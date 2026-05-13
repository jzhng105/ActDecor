"""Tests for the unified-endpoint (model="provider/model") path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from actdecor import (  # noqa: E402
    ActLLMFunctionSummarizer,
    SummaryResult,
    summarize_with_llm,
)
from actdecor import decorator as decorator_module  # noqa: E402


class _StubCompletion:
    """Stand-in for actllminfer.completion that records each call."""

    def __init__(self, content: str = "Stub commentary") -> None:
        self.content = content
        self.calls: list[dict] = []

    def __call__(self, *, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        return {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }


@pytest.fixture
def stub_completion(monkeypatch):
    stub = _StubCompletion()
    monkeypatch.setattr(decorator_module, "unified_completion", stub)
    return stub


def test_model_argument_dispatches_to_unified_completion(stub_completion):
    @summarize_with_llm(model="openai/gpt-4o-mini")
    def add(a: int, b: int) -> int:
        return a + b

    result = add(2, 3)

    assert isinstance(result, SummaryResult)
    assert result.value == 5
    assert result.commentary == "Stub commentary"
    assert len(stub_completion.calls) == 1
    call = stub_completion.calls[0]
    assert call["model"] == "openai/gpt-4o-mini"
    # The prompt template produces a system + human message pair.
    assert len(call["messages"]) == 2
    assert result.info["provider"] == "openai"
    assert result.info["model"] == "openai/gpt-4o-mini"


def test_completion_params_forwarded(stub_completion):
    @summarize_with_llm(
        model="anthropic/claude-haiku-4-5",
        completion_params={"temperature": 0.0, "max_tokens": 128},
    )
    def fetch() -> dict:
        return {"ok": True}

    fetch()

    kwargs = stub_completion.calls[0]["kwargs"]
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 128


def test_routing_swaps_active_model_for_unified_path(stub_completion, tmp_path):
    class FakeRouter:
        def __init__(self) -> None:
            self.calls = 0

        def route(self, func, **kwargs):
            self.calls += 1
            return {
                "chosen_model": "deepseek/deepseek-chat",
                "router": kwargs.get("router", "heuristic"),
                "estimated_cost_usd": 0.0001,
                "estimated_latency_ms": 200,
            }

    router = FakeRouter()

    @summarize_with_llm(
        model="openai/gpt-4o-mini",
        return_mode="dict",
        router_client=router,
        inventory=tmp_path / "inv.json",
    )
    def crunch(n: int) -> int:
        return n * 2

    result = crunch(7)

    # The routed model is what the completion call actually used.
    assert stub_completion.calls[0]["model"] == "deepseek/deepseek-chat"
    assert result["info"]["routing"]["model"] == "deepseek/deepseek-chat"
    assert result["info"]["routing"]["applied"] is True
    assert result["info"]["routing"]["source"] == "actrouter"
    assert result["info"]["model"] == "deepseek/deepseek-chat"
    # A second call reuses the routed model and does not re-hit the router.
    crunch(1)
    assert router.calls == 1
    assert stub_completion.calls[1]["model"] == "deepseek/deepseek-chat"


def test_routing_apply_model_false_keeps_original_string(stub_completion, tmp_path):
    class FakeRouter:
        def route(self, func, **kwargs):
            return {"chosen_model": "anthropic/claude-haiku-4-5", "router": "heuristic"}

    @summarize_with_llm(
        model="openai/gpt-4o-mini",
        router_client=FakeRouter(),
        inventory=tmp_path / "inv.json",
        apply_model=False,
    )
    def fn() -> int:
        return 1

    fn()

    assert stub_completion.calls[0]["model"] == "openai/gpt-4o-mini"


def test_init_requires_exactly_one_of_model_or_llm():
    with pytest.raises(ValueError):
        ActLLMFunctionSummarizer()
    with pytest.raises(ValueError):
        ActLLMFunctionSummarizer(llm=object(), model="openai/gpt-4o-mini")
