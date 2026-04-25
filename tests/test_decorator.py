from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableLambda

from actdecor import LangChainFunctionSummarizer, SummaryResult, summarize_with_llm


@pytest.fixture
def fake_llm() -> RunnableLambda:
    def _invoke(_prompt_value):
        return "Concise analysis"

    return RunnableLambda(_invoke)


def test_default_object_mode(fake_llm: RunnableLambda):
    @summarize_with_llm(fake_llm)
    def add(a: int, b: int) -> int:
        return a + b

    result = add(2, 3)

    assert isinstance(result, SummaryResult)
    assert result.value == 5
    assert result.commentary == "Concise analysis"
    assert result.info["summarized"] is True


def test_tuple_mode(fake_llm: RunnableLambda):
    @LangChainFunctionSummarizer(fake_llm, return_mode="tuple")
    def make_payload() -> dict:
        return {"ok": True}

    value, summary = make_payload()

    assert value == {"ok": True}
    assert summary == "Concise analysis"


def test_fail_silently_returns_original_when_llm_breaks():
    broken_llm = RunnableLambda(lambda _: (_ for _ in ()).throw(RuntimeError("boom")))

    @LangChainFunctionSummarizer(
        broken_llm,
        return_mode="dict",
        fail_silently=True,
    )
    def produce() -> list[int]:
        return [1, 2, 3]

    result = produce()

    assert result["value"] == [1, 2, 3]
    assert "failed" in result["commentary"].lower()
    assert result["info"]["summarized"] is False


def test_invalid_return_mode_raises(fake_llm: RunnableLambda):
    decorator = LangChainFunctionSummarizer(fake_llm)
    decorator.return_mode = "nope"  # type: ignore[assignment]

    @decorator
    def fn():
        return "x"

    with pytest.raises(ValueError):
        fn()
