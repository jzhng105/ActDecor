from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from actllminfer.runnables import RunnableLambda  # noqa: E402

from actdecor import (  # noqa: E402
    ActRouterClient,
    ActRouterError,
    FunctionInventory,
    apply_model_to_llm,
    build_function_metadata,
    function_fingerprint,
    summarize_with_llm,
)


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------

# Fields ActRouter rejects as "runtime data". build_function_metadata must
# never emit any of them.
_FORBIDDEN_FIELDS = {
    "args", "kwargs", "arguments", "inputs", "input", "input_data",
    "payload", "body", "result", "results", "return_value", "output",
    "outputs", "data",
}


def example_fn(x: int, y: str = "z") -> dict:
    """Do a thing with x and y."""
    return {"x": x, "y": y}


def test_build_function_metadata_static_only():
    meta = build_function_metadata(example_fn, task_kind="extraction", tags=["demo"])

    assert meta["name"] == "example_fn"
    assert meta["signature"] == "(x: int, y: str = 'z') -> dict"
    assert meta["parameter_types"] == ["int", "str"]
    assert meta["return_type"] == "dict"
    assert meta["task_kind"] == "extraction"
    assert meta["tags"] == ["demo"]
    assert meta["docstring"] == "Do a thing with x and y."
    # No runtime data leaked.
    assert _FORBIDDEN_FIELDS.isdisjoint(meta.keys())


def test_build_function_metadata_unknown_task_kind_falls_back():
    meta = build_function_metadata(example_fn, task_kind="not-a-real-kind")
    assert meta["task_kind"] == "unknown"


def test_fingerprint_is_stable_and_distinct():
    fp1 = function_fingerprint(example_fn)

    def other(x: int) -> int:
        return x

    assert fp1 == function_fingerprint(example_fn)
    assert fp1 != function_fingerprint(other)
    assert len(fp1) == 16


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------


def test_inventory_round_trip(tmp_path):
    path = tmp_path / "inv.json"
    inv = FunctionInventory(path)
    assert example_fn not in inv
    assert inv.get_model(example_fn) is None

    inv.record(example_fn, "gemini-2-5-flash", router="heuristic", estimated_cost_usd=0.0017)

    assert path.exists()
    data = json.loads(path.read_text())
    assert data["version"] == 1
    key = function_fingerprint(example_fn)
    assert data["functions"][key]["recommended_model"] == "gemini-2-5-flash"
    assert data["functions"][key]["function_name"] == "example_fn"

    # Fresh handle re-reads from disk.
    inv2 = FunctionInventory(path)
    assert example_fn in inv2
    assert inv2.get_model(example_fn) == "gemini-2-5-flash"


def test_inventory_handles_corrupt_file(tmp_path):
    path = tmp_path / "inv.json"
    path.write_text("{not valid json")
    inv = FunctionInventory(path)
    assert inv.get_model(example_fn) is None  # degrades gracefully


# ---------------------------------------------------------------------------
# apply_model_to_llm
# ---------------------------------------------------------------------------


def test_apply_model_sets_attribute():
    llm = types.SimpleNamespace(model="old")
    assert apply_model_to_llm(llm, "new-model") is True
    assert llm.model == "new-model"


def test_apply_model_prefers_set_model_method():
    captured = {}

    class LLM:
        def set_model(self, model):
            captured["model"] = model

    assert apply_model_to_llm(LLM(), "fancy") is True
    assert captured["model"] == "fancy"


def test_apply_model_noop_when_unknown_shape():
    assert apply_model_to_llm(object(), "x") is False
    assert apply_model_to_llm(types.SimpleNamespace(), "") is False


# ---------------------------------------------------------------------------
# ActRouterClient (transport mocked)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_router_client_route(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(
            json.dumps(
                {
                    "chosen_model": "claude-haiku-4-5",
                    "router": "heuristic",
                    "ranking": [{"model_id": "claude-haiku-4-5", "score": 0.8, "reasons": []}],
                    "estimated_cost_usd": 0.001,
                    "estimated_latency_ms": 250,
                }
            ).encode("utf-8")
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = ActRouterClient("https://api.example.com/")
    model = client.recommend_model(example_fn, task_kind="extraction")

    assert model == "claude-haiku-4-5"
    assert seen["url"] == "https://api.example.com/v1/route"
    assert seen["method"] == "POST"
    assert seen["body"]["function"]["name"] == "example_fn"
    assert seen["body"]["function"]["task_kind"] == "extraction"
    assert _FORBIDDEN_FIELDS.isdisjoint(seen["body"]["function"].keys())


def test_router_client_wraps_network_errors(monkeypatch):
    import urllib.error

    def boom(request, timeout=None):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(ActRouterError):
        ActRouterClient("https://api.example.com").recommend_model(example_fn)


# ---------------------------------------------------------------------------
# decorator integration
# ---------------------------------------------------------------------------


class FakeRouterClient:
    """Stand-in for ActRouterClient with a configurable response."""

    def __init__(self, chosen_model="gemini-2-5-flash"):
        self.chosen_model = chosen_model
        self.calls = 0

    def route(self, func, **kwargs):
        self.calls += 1
        return {
            "chosen_model": self.chosen_model,
            "router": kwargs.get("router", "heuristic"),
            "ranking": [{"model_id": self.chosen_model, "score": 0.9, "reasons": ["fake"]}],
            "estimated_cost_usd": 0.002,
            "estimated_latency_ms": 300,
        }


@pytest.fixture
def fake_llm() -> RunnableLambda:
    return RunnableLambda(lambda _: "Concise analysis")


def test_decorator_routes_and_records(tmp_path, fake_llm):
    inv_path = tmp_path / "inventory.json"
    router = FakeRouterClient("claude-haiku-4-5")

    @summarize_with_llm(fake_llm, return_mode="dict", router_client=router, inventory=inv_path)
    def crunch(n: int) -> int:
        return n * 2

    result = crunch(21)

    assert result["value"] == 42
    assert result["info"]["routing"]["model"] == "claude-haiku-4-5"
    assert result["info"]["routing"]["source"] == "actrouter"
    assert result["info"]["model"] == "claude-haiku-4-5"
    assert router.calls == 1

    # Recorded to the inventory file...
    data = json.loads(inv_path.read_text())
    key = function_fingerprint(crunch.__wrapped__)
    assert data["functions"][key]["recommended_model"] == "claude-haiku-4-5"

    # ...and a second call does not hit the router again.
    crunch(1)
    assert router.calls == 1


def test_decorator_prefers_existing_inventory(tmp_path, fake_llm):
    inv_path = tmp_path / "inventory.json"

    def target(n: int) -> int:
        return n

    FunctionInventory(inv_path).record(target, "preselected-model", router="rules")

    router = FakeRouterClient("should-not-be-used")

    @summarize_with_llm(fake_llm, return_mode="dict", router_client=router, inventory=inv_path)
    def target(n: int) -> int:  # noqa: F811 - same module/qualname/signature => same fingerprint
        return n

    result = target(5)

    assert result["info"]["routing"]["model"] == "preselected-model"
    assert result["info"]["routing"]["source"] == "inventory"
    assert router.calls == 0


def test_decorator_routing_failure_is_silent(tmp_path, fake_llm):
    class BrokenRouter:
        def route(self, func, **kwargs):
            raise ActRouterError("backend down")

    @summarize_with_llm(
        fake_llm,
        return_mode="dict",
        router_client=BrokenRouter(),
        inventory=tmp_path / "inv.json",
    )
    def still_works(x: int) -> int:
        return x + 1

    result = still_works(1)
    assert result["value"] == 2
    assert result["info"]["routing"]["model"] is None
    assert "backend down" in result["info"]["routing"]["error"]


def test_decorator_without_routing_is_unchanged(fake_llm):
    @summarize_with_llm(fake_llm, return_mode="tuple", auto_route=False)
    def f() -> dict:
        return {"ok": True}

    value, commentary = f()
    assert value == {"ok": True}
    assert commentary == "Concise analysis"
