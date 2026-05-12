from __future__ import annotations

import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from actllminfer.runnables import RunnableLambda  # noqa: E402

from actdecor import (  # noqa: E402
    available_candidate_models,
    detect_available_vendors,
    probe_vendor,
    summarize_with_llm,
)


# ---------------------------------------------------------------------------
# detect_available_vendors
# ---------------------------------------------------------------------------


def test_detect_from_env_keys():
    vendors = detect_available_vendors(env={"OPENAI_API_KEY": "sk-x", "ANTHROPIC_API_KEY": "   "})
    assert vendors == {"openai"}  # blank key is ignored


def test_detect_empty_env():
    assert detect_available_vendors(env={}) == set()


def test_detect_alias_env_var():
    assert detect_available_vendors(env={"GEMINI_API_KEY": "g"}) == {"google"}


def test_detect_extra_env_mapping():
    vendors = detect_available_vendors(env={"MY_LLAMA_KEY": "x"}, extra_env={"meta": ["MY_LLAMA_KEY"]})
    assert vendors == {"meta"}


def test_detect_include_exclude():
    env = {"OPENAI_API_KEY": "a", "ANTHROPIC_API_KEY": "b", "MISTRAL_API_KEY": "c"}
    assert detect_available_vendors(env=env, include=["openai", "mistral"]) == {"openai", "mistral"}
    assert detect_available_vendors(env=env, exclude=["anthropic"]) == {"openai", "mistral"}


# ---------------------------------------------------------------------------
# probe_vendor
# ---------------------------------------------------------------------------


class _FakeOK:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://api.example.com", code, "boom", None, None)


def test_probe_success(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: _FakeOK())
    assert probe_vendor("openai", api_key="sk-x") is True


def test_probe_unauthorized(monkeypatch):
    def raise_401(req, timeout=None):
        raise _http_error(401)

    monkeypatch.setattr("urllib.request.urlopen", raise_401)
    assert probe_vendor("openai", api_key="sk-x") is False


def test_probe_rate_limited_counts_as_available(monkeypatch):
    def raise_429(req, timeout=None):
        raise _http_error(429)

    monkeypatch.setattr("urllib.request.urlopen", raise_429)
    assert probe_vendor("openai", api_key="sk-x") is True


def test_probe_network_error(monkeypatch):
    def raise_urlerror(req, timeout=None):
        raise urllib.error.URLError("dns")

    monkeypatch.setattr("urllib.request.urlopen", raise_urlerror)
    assert probe_vendor("anthropic", api_key="sk-x") is False


def test_probe_without_known_endpoint_falls_back_to_key_presence():
    assert probe_vendor("minimax", api_key="present") is True
    assert probe_vendor("minimax", api_key=None, env={}) is False


def test_probe_known_endpoint_requires_key():
    assert probe_vendor("openai", api_key=None, env={}) is False


def test_detect_with_probe(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: _FakeOK())
    vendors = detect_available_vendors(env={"OPENAI_API_KEY": "sk-x"}, probe=True)
    assert vendors == {"openai"}

    def raise_401(req, timeout=None):
        raise _http_error(401)

    monkeypatch.setattr("urllib.request.urlopen", raise_401)
    assert detect_available_vendors(env={"OPENAI_API_KEY": "sk-x"}, probe=True) == set()


# ---------------------------------------------------------------------------
# available_candidate_models
# ---------------------------------------------------------------------------


def test_candidates_from_catalog_models():
    catalog = [
        {"id": "gpt-5-mini", "family": "openai"},
        {"id": "gpt-5.5", "family": "openai"},
        {"id": "claude-haiku-4-5", "family": "anthropic"},
    ]
    assert available_candidate_models({"openai"}, catalog_models=catalog) == ["gpt-5-mini", "gpt-5.5"]


def test_candidates_empty_when_no_vendors():
    assert available_candidate_models(set(), catalog_models=[{"id": "x", "family": "openai"}]) == []


def test_candidates_fallback_map_used_without_catalog():
    out = available_candidate_models({"anthropic"})
    assert "claude-haiku-4-5" in out


def test_candidates_uses_router_client_list_models():
    class RC:
        def list_models(self):
            return [{"id": "gemini-3-flash-preview", "family": "google"}, {"id": "x", "family": "openai"}]

    assert available_candidate_models({"google"}, router_client=RC()) == ["gemini-3-flash-preview"]


def test_candidates_falls_back_when_list_models_unavailable():
    class RC:
        pass  # no list_models method

    out = available_candidate_models({"openai"}, router_client=RC())
    assert "gpt-5-mini" in out  # came from the fallback map


# ---------------------------------------------------------------------------
# decorator integration
# ---------------------------------------------------------------------------


class FakeRouterClient:
    def __init__(self, chosen_model="gpt-5-mini", catalog=None):
        self.chosen_model = chosen_model
        self.catalog = catalog or [
            {"id": "gpt-5-mini", "family": "openai"},
            {"id": "gpt-5.5", "family": "openai"},
            {"id": "claude-haiku-4-5", "family": "anthropic"},
            {"id": "gemini-3-flash-preview", "family": "google"},
        ]
        self.route_calls: list[dict] = []

    def list_models(self):
        return list(self.catalog)

    def route(self, func, **kwargs):
        self.route_calls.append(kwargs)
        return {"chosen_model": self.chosen_model, "router": kwargs.get("router", "heuristic"), "ranking": []}


@pytest.fixture
def fake_llm() -> RunnableLambda:
    return RunnableLambda(lambda _: "Concise analysis")


def test_decorator_restricts_candidates_to_available_vendors(tmp_path, fake_llm):
    rc = FakeRouterClient(chosen_model="gpt-5-mini")

    @summarize_with_llm(
        fake_llm,
        return_mode="dict",
        router_client=rc,
        inventory=tmp_path / "inv.json",
        vendors=["openai"],
    )
    def f(x: int) -> int:
        return x

    result = f(1)

    assert rc.route_calls and rc.route_calls[0]["candidate_models"] == ["gpt-5-mini", "gpt-5.5"]
    routing = result["info"]["routing"]
    assert routing["available_vendors"] == ["openai"]
    assert routing["candidate_models"] == ["gpt-5-mini", "gpt-5.5"]
    assert routing["model"] == "gpt-5-mini"


def test_decorator_no_restriction_sends_no_candidates(tmp_path, fake_llm):
    rc = FakeRouterClient()

    @summarize_with_llm(
        fake_llm,
        return_mode="dict",
        router_client=rc,
        inventory=tmp_path / "inv.json",
        restrict_to_available_vendors=False,
    )
    def g(x: int) -> int:
        return x

    g(1)
    assert rc.route_calls[0]["candidate_models"] is None


def test_decorator_explicit_candidate_models_win(tmp_path, fake_llm):
    rc = FakeRouterClient(chosen_model="claude-haiku-4-5")

    @summarize_with_llm(
        fake_llm,
        return_mode="dict",
        router_client=rc,
        inventory=tmp_path / "inv.json",
        candidate_models=["claude-haiku-4-5"],
        vendors=["openai"],  # ignored because candidate_models is explicit
    )
    def h(x: int) -> int:
        return x

    h(1)
    assert rc.route_calls[0]["candidate_models"] == ["claude-haiku-4-5"]


def test_decorator_unknown_vendor_yields_empty_then_no_restriction(tmp_path, fake_llm):
    rc = FakeRouterClient()

    @summarize_with_llm(
        fake_llm,
        return_mode="dict",
        router_client=rc,
        inventory=tmp_path / "inv.json",
        vendors=["nonexistent-vendor"],
    )
    def k(x: int) -> int:
        return x

    k(1)
    # Nothing matched -> fall back to letting ActRouter use the full catalog.
    assert rc.route_calls[0]["candidate_models"] is None
