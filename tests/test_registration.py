"""Tests for the OSFI E-23 style tool registration feature."""

from __future__ import annotations

import json

import pytest
from actllminfer.runnables import RunnableLambda

from actdecor import (
    E23_REGISTRATION_FIELDS,
    FunctionInventory,
    build_registration_record,
    extract_registration_fields,
    summarize_with_llm,
)

_REGISTRATION_JSON = json.dumps(
    {
        "description": "Adds two integers and returns the sum.",
        "intended_use": "Arithmetic helper for reporting pipelines.",
        "model_type": "deterministic calculation",
        "methodology": "Integer addition.",
        "inputs": ["a: first addend", "b: second addend"],
        "outputs": ["the integer sum"],
        "assumptions": ["inputs are numeric"],
        "limitations": ["no overflow checks"],
        "risk_rating": "LOW",
        "risk_rationale": "Simple deterministic arithmetic.",
        "regulatory_use": "no",
        "ignored_unknown_field": "dropped",
    }
)


@pytest.fixture
def registering_llm() -> RunnableLambda:
    """Fake LLM: returns registration JSON for the registration prompt."""

    def _invoke(messages):
        text = " ".join(str(getattr(m, "content", m)) for m in messages)
        if "model inventory" in text:
            return _REGISTRATION_JSON
        return "Concise analysis"

    return RunnableLambda(_invoke)


def test_registration_populates_e23_fields(registering_llm, tmp_path):
    inv_path = tmp_path / "inv.json"

    @summarize_with_llm(
        registering_llm,
        auto_route=False,
        router_url=None,
        register_tool=True,
        inventory=inv_path,
    )
    def add(a: int, b: int) -> int:
        return a + b

    result = add(2, 3)

    reg = result.info["tool_registration"]
    assert reg is not None
    # All declared E-23 fields are present in the record.
    assert set(reg) == set(E23_REGISTRATION_FIELDS)
    # LLM-derived descriptive fields.
    assert reg["description"].startswith("Adds two integers")
    assert reg["model_type"] == "deterministic calculation"
    assert reg["risk_rating"] == "low"  # normalized from "LOW"
    assert reg["limitations"] == ["no overflow checks"]
    # Deterministic identity / lifecycle fields.
    assert reg["name"] == "add"
    assert reg["status"] == "registered"
    assert reg["lifecycle_stage"] == "in_use"
    assert reg["date_registered"] is not None


def test_registration_persisted_to_inventory(registering_llm, tmp_path):
    inv_path = tmp_path / "inv.json"

    @summarize_with_llm(
        registering_llm,
        auto_route=False,
        router_url=None,
        register_tool=True,
        inventory=inv_path,
    )
    def multiply(a: int, b: int) -> int:
        return a * b

    multiply(2, 3)

    saved = json.loads(inv_path.read_text())
    records = list(saved["functions"].values())
    assert len(records) == 1
    assert "tool_registration" in records[0]
    assert records[0]["tool_registration"]["name"] == "multiply"

    # A fresh process (new inventory instance) reuses the stored registration.
    reused = FunctionInventory(inv_path).get_registration(multiply)
    assert reused is not None and reused["name"] == "multiply"


def test_registration_disabled_by_default(registering_llm, tmp_path):
    @summarize_with_llm(
        registering_llm,
        auto_route=False,
        router_url=None,
        inventory=tmp_path / "inv.json",
    )
    def noop() -> int:
        return 1

    assert noop().info["tool_registration"] is None


def test_extract_registration_fields_tolerates_fences():
    fenced = "```json\n" + _REGISTRATION_JSON + "\n```"
    fields = extract_registration_fields(fenced)
    assert fields["model_type"] == "deterministic calculation"
    assert "ignored_unknown_field" not in fields

    assert extract_registration_fields("not json") == {}


def test_build_registration_record_without_llm_fields():
    def sample(x: int) -> str:
        """A sample function."""
        return str(x)

    record = build_registration_record(sample, {}, model="openai/gpt-4o-mini")
    assert set(record) == set(E23_REGISTRATION_FIELDS)
    assert record["name"] == "sample"
    assert record["registered_by_model"] == "openai/gpt-4o-mini"
    # Falls back to static type hints when the LLM provides nothing.
    assert record["inputs"] == ["int"]
    assert record["outputs"] == "str"
    assert record["description"] is None
