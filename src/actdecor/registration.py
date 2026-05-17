"""OSFI E-23 style "tool registration" for decorated functions.

When a function wrapped by ActDecor runs, the same LLM that produces the
commentary can also produce a structured *tool registration* record describing
the function as a model/tool. The fields mirror what OSFI Guideline E-23
(Model Risk Management) expects an institution's model inventory to capture, so
``actdecor_inventory.json`` doubles as a lightweight model inventory.

The registration is a best-effort artifact: descriptive and judgement fields
are filled from the LLM response; identity and lifecycle fields are filled
deterministically from the function's static metadata. Fields that cannot be
determined are left as ``None`` so gaps are visible rather than guessed.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from actllminfer.prompts import ChatPromptTemplate

from .metadata import build_function_metadata, function_fingerprint, signature_string

# Descriptive / judgement fields the LLM is asked to populate. These map onto
# the qualitative parts of an OSFI E-23 model inventory entry.
E23_LLM_FIELDS = (
    "description",
    "intended_use",
    "model_type",
    "methodology",
    "inputs",
    "outputs",
    "data_sources",
    "assumptions",
    "limitations",
    "risk_rating",
    "risk_rationale",
    "materiality",
    "dependencies",
    "downstream_use",
    "regulatory_use",
)

# Full ordered set of registration fields stored in the inventory. The fields
# outside ``E23_LLM_FIELDS`` are filled deterministically (identity, ownership
# and lifecycle), since OSFI E-23 expects those to be authoritative rather than
# inferred.
E23_REGISTRATION_FIELDS = (
    "tool_id",
    "name",
    "qualname",
    "module",
    "signature",
    *E23_LLM_FIELDS,
    "owner",
    "developer",
    "lifecycle_stage",
    "status",
    "vendor_or_internal",
    "date_registered",
    "last_reviewed",
    "next_review_due",
    "registered_by_model",
)

_VALID_RISK = {"low", "medium", "high"}


REGISTRATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a model-risk documentation assistant. Given a Python function, you
produce a structured registration record for an institutional model inventory
aligned with OSFI Guideline E-23 (Model Risk Management).

Base your answer only on the information provided. When a field cannot be
determined from the information given, use null instead of guessing. Respond
with a single JSON object and nothing else.
""",
        ),
        (
            "human",
            """
Document the following Python function as an entry in a model inventory.

Function name:
{function_name}

Module:
{module}

Signature:
{signature}

Docstring:
{docstring}

Sample arguments:
{arguments}

Sample return value:
{return_value}

Return a JSON object with exactly these keys:
- description: one-sentence plain-language description of what the function does
- intended_use: the business purpose / intended use of the function
- model_type: category, e.g. "deterministic calculation", "data transformation",
  "statistical model", "machine learning model", "rules-based logic", "LLM-based"
- methodology: the technique or approach used
- inputs: list describing the inputs and their meaning
- outputs: list describing the outputs and their meaning
- data_sources: data sources or systems the function relies on (or null)
- assumptions: list of key assumptions
- limitations: list of known limitations
- risk_rating: one of "low", "medium", "high"
- risk_rationale: brief justification of the risk rating
- materiality: assessment of materiality / business impact
- dependencies: upstream functions, models or services it depends on (or null)
- downstream_use: how the output is consumed downstream (or null)
- regulatory_use: whether the output feeds regulatory or financial reporting
  ("yes", "no" or "unknown")

Respond with only the JSON object.
""",
        ),
    ]
)


def _loads_lenient(text: str) -> Optional[Any]:
    """Parse JSON from an LLM response, tolerating code fences and prose."""

    text = text.strip()
    fence = re.match(r"^```[a-zA-Z0-9]*\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def extract_registration_fields(text: str) -> dict[str, Any]:
    """Pull the recognised E-23 fields out of an LLM registration response.

    Unknown keys are dropped; a malformed response yields an empty dict.
    """

    if not text:
        return {}
    data = _loads_lenient(text)
    if not isinstance(data, dict):
        return {}
    return {key: value for key, value in data.items() if key in E23_LLM_FIELDS}


def build_registration_record(
    func: Callable[..., Any],
    llm_fields: Optional[dict[str, Any]] = None,
    *,
    model: Optional[str] = None,
    owner: Optional[str] = None,
) -> dict[str, Any]:
    """Build an OSFI E-23 style registration record for ``func``.

    Identity, ownership and lifecycle fields are filled from static metadata;
    descriptive fields are taken from ``llm_fields`` (the parsed LLM response).
    Anything still unknown is left as ``None``.
    """

    llm_fields = llm_fields or {}
    meta = build_function_metadata(func)
    now = datetime.now(timezone.utc).isoformat()

    record: dict[str, Any] = {field: None for field in E23_REGISTRATION_FIELDS}
    record.update(
        {
            "tool_id": function_fingerprint(func),
            "name": meta["name"],
            "qualname": meta["qualname"],
            "module": meta["module"],
            "signature": meta["signature"],
            "owner": owner or meta["module"] or "unknown",
            "developer": meta["module"] or "unknown",
            "lifecycle_stage": "in_use",
            "status": "registered",
            "vendor_or_internal": "internal",
            "date_registered": now,
            "last_reviewed": now,
            "next_review_due": None,
            "registered_by_model": model,
        }
    )

    for field in E23_LLM_FIELDS:
        value = llm_fields.get(field)
        if value not in (None, "", [], {}):
            record[field] = value

    rating = record.get("risk_rating")
    if isinstance(rating, str):
        rating = rating.strip().lower()
        record["risk_rating"] = rating if rating in _VALID_RISK else None

    # Fall back to static type hints when the LLM did not describe inputs/outputs.
    if record["inputs"] is None and meta["parameter_types"]:
        record["inputs"] = meta["parameter_types"]
    if record["outputs"] is None and meta["return_type"]:
        record["outputs"] = meta["return_type"]

    return record


def build_registration_messages(
    func: Callable[..., Any],
    *,
    arguments_text: str,
    return_value_text: str,
) -> list:
    """Format the registration prompt for ``func`` into LLM messages."""

    import inspect

    target = inspect.unwrap(func) if callable(func) else func
    return REGISTRATION_PROMPT.format_messages(
        function_name=getattr(target, "__name__", None) or "anonymous",
        module=getattr(target, "__module__", None) or "",
        signature=signature_string(target) or "",
        docstring=inspect.getdoc(target) or "(no docstring)",
        arguments=arguments_text,
        return_value=return_value_text,
    )
