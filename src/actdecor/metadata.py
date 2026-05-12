"""Extract static, privacy-safe metadata from a Python callable.

The metadata produced here matches ActRouter's ``FunctionInfo`` request schema.
Only *static* information about a function is collected — never call arguments
or return values — so it is safe to send to a remote routing service.
"""

from __future__ import annotations

import hashlib
import inspect
from typing import Any, Callable, Optional

# Task kinds understood by ActRouter (mirrors actrouter.schemas.TaskKind).
KNOWN_TASK_KINDS = frozenset(
    {
        "code_generation",
        "code_review",
        "summarization",
        "translation",
        "qa",
        "reasoning",
        "math",
        "classification",
        "extraction",
        "chat",
        "unknown",
    }
)


def _unwrap(func: Callable[..., Any]) -> Callable[..., Any]:
    return inspect.unwrap(func) if callable(func) else func


def _safe_signature(func: Callable[..., Any]) -> Optional[inspect.Signature]:
    try:
        return inspect.signature(func)
    except (ValueError, TypeError):
        return None


def _annotation_name(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty or annotation is inspect.Signature.empty:
        return ""
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, "__name__", None) or str(annotation)


def signature_string(func: Callable[..., Any]) -> Optional[str]:
    sig = _safe_signature(func)
    return str(sig) if sig is not None else None


def parameter_types(func: Callable[..., Any]) -> list[str]:
    sig = _safe_signature(func)
    if sig is None:
        return []
    out: list[str] = []
    for param in sig.parameters.values():
        name = _annotation_name(param.annotation)
        if name:
            out.append(name)
    return out


def return_type(func: Callable[..., Any]) -> Optional[str]:
    sig = _safe_signature(func)
    if sig is None:
        return None
    name = _annotation_name(sig.return_annotation)
    return name or None


def function_fingerprint(func: Callable[..., Any]) -> str:
    """A stable short hash identifying a function.

    Used as the hash-map key in the function inventory file. Derived from the
    function's module, qualified name and signature so that renaming the local
    binding does not change the key but a real signature change does.
    """

    target = _unwrap(func)
    module = getattr(target, "__module__", "") or ""
    qualname = getattr(target, "__qualname__", None) or getattr(target, "__name__", "") or ""
    sig = signature_string(target) or ""
    raw = f"{module}:{qualname}:{sig}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def build_function_metadata(
    func: Callable[..., Any],
    *,
    task_kind: str = "unknown",
    tags: Optional[list[str]] = None,
    decorators: Optional[list[str]] = None,
    estimated_prompt_tokens: Optional[int] = None,
    estimated_completion_tokens: Optional[int] = None,
    latency_budget_ms: Optional[int] = None,
    cost_budget_usd: Optional[float] = None,
    privacy_tier: str = "internal",
) -> dict[str, Any]:
    """Build the ``function`` payload for ActRouter's ``POST /v1/route``.

    The returned dict contains static metadata only and never includes runtime
    arguments or return values.
    """

    target = _unwrap(func)
    if task_kind not in KNOWN_TASK_KINDS:
        task_kind = "unknown"

    metadata: dict[str, Any] = {
        "name": getattr(target, "__name__", None) or "anonymous",
        "qualname": getattr(target, "__qualname__", None),
        "module": getattr(target, "__module__", None),
        "signature": signature_string(target),
        "parameter_types": parameter_types(target),
        "return_type": return_type(target),
        "decorators": [d.strip() for d in (decorators or []) if d and d.strip()],
        "docstring": inspect.getdoc(target),
        "task_kind": task_kind,
        "tags": [t.strip() for t in (tags or []) if t and t.strip()],
        "privacy_tier": privacy_tier,
    }
    if estimated_prompt_tokens is not None:
        metadata["estimated_prompt_tokens"] = int(estimated_prompt_tokens)
    if estimated_completion_tokens is not None:
        metadata["estimated_completion_tokens"] = int(estimated_completion_tokens)
    if latency_budget_ms is not None:
        metadata["latency_budget_ms"] = int(latency_budget_ms)
    if cost_budget_usd is not None:
        metadata["cost_budget_usd"] = float(cost_budget_usd)
    return metadata
