from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Optional, Union

from actllminfer.output_parsers import StrOutputParser
from actllminfer.prompts import ChatPromptTemplate

from .inventory import FunctionInventory
from .router_client import ActRouterClient
from .vendors import available_candidate_models, detect_available_vendors

log = logging.getLogger("actdecor")

ReturnMode = Literal["summary", "original", "tuple", "dict", "object"]

PathLike = Union[str, Path]

# Attributes commonly used by LLM client libraries to hold the model id.
_MODEL_ATTRS = ("model", "model_name", "model_id")


@dataclass
class SummaryResult:
    """Container for a function result, generated commentary, and metadata."""

    value: Any
    commentary: str
    info: dict[str, Any]


def serialize_for_llm(value: Any, max_chars: int = 12000) -> str:
    """Serialize common Python values into concise text for LLM input."""

    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            return value.head(50).to_string()[:max_chars]

        if isinstance(value, pd.Series):
            return value.to_string()[:max_chars]
    except Exception:
        # pandas is optional and serialization should degrade gracefully.
        pass

    if hasattr(value, "model_dump"):
        try:
            return json.dumps(value.model_dump(), default=str, indent=2)[:max_chars]
        except Exception:
            pass

    try:
        return json.dumps(value, default=str, indent=2)[:max_chars]
    except Exception:
        return str(value)[:max_chars]


def apply_model_to_llm(llm: Any, model: str) -> bool:
    """Best-effort: point an LLM client object at ``model``.

    Returns ``True`` if the model was applied. Tries a ``set_model`` method
    first, then common attribute names. Never raises.
    """

    if not model:
        return False
    setter = getattr(llm, "set_model", None)
    if callable(setter):
        try:
            setter(model)
            return True
        except Exception:  # pragma: no cover - defensive
            pass
    for attr in _MODEL_ATTRS:
        if hasattr(llm, attr):
            try:
                setattr(llm, attr, model)
                return True
            except Exception:
                continue
    return False


def _routing_info(routing: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": routing.get("model"),
        "source": routing.get("source", "none"),
        "applied": bool(routing.get("applied")),
        "router": routing.get("router"),
        "estimated_cost_usd": routing.get("estimated_cost_usd"),
        "estimated_latency_ms": routing.get("estimated_latency_ms"),
        "available_vendors": routing.get("available_vendors"),
        "candidate_models": routing.get("candidate_models"),
        "error": routing.get("error"),
    }


class ActLLMFunctionSummarizer:
    """Decorator class that adds LLM commentary to wrapped function outputs.

    Optionally consults `ActRouter <https://github.com/jzhng105/actrouter>`_ to
    pick the best LLM for the wrapped function. Candidate models are first
    narrowed to vendors usable from the current environment (see
    :func:`~actdecor.vendors.detect_available_vendors`). The chosen model is
    cached in a *function inventory file* (see
    :class:`~actdecor.inventory.FunctionInventory`) keyed by a stable
    fingerprint of the function, so future calls reuse it without another
    round-trip to ActRouter.
    """

    def __init__(
        self,
        llm: Any,
        *,
        return_mode: ReturnMode = "object",
        result_key: str = "value",
        summary_key: str = "commentary",
        metadata_key: str = "info",
        max_chars: int = 12000,
        include_args: bool = True,
        fail_silently: bool = False,
        prompt: Optional[ChatPromptTemplate] = None,
        # --- ActRouter integration ---------------------------------------
        router_url: Optional[str] = None,
        router_client: Optional[ActRouterClient] = None,
        router: str = "heuristic",
        route_top_k: int = 3,
        candidate_models: Optional[list[str]] = None,
        route_hints: Optional[dict[str, Any]] = None,
        auto_route: bool = True,
        apply_model: bool = True,
        use_inventory: bool = True,
        inventory: Optional[Union[FunctionInventory, PathLike]] = None,
        route_fail_silently: bool = True,
        # --- restrict candidates to locally usable vendors ---------------
        restrict_to_available_vendors: bool = True,
        probe_vendors: bool = False,
        vendor_probe_timeout: float = 3.0,
        vendors: Optional[Iterable[str]] = None,
        vendor_env: Optional[Mapping[str, Iterable[str]]] = None,
    ):
        self.llm = llm
        self.return_mode = return_mode
        self.result_key = result_key
        self.summary_key = summary_key
        self.metadata_key = metadata_key
        self.max_chars = max_chars
        self.include_args = include_args
        self.fail_silently = fail_silently

        # ActRouter / inventory configuration.
        if router_client is None and router_url:
            router_client = ActRouterClient(router_url, router=router)
        self.router_client = router_client
        self.router = router
        self.route_top_k = route_top_k
        self.candidate_models = candidate_models
        self.route_hints: dict[str, Any] = dict(route_hints or {})
        self.route_hints.setdefault("decorators", ["summarize_with_llm"])
        self.auto_route = auto_route
        self.apply_model = apply_model
        self.route_fail_silently = route_fail_silently

        # Restrict candidate models to vendors the caller can actually use.
        self.restrict_to_available_vendors = restrict_to_available_vendors
        self.probe_vendors = probe_vendors
        self.vendor_probe_timeout = vendor_probe_timeout
        self.vendors = list(vendors) if vendors is not None else None
        self.vendor_env = dict(vendor_env) if vendor_env else None
        self._available_vendors: Optional[list[str]] = None
        self._candidate_models_cache: Optional[list[str]] = None
        self._candidate_models_resolved = False

        if not use_inventory:
            self.inventory: Optional[FunctionInventory] = None
        elif isinstance(inventory, FunctionInventory):
            self.inventory = inventory
        else:
            self.inventory = FunctionInventory(inventory)

        self.prompt = prompt or ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are an analytical assistant.

Summarize the return value of a Python function.
Focus on key results, anomalies, risks, and suggested next steps.
Be concise but useful.
""",
                ),
                (
                    "human",
                    """
Function name:
{function_name}

Function arguments:
{arguments}

Function return value:
{return_value}
""",
                ),
            ]
        )

        self.chain = self.prompt | self.llm | StrOutputParser()

    # -- ActRouter routing --------------------------------------------------

    def _routing_active(self) -> bool:
        return self.auto_route and (self.inventory is not None or self.router_client is not None)

    def resolve_candidate_models(self) -> Optional[list[str]]:
        """Candidate model ids to send to ActRouter.

        If ``candidate_models`` was given explicitly it is used as-is. Otherwise,
        when ``restrict_to_available_vendors`` is set, detect which major LLM
        vendors are usable from this environment (optionally probing their APIs)
        and limit candidates to those vendors' models. Returns ``None`` to mean
        "no restriction — let ActRouter use its full catalog".
        """

        if self.candidate_models is not None:
            return list(self.candidate_models)
        if not self.restrict_to_available_vendors:
            return None
        if not self._candidate_models_resolved:
            self._candidate_models_resolved = True
            if self.vendors is not None:
                vendors = set(self.vendors)
            else:
                vendors = detect_available_vendors(
                    probe=self.probe_vendors,
                    timeout=self.vendor_probe_timeout,
                    extra_env=self.vendor_env,
                )
            self._available_vendors = sorted(vendors)
            self._candidate_models_cache = available_candidate_models(
                vendors, router_client=self.router_client
            )
            if not self._candidate_models_cache:
                log.info(
                    "ActDecor: no usable LLM vendors detected; "
                    "letting ActRouter choose from its full catalog"
                )
        return self._candidate_models_cache or None

    def resolve_model(self, func: Callable[..., Any]) -> dict[str, Any]:
        """Determine the model ``func`` should use.

        Looks the function up in the inventory file first; on a miss, calls the
        ActRouter API (if configured) and records the recommendation. Routing
        failures are swallowed unless ``route_fail_silently=False``.
        """

        routing: dict[str, Any] = {
            "model": None,
            "source": "none",
            "router": None,
            "estimated_cost_usd": None,
            "estimated_latency_ms": None,
            "ranking": None,
            "available_vendors": None,
            "candidate_models": None,
            "error": None,
        }
        if not self._routing_active():
            return routing

        if self.inventory is not None:
            record = self.inventory.get(func)
            if record and record.get("recommended_model"):
                routing.update(
                    model=record["recommended_model"],
                    source="inventory",
                    router=record.get("router"),
                    estimated_cost_usd=record.get("estimated_cost_usd"),
                    estimated_latency_ms=record.get("estimated_latency_ms"),
                    ranking=record.get("ranking"),
                )
                return routing

        if self.router_client is None:
            return routing

        candidates = self.resolve_candidate_models()
        routing["available_vendors"] = self._available_vendors
        routing["candidate_models"] = candidates

        try:
            response = self.router_client.route(
                func,
                router=self.router,
                top_k=self.route_top_k,
                candidate_models=candidates,
                **self.route_hints,
            )
        except Exception as exc:
            routing["error"] = str(exc)
            log.warning(
                "ActRouter routing failed for %s: %s",
                getattr(func, "__name__", func),
                exc,
            )
            if not self.route_fail_silently:
                raise
            return routing

        model = (response or {}).get("chosen_model")
        if not model:
            routing["error"] = "ActRouter response did not include 'chosen_model'"
            return routing

        routing.update(
            model=model,
            source="actrouter",
            router=response.get("router", self.router),
            estimated_cost_usd=response.get("estimated_cost_usd"),
            estimated_latency_ms=response.get("estimated_latency_ms"),
            ranking=response.get("ranking"),
        )
        if self.inventory is not None:
            try:
                self.inventory.record(
                    func,
                    model,
                    router=routing["router"],
                    estimated_cost_usd=routing["estimated_cost_usd"],
                    estimated_latency_ms=routing["estimated_latency_ms"],
                    ranking=routing["ranking"],
                )
            except OSError as exc:  # pragma: no cover - filesystem dependent
                log.warning(
                    "Could not persist ActDecor inventory at %s: %s",
                    getattr(self.inventory, "path", "?"),
                    exc,
                )
        return routing

    # -- decoration ---------------------------------------------------------

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        # Per-decorated-function routing state, resolved lazily on first call.
        _state: dict[str, Any] = {"resolved": False, "routing": None}

        def _ensure_routing() -> dict[str, Any]:
            if not _state["resolved"]:
                _state["resolved"] = True
                routing = self.resolve_model(func)
                if routing.get("model") and self.apply_model:
                    routing["applied"] = apply_model_to_llm(self.llm, routing["model"])
                _state["routing"] = routing
            return _state["routing"]

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            routing = _ensure_routing()
            # Re-point the (possibly shared) llm at this function's model.
            if routing.get("model") and self.apply_model:
                apply_model_to_llm(self.llm, routing["model"])

            original_value = func(*args, **kwargs)

            try:
                return_value_text = serialize_for_llm(
                    original_value,
                    max_chars=self.max_chars,
                )

                if self.include_args:
                    try:
                        bound = inspect.signature(func).bind_partial(*args, **kwargs)
                        bound.apply_defaults()
                        arguments_text = serialize_for_llm(
                            bound.arguments,
                            max_chars=3000,
                        )
                    except Exception:
                        arguments_text = "Arguments could not be serialized."
                else:
                    arguments_text = "Not included."

                commentary = self.chain.invoke(
                    {
                        "function_name": func.__name__,
                        "arguments": arguments_text,
                        "return_value": return_value_text,
                    }
                )

                info = {
                    "function_name": func.__name__,
                    "summarized": True,
                    "provider": self.llm.__class__.__name__,
                    "model": routing.get("model"),
                    "routing": _routing_info(routing),
                }

            except Exception as exc:
                if not self.fail_silently:
                    raise

                commentary = f"LLM summarization failed: {exc}"
                info = {
                    "function_name": func.__name__,
                    "summarized": False,
                    "error": str(exc),
                    "model": routing.get("model"),
                    "routing": _routing_info(routing),
                }

            if self.return_mode == "summary":
                return commentary

            if self.return_mode == "original":
                return original_value

            if self.return_mode == "tuple":
                return original_value, commentary

            if self.return_mode == "dict":
                return {
                    self.result_key: original_value,
                    self.summary_key: commentary,
                    self.metadata_key: info,
                }

            if self.return_mode == "object":
                return SummaryResult(
                    value=original_value,
                    commentary=commentary,
                    info=info,
                )

            raise ValueError(f"Unsupported return_mode: {self.return_mode}")

        # Expose routing state for introspection / testing.
        wrapper.actdecor_resolve_routing = _ensure_routing  # type: ignore[attr-defined]
        wrapper.actdecor_summarizer = self  # type: ignore[attr-defined]
        return wrapper


def summarize_with_llm(
    llm: Any,
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Convenience decorator factory for :class:`ActLLMFunctionSummarizer`."""

    return ActLLMFunctionSummarizer(llm=llm, **kwargs)
