from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Optional, Union

from actllminfer import completion as unified_completion
from actllminfer.messages import SystemMessage
from actllminfer.output_parsers import StrOutputParser
from actllminfer.prompts import ChatPromptTemplate

from .inventory import FunctionInventory
from .router_client import DEFAULT_URL, ActRouterClient, configured_api_key, configured_url, is_hosted
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

def read_model_from_llm(llm: Any) -> Optional[str]:
    """Read the model ID current set for LLM client.
    Try get_model method first then try common attribute names
    """
    getter = getattr(llm, "get_model", None)
    if callable(getter):
        try:
            value = getter()
            if value:
                return str(value)
        except Exception:
            pass

    for attr in _MODEL_ATTRS:
        value = getattr(llm, attr, None)
        if value:
            return str(value)

    return None

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


def _provider_from_model(model: str) -> Optional[str]:
    """Pull the provider prefix off a LiteLLM-style ``provider/model`` string."""

    if not model:
        return None
    for sep in (":", "/"):
        if sep in model:
            head = model.split(sep, 1)[0].strip()
            if head:
                return head.lower()
    return None


class ActLLMFunctionSummarizer:
    """Decorator class that adds LLM commentary to wrapped function outputs.

    There are two ways to point the decorator at a model:

    * ``model="provider/model"`` — use the **unified** ``actllminfer.completion()``
      endpoint (LiteLLM-style). No provider client object is constructed; the
      decorator dispatches each call through :func:`actllminfer.completion`.
    * ``llm=<runnable>`` — legacy path. Any LangChain-Core-style runnable
      (e.g. ``ChatOpenAI(...)``) is wired into a ``prompt | llm | StrOutputParser``
      chain. This is preserved for backward compatibility and for tests that
      stub out the LLM with a ``RunnableLambda``.

    Pass ``instructions`` to layer task-specific guidance on top of the prompt
    without replacing it — the text is merged into the prompt's system message
    on every call (e.g. ``instructions="Respond in French."``).

    Optionally consults `ActRouter <https://r93hdy7zti.execute-api.us-east-1.amazonaws.com>`_ to
    pick the best LLM for the wrapped function. The chosen model is cached in a
    *function inventory file* (see :class:`~actdecor.inventory.FunctionInventory`)
    keyed by a stable fingerprint of the function, so future calls reuse it
    without another round-trip to ActRouter.

    ``router_url`` defaults to the hosted ActRouter, which is only called when
    an API key is configured (``router_api_key=`` or ``ACTROUTER_API_KEY``).
    Any other address (``router_url=`` or ``ACTROUTER_URL``, e.g. a
    self-hosted deployment) is called with or without a key, and
    ``router_client=`` is always used. Otherwise, or with ``router_url=None``,
    the function runs on the model it was given, with no calls to ActRouter.
    """

    def __init__(
        self,
        llm: Any = None,
        *,
        model: Optional[str] = None,
        completion_params: Optional[Mapping[str, Any]] = None,
        return_mode: ReturnMode = "object",
        result_key: str = "value",
        summary_key: str = "commentary",
        metadata_key: str = "info",
        max_chars: int = 12000,
        include_args: bool = True,
        fail_silently: bool = False,
        prompt: Optional[ChatPromptTemplate] = None,
        instructions: Optional[str] = None,
        # --- ActRouter integration ---------------------------------------
        router_url: Optional[str] = DEFAULT_URL,
        router_api_key: Optional[str] = None,
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
        if llm is None and model is None:
            raise ValueError(
                "ActLLMFunctionSummarizer requires either model='provider/model' "
                "(unified endpoint) or llm=<runnable>"
            )
        if llm is not None and model is not None:
            raise ValueError(
                "Pass either model='provider/model' or llm=<runnable>, not both"
            )

        self.llm = llm
        self.model = model
        self.completion_params: dict[str, Any] = dict(completion_params or {})
        self.return_mode = return_mode
        self.result_key = result_key
        self.summary_key = summary_key
        self.metadata_key = metadata_key
        self.max_chars = max_chars
        self.include_args = include_args
        self.fail_silently = fail_silently
        self.instructions = instructions

        # ActRouter / inventory configuration.
        if router_client is None and router_url:
            if is_hosted(router_url):
                # ACTROUTER_URL still takes precedence over the default.
                router_url = configured_url() or router_url
            # The hosted API rejects requests without a key, so skip it unless one is set.
            if configured_api_key(router_api_key) or not is_hosted(router_url):
                router_client = ActRouterClient(router_url, api_key=router_api_key, router=router)
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

        # Legacy chain is only used when an ``llm`` instance was supplied.
        if self.llm is not None:
            self.chain = self.prompt | self.llm | StrOutputParser()
        else:
            self.chain = None

    # -- commentary generation ---------------------------------------------
    def _inject_instructions(self, messages: list) -> list:
        """Merge ``self.instructions`` into the system message of ``messages``.
        The extra guidance is appended to the first system message (so the
        message order stays ``system, human, ...``); when the prompt has no
        system message a fresh one is prepended. A no-op when no instructions
        were configured.
        """

        if not self.instructions:
            return messages
        addition = f"Additional instructions:\n{self.instructions.strip()}"
        merged: list = []
        injected = False
        for msg in messages:
            if not injected and isinstance(msg, SystemMessage):
                merged.append(SystemMessage(content=f"{msg.content}\n\n{addition}"))
                injected = True
            else:
                merged.append(msg)
        if not injected:
            merged.insert(0, SystemMessage(content=addition))
        return merged

    def _build_messages(
        self,
        *,
        function_name: str,
        arguments_text: str,
        return_value_text: str,
    ) -> list:
        """Format the prompt template and splice in any extra instructions."""

        messages = self.prompt.format_messages(
            function_name=function_name,
            arguments=arguments_text,
            return_value=return_value_text,
        )
        return self._inject_instructions(messages)


    def _generate_commentary(
        self,
        active_model: Optional[str],
        *,
        function_name: str,
        arguments_text: str,
        return_value_text: str,
    ) -> str:
        """Run the summarization step using whichever LLM path is configured."""

        messages = self._build_messages(
            function_name=function_name,
            arguments_text=arguments_text,
            return_value_text=return_value_text,
        )

        if self.model is not None:
            model_to_use = active_model or self.model
            response = unified_completion(
                model=model_to_use,
                messages=messages,
                **self.completion_params,
            )
            choices = response.get("choices") or []
            if not choices:
                return ""
            message = choices[0].get("message") or {}
            return message.get("content") or ""

        output = self.llm.invoke(messages)
        return StrOutputParser().invoke(output)

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

    def _apply_routing(self, routing: dict[str, Any], state: dict[str, Any]) -> None:
        """Push the routed model onto the active LLM target (llm obj or model str)."""

        chosen = routing.get("model")
        if not chosen or not self.apply_model:
            return
        if self.model is not None:
            state["active_model"] = chosen
            routing["applied"] = True
        else:
            routing["applied"] = apply_model_to_llm(self.llm, chosen)

    def _provider_label(self) -> str:
        if self.llm is not None:
            return self.llm.__class__.__name__
        provider = _provider_from_model(self.model or "")
        return provider or "actllminfer.completion"

    # -- decoration ---------------------------------------------------------

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        # Per-decorated-function routing state, resolved lazily on first call.
        _state: dict[str, Any] = {
            "resolved": False,
            "routing": None,
            "active_model": self.model,
        }

        def _ensure_routing() -> dict[str, Any]:
            if not _state["resolved"]:
                _state["resolved"] = True
                routing = self.resolve_model(func)
                self._apply_routing(routing, _state)
                _state["routing"] = routing
            return _state["routing"]

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            routing = _ensure_routing()
            # Re-point the (possibly shared) llm at this function's model on
            # every call so concurrent decorators sharing one llm don't clobber
            # each other. (No-op for the unified ``model=`` path — the model
            # string lives in ``_state``.)
            if (
                self.llm is not None
                and routing.get("model")
                and self.apply_model
            ):
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

                commentary = self._generate_commentary(
                    _state.get("active_model"),
                    function_name=func.__name__,
                    arguments_text=arguments_text,
                    return_value_text=return_value_text,
                )

                if self.llm is not None:
                    actual_model = read_model_from_llm(self.llm)
                else:
                    actual_model = _state.get("active_model")

                info = {
                    "function_name": func.__name__,
                    "summarized": True,
                    "provider": self._provider_label(),
                    "model": actual_model or routing.get("model"),
                    "recommended_model": routing.get("model"),
                    "routing": _routing_info(routing),
                }

            except Exception as exc:
                if not self.fail_silently:
                    raise

                commentary = f"LLM summarization failed: {exc}"
                if self.llm is not None:
                    actual_model = read_model_from_llm(self.llm)
                else:
                    actual_model = _state.get("active_model")
                info = {
                    "function_name": func.__name__,
                    "summarized": False,
                    "error": str(exc),
                    "model": actual_model or routing.get("model"),
                    "recommended_model": routing.get("model"),
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
    llm: Any = None,
    *,
    model: Optional[str] = None,
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Convenience decorator factory for :class:`ActLLMFunctionSummarizer`.

    Use ``model="provider/model"`` to dispatch through the unified
    ``actllminfer.completion()`` endpoint, or pass a runnable ``llm=`` for the
    legacy chain-style path. Exactly one must be provided.Pass ``instructions="..."`` to
    add task-specific guidance to the prompt's system message.
    """

    return ActLLMFunctionSummarizer(llm=llm, model=model, **kwargs)
