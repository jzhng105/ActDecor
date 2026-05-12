"""ActDecor: add LLM commentary to Python function outputs.

Optionally integrates with `ActRouter <https://github.com/jzhng105/actrouter>`_:
the decorator can detect which major LLM vendors are usable from the current
environment, ask ActRouter which of *those* models best fits a wrapped function,
and cache that choice in a *function inventory file* so future calls reuse it.
"""

from .decorator import (
    ActLLMFunctionSummarizer,
    ReturnMode,
    SummaryResult,
    apply_model_to_llm,
    serialize_for_llm,
    summarize_with_llm,
)
from .inventory import (
    DEFAULT_INVENTORY_ENV,
    DEFAULT_INVENTORY_FILENAME,
    FunctionInventory,
    default_inventory_path,
)
from .metadata import build_function_metadata, function_fingerprint
from .router_client import ActRouterClient, ActRouterError
from .vendors import (
    FALLBACK_FAMILY_MODELS,
    VENDOR_API_KEYS,
    available_candidate_models,
    detect_available_vendors,
    probe_vendor,
)

__all__ = [
    "ActLLMFunctionSummarizer",
    "ReturnMode",
    "SummaryResult",
    "apply_model_to_llm",
    "serialize_for_llm",
    "summarize_with_llm",
    "ActRouterClient",
    "ActRouterError",
    "FunctionInventory",
    "default_inventory_path",
    "DEFAULT_INVENTORY_ENV",
    "DEFAULT_INVENTORY_FILENAME",
    "build_function_metadata",
    "function_fingerprint",
    "detect_available_vendors",
    "probe_vendor",
    "available_candidate_models",
    "VENDOR_API_KEYS",
    "FALLBACK_FAMILY_MODELS",
]
