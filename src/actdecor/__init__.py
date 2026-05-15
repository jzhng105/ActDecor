"""ActDecor: add LLM commentary to Python function outputs.

ActDecor wraps any Python function with an LLM-generated summary of its return
value. The recommended way to point it at an LLM is the **unified endpoint**
from :mod:`actllminfer` — pass ``model="provider/model"`` and the decorator
dispatches each call through :func:`actllminfer.completion`. The legacy
``llm=<runnable>`` path is preserved for backward compatibility.

Optionally integrates with `ActRouter <https://r93hdy7zti.execute-api.us-east-1.amazonaws.com>`_:
the decorator can detect which major LLM vendors are usable from the current
environment, ask ActRouter which of *those* models best fits a wrapped function,
and cache that choice in a *function inventory file* so future calls reuse it.
"""

from actllminfer import (
    ModelResponse,
    Router,
    StreamingModelResponse,
    acompletion,
    aembedding,
    completion,
    embedding,
)

from .decorator import (
    ActLLMFunctionSummarizer,
    ReturnMode,
    SummaryResult,
    apply_model_to_llm,
    read_model_from_llm,
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
    "read_model_from_llm",
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
    # Re-exported unified endpoint from actllminfer.
    "completion",
    "acompletion",
    "embedding",
    "aembedding",
    "Router",
    "ModelResponse",
    "StreamingModelResponse",
]
