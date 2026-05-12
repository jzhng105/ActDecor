"""Detect which major LLM vendors are usable from the current environment.

This lets ``actdecor`` restrict the candidate-model list it sends to ActRouter
so the router only ever picks a model the caller can actually invoke. Vendor
keys here match ActRouter's catalog ``family`` field (``openai``, ``anthropic``,
``google``, ``deepseek``, ``xai``, ``mistral``, ``cohere``, ``meta``,
``alibaba``, ``moonshot``, ``zhipuai``, ``minimax``, ``nvidia_nim``).
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from typing import Any, Iterable, Mapping, Optional

log = logging.getLogger("actdecor")

# Vendor (== ActRouter catalog "family") -> environment variables whose presence
# indicates the vendor's API is configured. The first matching, non-empty value
# is also used as the API key for the optional live probe.
VENDOR_API_KEYS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "xai": ("XAI_API_KEY", "GROK_API_KEY"),
    "mistral": ("MISTRAL_API_KEY",),
    "cohere": ("COHERE_API_KEY", "CO_API_KEY"),
    "meta": ("LLAMA_API_KEY", "META_API_KEY", "TOGETHER_API_KEY", "GROQ_API_KEY", "FIREWORKS_API_KEY"),
    "alibaba": ("DASHSCOPE_API_KEY", "ALIBABA_API_KEY", "QWEN_API_KEY"),
    "moonshot": ("MOONSHOT_API_KEY",),
    "zhipuai": ("ZHIPUAI_API_KEY", "ZHIPU_API_KEY", "GLM_API_KEY"),
    "minimax": ("MINIMAX_API_KEY",),
    "nvidia_nim": ("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY", "NIM_API_KEY"),
}

# Offline fallback model lists (mirror of actrouter's models_catalog.yaml), used
# only when ActRouter's ``GET /v1/models`` cannot be consulted.
FALLBACK_FAMILY_MODELS: dict[str, tuple[str, ...]] = {
    "openai": ("gpt-5.5", "gpt-5.4-mini", "gpt-5-mini"),
    "anthropic": ("claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"),
    "google": ("gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-2.5-pro"),
    "deepseek": ("deepseek-chat", "deepseek-reasoner", "deepseek-coder"),
    "xai": ("grok-4", "grok-4-fast-reasoning", "grok-3"),
    "mistral": ("mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"),
    "cohere": ("command-a-03-2025", "command-r-plus-08-2024", "command-r-08-2024"),
    "meta": ("llama-4-maverick", "llama-4-scout", "llama-3.3-70b-instruct"),
    "alibaba": ("qwen3-next-80b-a3b-instruct", "qwen3-vl-235b-a22b-instruct", "qwen-max"),
    "moonshot": ("kimi-k2.5", "kimi-k2-thinking", "kimi-k2-0711-preview"),
    "zhipuai": ("glm-4.6", "glm-4.5", "glm-4.5-air"),
    "minimax": ("minimax-m2.1", "minimax-m2", "minimax-m1"),
    "nvidia_nim": (
        "meta/llama-4-maverick-17b-128e-instruct",
        "meta/llama-3.3-70b-instruct",
        "nvidia/llama-3.1-nemotron-70b-instruct",
    ),
}

# Lightweight "is the API reachable and the key accepted?" probes. Vendors not
# listed here are considered available purely based on env-var presence.
_PROBES: dict[str, dict[str, Any]] = {
    "openai": {"url": "https://api.openai.com/v1/models", "auth": "bearer"},
    "anthropic": {
        "url": "https://api.anthropic.com/v1/models",
        "extra_headers": lambda key: {"x-api-key": key, "anthropic-version": "2023-06-01"},
    },
    "google": {"url": "https://generativelanguage.googleapis.com/v1beta/models?key={key}"},
    "deepseek": {"url": "https://api.deepseek.com/models", "auth": "bearer"},
    "xai": {"url": "https://api.x.ai/v1/models", "auth": "bearer"},
    "mistral": {"url": "https://api.mistral.ai/v1/models", "auth": "bearer"},
    "cohere": {"url": "https://api.cohere.com/v1/models", "auth": "bearer"},
    "moonshot": {"url": "https://api.moonshot.cn/v1/models", "auth": "bearer"},
    "alibaba": {"url": "https://dashscope.aliyun.com/compatible-mode/v1/models", "auth": "bearer"},
    "zhipuai": {"url": "https://open.bigmodel.cn/api/paas/v4/models", "auth": "bearer"},
    "nvidia_nim": {"url": "https://integrate.api.nvidia.com/v1/models", "auth": "bearer"},
}


def _merged_env_map(extra_env: Optional[Mapping[str, Iterable[str]]]) -> dict[str, tuple[str, ...]]:
    merged: dict[str, tuple[str, ...]] = {k: tuple(v) for k, v in VENDOR_API_KEYS.items()}
    if extra_env:
        for vendor, vars_ in extra_env.items():
            extra = (vars_,) if isinstance(vars_, str) else tuple(vars_)
            merged[vendor] = tuple(dict.fromkeys(merged.get(vendor, ()) + extra))
    return merged


def _api_key_for(vendor: str, env: Mapping[str, str], env_map: Mapping[str, Iterable[str]]) -> Optional[str]:
    for var in env_map.get(vendor, ()):  # type: ignore[union-attr]
        value = env.get(var)
        if value and value.strip():
            return value.strip()
    return None


def probe_vendor(
    vendor: str,
    *,
    api_key: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout: float = 3.0,
    extra_env: Optional[Mapping[str, Iterable[str]]] = None,
) -> bool:
    """Return ``True`` if ``vendor``'s API answers an authenticated request.

    For vendors without a known probe endpoint this degrades to "is an API key
    configured?". Never raises.
    """

    env = os.environ if env is None else env
    env_map = _merged_env_map(extra_env)
    key = api_key or _api_key_for(vendor, env, env_map)
    spec = _PROBES.get(vendor)
    if spec is None:
        return bool(key)

    url_template = spec["url"]
    needs_key_in_url = "{key}" in url_template
    if not key:
        # Every probe here needs credentials one way or another.
        return False

    url = url_template.format(key=key) if needs_key_in_url else url_template
    headers: dict[str, str] = {"accept": "application/json", "user-agent": "actdecor-vendor-probe"}
    if spec.get("auth") == "bearer":
        headers["authorization"] = f"Bearer {key}"
    extra_headers = spec.get("extra_headers")
    if callable(extra_headers):
        try:
            headers.update(extra_headers(key))
        except Exception:  # pragma: no cover - defensive
            pass

    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return True  # urllib raises on >= 400, follows 3xx => this is 2xx
    except urllib.error.HTTPError as exc:
        # 401/403 => key rejected; other errors => be conservative. 429 means
        # the vendor is clearly up, just rate limiting us.
        return exc.code == 429
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
    except Exception:  # pragma: no cover - defensive
        return False


def detect_available_vendors(
    *,
    env: Optional[Mapping[str, str]] = None,
    probe: bool = False,
    timeout: float = 3.0,
    extra_env: Optional[Mapping[str, Iterable[str]]] = None,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> set[str]:
    """Return the set of vendor/family names usable from this environment.

    By default availability means "an API-key environment variable is set". With
    ``probe=True`` each candidate vendor is additionally pinged (best effort,
    short timeout) and only those that respond are kept.
    """

    env = os.environ if env is None else env
    env_map = _merged_env_map(extra_env)
    vendors = set(include) if include is not None else set(env_map.keys())
    if exclude:
        vendors -= set(exclude)

    available: set[str] = set()
    for vendor in sorted(vendors):
        key = _api_key_for(vendor, env, env_map)
        if not key:
            continue  # no credentials => can't use it (even with probing)
        if probe:
            if probe_vendor(vendor, api_key=key, env=env, timeout=timeout, extra_env=extra_env):
                available.add(vendor)
        else:
            available.add(vendor)
    return available


def available_candidate_models(
    available_vendors: Iterable[str],
    *,
    catalog_models: Optional[Iterable[Mapping[str, Any]]] = None,
    router_client: Any = None,
) -> list[str]:
    """Translate available vendors into concrete candidate model ids.

    ``catalog_models`` is an iterable of dicts with at least ``id`` and
    ``family`` keys (the shape ActRouter's ``GET /v1/models`` returns). If it is
    not given, the catalog is fetched from ``router_client`` when one is
    supplied, otherwise the built-in :data:`FALLBACK_FAMILY_MODELS` map is used.
    """

    wanted = {v for v in available_vendors}
    if not wanted:
        return []

    models: list[Mapping[str, Any]] = []
    if catalog_models is not None:
        models = list(catalog_models)
    elif router_client is not None:
        try:
            models = list(router_client.list_models())
        except Exception as exc:  # network error, missing method, etc.
            log.warning("Could not fetch ActRouter model catalog: %s", exc)
            models = []

    if models:
        ids = [str(m["id"]) for m in models if m.get("family") in wanted and m.get("id")]
        if ids:
            return ids

    fallback: list[str] = []
    for vendor in sorted(wanted):
        fallback.extend(FALLBACK_FAMILY_MODELS.get(vendor, ()))
    return fallback
