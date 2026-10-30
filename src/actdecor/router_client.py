"""ActDecor's client for the ActRouter API.

ActRouter exposes ``POST /v1/route`` which takes *function metadata only* and
returns the recommended LLM for that function. This client builds the metadata
payload from a Python callable and sends it through
:class:`actrouter_client.HttpRouter`, so ActDecor shares the transport (API key
header, plan-limit and bad-key messages, refusing redirects) with the other
ActRouter tools.

Settings not passed explicitly come from the environment, as they do for
``actrouter-client``:

* ``ACTROUTER_API_KEY`` -- your API key, sent as ``x-api-key``.
* ``ACTROUTER_URL`` -- the API's base URL; unset means the hosted ActRouter.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from actrouter_client import DEFAULT_URL, ActRouterError, HttpRouter
from actrouter_client.client import env

from .metadata import build_function_metadata

DEFAULT_TIMEOUT = 10.0
DEFAULT_ROUTER = "heuristic"

__all__ = [
    "ActRouterClient",
    "ActRouterError",
    "DEFAULT_URL",
    "configured_api_key",
    "configured_url",
    "is_hosted",
]


def configured_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """``api_key`` if given, else ``ACTROUTER_API_KEY``, else None."""

    return api_key or env("ACTROUTER_API_KEY")


def configured_url(base_url: Optional[str] = None) -> Optional[str]:
    """``base_url`` if given, else ``ACTROUTER_URL``, else None."""

    return base_url or env("ACTROUTER_URL")


def is_hosted(url: str) -> bool:
    """Whether ``url`` is the hosted ActRouter, which requires an API key."""

    return url.rstrip("/") == DEFAULT_URL.rstrip("/")


class ActRouterClient:
    """Thin client for ActRouter's ``/v1/route`` (and ``/v1/models``) endpoints."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        router: str = DEFAULT_ROUTER,
        timeout: float = DEFAULT_TIMEOUT,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.base_url = (configured_url(base_url) or DEFAULT_URL).rstrip("/")
        self.router = router
        self.timeout = timeout
        merged: dict[str, str] = {}
        if key := configured_api_key(api_key):
            merged["x-api-key"] = key
        if headers:
            merged.update(headers)
        self._http = HttpRouter(self.base_url, timeout=timeout, headers=merged)
        self.headers = self._http.headers

    # -- public API ---------------------------------------------------------

    def route(
        self,
        func: Callable[..., Any],
        *,
        router: Optional[str] = None,
        candidate_models: Optional[list[str]] = None,
        top_k: int = 3,
        **metadata_hints: Any,
    ) -> dict[str, Any]:
        """Ask ActRouter which model ``func`` should use; returns the full response."""

        return self._http.route_function(
            build_function_metadata(func, **metadata_hints),
            router=router or self.router,
            candidate_models=list(candidate_models) if candidate_models is not None else None,
            top_k=top_k,
        )

    def recommend_model(self, func: Callable[..., Any], **kwargs: Any) -> str:
        """Convenience wrapper returning only ``chosen_model``."""

        response = self.route(func, **kwargs)
        model = response.get("chosen_model")
        if not model:
            raise ActRouterError(f"ActRouter response did not include 'chosen_model': {response!r}")
        return model

    def list_models(self) -> list[dict[str, Any]]:
        return self._http.list_models()
