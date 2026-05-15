"""A tiny, dependency-free HTTP client for the ActRouter API.

ActRouter exposes ``POST /v1/route`` which takes *function metadata only* and
returns the recommended LLM for that function. This client builds the metadata
payload from a Python callable and calls the endpoint using the standard
library, so ``actdecor`` does not gain a hard dependency on ``requests``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

from .metadata import build_function_metadata

DEFAULT_TIMEOUT = 10.0
DEFAULT_ROUTER = "heuristic"
DEFAULT_URL =  "https://r93hdy7zti.execute-api.us-east-1.amazonaws.com"

class ActRouterError(RuntimeError):
    """Raised when the ActRouter API is unreachable or returns an error."""


class ActRouterClient:
    """Thin client for ActRouter's ``/v1/route`` (and ``/v1/models``) endpoints."""

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        *,
        router: str = DEFAULT_ROUTER,
        timeout: float = DEFAULT_TIMEOUT,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required for ActRouterClient")
        self.base_url = base_url.rstrip("/")
        self.router = router
        self.timeout = timeout
        self.headers = {"content-type": "application/json", "accept": "application/json"}
        if headers:
            self.headers.update(headers)

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

        payload: dict[str, Any] = {
            "function": build_function_metadata(func, **metadata_hints),
            "router": router or self.router,
            "top_k": top_k,
        }
        if candidate_models is not None:
            payload["candidate_models"] = list(candidate_models)
        return self._request("POST", "/v1/route", payload)

    def recommend_model(self, func: Callable[..., Any], **kwargs: Any) -> str:
        """Convenience wrapper returning only ``chosen_model``."""

        response = self.route(func, **kwargs)
        model = response.get("chosen_model")
        if not model:
            raise ActRouterError(f"ActRouter response did not include 'chosen_model': {response!r}")
        return model

    def list_models(self) -> list[dict[str, Any]]:
        response = self._request("GET", "/v1/models", None)
        data = response.get("data")
        return list(data) if isinstance(data, list) else []

    # -- transport ----------------------------------------------------------

    def _request(self, method: str, path: str, payload: Optional[dict[str, Any]]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, headers=self.headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:  # 4xx / 5xx
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:  # pragma: no cover - defensive
                pass
            raise ActRouterError(f"ActRouter returned HTTP {exc.code} for {url}: {body}") from exc
        except urllib.error.URLError as exc:  # DNS / connection / timeout
            raise ActRouterError(f"Could not reach ActRouter at {url}: {exc.reason}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ActRouterError(f"ActRouter returned non-JSON response from {url}: {raw!r}") from exc
        if not isinstance(parsed, dict):
            raise ActRouterError(f"ActRouter returned unexpected payload from {url}: {parsed!r}")
        return parsed
