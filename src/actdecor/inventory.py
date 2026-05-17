"""The function inventory file: a persistent hash map of function -> model.

When ActRouter recommends an LLM for a decorated function, ``actdecor`` records
the choice here keyed by a stable :func:`function_fingerprint`. On later runs
(even in a fresh process) the decorator looks the function up in this file and
reuses the recommended model by default, instead of calling ActRouter again.

The file is plain JSON so it can be inspected, edited, or committed::

    {
      "version": 1,
      "functions": {
        "9f1c2a3b4d5e6f70": {
          "function_name": "run_report",
          "qualname": "run_report",
          "module": "reports.sales",
          "recommended_model": "gemini-2-5-flash",
          "router": "heuristic",
          "estimated_cost_usd": 0.0017,
          "estimated_latency_ms": 330,
          "updated_at": "2026-05-12T12:00:00+00:00"
        }
      }
    }
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Union

from .metadata import function_fingerprint

PathLike = Union[str, os.PathLike]

INVENTORY_VERSION = 1
DEFAULT_INVENTORY_ENV = "ACTDECOR_INVENTORY"
DEFAULT_INVENTORY_FILENAME = "actdecor_inventory.json"


def default_inventory_path() -> Path:
    """Path used when no explicit inventory location is given.

    Honors the ``ACTDECOR_INVENTORY`` environment variable; otherwise
    ``./actdecor_inventory.json`` in the current working directory.
    """

    env = os.environ.get(DEFAULT_INVENTORY_ENV)
    if env:
        return Path(env).expanduser()
    return Path.cwd() / DEFAULT_INVENTORY_FILENAME


def _empty() -> dict[str, Any]:
    return {"version": INVENTORY_VERSION, "functions": {}}


class FunctionInventory:
    """A JSON-backed mapping ``function fingerprint -> recommendation record``."""

    def __init__(self, path: Optional[PathLike] = None) -> None:
        self.path = Path(path).expanduser() if path is not None else default_inventory_path()
        self._data = self._load()

    # -- persistence --------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return _empty()
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return _empty()
        if not isinstance(data, dict):
            return _empty()
        functions = data.get("functions")
        if not isinstance(functions, dict):
            functions = {}
        return {"version": data.get("version", INVENTORY_VERSION), "functions": functions}

    def reload(self) -> "FunctionInventory":
        self._data = self._load()
        return self

    def save(self) -> None:
        """Atomically write the inventory to disk."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # -- accessors ----------------------------------------------------------

    @staticmethod
    def key_for(func: Callable[..., Any]) -> str:
        return function_fingerprint(func)

    def get(self, func: Callable[..., Any]) -> Optional[dict[str, Any]]:
        return self._data["functions"].get(self.key_for(func))

    def get_model(self, func: Callable[..., Any]) -> Optional[str]:
        record = self.get(func)
        return record.get("recommended_model") if record else None

    def __contains__(self, func: object) -> bool:
        return callable(func) and self.key_for(func) in self._data["functions"]

    def as_dict(self) -> dict[str, Any]:
        return {"version": self._data["version"], "functions": dict(self._data["functions"])}

    # -- mutation -----------------------------------------------------------

    def record(
        self,
        func: Callable[..., Any],
        model: str,
        *,
        router: Optional[str] = None,
        estimated_cost_usd: Optional[float] = None,
        estimated_latency_ms: Optional[int] = None,
        ranking: Optional[list[Any]] = None,
        extra: Optional[dict[str, Any]] = None,
        save: bool = True,
    ) -> dict[str, Any]:
        """Store (and by default persist) a model recommendation for ``func``."""

        record: dict[str, Any] = {
            "function_name": getattr(func, "__name__", None) or "anonymous",
            "qualname": getattr(func, "__qualname__", None),
            "module": getattr(func, "__module__", None),
            "recommended_model": model,
            "router": router,
            "estimated_cost_usd": estimated_cost_usd,
            "estimated_latency_ms": estimated_latency_ms,
            "ranking": ranking,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            record.update(extra)
        self._data["functions"][self.key_for(func)] = record
        if save:
            self.save()
        return record

    def set_model(self, func: Callable[..., Any], model: str, *, save: bool = True) -> dict[str, Any]:
        return self.record(func, model, save=save)

    # -- tool registration (OSFI E-23 style inventory) ----------------------

    def register(
        self,
        func: Callable[..., Any],
        registration: dict[str, Any],
        *,
        save: bool = True,
    ) -> dict[str, Any]:
        """Attach an OSFI E-23 style ``tool_registration`` block to ``func``.

        Creates a function record if one does not exist yet (e.g. when routing
        was never run), so the inventory file doubles as a model inventory.
        """

        key = self.key_for(func)
        record = self._data["functions"].get(key)
        if record is None:
            record = {
                "function_name": getattr(func, "__name__", None) or "anonymous",
                "qualname": getattr(func, "__qualname__", None),
                "module": getattr(func, "__module__", None),
            }
            self._data["functions"][key] = record
        record["tool_registration"] = registration
        record["registered_at"] = datetime.now(timezone.utc).isoformat()
        if save:
            self.save()
        return record

    def get_registration(self, func: Callable[..., Any]) -> Optional[dict[str, Any]]:
        record = self.get(func)
        return record.get("tool_registration") if record else None

    def is_registered(self, func: Callable[..., Any]) -> bool:
        return self.get_registration(func) is not None

    def remove(self, func: Callable[..., Any], *, save: bool = True) -> bool:
        existed = self._data["functions"].pop(self.key_for(func), None) is not None
        if existed and save:
            self.save()
        return existed
