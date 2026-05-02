from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Literal, Optional

from actllminfer.output_parsers import StrOutputParser
from actllminfer.prompts import ChatPromptTemplate

ReturnMode = Literal["summary", "original", "tuple", "dict", "object"]


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


class ActLLMFunctionSummarizer:
    """Decorator class that adds LLM commentary to wrapped function outputs."""

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
    ):
        self.llm = llm
        self.return_mode = return_mode
        self.result_key = result_key
        self.summary_key = summary_key
        self.metadata_key = metadata_key
        self.max_chars = max_chars
        self.include_args = include_args
        self.fail_silently = fail_silently

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

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
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
                }

            except Exception as exc:
                if not self.fail_silently:
                    raise

                commentary = f"LLM summarization failed: {exc}"
                info = {
                    "function_name": func.__name__,
                    "summarized": False,
                    "error": str(exc),
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

        return wrapper


def summarize_with_llm(
    llm: Any,
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Convenience decorator factory for :class:`LangChainFunctionSummarizer`."""

    return ActLLMFunctionSummarizer(llm=llm, **kwargs)
