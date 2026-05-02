"""ActDecor: add LLM commentary to Python function outputs."""

from .decorator import (
    ActLLMFunctionSummarizer,
    ReturnMode,
    SummaryResult,
    serialize_for_llm,
    summarize_with_llm,
)

__all__ = [
    "ActLLMFunctionSummarizer",
    "ReturnMode",
    "SummaryResult",
    "serialize_for_llm",
    "summarize_with_llm",
]
