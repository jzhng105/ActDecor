"""ActDecor: add LLM commentary to Python function outputs."""

from .decorator import (
    LangChainFunctionSummarizer,
    ReturnMode,
    SummaryResult,
    serialize_for_llm,
    summarize_with_llm,
)

__all__ = [
    "LangChainFunctionSummarizer",
    "ReturnMode",
    "SummaryResult",
    "serialize_for_llm",
    "summarize_with_llm",
]
