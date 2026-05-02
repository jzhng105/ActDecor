# ActDecor

AI commentary layer for Python pipelines.

ActDecor provides a decorator wrapper that can be attached to any Python function. It executes your function normally, then sends the return value (and optional arguments) to a ActLLM-compatible LLM and returns both the original value and a concise summary.

## Installation

```bash
pip install -e .
```

## Quick start

```python
from actllminfer import ChatOpenAI
from actdecor import summarize_with_llm

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@summarize_with_llm(llm, return_mode="object")
def run_report(region: str) -> dict:
    return {
        "region": region,
        "revenue": 128_000,
        "cost": 92_500,
        "delta_vs_plan": -8.1,
    }

result = run_report("US-West")

print(result.value)       # original function output
print(result.commentary)  # LLM-generated summary
print(result.info)        # metadata
```

## Return modes

- `"summary"`: Return only LLM commentary (`str`).
- `"original"`: Return only original function output.
- `"tuple"`: Return `(original_output, commentary)`.
- `"dict"`: Return `{value, commentary, info}` (customizable keys).
- `"object"` (default): Return `SummaryResult` dataclass.

## API

- `summarize_with_llm(llm, **kwargs)` — decorator factory.
- `ActLLMFunctionSummarizer(llm, **kwargs)` — decorator class.
- `serialize_for_llm(value, max_chars=12000)` — serialization helper.
- `SummaryResult` — dataclass response object.
