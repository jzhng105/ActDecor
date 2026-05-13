# ActDecor

AI commentary layer for Python pipelines.

ActDecor provides a decorator wrapper that can be attached to any Python function. It executes your function normally, then sends the return value (and optional arguments) to an LLM and returns both the original value and a concise summary.

Under the hood ActDecor dispatches through the **unified endpoint** from
[ActLLMInfer](https://github.com/jzhng105/actllminfer) — one
``completion()`` call across every supported provider, selected by a single
``provider/model`` string (LiteLLM-style).

It can also ask [ActRouter](https://github.com/jzhng105/actrouter) which LLM is the best fit for the wrapped function, cache that choice in a *function inventory file*, and reuse it on subsequent calls.

## Installation

```bash
pip install -e .
```

## Quick start (unified endpoint)

```python
from actdecor import summarize_with_llm

@summarize_with_llm(model="openai/gpt-4o-mini", completion_params={"temperature": 0})
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
print(result.info)        # metadata (model, routing, etc.)
```

The ``model`` argument accepts any string ``actllminfer.completion`` accepts:
``"openai/gpt-4o-mini"``, ``"anthropic/claude-haiku-4-5"``,
``"deepseek/deepseek-chat"``, ``"kimi/moonshot-v1-8k"``, ``"zhipuai/glm-4-plus"``,
``"nvidia_nim/meta/llama-3.3-70b-instruct"``, etc. Per-call generation knobs
(``temperature``, ``max_tokens``, ``tools``, ``response_format``, ``stop`` …)
go in ``completion_params``; everything else (``api_key``, ``base_url``,
``request_timeout`` …) is forwarded to the underlying client the first time it
is constructed.

### Legacy ``llm=`` (chain-style) path

If you prefer to construct a provider class yourself — or you need to stub the
LLM out in tests — pass ``llm=`` instead of ``model=``. ActDecor wires it into
a ``prompt | llm | StrOutputParser`` chain just like before:

```python
from actllminfer import ChatOpenAI
from actdecor import summarize_with_llm

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@summarize_with_llm(llm)
def run_report(region: str) -> dict:
    ...
```

Exactly one of ``model=`` / ``llm=`` must be provided.

## Return modes

- `"summary"`: Return only LLM commentary (`str`).
- `"original"`: Return only original function output.
- `"tuple"`: Return `(original_output, commentary)`.
- `"dict"`: Return `{value, commentary, info}` (customizable keys).
- `"object"` (default): Return `SummaryResult` dataclass.

## Routing with ActRouter

Point the decorator at a running [ActRouter](https://github.com/jzhng105/actrouter)
deployment and it will pick the model for you. ActRouter only ever receives the
function's **static metadata** (name, signature, docstring, type hints, optional
budgets) — never call arguments or return values.

```python
from actdecor import summarize_with_llm

@summarize_with_llm(
    model="openai/gpt-4o-mini",                # initial / fallback model
    router_url="https://<your-actrouter>",     # ActRouter base URL
    router="heuristic",                        # routing strategy
    route_hints={"task_kind": "summarization", "cost_budget_usd": 0.05},
)
def summarize_article(text: str) -> str:
    """Summarize a news article in 3 bullet points."""
    ...

result = summarize_article(article_text)
print(result.info["routing"])
# {'model': 'gemini-2-5-flash', 'source': 'actrouter', 'applied': True, ...}
```

When the decorator uses ``model="provider/model"``, "applying" a routed model
simply swaps the string used for subsequent ``completion()`` calls — no client
mutation involved. When the legacy ``llm=`` path is used, ``apply_model_to_llm``
patches the runnable's ``model`` / ``model_name`` / ``model_id`` attribute (or
calls ``set_model`` if present).

On the **first call** ActDecor:

1. detects which major LLM vendors are usable here and limits candidates to them;
2. looks the function up in the inventory file (a miss the first time);
3. calls `POST /v1/route` on ActRouter with the function metadata;
4. records the recommended `chosen_model` in the inventory file, keyed by a
   stable fingerprint of the function;
5. points the active LLM target at that model (a string swap for ``model=``,
   or an attribute patch for ``llm=``).

On **every later call** (including in a fresh process) it reads the model from
the inventory file and skips the ActRouter round-trip.

### The function inventory file

A plain JSON hash map of `function fingerprint -> recommendation`:

```json
{
  "version": 1,
  "functions": {
    "9f1c2a3b4d5e6f70": {
      "function_name": "summarize_article",
      "qualname": "summarize_article",
      "module": "news.pipeline",
      "recommended_model": "gemini-2-5-flash",
      "router": "heuristic",
      "estimated_cost_usd": 0.0017,
      "estimated_latency_ms": 330,
      "updated_at": "2026-05-12T12:00:00+00:00"
    }
  }
}
```

- Default location: `./actdecor_inventory.json`, overridable with the
  `ACTDECOR_INVENTORY` environment variable or the `inventory=` argument
  (a path or a `FunctionInventory` instance).
- You can commit the file, hand-edit it, or pre-populate it to pin a model
  without ever calling ActRouter.

### Restricting to vendors you can actually use

Before routing, ActDecor figures out which **major LLM vendors** are usable from
the current environment and tells ActRouter to only consider those vendors'
models. By default a vendor counts as "available" when its API-key environment
variable is set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` /
`GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `XAI_API_KEY`, `MISTRAL_API_KEY`,
`COHERE_API_KEY`, `DASHSCOPE_API_KEY`, `MOONSHOT_API_KEY`, `ZHIPUAI_API_KEY`,
`MINIMAX_API_KEY`, `NVIDIA_API_KEY`, …). Set `probe_vendors=True` to also ping
each vendor's `/models` endpoint and drop ones whose key is rejected.

```python
@summarize_with_llm(
    model="openai/gpt-4o-mini",
    router_url="https://<your-actrouter>",
    probe_vendors=True,                 # actually call each vendor's API
    # vendors=["openai", "anthropic"],  # or pin the set explicitly
    # vendor_env={"meta": ["TOGETHER_API_KEY"]},  # extra key env vars
)
def summarize_article(text: str) -> str:
    """Summarize a news article in 3 bullet points."""
    ...

print(summarize_article("...").info["routing"]["available_vendors"])  # ['anthropic', 'openai']
print(summarize_article("...").info["routing"]["candidate_models"])    # ['gpt-5-mini', 'claude-haiku-4-5', ...]
```

The candidate ids are taken from ActRouter's `GET /v1/models` (filtered by
vendor `family`), falling back to a built-in catalog snapshot if that call
fails. If no vendor is detected, ActDecor sends no restriction and lets
ActRouter pick from its full catalog. Set `restrict_to_available_vendors=False`
to disable this entirely, or pass an explicit `candidate_models=[...]` to
bypass detection.

You can also probe / map vendors directly:

```python
from actdecor import detect_available_vendors, available_candidate_models, ActRouterClient

vendors = detect_available_vendors(probe=True)            # {'openai', 'anthropic'}
client = ActRouterClient("https://<your-actrouter>")
candidates = available_candidate_models(vendors, router_client=client)
```

You can also drive routing manually:

```python
from actdecor import ActRouterClient, FunctionInventory

client = ActRouterClient("https://<your-actrouter>")
model = client.recommend_model(
    summarize_article,
    task_kind="summarization",
    candidate_models=candidates,
)

FunctionInventory().record(summarize_article, model, router="heuristic")
```

### Decorator options

| Argument             | Default        | Purpose                                                        |
| -------------------- | -------------- | -------------------------------------------------------------- |
| `model`              | `None`         | Unified-endpoint model id, e.g. ``"openai/gpt-4o-mini"``.      |
| `llm`                | `None`         | Legacy: a LangChain-Core-style runnable. Exclusive with ``model=``. |
| `completion_params`  | `None`         | Per-call kwargs forwarded to ``actllminfer.completion`` (e.g. ``temperature``, ``max_tokens``, ``tools``). |
| `router_url`         | `None`         | ActRouter base URL. Enables routing when set.                  |
| `router_client`      | `None`         | Pre-built `ActRouterClient` (overrides `router_url`).          |
| `router`             | `"heuristic"`  | Strategy: `heuristic` \| `rules` \| `random` \| `smallest` \| `largest`. |
| `route_top_k`        | `3`            | How many ranked candidates to request.                         |
| `candidate_models`   | `None`         | Explicit whitelist of model ids (bypasses vendor detection).   |
| `route_hints`        | `None`         | Extra static metadata (`task_kind`, `tags`, token estimates, `latency_budget_ms`, `cost_budget_usd`, `privacy_tier`). |
| `auto_route`         | `True`         | Master switch for routing + inventory lookups.                 |
| `apply_model`        | `True`         | Whether to point the active LLM target at the chosen model.    |
| `use_inventory`      | `True`         | Whether to read/write the inventory file.                      |
| `inventory`          | `None`         | Inventory file path or `FunctionInventory` instance.           |
| `route_fail_silently`| `True`         | Swallow ActRouter errors and keep the original model.          |
| `restrict_to_available_vendors` | `True` | Limit candidate models to vendors usable from this environment. |
| `probe_vendors`      | `False`        | Also ping each vendor's API (not just check env vars).         |
| `vendor_probe_timeout` | `3.0`        | Per-vendor probe timeout, seconds.                             |
| `vendors`            | `None`         | Explicit vendor/family set; skips auto-detection.              |
| `vendor_env`         | `None`         | Extra `{vendor: [ENV_VAR, ...]}` mappings for key detection.   |

## API

- `summarize_with_llm(llm=None, *, model=None, **kwargs)` — decorator factory.
- `ActLLMFunctionSummarizer(llm=None, *, model=None, **kwargs)` — decorator class.
- `serialize_for_llm(value, max_chars=12000)` — serialization helper.
- `SummaryResult` — dataclass response object.
- `ActRouterClient(base_url, ...)` — HTTP client for ActRouter (`route`, `recommend_model`, `list_models`).
- `ActRouterError` — raised when ActRouter is unreachable or errors.
- `FunctionInventory(path=None)` — JSON-backed function → model hash map.
- `build_function_metadata(func, ...)` — static, privacy-safe metadata for ActRouter.
- `function_fingerprint(func)` — stable inventory key for a function.
- `apply_model_to_llm(llm, model)` / `read_model_from_llm(llm)` — best-effort model setter/getter for legacy runnables.
- `default_inventory_path()` — resolves the default inventory file location.
- `detect_available_vendors(*, probe=False, ...)` — vendor/family names usable here.
- `probe_vendor(vendor, *, api_key=None, ...)` — does this vendor's API answer?
- `available_candidate_models(vendors, *, catalog_models=None, router_client=None)` — vendors → model ids.
- `VENDOR_API_KEYS`, `FALLBACK_FAMILY_MODELS` — the built-in vendor maps.
- `completion`, `acompletion`, `embedding`, `aembedding`, `Router` — re-exported from `actllminfer` for direct use.
