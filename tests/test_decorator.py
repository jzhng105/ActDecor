from __future__ import annotations
from actllminfer import ChatOpenAI
import pandas as pd

import pytest
from actllminfer.runnables import RunnableLambda

from actdecor import ActLLMFunctionSummarizer, SummaryResult, summarize_with_llm


@pytest.fixture
def fake_llm() -> RunnableLambda:
    def _invoke(_prompt_value):
        return "Concise analysis"

    return RunnableLambda(_invoke)


def test_default_object_mode(fake_llm: RunnableLambda):
    @summarize_with_llm(fake_llm)
    def add(a: int, b: int) -> int:
        return a + b

    result = add(2, 3)

    assert isinstance(result, SummaryResult)
    assert result.value == 5
    assert result.commentary == "Concise analysis"
    assert result.info["summarized"] is True


def test_tuple_mode(fake_llm: RunnableLambda):
    @ActLLMFunctionSummarizer(fake_llm, return_mode="tuple")
    def make_payload() -> dict:
        return {"ok": True}

    value, summary = make_payload()

    assert value == {"ok": True}
    assert summary == "Concise analysis"


def test_fail_silently_returns_original_when_llm_breaks():
    broken_llm = RunnableLambda(lambda _: (_ for _ in ()).throw(RuntimeError("boom")))

    @ActLLMFunctionSummarizer(
        broken_llm,
        return_mode="dict",
        fail_silently=True,
    )
    def produce() -> list[int]:
        return [1, 2, 3]

    result = produce()

    assert result["value"] == [1, 2, 3]
    assert "failed" in result["commentary"].lower()
    assert result["info"]["summarized"] is False


def test_invalid_return_mode_raises(fake_llm: RunnableLambda):
    decorator = ActLLMFunctionSummarizer(fake_llm)
    decorator.return_mode = "nope"  # type: ignore[assignment]

    @decorator
    def fn():
        return "x"

    with pytest.raises(ValueError):
        fn()


# -----------------------------
# Full test with LLM
# -----------------------------

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)

@summarize_with_llm(llm, router_url="https://r93hdy7zti.execute-api.us-east-1.amazonaws.com/", probe_vendors=True)
def add(a, b):
    return a + b

result = add(2, 3)

print(result.value)        # 5
print(result.commentary)   # real LLM output

raw_data = [
    {"policy_id": "P001", "premium": "$1,200 ", "province": "ON", "date": "2024-01-15"},
    {"policy_id": "P002", "premium": "$850", "province": "ON", "date": "2024-02-20"},
    {"policy_id": "P003", "premium": None, "province": "QC", "date": "2024-03-10"},
    {"policy_id": "P004", "premium": "$2,500", "province": "ON", "date": "invalid_date"},
]

df_raw = pd.DataFrame(raw_data)


# -----------------------------
# ETL Function
# -----------------------------
@summarize_with_llm(llm, router_url="https://r93hdy7zti.execute-api.us-east-1.amazonaws.com/", probe_vendors=True, route_hints={"task_kind": "summarization", "cost_budget_usd": 0.05})
def run_etl(df: pd.DataFrame, output_path: str = "cleaned_data.csv") -> pd.DataFrame:
    # ---- Transform ----
    df_clean = df.copy()

    # Clean premium
    df_clean["premium"] = (
        df_clean["premium"]
        .astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .str.strip()
    )
    df_clean["premium"] = pd.to_numeric(df_clean["premium"], errors="coerce")

    # Convert date
    df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")

    # Handle missing
    df_clean["premium"] = df_clean["premium"].fillna(0)

    # Feature engineering
    df_clean["premium_band"] = pd.cut(
        df_clean["premium"],
        bins=[-1, 1000, 2000, float("inf")],
        labels=["Low", "Medium", "High"]
    )

    return df_clean

result = run_etl(df_raw)
result.commentary
result.info["routing"]
result.value