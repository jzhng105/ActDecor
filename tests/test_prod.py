import numpy as np
from actllminfer import ChatOpenAI, OpenAIEmbeddings, ChatDeepSeek, ChatNVIDIA
from actdecor import summarize_with_llm, completion
from typing import Sequence

llm = ChatOpenAI(
    model="gpt-5.4",
    temperature=0,
)

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
)

llm = ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct",
    temperature=0,
)

response = llm.invoke("test")

response = completion(
    model="nvidia_nim/meta/llama-3.3-70b-instruct",
    messages="test",
)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

@summarize_with_llm(model="nvidia_nim/meta/llama-3.3-70b-instruct", instructions = "The prior accident-quarter loss ratio was 70%, be concise with your response")
def calculate_loss_ratio(earned_premium: float, incurred_losses: float) -> dict:
    loss_ratio = incurred_losses / earned_premium

    return {
        "earned_premium": earned_premium,
        "incurred_losses": incurred_losses,
        "loss_ratio": loss_ratio,
    }

result = calculate_loss_ratio(
    earned_premium=10_000_000,
    incurred_losses=7_800_000
)

result.value
result.commentary
result.info