import os

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .schema import LLM


def get_llm(llm: LLM) -> BaseChatModel:
    """
    Get an LLM instance based on the LLM enum.
    """
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    kwargs = {"base_url": base_url} if base_url else {}

    match llm:
        case LLM.OPENAI_GPT_41:
            return ChatOpenAI(model="gpt-4.1", **kwargs)
        case _:
            raise ValueError(f"Unknown LLM: {llm}")
