# LLM factory — swap providers by changing LLM_PROVIDER in .env
# everything downstream just calls get_llm() and doesn't care which one it is

from functools import lru_cache
from src.config import (
    LLM_PROVIDER, LLM_MODEL,
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY,
)


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.0):
    # cached LLM instance — temp 0 for deterministic eval results
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=LLM_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=temperature,
        )

    elif LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=LLM_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=temperature,
        )

    elif LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=LLM_MODEL,
            api_key=GROQ_API_KEY,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{LLM_PROVIDER}'. "
            f"Supported: openai, anthropic, groq"
        )
