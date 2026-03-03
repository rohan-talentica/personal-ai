"""
Shared LLM and embeddings factory.

All modules in src/ should import from here to keep configuration in one place.
"""
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
DEFAULT_CHAT_MODEL: str = "openai/gpt-3.5-turbo"
DEFAULT_EMBED_MODEL: str = "text-embedding-3-small"



@lru_cache(maxsize=4)
def get_llm(
    model: str = DEFAULT_CHAT_MODEL,
    temperature: float = 0.7,
) -> ChatOpenAI:
    """Return a cached ChatOpenAI instance pointed at OpenRouter.

    Args:
        model: OpenRouter model identifier (e.g. "openai/gpt-3.5-turbo").
        temperature: Sampling temperature. Use 0 for agents, 0.7 for chat.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file."
        )
    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=temperature,
    )


@lru_cache(maxsize=2)
def get_embeddings(model: str = DEFAULT_EMBED_MODEL) -> OpenAIEmbeddings:
    """Return a cached OpenAIEmbeddings instance pointed at OpenRouter.

    Args:
        model: Embedding model identifier.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file."
        )
    return OpenAIEmbeddings(
        model=model,
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_BASE_URL,
    )
