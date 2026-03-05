"""
Shared LLM and embeddings factory.

Primary provider : Groq        (fast inference, low latency)
Fallback provider: OpenRouter  (broad model catalogue)

To swap models for a use-case, edit MODEL_REGISTRY below — nothing else changes.
"""
import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

logger = logging.getLogger(__name__)

# ── Provider base URLs ────────────────────────────────────────────────────────
GROQ_BASE_URL: str       = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# ── Model Registry ────────────────────────────────────────────────────────────
# Edit this dict to change which model handles each use-case.
#   "groq"       → model ID sent to Groq API       (primary)
#   "openrouter" → model ID sent to OpenRouter API  (fallback when GROQ_API_KEY absent)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    # General conversational chat
    "chat": {
        "groq":       "llama-3.3-70b-versatile",
        "openrouter": "openai/gpt-3.5-turbo",
    },
    # RAG / document Q&A — benefits from large context window
    "rag": {
        "groq":       "meta-llama/llama-4-scout-17b-16e-instruct",  # 30K ctx
        "openrouter": "openai/gpt-3.5-turbo",
    },
    # Tool-calling ReAct agent — compound model built for agentic tasks
    "agent": {
        "groq":       "compound-beta",
        "openrouter": "openai/gpt-3.5-turbo",
    },
    # Notion revision summary — long notes need wide context
    "revision": {
        "groq":       "meta-llama/llama-4-scout-17b-16e-instruct",  # 30K ctx
        "openrouter": "openai/gpt-3.5-turbo",
    },
    # Progress / weakness report — analytical summarisation
    "progress": {
        "groq":       "llama-3.3-70b-versatile",
        "openrouter": "openai/gpt-3.5-turbo",
    },
    # Socratic quiz — needs reliable structured JSON-schema output
    "quiz": {
        "groq":       "meta-llama/llama-4-scout-17b-16e-instruct",  # supports json_schema structured outputs
        "openrouter": "openai/gpt-4o-mini",
    },
    # Date extraction — simple deterministic task; use the fastest/lightest model
    "date_extraction": {
        "groq":       "llama-3.1-8b-instant",
        "openrouter": "nvidia/nemotron-nano-9b-v2:free",
    },
    # Embeddings — vector representations for RAG/vector search
    "embeddings": {
        "groq":       None,  # Groq doesn't offer embeddings
        "openrouter": "text-embedding-3-small",
    },
}


@lru_cache(maxsize=16)
def get_llm(
    use_case: str = "chat",
    temperature: float = 0.7,
) -> ChatOpenAI:
    """Return a cached LLM instance for the given use-case.

    Tries Groq first (low-latency); falls back to OpenRouter when
    GROQ_API_KEY is not set.

    Args:
        use_case:    Key from MODEL_REGISTRY (e.g. "chat", "rag", "quiz").
        temperature: Sampling temperature. Use 0 for deterministic tasks.

    Raises:
        ValueError:       If use_case is not a key in MODEL_REGISTRY.
        EnvironmentError: If neither GROQ_API_KEY nor OPENROUTER_API_KEY is set.
    """
    if use_case not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown use_case '{use_case}'. "
            f"Valid keys: {sorted(MODEL_REGISTRY)}"
        )

    entry = MODEL_REGISTRY[use_case]
    groq_key = os.environ.get("GROQ_API_KEY", "")

    if groq_key:
        logger.debug("LLM ▶ Groq / %s  (use_case=%s)", entry["groq"], use_case)
        return ChatOpenAI(
            model=entry["groq"],
            openai_api_key=groq_key,
            openai_api_base=GROQ_BASE_URL,
            temperature=temperature,
        )

    # ── Fallback: OpenRouter ──────────────────────────────────────────────────
    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not or_key:
        raise EnvironmentError(
            "Neither GROQ_API_KEY nor OPENROUTER_API_KEY is set. "
            "Add at least one to your .env file."
        )
    logger.warning(
        "GROQ_API_KEY not set — falling back to OpenRouter / %s  (use_case=%s)",
        entry["openrouter"], use_case,
    )
    return ChatOpenAI(
        model=entry["openrouter"],
        openai_api_key=or_key,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=temperature,
    )


@lru_cache(maxsize=2)
def get_embeddings(use_case: str = "embeddings") -> OpenAIEmbeddings:
    """Return a cached OpenAIEmbeddings instance using the registry pattern.

    Always uses OpenRouter since Groq doesn't offer embedding models.
    Falls back gracefully if no API keys are available.

    Args:
        use_case: Key from MODEL_REGISTRY (should be "embeddings").

    Raises:
        ValueError:       If use_case is not a key in MODEL_REGISTRY.
        EnvironmentError: If OPENROUTER_API_KEY is not set.
    """
    if use_case not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown use_case '{use_case}'. "
            f"Valid keys: {sorted(MODEL_REGISTRY)}"
        )

    entry = MODEL_REGISTRY[use_case]

    # Groq doesn't offer embeddings, so always use OpenRouter
    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not or_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is required for embeddings. "
            "Add it to your .env file."
        )

    model = entry["openrouter"]
    logger.debug("Embeddings ▶ OpenRouter / %s  (use_case=%s)", model, use_case)

    return OpenAIEmbeddings(
        model=model,
        openai_api_key=or_key,
        openai_api_base=OPENROUTER_BASE_URL,
    )
