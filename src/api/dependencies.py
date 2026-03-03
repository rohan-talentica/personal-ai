"""
FastAPI dependency injection for shared resources.

Resources are created once at startup (via module-level singletons) and
injected into endpoint handlers via ``Depends()``.  This avoids
re-initialising expensive objects (LLM client, vector store) on every request.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from langchain_classic.agents import AgentExecutor

from src.agents.react_agent import build_react_agent
from src.chains.chat import build_chat_chain
from src.chains.rag import build_rag_chain
from src.memory.vector_store import DEFAULT_COLLECTION, DEFAULT_PERSIST_DIR

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_chat_chain():
    """Singleton chat chain (stateless, safe to share across requests)."""
    logger.info("Initialising chat chain…")
    return build_chat_chain()


@lru_cache(maxsize=1)
def get_rag_chain():
    """Singleton RAG chain backed by the default ChromaDB collection."""
    logger.info("Initialising RAG chain…")
    return build_rag_chain(
        persist_directory=DEFAULT_PERSIST_DIR,
        collection_name=DEFAULT_COLLECTION,
    )


@lru_cache(maxsize=1)
def get_agent() -> AgentExecutor:
    """Singleton ReAct agent executor."""
    logger.info("Initialising ReAct agent…")
    return build_react_agent()
