"""
ChromaDB adapter — implements VectorStoreAdapter using Chroma Cloud (or
a local PersistentClient fallback when Cloud credentials are absent).

To switch to a different vector store, add a new file in this directory
(e.g. pinecone.py, qdrant.py) and register it in src/memory/factory.py.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.memory.base import VectorStoreAdapter
from src.utils.llm import get_embeddings

logger = logging.getLogger(__name__)

# Local fallback directory (only used when Cloud credentials are absent)
_DEFAULT_LOCAL_DIR = os.getenv("CHROMA_DB_PATH", "chroma_db")


def _build_client() -> chromadb.ClientAPI:
    """Return CloudClient when credentials are present, PersistentClient otherwise."""
    api_key  = os.getenv("CHROMA_API_KEY", "").strip()
    tenant   = os.getenv("CHROMA_TENANT", "").strip()
    database = os.getenv("CHROMA_DATABASE", "").strip()

    if api_key and tenant and database:
        logger.info("ChromaAdapter: connecting to Cloud (tenant=%s, db=%s)", tenant, database)
        return chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database,
        )

    logger.info("ChromaAdapter: using local PersistentClient at '%s'", _DEFAULT_LOCAL_DIR)
    Path(_DEFAULT_LOCAL_DIR).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=_DEFAULT_LOCAL_DIR)


class ChromaAdapter(VectorStoreAdapter):
    """VectorStoreAdapter backed by ChromaDB (Cloud or local).

    Each instance manages a single named collection.

    Args:
        collection_name: The ChromaDB collection this adapter operates on.
    """

    def __init__(self, collection_name: str) -> None:
        self._collection_name = collection_name
        self._client = _build_client()
        self._embeddings = get_embeddings()

    def _store(self) -> Chroma:
        """Return a LangChain Chroma wrapper bound to this collection."""
        return Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
            client=self._client,
        )

    # ── VectorStoreAdapter interface ──────────────────────────────────────────

    def add_documents(self, docs: List[Document]) -> None:
        self._store().add_documents(docs)
        logger.debug(
            "ChromaAdapter: added %d docs to '%s'", len(docs), self._collection_name
        )

    def as_retriever(self, k: int = 4) -> Any:
        return self._store().as_retriever(search_kwargs={"k": k})

    def list_documents(self, filter: dict | None = None) -> List[Document]:
        """Fetch documents from ChromaDB, optionally filtered by metadata.

        Uses Chroma's native ``where`` clause for server-side filtering.
        """
        try:
            kwargs: dict = {}
            if filter:
                kwargs["where"] = filter
            raw = self._store().get(**kwargs)
        except Exception as exc:
            logger.warning(
                "ChromaAdapter.list_documents: query failed — %s", exc
            )
            return []

        if not raw or not raw.get("documents"):
            return []

        return [
            Document(page_content=content, metadata=meta or {})
            for content, meta in zip(raw["documents"], raw["metadatas"])
        ]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
    ) -> List[Document]:
        """Semantic similarity search via Chroma's built-in MMR/cosine."""
        kwargs: dict = {"k": k}
        if filter:
            kwargs["filter"] = filter
        return self._store().similarity_search(query, **kwargs)

    def delete(self) -> None:
        """Delete the entire collection from ChromaDB."""
        try:
            self._client.delete_collection(self._collection_name)
            logger.info(
                "ChromaAdapter: deleted collection '%s'", self._collection_name
            )
        except Exception as exc:
            logger.warning(
                "ChromaAdapter.delete: could not delete '%s' — %s",
                self._collection_name, exc,
            )
