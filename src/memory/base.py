"""
Abstract base class for vector store adapters.

All vector store providers (Chroma, Pinecone, Qdrant, etc.) must implement
this interface. Consumers depend only on VectorStoreAdapter — never on a
concrete provider class.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

from langchain_core.documents import Document


class VectorStoreAdapter(ABC):
    """Minimal interface for a vector store used in this project.

    Only the 4 operations actually needed are defined here — keep it thin.
    Concrete implementations live in src/memory/providers/.
    """

    @abstractmethod
    def add_documents(self, docs: List[Document]) -> None:
        """Embed and upsert documents into the store.

        Args:
            docs: LangChain Document objects to embed and store.
        """

    @abstractmethod
    def as_retriever(self, k: int = 4) -> Any:
        """Return a LangChain-compatible retriever for this collection.

        Args:
            k: Number of documents to return per query.

        Returns:
            An object with a `.invoke(query: str) -> List[Document]` method.
        """

    @abstractmethod
    def list_documents(self, filter: dict | None = None) -> List[Document]:
        """Return documents from the store, optionally filtered by metadata.

        This is the generalised form of ChromaDB's `.get(where=...)`.
        Implementations for providers that don't support metadata filtering
        (e.g. basic Pinecone) can fetch all docs and filter in Python.

        Args:
            filter: Metadata key-value pairs to filter on (exact match).
                    e.g. {"is_correct": "false"}

        Returns:
            List of matching Documents.
        """

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
        score_threshold: float | None = None,
    ) -> List[Document]:
        """Return the k most semantically similar documents to query.

        Args:
            query:  Natural language question/topic to search for.
            k:      Number of results to return.
            filter: Optional metadata filters applied before ranking.
                    e.g. {"is_correct": False} or {"session_id": "abc"}
            score_threshold: Optional maximum distance/similarity score.
                             Only return documents with a score <= threshold.

        Returns:
            List of Documents ordered by descending similarity.
        """

    @abstractmethod
    def delete(self) -> None:
        """Delete the entire collection managed by this adapter.

        Used when overwrite=True is passed to build_vector_store().
        """
