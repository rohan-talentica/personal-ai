"""
Vector store public API — provider-agnostic.

All callers (rag.py, main.py, dependencies.py) import from here.
The concrete provider is chosen by get_store() in factory.py — controlled
by the VECTOR_STORE_PROVIDER env var (default: "chroma").

To swap providers:
    1. Set VECTOR_STORE_PROVIDER=<name> in .env
    2. Nothing else changes here.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.memory.factory import get_store

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "complete_kb"


def build_vector_store(
    documents: list[Document],
    collection_name: str = DEFAULT_COLLECTION,
    overwrite: bool = False,
):
    """Create (or overwrite) a collection from a list of Documents.

    Args:
        documents:       LangChain Document objects to index.
        collection_name: Name for the collection.
        overwrite:       If True, delete the existing collection first.

    Returns:
        The adapter instance (for chaining / testing).
    """
    store = get_store(collection_name)

    if overwrite:
        store.delete()
        # Re-create the underlying adapter so it targets a fresh collection
        store = get_store(collection_name)

    store.add_documents(documents)
    return store


def get_retriever(
    collection_name: str = DEFAULT_COLLECTION,
    k: int = 4,
) -> Any:
    """Return a LangChain retriever for the given collection.

    Args:
        collection_name: Collection name.
        k: Number of documents to retrieve per query.
    """
    return get_store(collection_name).as_retriever(k=k)


def ingest_documents(
    *,
    url: str | None = None,
    text: str | None = None,
    title: str = "Untitled",
    source: str = "",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    collection_name: str = DEFAULT_COLLECTION,
    # Legacy param kept for backwards compatibility — no longer used
    persist_directory: str | None = None,
) -> int:
    """Chunk and upsert text or a public URL into a collection.

    Args:
        url:             Public URL to fetch and ingest.
        text:            Raw text to ingest (used when url is None).
        title:           Stored in document metadata as ``title``.
        source:          Stored in document metadata as ``source``.
        chunk_size:      Target character count per chunk.
        chunk_overlap:   Overlap characters between consecutive chunks.
        collection_name: Collection to write into.

    Returns:
        Number of chunks added.

    Raises:
        ValueError: If neither url nor text is supplied.
    """
    if not url and not text:
        raise ValueError("Provide either 'url' or 'text'.")

    # ── Fetch content ─────────────────────────────────────────────────────────
    if url:
        import requests
        from bs4 import BeautifulSoup

        logger.info("Fetching URL for ingest: %s", url)
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        raw_text = soup.get_text(separator="\n", strip=True)
        source = source or url
    else:
        raw_text = text  # type: ignore[assignment]

    # ── Split ─────────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(raw_text)
    if not chunks:
        logger.warning("ingest_documents: no chunks produced — is the text empty?")
        return 0

    docs = [
        Document(
            page_content=chunk,
            metadata={
                "title": title,
                "source": source,
                "document_type": "url" if url else "text",
            },
        )
        for chunk in chunks
    ]

    # ── Upsert ────────────────────────────────────────────────────────────────
    get_store(collection_name).add_documents(docs)
    logger.info(
        "ingest_documents: added %d chunks to collection '%s'",
        len(docs), collection_name,
    )
    return len(docs)
