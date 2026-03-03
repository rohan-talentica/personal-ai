"""
Vector store helpers for building, loading, and ingesting into ChromaDB.

Used by the RAG chain (src/chains/rag.py) and can be called directly
from the FastAPI startup handler to warm up the retriever.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.llm import get_embeddings

logger = logging.getLogger(__name__)

# Default persist location — overridden by CHROMA_DB_PATH env var on ECS
import os as _os
DEFAULT_PERSIST_DIR: str = _os.getenv("CHROMA_DB_PATH", "notebooks/chroma_db_complete")
DEFAULT_COLLECTION = "complete_kb"


def build_vector_store(
    documents: list[Document],
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    overwrite: bool = False,
) -> Chroma:
    """Create (or overwrite) a ChromaDB collection from a list of Documents.

    Args:
        documents: LangChain ``Document`` objects to index.
        persist_directory: Directory where ChromaDB will persist data.
        collection_name: Name for the ChromaDB collection.
        overwrite: If True, delete the existing collection before indexing.

    Returns:
        The populated ``Chroma`` vector store instance.
    """
    embeddings = get_embeddings()

    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)

    if overwrite:
        import shutil

        if persist_path.exists():
            shutil.rmtree(persist_path)
            persist_path.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_path),
    )
    return vectorstore


def load_vector_store(
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> Chroma:
    """Load an existing ChromaDB collection from disk.

    Args:
        persist_directory: Path to the ChromaDB persist directory.
        collection_name: Collection name inside ChromaDB.

    Returns:
        A ``Chroma`` instance connected to the existing collection.

    Raises:
        FileNotFoundError: If the persist directory does not exist.
    """
    persist_path = Path(persist_directory)
    if not persist_path.exists():
        raise FileNotFoundError(
            f"ChromaDB persist directory not found: {persist_path}. "
            "Run build_vector_store() first or point to an existing collection."
        )

    embeddings = get_embeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_path),
    )


def get_retriever(
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    k: int = 4,
) -> Any:
    """Convenience wrapper — load the store and return its retriever.

    Args:
        persist_directory: Path to the ChromaDB persist directory.
        collection_name: Collection name inside ChromaDB.
        k: Number of documents to retrieve per query.
    """
    store = load_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    return store.as_retriever(search_kwargs={"k": k})


def ingest_documents(
    *,
    url: str | None = None,
    text: str | None = None,
    title: str = "Untitled",
    source: str = "",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    persist_directory: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
) -> int:
    """Chunk and upsert text or a public URL into a ChromaDB collection.

    Creates the collection (and the persist directory) if they don't exist
    yet — important for ECS where the disk starts empty on every task start.

    Args:
        url: Public URL to fetch and ingest (BS4 used to extract text).
        text: Raw text to ingest (used when url is None).
        title: Added to document metadata as ``title``.
        source: Added to document metadata as ``source``.
        chunk_size: Target character count per chunk.
        chunk_overlap: Overlap characters between consecutive chunks.
        persist_directory: ChromaDB directory; defaults to DEFAULT_PERSIST_DIR.
        collection_name: Collection to write into.

    Returns:
        Number of chunks added.

    Raises:
        ValueError: If neither url nor text is supplied.
    """
    if not url and not text:
        raise ValueError("Provide either 'url' or 'text'.")

    persist_dir = persist_directory or DEFAULT_PERSIST_DIR
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    # ── Fetch content ─────────────────────────────────────────────────────
    if url:
        import requests
        from bs4 import BeautifulSoup

        logger.info("Fetching URL for ingest: %s", url)
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove script/style clutter
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        raw_text = soup.get_text(separator="\n", strip=True)
        source = source or url
    else:
        raw_text = text  # type: ignore[assignment]

    # ── Split ─────────────────────────────────────────────────────────────
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

    # ── Upsert into ChromaDB ──────────────────────────────────────────────
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_path),
    )
    vectorstore.add_documents(docs)
    logger.info(
        "ingest_documents: added %d chunks to collection '%s' at '%s'",
        len(docs), collection_name, persist_dir,
    )
    return len(docs)
