"""
Vector store helpers for building and loading ChromaDB collections.

Used by the RAG chain (src/chains/rag.py) and can be called directly
from the FastAPI startup handler to warm up the retriever.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.utils.llm import get_embeddings

# Default persist location (relative to project root)
DEFAULT_PERSIST_DIR = "notebooks/chroma_db_complete"
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
