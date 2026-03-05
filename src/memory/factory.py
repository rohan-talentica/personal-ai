"""
Vector store factory.

Reads VECTOR_STORE_PROVIDER from the environment (default: "chroma") and
returns the appropriate VectorStoreAdapter implementation.

To add a new provider:
    1. Create src/memory/providers/<name>.py with a class that subclasses VectorStoreAdapter
    2. Add an elif branch below
    3. Set VECTOR_STORE_PROVIDER=<name> in your .env

No other files need to change.
"""
from __future__ import annotations

import os

from src.memory.base import VectorStoreAdapter


def get_store(collection_name: str) -> VectorStoreAdapter:
    """Return a VectorStoreAdapter for the given collection.

    The provider is selected from the VECTOR_STORE_PROVIDER env var:
        - "chroma"   (default) — ChromaDB Cloud or local PersistentClient
        - "pinecone"           — Pinecone (add providers/pinecone.py to enable)
        - "qdrant"             — Qdrant   (add providers/qdrant.py to enable)

    Args:
        collection_name: The name of the collection / index to operate on.

    Returns:
        A concrete VectorStoreAdapter instance.

    Raises:
        ValueError: If VECTOR_STORE_PROVIDER names an unrecognised provider.
    """
    provider = os.getenv("VECTOR_STORE_PROVIDER", "chroma").lower().strip()

    if provider == "chroma":
        from src.memory.providers.chroma import ChromaAdapter
        return ChromaAdapter(collection_name)

    # ── Future providers (uncomment + install deps to activate) ───────────────
    # elif provider == "pinecone":
    #     from src.memory.providers.pinecone import PineconeAdapter
    #     return PineconeAdapter(collection_name)
    #
    # elif provider == "qdrant":
    #     from src.memory.providers.qdrant import QdrantAdapter
    #     return QdrantAdapter(collection_name)

    raise ValueError(
        f"Unknown VECTOR_STORE_PROVIDER: '{provider}'. "
        "Supported values: 'chroma'. "
        "To add a new provider, see src/memory/factory.py."
    )
