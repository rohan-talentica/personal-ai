"""
Notion notes memory — Phase 5: Notion Pre-Embedding.

Stores pre-embedded Notion page chunks in the ``notion_notes`` pgvector table.
Each page is split into heading-scoped chunks (~500 tokens each) so that
the quiz agent can retrieve semantically relevant content without a live Notion
API call.

Key operations
--------------
- ingest_page()   — chunk, embed, and upsert a single Notion page
- delete_page()   — remove all existing chunks for a page (used before re-ingest)
- search_notes()  — semantic similarity search, optional date filter

Environment variables required (same as pgvector adapter):
    DATABASE_URL   — postgresql://user:password@host:5432/dbname

The shared psycopg pool is registered via set_pool() from the FastAPI lifespan,
mirroring the pattern used in quiz_memory.py.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

import psycopg
from pgvector.psycopg import register_vector
from langchain_core.documents import Document

from src.utils.llm import get_embeddings

try:
    from psycopg_pool import ConnectionPool as PsycopgPool
except ImportError:
    PsycopgPool = None  # type: ignore

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TABLE = "notion_notes"
MAX_CHUNK_CHARS = 1800  # ~500 tokens at ~3.5 chars/token — safe for most embedders

# Column order used in SELECT queries — must match _row_to_doc()
_SELECT_COLS = "page_id, title, date, last_edited_time, chunk_index, content"

# Module-level singletons set during FastAPI lifespan
_pool = None
_embeddings = None


# ── Lifecycle helpers ─────────────────────────────────────────────────────────

def set_pool(pool) -> None:
    """Register the shared psycopg ConnectionPool for this process.

    Called once from the FastAPI lifespan after the pool is created.
    """
    global _pool
    _pool = pool
    logger.info("notion_memory: shared connection pool registered")


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embeddings()
    return _embeddings


def _connect():
    """Return a context manager that yields a ready psycopg connection.

    Pool path (production): borrows from shared pool.
    Direct path (fallback):  opens a new connection (scripts / local dev).
    """
    import os
    if _pool is not None:
        return _pool.connection()
    dsn = os.getenv("DATABASE_URL", "").strip()
    if not dsn:
        raise EnvironmentError("DATABASE_URL is required for notion_memory.")
    conn = psycopg.connect(dsn, prepare_threshold=None)
    register_vector(conn)
    return conn


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_by_heading(text: str) -> List[str]:
    """Split page content at Markdown heading boundaries.

    Each chunk contains one heading and all the content beneath it until
    the next heading. Chunks are further split if they exceed MAX_CHUNK_CHARS
    to keep embeddings focused.

    Args:
        text: Full plain-text content of a Notion page.

    Returns:
        List of non-empty text chunks.
    """
    # Split on lines that start with 1–3 '#' characters
    heading_pattern = re.compile(r"(?m)^#{1,3} .+")
    boundaries = [m.start() for m in heading_pattern.finditer(text)]

    if not boundaries:
        # No headings — treat entire page as one chunk (split if too long)
        return _split_long_chunk(text.strip())

    chunks: List[str] = []

    # Text before the first heading (e.g. an intro paragraph)
    preamble = text[: boundaries[0]].strip()
    if preamble:
        chunks.extend(_split_long_chunk(preamble))

    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        section = text[start:end].strip()
        if section:
            chunks.extend(_split_long_chunk(section))

    return [c for c in chunks if c.strip()]


def _split_long_chunk(text: str) -> List[str]:
    """Split a single chunk that exceeds MAX_CHUNK_CHARS on paragraph boundaries."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    paragraphs = text.split("\n\n")
    result: List[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 > MAX_CHUNK_CHARS and current:
            result.append(current.strip())
            current = para
        else:
            current = (current + "\n\n" + para).strip() if current else para

    if current.strip():
        result.append(current.strip())

    return result


# ── Row conversion ────────────────────────────────────────────────────────────

def _row_to_doc(row: tuple) -> Document:
    """Convert a DB row (in _SELECT_COLS order) to a LangChain Document."""
    page_id, title, date, last_edited_time, chunk_index, content = row
    return Document(
        page_content=content,
        metadata={
            "page_id": page_id,
            "title": title,
            "date": date,
            "last_edited_time": last_edited_time,
            "chunk_index": chunk_index,
        },
    )


# ── Public API ────────────────────────────────────────────────────────────────

def delete_page(page_id: str) -> int:
    """Delete all stored chunks for a Notion page.

    Call this before re-ingesting an updated page to avoid stale/duplicate chunks.

    Args:
        page_id: The Notion page UUID.

    Returns:
        Number of rows deleted.
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {TABLE} WHERE page_id = %s", (page_id,))
            deleted = cur.rowcount
    logger.info("notion_memory: deleted %d chunk(s) for page %s", deleted, page_id)
    return deleted


def ingest_page(
    *,
    page_id: str,
    title: str,
    content: str,
    date: str,
    last_edited_time: str,
) -> int:
    """Chunk, embed, and upsert a Notion page into notion_notes.

    Splits the page content at heading boundaries, embeds each chunk, and
    inserts all rows in a single executemany call.

    Args:
        page_id:          Notion page UUID.
        title:            Page title (stored as metadata).
        content:          Full plain-text content of the page.
        date:             YYYY-MM-DD from the Notion Date property.
        last_edited_time: ISO 8601 timestamp from Notion.

    Returns:
        Number of chunks upserted.
    """
    chunks = _chunk_by_heading(content)
    if not chunks:
        logger.warning("notion_memory: page %s produced 0 chunks — skipping", page_id)
        return 0

    embeddings_model = _get_embeddings()
    vectors = embeddings_model.embed_documents(chunks)

    rows = [
        (page_id, title, date, last_edited_time, idx, chunk, vector)
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]

    sql = f"""
        INSERT INTO {TABLE}
            (page_id, title, date, last_edited_time, chunk_index, content, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)

    logger.info(
        "notion_memory: ingested %d chunk(s) for page '%s' (%s)",
        len(rows), title, page_id,
    )
    return len(rows)


def search_notes(
    query: str,
    k: int = 6,
    date_filter: Optional[str] = None,
    score_threshold: float = 0.7,
) -> List[Document]:
    """Semantic similarity search over embedded Notion notes.

    Args:
        query:          Natural language question or topic to search for.
        k:              Number of chunks to return.
        date_filter:    Optional YYYY-MM-DD to restrict results to a single day.
        score_threshold: Maximum cosine distance to consider relevant
                        (lower = more similar; 0 = identical, 2 = opposite).

    Returns:
        List of Documents, most similar first.
    """
    embeddings_model = _get_embeddings()
    query_vector = embeddings_model.embed_query(query)

    conditions: list[str] = []
    params: list = []

    if date_filter:
        conditions.append("date = %s")
        params.append(date_filter)

    if score_threshold is not None:
        conditions.append(f"embedding <=> %s::vector <= %s")
        params.extend([query_vector, score_threshold])

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"""
        SELECT {_SELECT_COLS}
        FROM {TABLE}
        {where}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    params.extend([query_vector, k])

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    logger.info(
        "notion_memory: search_notes returned %d result(s) for query='%s'",
        len(rows), query[:60],
    )
    return [_row_to_doc(r) for r in rows]
