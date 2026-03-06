"""
PgVector adapter — implements VectorStoreAdapter using Supabase/Postgres + pgvector.

Uses raw psycopg3 (psycopg package) for all queries — no ORM, no LangChain
vector store abstraction. This gives us proper SQL: ORDER BY, LIMIT, FILTER
on typed columns (BOOLEAN, DATE, FLOAT), and cosine-distance similarity search.

Prerequisites
-------------
1. Run infrastructure/migrations/001_quiz_history.sql against your Supabase DB.
2. Set DATABASE_URL in your .env:
       DATABASE_URL=postgresql://postgres:<password>@<host>:5432/postgres

Connection pool
---------------
In production the app creates a single ``psycopg_pool.ConnectionPool`` at
startup and passes it to ``PgVectorAdapter``.  Every query borrows a
connection from that pool, uses it, and returns it — no open/close overhead
per request, identical to the NestJS TypeORM / pg-pool pattern.

Fallback: when no pool is supplied (e.g. direct script usage), the adapter
opens a direct ``psycopg.connect()`` per call as before.
"""
from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

import psycopg
from pgvector.psycopg import register_vector
from langchain_core.documents import Document

from src.memory.base import VectorStoreAdapter
from src.utils.llm import get_embeddings

try:
    from psycopg_pool import ConnectionPool as PsycopgPool
except ImportError:  # psycopg-pool not installed — pool path unavailable
    PsycopgPool = None  # type: ignore

logger = logging.getLogger(__name__)

# Column order used in SELECT queries — must match _row_to_doc()
_SELECT_COLS = (
    "session_id, notes_date, quiz_taken_at, concept, question, "
    "answer, feedback, is_correct, confidence_score, content"
)


def _get_dsn() -> str:
    dsn = os.getenv("DATABASE_URL", "").strip()
    if not dsn:
        raise EnvironmentError(
            "DATABASE_URL is required for the pgvector provider. "
            "Add it to your .env — format: postgresql://user:password@host:5432/dbname"
        )
    return dsn


class PgVectorAdapter(VectorStoreAdapter):
    """VectorStoreAdapter backed by Postgres + pgvector (raw psycopg3).

    Operates exclusively on the ``quiz_history`` table defined in
    infrastructure/migrations/001_quiz_history.sql.

    The ``collection_name`` argument is accepted for interface compatibility
    but ignored — this adapter is single-table by design.

    Args:
        collection_name: Kept for interface compatibility, ignored internally.
        pool:            A ``psycopg_pool.ConnectionPool`` created once at
                         application startup.  When provided, every query
                         borrows a connection from the shared pool instead of
                         opening a new connection.  Pass ``None`` to fall back
                         to a direct ``psycopg.connect()`` per call (useful in
                         scripts / tests).
    """

    def __init__(self, collection_name: str, pool=None) -> None:
        self._collection_name = collection_name  # kept for logging only
        self._dsn = _get_dsn()
        self._pool = pool
        self._embeddings = get_embeddings()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _connect(self):
        """Return a context manager that yields a ready-to-use psycopg Connection.

        Pool path  (production): borrows a connection from the shared pool,
                                 returns it on exit — zero open/close overhead.
        Direct path (fallback):  opens a new connection, closes it on exit.

        Both paths support ``with self._connect() as conn:`` identically.
        """
        if self._pool is not None:
            return self._pool.connection()
        # Fallback: direct connection (scripts, tests, local dev without pool)
        conn = psycopg.connect(self._dsn, prepare_threshold=None)
        register_vector(conn)
        return conn

    @staticmethod
    def _row_to_doc(row: tuple) -> Document:
        """Convert a DB row (in _SELECT_COLS order) to a LangChain Document."""
        (
            session_id, notes_date, quiz_taken_at, concept,
            question, answer, feedback, is_correct,
            confidence_score, content,
        ) = row
        return Document(
            page_content=content,
            metadata={
                "session_id": session_id,
                "notes_date": str(notes_date),
                "quiz_taken_at": str(quiz_taken_at),
                "concept": concept,
                "question": question,
                "answer": answer,
                "feedback": feedback,
                "is_correct": is_correct,           # bool
                "confidence_score": confidence_score,
            },
        )

    # ── VectorStoreAdapter interface ───────────────────────────────────────────

    def add_documents(self, docs: List[Document]) -> None:
        """Embed and insert documents into quiz_history.

        Each Document's metadata must contain the quiz-specific fields set by
        quiz_memory.ingest_quiz_qa().
        """
        if not docs:
            return

        texts = [doc.page_content for doc in docs]
        vectors = self._embeddings.embed_documents(texts)

        rows = []
        for doc, vector in zip(docs, vectors):
            m = doc.metadata
            rows.append((
                m["session_id"],
                m["notes_date"],
                m["quiz_taken_at"],
                m["concept"],
                m["question"],
                m["answer"],
                m["feedback"],
                bool(m["is_correct"]),
                float(m["confidence_score"]),
                doc.page_content,
                vector,                        # list[float] → VECTOR via pgvector adapter
            ))

        sql = """
            INSERT INTO quiz_history
                (session_id, notes_date, quiz_taken_at, concept, question,
                 answer, feedback, is_correct, confidence_score, content, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
        logger.debug("PgVectorAdapter: inserted %d rows into quiz_history", len(rows))

    def as_retriever(self, k: int = 4) -> Any:
        """Return a simple retriever wrapper backed by similarity_search."""
        adapter = self

        class _Retriever:
            def invoke(self, query: str) -> List[Document]:
                return adapter.similarity_search(query, k=k)

        return _Retriever()

    def list_documents(self, filter: dict | None = None) -> List[Document]:
        """Fetch rows, optionally filtered by metadata fields.

        Filter keys must be valid column names (``is_correct``, ``session_id``,
        ``notes_date``, etc.). Values are passed as SQL parameters.

        Example::

            adapter.list_documents(filter={"is_correct": False})
            adapter.list_documents(filter={"session_id": "thread-abc"})
        """
        conditions: list[str] = []
        params: list = []

        if filter:
            for col, val in filter.items():
                conditions.append(f"{col} = %s")
                params.append(val)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"SELECT {_SELECT_COLS} FROM quiz_history {where}"

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or None)
                rows = cur.fetchall()

        return [self._row_to_doc(r) for r in rows]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
        score_threshold: float | None = None,
    ) -> List[Document]:
        """Return the k most semantically similar documents to query.

        Uses cosine distance (``<=>`` operator) on the ``embedding`` column.
        Optional filter is applied as a WHERE clause before ranking, enabling
        session-scoped or correctness-filtered topic search.

        Args:
            query:           Natural language question or topic.
            k:               Number of results to return.
            filter:          Column-value pairs applied as exact-match WHERE conditions.
                             e.g. ``{"is_correct": False}`` or ``{"session_id": "x"}``
            score_threshold: Optional maximum distance/similarity score.

        Returns:
            List of Documents, most similar first.
        """
        query_vector = self._embeddings.embed_query(query)

        conditions: list[str] = []
        params: list = []

        if filter:
            for col, val in filter.items():
                conditions.append(f"{col} = %s")
                params.append(val)

        if score_threshold is not None:
            conditions.append("embedding <=> %s::vector <= %s")
            params.extend([query_vector, score_threshold])

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        sql = f"""
            SELECT {_SELECT_COLS}
            FROM quiz_history
            {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params.extend([query_vector, k])

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [self._row_to_doc(r) for r in rows]

    def delete(self) -> None:
        """Delete all rows from quiz_history (drops data, keeps the table)."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM quiz_history")
        logger.info("PgVectorAdapter: deleted all rows from quiz_history")

    # ── Quiz-specific methods (beyond the base interface) ──────────────────────

    def get_last_n_session_ids(self, n: int) -> List[str]:
        """Return session_ids of the n most recently taken quiz sessions.

        Uses a GROUP BY + MAX(quiz_taken_at) + ORDER BY — impossible in ChromaDB,
        trivial in Postgres.

        Args:
            n: Number of sessions to return.

        Returns:
            List of session_id strings, most-recent first.
        """
        sql = """
            SELECT session_id
            FROM quiz_history
            GROUP BY session_id
            ORDER BY MAX(quiz_taken_at) DESC
            LIMIT %s
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (n,))
                rows = cur.fetchall()

        return [row[0] for row in rows]

    def similarity_search_by_sessions(
        self,
        query: str,
        session_ids: List[str],
        k: int = 10,
        score_threshold: float | None = None,
    ) -> List[Document]:
        """Similarity search scoped to a specific set of sessions.

        Uses ``session_id = ANY(%s)`` for the scope filter so the planner can
        combine the HNSW index with a session filter efficiently.

        Args:
            query:           Natural language question.
            session_ids:     Restrict results to these session IDs.
            k:               Number of results to return.
            score_threshold: Optional maximum distance/similarity score.
        """
        query_vector = self._embeddings.embed_query(query)

        conditions = ["session_id = ANY(%s)"]
        params: list = [session_ids]
        
        if score_threshold is not None:
            conditions.append("embedding <=> %s::vector <= %s")
            params.extend([query_vector, score_threshold])

        where = "WHERE " + " AND ".join(conditions)

        sql = f"""
            SELECT {_SELECT_COLS}
            FROM quiz_history
            {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params.extend([query_vector, k])

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [self._row_to_doc(r) for r in rows]

    def get_stats(self, session_ids: Optional[List[str]] = None) -> dict:
        """Return aggregate counts, optionally scoped to session_ids.

        Args:
            session_ids: If given, restrict counts to these session IDs.

        Returns:
            dict with keys: total_qa_pairs, weak_qa_pairs, sessions
        """
        if session_ids is not None:
            sql = """
                SELECT
                    COUNT(*)                                       AS total,
                    COUNT(*) FILTER (WHERE is_correct = false)    AS weak,
                    COUNT(DISTINCT session_id)                     AS sessions
                FROM quiz_history
                WHERE session_id = ANY(%s)
            """
            params: list = [session_ids]
        else:
            sql = """
                SELECT
                    COUNT(*)                                       AS total,
                    COUNT(*) FILTER (WHERE is_correct = false)    AS weak,
                    COUNT(DISTINCT session_id)                     AS sessions
                FROM quiz_history
            """
            params = []

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or None)
                row = cur.fetchone()

        return {
            "total_qa_pairs": int(row[0]) if row else 0,
            "weak_qa_pairs":  int(row[1]) if row else 0,
            "sessions":       int(row[2]) if row else 0,
        }
