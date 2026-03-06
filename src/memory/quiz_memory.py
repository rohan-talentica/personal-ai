"""
Quiz memory helpers — Phase 3: Long-Term Progress Tracker.

Backed by Supabase/Postgres + pgvector via PgVectorAdapter.
Key advantages over the previous ChromaDB implementation:
  - get_last_n_session_ids uses SQL ORDER BY + LIMIT (no full-table scan)
  - query_by_topic uses proper cosine-distance similarity search
  - is_correct is a real BOOLEAN column (not a "true"/"false" string)
"""
from __future__ import annotations

import logging
from datetime import date as date_cls
from typing import List, Optional

from langchain_core.documents import Document

from src.memory.factory import get_store
from src.memory.providers.pgvector import PgVectorAdapter

logger = logging.getLogger(__name__)

QUIZ_HISTORY_COLLECTION = "quiz_history"


def _get_adapter() -> PgVectorAdapter:
    """Return the PgVectorAdapter for the quiz_history collection.

    Raises RuntimeError if the configured provider is not pgvector.
    """
    adapter = get_store(QUIZ_HISTORY_COLLECTION)
    if not isinstance(adapter, PgVectorAdapter):
        raise RuntimeError(
            "Quiz memory requires VECTOR_STORE_PROVIDER=pgvector. "
            f"Got: {type(adapter).__name__}"
        )
    return adapter


def ingest_quiz_qa(
    *,
    session_id: str,
    notes_date: str,
    question: str,
    answer: str,
    feedback: str,
    concept: str,
    is_correct: bool,
    confidence_score: float,
) -> None:
    """Embed a single Q&A pair and insert it into quiz_history.

    Args:
        session_id:       LangGraph thread_id for this quiz session.
        notes_date:       YYYY-MM-DD of the Notion notes being quizzed.
        question:         The question asked by the quiz engine.
        answer:           The developer's answer.
        feedback:         Evaluator feedback on the answer.
        concept:          The core concept/topic being tested.
        is_correct:       Whether the developer answered correctly.
        confidence_score: 0.0–1.0 LLM-rated score of understanding.
    """
    content = (
        f"Concept: {concept}\n"
        f"Q: {question}\n"
        f"A: {answer}\n"
        f"Feedback: {feedback}"
    )
    doc = Document(
        page_content=content,
        metadata={
            "session_id": session_id,
            "notes_date": notes_date,
            "quiz_taken_at": date_cls.today().isoformat(),
            "concept": concept,
            "question": question,
            "answer": answer,
            "feedback": feedback,
            "is_correct": bool(is_correct),          # real bool — no string conversion
            "confidence_score": float(confidence_score),
        },
    )
    _get_adapter().add_documents([doc])
    logger.info(
        "quiz_memory: ingested Q&A concept='%s' is_correct=%s session=%s",
        concept, is_correct, session_id,
    )


def query_weak_areas(
    limit: int = 30,
    session_ids: Optional[List[str]] = None,
) -> List[Document]:
    """Retrieve Q&A pairs where the developer answered incorrectly.

    Args:
        limit:       Maximum number of documents to return.
        session_ids: If given, restrict results to these session IDs.

    Returns:
        List of Documents for use by the progress chain.
    """
    adapter = _get_adapter()
    docs = adapter.list_documents(filter={"is_correct": False})
    if session_ids is not None:
        session_set = set(session_ids)
        docs = [d for d in docs if d.metadata.get("session_id") in session_set]
    return docs[:limit]


def query_by_topic(
    question: str,
    k: int = 15,
    session_ids: Optional[List[str]] = None,
    score_threshold: float = 0.65,
) -> List[Document]:
    """Retrieve Q&A records most semantically relevant to a natural language question.

    Uses cosine similarity search on the embedding column.

    Args:
        question:        Natural language question, e.g. "how did I do on caching?"
        k:               Number of results to return.
        session_ids:     If given, restrict the search to these session IDs.
        score_threshold: Maximum distance to consider a record relevant.

    Returns:
        List of Documents, most similar first.
    """
    adapter = _get_adapter()
    if session_ids is not None:
        return adapter.similarity_search_by_sessions(question, session_ids, k=k, score_threshold=score_threshold)
    return adapter.similarity_search(question, k=k, score_threshold=score_threshold)


def get_last_n_session_ids(n: int) -> List[str]:
    """Return the session_ids of the n most recently taken quiz sessions.

    Delegates to SQL: GROUP BY session_id ORDER BY MAX(quiz_taken_at) DESC LIMIT n.
    No full-table scan needed.

    Args:
        n: Number of sessions to return.

    Returns:
        List of session_id strings, most-recent first.
    """
    return _get_adapter().get_last_n_session_ids(n)


def get_all_stats(session_ids: Optional[List[str]] = None) -> dict:
    """Return aggregate counts from the quiz_history table.

    Args:
        session_ids: If given, restrict counts to these session IDs.

    Returns:
        dict with keys: total_qa_pairs, weak_qa_pairs, sessions
    """
    return _get_adapter().get_stats(session_ids=session_ids)
