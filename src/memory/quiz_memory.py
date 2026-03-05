"""
Quiz memory helpers — Phase 3: Long-Term Progress Tracker.

Uses the factory (src/memory/factory.py) to get a provider-agnostic
VectorStoreAdapter, so the quiz history collection automatically moves
to whichever vector store is configured in VECTOR_STORE_PROVIDER.
"""
from __future__ import annotations

import logging
from datetime import date as date_cls
from typing import List, Optional

from langchain_core.documents import Document

from src.memory.factory import get_store

logger = logging.getLogger(__name__)

QUIZ_HISTORY_COLLECTION = "quiz_history"


def ingest_quiz_qa(
    *,
    session_id: str,
    date: str,
    question: str,
    answer: str,
    feedback: str,
    concept: str,
    is_correct: bool,
    confidence_score: float,
) -> None:
    """Embed a single Q&A pair into the quiz_history collection.

    Args:
        session_id:       LangGraph thread_id for this quiz session.
        date:             YYYY-MM-DD of the Notion notes being quizzed.
        question:         The question asked by the quiz engine.
        answer:           The developer's answer.
        feedback:         Evaluator feedback on the answer.
        concept:          The core concept/topic being tested.
        is_correct:       Whether the developer answered correctly.
        confidence_score: 0.0–1.0 LLM-rated score of understanding.
    """
    text = (
        f"Concept: {concept}\n"
        f"Q: {question}\n"
        f"A: {answer}\n"
        f"Feedback: {feedback}"
    )
    doc = Document(
        page_content=text,
        metadata={
            "session_id": session_id,
            "date": date,
            "concept": concept,
            "is_correct": str(is_correct).lower(),   # must be str for Chroma where-clause
            "confidence_score": confidence_score,
            "question": question,
            "quiz_taken_at": date_cls.today().isoformat(),  # date the quiz was actually taken
        },
    )
    get_store(QUIZ_HISTORY_COLLECTION).add_documents([doc])
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
    docs = get_store(QUIZ_HISTORY_COLLECTION).list_documents(
        filter={"is_correct": "false"}
    )
    if session_ids is not None:
        session_set = set(session_ids)
        docs = [d for d in docs if d.metadata.get("session_id") in session_set]
    return docs[:limit]


def get_last_n_session_ids(n: int) -> List[str]:
    """Return the session_ids of the n most recently taken quiz sessions.

    Sessions are ranked by the latest ``quiz_taken_at`` date recorded among
    their Q&A pairs. Sessions that pre-date the ``quiz_taken_at`` field (old
    records) are sorted to the end.

    Args:
        n: Number of sessions to return.

    Returns:
        List of session_id strings, most-recent first.
    """
    all_docs = get_store(QUIZ_HISTORY_COLLECTION).list_documents()
    # Map each session_id → its latest quiz_taken_at value
    session_dates: dict[str, str] = {}
    for doc in all_docs:
        sid = doc.metadata.get("session_id")
        taken_at = doc.metadata.get("quiz_taken_at", "")
        if sid and taken_at > session_dates.get(sid, ""):
            session_dates[sid] = taken_at
    sorted_sessions = sorted(
        session_dates.keys(),
        key=lambda s: session_dates[s],
        reverse=True,
    )
    return sorted_sessions[:n]


def get_all_stats(session_ids: Optional[List[str]] = None) -> dict:
    """Return aggregate counts from the quiz_history collection.

    Args:
        session_ids: If given, restrict counts to these session IDs.

    Returns:
        dict with keys: total_qa_pairs, weak_qa_pairs, sessions
    """
    all_docs = get_store(QUIZ_HISTORY_COLLECTION).list_documents()

    if session_ids is not None:
        session_set = set(session_ids)
        all_docs = [d for d in all_docs if d.metadata.get("session_id") in session_set]

    total = len(all_docs)
    weak = sum(1 for d in all_docs if d.metadata.get("is_correct") == "false")
    sessions = len({
        d.metadata.get("session_id")
        for d in all_docs
        if d.metadata.get("session_id")
    })

    return {"total_qa_pairs": total, "weak_qa_pairs": weak, "sessions": sessions}
