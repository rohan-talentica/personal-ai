"""
Progress chain — Phase 3: Long-Term Progress Tracker.

Takes weak-area Q&A documents retrieved from ChromaDB and summarises
them into a structured markdown report using an LLM, identifying the
most recurring knowledge gaps.
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from src.utils.llm import get_llm


_SYSTEM_PROMPT = """\
You are a learning coach reviewing a developer's quiz history.
You will be given a list of Q&A records where the developer struggled (answered incorrectly or with low confidence).

Your task: produce a concise markdown report that:
1. Groups entries by concept/topic
2. Ranks topics from most-struggled (most entries) to least
3. For each topic, explains the specific gap revealed by the answers
4. Ends with 1-2 actionable revision suggestions

Format your output as clean markdown with headers. Be direct and specific — avoid generic advice.\
"""


def _format_docs(docs: List[Document]) -> str:
    """Convert documents into a numbered list for the LLM prompt."""
    if not docs:
        return "(No weak areas recorded yet. Complete a quiz session first.)"
    lines = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        concept = meta.get("concept", "Unknown concept")
        date = meta.get("date", "?")
        score = meta.get("confidence_score", "?")
        lines.append(
            f"{i}. [{date}] Concept: {concept} | Confidence: {score}\n"
            f"   {doc.page_content}"
        )
    return "\n\n".join(lines)


def build_progress_chain():
    """Return an LCEL runnable that accepts List[Document] and returns a markdown string.

    Usage:
        chain = build_progress_chain()
        report: str = chain.invoke(docs)
    """
    llm = get_llm(use_case="progress")
    parser = StrOutputParser()

    def run(docs: List[Document]) -> str:
        formatted = _format_docs(docs)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Here are the quiz records where I struggled:\n\n"
                    f"{formatted}\n\n"
                    "Please produce my weakness report."
                )
            ),
        ]
        return parser.invoke(llm.invoke(messages))

    return RunnableLambda(run)
