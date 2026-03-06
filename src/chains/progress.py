"""
Progress chain — Phase 3: Long-Term Progress Tracker.

Two modes:
  - Weakness report (default): summarises incorrect Q&A pairs into a ranked
    list of recurring knowledge gaps.
  - Topic analysis (when ``question`` is supplied): answers a specific natural
    language question about quiz performance using retrieved relevant records.
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from src.utils.llm import get_llm
from src.api.models import StructuredProgressReport


_SYSTEM_PROMPT = """\
You are a learning coach reviewing a developer's quiz history.
You will be given a list of Q&A records where the developer struggled (answered incorrectly or with low confidence).

Your task: produce a structured report that:
1. Groups entries by concept/topic, counting occurrences
2. Assigns a severity score (1-5) to each gap
3. Explains the specific gap revealed by the answers
4. Provides 1-2 actionable revision suggestions

Be direct and specific — avoid generic advice.\
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


_TOPIC_SYSTEM_PROMPT = """\
You are a learning coach reviewing a developer's quiz history.
You will be given quiz records retrieved as relevant to a specific question or topic.

Your task: directly answer the developer's question based on the evidence in these records.
- Be specific: cite concepts and examples from the records.
- Distinguish between areas of strength and areas of weakness found in the records.
- Provide a severity score (1-5) for each identified gap.
- ONLY output concepts in `weak_concepts` if a clear weakness or gap is identified (do not include strengths or neutral concepts there).
- Include 1-2 targeted revision suggestions for any gaps identified.

Be concise and direct.\
"""


def build_progress_chain(question: str | None = None):
    """Return an LCEL runnable that accepts List[Document] and returns a StructuredProgressReport.

    Args:
        question: Optional natural language question about quiz performance.
                  When provided, the chain answers the specific question instead
                  of producing a general weakness report.

    Usage:
        # General weakness report
        chain = build_progress_chain()
        report: StructuredProgressReport = chain.invoke(docs)

        # Topic-specific analysis
        chain = build_progress_chain(question="how did I do on caching?")
        report: StructuredProgressReport = chain.invoke(docs)
    """
    llm = get_llm(use_case="progress").with_structured_output(StructuredProgressReport)
    system_prompt = _TOPIC_SYSTEM_PROMPT if question else _SYSTEM_PROMPT

    def run(docs: List[Document]) -> StructuredProgressReport:
        formatted = _format_docs(docs)
        if question:
            human_content = (
                f"My question: {question}\n\n"
                f"Relevant quiz records:\n\n{formatted}\n\n"
                "Please answer my question based on these records."
            )
        else:
            human_content = (
                f"Here are the quiz records where I struggled:\n\n"
                f"{formatted}\n\n"
                "Please produce my weakness report."
            )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ]
        return llm.invoke(messages)

    return RunnableLambda(run)
