"""
Revision chain — takes Notion daily notes and generates a structured revision summary.

Usage:
    chain = build_revision_chain()
    result = chain.invoke({
        "date": "2026-03-03",
        "question": "what did I learn on Monday?",
        "content": "<concatenated page content>",
    })
    # result is a plain string (markdown-formatted revision summary)
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.utils.llm import get_llm

_REVISION_SYSTEM = """\
You are a focused study assistant helping a software engineer revise their daily learning notes.
Your job is to transform raw notes into a clear, structured revision summary.

Return the summary in this exact format:

## 📅 {date} — Revision Summary

### 🧠 Key Concepts
(Bullet list of the most important concepts or ideas from the notes)

### 📌 Topics Covered
(Short descriptions of each major topic — one or two sentences each)

### 💡 Things to Remember
(Specific facts, definitions, formulas, patterns, or gotchas worth memorising)

### ❓ Questions to Explore
(1-3 follow-up questions or gaps in understanding you could investigate next)

Keep the language concise and direct. Focus on what matters for retention, not on restating everything verbatim.\
"""

_REVISION_HUMAN = """\
Here are the notes from {date}:

{content}

The user asked: "{question}"

Generate the revision summary.\
"""


def build_revision_chain():
    """Build an LCEL chain for generating structured revision summaries.

    Input keys required by chain.invoke():
        date     — ISO date string, e.g. "2026-03-03"
        question — the user's original natural-language question
        content  — concatenated text of Notion page content(s)

    Returns:
        A Runnable that produces a markdown string.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", _REVISION_SYSTEM),
        ("human", _REVISION_HUMAN),
    ])
    llm = get_llm()
    parser = StrOutputParser()

    return prompt | llm | parser
