"""
Pydantic v2 request/response models for the FastAPI application.

Keep all input/output schemas here so they are easy to extend.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Chat  (/chat)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question or message.")
    history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Prior conversation turns, each {'role': ..., 'content': ...}.",
    )


class ChatResponse(BaseModel):
    answer: str
    model: str


# ---------------------------------------------------------------------------
# RAG  (/rag/query)
# ---------------------------------------------------------------------------

class RAGRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer from the knowledge base.")
    k: int = Field(default=4, ge=1, le=10, description="Number of documents to retrieve.")


class RAGSource(BaseModel):
    title: str
    source: str
    document_type: str
    snippet: str


class RAGResponse(BaseModel):
    answer: str
    sources: list[RAGSource]


# ---------------------------------------------------------------------------
# Agent  (/agent/run)
# ---------------------------------------------------------------------------

class AgentRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Task or question for the ReAct agent.")


class AgentStep(BaseModel):
    tool: str
    tool_input: Any
    observation: str


class AgentResponse(BaseModel):
    output: str
    steps: list[AgentStep]
