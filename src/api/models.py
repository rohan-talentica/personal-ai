"""
Pydantic v2 request/response models for the FastAPI application.

Keep all input/output schemas here so they are easy to extend.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


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
# RAG  (/rag/query  &  /rag/ingest)
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


class IngestRequest(BaseModel):
    """Body for POST /rag/ingest.

    Provide exactly one of ``url`` or ``text``.
    """
    url: Optional[str] = Field(None, description="Public URL to fetch and ingest.")
    text: Optional[str] = Field(None, description="Raw text to ingest directly.")
    title: str = Field("Untitled", description="Human-readable title stored as metadata.")
    source: str = Field("", description="Source label stored as metadata.")
    chunk_size: int = Field(500, ge=50, le=5000, description="Characters per chunk.")
    chunk_overlap: int = Field(50, ge=0, le=500, description="Overlap between consecutive chunks.")

    @model_validator(mode="after")
    def _require_url_or_text(self) -> "IngestRequest":
        if not self.url and not self.text:
            raise ValueError("Provide either 'url' or 'text'.")
        return self


class IngestResponse(BaseModel):
    chunks_added: int
    collection: str
    message: str = "OK"


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
