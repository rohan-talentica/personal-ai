"""
Day 9 — Production FastAPI Application
=======================================

Endpoints
---------
GET  /health                 Liveness check
POST /chat                   Stateless conversational chain (Days 1-2)
POST /chat/stream            Same, but streams tokens via SSE
POST /rag/ingest             Ingest a URL or raw text into ChromaDB
POST /rag/query              RAG pipeline with citations (Days 3-4)
POST /agent/run              ReAct agent with tools (Day 5)

Run locally
-----------
    uvicorn src.api.main:app --reload --port 8000

Or via the helper at the bottom:
    python -m src.api.main
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

load_dotenv()

# ── Internal imports ─────────────────────────────────────────────────────────
from src.api.dependencies import get_agent, get_chat_chain, get_rag_chain
from src.api.models import (
    AgentRequest,
    AgentResponse,
    AgentStep,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    RAGRequest,
    RAGResponse,
    RAGSource,
    RevisionRequest,
    RevisionResponse,
)
from src.memory.vector_store import (
    DEFAULT_COLLECTION,
    DEFAULT_PERSIST_DIR,
    ingest_documents,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up expensive singletons on startup."""
    logger.info("🚀 Starting Personal AI API…")
    try:
        get_chat_chain()
        logger.info("✅ Chat chain ready")
    except Exception as exc:
        logger.warning("⚠️  Chat chain init failed: %s", exc)

    try:
        get_rag_chain()
        logger.info("✅ RAG chain ready")
    except Exception as exc:
        logger.warning("⚠️  RAG chain init failed (no vector store?): %s", exc)

    try:
        get_agent()
        logger.info("✅ Agent ready")
    except Exception as exc:
        logger.warning("⚠️  Agent init failed: %s", exc)

    logger.info("🟢 API is ready to serve requests")
    yield
    logger.info("🔴 Shutting down Personal AI API")


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Personal AI Research Assistant",
    description=(
        "Production API wrapping the LangChain / LangGraph pipelines "
        "built across the 10-day learning plan."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs for details."},
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Liveness / readiness probe."""
    return HealthResponse()


# ── Chat ─────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(body: ChatRequest, chain=Depends(get_chat_chain)):
    """Answer a question using the conversational LLM chain.

    Optionally pass ``history`` (list of ``{"role": ..., "content": ...}``
    dicts) for multi-turn context.
    """
    try:
        from langchain_core.messages import AIMessage, HumanMessage

        # Convert raw history dicts → LangChain message objects
        lc_history = []
        for msg in body.history:
            role = msg.get("role", "human").lower()
            content = msg.get("content", "")
            if role in ("human", "user"):
                lc_history.append(HumanMessage(content=content))
            else:
                lc_history.append(AIMessage(content=content))

        answer: str = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: chain.invoke({"question": body.question, "history": lc_history}),
        )
        return ChatResponse(answer=answer, model="openai/gpt-3.5-turbo")
    except Exception as exc:
        logger.error("Chat error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(body: ChatRequest, chain=Depends(get_chat_chain)):
    """Stream tokens from the conversational chain via Server-Sent Events.

    Consume with:
    ```
    EventSource('/chat/stream', { method: 'POST', body: JSON.stringify({question: '...'}) })
    ```
    """
    from langchain_core.messages import AIMessage, HumanMessage

    lc_history = []
    for msg in body.history:
        role = msg.get("role", "human").lower()
        content = msg.get("content", "")
        lc_history.append(
            HumanMessage(content=content) if role in ("human", "user") else AIMessage(content=content)
        )

    async def token_generator() -> AsyncGenerator[str, None]:
        try:
            async for chunk in chain.astream({"question": body.question, "history": lc_history}):
                if chunk:
                    yield f"data: {chunk}\n\n"
        except Exception as exc:
            logger.error("Stream error: %s", exc)
            yield f"data: [ERROR] {exc}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── RAG ───────────────────────────────────────────────────────────────────────

@app.post("/rag/ingest", response_model=IngestResponse, tags=["RAG"])
async def rag_ingest(body: IngestRequest):
    """Ingest a URL or raw text into ChromaDB.

    The document is fetched (if a URL), split into chunks, embedded, and
    upserted into the default ChromaDB collection.  Subsequent calls to
    ``POST /rag/query`` will be able to retrieve the newly added content.
    """
    try:
        chunks_added: int = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: ingest_documents(
                url=body.url,
                text=body.text,
                title=body.title,
                source=body.source,
                chunk_size=body.chunk_size,
                chunk_overlap=body.chunk_overlap,
                persist_directory=DEFAULT_PERSIST_DIR,
                collection_name=DEFAULT_COLLECTION,
            ),
        )
        return IngestResponse(
            chunks_added=chunks_added,
            collection=DEFAULT_COLLECTION,
        )
    except Exception as exc:
        logger.error("Ingest error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/rag/query", response_model=RAGResponse, tags=["RAG"])
async def rag_query(body: RAGRequest, chain=Depends(get_rag_chain)):
    """Answer a question using the RAG pipeline, returning citations.

    Retrieves the top-``k`` most relevant documents from the ChromaDB
    vector store then synthesises an answer with source attribution.
    """
    try:
        result: dict = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: chain.invoke({"question": body.question}),
        )
        return RAGResponse(
            answer=result["answer"],
            sources=[RAGSource(**s) for s in result["sources"]],
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Vector store not initialised. "
                "Run the Day 3/4 notebook to populate the ChromaDB collection."
            ),
        ) from exc
    except Exception as exc:
        logger.error("RAG error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# ── Notion / Revision ─────────────────────────────────────────────────────────

@app.post("/notion/revise", response_model=RevisionResponse, tags=["Notion"])
async def notion_revise(body: RevisionRequest):
    """Fetch Notion notes for a given day and return a structured revision summary.

    Supply a natural-language query like:
    - ``"what did I learn on Monday?"``
    - ``"revise yesterday's notes"``
    - ``"summarise my notes from March 3rd"``

    The endpoint will:
    1. Use the LLM to resolve the date from the query.
    2. Fetch all Notion pages for that date.
    3. Run the revision chain to produce a structured summary.
    """
    from datetime import date as date_cls
    import re
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.tools.notion_tool import get_daily_notes
    from src.chains.revision import build_revision_chain
    from src.utils.llm import get_llm

    # ── Step 1: Resolve the date from the natural-language query ──────────────
    # Use the LLM with temperature=0 (deterministic) and a focused prompt.
    # The prompt separates "what date is implied?" from "answer the user's question"
    # so the model doesn't confuse a date-extraction task with a personal query.
    try:
        today = date_cls.today().isoformat()
        llm = get_llm(temperature=0)
        date_resolution_prompt = [
            SystemMessage(
                content=(
                    f"Today's date is {today}. "
                    "Your task: identify which calendar date the user is referring to in their message. "
                    "Do NOT answer their question. Do NOT explain. "
                    "Output ONLY a single date in YYYY-MM-DD format and nothing else. "
                    "Examples:\n"
                    "  'what did I learn today?' → {today}\n"
                    "  'revise Monday's notes' → the most recent Monday\n"
                    "  'what did I study yesterday?' → yesterday's date\n"
                    "  'summarise March 3rd' → 2026-03-03"
                ).format(today=today)
            ),
            HumanMessage(content=body.query),
        ]
        resolved_date: str = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: llm.invoke(date_resolution_prompt).content.strip(),
        )
        logger.info("Resolved date '%s' from query: %s", resolved_date, body.query)
    except Exception as exc:
        logger.error("Date resolution failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Could not parse date from query: {exc}") from exc

    # Sanity check — LLM must return something that looks like YYYY-MM-DD
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", resolved_date):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Could not extract a date from your query. Got: '{resolved_date}'. "
                "Try: 'what did I learn on Monday?' or 'revise today's notes'."
            ),
        )

    # ── Step 2: Fetch Notion pages ─────────────────────────────────────────────
    try:
        notes = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: get_daily_notes(resolved_date),
        )
    except Exception as exc:
        logger.error("Notion fetch failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Notion API error: {exc}") from exc

    if not notes:
        raise HTTPException(
            status_code=404,
            detail=f"No notes found in Notion for {resolved_date}. Make sure a page with that date exists and is shared with your integration.",
        )

    # ── Step 3: Concatenate all page content and run revision chain ────────────
    combined_content = "\n\n---\n\n".join(
        f"**{note['title']}**\n\n{note['content']}" for note in notes
    )

    try:
        chain = build_revision_chain()
        summary: str = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: chain.invoke({
                "date": resolved_date,
                "question": body.query,
                "content": combined_content,
            }),
        )
    except Exception as exc:
        logger.error("Revision chain error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return RevisionResponse(
        date=resolved_date,
        pages_found=len(notes),
        summary=summary,
    )


# ── Agent ─────────────────────────────────────────────────────────────────────

@app.post("/agent/run", response_model=AgentResponse, tags=["Agent"])
async def agent_run(body: AgentRequest, executor=Depends(get_agent)):
    """Run the ReAct agent on a query.

    The agent can call ``calculator``, ``word_counter``, ``get_weather``,
    and ``text_summarizer`` tools.  Returns the final answer plus a trace of
    every tool invocation.
    """
    try:
        result: dict = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: executor.invoke({"input": body.query}),
        )
        steps = [
            AgentStep(
                tool=step[0].tool,
                tool_input=step[0].tool_input,
                observation=str(step[1]),
            )
            for step in result.get("intermediate_steps", [])
        ]
        return AgentResponse(output=result.get("output", ""), steps=steps)
    except Exception as exc:
        logger.error("Agent error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# ── Dev entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        log_level="info",
    )
