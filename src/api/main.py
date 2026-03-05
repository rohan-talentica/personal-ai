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
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

load_dotenv()

# ── Internal imports ─────────────────────────────────────────────────────────
from src.api.dependencies import get_agent, get_chat_chain, get_quiz_graph, get_rag_chain
from src.api.models import (
    AgentRequest,
    AgentResponse,
    AgentStep,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    ProgressResponse,
    RAGRequest,
    RAGResponse,
    RAGSource,
    RevisionRequest,
    RevisionResponse,
    QuizStartRequest,
    QuizStartResponse,
    QuizAnswerRequest,
    QuizAnswerResponse,
)
from src.memory.vector_store import (
    DEFAULT_COLLECTION,
    ingest_documents,
)
from src.memory.quiz_memory import (
    ingest_quiz_qa,
    query_weak_areas,
    get_all_stats,
    get_last_n_session_ids,
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
        get_quiz_graph()
        logger.info("✅ Quiz graph ready")
    except Exception as exc:
        logger.warning("⚠️  Quiz graph init failed: %s", exc)

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
    # Use a small, fast model — date extraction is a simple deterministic task.
    try:
        today = date_cls.today().isoformat()
        llm = get_llm(use_case="date_extraction", temperature=0)
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


# ── Socratic Quiz (Phase 2) ──────────────────────────────────────────────────

@app.post("/notion/quiz", response_model=QuizStartResponse, tags=["Notion"])
async def notion_quiz_start(body: QuizStartRequest, graph=Depends(get_quiz_graph)):
    """Start a Socratic revision session.
    
    1. Extracts date from query
    2. Fetches Notion notes
    3. Starts a LangGraph session to generate the first question
    """
    import uuid
    from datetime import date as date_cls
    import re
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.tools.notion_tool import get_daily_notes
    from src.utils.llm import get_llm

    try:
        today = date_cls.today().isoformat()
        llm = get_llm(use_case="date_extraction", temperature=0)
        date_resolution_prompt = [
            SystemMessage(
                content=(
                    f"Today's date is {today}. "
                    "Your task: identify which calendar date the user is referring to in their message. "
                    "Output ONLY a single date in YYYY-MM-DD format and nothing else. "
                    "Examples:\n"
                    "  'what did I learn today?' → {today}\n"
                    "  'test me on Monday's notes' → the most recent Monday\n"
                    "  'quiz me on March 3rd' → 2026-03-03"
                ).format(today=today)
            ),
            HumanMessage(content=body.query),
        ]
        resolved_date: str = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: llm.invoke(date_resolution_prompt).content.strip(),
        )
    except Exception as exc:
        logger.error("Date resolution failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Could not parse date from query: {exc}") from exc

    if not re.match(r"^\d{4}-\d{2}-\d{2}$", resolved_date):
        raise HTTPException(status_code=422, detail=f"Invalid date resolved: {resolved_date}")

    try:
        notes = await asyncio.get_event_loop().run_in_executor(None, lambda: get_daily_notes(resolved_date))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Notion API error: {exc}") from exc

    if not notes:
        raise HTTPException(status_code=404, detail=f"No notes found for {resolved_date}.")

    combined_content = "\n\n---\n\n".join(f"**{note['title']}**\n\n{note['content']}" for note in notes)
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    try:
        # Initialize the graph state and run until END
        initial_state = {
            "date": resolved_date,
            "content": combined_content,
            "questions_asked": 0,
            "weak_areas": [],
            "asked_concepts": [],
            "messages": []
        }
        
        state = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: graph.invoke(initial_state, config=config)
        )
        
        # The generated question is the last message
        latest_message = state["messages"][-1]
        
        return QuizStartResponse(
            session_id=session_id,
            date=resolved_date,
            pages_found=len(notes),
            question=latest_message.content
        )
    except Exception as exc:
        logger.error("Quiz graph error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/notion/quiz/{session_id}/answer", response_model=QuizAnswerResponse, tags=["Notion"])
async def notion_quiz_answer(session_id: str, body: QuizAnswerRequest, graph=Depends(get_quiz_graph)):
    """Answer a quiz question and advance the graph."""
    from langchain_core.messages import HumanMessage
    
    config = {"configurable": {"thread_id": session_id}}
    
    # 1. First, check if thread exists
    try:
        stored_state = graph.get_state(config)
        if not stored_state.values:
            raise HTTPException(status_code=404, detail="Quiz session not found or expired.")
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        logger.error("Error retrieving state: %s", exc)
        raise HTTPException(status_code=500, detail="Error accessing session store.")

    state_values = stored_state.values
    if state_values.get("is_completed", False):
        return QuizAnswerResponse(
            evaluation="This quiz session is already completed.",
            is_completed=True,
            next_question=None
        )

    try:
        # 2. Add the user's answer (as a HumanMessage) to the state thread
        user_msg = HumanMessage(content=body.answer)
        
        # update_state appends the message to the state
        graph.update_state(config, {"messages": [user_msg]})
        
        # 3. Resume the graph (it was interrupted before 'evaluate_answer')
        # We pass None as input because the state is already updated.
        state = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: graph.invoke(None, config=config)
        )

        # is_completed when the graph has no further nodes to run (reached END)
        next_nodes = graph.get_state(config).next
        is_completed = len(next_nodes) == 0

        feedback = state.get("evaluation_feedback", "Evaluated.")
        concept = state.get("_last_concept", "Unknown")  # may be absent on older sessions

        next_question = None
        if not is_completed:
            # The last message is the new question generated after evaluation
            next_question = state["messages"][-1].content

        # ── Phase 3: Ingest this Q&A pair into quiz_history ──────────────────
        # We reconstruct the Q&A from the last two messages before resumption:
        # messages[-2] = AI question, messages[-1] (before graph resumed) = user answer
        # After graph.invoke the evaluate node has run and state reflects results.
        # We read the conversation messages to find the most recent Q+A pair.
        try:
            msgs = state.get("messages", [])
            # Most recent AI question is the second-to-last message (last is new question if any)
            # Find the last HumanMessage (answer) and the AIMessage just before it
            ai_question = ""
            user_answer = ""
            for i in range(len(msgs) - 1, -1, -1):
                msg = msgs[i]
                if hasattr(msg, "type"):
                    if msg.type == "human" and not user_answer:
                        user_answer = msg.content
                    elif msg.type == "ai" and user_answer and not ai_question:
                        ai_question = msg.content
                        break

            if ai_question and user_answer:
                eval_feedback = state.get("evaluation_feedback", "")
                # Use real LLM-scored values from QuizState (set by evaluate_answer node)
                concept = state.get("last_concept", "General")
                conf_score = state.get("last_confidence_score", 0.5)
                weak_areas = state.get("weak_areas", [])
                # A concept is in weak_areas only when is_correct=False
                qa_is_correct = concept not in weak_areas

                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: ingest_quiz_qa(
                        session_id=session_id,
                        date=state_values.get("date", "unknown"),
                        question=ai_question,
                        answer=user_answer,
                        feedback=eval_feedback,
                        concept=concept,
                        is_correct=qa_is_correct,
                        confidence_score=conf_score,
                    )
                )
                logger.info("Phase 3: ingested Q&A for session %s", session_id)
        except Exception as ingest_exc:
            # Ingest failure must NOT break the quiz flow
            logger.warning("Phase 3 ingest error (non-fatal): %s", ingest_exc)

        return QuizAnswerResponse(
            evaluation=feedback,
            is_completed=is_completed,
            next_question=next_question
        )
    except Exception as exc:
        logger.error("Graph resuming failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))



# ── Progress Tracker (Phase 3) ──────────────────────────────────────────────

@app.get("/notion/progress", response_model=ProgressResponse, tags=["Notion"])
async def notion_progress(
    last_n_sessions: Optional[int] = Query(
        None,
        ge=1,
        description="Limit analysis to the N most recent quiz sessions. Omit to analyse all sessions.",
    )
):
    """Return a personalised weakness report based on past quiz sessions.

    Retrieves Q&A pairs where you answered incorrectly from the persistent
    ``quiz_history`` ChromaDB collection, then asks the LLM to identify
    recurring knowledge gaps and rank them by frequency.

    Pass ``?last_n_sessions=N`` to restrict the report to the N most recent sessions.
    Omit the parameter to analyse all sessions.
    """
    from src.chains.progress import build_progress_chain

    try:
        target_session_ids = None
        if last_n_sessions is not None:
            target_session_ids = await asyncio.get_event_loop().run_in_executor(
                None, lambda: get_last_n_session_ids(last_n_sessions)
            )

        stats = await asyncio.get_event_loop().run_in_executor(
            None, lambda: get_all_stats(session_ids=target_session_ids)
        )

        docs = await asyncio.get_event_loop().run_in_executor(
            None, lambda: query_weak_areas(session_ids=target_session_ids)
        )

        chain = build_progress_chain()
        report: str = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: chain.invoke(docs),
        )

        return ProgressResponse(
            sessions_analysed=stats["sessions"],
            total_qa_pairs=stats["total_qa_pairs"],
            weak_qa_pairs=stats["weak_qa_pairs"],
            report=report,
        )
    except Exception as exc:
        logger.error("Progress endpoint error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


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
