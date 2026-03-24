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
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.agents.quiz_graph import build_quiz_graph
from src.api.dependencies import get_agent, get_chat_chain, get_quiz_graph, get_rag_chain
from src.memory.providers.pgvector import PgVectorAdapter
from src.memory.quiz_memory import QUIZ_HISTORY_COLLECTION, set_adapter
from src.api.models import (
    AgentRequest,
    AgentResponse,
    AgentStep,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    NotionIngestRequest,
    NotionIngestResponse,
    NotionWebhookResponse,
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
    query_by_topic,
    get_all_stats,
    get_last_n_session_ids,
)
import src.memory.notion_memory as notion_memory


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

    # ── Postgres connection pool (shared across all requests) ──────────────────────
    # One pool for the whole process — same pattern as NestJS TypeORM / pg-pool.
    # configure= registers the pgvector type on every connection when created.
    database_url = os.getenv("DATABASE_URL", "")
    from psycopg_pool import ConnectionPool as PsycopgPool
    from pgvector.psycopg import register_vector

    pg_pool = PsycopgPool(
        database_url,
        min_size=2,
        max_size=10,
        configure=register_vector,  # registers pgvector type on each connection
        open=True,                  # open the pool immediately (blocking, fast)
    )
    app.state.pg_pool = pg_pool
    logger.info("✅ Postgres connection pool ready (min=2 max=10)")

    # Wire the shared PgVectorAdapter singleton into quiz_memory.
    # All calls to ingest_quiz_qa / query_weak_areas / etc. reuse this adapter
    # rather than opening a new connection per call.
    pg_adapter = PgVectorAdapter(QUIZ_HISTORY_COLLECTION, pool=pg_pool)
    set_adapter(pg_adapter)
    logger.info("✅ PgVectorAdapter singleton registered")

    # Wire the shared pool into notion_memory too
    notion_memory.set_pool(pg_pool)
    logger.info("✅ notion_memory pool registered")


    # ── Async Postgres checkpointer ───────────────────────────────────────────
    async with AsyncPostgresSaver.from_conn_string(database_url) as checkpointer:
        try:
            await checkpointer.setup()  # idempotent — creates tables if needed
            app.state.quiz_graph = build_quiz_graph(checkpointer=checkpointer)
            logger.info("✅ Quiz graph ready (Postgres checkpointer)")
        except Exception as exc:
            logger.warning("⚠️  Quiz graph init failed: %s", exc)
            app.state.quiz_graph = None

        logger.info("🟢 API is ready to serve requests")
        yield

    pg_pool.close()
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
    
    1. Extracts date or topic from query
    2. Fetches relevant notes from pgvector (NO live Notion fetch)
    3. Starts a LangGraph session to generate the first question
    """
    import uuid
    from datetime import date as date_cls
    import json
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.utils.llm import get_llm

    try:
        today = date_cls.today().isoformat()
        llm = get_llm(use_case="date_extraction", temperature=0)
        
        extraction_prompt = [
            SystemMessage(
                content=(
                    f"Today's date is {today}. "
                    "Your task: identify if the user wants to be quizzed on a specific date OR a specific topic. "
                    "Output ONLY a JSON object with 'date' (YYYY-MM-DD or null) and 'topic' (string or null). "
                    "Examples:\n"
                    "  'quiz me on today' → {{\"date\": \"{today}\", \"topic\": null}}\n"
                    "  'test me on React' → {{\"date\": null, \"topic\": \"React\"}}\n"
                    "  'quiz me on Monday' → {{\"date\": \"2026-03-23\", \"topic\": null}}\n"
                    "  'quiz me on Python decorators' → {{\"date\": null, \"topic\": \"Python decorators\"}}"
                ).format(today=today)
            ),
            HumanMessage(content=body.query),
        ]
        
        extraction_result: str = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: llm.invoke(extraction_prompt).content.strip(),
        )
        
        # Clean up Markdown if the LLM wrapped it in ```json
        if extraction_result.startswith("```"):
            extraction_result = extraction_result.strip("```json").strip("```").strip()
            
        extraction_data = json.loads(extraction_result)
        resolved_date = extraction_data.get("date")
        resolved_topic = extraction_data.get("topic")
        
        logger.info("Extracted from quiz query: date=%s, topic=%s", resolved_date, resolved_topic)
    except Exception as exc:
        logger.error("Extraction failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Could not understand query: {exc}") from exc

    if not resolved_date and not resolved_topic:
        raise HTTPException(
            status_code=422, 
            detail="Could not identify a date or topic for the quiz. Try 'quiz me on React' or 'quiz me on yesterday'."
        )

    # ── Step 2: Semantic Retrieval ──────────────────────────────────────────
    import src.memory.notion_memory as notion_memory
    
    chunks = []
    search_label = ""
    
    try:
        if resolved_topic:
            # Topic-based search: global search across all dates
            search_label = f"topic '{resolved_topic}'"
            chunks = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: notion_memory.search_notes(
                    query=resolved_topic,
                    k=15,
                    date_filter=None,
                    score_threshold=0.85  # Allow more breadth for topic search
                )
            )
        else:
            # Date-based search: restricted to that specific date
            search_label = f"date {resolved_date}"
            chunks = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: notion_memory.search_notes(
                    query="technical concepts",
                    k=20,
                    date_filter=resolved_date,
                    score_threshold=None
                )
            )
    except Exception as exc:
        logger.error("Quiz retrieval failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Error retrieving notes from semantic store: {exc}")

    if not chunks:
        raise HTTPException(
            status_code=404, 
            detail=(
                f"No notes found for {search_label} in the semantic store. "
                "Please ensure you have ingested the relevant Notion pages first using POST /notion/ingest."
            )
        )

    logger.info("notion/quiz: Initialised using %d chunks for %s", len(chunks), search_label)
    combined_content = "\n\n---\n\n".join(doc.page_content for doc in chunks)
    pages_found = len(set(doc.metadata.get("page_id") for doc in chunks))

    # ── Step 3: Start LangGraph Session ──────────────────────────────────────
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    try:
        initial_state = {
            "date": resolved_date or "multiple dates",
            "topic": resolved_topic,
            "content": combined_content,
            "questions_asked": 0,
            "weak_areas": [],
            "asked_concepts": [],
            "messages": []
        }

        state = await graph.ainvoke(initial_state, config=config)
        latest_message = state["messages"][-1]

        return QuizStartResponse(
            session_id=session_id,
            date=resolved_date or f"Topic: {resolved_topic}",
            pages_found=pages_found,
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
        stored_state = await graph.aget_state(config)
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

        # aupdate_state appends the message to the persisted state
        await graph.aupdate_state(config, {"messages": [user_msg]})

        # 3. Resume the graph (it was interrupted before 'evaluate_answer')
        # We pass None as input because the state is already updated.
        state = await graph.ainvoke(None, config=config)

        # is_completed when the graph has no further nodes to run (reached END)
        next_nodes = (await graph.aget_state(config)).next
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
                        notes_date=state_values.get("date", "unknown"),
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

        # ── Cleanup: delete checkpoint data for completed sessions ────────────
        # checkpoint_writes/blobs/checkpoints accumulate indefinitely.
        # Once a quiz is done there is nothing to resume, so we purge the
        # thread immediately. The Q&A history is already persisted in
        # quiz_history (Phase 3) so no data is lost.
        if is_completed:
            try:
                await graph.checkpointer.adelete_thread(session_id)
                logger.info("Checkpoint data pruned for completed session %s", session_id)
            except Exception as prune_exc:
                logger.warning("Checkpoint prune failed (non-fatal): %s", prune_exc)

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
    question: Optional[str] = Query(
        None,
        description="Natural language question about quiz performance, e.g. 'how did I do on caching?'. "
                    "When provided, uses semantic search instead of a general weakness report.",
    ),
    last_n_sessions: Optional[int] = Query(
        None,
        ge=1,
        description="Limit analysis to the N most recent quiz sessions. Omit to analyse all sessions.",
    ),
):
    """Return a personalised progress report based on past quiz sessions.

    Retrieval strategy depends on the combination of parameters:

    - **No params**: general weakness report across all sessions.
    - **question only**: semantic search across all sessions, answers the specific question.
    - **last_n_sessions only**: general weakness report scoped to the N most recent sessions.
    - **Both**: semantic search scoped to the N most recent sessions.
    """
    from src.chains.progress import build_progress_chain

    try:
        loop = asyncio.get_event_loop()

        # Resolve session scope
        target_session_ids = None
        if last_n_sessions is not None:
            target_session_ids = await loop.run_in_executor(
                None, lambda: get_last_n_session_ids(last_n_sessions)
            )

        # Retrieve documents using the appropriate strategy
        if question:
            docs = await loop.run_in_executor(
                None, lambda: query_by_topic(question, session_ids=target_session_ids)
            )
        else:
            docs = await loop.run_in_executor(
                None, lambda: query_weak_areas(session_ids=target_session_ids)
            )

        stats = await loop.run_in_executor(
            None, lambda: get_all_stats(session_ids=target_session_ids)
        )

        chain = build_progress_chain(question=question)
        report = await loop.run_in_executor(
            None, lambda: chain.invoke(docs)
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


# ── Notion Ingest (Phase 5) ─────────────────────────────────────────────────

_INGEST_TOKEN = os.getenv("INGEST_TOKEN", "")


@app.post("/notion/ingest", response_model=NotionIngestResponse, tags=["Notion"])
async def notion_ingest(body: NotionIngestRequest, request: Request):
    """Manually sync a single Notion page into the vector store.

    Protected by ``X-Ingest-Token`` header — set the ``INGEST_TOKEN`` env var.

    The page content is fetched from Notion, split into heading-scoped chunks,
    embedded, and upserted into the ``notion_notes`` pgvector table.  Any
    existing chunks for this page are deleted first so updates are idempotent.
    """
    token = request.headers.get("X-Ingest-Token", "")
    if not _INGEST_TOKEN or token != _INGEST_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Ingest-Token header.")

    from src.tools.notion_tool import extract_page_content, get_page_title, fetch_pages_by_date
    import re

    try:
        # Fetch the page object to get title and date
        client = __import__("src.tools.notion_tool", fromlist=["_get_client"]) 
        from src.tools.notion_tool import _get_client, _get_database_id
        notion_client = _get_client()

        page = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: notion_client.pages.retrieve(page_id=body.page_id),
        )

        from src.tools.notion_tool import get_page_title
        title = get_page_title(page)

        # Extract the Date property value
        props = page.get("properties", {})
        date_str = ""
        for prop in props.values():
            if prop.get("type") == "date" and prop.get("date"):
                date_str = prop["date"].get("start", "")
                break

        last_edited_time = page.get("last_edited_time", "")

        content = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: extract_page_content(body.page_id),
        )

        # Delete stale chunks before re-ingesting
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: notion_memory.delete_page(body.page_id),
        )

        chunks_upserted = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: notion_memory.ingest_page(
                page_id=body.page_id,
                title=title,
                content=content,
                date=date_str,
                last_edited_time=last_edited_time,
            ),
        )

        logger.info("notion/ingest: page_id=%s title='%s' chunks=%d", body.page_id, title, chunks_upserted)
        return NotionIngestResponse(
            page_id=body.page_id,
            title=title,
            chunks_upserted=chunks_upserted,
        )
    except Exception as exc:
        logger.error("notion/ingest error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/notion/webhook", response_model=NotionWebhookResponse, tags=["Notion"])
async def notion_webhook(request: Request):
    """Receive Notion webhook events and re-ingest updated pages.

    Notion calls this endpoint automatically when a page is created or updated.
    Verifies the ``X-Notion-Signature`` HMAC header using ``NOTION_WEBHOOK_SECRET``.
    """
    import hmac
    import hashlib
    import json

    webhook_secret = os.getenv("NOTION_WEBHOOK_SECRET", "")

    # ── Signature verification ────────────────────────────────────────────────
    if webhook_secret:
        signature_header = request.headers.get("X-Notion-Signature", "")
        raw_body = await request.body()
        expected_sig = hmac.new(
            webhook_secret.encode(),
            raw_body,
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(f"sha256={expected_sig}", signature_header):
            raise HTTPException(status_code=401, detail="Invalid webhook signature.")
        payload = json.loads(raw_body)
    else:
        # No secret configured — skip verification (dev/testing only)
        logger.warning("notion/webhook: NOTION_WEBHOOK_SECRET not set — skipping signature check")
        payload = await request.json()

    # ── Handle Workspace Verification ────────────────────────────────────────
    # Notion sends a workspace.verification event when setting up a webhook.
    # We must respond with an HTTP 200 containing the token and workspace_id.
    payload_type = payload.get("type", "")
    if payload_type == "workspace.verification":
        from fastapi.responses import JSONResponse
        event_data = payload.get("workspace.verification", {})
        token = event_data.get("token", "")
        workspace_id = event_data.get("workspace_id", "")
        
        logger.info("notion/webhook: handling workspace.verification event for workspace %s", workspace_id)
        return JSONResponse(
            content={
                "type": "workspace.verification",
                "workspace.verification": {
                    "token": token,
                    "workspace_id": workspace_id,
                }
            }
        )

    # ── Extract page_id from payload ─────────────────────────────────────────
    # Notion webhook payload structure: { "type": "page.updated", "entity": { "id": "..." } }
    entity = payload.get("entity", {})
    page_id = entity.get("id", "")

    if not page_id:
        logger.warning("notion/webhook: no page_id in payload — ignoring")
        return NotionWebhookResponse(page_id="", status="skipped", message="No page_id in payload")

    logger.info("notion/webhook: received event for page_id=%s", page_id)

    try:
        from src.tools.notion_tool import extract_page_content, get_page_title
        from src.tools.notion_tool import _get_client

        notion_client = _get_client()
        page = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: notion_client.pages.retrieve(page_id=page_id),
        )

        title = get_page_title(page)

        props = page.get("properties", {})
        date_str = ""
        for prop in props.values():
            if prop.get("type") == "date" and prop.get("date"):
                date_str = prop["date"].get("start", "")
                break

        last_edited_time = page.get("last_edited_time", "")

        content = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: extract_page_content(page_id),
        )

        await asyncio.get_event_loop().run_in_executor(
            None, lambda: notion_memory.delete_page(page_id)
        )
        chunks_upserted = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: notion_memory.ingest_page(
                page_id=page_id,
                title=title,
                content=content,
                date=date_str,
                last_edited_time=last_edited_time,
            ),
        )
        logger.info("notion/webhook: re-ingested page_id=%s chunks=%d", page_id, chunks_upserted)
        return NotionWebhookResponse(page_id=page_id, status="ingested", message=f"{chunks_upserted} chunks upserted")

    except Exception as exc:
        logger.error("notion/webhook error for page_id=%s: %s", page_id, exc)
        # Return 200 even on error — Notion will retry on non-2xx, causing infinite loops.
        return NotionWebhookResponse(page_id=page_id, status="error", message=str(exc))


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
