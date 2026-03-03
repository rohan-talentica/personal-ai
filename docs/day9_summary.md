# Day 9: Production Patterns — Summary

**Date completed:** 2 March 2026

---

## Overview

Day 9 converted all the notebook-based work from Days 1–8 into a production-ready Python package and exposed it via a FastAPI REST API. The app was then containerised with Docker and docker-compose.

---

## Files Created

### `src/` — Python Package

| File | Purpose |
|------|---------|
| `src/__init__.py` | Marks `src/` as a Python package |
| `src/utils/llm.py` | Shared `get_llm()` and `get_embeddings()` factory using `lru_cache`. All modules import from here so OpenRouter config lives in one place. |
| `src/chains/chat.py` | Stateless conversational LCEL chain (`prompt \| llm \| StrOutputParser`). Accepts an optional `history` list of `BaseMessage` for multi-turn context. |
| `src/chains/rag.py` | RAG chain: loads a persisted ChromaDB collection, retrieves the top-k docs, and asks the LLM to answer with inline `[Source: title]` citations. Returns `{"answer": ..., "sources": [...]}`. |
| `src/tools/custom_tools.py` | Four `@tool`-decorated functions ported from the Day 5 notebook: `calculator`, `word_counter`, `get_weather`, `text_summarizer`. Exported as `ALL_TOOLS`. |
| `src/agents/react_agent.py` | `build_react_agent()` — wraps `create_tool_calling_agent` + `AgentExecutor` from `langchain_classic`. Returns a configured executor with all four custom tools. `run_agent()` is a convenience wrapper that also returns intermediate tool-call steps. |
| `src/memory/vector_store.py` | `build_vector_store()` — creates/overwrites a ChromaDB collection from a list of `Document`s. `load_vector_store()` — loads an existing collection from disk. `get_retriever()` — shortcut returning a retriever with a configurable `k`. |
| `src/api/models.py` | Pydantic v2 request/response schemas for all API endpoints: `ChatRequest/Response`, `RAGRequest/Response`, `AgentRequest/Response`, `HealthResponse`. |
| `src/api/dependencies.py` | FastAPI dependency injection. `get_chat_chain()`, `get_rag_chain()`, `get_agent()` are each decorated with `@lru_cache` so the expensive objects (LLM client, vector store, agent) are initialised once and reused across requests. |
| `src/api/main.py` | The FastAPI application. See endpoints table below. |

### FastAPI Endpoints (`src/api/main.py`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe — returns `{"status": "ok", "version": "1.0.0"}` |
| `POST` | `/chat` | Stateless conversational chain. Accepts `question` + optional `history`. |
| `POST` | `/chat/stream` | Same chain, but streams tokens as Server-Sent Events (SSE). Consume with `EventSource` or `httpx` streaming. |
| `POST` | `/rag/query` | RAG pipeline. Returns `answer` + list of `sources` with title, document type, and snippet. Returns `503` if the ChromaDB collection hasn't been populated. |
| `POST` | `/agent/run` | ReAct agent. Returns `output` + `steps` (list of tool name, input, observation). |

#### Production patterns used in the API

- **Lifespan handler** — warms up all singletons at startup; logs warnings if a resource (e.g. vector store) isn't available yet.
- **CORS middleware** — open for local dev; restrict `allow_origins` before production deployment.
- **`run_in_executor`** — wraps synchronous LangChain calls so the async FastAPI event loop is never blocked.
- **Global exception handler** — catches unhandled exceptions, logs them, and returns a clean `500` JSON response.

### Containerisation

| File | Purpose |
|------|---------|
| `Dockerfile` | **Multi-stage build.** Stage 1 (`builder`) installs all Python wheels into `/opt/venv`. Stage 2 (`runtime`) copies only the venv — no build tools in the final image. Runs as non-root user `appuser` (UID 1001). Includes a `HEALTHCHECK` hitting `/health`. |
| `docker-compose.yml` | Defines the `api` service (built from `Dockerfile`), maps port `8000`, loads `.env`, and mounts a named volume `chroma_data` at `/app/data/chroma` so ChromaDB persists across container restarts. |

### Notebook

`notebooks/day9_production_patterns.ipynb` — 14-cell interactive walkthrough:

1. Environment setup + `sys.path` configuration
2. Smoke-test each `src/` module (LLM factory, tools, chat chain, agent)
3. Inspect registered FastAPI routes
4. Start `uvicorn` as a background subprocess and wait for `/health` to respond
5. Test every endpoint with `httpx` (health, chat, multi-turn chat, streaming, RAG, agent)
6. Docker build/run instructions
7. Teardown cell to stop the background server
8. Day 9 summary and Day 10 preview

---

## Key Concepts Learned

| Concept | How it was applied |
|---------|--------------------|
| **Module packaging** | Notebook one-off code → importable `src/` with `__init__.py` re-exports |
| **Dependency injection** | `@lru_cache` singletons + FastAPI `Depends()` — one initialisation, many requests |
| **Async + sync bridge** | `asyncio.get_event_loop().run_in_executor(None, ...)` keeps the ASGI event loop free |
| **SSE streaming** | `StreamingResponse` wrapping an `async def` generator that yields `data: <token>\n\n` lines |
| **Multi-stage Docker** | Builder stage installs deps; runtime stage is lean — no compiler toolchain in prod |
| **Volume persistence** | Named Docker volume keeps ChromaDB data alive across `docker compose down/up` cycles |

---

## How to Run

```bash
# Local dev
uvicorn src.api.main:app --reload --port 8000
# → Swagger UI at http://localhost:8000/docs

# Docker
docker compose up --build
```

---

## What's Next — Day 10: AWS Deployment

- Define ECS Fargate task + service with AWS CDK (Python)
- Push Docker image to ECR
- Application Load Balancer with HTTPS termination
- Secrets (API keys) via AWS Secrets Manager
- CloudWatch Logs + LangSmith tracing in production
