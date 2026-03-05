# Personal AI — Project Roadmap

**Goal**: Iteratively improve this project while learning Python and agent orchestration.
No fixed deadline — prioritise learning over speed.

---

## 🎯 North Star Use Case

**Notion Weekly Review Agent** — a personal learning assistant that reads your
daily Notion notes and helps you revise, recall, and consolidate what you studied.

This use case was chosen because:
- You are already taking daily notes in Notion (system design, backend, etc.)
- It requires real agent orchestration skills (tools, memory, HITL)
- You will use it constantly, so quality feedback is immediate

---

## ✅ Phase 1 — Notion Daily Revision (Mode 1) — COMPLETE

**What it does**: Ask in natural language what you learned on a given day,
get a structured revision summary pulled live from Notion.

**Endpoint**: `POST /notion/revise`
**Request**: `{ "query": "what did I learn on Monday?" }`
**Response**: `{ "date": "...", "pages_found": N, "summary": "..." }`

### How it works
1. LLM (temperature=0) extracts the target date from the natural-language query
2. Notion API fetches all pages in your database matching that date
3. Block-level content is extracted (headings, bullets, paragraphs)
4. Revision chain generates a structured markdown summary:
   - 🧠 Key Concepts
   - 📌 Topics Covered
   - 💡 Things to Remember
   - ❓ Questions to Explore

### Files added
| File | Purpose |
|------|---------|
| `src/tools/notion_tool.py` | Notion API client — date-filtered DB query + block extraction |
| `src/chains/revision.py` | LCEL revision chain with structured prompt |
| `src/api/models.py` | Added `RevisionRequest`, `RevisionResponse` |
| `src/api/main.py` | Added `POST /notion/revise` endpoint |

### Key decisions made
- **notion-client==2.2.1** pinned — v3 removed `databases.query()`
- **LLM date resolution (temperature=0)** — handles "Monday", "yesterday", "March 3rd" naturally without hardcoding
- **Lazy Notion client singleton** — init happens at first call, not import time

### Still to do for Mode 1
- [ ] Add `NOTION_API_KEY` + `NOTION_DATABASE_ID` to ECS via Secrets Manager
- [ ] Redeploy to ECS and verify end-to-end on production

---

## ✅ Phase 2 — Socratic Revision (Mode 2) — COMPLETE

**What it does**: Instead of just summarising, the agent *quizzes* you.
You answer → it evaluates → it follows up on weak areas.

**Why this is interesting**: This is real Human-in-the-Loop (HITL) in LangGraph —
the agent decides the next question based on your previous answer.

### What to build
- Replace the simple revision chain with a **LangGraph graph**:
  ```
  fetch_notes → generate_question → [user answers] → evaluate_answer
       ↑_____________[weak area? ask follow-up]__________________|
  ```
- New endpoint: `POST /notion/quiz` — starts a quiz session
- State stored per `session_id` (SQLite checkpointer)
- Returns one question at a time; accepts answers via `POST /notion/quiz/{session_id}/answer`

### What you will learn
- LangGraph state graphs in production (not just notebooks)
- Conditional edges and interrupt/resume patterns
- Session-based state with a checkpointer (SQLite → Postgres later)

---

---
 
 ## ✅ Phase 3 — Long-Term Progress Tracker (Mode 3) — COMPLETE
 
 **What it does**: Tracks your knowledge over multiple weeks by storing quiz Q&A
 into a persistent memory store. Generates an LLM-powered weakness report to
 highlight recurring gaps.
 
 **Endpoint**: `GET /notion/progress`
 
 ### What was built
 - **Quiz Persistence Hook**: Each answer in a quiz session is embedded into the `quiz_history` collection with metadata (`concept`, `is_correct`, `confidence_score`).
 - **Progress Chain**: An LCEL chain that retrieves your 30 most recent weak areas and generates a markdown report with actionable revision suggestions.
 - **ChromaDB Cloud Migration**: Switched from ephemeral local storage to persistent ChromaDB Cloud.
 - **Vector Store Abstraction**: Implemented the **Strategy Pattern** for the memory layer. The code is now provider-agnostic (`VectorStoreAdapter`).
 
 ### Files added/modified
 | File | Purpose |
 |------|---------|
 | `src/memory/quiz_memory.py` | Quiz history ingestion and retrieval logic |
 | `src/chains/progress.py` | LLM weakness report generator chain |
 | `src/memory/base.py` | `VectorStoreAdapter` interface (ABC) |
 | `src/memory/providers/` | Concrete implementations (starting with `ChromaAdapter`) |
 | `src/memory/factory.py` | Provider factory controlled by `VECTOR_STORE_PROVIDER` env var |
 
 ### Key lessons learned
 - **Repository Pattern in Agents**: Decoupling the vector store provider makes the agent infrastructure portable (Pinecone/Qdrant ready).
 - **Granular Analytics**: Storing Q&A *per answer* allows for better tracking than storing *per session*.
 - **LLM-as-Analyst**: Using a dedicated chain to summarize retrieved failures into a coherent learning path.

---

## 🔮 Future Options (Post Mode 3)

These are the broader directions we discussed. Pick based on what is most interesting
at the time.

| Option | What it is | Key learning |
|--------|-----------|--------------|
| **Stateful LangGraph Agent** | Replace the ReAct agent with a proper LangGraph graph — persistent state, conditional edges, real HITL | LangGraph in prod, thread persistence |
| **Real Tools** | Add web search (Tavily), URL reader, code executor | Tool schemas, error handling, sandboxing |
| **Better RAG** | Hybrid search (BM25 + vector), re-ranking | Advanced retrieval, evaluation |
| **Multi-agent** | Orchestrator + specialist agents (researcher, summariser, quizzer) | Agent-to-agent coordination |

---

## 🏗️ What Is Deployed Today

| **API** | FastAPI on ECS Fargate (ap-south-1), behind ALB |
| **LLM** | OpenRouter → GPT-3.5-turbo (default) |
| **Vector DB** | **ChromaDB Cloud** (Persistent, multi-tenant) |
| **Store Provider**| Pluggable Adapter Pattern (Strategy) |
| **Notion** | Integration connected, DB ID `319a5bf6db5e80549961c5f23841073e` |
| **Infra** | AWS CDK (Python), ECR for Docker images |

### Useful commands
```bash
# Run locally
uvicorn src.api.main:app --reload --port 8000

# Deploy to ECS
cd infrastructure && cdk deploy --profile personal-ai --require-approval never

# Test revision summary
curl -X POST http://localhost:8000/notion/revise \
  -H "Content-Type: application/json" \
  -d '{"query": "what did I learn on Monday?"}'

# Test progress report (NEW)
curl -s http://localhost:8000/notion/progress | python3 -m json.tool
```
