# Personal AI Research Assistant

A learning project to master LangChain, LangGraph, and AI agent development with human-in-the-loop workflows.

## 🎯 Learning Objectives
- Master Python for AI development
- Learn LangChain chains and prompts
- Build stateful workflows with LangGraph
- Implement human-in-the-loop patterns
- Understand memory systems (episodic, semantic, long-term)
- Build RAG (Retrieval Augmented Generation) systems
- Work with vector databases
- Implement ReAct agents
- Set up observability with LangSmith
- Deploy to AWS using CDK

## 🗓️ 10-Day Learning Plan

### ✅ Phase 1: Foundations (Days 1-2) — COMPLETE
- **Day 1**: ✅ Python essentials + First LangChain chain + LangSmith setup
- **Day 2**: ✅ Chain types, prompt templates, conversation memory
- **Day 3**: ✅ Vector embeddings, vector DBs (ChromaDB), RAG basics

### ✅ Phase 2: Intelligence Layer (Days 4-5) — COMPLETE
- **Day 4**: ✅ Advanced RAG with multiple sources and citations
- **Day 5**: ✅ ReAct agents with tools and function calling

### ✅ Phase 3: Stateful Workflows (Days 6-8) — COMPLETE
- **Day 6**: ✅ Introduction to LangGraph and state management
- **Day 7**: ✅ Human-in-the-loop interrupts and approvals
- **Day 8**: ✅ Memory systems across sessions

### ✅ Phase 4: Production (Days 9-10) — IN PROGRESS
- **Day 9**: ✅ Production patterns, FastAPI, containerization
- **Day 10**: AWS deployment with CDK (ECS Fargate)

## 🚀 Quick Start

### 1. Set up Python environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure environment variables
```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

### 3. Open the latest notebook
Open `notebooks/day9_production_patterns.ipynb` in VS Code

### 4. Run the FastAPI server
```bash
uvicorn src.api.main:app --reload --port 8000
# Swagger UI → http://localhost:8000/docs
```

### 5. Run with Docker
```bash
docker compose up --build
```

> **Current progress**: Day 9 complete — resuming at Day 10 (AWS Deployment)

## 📁 Project Structure
```
personal-ai/
├── notebooks/              # Daily learning notebooks
├── src/
│   ├── utils/
│   │   └── llm.py          # Shared LLM + embeddings factory
│   ├── chains/
│   │   ├── chat.py         # Conversational chain (Days 1-2)
│   │   └── rag.py          # RAG chain with citations (Days 3-4)
│   ├── tools/
│   │   └── custom_tools.py # calculator, word_counter, get_weather, summarizer
│   ├── agents/
│   │   └── react_agent.py  # ReAct tool-calling agent (Day 5)
│   ├── memory/
│   │   └── vector_store.py # ChromaDB helpers
│   └── api/
│       ├── main.py         # FastAPI app (Day 9)
│       ├── models.py       # Pydantic schemas
│       └── dependencies.py # FastAPI DI singletons
├── Dockerfile              # Multi-stage production image
├── docker-compose.yml      # App + ChromaDB volume
├── docs/                   # Learning notes
├── infrastructure/         # AWS CDK code (Day 10)
└── requirements.txt        # Python dependencies
```

## 🔑 API Keys Needed
- **OpenRouter**: For LLM access (multiple models)
- **LangSmith** (Optional): For tracing and observability

## 📚 Resources
- [LangChain Docs](https://python.langchain.com/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangSmith Docs](https://docs.smith.langchain.com/)
- [OpenRouter](https://openrouter.ai/)

## 🎓 Daily Progress
- [x] Day 1: Python + LangChain Basics ✅
- [x] Day 2: Chains & Memory ✅
- [x] Day 3: RAG Fundamentals ✅
- [x] Day 4: Advanced RAG with Citations ✅
- [x] Day 5: ReAct Agents ✅
- [x] Day 6: LangGraph Intro ✅
- [x] Day 7: Human-in-the-Loop ✅
- [x] Day 8: Memory Systems ✅
- [x] Day 9: Production Patterns — FastAPI + Docker ✅
- [ ] Day 10: AWS Deployment
