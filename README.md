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

### Phase 2: Intelligence Layer (Days 3-5)
- **Day 3**: Vector embeddings, vector DBs, RAG basics
- **Day 4**: Advanced RAG with multiple sources and citations
- **Day 5**: ReAct agents with tools and function calling

### Phase 3: Stateful Workflows (Days 6-8)
- **Day 6**: Introduction to LangGraph and state management
- **Day 7**: Human-in-the-loop interrupts and approvals
- **Day 8**: Memory systems across sessions

### Phase 4: Production (Days 9-10)
- **Day 9**: Production patterns, FastAPI, containerization
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

### 3. Start with Day 1
Open `notebooks/day1_python_langchain_basics.ipynb` in VS Code

## 📁 Project Structure
```
personal-ai/
├── notebooks/          # Daily learning notebooks
├── src/
│   ├── agents/        # LangGraph agents
│   ├── chains/        # LangChain chains
│   ├── memory/        # Memory implementations
│   ├── tools/         # Custom tools
│   └── api/           # FastAPI endpoints
├── docs/              # Learning notes
├── infrastructure/    # AWS CDK code
└── requirements.txt   # Python dependencies
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
- [ ] Day 2: Chains & Memory
- [ ] Day 3: RAG Fundamentals
- [ ] Day 4: Advanced RAG
- [ ] Day 5: ReAct Agents
- [ ] Day 6: LangGraph Intro
- [ ] Day 7: Human-in-the-Loop
- [ ] Day 8: Memory Systems
- [ ] Day 9: Production Prep
- [ ] Day 10: AWS Deployment
