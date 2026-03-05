# Available LLM Models for TradingAgents

This document lists all tested and available models for use with the TradingAgents framework.

## ✅ Google Gemini Models (Free Tier)

All models tested on: 2026-02-15

### Recommended Models

| Model Name | Status | Context Window | Best For | Rate Limits |
|------------|--------|----------------|----------|-------------|
| `gemini-2.5-flash-preview-09-2025` | ✅ Working | 1M+ tokens | **Primary choice** - Latest tech, best performance | 15 RPM, 1500 RPD |
| `gemini-flash-latest` | ✅ Working | 1M+ tokens | Stable fallback - Proven reliability (Gemini 1.5) | 15 RPM, 1500 RPD |
| `gemini-3-flash-preview` | ✅ Working | 1M+ tokens | Experimental - Cutting edge features | 15 RPM, 1500 RPD |
| `gemini-flash-lite-latest` | ✅ Working | 1M+ tokens | Speed-optimized - Faster responses | 15 RPM, 1500 RPD |

### Not Available on Free Tier

| Model Name | Status | Reason |
|------------|--------|--------|
| `gemini-2.5-pro` | ❌ Not Available | Requires paid plan (quota limit: 0) |
| `gemini-pro-latest` | ❌ Not Available | Requires paid plan (quota limit: 0) |
| `gemini-2.0-flash-exp` | ❌ Not Found | Deprecated or removed |

**Rate Limit Abbreviations:**
- RPM = Requests Per Minute
- RPD = Requests Per Day
- TPM = Tokens Per Minute

## 🏠 Local Models (Ollama)

### Available Models

| Model Name | Size | Context Window | Best For | Requirements |
|------------|------|----------------|----------|--------------|
| `llama3.2:3b` | 3B | 8K-32K tokens | Simple tasks, fast inference | 4GB RAM |
| `llama3.2:7b` | 7B | 8K-32K tokens | General purpose | 8GB RAM |
| `mistral:7b` | 7B | 8K-32K tokens | Code and reasoning | 8GB RAM |
| `qwen2.5:7b` | 7B | 32K tokens | Multilingual, good reasoning | 8GB RAM |

### Pros & Cons

**Advantages:**
- ✅ Completely free, unlimited requests
- ✅ Full privacy - data never leaves your machine
- ✅ Works offline
- ✅ No rate limits

**Disadvantages:**
- ❌ Smaller context window (8K-32K vs 1M+)
- ❌ Lower reasoning quality than Gemini
- ❌ Slower on CPU (needs GPU for good performance)
- ❌ More prone to hallucinations

## 🌐 OpenRouter Models (Free Tier)

OpenRouter provides unified API access to 31+ free models from various providers. **9 models verified working.**

### Recommended Models

| Model Name | Status | Context Window | Parameters | Best For |
|------------|--------|----------------|------------|----------|
| `openrouter/free` | ✅ Working | 200K tokens | Auto-router | **Auto-selects best free model** |
| `openai/gpt-oss-120b:free` | ✅ Working | 131K tokens | 120B | **Large model, high capability** |
| `deepseek/deepseek-r1-0528:free` | ✅ Working | Unknown | Unknown | **DeepSeek reasoning model** |
| `arcee-ai/trinity-large-preview:free` | ✅ Working | 131K tokens | Unknown | **Large preview model** |
| `nvidia/nemotron-3-nano-30b-a3b:free` | ✅ Working | 256K tokens | 30B | Large context analysis |
| `stepfun/step-3.5-flash:free` | ✅ Working | 256K tokens | Unknown | Fast inference |
| `upstage/solar-pro-3:free` | ✅ Working | 128K tokens | Unknown | General purpose |
| `z-ai/glm-4.5-air:free` | ✅ Working | 131K tokens | Unknown | Balanced performance |
| `nvidia/nemotron-nano-9b-v2:free` | ✅ Working | 128K tokens | 9B | Efficient inference |

### Rate-Limited Models

| Model Name | Status | Reason |
|------------|--------|--------|
| `qwen/qwen3-coder:free` | ⚠️ Rate Limited | 480B model - high demand |
| `qwen/qwen3-next-80b-a3b-instruct:free` | ⚠️ Rate Limited | 80B model - high demand |

### Pros & Cons

**Advantages:**
- ✅ 31+ free models available
- ✅ Unified API for all providers
- ✅ Auto-router selects best available model
- ✅ Large context windows (up to 256K)
- ✅ Access to powerful models (up to 480B parameters)
- ✅ No need for multiple API keys

**Disadvantages:**
- ❌ Popular models may be rate-limited
- ❌ Requires internet connection
- ❌ Free tier has usage limits
- ❌ Model availability can change

## 📊 Comparison for Trading Use Cases

| Use Case | Recommended Model | Alternative 1 | Alternative 2 |
|----------|-------------------|---------------|---------------|
| **Market Analysis** | `gemini-2.5-flash-preview-09-2025` | `nvidia/nemotron-3-nano-30b-a3b:free` (OpenRouter) | `llama3.2:7b` (offline) |
| **Strategy Backtesting** | `gemini-2.5-flash-preview-09-2025` | `openrouter/free` | `gemini-flash-latest` |
| **Real-time Trading** | `gemini-flash-lite-latest` (speed) | `stepfun/step-3.5-flash:free` (OpenRouter) | Local model (privacy) |
| **Research & Planning** | `gemini-2.5-flash-preview-09-2025` | `upstage/solar-pro-3:free` (OpenRouter) | `gemini-3-flash-preview` |
| **Simple Formatting** | `llama3.2:3b` (local) | `gemini-flash-lite-latest` | `nvidia/nemotron-nano-9b-v2:free` |
| **Development/Testing** | `llama3.2:7b` (local) | `openrouter/free` | Any Gemini Flash |
| **Code Generation** | `qwen/qwen3-coder:free` (OpenRouter) | `gemini-2.5-flash-preview-09-2025` | `llama3.2:7b` |

## 🔧 Configuration

### For Initial Testing (Local Model)

Set in `.env`:
```bash
# Use Ollama for initial testing
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:7b
```

### For Production (Gemini)

Set in `.env`:
```bash
# Use Gemini 2.5 Flash for production
LLM_PROVIDER=google
GOOGLE_MODEL=gemini-2.5-flash-preview-09-2025
GOOGLE_API_KEY=your_api_key_here
```

### OpenRouter (Unified Access)

Set in `.env`:
```bash
# Use OpenRouter for access to multiple providers
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=openrouter/free  # Auto-selects best free model
# Or choose specific model:
# OPENROUTER_MODEL=nvidia/nemotron-3-nano-30b-a3b:free
```

### Hybrid Setup (Recommended)

Use multiple providers for maximum reliability:
```bash
# Primary: Gemini for best performance
GOOGLE_API_KEY=your_api_key_here
GOOGLE_MODEL=gemini-2.5-flash-preview-09-2025

# Fallback 1: OpenRouter for variety
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=openrouter/free

# Fallback 2: Ollama for offline/privacy
OLLAMA_MODEL=llama3.2:7b
```

## 📝 Testing Your Setup

Run the test scripts to verify your API keys:
```bash
# Test Google Gemini
python3 test_google_api.py

# Test OpenRouter
python3 test_openrouter_api.py
```

For Ollama, ensure it's running:
```bash
ollama serve
ollama pull llama3.2:7b
```

## 🔄 Model Selection Strategy

1. **Start with local** (`llama3.2:7b`) for development and testing
2. **Switch to cloud** when you need better performance:
   - **Gemini** (`gemini-2.5-flash-preview-09-2025`) - Best quality, 1M+ context
   - **OpenRouter** (`openrouter/free`) - Auto-routing, variety of models
3. **Use hybrid** for optimal reliability:
   - Gemini → OpenRouter → Ollama fallback chain
   - Automatic failover if one service is down or rate-limited

---

*Last updated: 2026-02-15*
*Tested with Google API and OpenRouter API*
