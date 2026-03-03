# Next Session — ECS TODO

## 1. Redeploy (pending code fix)
`src/utils/llm.py` was fixed to read `OPENROUTER_API_KEY` lazily at call-time instead of at module import (the root cause of the 401 errors on ECS).

```bash
cd infrastructure && cdk deploy --profile personal-ai --require-approval never
```

## 2. Add a `/rag/ingest` API endpoint
ChromaDB on ECS starts empty every task restart (ephemeral storage). Without an ingest endpoint, RAG always returns empty `sources`.

Add `POST /rag/ingest` to `src/api/main.py` that accepts a URL or raw text, chunks it, and stores it in ChromaDB. This is the only practical way to feed data in without ECS Exec.

## 3. Verify the fix end-to-end
After deploy:
1. Hit `POST /chat` → should no longer 401
2. Ingest something via `POST /rag/ingest`
3. Hit `POST /rag/query` → should return non-empty `sources`

## 4. (Optional) Enable ECS Exec for debugging
Lets you shell into the running container to inspect ChromaDB directly.

Add to `personal_ai_stack.py` → `ApplicationLoadBalancedFargateService`:
```python
enable_execute_command=True,
```
Then: `aws ecs execute-command --cluster personal-ai-cluster --task <task-id> --interactive --command "/bin/sh" --profile personal-ai --region ap-south-1`
