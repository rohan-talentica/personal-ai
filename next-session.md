# Next Session: Enhancing Progress Endpoint with Topic-Specific Queries

## What We Want to Do

**Goal**: Upgrade `/notion/progress` to accept natural language questions about quiz performance, using vector search for topic-specific analysis.

**New API Design**:
- Add optional `question` query parameter: `GET /notion/progress?question=how did I do on caching&last_n_sessions=2`
- Keep existing `last_n_sessions` parameter for session scoping.

**Retrieval Strategies**:
1. **No `question`**: Metadata filter → general weakness report (current behavior).
2. **`question` only**: Similarity search → topic-specific analysis.
3. **`last_n_sessions` only**: Metadata filter scoped to N sessions → general report for those sessions.
4. **Both params**: Similarity search scoped to N sessions → topic analysis within those sessions.

**Implementation Steps**:
- Add `similarity_search(query, k, filter)` method to `VectorStoreAdapter` and `ChromaAdapter`.
- New `query_by_topic(question, k, session_ids=None)` function in `quiz_memory.py`.
- Update `progress_chain` to answer specific questions using retrieved docs.
- Update endpoint in `main.py` to route based on params.

## Limitations

- **ChromaDB Filtering**: No `ORDER BY` or sorted `LIMIT` — session ranking requires full scan + Python processing.
- **No Intent Classification**: Not needed; LLM handles question interpretation from retrieved docs.
- **Scale**: At current data volumes, Python filtering is fine; vector search is efficient.

This design keeps the API simple while enabling powerful topic-specific insights. Ready to implement in the next session?