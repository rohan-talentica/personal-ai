-- Migration 002: notion_notes table with pgvector support
--
-- Stores pre-embedded Notion page chunks for semantic retrieval.
-- Each row is one chunk of a Notion page, with the full chunk text + embedding.
--
-- Usage (psql):
--   psql $DATABASE_URL -f infrastructure/migrations/002_notion_notes.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS notion_notes (
    id               UUID    PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Page provenance
    page_id          TEXT    NOT NULL,   -- Notion page UUID
    title            TEXT    NOT NULL,   -- Page title
    date             TEXT    NOT NULL,   -- YYYY-MM-DD from the Notion Date property
    last_edited_time TEXT    NOT NULL,   -- ISO timestamp from Notion (used for stale-check)
    chunk_index      INT     NOT NULL,   -- 0-based position of this chunk within the page

    -- Content
    content          TEXT    NOT NULL,   -- Full text of this chunk (heading + body)

    -- Vector column (text-embedding-3-small → 1536 dims)
    embedding        VECTOR(1536)
);

-- ── Scalar indexes ────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_nn_page_id
    ON notion_notes (page_id);

CREATE INDEX IF NOT EXISTS idx_nn_date
    ON notion_notes (date);

-- ── ANN index (semantic similarity search) ────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_nn_embedding
    ON notion_notes USING hnsw (embedding vector_cosine_ops);
