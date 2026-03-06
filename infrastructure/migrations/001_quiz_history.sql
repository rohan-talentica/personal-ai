-- Migration 001: quiz_history table with pgvector support
--
-- Run this once against your Supabase database before starting the app.
-- Supabase enables the vector extension by default; the CREATE EXTENSION
-- line is included for completeness / non-Supabase Postgres setups.
--
-- Usage (psql):
--   psql $DATABASE_URL -f infrastructure/migrations/001_quiz_history.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS quiz_history (
    id               UUID    PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Session / provenance
    session_id       TEXT    NOT NULL,          -- LangGraph thread_id for this quiz session
    notes_date       TEXT    NOT NULL,          -- YYYY-MM-DD of the Notion notes being quizzed
    quiz_taken_at    DATE    NOT NULL,          -- calendar date the quiz was actually run

    -- Q&A content
    concept          TEXT    NOT NULL,          -- core concept / topic being tested
    question         TEXT    NOT NULL,
    answer           TEXT    NOT NULL,          -- developer's answer
    feedback         TEXT    NOT NULL,          -- LLM evaluator feedback

    -- Scoring
    is_correct       BOOLEAN NOT NULL,          -- true = answered correctly
    confidence_score FLOAT   NOT NULL,          -- 0.0–1.0 LLM-rated understanding score

    -- Embedding source
    content          TEXT    NOT NULL,          -- "Concept: X\nQ: ...\nA: ...\nFeedback: ..."

    -- Vector column (text-embedding-3-small → 1536 dims)
    embedding        VECTOR(1536)
);

-- ── Scalar indexes (filtering / sorting) ────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_qh_session
    ON quiz_history (session_id);

CREATE INDEX IF NOT EXISTS idx_qh_taken_at
    ON quiz_history (quiz_taken_at DESC);

CREATE INDEX IF NOT EXISTS idx_qh_is_correct
    ON quiz_history (is_correct);

-- ── ANN index (semantic similarity search) ──────────────────────────────────
-- HNSW builds incrementally — no minimum row count needed before it's useful.

CREATE INDEX IF NOT EXISTS idx_qh_embedding
    ON quiz_history USING hnsw (embedding vector_cosine_ops);
