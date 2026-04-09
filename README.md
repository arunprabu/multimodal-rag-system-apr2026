How to run this fast apu project?
`uv run uvicorn main:app --reload`

Install the necessary packages

```
uv add fastapi uvicorn[standard]
uv add sqlalchemy[asyncio] asyncpg pgvector
uv add docling
uv add langgraph langchain
uv add google-genai
uv add python-multipart
```

# To run ingestion

```
uv run python -m src.ingestion.ingestion
```

====
db query
===

```
-- Enable the pgvector extension (once per DB)-- Enable the pgvector extension (once per DB)
CREATE EXTENSION IF NOT EXISTS vector;

-- Document registry
CREATE TABLE documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename    TEXT NOT NULL UNIQUE,
    source_path TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Multimodal chunks
CREATE TABLE multimodal_chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id       UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_type   TEXT NOT NULL CHECK (chunk_type IN ('text', 'table', 'image')),
    element_type TEXT,
    content      TEXT NOT NULL,
    image_bytes  BYTEA,
    mime_type    TEXT,
    page_number  INTEGER,
    section      TEXT,
    source_file  TEXT,
    position     JSONB,
    embedding    VECTOR(1536),
    metadata     JSONB,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW index for fast cosine similarity search
CREATE INDEX multimodal_chunks_embedding_hnsw_idx
    ON multimodal_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```
