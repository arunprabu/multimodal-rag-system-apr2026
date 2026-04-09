import base64
import hashlib
import json
import os
import pathlib

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from google import genai
from google.genai import types as genai_types

load_dotenv()

# ---------------------------------------------------------------------------
# Connection setup
#
# The .env connection string uses SQLAlchemy's dialect prefix
# "postgresql+psycopg://" so that LangChain can parse it.
# psycopg.connect() expects the standard "postgresql://" URI, so we strip
# the dialect marker before passing it to psycopg.
# ---------------------------------------------------------------------------
_PG_CONNECTION = os.getenv("PG_CONNECTION_STRING", "")
_PG_DSN = _PG_CONNECTION.replace("postgresql+psycopg://", "postgresql://")

# How many chunks to embed per API call.
# Google's embedding API accepts up to 100 texts per batch.
_EMBED_BATCH_SIZE = 50

# ---------------------------------------------------------------------------
# Issue 8 + 11 fix: Module-level google-genai client singleton.
#
# gemini-embedding-2-preview is a multimodal embedding model: it places text
# AND image bytes into the SAME vector space (co-embedding).  This enables
# true cross-modal retrieval — a text query like "revenue bar chart" can
# match the actual chart image directly, not just its caption text.
#
# _embed_texts()  → batched text embedding for text/table chunks and queries
# _embed_image()  → visual embedding of raw PNG bytes for image chunks
# ---------------------------------------------------------------------------
_EMBED_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-2-preview")
# Matryoshka dimensionality supported by gemini-embedding-2-preview:
# 768 | 1024 | 1536 | 3072.  1536 matches the existing DB schema.
_EMBED_DIMENSIONS = 1536

_genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of text strings using gemini-embedding-2-preview.

    Batches up to _EMBED_BATCH_SIZE texts per API call to minimise
    round-trips while staying within the model's per-request limit.
    """
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[i : i + _EMBED_BATCH_SIZE]
        contents = [
            genai_types.Content(parts=[genai_types.Part.from_text(text=t)])
            for t in batch
        ]
        result = _genai_client.models.embed_content(
            model=_EMBED_MODEL,
            contents=contents,
            config=genai_types.EmbedContentConfig(
                output_dimensionality=_EMBED_DIMENSIONS
            ),
        )
        embeddings.extend(e.values for e in result.embeddings)
    return embeddings


def _embed_image(image_bytes: bytes) -> list[float]:
    """Embed raw PNG image bytes using gemini-embedding-2-preview.

    Passes the image directly to the multimodal model so the embedding
    captures visual structure, colour, layout and text-in-image — not just
    caption words.  The resulting vector lives in the same space as text
    embeddings from _embed_texts(), so a natural-language query retrieves
    visually matching images even when captions are absent or generic.
    """
    result = _genai_client.models.embed_content(
        model=_EMBED_MODEL,
        contents=genai_types.Content(
            parts=[
                genai_types.Part.from_bytes(
                    data=image_bytes, mime_type="image/png"
                )
            ]
        ),
        config=genai_types.EmbedContentConfig(
            output_dimensionality=_EMBED_DIMENSIONS
        ),
    )
    return result.embeddings[0].values

# ---------------------------------------------------------------------------
# Issue 9 fix: Lazy connection pool — reuses existing TCP connections instead
# of opening a new one per request. Created on first use to avoid failing at
# import time when the DB is not yet available (e.g. during tests).
# ---------------------------------------------------------------------------
_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    """Return the module-level connection pool, creating it on first call."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            _PG_DSN,
            min_size=2,
            max_size=10,
            kwargs={"row_factory": dict_row},
        )
    return _pool


def get_db_conn():
    """Return a pooled connection context manager.

    Usage:
        with get_db_conn() as conn:
            with conn.cursor() as cur: ...
    """
    return _get_pool().connection()


# ---------------------------------------------------------------------------
# Document registry
# ---------------------------------------------------------------------------

def upsert_document(filename: str, source_path: str) -> str:
    """Insert a document record and return its UUID.

    Uses ON CONFLICT so re-ingesting the same filename updates the path
    and returns the *existing* doc_id rather than creating a duplicate.
    This makes ingestion idempotent at the document level.
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (filename, source_path)
                VALUES (%s, %s)
                ON CONFLICT (filename) DO UPDATE
                    SET source_path = EXCLUDED.source_path,
                        ingested_at  = now()
                RETURNING id
                """,
                (filename, source_path),
            )
            row = cur.fetchone()
        conn.commit()
    return str(row["id"])


# ---------------------------------------------------------------------------
# Chunk storage
# ---------------------------------------------------------------------------

def store_chunks(chunks: list[dict], doc_id: str) -> int:
    """Embed each chunk and insert it into the multimodal_chunks table.

    Args:
        chunks:  List of dicts produced by parse_document() / ingestion.py.
                 Each dict must have: content (str), content_type (str),
                 metadata (dict with page_number, section, source_file,
                 element_type, position, image_base64).
        doc_id:  UUID string of the parent document (from upsert_document).

    Returns:
        Number of rows inserted.

    Embedding strategy:
        text/table chunks are batch-embedded via _embed_texts() using
        gemini-embedding-2-preview (up to _EMBED_BATCH_SIZE per API call).
        image chunks are embedded via _embed_image() which sends the raw
        PNG bytes to the same model, placing images in the same vector
        space as text — enabling true cross-modal retrieval.

    Vector storage:
        pgvector accepts the '[f1,f2,…]' string literal when cast with
        ::vector. We build that string directly to avoid needing the
        separate pgvector Python package.

    Image storage:
        image_base64 from metadata is decoded to raw bytes and stored in
        the BYTEA column. The JSONB metadata column does NOT duplicate it,
        keeping metadata lean.
    """
    if not chunks:
        return 0

    # ── Compute embeddings per modality ──────────────────────────────────────
    # text/table chunks → batched text embedding (efficient, one API call per 50)
    # image chunks      → multimodal visual embedding of raw PNG bytes
    #
    # Both use gemini-embedding-2-preview which projects text AND images into
    # the same vector space — this is what enables cross-modal retrieval.
    all_embeddings: list[list[float]] = [None] * len(chunks)  # type: ignore[list-item]

    # Batch-embed all text and table chunks together for efficiency
    text_indices = [
        i for i, c in enumerate(chunks) if c["content_type"] in ("text", "table")
    ]
    if text_indices:
        text_contents = [chunks[i]["content"] for i in text_indices]
        for idx, emb in zip(text_indices, _embed_texts(text_contents)):
            all_embeddings[idx] = emb

    # Embed image chunks via raw PNG bytes — captures visual content directly
    for i, chunk in enumerate(chunks):
        if chunk["content_type"] != "image":
            continue
        img_b64 = chunk["metadata"].get("image_base64")
        if img_b64:
            all_embeddings[i] = _embed_image(base64.b64decode(img_b64))
        else:
            # No image bytes (caption-only fallback) — embed caption text
            all_embeddings[i] = _embed_texts([chunk["content"]])[0]

    # ── Insert rows ───────────────────────────────────────────────────────────
    # Issue 10 fix: Only store fields in JSONB that don't already have a
    # dedicated column — the rest are redundant and waste storage.
    _DEDICATED_COLUMNS = {
        "content_type", "element_type", "section",
        "page_number", "source_file", "position", "image_base64",
    }

    rows_inserted = 0
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            # Issue 4 fix: Delete stale chunks before re-inserting so that
            # re-ingesting the same document does not create duplicates.
            cur.execute(
                "DELETE FROM multimodal_chunks WHERE doc_id = %s::uuid",
                (doc_id,),
            )

            for chunk, embedding in zip(chunks, all_embeddings):
                meta = chunk["metadata"]

                # Issue 18 fix: Save image bytes to the filesystem and store
                # only the file path in the DB. This avoids bloating PostgreSQL
                # with large BYTEA columns that slow down vacuuming and queries.
                img_b64 = meta.get("image_base64")
                image_path: str | None = None
                mime_type = "image/png" if img_b64 else None
                if img_b64:
                    image_bytes = base64.b64decode(img_b64)
                    img_dir = pathlib.Path("data/images")
                    img_dir.mkdir(parents=True, exist_ok=True)
                    img_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
                    img_file = img_dir / f"{doc_id}_{img_hash}.png"
                    img_file.write_bytes(image_bytes)
                    image_path = str(img_file)

                # pgvector vector literal: '[0.1, 0.2, …]'
                embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

                # Exclude fields that already have dedicated columns from JSONB.
                clean_meta = {k: v for k, v in meta.items() if k not in _DEDICATED_COLUMNS}

                cur.execute(
                    """
                    INSERT INTO multimodal_chunks (
                        doc_id, chunk_type, element_type, content,
                        image_path, mime_type,
                        page_number, section, source_file,
                        position, embedding, metadata
                    ) VALUES (
                        %s::uuid, %s, %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        %s::jsonb, %s::vector, %s::jsonb
                    )
                    """,
                    (
                        doc_id,
                        chunk["content_type"],       # chunk_type column
                        meta.get("element_type"),    # raw Docling label
                        chunk["content"],            # text / markdown / caption
                        image_path,                  # filesystem path (None for text/table)
                        mime_type,
                        meta.get("page_number"),
                        meta.get("section"),
                        meta.get("source_file"),
                        json.dumps(meta.get("position")) if meta.get("position") else None,
                        embedding_str,               # ::vector cast
                        json.dumps(clean_meta),      # JSONB catch-all
                    ),
                )
                rows_inserted += 1
        conn.commit()

    return rows_inserted


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------

def similarity_search(
    query: str,
    k: int = 5,
    chunk_type: str | None = None,
) -> list[dict]:
    """Find the k most similar chunks to a natural-language query.

    Args:
        query:      Natural-language question or search string.
        k:          Number of results to return.
        chunk_type: Optional filter — 'text', 'table', or 'image'.

    Returns:
        List of dicts with keys: content, chunk_type, page_number, section,
        source_file, element_type, image_base64, mime_type, position,
        metadata, similarity (0–1 cosine similarity score).

    The <=> operator is pgvector's cosine distance operator.
    Similarity = 1 − cosine_distance, so 1.0 = identical, 0.0 = orthogonal.
    """
    # Embed the query text in the same vector space as documents and images.
    # Using _embed_texts (not a separate text-only model) ensures the query
    # vector aligns with both text chunks and image chunks in the DB.
    query_vec = _embed_texts([query])[0]
    embedding_str = "[" + ",".join(str(v) for v in query_vec) + "]"

    # Conditionally add a chunk_type filter without SQL injection risk
    # (chunk_type is always passed as a parameterised value, never interpolated)
    type_clause = "AND chunk_type = %(chunk_type)s" if chunk_type else ""

    sql = f"""
        SELECT
            content, chunk_type, page_number, section,
            source_file, element_type, image_path, mime_type,
            position, metadata,
            1 - (embedding <=> %(vec)s::vector) AS similarity
        FROM multimodal_chunks
        WHERE 1=1 {type_clause}
        ORDER BY embedding <=> %(vec)s::vector
        LIMIT %(k)s
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"vec": embedding_str, "chunk_type": chunk_type, "k": k})
            rows = cur.fetchall()

    # Read image from filesystem and re-encode as base64 for callers.
    results = []
    for row in rows:
        row = dict(row)
        img_path = row.pop("image_path", None)
        if img_path and os.path.exists(img_path):
            row["image_base64"] = base64.b64encode(
                pathlib.Path(img_path).read_bytes()
            ).decode()
        else:
            row["image_base64"] = None
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Chunk listing (for preview / debugging)
# ---------------------------------------------------------------------------

def get_all_chunks(chunk_type: str | None = None, limit: int = 200) -> list[dict]:
    """Return all stored chunks, optionally filtered by type.

    Args:
        chunk_type: Optional filter — 'text', 'table', or 'image'.
        limit:      Max rows to return (default 200, safety cap).

    Returns:
        List of dicts with keys: id, content, chunk_type, page_number,
        section, source_file, element_type, image_base64, mime_type,
        position, metadata.
    """
    type_clause = "WHERE chunk_type = %(chunk_type)s" if chunk_type else ""

    sql = f"""
        SELECT
            id, content, chunk_type, page_number, section,
            source_file, element_type, image_path, mime_type,
            position, metadata
        FROM multimodal_chunks
        {type_clause}
        ORDER BY page_number ASC NULLS LAST, id ASC
        LIMIT %(limit)s
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"chunk_type": chunk_type, "limit": limit})
            rows = cur.fetchall()

    results = []
    for row in rows:
        row = dict(row)
        img_path = row.pop("image_path", None)
        if img_path and os.path.exists(img_path):
            row["image_base64"] = base64.b64encode(
                pathlib.Path(img_path).read_bytes()
            ).decode()
        else:
            row["image_base64"] = None
        results.append(row)

    return results

    return results