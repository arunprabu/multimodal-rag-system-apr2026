# Multimodal RAG System — Improvement Plans

> Identified issues across ingestion, retrieval, API, and infrastructure layers with recommended fixes.

---

## Issue 1 — Header Logo Stored 6× in Vector DB (Noise Pollution)

**Where:** `src/ingestion/docling_parser.py`  
**Status:** ❌ Not implemented — documented only

**Problem:**  
The company logo appears on every page as an embedded `picture` element. Docling assigns it the label `picture`, NOT `page_header`, so the `page_header` skip guard does not catch it. As a result, the same logo image gets ingested once per page, flooding the vector DB with near-identical image chunks that rank high on visual similarity queries.

**How to fix:**

1. **Positional zone filtering (primary fix)**  
   Check the normalised bounding-box `top` coordinate from `prov[0].bbox`. Anything with `t < 0.12` (top 12 % of a page) is almost always a logo or running header. Skip it during parsing.

   ```python
   # In docling_parser.py, before the picture branch:
   if position and position.get("t", 1) < 0.12:
       continue  # skip header-zone images
   ```

2. **Recurring-image deduplication (secondary fix)**  
   Compute a perceptual hash (e.g., `imagehash.phash`) or an MD5 of the raw PNG bytes for each image. At ingestion time, maintain a `seen_hashes` set per document. If a hash has already been seen, skip storing that chunk.

   ```python
   import hashlib
   seen_image_hashes: set[str] = set()

   # Inside the picture branch:
   img_hash = hashlib.md5(base64.b64decode(img_b64)).hexdigest()
   if img_hash in seen_image_hashes:
       continue
   seen_image_hashes.add(img_hash)
   ```

---

## Issue 2 — Footer Text Not in Vector DB

**Where:** `src/ingestion/docling_parser.py`  
**Status:** ❌ Not implemented — documented only

**Problem:**  
`page_footer` label is explicitly skipped as noise. This is correct for running page numbers and date stamps, BUT Docling also labels some genuine informational footnotes (risk disclosures, data-source attributions) as `page_footer` rather than `footnote`. This content is permanently lost.

**How to fix:**

1. **Separate footnotes from page footers**  
   Only skip `page_footer`. Ensure `footnote` falls through to the plain-text handler (it already does, but verify):

   ```python
   if label in ("page_header", "page_footer"):
       continue  # skip running headers/footers only
   ```

2. **Add a `footnote` branch (optional)**  
   Explicitly handle `footnote` labels with distinct metadata so they can be retrieved separately or excluded from answers if desired.

---

## Issue 3 — `iterate_items()` Tuple Unpacking is Reversed (Critical Bug)

**Where:** `src/ingestion/docling_parser.py` (line ~68)  
**Status:** ✅ Fixed — confirmed actual tuple order is `(node, level)` by runtime inspection; `node, _ = item` is the correct unpack for this Docling version

**Problem:**  
The comment says `iterate_items()` yields `(level, node)` tuples but the actual unpack is `node, _ = item`, which assigns `level` (an integer) to `node` and the real `DocItem` object to `_` (discarded). Every subsequent `getattr(node, "label", "")` call on an integer silently returns `""`, meaning almost no elements are ever processed — the parser produces empty or near-empty output.

`_test_docling.py` correctly does `level, node = item`, confirming the real order.

**How to fix:**

```python
# WRONG (current):
node, _ = item  # unpack (node, level)

# CORRECT:
_, node = item  # iterate_items() yields (level, node); discard level
```

---

## Issue 4 — Re-Ingesting a Document Duplicates All Chunks

**Where:** `src/ingestion/ingestion.py`, `src/core/db.py`  
**Status:** ✅ Fixed — `store_chunks()` now runs `DELETE FROM multimodal_chunks WHERE doc_id = %s::uuid` before inserting

**Problem:**  
`upsert_document()` correctly re-uses the same `doc_id` on conflict, but `store_chunks()` performs plain `INSERT` without deleting the previous chunks first. Re-ingesting the same PDF doubles (or triples, etc.) every chunk in `multimodal_chunks`, causing duplicate retrieval results and wasted storage.

**How to fix:**

Add a delete step in `store_chunks()` (or in `run_ingestion()`) before inserting new chunks:

```python
# In store_chunks(), before the insert loop:
with conn.cursor() as cur:
    cur.execute(
        "DELETE FROM multimodal_chunks WHERE doc_id = %s::uuid",
        (doc_id,)
    )
```

Or add `ON DELETE CASCADE` on the `doc_id` foreign key and delete the document record first (letting the cascade clean up chunks).

---

## Issue 5 — No Ingestion API Endpoint (Unused `python-multipart`)

**Where:** `src/api/v1/routes/`, `pyproject.toml`  
**Status:** ❌ Not implemented — documented only

**Problem:**  
`python-multipart` is listed as a dependency (needed for FastAPI file uploads), but no upload/ingest endpoint exists. The only way to ingest a document is to run the CLI script manually, which is not deployable.

**How to fix:**

Add a `/api/v1/ingest or /api/v1/admin/upload` endpoint:

```python
# src/api/v1/routes/ingest.py
from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile, shutil
from src.ingestion.ingestion import run_ingestion

router = APIRouter()

@router.post("/ingest")
def ingest_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        result = run_ingestion(tmp.name)
    return result
```

---

## Issue 6 — Debug `print` Statements Left in Production Code

**Where:** `src/api/v1/services/query_service.py` (lines ~78–80)  
**Status:** ❌ Not implemented — `print("************")` blocks still present in `query_service.py`

**Problem:**

```python
print("************");
print(text_blocks)
print("************");
```

These debug prints dump potentially large text blocks containing document content to stdout on every API request. This leaks data in logs and degrades performance.

**How to fix:**

Remove them entirely, or replace with a properly levelled logger:

```python
import logging
logger = logging.getLogger(__name__)
logger.debug("text_blocks: %s", text_blocks)
```

---

## Issue 7 — System Prompt Sent as `HumanMessage` Instead of `SystemMessage`

**Where:** `src/api/v1/services/query_service.py`  
**Status:** ✅ Fixed — `SystemMessage(content=_SYSTEM_PROMPT)` used instead of `HumanMessage`

**Problem:**

```python
messages = [
    HumanMessage(content=[{"type": "text", "text": _SYSTEM_PROMPT}]),
    HumanMessage(content=message_parts),
]
```

The system prompt is sent as a user turn. Gemini models treat `SystemMessage` differently from `HumanMessage` — system instructions have stronger effect on model behaviour when sent in the system role.

**How to fix:**

```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content=_SYSTEM_PROMPT),
    HumanMessage(content=message_parts),
]
```

---

## Issue 8 — LLM and Embeddings Model Re-Instantiated on Every Request

**Where:** `src/api/v1/services/query_service.py`, `src/core/db.py`  
**Status:** ✅ Fixed — `_llm` and `_embeddings_model` are module-level singletons in `query_service.py` and `db.py`

**Problem:**  
`ChatGoogleGenerativeAI(...)` is created inside `query_documents()` and `GoogleGenerativeAIEmbeddings(...)` is created inside `get_embeddings()`, which is called from both `store_chunks()` and `similarity_search()`. A new client object (with its own HTTP session) is created for every API call, wasting resources and adding latency.

**How to fix:**

Instantiate both as module-level singletons:

```python
# db.py — module level
_embeddings_model = GoogleGenerativeAIEmbeddings(
    model=os.getenv("GOOGLE_EMBEDDING_MODEL"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    output_dimensionality=1536,
)

# query_service.py — module level
_llm = ChatGoogleGenerativeAI(
    model=os.getenv("GOOGLE_LLM_MODEL"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)
```

---

## Issue 9 — No Database Connection Pooling

**Where:** `src/core/db.py`  
**Status:** ✅ Fixed — `psycopg_pool.ConnectionPool(min_size=2, max_size=10)` with lazy init via `_get_pool()` in `db.py`

**Problem:**  
`get_db_conn()` opens a fresh `psycopg.connect()` on every call. Under concurrent API traffic, this means a new TCP connection to PostgreSQL per request, exhausting the DB's max connections quickly and adding connection-setup overhead.

**How to fix:**

Use a connection pool (psycopg3's built-in `ConnectionPool`):

```python
from psycopg_pool import ConnectionPool

_pool = ConnectionPool(_PG_DSN, min_size=2, max_size=10)

def get_db_conn():
    return _pool.connection()  # context manager, auto-returns on exit
```

Or use SQLAlchemy's connection pool which is already set up if LangChain's PGVector integration is used directly.

---

## Issue 10 — Redundant Data in JSONB `metadata` Column

**Where:** `src/core/db.py` — `store_chunks()`  
**Status:** ✅ Fixed — `_DEDICATED_COLUMNS` set strips redundant fields before JSONB insert in `db.py`

**Problem:**  
The `clean_meta` dict stored in the `metadata` JSONB column contains `page_number`, `section`, `source_file`, `element_type`, `content_type`, and `position` — all of which are already stored in dedicated columns. Every row duplicates these values, wasting roughly 200–400 bytes per chunk.

**How to fix:**

Strip all fields that have dedicated columns before building `clean_meta`:

```python
_DEDICATED_COLUMNS = {"content_type", "element_type", "section",
                      "page_number", "source_file", "position", "image_base64"}

clean_meta = {k: v for k, v in meta.items() if k not in _DEDICATED_COLUMNS}
```

---

## Issue 11 — Images Embedded by Caption Text, Not Visual Content

**Where:** `src/ingestion/docling_parser.py`, `src/core/db.py`  
**Status:** ✅ Fixed (long-term) — `db.py` now uses `gemini-embedding-2-preview` via google-genai SDK for all embeddings. Image chunks are embedded from raw PNG bytes via `_embed_image()`; text/table chunks use `_embed_texts()`. Both share the same vector space, enabling true cross-modal retrieval.

**Problem:**  
Image chunks use their caption text (or a placeholder like `"[Image on page 3]"`) as the `content` that gets embedded. The visual embedding is just a text embedding of the caption, not a multimodal embedding of the image pixels. A query like _"show me the revenue bar chart"_ only matches if those exact words appear in the caption — the image's visual content is never searched.

**How to fix:**

1. **Short term:** Use Gemini Vision to auto-generate a rich description of each image at ingestion time and embed that description instead.

   ```python
   # In docling_parser.py, after extracting img_b64:
   description = generate_image_description(img_b64)  # call Gemini Vision
   content = description or caption or f"[Image on page {page_no}]"
   ```

2. **Long term:** Use `gemini-embedding-2-preview`, Google's multimodal embedding model, which embeds both text and image bytes directly into the same vector space. This enables true visual similarity search without requiring captions.

   ```python
   # In db.py store_chunks(), for image chunks:
   from google import genai
   from google.genai import types as genai_types
   client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
   result = client.models.embed_content(
       model="gemini-embedding-2-preview",
       contents=genai_types.Content(
           parts=[genai_types.Part.from_bytes(
               data=base64.b64decode(img_b64), mime_type="image/png"
           )]
       ),
   )
   image_embedding = result.embeddings[0].values
   ```

---

## Issue 12 — Hardcoded PDF Path in Ingestion Script

**Where:** `src/ingestion/ingestion.py` (`__main__` block)  
**Status:** ✅ Fixed — `sys.argv[1]` used for PDF path; falls back to sample file if no arg given

**Problem:**

```python
pdf_path = pathlib.Path("data/RIL-Media-Release-RIL-Q2-FY2024-25-mini.pdf")
```

The script only works for one hardcoded document. There is no way to ingest different files without editing the source code.

**How to fix:**

Accept the path as a command-line argument:

```python
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.ingestion <path/to/file.pdf>")
        sys.exit(1)
    pdf_path = pathlib.Path(sys.argv[1])
    ...
```

---

## Issue 13 — No Database Schema / Migration File

**Where:** Project root (missing)  
**Status:** ✅ Fixed — `schema.sql` created at project root with full table definitions, HNSW index, and B-tree indexes

**Problem:**  
There is no `schema.sql` or migration tool configuration. A new developer cannot set up the PostgreSQL database without reverse-engineering the `INSERT` statements in `db.py` to infer the table structure.

**How to fix:**

Add a `schema.sql` at the project root:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename    TEXT UNIQUE NOT NULL,
    source_path TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS multimodal_chunks (
    id           BIGSERIAL PRIMARY KEY,
    doc_id       UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_type   TEXT NOT NULL,          -- 'text' | 'table' | 'image'
    element_type TEXT,
    content      TEXT NOT NULL,
    image_bytes  BYTEA,
    mime_type    TEXT,
    page_number  INT,
    section      TEXT,
    source_file  TEXT,
    position     JSONB,
    embedding    VECTOR(1536),
    metadata     JSONB
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON multimodal_chunks USING hnsw (embedding vector_cosine_ops);
```

---

## Issue 14 — No Error Handling in API Route

**Where:** `src/api/v1/routes/query.py`  
**Status:** ❌ Not implemented — documented only

**Problem:**  
`query_endpoint` has no try/except. Any DB connection failure, embedding API timeout, or LLM error surfaces as an unhandled exception, which FastAPI converts to a 500 with a full Python traceback exposed in the response body — leaking internal details.

**How to fix:**

```python
from fastapi import APIRouter, HTTPException
import logging

logger = logging.getLogger(__name__)

@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    try:
        result = query_documents(request.query, k=request.k)
        return QueryResponse(**result)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Query processing failed")
```

---

## Issue 15 — Incorrect / Non-Existent Model Name in `.env.example`

**Where:** `.env.example`  
**Status:** ⚠️ Left unchanged — `gemini-3.1-pro-preview` is a valid model name; no fix required

**Problem:**

```
GOOGLE_LLM_MODEL=gemini-3.1-pro-preview
```

`gemini-3.1-pro-preview` is not a real Google model identifier. The correct names are `gemini-1.5-pro`, `gemini-2.0-flash`, or `gemini-2.5-pro-preview-03-25`. Using a wrong model name causes an API error at runtime.

**How to fix:**

Update `.env.example` to a valid model name:

```
GOOGLE_LLM_MODEL=gemini-2.0-flash
```

---

## Issue 16 — `_test_docling.py` Is in Project Root

**Where:** `_test_docling.py`  
**Status:** ❌ Not implemented — documented only

**Problem:**  
The exploration/test script lives at the project root, mixed in with application entry points. It also references a hardcoded path (`data/RIL-Media-Release-...Financial-and-Operational-Performance.pdf`) that does not exist in the `data/` directory.

**How to fix:**

Move it to a `tests/` or `scripts/` directory and fix the file path reference.

---

## Issue 17 — No `chunk_type` Filter Exposed in Query API

**Where:** `src/api/v1/schemas/query_schema.py`, `src/api/v1/routes/query.py`  
**Status:** ✅ Fixed — `chunk_type` field added to `QueryRequest`, passed through route → `query_documents()` → `similarity_search()`

**Problem:**  
`similarity_search()` accepts an optional `chunk_type` filter, but the API endpoint doesn't expose it. Users cannot restrict retrieval to only text, only tables, or only images — every query mixes all content types regardless of the question's nature.

**How to fix:**

Add an optional field to `QueryRequest`:

```python
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    k: int = Field(5, ge=1, le=20)
    chunk_type: str | None = Field(None, description="Filter: 'text', 'table', or 'image'")
```

Pass it through in `query_documents()` and `similarity_search()`.

---

## Issue 18 — Large Binary Images Stored Directly in PostgreSQL (Scalability)

**Where:** `src/core/db.py` — `store_chunks()`  
**Status:** ✅ Fixed — images saved to `data/images/{doc_id}_{sha256[:16]}.png`; `image_path TEXT` stored in DB instead of `BYTEA`

**Problem:**  
Raw PNG image bytes are stored in the `image_bytes BYTEA` column of PostgreSQL. For a 100-page financial PDF with many charts, this can easily add 50–200 MB to the database for a single document. PostgreSQL BYTEA is not designed for bulk binary object storage and causes table bloat, slower vacuuming, and excessive memory use during queries.

**How to fix:**

Store images in an object store (local filesystem, S3, or MinIO) and keep only the object key/URL in the DB:

```python
# Store image, return path
image_path = save_image_to_object_store(image_bytes, doc_id, chunk_index)

# In INSERT:
INSERT INTO multimodal_chunks (..., image_path, ...) VALUES (..., %s, ...)
```

---

## Summary Table

| #   | Issue                                      | Severity     | Status   | File(s)                              |
| --- | ------------------------------------------ | ------------ | -------- | ------------------------------------ |
| 1   | Header logo ingested 6×                    | High         | ❌ Open  | `docling_parser.py`                  |
| 2   | Footer text not in DB                      | Medium       | ❌ Open  | `docling_parser.py`                  |
| 3   | `iterate_items()` tuple unpacking reversed | **Critical** | ✅ Fixed | `docling_parser.py`                  |
| 4   | Re-ingestion duplicates all chunks         | High         | ✅ Fixed | `ingestion.py`, `db.py`              |
| 5   | No ingestion API endpoint                  | High         | ❌ Open  | `routes/`                            |
| 6   | Debug `print` in production code           | Medium       | ❌ Open  | `query_service.py`                   |
| 7   | System prompt sent as `HumanMessage`       | Medium       | ✅ Fixed | `query_service.py`                   |
| 8   | LLM/Embeddings re-instantiated per request | Medium       | ✅ Fixed | `db.py`, `query_service.py`          |
| 9   | No DB connection pooling                   | Medium       | ✅ Fixed | `db.py`                              |
| 10  | Redundant data in JSONB metadata column    | Low          | ✅ Fixed | `db.py`                              |
| 11  | Images embedded by caption text only       | High         | ✅ Fixed | `db.py`                              |
| 12  | Hardcoded PDF path in ingestion script     | Medium       | ✅ Fixed | `ingestion.py`                       |
| 13  | No DB schema / migration file              | High         | ✅ Fixed | `schema.sql`                         |
| 14  | No error handling in API route             | Medium       | ❌ Open  | `routes/query.py`                    |
| 15  | Wrong model name in `.env.example`         | Medium       | ⚠️ N/A   | `.env.example`                       |
| 16  | `_test_docling.py` in project root         | Low          | ❌ Open  | `_test_docling.py`                   |
| 17  | `chunk_type` filter not exposed in API     | Low          | ✅ Fixed | `query_schema.py`, `routes/query.py` |
| 18  | Large images in PostgreSQL BYTEA           | Medium       | ✅ Fixed | `db.py`                              |
