# Multimodal RAG System — Complete Implementation Guide

> **For trainees:** This guide walks you through building the entire Multimodal RAG system from scratch, file by file, line by line. Follow every section in order. Each section explains _what_ to build, _why_ it is needed, and _exactly how_ to write the code.

---

## Table of Contents

1. [What is a Multimodal RAG System?](#1-what-is-a-multimodal-rag-system)
2. [Architecture Overview](#2-architecture-overview)
3. [Prerequisites](#3-prerequisites)
4. [Project Setup](#4-project-setup)
5. [Database Setup — PostgreSQL + pgvector](#5-database-setup--postgresql--pgvector)
6. [File 1 — `schema.sql`](#6-file-1--schemasql)
7. [File 2 — `.env` Configuration](#7-file-2--env-configuration)
8. [File 3 — `src/ingestion/docling_parser.py`](#8-file-3--srcinestiondocling_parserpy)
9. [File 4 — `src/ingestion/ingestion.py`](#9-file-4--srcingestioningestionpy)
10. [File 5 — `src/core/db.py`](#10-file-5--srccoredbpy)
11. [File 6 — `src/api/v1/schemas/query_schema.py`](#11-file-6--srcapiv1schemasquery_schemapy)
12. [File 7 — `src/api/v1/services/query_service.py`](#12-file-7--srcapiv1servicesquery_servicepy)
13. [File 8 — `src/api/v1/routes/query.py`](#13-file-8--srcapiv1routesquerypy)
14. [File 9 — `main.py`](#14-file-9--mainpy)
15. [Running the System End-to-End](#15-running-the-system-end-to-end)
16. [Testing with curl](#16-testing-with-curl)
17. [How Data Flows Through the System](#17-how-data-flows-through-the-system)
18. [Common Errors and Fixes](#18-common-errors-and-fixes)

---

## 1. What is a Multimodal RAG System?

### Standard RAG (text only)

RAG stands for **Retrieval-Augmented Generation**. The idea is simple:

1. You have a large set of documents.
2. When a user asks a question, instead of relying only on an LLM's training data, you **retrieve** the most relevant pieces from your documents.
3. You **augment** the question with those retrieved pieces.
4. You **generate** the answer using the LLM, which now has document-specific context.

### Multimodal RAG

A real-world PDF like a financial report or a product manual contains:

- Text paragraphs
- Data tables (revenue breakdowns, comparison charts)
- Images (charts, graphs, product photos, logos)

A text-only RAG system would miss the information locked inside tables and images. A **Multimodal RAG** system parses all three content types, stores them in a shared vector database, and sends the relevant content (including actual image pixels) to a Vision-capable LLM like Gemini Pro Vision to generate the answer.

### What this project does

```
PDF File
   │
   ▼
┌─────────────────────────────┐
│  Docling Parser             │  → Extracts text, tables, images
│  (docling_parser.py)        │
└──────────────┬──────────────┘
               │ list of chunks: {content, content_type, metadata}
               ▼
┌─────────────────────────────┐
│  Ingestion Pipeline         │  → Splits long text, embeds all chunks,
│  (ingestion.py)             │    saves images to disk, stores in Postgres
└──────────────┬──────────────┘
               │ embeddings stored in PGVector
               ▼
┌─────────────────────────────┐
│  PostgreSQL + pgvector      │  → Vector database (multimodal_chunks table)
│  (db.py)                    │
└──────────────┬──────────────┘
               │ cosine similarity search
               ▼
┌─────────────────────────────┐
│  Query Service              │  → Retrieves top-k chunks, builds Gemini
│  (query_service.py)         │    prompt with text + images, returns answer
└──────────────┬──────────────┘
               │ answer + sources
               ▼
┌─────────────────────────────┐
│  FastAPI REST API           │  → POST /api/v1/query
│  (main.py, routes/)         │
└─────────────────────────────┘
```

---

## 2. Architecture Overview

### Tech Stack

| Component        | Technology                            | Why                                            |
| ---------------- | ------------------------------------- | ---------------------------------------------- |
| Document Parsing | Docling                               | Handles complex PDFs — tables, images, OCR     |
| Embeddings       | Google `gemini-embedding-001`         | 1536-dim, high quality, supports batching      |
| Vector Store     | PostgreSQL + pgvector                 | Production-grade, supports HNSW indexing       |
| LLM              | Gemini Pro (`gemini-3.1-pro-preview`) | Multimodal — reads both text and images        |
| API Framework    | FastAPI                               | Fast, async, auto-generates docs at `/docs`    |
| Package Manager  | `uv`                                  | Faster than pip, generates lockfiles           |
| DB Driver        | psycopg3 + psycopg_pool               | Modern Postgres driver with connection pooling |

### Project Folder Structure

```
multimodal-rag-system1/
├── main.py                        ← FastAPI app entry point
├── schema.sql                     ← Database schema (run once to set up DB)
├── pyproject.toml                 ← Dependencies
├── .env.example                   ← Template — copy to .env
├── .env                           ← Your actual secrets (never commit this)
├── data/
│   ├── *.pdf                      ← Source PDFs to ingest
│   └── images/                    ← Extracted images saved here during ingestion
├── references/
│   └── multimodal-rag-implementation-guide.md  ← This file
└── src/
    ├── core/
    │   └── db.py                  ← DB connection, embeddings, store/search
    ├── ingestion/
    │   ├── docling_parser.py      ← PDF → structured chunks
    │   └── ingestion.py           ← Orchestrates parse → embed → store
    └── api/
        └── v1/
            ├── routes/
            │   └── query.py       ← POST /api/v1/query endpoint
            ├── schemas/
            │   └── query_schema.py ← Pydantic request/response models
            └── services/
                └── query_service.py ← RAG logic: retrieve + build prompt + call LLM
```

---

## 3. Prerequisites

Before writing any code, you need these installed on your machine:

### 3.1 Python 3.11+

```bash
python3 --version   # should show 3.11 or higher
```

### 3.2 `uv` Package Manager

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify
uv --version
```

### 3.3 PostgreSQL

On macOS with Homebrew:

```bash
brew install postgresql@16
brew services start postgresql@16
```

On Ubuntu/Debian:

```bash
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql
```

### 3.4 `pgvector` Extension

pgvector adds vector similarity search to PostgreSQL.

On macOS:

```bash
brew install pgvector
```

On Ubuntu (after installing postgres):

```bash
sudo apt install postgresql-16-pgvector
```

### 3.5 Google API Key

1. Go to [https://aistudio.google.com](https://aistudio.google.com)
2. Click **Get API Key** → **Create API key**
3. Copy the key — you will put it in `.env`

---

## 4. Project Setup

### 4.1 Clone/Create the Project

The project folder already exists. Navigate into it:

```bash
cd multimodal-rag-system1
```

### 4.2 Create the Virtual Environment and Install Dependencies

```bash
# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# Install all dependencies from pyproject.toml
uv sync
```

The `pyproject.toml` lists all dependencies:

```toml
[project]
name = "multimodal-rag-system1"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.135.2",        # Web framework
    "uvicorn>=0.42.0",         # ASGI server to run FastAPI
    "python-dotenv>=1.2.2",    # Load .env files
    "docling>=2.85.0",         # PDF parsing
    "langchain-postgres>=0.0.17",  # LangChain + pgvector integration
    "langchain-google-genai>=4.2.1",  # Gemini via LangChain
    "langgraph>=1.1.6",        # Stateful pipeline orchestration
    "langchain>=1.2.15",       # Core LangChain
    "python-multipart>=0.0.24",  # File upload support for FastAPI
]
```

---

## 5. Database Setup — PostgreSQL + pgvector

### 5.1 Create the Database

Open a terminal and connect to PostgreSQL as the superuser:

```bash
psql -U postgres
```

Inside the psql shell:

```sql
-- Create a dedicated database for this project
CREATE DATABASE multimodal_rag_db;

-- Create a user (optional, for cleaner access control)
CREATE USER rag_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE multimodal_rag_db TO rag_user;

-- Exit psql
\q
```

### 5.2 Enable the pgvector Extension

Connect to the new database:

```bash
psql -U postgres -d multimodal_rag_db
```

```sql
-- This must be run once. It enables the VECTOR data type and similarity operators.
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### 5.3 Run the Schema

Exit psql and run `schema.sql` (we write this file in the next section):

```bash
psql -U postgres -d multimodal_rag_db -f schema.sql
```

---

## 6. File 1 — `schema.sql`

**Location:** `schema.sql` (project root)

**Purpose:** Defines the two database tables the system needs. Run this once before ingesting any documents.

```sql
-- =============================================================================
-- Enable pgvector — adds the VECTOR data type and <=> cosine distance operator
-- =============================================================================
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- documents table
-- Every PDF you ingest gets one row here.
-- The UUID becomes a foreign key that links all chunks extracted from that PDF.
-- ON CONFLICT (filename) means re-ingesting the same file updates the row
-- instead of creating a duplicate — this is called an "upsert".
-- =============================================================================
CREATE TABLE IF NOT EXISTS documents (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    filename    TEXT        UNIQUE NOT NULL,
    source_path TEXT        NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =============================================================================
-- multimodal_chunks table
-- Every piece of content extracted from a document becomes one row here.
-- chunk_type tells us whether this was a text paragraph, a table, or an image.
--
-- image_path: Instead of storing raw image bytes in Postgres (wasteful), we
--   save images as PNG files in data/images/ and only keep the file path here.
--
-- embedding: A 1536-dimensional float vector. The <=> operator on this column
--   lets pgvector find the most similar chunks to a query vector.
--
-- metadata: A JSONB catch-all for any extra fields not worth a dedicated column.
-- =============================================================================
CREATE TABLE IF NOT EXISTS multimodal_chunks (
    id           BIGSERIAL    PRIMARY KEY,
    doc_id       UUID         NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    chunk_type   TEXT         NOT NULL CHECK (chunk_type IN ('text', 'table', 'image')),
    element_type TEXT,

    content      TEXT         NOT NULL,
    image_path   TEXT,
    mime_type    TEXT,

    page_number  INT,
    section      TEXT,
    source_file  TEXT,
    position     JSONB,

    embedding    VECTOR(1536),
    metadata     JSONB
);

-- =============================================================================
-- Indexes — these make searches fast
-- =============================================================================

-- HNSW (Hierarchical Navigable Small World) index for approximate nearest
-- neighbour search on the embedding column. "vector_cosine_ops" means we
-- use cosine similarity as the distance metric.
-- Without this index, every query does a full table scan — extremely slow.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON multimodal_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id    ON multimodal_chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type ON multimodal_chunks (chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_page_number ON multimodal_chunks (page_number);
```

**Run it:**

```bash
psql -U postgres -d multimodal_rag_db -f schema.sql
```

---

## 7. File 2 — `.env` Configuration

**Location:** `.env` (project root — created from `.env.example`)

**Purpose:** Stores secrets and configuration. Never commit this file to Git.

```bash
# Copy the template
cp .env.example .env
```

Open `.env` and fill in real values:

```dotenv
# Your Google AI Studio API key
GOOGLE_API_KEY=AIzaSy...your-key-here...

# Embedding model — 1536 dimensions, works for text
GOOGLE_EMBEDDING_MODEL=gemini-embedding-001

# LLM model — supports vision (can read images)
GOOGLE_LLM_MODEL=gemini-3.1-pro-preview

# PostgreSQL connection string
# Format: postgresql+psycopg://username:password@host:port/database
PG_CONNECTION_STRING=postgresql+psycopg://postgres:yourpassword@localhost:5432/multimodal_rag_db
```

> **Why `postgresql+psycopg://` prefix?**  
> LangChain uses SQLAlchemy to parse connection strings, and SQLAlchemy needs the dialect prefix to know which driver to use. When we connect directly with psycopg3, we strip this prefix. You will see this in `db.py`.

---

## 8. File 3 — `src/ingestion/docling_parser.py`

**Purpose:** Takes a PDF file path and returns a flat list of typed chunks — one chunk per meaningful piece of content (heading, paragraph, table, image).

**Why Docling?** Docling is a document parsing library by IBM that does:

- Layout analysis (understands where columns, headers, footers are)
- Table structure recognition (reconstructs rows and columns)
- OCR (extracts text from scanned pages)
- Image extraction (renders chart/figure areas as PNG images)

### Complete Code

```python
import base64
import io
import os

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def parse_document(file_path: str) -> list[dict]:
    """Parse a PDF into a flat list of typed content chunks using Docling.

    Returns a list of dicts. Each dict has:
      content      — the text content of this chunk (or a caption for images)
      content_type — "text", "table", or "image"
      metadata     — dict with page_number, section, element_type,
                     source_file, position, image_base64
    """

    # ── Step 1: Configure the Docling pipeline ───────────────────────────────
    # These three options are critical:
    #   do_ocr=True              → extract text even from scanned/rasterised pages
    #   do_table_structure=True  → reconstruct table rows and columns
    #   generate_picture_images  → render picture elements to PIL Image objects
    #                              so we can extract them as PNG bytes
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        generate_picture_images=True,
    )

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
    )

    # ── Step 2: Convert the PDF ───────────────────────────────────────────────
    # converter.convert() runs the full pipeline:
    #   layout analysis → OCR → table structure → picture rendering
    # result.document is a DoclingDocument — a tree of typed elements.
    result = converter.convert(file_path)
    doc = result.document

    parsed_chunks: list[dict] = []

    # current_section tracks the most recently seen heading.
    # Every chunk we produce will carry this section name in its metadata,
    # so at query time we can tell the user "this came from the Revenue section".
    current_section: str | None = None
    source_file = os.path.basename(file_path)

    # ── Step 3: Walk the element tree ────────────────────────────────────────
    # doc.iterate_items() yields (level, node) tuples.
    # level = heading depth (1 = top-level section, 2 = subsection, etc.)
    # node = the actual DocItem object with .label, .text, .prov, etc.
    for item in doc.iterate_items():
        if isinstance(item, tuple):
            _, node = item  # discard level, keep node
        else:
            node = item     # older Docling versions yield bare nodes

        # node.label is a DocItemLabel enum — we convert to lowercase string
        # so we can do simple string matching ("table" in label, etc.)
        label = str(getattr(node, "label", "")).lower()

        # ── Skip running headers and footers ─────────────────────────────────
        # page_header = the document title or company name that appears at the
        #               top of every page (NOISE — not useful for retrieval)
        # page_footer = page number, date stamp at the bottom (also NOISE)
        if label in ("page_header", "page_footer"):
            continue

        # ── Extract position on the page ─────────────────────────────────────
        # "prov" (provenance) carries the bounding box (l, t, r, b) which
        # tells us where on the page this element was located.
        # Useful for filtering (e.g., skip images in the top 12% of a page
        # which are likely logos). Stored as JSONB for optional use.
        prov = getattr(node, "prov", None)
        page_no = prov[0].page_no if prov else None
        position: dict | None = None
        if prov and hasattr(prov[0], "bbox") and prov[0].bbox is not None:
            b = prov[0].bbox
            position = {"l": b.l, "t": b.t, "r": b.r, "b": b.b}

        def _make_metadata(content_type: str, element_type: str, img_b64=None):
            """Helper: create the metadata dict that travels with every chunk."""
            return {
                "content_type": content_type,
                "element_type": element_type,
                "section": current_section,
                "page_number": page_no,
                "source_file": source_file,
                "position": position,
                "image_base64": img_b64,
            }

        # ── CASE 1: Section headings and document title ───────────────────────
        # We update current_section so all subsequent chunks inherit this label.
        # We also store the heading itself as a text chunk so it is searchable.
        if "section_header" in label or label == "title":
            text = getattr(node, "text", "").strip()
            if text:
                current_section = text
                parsed_chunks.append({
                    "content": text,
                    "content_type": "text",
                    "metadata": _make_metadata("text", label),
                })

        # ── CASE 2: Tables ────────────────────────────────────────────────────
        # Strategy: Convert table cells to readable plain text.
        # "Col1: Val1  |  Col2: Val2" format keeps column context with values,
        # making the embedding more meaningful for retrieval.
        #
        # We try three approaches in order:
        #   1. export_to_dataframe() — best quality, uses Docling's cell grid
        #   2. export_to_html() stripped of tags — fallback
        #   3. node.text — last resort
        elif "table" in label:
            table_text = ""

            if hasattr(node, "export_to_dataframe"):
                try:
                    df = node.export_to_dataframe()
                    if df is not None and not df.empty:
                        rows_text: list[str] = []
                        headers = [str(c).strip() for c in df.columns]
                        for _, row in df.iterrows():
                            pairs = [
                                f"{h}: {str(v).strip()}"
                                for h, v in zip(headers, row)
                                if str(v).strip() not in ("", "nan", "None")
                            ]
                            if pairs:
                                rows_text.append("  |  ".join(pairs))
                        table_text = "\n".join(rows_text)
                except Exception:
                    pass

            if not table_text and hasattr(node, "export_to_html"):
                try:
                    import re as _re
                    raw_html = node.export_to_html(doc)
                    table_text = _re.sub(r"<[^>]+>", " ", raw_html or "")
                    table_text = _re.sub(r"\s+", " ", table_text).strip()
                except Exception:
                    pass

            if not table_text:
                table_text = getattr(node, "text", "")

            if table_text and table_text.strip():
                parsed_chunks.append({
                    "content": table_text.strip(),
                    "content_type": "table",
                    "metadata": _make_metadata("table", "table"),
                })

        # ── CASE 3: Pictures, figures, charts ─────────────────────────────────
        # Images are stored as base64-encoded PNG strings in metadata.
        # The "content" field holds the caption (or a placeholder if no caption
        # exists) — this is what gets embedded as a vector.
        #
        # Note (Issue 11): Using caption text as the embedding is a limitation.
        # Short-term fix: generate richer descriptions using Gemini Vision.
        # Long-term fix: use gemini-embedding-2-preview for true image embeddings.
        elif "picture" in label or "figure" in label or label == "chart":
            img_b64 = None
            caption = getattr(node, "text", "") or ""

            try:
                if hasattr(node, "get_image"):
                    pil_img = node.get_image(doc)
                    if pil_img:
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        img_b64 = base64.b64encode(buf.getvalue()).decode()

                # Fallback for older Docling versions
                if img_b64 is None and hasattr(node, "image") and node.image:
                    pil_img = getattr(node.image, "pil_image", None)
                    if pil_img:
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        img_b64 = base64.b64encode(buf.getvalue()).decode()
            except Exception:
                pass  # image extraction is best-effort; caption still gets indexed

            content = caption.strip() or f"[Image on page {page_no}]"
            parsed_chunks.append({
                "content": content,
                "content_type": "image",
                "metadata": _make_metadata("image", "picture", img_b64),
            })

        # ── CASE 4: Everything else (paragraphs, list items, footnotes) ───────
        else:
            text = getattr(node, "text", "")
            if text and text.strip():
                parsed_chunks.append({
                    "content": text.strip(),
                    "content_type": "text",
                    "metadata": _make_metadata("text", label),
                })

    return parsed_chunks
```

### What Docling Labels Mean

| Label                | What it is           | How we handle it                                   |
| -------------------- | -------------------- | -------------------------------------------------- |
| `title`              | Document title       | Store as text, set as `current_section`            |
| `section_header`     | Section heading      | Store as text, update `current_section`            |
| `text` / `paragraph` | Body paragraph       | Store as text chunk                                |
| `list_item`          | Bullet/numbered item | Store as text chunk                                |
| `caption`            | Figure/table caption | Store as text chunk                                |
| `footnote`           | Page footnote        | Store as text chunk                                |
| `table`              | Data table           | Convert to "Col: Val" format, store as table chunk |
| `picture` / `chart`  | Image or chart       | Extract PNG bytes → base64, store as image chunk   |
| `page_header`        | Running page header  | **SKIP** — noise                                   |
| `page_footer`        | Running page footer  | **SKIP** — noise                                   |

---

## 9. File 4 — `src/ingestion/ingestion.py`

**Purpose:** Orchestrates the full ingestion pipeline for one PDF:

1. Register the document in the `documents` table
2. Parse with Docling
3. Split long text into overlapping chunks
4. Embed and store in `multimodal_chunks`

### Why chunk splitting?

The embedding model has a context window limit (~2000 tokens). A long financial paragraph might be 3000 characters. We split it into overlapping windows so:

- No chunk exceeds the model's limit
- Sentences that fall at a boundary appear in both the preceding and following chunk, ensuring they can be found by either half

### Complete Code

```python
import os
import pathlib
import sys

from dotenv import load_dotenv

from src.core.db import store_chunks, upsert_document
from src.ingestion.docling_parser import parse_document

load_dotenv()

# ── Chunking configuration ────────────────────────────────────────────────────
# _TEXT_CHUNK_SIZE    = max characters per chunk
# _TEXT_CHUNK_OVERLAP = characters shared between adjacent chunks
# Only text is split — tables and images are always stored as atomic units
_TEXT_CHUNK_SIZE = 1500
_TEXT_CHUNK_OVERLAP = 300


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a long string into overlapping character windows.

    Example with chunk_size=10, overlap=3, text="ABCDEFGHIJKLM":
      Window 1: ABCDEFGHIJ   (chars 0–9)
      Window 2: HIJKLM       (chars 7–12, overlap of 3 = HI + J already seen)

    This ensures words at boundaries are not cut off without context.
    """
    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap          # advance by (chunk_size - overlap) each time
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step
    return chunks


def run_ingestion(file_path: str) -> dict:
    """Run the full ingestion pipeline for a single PDF file.

    Args:
        file_path: Path to the PDF (relative or absolute).

    Returns:
        {"status": "success", "doc_id": "...", "chunks_ingested": N}
    """
    resolved = pathlib.Path(file_path).resolve()

    # ── Step 1: Register document in the database ─────────────────────────────
    # upsert_document() does INSERT ... ON CONFLICT DO UPDATE.
    # If this PDF was ingested before, it returns the same UUID — no duplicate.
    # This UUID is also used to delete old chunks before re-inserting (idempotent).
    doc_id = upsert_document(resolved.name, str(resolved))
    print(f"[ingestion] doc_id={doc_id}  file={file_path}")

    # ── Step 2: Parse the PDF with Docling ────────────────────────────────────
    print(f"[ingestion] Parsing: {file_path}")
    parsed_elements = parse_document(file_path)
    print(f"[ingestion] Docling produced {len(parsed_elements)} elements")

    # ── Step 3: Split long text elements ─────────────────────────────────────
    # Tables and images are never split — you can't cut a table row in half.
    # Only text paragraphs longer than _TEXT_CHUNK_SIZE get windowed.
    chunks: list[dict] = []
    for elem in parsed_elements:
        if elem["content_type"] == "text" and len(elem["content"]) > _TEXT_CHUNK_SIZE:
            for sub in _split_text(elem["content"], _TEXT_CHUNK_SIZE, _TEXT_CHUNK_OVERLAP):
                # Each sub-chunk inherits the parent's metadata
                # (same page number, section, source file)
                chunks.append({
                    "content": sub,
                    "content_type": elem["content_type"],
                    "metadata": elem["metadata"],
                })
        else:
            chunks.append(elem)

    print(f"[ingestion] {len(chunks)} chunks ready for embedding")

    # ── Step 4: Embed and store ───────────────────────────────────────────────
    # store_chunks() in db.py:
    #   - Deletes old chunks for this doc_id (so re-ingestion doesn't duplicate)
    #   - Calls the Google embedding API in batches of 50
    #   - Saves image files to data/images/
    #   - INSERTs each row into multimodal_chunks
    count = store_chunks(chunks, doc_id)
    print(f"[ingestion] Stored {count} chunks → multimodal_chunks")

    return {"status": "success", "doc_id": doc_id, "chunks_ingested": count}


# ── CLI entry point ───────────────────────────────────────────────────────────
# Run:  uv run python -m src.ingestion.ingestion path/to/file.pdf
# If no argument is given, falls back to the default development PDF.
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        pdf_path = pathlib.Path(sys.argv[1])
    else:
        pdf_path = pathlib.Path("data/RIL-Media-Release-RIL-Q2-FY2024-25-mini.pdf")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path.resolve()}")

    result = run_ingestion(str(pdf_path))
    print(f"\nIngestion complete: {result}")
```

---

## 10. File 5 — `src/core/db.py`

**Purpose:** Everything that touches the database lives here:

- Module-level singletons (embeddings model, connection pool)
- `upsert_document()` — register a document
- `store_chunks()` — embed and insert chunks
- `similarity_search()` — find relevant chunks for a query
- `get_all_chunks()` — list chunks for debugging

### Why a connection pool?

Opening a new TCP connection to PostgreSQL takes ~10–50ms and consumes a server process. A **connection pool** keeps a small number of connections open and reuses them across requests. With `min_size=2, max_size=10`, the pool always has at least 2 ready connections and can grow to handle 10 simultaneous requests.

### Why a module-level embeddings singleton?

`GoogleGenerativeAIEmbeddings(...)` creates an HTTP client internally. If we call it inside `store_chunks()` and `similarity_search()`, a new client is created on every function call — wasteful. Creating it once at module import time (`_embeddings_model = ...`) means it is created once and reused.

### Complete Code

```python
import base64
import hashlib
import json
import os
import pathlib

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# ── Connection setup ──────────────────────────────────────────────────────────
# The .env file uses "postgresql+psycopg://" (SQLAlchemy format).
# psycopg3 needs the standard "postgresql://" URI, so we strip the prefix.
_PG_CONNECTION = os.getenv("PG_CONNECTION_STRING", "")
_PG_DSN = _PG_CONNECTION.replace("postgresql+psycopg://", "postgresql://")

_EMBED_BATCH_SIZE = 50  # Google embedding API: up to 100 texts per batch

# ── Module-level singletons ───────────────────────────────────────────────────
# Created once when this module is first imported. Reused for every request.

_embeddings_model = GoogleGenerativeAIEmbeddings(
    model=os.getenv("GOOGLE_EMBEDDING_MODEL"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    output_dimensionality=1536,   # 1536-dim vectors to match our schema
)

# Lazy connection pool — created on first use, not at import time.
# This prevents connection failures during unit tests that don't need a DB.
_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            _PG_DSN,
            min_size=2,
            max_size=10,
            kwargs={"row_factory": dict_row},  # rows come back as dicts, not tuples
        )
    return _pool


def get_db_conn():
    """Return a pooled connection as a context manager.

    Usage:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
    The connection is automatically returned to the pool on exit.
    """
    return _get_pool().connection()


# ── Document registry ─────────────────────────────────────────────────────────

def upsert_document(filename: str, source_path: str) -> str:
    """Insert or update a document record, return its UUID as a string.

    ON CONFLICT (filename) DO UPDATE means: if a document with this filename
    already exists, update its source_path and ingested_at timestamp and
    return the existing id. This makes ingestion idempotent.
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


# ── Chunk storage ─────────────────────────────────────────────────────────────

def store_chunks(chunks: list[dict], doc_id: str) -> int:
    """Embed each chunk and insert it into multimodal_chunks.

    Key behaviours:
    - Deletes all existing chunks for doc_id first (idempotent re-ingestion)
    - Embeds text content in batches of _EMBED_BATCH_SIZE
    - Saves image PNG files to data/images/ (not stored in Postgres BYTEA)
    - Strips redundant fields from JSONB metadata (they have dedicated columns)
    """
    if not chunks:
        return 0

    contents = [c["content"] for c in chunks]

    # ── Batch embed ───────────────────────────────────────────────────────────
    # embed_documents() takes a list of strings and returns a list of vectors.
    # We call it in batches to stay within the API's payload limit.
    all_embeddings: list[list[float]] = []
    for i in range(0, len(contents), _EMBED_BATCH_SIZE):
        batch = contents[i : i + _EMBED_BATCH_SIZE]
        all_embeddings.extend(_embeddings_model.embed_documents(batch))

    # Fields already stored in dedicated columns — exclude from JSONB to avoid
    # data duplication (every saved byte is read back on every query result).
    _DEDICATED_COLUMNS = {
        "content_type", "element_type", "section",
        "page_number", "source_file", "position", "image_base64",
    }

    rows_inserted = 0
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            # Delete stale chunks — this is what makes re-ingestion safe.
            # Without this DELETE, re-ingesting the same PDF would DOUBLE
            # every chunk in the database.
            cur.execute(
                "DELETE FROM multimodal_chunks WHERE doc_id = %s::uuid",
                (doc_id,),
            )

            for chunk, embedding in zip(chunks, all_embeddings):
                meta = chunk["metadata"]

                # ── Image file storage ────────────────────────────────────────
                # Instead of storing raw PNG bytes in a BYTEA column (which
                # would bloat the DB and slow vacuuming), we write the PNG to
                # data/images/ and store only the path.
                #
                # We name the file using: doc_id + first 16 chars of SHA-256
                # of the image bytes. SHA-256 ensures that two identical
                # images (e.g., the same logo on every page) always produce
                # the same filename → only stored once on disk.
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

                # pgvector stores vectors as '[f1,f2,…,f1536]' string literals
                # cast with ::vector. We build this string ourselves to avoid
                # needing the separate pgvector Python package.
                embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

                # Build the lean metadata — no redundant fields
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
                        chunk["content_type"],       # "text", "table", or "image"
                        meta.get("element_type"),    # raw Docling label
                        chunk["content"],            # the text that was embedded
                        image_path,                  # filesystem path or None
                        mime_type,
                        meta.get("page_number"),
                        meta.get("section"),
                        meta.get("source_file"),
                        json.dumps(meta.get("position")) if meta.get("position") else None,
                        embedding_str,
                        json.dumps(clean_meta),
                    ),
                )
                rows_inserted += 1
        conn.commit()

    return rows_inserted


# ── Similarity search ─────────────────────────────────────────────────────────

def similarity_search(
    query: str,
    k: int = 5,
    chunk_type: str | None = None,
) -> list[dict]:
    """Find the k most similar chunks to a natural-language query.

    How it works:
    1. Embed the query string into a 1536-dim vector using the same model
       that was used during ingestion.
    2. Run a SQL query that computes cosine distance between the query vector
       and every stored embedding, then returns the k closest rows.
    3. The <=> operator is pgvector's cosine distance. The HNSW index makes
       this approximate nearest-neighbour search very fast.
    4. Similarity = 1 - cosine_distance
       1.0 = identical vectors, 0.0 = completely unrelated.
    5. For image chunks, read the PNG from disk and re-encode as base64.

    Args:
        query:      The user's question.
        k:          How many chunks to return.
        chunk_type: Optional filter — only return 'text', 'table', or 'image'.
    """
    query_vec = _embeddings_model.embed_query(query)
    embedding_str = "[" + ",".join(str(v) for v in query_vec) + "]"

    # chunk_type is always passed as a SQL parameter (%s), never interpolated
    # into the SQL string — this prevents SQL injection.
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

    # Replace the filesystem image_path with base64 content for the caller.
    # The API and query service work exclusively with base64 — they never
    # need to know where the file lives on disk.
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


# ── Debug helper ──────────────────────────────────────────────────────────────

def get_all_chunks(chunk_type: str | None = None, limit: int = 200) -> list[dict]:
    """Return stored chunks for inspection/debugging.

    Useful for verifying ingestion worked correctly:
        from src.core.db import get_all_chunks
        chunks = get_all_chunks(chunk_type="table")
        for c in chunks[:3]:
            print(c["content"][:200])
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
```

---

## 11. File 6 — `src/api/v1/schemas/query_schema.py`

**Purpose:** Defines the shape of the data that goes **into** and **out of** the API. FastAPI uses these Pydantic models to:

- Validate incoming request bodies automatically
- Generate the OpenAPI (Swagger) documentation at `/docs`
- Serialize responses

```python
from pydantic import BaseModel, Field
from typing import List, Optional


# ── Request ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(
        ...,              # "..." means the field is required
        description="The user's question in natural language"
    )
    k: int = Field(
        5,                # default value
        ge=1,             # minimum value (ge = greater than or equal)
        le=20,            # maximum value (le = less than or equal)
        description="Number of chunks to retrieve from the vector database"
    )
    chunk_type: Optional[str] = Field(
        None,             # default None = no filter, search all types
        description="Filter results to one type: 'text', 'table', or 'image'"
    )


# ── Response ──────────────────────────────────────────────────────────────────
class QueryResponse(BaseModel):
    answer: str            # The generated answer from Gemini
    sources: List[dict]    # List of source chunks used to generate the answer
                           # Each dict contains: chunk_type, page_number,
                           # section, source_file, element_type, similarity
```

---

## 12. File 7 — `src/api/v1/services/query_service.py`

**Purpose:** The heart of the RAG system. This module:

1. Calls `similarity_search()` to retrieve the top-k relevant chunks
2. Builds a multimodal prompt — text blocks and base64 images interleaved
3. Calls the Gemini LLM with the prompt
4. Returns the answer and source metadata

### Why `SystemMessage` vs `HumanMessage`?

The Gemini API supports different message roles:

- `SystemMessage` — Instructions to the model about **how** to behave
- `HumanMessage` — The actual content and question

Sending the system prompt as `SystemMessage` ensures the model treats it as behavioural instructions rather than user input, giving it stronger effect on model behaviour.

### Complete Code

```python
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.db import similarity_search

load_dotenv()

# The system prompt instructs the LLM on its role and constraints.
# "ONLY the provided context" prevents hallucination — the model must
# answer based on retrieved document content, not its training data.
_SYSTEM_PROMPT = (
    "You are a helpful assistant for document question-answering. "
    "Answer the question using ONLY the provided context (text, tables, and images). "
    "If the answer is not present in the context, say you don't know. "
    "When citing information, mention the page number and section."
)

# Module-level LLM singleton — created once, reused for every request.
# Creating a new ChatGoogleGenerativeAI() per request would open a new
# HTTP client each time, adding unnecessary latency.
_llm = ChatGoogleGenerativeAI(
    model=os.getenv("GOOGLE_LLM_MODEL"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


def query_documents(query: str, k: int = 5, chunk_type: str | None = None) -> dict:
    """Run the full RAG pipeline for a user query.

    Args:
        query:      The user's natural-language question.
        k:          Number of chunks to retrieve.
        chunk_type: Optional — restrict retrieval to 'text', 'table', or 'image'.

    Returns:
        {"answer": "...", "sources": [...]}
    """

    # ── Step 1: Vector similarity search ─────────────────────────────────────
    # similarity_search() embeds the query, runs the <=> cosine search in
    # PostgreSQL, and returns the k most relevant chunks.
    chunks = similarity_search(query, k=k, chunk_type=chunk_type)

    # ── Step 2: Build the multimodal prompt ───────────────────────────────────
    # Gemini's multimodal API accepts a list of "content parts".
    # Each part is either {"type": "text", "text": "..."} or
    # {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    #
    # We accumulate text chunks into a single text block (more efficient than
    # many small text parts). When we encounter an image chunk, we flush the
    # accumulated text as a text part first, then add the image part.
    # This ordering matters — Gemini processes content parts sequentially.
    message_parts: list[dict] = []
    sources: list[dict] = []
    text_blocks: list[str] = []

    for chunk in chunks:
        ctype = chunk["chunk_type"]
        page = chunk.get("page_number")
        section = chunk.get("section") or "—"

        # Build the sources list for the API response.
        # Trainees: similarity scores close to 1.0 mean the chunk was very
        # relevant; close to 0.0 means it was loosely related.
        sources.append({
            "chunk_type": ctype,
            "page_number": page,
            "section": section,
            "source_file": chunk.get("source_file", ""),
            "element_type": chunk.get("element_type"),
            "similarity": round(chunk.get("similarity", 0), 4),
        })

        if ctype in ("text", "table"):
            label = "TABLE" if ctype == "table" else "TEXT"
            # Include the chunk type, page, and section in the label so the
            # LLM can cite its sources accurately in the answer.
            text_blocks.append(
                f"[{label} | page {page} | {section}]\n{chunk['content']}"
            )

        elif ctype == "image" and chunk.get("image_base64"):
            # Flush accumulated text before inserting an image part.
            # Gemini reads content parts in sequence; mixing text and images
            # freely works, but flushing text first keeps it readable.
            if text_blocks:
                message_parts.append({
                    "type": "text",
                    "text": "\n\n".join(text_blocks),
                })
                text_blocks = []

            # Send the image as a data URI embedded in the message.
            # Gemini Vision reads the base64 PNG bytes directly.
            message_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{chunk['image_base64']}"
                },
            })

    # Flush any remaining text blocks after the loop.
    if text_blocks:
        message_parts.append({
            "type": "text",
            "text": "\n\n".join(text_blocks),
        })

    # Append the question at the very end of the prompt.
    message_parts.append({
        "type": "text",
        "text": f"\n\nQuestion: {query}",
    })

    # ── Step 3: Invoke Gemini ─────────────────────────────────────────────────
    # SystemMessage = behavioural instructions (sent in the "system" role)
    # HumanMessage  = the actual content + question (sent in the "user" role)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=message_parts),
    ]

    response = _llm.invoke(messages)

    # response.content is usually a plain string, but thinking/reasoning models
    # (like gemini-3.1-pro-preview) return a list of content parts.
    # We handle both cases to keep the code robust.
    content = response.content
    if isinstance(content, list):
        answer = " ".join(
            part["text"] for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    else:
        answer = content

    return {
        "answer": answer,
        "sources": sources,
    }
```

---

## 13. File 8 — `src/api/v1/routes/query.py`

**Purpose:** Defines the FastAPI route that handles `POST /api/v1/query` requests.

```python
from fastapi import APIRouter
from src.api.v1.schemas.query_schema import QueryRequest, QueryResponse
from src.api.v1.services.query_service import query_documents

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """POST /api/v1/query

    Request body (JSON):
      {
        "query": "What was the revenue for Q2 FY2024-25?",
        "k": 5,
        "chunk_type": null          // optional: "text", "table", or "image"
      }

    Response body (JSON):
      {
        "answer": "The revenue was ...",
        "sources": [
          {"chunk_type": "table", "page_number": 4, "section": "Revenue", ...},
          ...
        ]
      }

    FastAPI automatically:
    - Validates request JSON against QueryRequest
    - Returns 422 Unprocessable Entity if validation fails
    - Serializes the return value against QueryResponse
    - Documents this endpoint at http://localhost:8000/docs
    """
    result = query_documents(
        request.query,
        k=request.k,
        chunk_type=request.chunk_type,
    )
    return QueryResponse(**result)
```

---

## 14. File 9 — `main.py`

**Purpose:** Creates the FastAPI application and registers all routers.

```python
from fastapi import FastAPI
from src.api.v1.routes.query import router as query_router

# Create the FastAPI app instance.
# The title appears in the auto-generated docs at http://localhost:8000/docs
app = FastAPI(title="Multimodal RAG API")


@app.get("/")
def read_root():
    """GET / — Basic health indicator."""
    return {"message": "Multimodal RAG API is running."}


@app.get("/health")
def health_check():
    """GET /health — Used by load balancers and monitoring tools."""
    return {"status": "ok"}


# Register the query router under /api/v1
# All routes defined in query.py will be accessible at /api/v1/<route>
app.include_router(query_router, prefix="/api/v1")
```

---

## 15. Running the System End-to-End

Follow these steps in order every time you start working.

### Step 1 — Start PostgreSQL

```bash
# macOS
brew services start postgresql@16

# Ubuntu
sudo systemctl start postgresql
```

### Step 2 — Set Up the Database (first time only)

```bash
psql -U postgres -c "CREATE DATABASE multimodal_rag_db;"
psql -U postgres -d multimodal_rag_db -f schema.sql
```

### Step 3 — Activate the Virtual Environment

```bash
source .venv/bin/activate
```

### Step 4 — Verify the `.env` File

Make sure these are filled in:

```bash
cat .env
```

You should see real values (not placeholders) for `GOOGLE_API_KEY` and `PG_CONNECTION_STRING`.

### Step 5 — Ingest a PDF

```bash
# Ingest the sample PDF
uv run python -m src.ingestion.ingestion data/RIL-Media-Release-RIL-Q2-FY2024-25-mini.pdf
```

Expected output:

```
[ingestion] doc_id=3f2a1c...  file=data/RIL-Media-Release-...pdf
[ingestion] Parsing: data/RIL-Media-Release-...pdf
[ingestion] Docling produced 142 elements
[ingestion] 187 chunks ready for embedding
[ingestion] Stored 187 chunks → multimodal_chunks

Ingestion complete: {'status': 'success', 'doc_id': '3f2a1c...', 'chunks_ingested': 187}
```

> **Note:** First run downloads Docling's ML models (~1 GB). Subsequent runs are fast.

### Step 6 — Verify Ingestion in the Database

```bash
psql -U postgres -d multimodal_rag_db -c "
SELECT chunk_type, COUNT(*) as count
FROM multimodal_chunks
GROUP BY chunk_type
ORDER BY count DESC;
"
```

Expected output:

```
 chunk_type | count
------------+-------
 text       |   145
 table      |    28
 image      |    14
```

### Step 7 — Start the API Server

```bash
uv run uvicorn main:app --reload
```

The `--reload` flag restarts the server automatically whenever you edit a Python file — very useful during development.

Expected output:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Application startup complete.
```

---

## 16. Testing with curl

### Health Check

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### Query the System

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was the revenue for Q2 FY2024-25?",
    "k": 5
  }'
```

### Query Only Tables

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me a revenue breakdown by segment",
    "k": 3,
    "chunk_type": "table"
  }'
```

### Query with Images

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Describe any charts showing financial performance",
    "k": 3,
    "chunk_type": "image"
  }'
```

### Using the Auto-Generated Docs (easier for beginners)

Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser. FastAPI shows you a Swagger UI where you can run queries interactively.

---

## 17. How Data Flows Through the System

### Ingestion Flow (offline — run once)

```
PDF file on disk
      │
      ▼
parse_document()         ← docling_parser.py
  Docling: layout analysis → OCR → table structure → picture rendering
  Output: list of {content, content_type, metadata}
  Examples:
    {"content": "Financial Performance Overview", "content_type": "text", ...}
    {"content": "Revenue: 2.3T  |  EBITDA: 450B", "content_type": "table", ...}
    {"content": "[Image on page 4]", "content_type": "image", metadata includes image_base64}
      │
      ▼
_split_text()            ← ingestion.py
  Long text paragraphs → overlapping 1500-char windows
  Tables and images → kept as-is
      │
      ▼
upsert_document()        ← db.py
  INSERT INTO documents (filename, source_path) ON CONFLICT DO UPDATE
  Returns doc_id (UUID)
      │
      ▼
store_chunks()           ← db.py
  1. DELETE FROM multimodal_chunks WHERE doc_id = ?   (clean slate)
  2. embed_documents(all_text_contents)               (batched API calls)
  3. For images: decode base64 → write PNG to data/images/
  4. INSERT INTO multimodal_chunks (..., embedding, image_path, ...)
```

### Query Flow (online — every API request)

```
POST /api/v1/query {"query": "...", "k": 5}
      │
      ▼
query_endpoint()         ← routes/query.py
  Validates request with Pydantic
      │
      ▼
query_documents()        ← query_service.py

  Step 1: similarity_search(query, k=5)     ← db.py
    embed_query(query)   → 1536-dim vector
    SELECT ... FROM multimodal_chunks
    ORDER BY embedding <=> query_vector
    LIMIT 5
    For image rows: read PNG from disk → base64
    Returns list of 5 most relevant chunks

  Step 2: Build multimodal prompt
    Text/table chunks → accumulated text blocks
    Image chunks     → data URI image parts
    Interleave text and image parts in message_parts list

  Step 3: Invoke Gemini
    messages = [
      SystemMessage("You are a helpful assistant...")
      HumanMessage([text_block, image_url, text_block, question])
    ]
    _llm.invoke(messages) → response

  Returns {"answer": "...", "sources": [...]}
      │
      ▼
QueryResponse serialized to JSON
POST response → {"answer": "...", "sources": [...]}
```

---

## 18. Common Errors and Fixes

### `ModuleNotFoundError: No module named 'src'`

You must run Python from the project root, not from inside a subdirectory:

```bash
# WRONG
cd src/ingestion
python ingestion.py

# CORRECT
cd multimodal-rag-system1
uv run python -m src.ingestion.ingestion
```

---

### `connection refused` or `psycopg.OperationalError`

PostgreSQL is not running or the connection string is wrong.

```bash
# Check if PostgreSQL is running
pg_isready

# Start it
brew services start postgresql@16   # macOS
sudo systemctl start postgresql     # Linux
```

Check your `.env`:

```
PG_CONNECTION_STRING=postgresql+psycopg://postgres:yourpassword@localhost:5432/multimodal_rag_db
```

---

### `google.api_core.exceptions.InvalidArgument: API key not valid`

The `GOOGLE_API_KEY` in your `.env` is incorrect or uses placeholder text.

---

### `relation "multimodal_chunks" does not exist`

You forgot to run `schema.sql`:

```bash
psql -U postgres -d multimodal_rag_db -f schema.sql
```

---

### `KeyError: 'image_base64'` or images not appearing in responses

The image PNG file referenced by `image_path` was deleted or the `data/images/` directory is missing. Re-ingest the PDF to regenerate:

```bash
uv run python -m src.ingestion.ingestion data/your-file.pdf
```

---

### Docling downloads ML models on first run (slow)

Expected behaviour. Docling downloads:

- EasyOCR models for text recognition
- Table structure detection models
- Layout analysis models

Total ~1 GB. This only happens once. Subsequent runs use cached models.

---

### Empty `answer` from Gemini

Usually means no relevant chunks were found. Check:

1. Was the document actually ingested?
   ```bash
   psql -U postgres -d multimodal_rag_db \
     -c "SELECT COUNT(*) FROM multimodal_chunks;"
   ```
2. Is your query too different in wording from the document content? Try rephrasing.
3. Increase `k` to retrieve more chunks.

---

## Quick Reference — All Commands

```bash
# Setup (run once)
uv venv && source .venv/bin/activate && uv sync
psql -U postgres -c "CREATE DATABASE multimodal_rag_db;"
psql -U postgres -d multimodal_rag_db -f schema.sql

# Ingest a PDF
uv run python -m src.ingestion.ingestion data/your-file.pdf

# Start the API
uv run uvicorn main:app --reload

# Query via curl
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "your question here", "k": 5}'

# Check stored chunks
psql -U postgres -d multimodal_rag_db \
  -c "SELECT chunk_type, COUNT(*) FROM multimodal_chunks GROUP BY chunk_type;"

# Browse the API docs in your browser
open http://localhost:8000/docs
```
