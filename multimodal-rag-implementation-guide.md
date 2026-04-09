# Multimodal RAG System — Implementation Guide

> A hands-on course project guide for building a Multimodal Retrieval-Augmented Generation system using Docling, Google Gemini, PGVector, LangChain, and FastAPI.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Environment Setup](#3-environment-setup)
4. [5.1 — Working with Complex Documents](#51--working-with-complex-documents)
5. [5.2 — Multimodal Embeddings](#52--multimodal-embeddings)
6. [5.3 — Building Multimodal RAG Pipeline](#53--building-multimodal-rag-pipeline)
7. [5.4 — FastAPI with Multimodal Responses](#54--fastapi-with-multimodal-responses)

---

## 1. Project Overview

This project implements a **Multimodal RAG (Retrieval-Augmented Generation)** system that can:

- Parse complex documents (PDFs with tables, images, and mixed layouts) using **Docling**
- Generate multimodal embeddings using **Google's embedding models**
- Store and retrieve vectors using **PGVector** (PostgreSQL)
- Answer user queries using **Gemini Pro Vision** with text and visual context
- Expose the pipeline via a **FastAPI** REST API with multimodal responses

### Tech Stack

| Component        | Technology                      |
| ---------------- | ------------------------------- |
| Document Parsing | Docling                         |
| Embeddings       | Google Generative AI Embeddings |
| Vector Store     | PGVector (PostgreSQL)           |
| LLM              | Gemini Pro (via LangChain)      |
| Orchestration    | LangChain / LangGraph           |
| API Framework    | FastAPI                         |
| Package Manager  | uv                              |

---

## 2. Project Structure

```
multimodal-rag-system1/
├── main.py                          # FastAPI app entry point
├── pyproject.toml                   # Dependencies and project metadata
├── .env.example                     # Environment variable template
├── data/                            # Source documents (PDFs, images)
├── src/
│   ├── core/
│   │   └── db.py                    # PGVector store & embedding config
│   ├── ingestion/
│   │   ├── docling_parser.py        # Docling-based document parser
│   │   └── ingestion.py             # Ingestion pipeline orchestrator
│   ├── query/                       # Query pipeline (retrieval + generation)
│   └── api/
│       └── v1/
│           ├── routes/
│           │   └── query.py         # API route definitions
│           ├── schemas/
│           │   └── query_schema.py  # Pydantic request/response models
│           └── services/
│               └── query_service.py # RAG query logic
```

---

## 3. Environment Setup

### 3.1 Prerequisites

- Python 3.11+
- PostgreSQL with the `pgvector` extension enabled
- A Google Cloud API key with access to Gemini and Embedding models

### 3.2 Install Dependencies

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install packages
uv venv
source .venv/bin/activate
uv sync
```

### 3.3 Configure Environment Variables

Copy `.env.example` to `.env` and fill in real values:

```bash
cp .env.example .env
```

Required variables (defined in `.env.example`):

| Variable                 | Description                                        |
| ------------------------ | -------------------------------------------------- |
| `GOOGLE_API_KEY`         | Google Cloud API key for Gemini & Embeddings       |
| `GOOGLE_EMBEDDING_MODEL` | Embedding model name (e.g. `google-embedding-001`) |
| `GOOGLE_LLM_MODEL`       | LLM model name (e.g. `gemini-3.1-pro-preview`)     |
| `PG_CONNECTION_STRING`   | PostgreSQL connection URI with pgvector            |

### 3.4 Database Setup

```sql
-- Connect to PostgreSQL and create the database
CREATE DATABASE multimodal_rag_db;

-- Enable the pgvector extension
\c multimodal_rag_db
CREATE EXTENSION IF NOT EXISTS vector;
```

### 3.5 Run the Application

```bash
uv run uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`. Verify with:

- `GET /` — Hello World
- `GET /health` — Health check
- `POST /api/v1/query` — RAG query endpoint

---

## 5.1 — Working with Complex Documents

> **Goal:** Parse PDFs containing tables, images, and mixed layouts into structured, indexable content.

### Topics Covered

| Topic                                  | Description                                                             |
| -------------------------------------- | ----------------------------------------------------------------------- |
| PDF parsing with tables and images     | Extract structured content from complex PDF layouts                     |
| Docling for Document Parsing           | Use Docling library to parse documents with high fidelity               |
| Table extraction and structuring       | Convert embedded tables into structured data (markdown/JSON)            |
| Image captioning and OCR               | Extract text from images and generate captions for visual elements      |
| Preserving document structure & layout | Maintain headings, sections, and spatial relationships between elements |

### Implementation: `src/ingestion/docling_parser.py`

This module uses the **Docling** library to parse PDF documents into structured elements.

#### Key Concepts

```python
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
```

**Step-by-step approach:**

1. **Initialize the DocumentConverter** with pipeline options that enable OCR and table structure recognition:

   ```python
   pipeline_options = PdfPipelineOptions()
   pipeline_options.do_ocr = True                 # Enable OCR for scanned content
   pipeline_options.do_table_structure = True      # Enable table structure recognition

   converter = DocumentConverter(
       allowed_formats=["pdf"],
       pipeline_options={"pdf": pipeline_options}
   )
   ```

2. **Convert the document** to get structured output:

   ```python
   result = converter.convert(file_path)
   doc = result.document
   ```

3. **Iterate over document elements** — Docling breaks the document into typed elements:

   ```python
   for element in doc.iterate_items():
       if element.label == "table":
           # Extract table as markdown or structured data
           table_md = element.export_to_markdown()
       elif element.label == "picture":
           # Extract image bytes, generate caption via OCR or model
           image_data = element.get_image()
       else:
           # Text content (headings, paragraphs, lists)
           text = element.text
   ```

4. **Preserve document structure** — Track section hierarchy and page numbers:

   ```python
   # Each element carries metadata about its position
   metadata = {
       "page_number": element.prov[0].page_no if element.prov else None,
       "element_type": element.label,
       "section": current_section_heading,
   }
   ```

#### What to Implement

- A `parse_document(file_path: str)` function that:
  - Accepts a PDF file path
  - Returns a list of parsed chunks, each with `content`, `content_type` (text/table/image), and `metadata`
  - Handles tables by converting them to markdown representation
  - Handles images by extracting them and generating base64 representations
  - Preserves section headings and page numbers in metadata

---

### Implementation: `src/ingestion/ingestion.py`

This module orchestrates the full ingestion pipeline: parse → chunk → embed → store.

#### What to Implement

- A `run_ingestion(file_path: str)` function that:
  1. Calls `docling_parser.parse_document()` to get structured elements
  2. Splits long text elements into smaller chunks (with overlap for context continuity)
  3. Creates `LangChain Document` objects with content and metadata
  4. Stores them in PGVector using the vector store from `src/core/db.py`

```python
from langchain.schema import Document
from src.ingestion.docling_parser import parse_document
from src.core.db import get_vector_store

def run_ingestion(file_path: str, collection_name: str = "multimodal_rag_db"):
    # 1. Parse document using Docling
    parsed_elements = parse_document(file_path)

    # 2. Create LangChain Document objects
    documents = []
    for elem in parsed_elements:
        doc = Document(
            page_content=elem["content"],
            metadata=elem["metadata"]
        )
        documents.append(doc)

    # 3. Store in PGVector
    vector_store = get_vector_store(collection_name)
    vector_store.add_documents(documents)

    return {"status": "success", "chunks_ingested": len(documents)}
```

---

## 5.2 — Multimodal Embeddings

> **Goal:** Generate embeddings that capture both text and visual content in a shared vector space.

### Topics Covered

| Topic                                | Description                                                             |
| ------------------------------------ | ----------------------------------------------------------------------- |
| Google's multimodal embedding models | Use `google-embedding-001` for unified text and image embeddings        |
| Image and text co-embeddings         | Embed images and text into the same vector space for cross-modal search |
| Cross-modal retrieval                | Retrieve images using text queries and vice versa                       |
| Combining text and visual context    | Merge text and image embeddings for richer document representations     |

### Implementation: `src/core/db.py` (already partially implemented)

The existing `db.py` configures Google's embedding model and PGVector store:

```python
# Current implementation
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL"),   # google-embedding-001
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
```

#### Key Concepts

1. **Text Embeddings** — Standard text content is embedded directly:

   ```python
   embeddings = get_embeddings()
   vector = embeddings.embed_query("What was the quarterly revenue?")
   ```

2. **Image Embeddings** — Images can be described (captioned) and then embedded as text, or if the model supports it, embedded directly as image data.

3. **Co-embedding Strategy** — For tables and images in documents:
   - Tables: Convert to markdown text → embed as text
   - Images: Generate a text description/caption → embed the caption
   - This ensures all content types live in the same vector space

4. **Cross-modal Retrieval** — A text query can retrieve image-based content because both share the same embedding space. The similarity search in PGVector works across all content types:

   ```python
   # A text query retrieves the most relevant chunks regardless of original content type
   results = vector_store.similarity_search("revenue breakdown chart", k=5)
   # Results may include text chunks, table chunks, or image-description chunks
   ```

#### Metadata-Enriched Storage

When storing embeddings, attach metadata that identifies the content type so the retrieval layer knows how to present results:

```python
Document(
    page_content="Table showing Q2 FY2024-25 revenue by segment...",
    metadata={
        "content_type": "table",       # text | table | image
        "source_file": "report.pdf",
        "page_number": 5,
        "image_base64": None,          # populated for image elements
    }
)
```

---

## 5.3 — Building Multimodal RAG Pipeline

> **Goal:** Build a retrieval-augmented generation pipeline that combines text and visual context to generate answers.

### Topics Covered

| Topic                                       | Description                                                     |
| ------------------------------------------- | --------------------------------------------------------------- |
| Indexing images and text together           | Store all content types in a unified PGVector index             |
| Retrieval strategies for multimodal content | Similarity search with content-type-aware filtering             |
| Context assembly from mixed sources         | Combine text, table, and image context into a coherent prompt   |
| Gemini Pro Vision for image understanding   | Use Gemini's vision capabilities to interpret images in context |
| Generating responses with visual context    | Produce answers that reference and explain visual elements      |

### Implementation: `src/query/` module

#### Step 1 — Retrieval with Content-Type Awareness

```python
# src/query/retriever.py
from src.core.db import get_vector_store

def retrieve_multimodal(query: str, k: int = 5):
    vector_store = get_vector_store()
    results = vector_store.similarity_search(query, k=k)

    text_chunks = []
    table_chunks = []
    image_chunks = []

    for doc in results:
        content_type = doc.metadata.get("content_type", "text")
        if content_type == "table":
            table_chunks.append(doc)
        elif content_type == "image":
            image_chunks.append(doc)
        else:
            text_chunks.append(doc)

    return {
        "text": text_chunks,
        "tables": table_chunks,
        "images": image_chunks,
        "all": results,
    }
```

#### Step 2 — Context Assembly

Combine retrieved multimodal content into a structured prompt:

```python
# src/query/context_builder.py

def build_multimodal_context(retrieved: dict) -> str:
    sections = []

    if retrieved["text"]:
        text_content = "\n\n".join([doc.page_content for doc in retrieved["text"]])
        sections.append(f"### Text Context\n{text_content}")

    if retrieved["tables"]:
        table_content = "\n\n".join([doc.page_content for doc in retrieved["tables"]])
        sections.append(f"### Table Data\n{table_content}")

    if retrieved["images"]:
        image_content = "\n\n".join([doc.page_content for doc in retrieved["images"]])
        sections.append(f"### Image Descriptions\n{image_content}")

    return "\n\n---\n\n".join(sections)
```

#### Step 3 — Gemini Pro Vision for Image Understanding

When retrieved chunks contain image data, use Gemini's multimodal capabilities:

```python
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.messages import HumanMessage

def query_with_vision(query: str, context: str, image_base64: str = None):
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    content_parts = [
        {"type": "text", "text": (
            "You are a helpful RAG assistant with vision capabilities. "
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )}
    ]

    # If image data is available, include it in the prompt
    if image_base64:
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
        })

    message = HumanMessage(content=content_parts)
    response = llm.invoke([message])
    return response.content
```

#### Step 4 — LangGraph Orchestration (Optional Advanced)

Use LangGraph to orchestrate the RAG pipeline as a stateful graph:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class RAGState(TypedDict):
    query: str
    retrieved_docs: dict
    context: str
    answer: str
    sources: List[dict]

def retrieve_node(state: RAGState) -> RAGState:
    state["retrieved_docs"] = retrieve_multimodal(state["query"])
    return state

def context_node(state: RAGState) -> RAGState:
    state["context"] = build_multimodal_context(state["retrieved_docs"])
    return state

def generate_node(state: RAGState) -> RAGState:
    state["answer"] = query_with_vision(state["query"], state["context"])
    state["sources"] = [doc.metadata for doc in state["retrieved_docs"]["all"]]
    return state

# Build graph
graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("build_context", context_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "build_context")
graph.add_edge("build_context", "generate")
graph.add_edge("generate", END)

rag_pipeline = graph.compile()
```

### Updating the Query Service

Update `src/api/v1/services/query_service.py` to use the multimodal pipeline:

```python
def query_documents(query: str) -> dict:
    result = rag_pipeline.invoke({"query": query})
    return {
        "answer": result["answer"],
        "sources": result["sources"],
    }
```

---

## 5.4 — FastAPI with Multimodal Responses

> **Goal:** Build API endpoints that handle file uploads, process documents asynchronously, and return mixed-content responses.

### Topics Covered

| Topic                                | Description                                                              |
| ------------------------------------ | ------------------------------------------------------------------------ |
| Handling file uploads (PDFs, images) | Accept PDF and image uploads via FastAPI `UploadFile`                    |
| Async file processing                | Process uploaded files asynchronously using `BackgroundTasks`            |
| Returning mixed content responses    | Return JSON responses containing text, tables, and base64-encoded images |
| Base64 encoding for images           | Encode extracted images as base64 strings for API transport              |
| Streaming multimodal responses       | Stream long responses using FastAPI `StreamingResponse`                  |

### Step 1 — File Upload Endpoint

Add an ingestion endpoint to accept document uploads:

```python
# src/api/v1/routes/query.py — add to existing router
from fastapi import UploadFile, File, BackgroundTasks
import shutil, os

UPLOAD_DIR = "data/uploads"

@router.post("/ingest")
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    # Validate file type
    if not file.filename.endswith((".pdf", ".png", ".jpg", ".jpeg")):
        return {"error": "Unsupported file type. Upload PDF or image files."}

    # Save uploaded file
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process in background
    background_tasks.add_task(run_ingestion, file_path)

    return {
        "message": "File uploaded and ingestion started",
        "filename": file.filename,
    }
```

### Step 2 — Multimodal Response Schema

Extend the response schema to include images and different content types:

```python
# src/api/v1/schemas/query_schema.py — extended
from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")

class ContentBlock(BaseModel):
    type: str = Field(..., description="Content type: text, table, or image")
    content: str = Field(..., description="Text content or base64-encoded image")
    caption: Optional[str] = Field(None, description="Caption for images/tables")

class QueryResponse(BaseModel):
    answer: str
    content_blocks: Optional[List[ContentBlock]] = None
    sources: List[dict]
```

### Step 3 — Returning Mixed Content

Build responses that include text, table markdown, and base64-encoded images:

```python
import base64

def build_multimodal_response(answer: str, retrieved_docs: dict) -> dict:
    content_blocks = []

    for doc in retrieved_docs["all"]:
        content_type = doc.metadata.get("content_type", "text")

        if content_type == "image" and doc.metadata.get("image_base64"):
            content_blocks.append({
                "type": "image",
                "content": doc.metadata["image_base64"],
                "caption": doc.page_content,
            })
        elif content_type == "table":
            content_blocks.append({
                "type": "table",
                "content": doc.page_content,
                "caption": f"Table from page {doc.metadata.get('page_number', '?')}",
            })

    return {
        "answer": answer,
        "content_blocks": content_blocks if content_blocks else None,
        "sources": [doc.metadata for doc in retrieved_docs["all"]],
    }
```

### Step 4 — Streaming Multimodal Responses

For long or complex responses, use FastAPI's `StreamingResponse`:

```python
from fastapi.responses import StreamingResponse
import json

@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    async def generate():
        # Retrieve documents
        retrieved = retrieve_multimodal(request.query)
        context = build_multimodal_context(retrieved)

        # Stream chunks from LLM
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_LLM_MODEL"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            streaming=True,
        )

        prompt = f"Context:\n{context}\n\nQuestion: {request.query}"

        async for chunk in llm.astream(prompt):
            yield json.dumps({"type": "text", "content": chunk.content}) + "\n"

        # After text, send any image blocks
        for doc in retrieved.get("images", []):
            if doc.metadata.get("image_base64"):
                yield json.dumps({
                    "type": "image",
                    "content": doc.metadata["image_base64"],
                    "caption": doc.page_content,
                }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")
```

---

## Quick Reference — Implementation Checklist

| #   | Task                                         | File                                   | Status |
| --- | -------------------------------------------- | -------------------------------------- | ------ |
| 1   | Docling PDF parser with table/image support  | `src/ingestion/docling_parser.py`      | ☐      |
| 2   | Ingestion pipeline (parse → embed → store)   | `src/ingestion/ingestion.py`           | ☐      |
| 3   | Google multimodal embedding configuration    | `src/core/db.py`                       | ✅     |
| 4   | Multimodal retriever                         | `src/query/retriever.py`               | ☐      |
| 5   | Context builder for mixed content            | `src/query/context_builder.py`         | ☐      |
| 6   | Gemini Vision query function                 | `src/query/vision_query.py`            | ☐      |
| 7   | LangGraph RAG orchestration                  | `src/query/rag_pipeline.py`            | ☐      |
| 8   | File upload endpoint                         | `src/api/v1/routes/query.py`           | ☐      |
| 9   | Multimodal response schema                   | `src/api/v1/schemas/query_schema.py`   | ☐      |
| 10  | Streaming response endpoint                  | `src/api/v1/routes/query.py`           | ☐      |
| 11  | Update query service for multimodal pipeline | `src/api/v1/services/query_service.py` | ☐      |

---

## Running & Testing

```bash
# Start the server
uv run uvicorn main:app --reload

# Test health check
curl http://localhost:8000/health

# Ingest a document
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@data/RIL-Media-Release-RIL-Q2-FY2024-25-Financial-and-Operational-Performance.pdf"

# Query the RAG system
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the revenue for Q2 FY2024-25?"}'

# Stream a response
curl -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the financial performance"}'
```

# Fixes

Header logo is present in vector db - 6 times

how to fix? 1. Positional zone filtering (primary fix) 2. Recurring-image deduplication (secondary fix)

Footer texts are not present in vector db
