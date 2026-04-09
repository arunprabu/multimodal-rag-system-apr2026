# Current Implementation vs. Required Features

> Status of each multimodal RAG feature against the actual codebase as of April 2026.

---

## Feature Checklist

### 1. Image Captioning and OCR

| Feature                       | Status             | Details                                                                                                                                                            |
| ----------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| OCR on scanned PDFs           | ✅ Implemented     | `PdfPipelineOptions(do_ocr=True)` in `docling_parser.py` — Docling runs OCR via its built-in pipeline                                                              |
| Image captioning at ingestion | ❌ Not implemented | Captions from Docling's `caption` label are used as-is; no Gemini Vision call to generate rich descriptions. See Issue 11 short-term fix in `improvement-plans.md` |
| Table structure extraction    | ✅ Implemented     | `do_table_structure=True` + `export_to_dataframe()` / `export_to_html()` fallback in `docling_parser.py`                                                           |

---

### 2. Google's Multimodal Embedding Models

| Feature                                              | Status             | Details                                                                                                                                                                                                                                 |
| ---------------------------------------------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Text embeddings                                      | ✅ Implemented     | `GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", output_dimensionality=1536)` in `db.py`                                                                                                                                     |
| Multimodal embeddings (`gemini-embedding-2-preview`) | ❌ Not implemented | Images are embedded using text captions only. The `gemini-embedding-2-preview` model (which embeds image bytes directly) is documented as a code comment in `query_service.py` but not wired in. See Issue 11 in `improvement-plans.md` |

---

### 3. Image and Text Co-Embeddings

| Feature                               | Status         | Details                                                                                                                                                                                                          |
| ------------------------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Text chunks embedded                  | ✅ Implemented | All `text` and `table` chunks embedded via `_embeddings_model.embed_documents()` in `db.py`                                                                                                                      |
| Image chunks embedded                 | ⚠️ Partial     | Image chunks **are** embedded and stored in `VECTOR(1536)`, but the embedding is of the caption text (e.g. `"[Image on page 3]"`), not the image pixels. True co-embedding requires `gemini-embedding-2-preview` |
| Same vector space for text and images | ⚠️ Partial     | Both types share the same `embedding` column and HNSW index, enabling joint retrieval. However, visual similarity is not captured — only lexical similarity of captions                                          |

---

### 4. Cross-Modal Retrieval

| Feature                       | Status         | Details                                                                                            |
| ----------------------------- | -------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| Retrieve text by text query   | ✅ Implemented | `similarity_search()` in `db.py` uses `<=>` cosine distance on `embedding` column                  |
| Retrieve images by text query | ⚠️ Partial     | Works only if the query words appear in the image caption. Visual content is not searchable        |
| Retrieve tables by text query | ✅ Implemented | Table rows serialised as `"Col: val                                                                | Col: val"` text — well-embedded and retrievable |
| Filter by chunk type          | ✅ Implemented | `chunk_type` parameter in `QueryRequest` → `similarity_search()` adds `AND chunk_type = %s` clause |

---

### 5. Combining Text and Visual Context

| Feature                                    | Status         | Details                                                                                     |
| ------------------------------------------ | -------------- | ------------------------------------------------------------------------------------------- |
| Mixed context assembly at query time       | ✅ Implemented | `query_service.py` interleaves text blocks and base64 image `data:` URIs in `message_parts` |
| Images passed to LLM as inline data URIs   | ✅ Implemented | `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` sent to Gemini   |
| Text blocks flushed before each image part | ✅ Implemented | `if text_blocks: message_parts.append(...)` pattern ensures correct ordering in the prompt  |

---

### 6. Indexing Images and Text Together

| Feature                              | Status             | Details                                                                                                                                                                       |
| ------------------------------------ | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Unified `multimodal_chunks` table    | ✅ Implemented     | Single table stores `chunk_type = 'text' \| 'table' \| 'image'` with shared `embedding VECTOR(1536)` column                                                                   |
| HNSW index on embeddings             | ✅ Implemented     | `CREATE INDEX idx_chunks_embedding USING hnsw (embedding vector_cosine_ops)` in `schema.sql`                                                                                  |
| Image filesystem storage             | ✅ Implemented     | PNG files stored to `data/images/{doc_id}_{sha256[:16]}.png`; `image_path TEXT` in DB (Issue 18 fix)                                                                          |
| SHA-256 image deduplication          | ✅ Implemented     | Identical images (e.g. logo appearing on every page) hash to the same filename on disk                                                                                        |
| Header logo deduplication (per-page) | ❌ Not implemented | SHA-256 dedup on disk prevents duplicate files but the **chunks** still get inserted once per occurrence. Positional zone filter (`t < 0.12`) is not yet applied. See Issue 1 |

---

### 7. Retrieval Strategies for Multimodal Content

| Feature                             | Status             | Details                                                                      |
| ----------------------------------- | ------------------ | ---------------------------------------------------------------------------- |
| Cosine similarity search (pgvector) | ✅ Implemented     | `ORDER BY embedding <=> %s::vector LIMIT %s` in `similarity_search()`        |
| `chunk_type` filter                 | ✅ Implemented     | Optional filter for `'text'`, `'table'`, or `'image'` exposed via API        |
| Hybrid search (keyword + vector)    | ❌ Not implemented | No full-text search (`tsvector`) or BM25 combined with vector search         |
| Reranking                           | ❌ Not implemented | Retrieved chunks used as-is; no cross-encoder or Gemini-based reranking step |
| MMR (Maximal Marginal Relevance)    | ❌ Not implemented | No diversity-based reranking to avoid redundant results                      |

---

### 8. Context Assembly from Mixed Sources

| Feature                        | Status             | Details                                                                                                                        |
| ------------------------------ | ------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| Text + table context blocks    | ✅ Implemented     | Accumulated in `text_blocks` list with `[TEXT \| page N \| section]` prefix                                                    |
| Image context inlined          | ✅ Implemented     | Read back from filesystem, re-encoded as base64, inlined in `message_parts`                                                    |
| Source attribution in response | ✅ Implemented     | `sources` list in `QueryResponse` includes `chunk_type`, `page_number`, `section`, `source_file`, `element_type`, `similarity` |
| Streaming assembled context    | ❌ Not implemented | Context is assembled in memory and sent as one batch; no streaming                                                             |

---

### 9. Gemini Pro Vision for Image Understanding

| Feature                                    | Status             | Details                                                                                             |
| ------------------------------------------ | ------------------ | --------------------------------------------------------------------------------------------------- |
| Images sent to Gemini at query time        | ✅ Implemented     | All retrieved image chunks with `image_base64` are included in the `HumanMessage` multimodal prompt |
| Gemini Vision answers with image context   | ✅ Implemented     | `_llm.invoke(messages)` with `ChatGoogleGenerativeAI(model=GOOGLE_LLM_MODEL)` handles vision input  |
| Gemini Vision captioning at ingestion time | ❌ Not implemented | Not called during `parse_document()`. Documented as Issue 11 short-term fix                         |

---

### 10. Generating Responses with Visual Context

| Feature                            | Status         | Details                                                                                           |
| ---------------------------------- | -------------- | ------------------------------------------------------------------------------------------------- |
| LLM response generation            | ✅ Implemented | `_llm.invoke(messages)` returns Gemini answer; handles list-type content for thinking models      |
| System prompt for grounded answers | ✅ Implemented | `SystemMessage(content=_SYSTEM_PROMPT)` instructs model to answer from context only (Issue 7 fix) |
| Response includes answer + sources | ✅ Implemented | `QueryResponse(answer=..., sources=[...])` returned from API                                      |

---

### 11. Async File Processing

| Feature                      | Status             | Details                                                                                                              |
| ---------------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------- |
| Async ingestion API endpoint | ❌ Not implemented | No `/api/v1/ingest` endpoint exists. Ingestion is CLI-only (`python -m src.ingestion.ingestion <file>`). See Issue 5 |
| Background task processing   | ❌ Not implemented | FastAPI `BackgroundTasks` not used; ingestion blocks the caller                                                      |
| Async DB queries             | ❌ Not implemented | Uses synchronous `psycopg3`; no `psycopg.AsyncConnection` or `asyncpg`                                               |

---

### 12. Returning Mixed Content Responses

| Feature                        | Status             | Details                                                                                                                                 |
| ------------------------------ | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| Text answer in response        | ✅ Implemented     | `answer: str` in `QueryResponse`                                                                                                        |
| Source metadata in response    | ✅ Implemented     | `sources: List[dict]` with chunk type, page, section, similarity                                                                        |
| Image bytes / URLs in response | ❌ Not implemented | Retrieved image `image_path` is read and sent to Gemini but NOT included in the API response body. Clients receive only the text answer |

---

### 13. Base64 Encoding for Images

| Feature                               | Status         | Details                                                                                              |
| ------------------------------------- | -------------- | ---------------------------------------------------------------------------------------------------- |
| Images base64-encoded at parse time   | ✅ Implemented | PIL Image → BytesIO → `base64.b64encode()` in `docling_parser.py`                                    |
| Base64 decoded and saved at ingestion | ✅ Implemented | `base64.b64decode(img_b64)` → write bytes to `data/images/` in `db.py`                               |
| Base64 re-encoded for LLM prompt      | ✅ Implemented | PNG file re-read and re-encoded as base64 `data:` URI at query time in `db.py` `similarity_search()` |

---

### 14. Streaming Multimodal Responses

| Feature                     | Status             | Details                                                                  |
| --------------------------- | ------------------ | ------------------------------------------------------------------------ |
| Streaming LLM response      | ❌ Not implemented | `_llm.invoke()` used (blocking, full response). `_llm.stream()` not used |
| FastAPI `StreamingResponse` | ❌ Not implemented | Route returns full `QueryResponse` object; no `StreamingResponse` or SSE |
| Server-Sent Events (SSE)    | ❌ Not implemented | Not implemented                                                          |

---

## Summary

| Category                                | Total Features | ✅ Implemented | ⚠️ Partial | ❌ Not Implemented |
| --------------------------------------- | -------------- | -------------- | ---------- | ------------------ |
| Image Captioning and OCR                | 3              | 2              | 0          | 1                  |
| Google Multimodal Embeddings            | 2              | 1              | 0          | 1                  |
| Image & Text Co-Embeddings              | 3              | 1              | 2          | 0                  |
| Cross-Modal Retrieval                   | 4              | 3              | 1          | 0                  |
| Combining Text & Visual Context         | 3              | 3              | 0          | 0                  |
| Indexing Images & Text Together         | 5              | 4              | 0          | 1                  |
| Retrieval Strategies                    | 5              | 2              | 0          | 3                  |
| Context Assembly                        | 4              | 3              | 0          | 1                  |
| Gemini Vision for Image Understanding   | 3              | 2              | 0          | 1                  |
| Response Generation with Visual Context | 3              | 3              | 0          | 0                  |
| Async File Processing                   | 3              | 0              | 0          | 3                  |
| Returning Mixed Content Responses       | 3              | 2              | 0          | 1                  |
| Base64 Encoding for Images              | 3              | 3              | 0          | 0                  |
| Streaming Multimodal Responses          | 3              | 0              | 0          | 3                  |
| **Total**                               | **47**         | **29**         | **3**      | **15**             |

---

## What Needs to Be Built

The following are the key gaps to make this a complete multimodal RAG system:

1. **Image captioning at ingestion** — Call Gemini Vision (`gemini-2.0-flash`) on each extracted image to generate a rich description. Embed that description instead of the bare caption string. (Issue 11 short-term fix)

2. **True multimodal embeddings** — Replace text-only embeddings for image chunks with `gemini-embedding-2-preview` which accepts image bytes directly. This enables visual similarity search. (Issue 11 long-term fix)

3. **Ingestion API endpoint** — Add `POST /api/v1/ingest` with `UploadFile` so PDFs can be ingested via HTTP rather than CLI only. (Issue 5)

4. **Streaming responses** — Use `_llm.stream()` + FastAPI `StreamingResponse` to stream the Gemini answer token-by-token to the client.

5. **Return images in API response** — Include retrieved image base64 data in `QueryResponse.sources` so clients can display the visual evidence alongside the text answer.

6. **Async ingestion** — Use FastAPI `BackgroundTasks` so the ingest endpoint returns immediately with a job ID while processing the PDF in the background.

7. **Hybrid search** — Add `tsvector` full-text search alongside vector search and combine scores for better precision on keyword-heavy queries.
