import json
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.db import similarity_search

load_dotenv()

_SYSTEM_PROMPT = (
    "You are a helpful assistant for document question-answering. "
    "Answer the question using ONLY the provided context (text, tables, and images). "
    "If the answer is not present in the context, say you don't know. "
    "When citing information, mention the page number and section."
)

# Module-level LLM singleton — avoids re-instantiating a new HTTP client per request.
_llm = ChatGoogleGenerativeAI(
    model=os.getenv("GOOGLE_LLM_MODEL"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


def _build_messages(
    query: str, k: int, chunk_type: str | None
) -> tuple[list, list[dict]]:
    """Retrieve relevant chunks and build the multimodal LangChain message list.

    Returns:
        messages: [SystemMessage, HumanMessage] ready to pass to the LLM.
        sources:  List of source metadata dicts. Image chunks include
                  'image_base64' so callers can return visual evidence to
                  the client alongside the text answer.
    """
    chunks = similarity_search(query, k=k, chunk_type=chunk_type)

    message_parts: list[dict] = []
    sources: list[dict] = []
    text_blocks: list[str] = []

    for chunk in chunks:
        ct = chunk["chunk_type"]
        page = chunk.get("page_number")
        section = chunk.get("section") or "—"

        # ── Build source entry ────────────────────────────────────────────────
        # Feature 12: include image bytes in the source so the API response
        # contains the actual visual evidence the LLM reasoned over.
        source_entry: dict = {
            "chunk_type": ct,
            "page_number": page,
            "section": section,
            "source_file": chunk.get("source_file", ""),
            "element_type": chunk.get("element_type"),
            "similarity": round(chunk.get("similarity", 0), 4),
        }
        if ct == "image" and chunk.get("image_base64"):
            source_entry["image_base64"] = chunk["image_base64"]
        sources.append(source_entry)

        # ── Assemble multimodal prompt parts ─────────────────────────────────
        if ct in ("text", "table"):
            label = "TABLE" if ct == "table" else "TEXT"
            text_blocks.append(
                f"[{label} | page {page} | {section}]\n{chunk['content']}"
            )
        elif ct == "image" and chunk.get("image_base64"):
            # Flush accumulated text before inserting an image part so the
            # LLM sees text context immediately before the relevant image.
            if text_blocks:
                message_parts.append({
                    "type": "text",
                    "text": "\n\n".join(text_blocks),
                })
                text_blocks = []
            message_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{chunk['image_base64']}"
                },
            })

    # Flush any remaining text blocks
    if text_blocks:
        message_parts.append({
            "type": "text",
            "text": "\n\n".join(text_blocks),
        })

    # Append the question at the end of the context
    message_parts.append({
        "type": "text",
        "text": f"\n\nQuestion: {query}",
    })

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=message_parts),
    ]
    return messages, sources


def _extract_text(content) -> str:
    """Extract plain text from a LangChain response content value.

    Handles both plain-string responses (standard models) and list-of-parts
    responses (thinking models like gemini-3.1-pro-preview).
    """
    if isinstance(content, list):
        return " ".join(
            part["text"] for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return str(content).strip()


def query_documents(query: str, k: int = 5, chunk_type: str | None = None) -> dict:
    """Run the full RAG pipeline and return the answer with sources.

    Returns:
        {
          "answer":  str — Gemini's answer grounded in the retrieved context,
          "sources": list[dict] — retrieved chunk metadata; image chunks
                     include an 'image_base64' key with the PNG bytes so the
                     client can render the visual evidence (Feature 12).
        }
    """
    messages, sources = _build_messages(query, k, chunk_type)
    response = _llm.invoke(messages)
    return {
        "answer": _extract_text(response.content),
        "sources": sources,
    }


async def stream_query_documents(
    query: str, k: int = 5, chunk_type: str | None = None
):
    """Async generator that streams the Gemini answer as SSE-formatted events.

    Yields newline-delimited SSE events (each ending with two newlines):

        data: {"type": "token",   "text": "..."}
            — one or more answer tokens from the LLM stream

        data: {"type": "sources", "sources": [...]}
            — retrieved chunk metadata sent after the stream completes.
              NOTE: image_base64 is omitted here to keep event size small.
              Use the non-streaming POST /query endpoint if you need images.

        data: {"type": "done"}
            — signals the stream is finished; client can close the connection.

    Usage (JavaScript):
        const es = new EventSource('/api/v1/query/stream', {method:'POST',...});
        es.onmessage = e => {
            const ev = JSON.parse(e.data);
            if (ev.type === 'token') appendText(ev.text);
            if (ev.type === 'done')  es.close();
        };
    """
    messages, sources = _build_messages(query, k, chunk_type)

    # Stream tokens from Gemini
    async for chunk in _llm.astream(messages):
        text = _extract_text(chunk.content)
        if text:
            yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"

    # Send source metadata (without image bytes — keep SSE payload small)
    slim_sources = [
        {k: v for k, v in s.items() if k != "image_base64"}
        for s in sources
    ]
    yield f"data: {json.dumps({'type': 'sources', 'sources': slim_sources})}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


