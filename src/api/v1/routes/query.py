from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from src.api.v1.schemas.query_schema import QueryRequest, QueryResponse
from src.api.v1.services.query_service import query_documents, stream_query_documents

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    result = query_documents(request.query, k=request.k, chunk_type=request.chunk_type)
    return QueryResponse(**result)


@router.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """Stream the Gemini answer token-by-token using Server-Sent Events (SSE).

    Each event is a JSON object on a `data:` line followed by two newlines:

        data: {"type": "token",   "text": "..."}      — answer fragment
        data: {"type": "sources", "sources": [...]}   — chunk metadata (no image bytes)
        data: {"type": "done"}                        — stream finished

    Example (JavaScript):
        const res = await fetch('/api/v1/query/stream', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query: '...', k: 5}),
        });
        const reader = res.body.getReader();
        // read lines and JSON.parse each data: payload
    """
    return StreamingResponse(
        stream_query_documents(request.query, k=request.k, chunk_type=request.chunk_type),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx proxy buffering
        },
    )
