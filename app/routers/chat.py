"""
chat.py — POST /chat

Runs the RAG query pipeline and returns an answer with source citations.
Also maintains in-memory chat history per document session.
"""

from fastapi import APIRouter, HTTPException
from app.models import ChatRequest, ChatResponse, SourceChunk
from app.services.rag import query_rag
from app.services.vector_store import document_exists

router = APIRouter()

# In-memory chat history store: { doc_id: [ {role, content}, ... ] }
# Level 3 upgrade: replace this with Redis or a database for persistence
_chat_histories: dict[str, list[dict]] = {}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Ask a question about an uploaded PDF.

    Sends the question through the RAG pipeline:
      question → embed → search vector DB → inject context → LLM → answer

    Chat history is maintained per doc_id so the LLM has conversation memory.
    """
    if not document_exists(request.doc_id):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.doc_id}' not found. Please upload a PDF first.",
        )

    # Load or initialize history for this document
    history = _chat_histories.setdefault(request.doc_id, [])

    # Run RAG
    result = query_rag(
        doc_id=request.doc_id,
        question=request.question,
        chat_history=history,
    )

    # Update history with this turn
    history.append({"role": "user", "content": request.question})
    history.append({"role": "assistant", "content": result["answer"]})

    # Keep last 10 turns (20 messages) to avoid unbounded memory growth
    _chat_histories[request.doc_id] = history[-20:]

    return ChatResponse(
        answer=result["answer"],
        sources=[
            SourceChunk(
                text=s["text"][:300],       # truncate for the response payload
                page=s["page"],
                chunk_index=s["chunk_index"],
            )
            for s in result["sources"]
        ],
        doc_id=request.doc_id,
    )


@router.delete("/chat/{doc_id}/history")
async def clear_history(doc_id: str):
    """Clear the chat history for a document session."""
    _chat_histories.pop(doc_id, None)
    return {"message": f"Chat history cleared for {doc_id}"}
