"""
history.py — GET /history/{doc_id}

Returns the chat history for a given document session.
"""

from fastapi import APIRouter, HTTPException
from app.models import HistoryResponse, HistoryEntry
from app.routers.chat import _chat_histories

router = APIRouter()


@router.get("/history/{doc_id}", response_model=HistoryResponse)
async def get_history(doc_id: str):
    """Return the full chat history for a document session."""
    history = _chat_histories.get(doc_id, [])
    return HistoryResponse(
        doc_id=doc_id,
        history=[HistoryEntry(**h) for h in history],
    )
