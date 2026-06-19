"""
models.py — Pydantic schemas for request/response bodies.

Think of these as "contracts":
  - What data the API expects (request models)
  - What data the API returns (response models)
"""

from pydantic import BaseModel
from typing import Optional


# ── Upload ─────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    doc_id: str          # unique ID for this document in the vector DB
    filename: str
    pages: int           # total pages in the PDF
    chunks: int          # how many chunks were created and stored
    message: str


# ── Chat ───────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    doc_id: str          # which document to search in
    question: str
    chat_history: Optional[list[dict]] = []   # [{"role": "user", "content": "..."}, ...]


class SourceChunk(BaseModel):
    text: str
    page: int
    chunk_index: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]   # the chunks used to generate the answer
    doc_id: str


# ── History ────────────────────────────────────────────────────────────────────

class HistoryEntry(BaseModel):
    role: str        # "user" or "assistant"
    content: str


class HistoryResponse(BaseModel):
    doc_id: str
    history: list[HistoryEntry]
