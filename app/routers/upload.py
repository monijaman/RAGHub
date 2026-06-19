"""
upload.py — POST /upload

Handles the full ingestion pipeline:
  PDF file → extract text → chunk → embed → store in vector DB
"""

import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.config import settings
from app.models import UploadResponse
from app.services.pdf_loader import load_pdf
from app.services.chunker import chunk_text
from app.services.embeddings import embed_batch
from app.services.vector_store import store_chunks, document_exists

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF and process it through the full RAG ingestion pipeline.

    Steps performed here:
      1. Save the uploaded file to disk
      2. Extract text from each page
      3. Split text into overlapping chunks
      4. Generate embeddings for all chunks (batched API call)
      5. Store chunks + embeddings in ChromaDB
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Create a unique ID for this document
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"

    # 1. Save the PDF to disk
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{doc_id}.pdf"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2. Extract text from each page
    pages, total_pages = load_pdf(str(file_path))
    if not pages:
        raise HTTPException(status_code=422, detail="No extractable text found in PDF.")

    # 3. Chunk the text
    chunks = chunk_text(pages)

    # 4. Generate embeddings for all chunks in one batched API call
    texts = [c["text"] for c in chunks]
    embeddings = embed_batch(texts)

    # 5. Store in vector DB
    store_chunks(doc_id, chunks, embeddings)

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        pages=total_pages,
        chunks=len(chunks),
        message=f"Successfully processed '{file.filename}' into {len(chunks)} chunks.",
    )
