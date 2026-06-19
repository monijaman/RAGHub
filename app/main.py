"""
main.py — FastAPI application entry point.

Run with:  uvicorn app.main:app --reload
Docs at:   http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routers import upload, chat, history

app = FastAPI(
    title="RAGHub — Chat with PDF",
    description="Upload a PDF, ask questions, get AI answers with source citations.",
    version="1.0.0",
)

# Allow the frontend (any origin in dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(chat.router,   prefix="/api", tags=["Chat"])
app.include_router(history.router, prefix="/api", tags=["History"])

# Serve the frontend HTML at "/"
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


@app.get("/api/health")
def health():
    return {"status": "ok"}
