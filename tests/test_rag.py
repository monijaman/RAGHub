"""
test_rag.py — Unit tests for the RAG pipeline components.

Run with:  pytest tests/ -v

These tests verify the logic without hitting the OpenAI API by using
mock objects. This is important because:
  1. API calls cost money
  2. Tests should run instantly, not wait for the network
  3. We want to test OUR code, not OpenAI's API
"""

import pytest
from unittest.mock import patch, MagicMock


# ── Test: chunker ─────────────────────────────────────────────────────────────

def test_chunk_text_basic():
    """Chunks should split long text and respect size/overlap settings."""
    from app.services.chunker import chunk_text

    # Create a page with enough text to produce multiple chunks
    long_text = "This is a test sentence. " * 100   # ~500+ tokens
    pages = [{"page": 1, "text": long_text}]

    chunks = chunk_text(pages)

    assert len(chunks) > 1, "Long text should produce multiple chunks"
    for c in chunks:
        assert "text" in c
        assert "page" in c
        assert "chunk_index" in c
        assert c["token_count"] <= 600   # should be close to CHUNK_SIZE


def test_chunk_preserves_page_numbers():
    """Each chunk should record which PDF page it came from."""
    from app.services.chunker import chunk_text

    pages = [
        {"page": 1, "text": "Page one content. " * 20},
        {"page": 2, "text": "Page two content. " * 20},
    ]
    chunks = chunk_text(pages)

    # At least the first chunk should be from page 1
    assert chunks[0]["page"] == 1


def test_chunk_indices_are_sequential():
    """chunk_index values should start at 0 and increase by 1."""
    from app.services.chunker import chunk_text

    pages = [{"page": 1, "text": "word " * 200}]
    chunks = chunk_text(pages)

    for i, c in enumerate(chunks):
        assert c["chunk_index"] == i


# ── Test: PDF loader ──────────────────────────────────────────────────────────

def test_load_pdf_file_not_found():
    """load_pdf should raise FileNotFoundError for missing files."""
    from app.services.pdf_loader import load_pdf

    with pytest.raises(FileNotFoundError):
        load_pdf("/nonexistent/path/file.pdf")


# ── Test: vector store ────────────────────────────────────────────────────────

def test_document_exists_returns_false_for_unknown():
    """document_exists should return False for a doc_id that was never stored."""
    from app.services.vector_store import document_exists

    assert document_exists("doc_nonexistent_000") is False


# ── Test: RAG pipeline (mocked) ───────────────────────────────────────────────

@patch("app.services.rag.embed_text")
@patch("app.services.rag.search_chunks")
@patch("app.services.rag._client")
def test_query_rag_returns_answer_and_sources(mock_client, mock_search, mock_embed):
    """query_rag should call embed → search → LLM and return structured result."""
    from app.services.rag import query_rag

    # Set up mock responses
    mock_embed.return_value = [0.1] * 1536

    mock_search.return_value = [
        {"text": "Refunds are processed in 5 days.", "page": 3, "chunk_index": 7, "score": 0.9},
        {"text": "Contact support for refund requests.", "page": 4, "chunk_index": 8, "score": 0.85},
    ]

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "According to page 3, refunds take 5 days."
    mock_client.chat.completions.create.return_value = mock_response

    result = query_rag(doc_id="doc_test123", question="What is the refund policy?")

    assert "answer" in result
    assert "sources" in result
    assert len(result["sources"]) == 2
    assert result["sources"][0]["page"] == 3

    # Verify the pipeline called each step
    mock_embed.assert_called_once_with("What is the refund policy?")
    mock_search.assert_called_once()
    mock_client.chat.completions.create.assert_called_once()
