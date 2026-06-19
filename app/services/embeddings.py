"""
embeddings.py — Step 4: Convert text into embedding vectors.

RAG concept: An embedding is a list of numbers (a vector) that represents
the MEANING of text in a high-dimensional space. Texts with similar
meanings end up close to each other in this space.

Example:
  "What is the refund policy?"  →  [0.12, -0.45, 0.88, ...]  (1536 numbers)
  "How do I get a refund?"      →  [0.13, -0.44, 0.87, ...]  (very close!)
  "The sky is blue."            →  [0.91,  0.23, -0.12, ...]  (far away)

We use this to find the PDF chunks most relevant to a user's question.

--- Ollama alternative (free, runs locally, no API key needed) ---
  pip install ollama
  ollama pull nomic-embed-text
  Then replace the OpenAI call with:
      import ollama
      response = ollama.embeddings(model="nomic-embed-text", prompt=text)
      return response["embedding"]
"""

from openai import OpenAI
from app.config import settings

_client = OpenAI(api_key=settings.openai_api_key)


def embed_text(text: str) -> list[float]:
    """Convert a single string into an embedding vector."""
    response = _client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )
    return response.data[0].embedding   # list of ~1536 floats


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple texts in one API call (more efficient than looping).
    OpenAI allows up to 2048 texts per request.
    """
    response = _client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    # The API returns embeddings in the same order as input
    return [item.embedding for item in response.data]
