"""
rag.py — Step 6: The full RAG query pipeline.

This is where everything comes together:

  User question
      ↓
  Embed question  →  [0.12, -0.45, ...]
      ↓
  Search vector DB  →  top 5 relevant chunks
      ↓
  Build prompt:
    "Answer using this context: <chunks>
     Question: <question>"
      ↓
  Send to LLM  →  "According to page 3, the refund policy..."
      ↓
  Return answer + source chunks (for citations)

RAG concept: The LLM doesn't know anything about your PDF. We "augment"
its knowledge by injecting relevant text chunks as context. The LLM then
synthesizes an answer from that context. This is called "Retrieval-Augmented
Generation" — Retrieve → Augment → Generate.
"""

from openai import OpenAI
from app.config import settings
from app.services.embeddings import embed_text
from app.services.vector_store import search_chunks

_client = OpenAI(api_key=settings.openai_api_key)


def _build_system_prompt(context_chunks: list[dict]) -> str:
    """
    Build the system prompt that contains the retrieved context.
    The LLM answers ONLY based on what's in these chunks.
    """
    context_text = "\n\n---\n\n".join(
        f"[Page {c['page']}]\n{c['text']}"
        for c in context_chunks
    )
    return f"""You are a helpful assistant that answers questions about a document.

Use ONLY the context below to answer the question. If the answer is not in the
context, say "I couldn't find that in the document."

Always mention the page number(s) where you found the answer.

CONTEXT:
{context_text}"""


def query_rag(
    doc_id: str,
    question: str,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Run the full RAG pipeline for one question.

    Args:
        doc_id:        which document to search
        question:      the user's question
        chat_history:  prior conversation turns for multi-turn context
                       [{"role": "user", "content": "..."}, ...]

    Returns:
        {
            "answer": "...",
            "sources": [{"text": "...", "page": 2, "chunk_index": 4}, ...]
        }
    """
    # Step 1: Embed the question
    query_embedding = embed_text(question)

    # Step 2: Find relevant chunks in the vector DB
    relevant_chunks = search_chunks(doc_id, query_embedding, settings.top_k)

    # Step 3: Build messages for the LLM
    messages = [
        {"role": "system", "content": _build_system_prompt(relevant_chunks)},
    ]

    # Include prior conversation turns (chat memory)
    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": question})

    # Step 4: Call the LLM
    response = _client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.2,     # low temperature = more factual, less creative
        max_tokens=1024,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": [
            {
                "text": c["text"],
                "page": c["page"],
                "chunk_index": c["chunk_index"],
            }
            for c in relevant_chunks
        ],
    }
