"""
vector_store.py — Step 5: Store and search embeddings using ChromaDB.

RAG concept: A vector database stores embeddings alongside their source
text and metadata. When we search with a query embedding, it finds the
stored embeddings that are geometrically closest (most similar meaning).

ChromaDB is a local, file-based vector DB — perfect for learning because:
  - No server setup needed
  - Data persists to disk automatically
  - Simple Python API

How similarity search works (cosine similarity):
  - Two vectors pointing in the same direction → score = 1.0 (identical meaning)
  - Vectors at 90°                            → score = 0.0 (unrelated)
  - Vectors pointing opposite directions      → score = -1.0 (opposite meaning)
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings


def _get_client() -> chromadb.PersistentClient:
    """Return a ChromaDB client that saves data to disk."""
    return chromadb.PersistentClient(
        path=settings.vector_db_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _get_collection(client: chromadb.PersistentClient, doc_id: str):
    """
    Get or create a ChromaDB collection for this document.
    Each PDF gets its own collection, named by doc_id.
    """
    return client.get_or_create_collection(
        name=doc_id,
        metadata={"hnsw:space": "cosine"},  # use cosine similarity
    )


def store_chunks(doc_id: str, chunks: list[dict], embeddings: list[list[float]]) -> None:
    """
    Save chunks + their embeddings into the vector database.

    Args:
        doc_id:     unique ID for this document (used as collection name)
        chunks:     list of chunk dicts from chunker.py
        embeddings: list of embedding vectors, one per chunk
    """
    client = _get_client()
    collection = _get_collection(client, doc_id)

    # ChromaDB expects parallel lists: ids, embeddings, documents, metadatas
    ids = [f"{doc_id}_chunk_{c['chunk_index']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{"page": c["page"], "chunk_index": c["chunk_index"]} for c in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def search_chunks(doc_id: str, query_embedding: list[float], top_k: int) -> list[dict]:
    """
    Find the top_k chunks most similar to the query embedding.

    Returns:
        [
            {
                "text": "...",
                "page": 2,
                "chunk_index": 4,
                "score": 0.87,   # cosine similarity (higher = more relevant)
            },
            ...
        ]
    """
    client = _get_client()
    collection = _get_collection(client, doc_id)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),   # can't request more than exists
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        # ChromaDB cosine distance = 1 - cosine_similarity
        similarity = 1.0 - distance

        chunks.append({
            "text": results["documents"][0][i],
            "page": results["metadatas"][0][i]["page"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "score": round(similarity, 4),
        })

    return chunks


def document_exists(doc_id: str) -> bool:
    """Check if this document has already been ingested."""
    client = _get_client()
    try:
        col = client.get_collection(doc_id)
        return col.count() > 0
    except Exception:
        return False


def delete_document(doc_id: str) -> None:
    """Remove all chunks for a document from the vector DB."""
    client = _get_client()
    try:
        client.delete_collection(doc_id)
    except Exception:
        pass
