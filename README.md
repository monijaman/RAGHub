# RAGHub — Chat with PDF

A full-stack RAG (Retrieval-Augmented Generation) app. Upload any PDF and ask questions in a chat UI — the AI answers using only that document's content, with page-number citations.

Built as a learning project that walks through every layer of a production RAG system from scratch.

---

## What is RAG?

RAG stands for **Retrieval-Augmented Generation**. Instead of asking an LLM to answer from memory (which leads to hallucinations), you:

1. **Retrieve** the most relevant passages from your document
2. **Augment** the LLM's prompt with those passages as context
3. **Generate** an answer grounded in the actual source material

The LLM never sees the whole PDF — only the 5 most relevant chunks for each question.

---

## How the pipeline works

```
                        INGESTION (on upload)
┌─────────┐   pypdf   ┌────────────┐  tiktoken  ┌────────────┐  OpenAI API  ┌───────────┐
│   PDF   │ ────────► │ Plain text │ ──────────► │   Chunks   │ ────────────► │ Embeddings│
│  file   │           │ per page   │             │ 500 tokens │              │ 1536-dim  │
└─────────┘           └────────────┘             │ 100 overlap│              └─────┬─────┘
                                                 └────────────┘                    │
                                                                                   ▼
                                                                             ┌───────────┐
                                                                             │  ChromaDB │
                                                                             │ (on disk) │
                                                                             └───────────┘

                        QUERYING (on each question)
┌──────────┐  OpenAI  ┌───────────┐  cosine   ┌───────────┐  GPT-4o-mini  ┌──────────┐
│ Question │ ────────► │ Embedding │ ────────► │  Top 5    │ ─────────────► │  Answer  │
│          │           │ 1536-dim  │  search   │  chunks   │               │ + sources│
└──────────┘           └───────────┘           └───────────┘               └──────────┘
```

---

## Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| **API framework** | [FastAPI](https://fastapi.tiangolo.com/) | REST endpoints, async, auto-docs |
| **ASGI server** | [Uvicorn](https://www.uvicorn.org/) | Runs the FastAPI app |
| **PDF parsing** | [pypdf](https://pypdf.readthedocs.io/) | Extracts text from each page |
| **Tokenizer** | [tiktoken](https://github.com/openai/tiktoken) | Token-aware chunking (same tokenizer as GPT-4) |
| **Embeddings** | [OpenAI](https://platform.openai.com/docs/guides/embeddings) `text-embedding-3-small` | Converts text to 1536-dim vectors |
| **Vector DB** | [ChromaDB](https://www.trychroma.com/) | Stores embeddings, cosine similarity search |
| **LLM** | [OpenAI](https://platform.openai.com/docs/guides/chat) `gpt-4o-mini` | Generates answers from retrieved context |
| **Config** | [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) | Typed settings loaded from `.env` |
| **Data models** | [Pydantic v2](https://docs.pydantic.dev/) | Request/response validation |
| **Frontend** | Plain HTML + JS | No framework — single `index.html` |
| **Runtime** | Python 3.12 | 3.14 lacks pre-built wheels for several deps |

---

## Project structure

```
RAGHub/
├── app/
│   ├── main.py               # FastAPI app entry point, middleware, route wiring
│   ├── config.py             # All settings from .env (pydantic-settings)
│   ├── models.py             # Pydantic schemas: UploadResponse, ChatRequest, etc.
│   ├── routers/
│   │   ├── upload.py         # POST /api/upload  — ingestion pipeline
│   │   ├── chat.py           # POST /api/chat    — RAG query + history
│   │   └── history.py        # GET  /api/history/{doc_id}
│   └── services/             # One file per pipeline step
│       ├── pdf_loader.py     # Step 1: PDF → list of {page, text} dicts
│       ├── chunker.py        # Step 2: text → overlapping token chunks
│       ├── embeddings.py     # Step 3: text → embedding vectors (batched)
│       ├── vector_store.py   # Step 4: ChromaDB store + similarity search
│       └── rag.py            # Step 5: full query pipeline (embed → search → LLM)
├── frontend/
│   └── index.html            # Chat UI (upload form, chat window, source citations)
├── tests/
│   └── test_rag.py
├── data/
│   ├── uploads/              # Saved PDF files (gitignored)
│   └── vector_db/            # ChromaDB files on disk (gitignored)
├── requirements.txt
├── .env                      # Your secrets (gitignored)
└── .env.example              # Template — copy this to .env
```

---

## Quick start

### 1. Install Python 3.12

> The system Python 3.14 on some distros is missing pre-built wheels for ChromaDB and other packages. Use Python 3.12.

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev build-essential
```

### 2. Create a virtual environment and install dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure your API key

```bash
cp .env.example .env
# Open .env and set OPENAI_API_KEY=sk-...
```

### 4. Run the server

```bash
# Use the venv's uvicorn, not the system one
.venv/bin/uvicorn app.main:app --reload
```

### 5. Open the app

- Frontend: `http://localhost:8000`
- Interactive API docs: `http://localhost:8000/docs`

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/upload` | Upload a PDF — runs the full ingestion pipeline, returns `doc_id` |
| `POST` | `/api/chat` | Ask a question about a document, returns answer + source chunks |
| `GET` | `/api/history/{doc_id}` | Retrieve chat history for a document session |
| `DELETE` | `/api/chat/{doc_id}/history` | Clear chat history for a document |
| `GET` | `/api/health` | Health check |

### Example: upload a PDF

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@my-document.pdf"
```

Response:
```json
{
  "doc_id": "doc_a3f9c12b4e87",
  "filename": "my-document.pdf",
  "pages": 24,
  "chunks": 87,
  "message": "Successfully processed 'my-document.pdf' into 87 chunks."
}
```

### Example: ask a question

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "doc_a3f9c12b4e87", "question": "What is the refund policy?"}'
```

Response:
```json
{
  "answer": "According to page 8, the refund policy allows returns within 30 days...",
  "sources": [
    {"text": "...refund within 30 days of purchase...", "page": 8, "chunk_index": 31}
  ],
  "doc_id": "doc_a3f9c12b4e87"
}
```

---

## Configuration

All settings live in `.env` and are validated at startup:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. Your OpenAI key. |
| `LLM_MODEL` | `gpt-4o-mini` | Model used to generate answers |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model used to create vectors |
| `CHUNK_SIZE` | `500` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `100` | Token overlap between adjacent chunks |
| `TOP_K` | `5` | Number of chunks retrieved per question |
| `UPLOAD_DIR` | `data/uploads` | Where PDFs are saved |
| `VECTOR_DB_DIR` | `data/vector_db` | Where ChromaDB stores its files |

---

## How each service works

### `pdf_loader.py` — PDF to text
Uses `pypdf` to read each page. Returns a list of `{"page": N, "text": "..."}` dicts, skipping blank or image-only pages.

### `chunker.py` — Text to chunks
Tokenizes the full document with `tiktoken` (the same tokenizer GPT-4 uses), then slides a window of `CHUNK_SIZE` tokens with `CHUNK_OVERLAP` step-back. This means a sentence that falls on a chunk boundary appears in both adjacent chunks — nothing gets lost.

### `embeddings.py` — Text to vectors
Calls the OpenAI embeddings API in one batched request for all chunks. Each chunk becomes a 1536-dimensional float vector representing its semantic meaning. Texts with similar meanings produce vectors that point in similar directions.

### `vector_store.py` — Store and search
Each document gets its own ChromaDB collection named by `doc_id`. Stores chunks, their embeddings, and metadata (page number, chunk index). At query time, computes cosine similarity between the query vector and all stored vectors, returning the top K matches.

### `rag.py` — Full query pipeline
1. Embed the user's question
2. Search ChromaDB for the top K most similar chunks
3. Build a system prompt that injects those chunks as context
4. Send the conversation (including prior turns) to GPT-4o-mini
5. Return the answer + the source chunks used

Chat history is kept in-memory per `doc_id` (last 20 messages), so the LLM remembers what was asked earlier in the session.

---

## Run tests

```bash
.venv/bin/pytest tests/ -v
```

---

## Use Ollama instead of OpenAI (free, local, no API key)

1. Install Ollama: https://ollama.ai
2. Pull the models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3
   ```
3. In `embeddings.py`, replace the OpenAI call:
   ```python
   import ollama
   response = ollama.embeddings(model="nomic-embed-text", prompt=text)
   return response["embedding"]
   ```
4. In `rag.py`, replace the chat completion call with the Ollama chat API.

---

## Possible extensions

**Level 2**
- [ ] Streaming responses (`stream=True` + SSE endpoint)
- [ ] Persistent chat history (replace in-memory dict with SQLite)
- [ ] Better source display (highlight the exact matched sentence)

**Level 3**
- [ ] Multi-document chat (search across all uploaded docs)
- [ ] Re-ranking (cross-encoder model to re-score retrieved chunks)
- [ ] Hybrid search (vector search + BM25 keyword search combined)

**Level 4 — Agentic RAG**
- [ ] Tool-calling: let the LLM decide whether to search or answer directly
- [ ] Multi-step reasoning: chain multiple searches for complex questions
- [ ] LangChain / LlamaIndex integration
