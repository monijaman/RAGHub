# RAGHub — Chat with PDF

A beginner-to-advanced RAG (Retrieval-Augmented Generation) learning project.
Upload any PDF and ask questions — the AI answers using only the document's content,
with page-number citations.

---

## How it works

```
PDF → Extract text → Chunk (500 tokens, 100 overlap) → Embed → ChromaDB
                                                                    ↑
User question → Embed → Similarity search → Top 5 chunks → LLM → Answer
```

---

## Quick start

### 1. Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your OpenAI API key
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run the server
```bash
uvicorn app.main:app --reload
```

### 4. Open the app
Visit `http://localhost:8000` in your browser.

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/upload` | Upload a PDF, returns `doc_id` |
| `POST` | `/api/chat` | Ask a question, returns answer + sources |
| `GET`  | `/api/history/{doc_id}` | Get chat history |
| `DELETE` | `/api/chat/{doc_id}/history` | Clear chat history |
| `GET`  | `/api/health` | Health check |

Interactive docs: `http://localhost:8000/docs`

---

## Project structure

```
RAGHub/
├── app/
│   ├── main.py               # FastAPI app, routes wired here
│   ├── config.py             # Settings from .env
│   ├── models.py             # Pydantic request/response schemas
│   ├── routers/
│   │   ├── upload.py         # POST /api/upload
│   │   ├── chat.py           # POST /api/chat
│   │   └── history.py        # GET  /api/history/{doc_id}
│   └── services/
│       ├── pdf_loader.py     # Step 2: Extract text from PDF
│       ├── chunker.py        # Step 3: Split text into chunks
│       ├── embeddings.py     # Step 4: Generate embedding vectors
│       ├── vector_store.py   # Step 5: Store/search in ChromaDB
│       └── rag.py            # Step 6: Full query pipeline
├── frontend/
│   └── index.html            # Chat UI (no framework, plain JS)
├── tests/
│   └── test_rag.py
├── data/
│   ├── uploads/              # Saved PDF files
│   └── vector_db/            # ChromaDB persistent storage
├── requirements.txt
└── .env.example
```

---

## Learning path (follow in this order)

### Phase 1 — Understand the concepts (1–2 days)
- Read `app/services/embeddings.py` — what is a vector?
- Read `app/services/chunker.py` — why do we overlap chunks?
- Read `app/services/vector_store.py` — what is cosine similarity?

### Phase 2 — Run and experiment (2–4 days)
- Upload a PDF and ask questions
- Change `CHUNK_SIZE` and `CHUNK_OVERLAP` in `.env` — how does it affect answers?
- Change `TOP_K` — what happens with 1 chunk vs 10?
- Read `app/services/rag.py` — understand the full pipeline

### Phase 3 — Extend the app (advanced)

**Level 2 upgrades (try these next)**
- [ ] Streaming responses (`stream=True` in OpenAI call + SSE endpoint)
- [ ] Persistent chat history (replace in-memory dict with SQLite)
- [ ] Better source display (highlight the exact sentence, not full chunk)

**Level 3 upgrades**
- [ ] Multi-document chat (search across all docs, not just one)
- [ ] Re-ranking (use a cross-encoder model to re-score retrieved chunks)
- [ ] Hybrid search (combine vector search with BM25 keyword search)

**Level 4 — Agent-based RAG**
- [ ] Tool-calling: let the LLM decide whether to search or answer directly
- [ ] Multi-step reasoning: chain multiple searches for complex questions
- [ ] LangChain / LlamaIndex integration

---

## Use Ollama instead of OpenAI (free, runs locally)

1. Install Ollama: https://ollama.ai
2. `ollama pull nomic-embed-text`
3. `ollama pull llama3`
4. In `embeddings.py`, replace the OpenAI call with:
   ```python
   import ollama
   response = ollama.embeddings(model="nomic-embed-text", prompt=text)
   return response["embedding"]
   ```
5. In `rag.py`, replace the OpenAI call with the Ollama chat API.

---

## Run tests
```bash
pytest tests/ -v
```
