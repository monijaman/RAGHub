"""
chunker.py — Step 3: Split text into overlapping chunks.

RAG concept: LLMs have context limits, and a full PDF can be thousands
of tokens. We split the text into smaller pieces so:
  1. Each piece fits in the LLM's context window.
  2. We can find the most RELEVANT piece for a given question.

Why overlap? So that information that spans a chunk boundary is still
captured. If chunk A ends mid-sentence, chunk B starts 100 tokens back
so neither chunk loses that sentence.

Visualization:
  |-------- chunk 0 --------|
                   |-------- chunk 1 --------|
                                    |-------- chunk 2 --------|
  <---------  overlap  ----------->
"""

import tiktoken
from app.config import settings


def _get_encoder():
    # cl100k_base is the tokenizer used by GPT-4 and text-embedding-3 models
    return tiktoken.get_encoding("cl100k_base")


def chunk_text(pages: list[dict]) -> list[dict]:
    """
    Split page text into overlapping token-based chunks.

    Args:
        pages: [{"page": 1, "text": "..."}, ...]

    Returns:
        chunks: [
            {
                "chunk_index": 0,
                "text": "...",
                "page": 1,          # page where this chunk starts
                "token_count": 487,
            },
            ...
        ]
    """
    enc = _get_encoder()
    size = settings.chunk_size
    overlap = settings.chunk_overlap

    # 1. Flatten all pages into one token stream, remembering page boundaries
    all_tokens: list[int] = []
    token_page_map: list[int] = []     # all_tokens[i] came from which page?

    for page_data in pages:
        tokens = enc.encode(page_data["text"])
        all_tokens.extend(tokens)
        token_page_map.extend([page_data["page"]] * len(tokens))

    # 2. Slide a window of `size` tokens with `overlap` step back
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(all_tokens):
        end = min(start + size, len(all_tokens))
        token_slice = all_tokens[start:end]

        chunk_text_str = enc.decode(token_slice)
        page_number = token_page_map[start]   # page of the first token in chunk

        chunks.append({
            "chunk_index": chunk_index,
            "text": chunk_text_str,
            "page": page_number,
            "token_count": len(token_slice),
        })

        chunk_index += 1
        start += size - overlap   # move forward, but keep `overlap` tokens

    return chunks
