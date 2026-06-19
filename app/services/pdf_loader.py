"""
pdf_loader.py — Step 2: Extract text from a PDF file.

RAG concept: Raw data must be converted to plain text before we can
split, embed, or search it. Here we use pypdf to read each page.
"""

from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path: str) -> tuple[list[dict], int]:
    """
    Read a PDF and return a list of page dicts.

    Returns:
        pages:  [{"page": 1, "text": "..."}, ...]
        total:  total number of pages
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    reader = PdfReader(str(path))
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:                            # skip blank/image-only pages
            pages.append({
                "page": i + 1,             # 1-indexed page numbers
                "text": text,
            })

    return pages, len(reader.pages)
