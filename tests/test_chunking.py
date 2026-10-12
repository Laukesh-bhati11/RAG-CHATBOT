"""
Unit Tests for Text Chunking Module (tests/test_chunking.py).

Tests verify:
1. Recursive text splitting divides long pages into smaller chunks.
2. Metadata (source, page, chunk_id) is preserved and enriched.
3. Configurable chunk size and overlap parameters function as expected.
4. Empty document lists are handled safely.
"""

from pathlib import Path
import pytest
from src.document_loader import load_single_pdf
from src.chunker import split_documents
from langchain_core.documents import Document

SAMPLE_PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "documents" / "sample_ai_notes.pdf"


def test_chunking_creates_more_chunks_than_pages():
    """Verify that chunking breaks pages into smaller chunks."""
    pages = load_single_pdf(SAMPLE_PDF_PATH)
    assert len(pages) > 0

    chunks = split_documents(pages, chunk_size=400, chunk_overlap=100)
    assert len(chunks) >= len(pages), "Smaller chunk size should produce at least as many chunks as pages"


def test_chunking_metadata_preservation():
    """Verify that every chunk maintains source, page, and chunk_id."""
    pages = load_single_pdf(SAMPLE_PDF_PATH)
    chunks = split_documents(pages, chunk_size=500, chunk_overlap=100)

    for idx, chunk in enumerate(chunks):
        assert "source" in chunk.metadata
        assert chunk.metadata["source"] == "sample_ai_notes.pdf"
        assert "page" in chunk.metadata
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["chunk_id"] == idx


def test_chunking_empty_list():
    """Splitting an empty list should safely return an empty list."""
    chunks = split_documents([])
    assert chunks == []


def test_chunk_size_limits():
    """Verify that chunk content lengths stay within the expected boundaries."""
    long_text = "Python is great! " * 200  # ~3400 characters
    doc = Document(page_content=long_text, metadata={"source": "test.pdf", "page": 0})
    
    chunk_size = 500
    overlap = 100
    chunks = split_documents([doc], chunk_size=chunk_size, chunk_overlap=overlap)

    assert len(chunks) > 1
    for chunk in chunks:
        # Chunks should not substantially exceed chunk_size
        assert len(chunk.page_content) <= chunk_size + 50
