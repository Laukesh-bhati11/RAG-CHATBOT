"""
Unit Tests for Document Loader Module (tests/test_loader.py).

Tests verify:
1. Loading a valid PDF extracts correct number of pages.
2. Metadata (source filename, page number) is preserved accurately.
3. Multiple PDF loading aggregates pages seamlessly.
4. Non-existent or invalid files are handled gracefully without exceptions.
"""

from pathlib import Path
import pytest
from src.document_loader import load_single_pdf, load_multiple_pdfs
from src.utils import is_valid_pdf

SAMPLE_PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "documents" / "sample_ai_notes.pdf"


def test_sample_pdf_exists():
    """Verify that the test fixture PDF is present."""
    assert SAMPLE_PDF_PATH.exists(), f"Sample PDF missing at {SAMPLE_PDF_PATH}"
    is_valid, msg = is_valid_pdf(SAMPLE_PDF_PATH)
    assert is_valid is True, f"Validation failed: {msg}"


def test_load_single_pdf_success():
    """Test loading a single valid PDF document."""
    documents = load_single_pdf(SAMPLE_PDF_PATH)
    assert len(documents) == 5, f"Expected 5 pages, got {len(documents)}"

    # Check first page content and metadata
    page_1 = documents[0]
    assert "Python" in page_1.page_content
    assert page_1.metadata["source"] == "sample_ai_notes.pdf"
    assert page_1.metadata["page"] == 0

    # Check third page (RAG)
    page_3 = documents[2]
    assert "Retrieval-Augmented Generation" in page_3.page_content
    assert page_3.metadata["page"] == 2


def test_metadata_preservation_across_all_pages():
    """Ensure every loaded page preserves source and page index."""
    documents = load_single_pdf(SAMPLE_PDF_PATH)
    for idx, doc in enumerate(documents):
        assert doc.metadata["source"] == "sample_ai_notes.pdf"
        assert doc.metadata["page"] == idx
        assert len(doc.page_content.strip()) > 50


def test_load_multiple_pdfs():
    """Test loading multiple files simultaneously."""
    docs, failed = load_multiple_pdfs([SAMPLE_PDF_PATH, SAMPLE_PDF_PATH])
    assert len(docs) == 10  # 5 + 5
    assert len(failed) == 0


def test_load_nonexistent_pdf():
    """Verify that a missing PDF returns empty list rather than raising an unhandled exception."""
    fake_path = Path("data/documents/does_not_exist.pdf")
    docs = load_single_pdf(fake_path)
    assert docs == []

    docs, failed = load_multiple_pdfs([fake_path])
    assert docs == []
    assert len(failed) == 1
