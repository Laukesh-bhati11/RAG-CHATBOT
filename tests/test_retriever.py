"""
Unit Tests for Vector Store and Retriever (tests/test_retriever.py).

Tests verify:
1. FAISS vector store creation using deterministic offline embeddings (FakeEmbeddings).
2. Index persistence to disk and subsequent loading without re-embedding.
3. Top-K similarity search and metadata retrieval.
4. Empty query handling.
5. Accurate source citation extraction and 1-based page numbering.
6. Safe clearing of the vector database.
"""

import shutil
from pathlib import Path
import pytest

from src.document_loader import load_single_pdf
from src.chunker import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import (
    create_vector_store,
    load_vector_store,
    vector_store_exists,
    clear_vector_store,
)
from src.retriever import retrieve_relevant_chunks, format_retrieved_chunks_debug
from src.utils import format_sources, get_sources_markdown

SAMPLE_PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "documents" / "sample_ai_notes.pdf"
TEST_VECTOR_DIR = Path(__file__).resolve().parent.parent / "vectorstore" / "test_faiss_index"


@pytest.fixture(scope="module")
def sample_chunks():
    """Loads and chunks the sample PDF once for test execution."""
    pages = load_single_pdf(SAMPLE_PDF_PATH)
    chunks = split_documents(pages, chunk_size=500, chunk_overlap=100)
    return chunks


@pytest.fixture(scope="module")
def test_embedding_model():
    """Returns offline deterministic embeddings for fast, API-free test execution."""
    return get_embedding_model(provider="fake")


def test_create_and_save_vector_store(sample_chunks, test_embedding_model):
    """Test creating a FAISS vector store and persisting it to disk."""
    if TEST_VECTOR_DIR.exists():
        shutil.rmtree(TEST_VECTOR_DIR)

    vs = create_vector_store(sample_chunks, test_embedding_model, save_path=TEST_VECTOR_DIR)
    assert vs is not None
    assert vector_store_exists(TEST_VECTOR_DIR) is True
    assert (TEST_VECTOR_DIR / "index.faiss").exists()
    assert (TEST_VECTOR_DIR / "index.pkl").exists()


def test_load_vector_store(test_embedding_model):
    """Test loading an existing index without reprocessing documents."""
    vs = load_vector_store(test_embedding_model, save_path=TEST_VECTOR_DIR)
    assert vs is not None


def test_retrieval_top_k(test_embedding_model):
    """Test similarity search retrieves the requested top_k documents."""
    vs = load_vector_store(test_embedding_model, save_path=TEST_VECTOR_DIR)
    assert vs is not None

    top_k = 3
    results = retrieve_relevant_chunks(vs, query="What is Python?", top_k=top_k)
    assert len(results) == top_k

    for doc, score in results:
        assert doc.page_content != ""
        assert "source" in doc.metadata
        assert "page" in doc.metadata
        assert isinstance(score, (float, int))


def test_retrieval_empty_query(test_embedding_model):
    """Retrieving with an empty or whitespace query should return an empty list."""
    vs = load_vector_store(test_embedding_model, save_path=TEST_VECTOR_DIR)
    results = retrieve_relevant_chunks(vs, query="   ", top_k=4)
    assert results == []


def test_source_citation_extraction(test_embedding_model):
    """Verify source extraction deduplicates and converts pages to 1-based indexing."""
    vs = load_vector_store(test_embedding_model, save_path=TEST_VECTOR_DIR)
    results = retrieve_relevant_chunks(vs, query="Retrieval-Augmented Generation", top_k=4)
    docs = [doc for doc, _ in results]

    sources = format_sources(docs)
    assert len(sources) > 0

    for src in sources:
        assert src["filename"] == "sample_ai_notes.pdf"
        # 1-indexed page should be >= 1
        assert src["page"] >= 1
        assert "📄 sample_ai_notes.pdf — Page" in src["display_text"]

    md = get_sources_markdown(sources)
    assert "**Sources:**" in md
    assert "sample_ai_notes.pdf" in md


def test_debug_formatting(test_embedding_model):
    """Test developer debug formatting for retrieved chunks."""
    vs = load_vector_store(test_embedding_model, save_path=TEST_VECTOR_DIR)
    results = retrieve_relevant_chunks(vs, query="LangGraph workflows", top_k=2)
    debug_list = format_retrieved_chunks_debug(results)

    assert len(debug_list) == 2
    assert debug_list[0]["rank"] == 1
    assert "score" in debug_list[0]
    assert "content_preview" in debug_list[0]


def test_clear_vector_store():
    """Verify clearing the vector store removes persisted files."""
    cleared = clear_vector_store(TEST_VECTOR_DIR)
    assert cleared is True
    assert vector_store_exists(TEST_VECTOR_DIR) is False
    # Cleanup directory
    if TEST_VECTOR_DIR.exists():
        shutil.rmtree(TEST_VECTOR_DIR)
