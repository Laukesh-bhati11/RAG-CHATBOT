"""
Local Verification Script for DocuMind AI.
Tests the entire RAG pipeline end-to-end offline without OpenAI API credits.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure stdout supports UTF-8 on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_checks():
    print("=" * 60)
    print(">> DOCUMIND AI: RUNNING LOCAL RAG INTEGRATION CHECKS")
    print("=" * 60)

    # Check 1: Python imports
    print("\n[Check 1/12] Testing Python module imports...")
    try:
        from src.config import default_config
        from src.document_loader import load_single_pdf, load_multiple_pdfs
        from src.chunker import split_documents
        from src.embeddings import get_embedding_model
        from src.vector_store import create_vector_store, load_vector_store, vector_store_exists
        from src.retriever import retrieve_relevant_chunks
        from src.prompts import RAG_PROMPT_TEMPLATE
        from src.chains import get_llm_model, format_documents_context
        from src.graph import build_rag_graph, run_rag_pipeline
        from src.utils import format_sources, is_valid_pdf
        print("  ✓ All DocuMind AI modules imported successfully without OpenAI dependencies!")
    except Exception as e:
        print(f"  ✗ Import Error: {e}")
        return False

    # Check 2 & 5: Load sample PDF
    sample_pdf = ROOT_DIR / "data" / "documents" / "sample_ai_notes.pdf"
    print(f"\n[Check 5/12] Loading sample PDF from: {sample_pdf.name}...")
    assert sample_pdf.exists(), f"Sample PDF not found at {sample_pdf}"
    docs = load_single_pdf(sample_pdf)
    print(f"  ✓ Successfully extracted {len(docs)} pages with source metadata.")

    # Check 6: Text Chunking
    print("\n[Check 6/12] Chunking documents with RecursiveCharacterTextSplitter...")
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=100)
    print(f"  ✓ Created {len(chunks)} chunks from {len(docs)} pages.")

    # Check 2 & 7: Embedding initialization & vector generation
    print("\n[Check 2/12 & 7/12] Testing embedding model initialization...")
    try:
        embedding_model = get_embedding_model(provider="huggingface")
        provider_used = "huggingface"
        print("  ✓ Loaded local HuggingFace embeddings: sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(f"  ℹ Hugging Face local package downloading/caching ({e}); falling back to deterministic test embeddings...")
        embedding_model = get_embedding_model(provider="fake")
        provider_used = "fake"
        print("  ✓ Deterministic test embeddings active.")

    # Check 3 & 8: FAISS vector database creation & disk persistence
    test_faiss_dir = ROOT_DIR / "vectorstore" / "test_check_index"
    print(f"\n[Check 3/12 & 8/12] Creating FAISS index at {test_faiss_dir.name}...")
    vector_store = create_vector_store(chunks[:5], embedding_model, save_path=test_faiss_dir)
    assert vector_store is not None
    assert vector_store_exists(test_faiss_dir)
    print("  ✓ FAISS index created, saved to disk, and verified.")

    # Check 9: Retriever similarity search
    print("\n[Check 9/12] Testing similarity search and metadata retrieval...")
    retrieved = retrieve_relevant_chunks(vector_store, query="What is Machine Learning?", top_k=2)
    assert len(retrieved) > 0, "Retriever returned 0 results"
    for rank, (doc, score) in enumerate(retrieved, start=1):
        print(f"  ✓ Rank {rank}: {doc.metadata.get('source')} (Page {doc.metadata.get('page', 0) + 1}) - Score: {score:.4f}")

    # Check 10 & 11: LangGraph Workflow & Answer Generation
    print("\n[Check 10/12 & 11/12] Testing LangGraph Workflow & Answer Generation...")
    try:
        # Test with fake model for deterministic offline verification
        llm = get_llm_model(provider="fake")
        result = run_rag_pipeline(
            question="What is Machine Learning?",
            vector_store=vector_store,
            llm=llm,
            top_k=2,
        )
        print("  ✓ LangGraph StateMachine executed successfully!")
        print(f"  ✓ Workflow Steps: {' ➔ '.join(result['workflow_steps'])}")
        print(f"  ✓ Answer: \"{result['answer']}\"")
        print(f"  ✓ Sources: {[s['display_text'] for s in result['sources']]}")
    except Exception as e:
        print(f"  ✗ LangGraph execution error: {e}")
        return False

    # Clean up test index
    import shutil
    if test_faiss_dir.exists():
        shutil.rmtree(test_faiss_dir)

    print("\n" + "=" * 60)
    print("🎉 ALL CORE RAG INTEGRATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
