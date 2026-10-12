"""
Retriever Module for DocuMind AI.

HOW RETRIEVAL WORKS IN RAG:
---------------------------
1. User enters a query: "What is Normalization?"
2. The query is transformed into a dense embedding vector using the embedding model.
3. FAISS performs an approximate nearest neighbors (ANN) search comparing the query
   vector against all stored chunk vectors.
4. The Top-K most similar chunks (by L2 distance or cosine similarity) are returned,
   complete with their text content, filename, and page numbers.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.config import default_config

logger = logging.getLogger("documind.retriever")


def retrieve_relevant_chunks(
    vector_store: FAISS,
    query: str,
    top_k: int = 4,
) -> List[Tuple[Document, float]]:
    """
    Retrieves the Top-K most relevant document chunks from the FAISS vector database
    along with their similarity distance scores.

    Parameters:
        vector_store: The loaded FAISS vector store.
        query: The user's question or search query.
        top_k: Number of most relevant chunks to retrieve (default: 4).

    Returns:
        List of Tuples: (Document, score)
        In FAISS with Euclidean (L2) distance, lower scores indicate closer distance (higher similarity).
    """
    clean_query = query.strip() if query else ""
    if not clean_query:
        logger.warning("Empty query provided to retriever.")
        return []

    if vector_store is None:
        logger.error("Vector store is not initialized or loaded.")
        return []

    effective_k = max(1, top_k)
    logger.info(f"Retrieving top {effective_k} chunks for query: '{clean_query[:50]}...'")

    try:
        raw_results = vector_store.similarity_search_with_score(
            query=clean_query,
            k=effective_k,
        )
        # Ensure scores are standard Python floats (FAISS returns np.float32)
        results = [(doc, float(score)) for doc, score in raw_results]
        logger.info(f"Retrieved {len(results)} chunks from FAISS.")
        return results
    except Exception as e:
        logger.error(f"Error during FAISS similarity search: {e}", exc_info=True)
        return []


def format_retrieved_chunks_debug(results: List[Tuple[Document, float]]) -> List[Dict[str, Any]]:
    """
    Formats retrieved chunks and their scores into a structured dictionary list
    for the Streamlit Debug / Developer mode.

    Parameters:
        results: List of (Document, score) tuples.

    Returns:
        List of dictionaries with debug metadata.
    """
    debug_info = []
    for rank, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", 0)
        # Convert 0-indexed to 1-indexed for human display
        display_page = int(page) + 1 if isinstance(page, (int, str)) and str(page).isdigit() else 1
        
        debug_info.append({
            "rank": rank,
            "source": source,
            "page": display_page,
            "score": round(float(score), 4),
            "content_preview": doc.page_content.strip()[:250] + ("..." if len(doc.page_content) > 250 else ""),
            "full_content": doc.page_content.strip(),
            "metadata": doc.metadata,
        })
    return debug_info
