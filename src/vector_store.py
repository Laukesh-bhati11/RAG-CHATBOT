"""
FAISS Vector Database Module for DocuMind AI.

WHAT IS FAISS?
--------------
FAISS (Facebook AI Similarity Search) is an open-source library specifically
optimized for high-efficiency similarity search and clustering of dense vectors.

Unlike traditional SQL databases that match exact strings, a vector database:
1. Stores floating-point embeddings alongside their document metadata (filename, page).
2. Uses mathematical indexing algorithms to rapidly locate the nearest neighbor
   vectors to a query vector in milliseconds.

PERSISTENCE AND EFFICIENCY:
---------------------------
Recomputing embeddings for large documents costs API tokens and time.
DocuMind AI creates the index once, saves it to disk (`vectorstore/faiss_index`),
and loads it instantly on subsequent queries or application restarts.
"""

import shutil
import logging
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from src.config import default_config

logger = logging.getLogger("documind.vectorstore")


def vector_store_exists(directory_path: Optional[Path | str] = None) -> bool:
    """
    Checks if a valid, serialized FAISS index already exists on disk.
    
    A valid LangChain FAISS directory typically contains:
    - index.faiss (the binary vector index)
    - index.pkl (the pickled document and metadata mappings)
    """
    target_dir = Path(directory_path or default_config.vectorstore_dir)
    faiss_file = target_dir / "index.faiss"
    pkl_file = target_dir / "index.pkl"
    return target_dir.exists() and faiss_file.exists() and pkl_file.exists()


def create_vector_store(
    chunks: List[Document],
    embedding_model: Embeddings,
    save_path: Optional[Path | str] = None,
) -> FAISS:
    """
    Creates a new FAISS vector database from document chunks and saves it locally.

    Parameters:
        chunks: List of Document chunks with metadata.
        embedding_model: Initialized Embeddings instance.
        save_path: Directory path where the index files should be persisted.

    Returns:
        FAISS vector store instance.
    """
    if not chunks:
        raise ValueError("Cannot create vector store: No document chunks provided.")

    target_dir = Path(save_path or default_config.vectorstore_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating embeddings and building FAISS index for {len(chunks)} chunks...")
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )

    # Persist the index and docstore mappings to disk
    vector_store.save_local(str(target_dir))
    logger.info(f"FAISS index successfully saved to: {target_dir}")
    print(f"[DocuMind VectorStore] Saved FAISS index with {len(chunks)} chunks to {target_dir}")

    return vector_store


def load_vector_store(
    embedding_model: Embeddings,
    save_path: Optional[Path | str] = None,
) -> Optional[FAISS]:
    """
    Loads an existing FAISS vector database from disk without re-embedding documents.

    Parameters:
        embedding_model: Embeddings instance matching the dimensions of the saved index.
        save_path: Directory path where index files are located.

    Returns:
        Loaded FAISS vector store, or None if the index does not exist.
    """
    target_dir = Path(save_path or default_config.vectorstore_dir)

    if not vector_store_exists(target_dir):
        logger.warning(f"No FAISS index found at {target_dir}.")
        return None

    try:
        logger.info(f"Loading existing FAISS index from: {target_dir}")
        # Note: allow_dangerous_deserialization=True is required by LangChain to load
        # the internal document metadata pickle file created by save_local.
        vector_store = FAISS.load_local(
            folder_path=str(target_dir),
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS vector store loaded successfully.")
        return vector_store
    except Exception as e:
        logger.error(f"Error loading FAISS vector store from {target_dir}: {e}", exc_info=True)
        return None


def add_documents_to_vector_store(
    vector_store: FAISS,
    new_chunks: List[Document],
    save_path: Optional[Path | str] = None,
) -> FAISS:
    """
    Appends new document chunks to an existing FAISS vector database and re-saves it.

    Parameters:
        vector_store: The active FAISS instance.
        new_chunks: List of new Document chunks to add.
        save_path: Directory path to persist the updated index.

    Returns:
        Updated FAISS vector store instance.
    """
    if not new_chunks:
        logger.warning("No new chunks provided to add to vector store.")
        return vector_store

    target_dir = Path(save_path or default_config.vectorstore_dir)
    logger.info(f"Adding {len(new_chunks)} new chunks to existing FAISS vector store...")
    
    vector_store.add_documents(new_chunks)
    vector_store.save_local(str(target_dir))
    logger.info(f"Updated FAISS vector store saved to: {target_dir}")
    
    return vector_store


def clear_vector_store(save_path: Optional[Path | str] = None) -> bool:
    """
    Deletes the FAISS index files from disk to reset the vector database.

    Parameters:
        save_path: Directory path where index files are located.

    Returns:
        True if successfully cleared or already empty, False on error.
    """
    target_dir = Path(save_path or default_config.vectorstore_dir)
    try:
        if target_dir.exists():
            shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cleared vector store directory: {target_dir}")
            return True
        return True
    except Exception as e:
        logger.error(f"Failed to clear vector store at {target_dir}: {e}")
        return False
