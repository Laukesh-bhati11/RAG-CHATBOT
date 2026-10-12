"""
Text Chunking Module for DocuMind AI.

Why is Chunking Essential in RAG?
Large Language Models have context limits, and embedding entire documents at once
causes semantic dilution (the main idea gets lost in hundreds of pages).
By splitting long texts into smaller, focused chunks (e.g., 1000 characters) with
slight overlap (e.g., 200 characters), we:
1. Preserve specific context within semantic boundaries.
2. Enable precise retrieval of the exact paragraph answering a question.
3. Allow accurate page-level citations.

We use RecursiveCharacterTextSplitter because it splits recursively by:
- Paragraphs ("\n\n") first,
- Then lines ("\n"),
- Then spaces (" "),
- And finally individual characters if necessary.
This ensures semantic paragraphs are kept intact whenever possible.
"""

import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("documind.chunker")


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Splits a list of Document pages into smaller, overlapping chunks while preserving metadata.

    Parameters:
        documents: List of Document objects loaded from PDF files.
        chunk_size: Target character length per chunk (default: 1000).
        chunk_overlap: Number of characters shared between adjacent chunks (default: 200).
                       Overlap prevents thoughts from being cut in half at chunk boundaries.

    Returns:
        List of smaller Document chunk objects with preserved and enriched metadata.
    """
    if not documents:
        logger.warning("No documents provided to chunker.")
        return []

    # Configure the recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    # Split the documents
    chunks = text_splitter.split_documents(documents)

    # Enrich metadata with unique chunk identifiers
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        # Ensure source and page remain intact
        chunk.metadata.setdefault("source", "Unknown Document")
        chunk.metadata.setdefault("page", 0)

    # Log statistics as requested
    logger.info(f"Loaded pages: {len(documents)}")
    logger.info(f"Created chunks: {len(chunks)}")
    print(f"[DocuMind Chunker] Loaded pages: {len(documents)} | Created chunks: {len(chunks)}")

    return chunks
