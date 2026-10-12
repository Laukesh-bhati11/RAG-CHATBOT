"""
Document Loader Module for DocuMind AI.

This module is responsible for:
1. Loading PDF files using LangChain's PyPDFLoader.
2. Extracting raw text page-by-page.
3. Preserving essential metadata (source filename and page number).
4. Gracefully handling corrupted or unreadable PDFs.
"""

import logging
from pathlib import Path
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

# Configure standard logger
logger = logging.getLogger("documind.loader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_single_pdf(file_path: Path | str) -> List[Document]:
    """
    Loads a single PDF document and extracts its pages into LangChain Document objects.

    Each page in the returned Document has:
    - page_content: The textual content of that page.
    - metadata: A dictionary containing:
        - 'source': Filename of the PDF.
        - 'page': Zero-based page index from the PDF.

    Parameters:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List of Document objects representing individual pages.
        Returns an empty list if the file cannot be loaded or is corrupted.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"Cannot load PDF: File does not exist at {path}")
        return []

    try:
        logger.info(f"Loading PDF document: {path.name}")
        loader = PyPDFLoader(str(path))
        raw_documents = loader.load()

        # Normalize and ensure clean metadata for downstream processing
        cleaned_documents = []
        for doc in raw_documents:
            # Standardize source to the base filename so UI citations are clean
            doc.metadata["source"] = path.name
            # Ensure page key exists (PyPDFLoader defaults to 0-based page)
            if "page" not in doc.metadata:
                doc.metadata["page"] = 0
            
            # Avoid indexing completely empty/whitespace-only pages
            if doc.page_content and doc.page_content.strip():
                cleaned_documents.append(doc)

        logger.info(f"Successfully loaded {len(cleaned_documents)} pages from '{path.name}'.")
        return cleaned_documents

    except Exception as e:
        logger.error(f"Error loading PDF '{path.name}': {e}", exc_info=True)
        # Returning empty list allows the caller to handle failures gracefully
        return []


def load_multiple_pdfs(file_paths: List[Path | str]) -> Tuple[List[Document], List[str]]:
    """
    Loads multiple PDF files, collecting all pages into a unified list.

    Parameters:
        file_paths: A list of file paths to load.

    Returns:
        Tuple:
        - List of all loaded Document pages across all valid PDFs.
        - List of error messages for any files that failed to load.
    """
    all_documents: List[Document] = []
    failed_files: List[str] = []

    for file_path in file_paths:
        path = Path(file_path)
        docs = load_single_pdf(path)
        if docs:
            all_documents.extend(docs)
        else:
            failed_files.append(f"Failed to load or parse '{path.name}'. File may be encrypted or corrupted.")

    logger.info(f"Batch loading complete. Loaded total {len(all_documents)} pages from {len(file_paths) - len(failed_files)} files.")
    return all_documents, failed_files
