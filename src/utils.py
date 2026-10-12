"""
Utility Functions for DocuMind AI.

This module provides helper utilities for:
1. Validating PDF documents safely.
2. Saving uploaded files from Streamlit.
3. Extracting and deduplicating source citations (Document name + Page number).
4. Formatting debug information for the Streamlit UI.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document


def is_valid_pdf(file_path: Path | str) -> Tuple[bool, str]:
    """
    Validates whether a given file path is a readable, non-empty PDF file.

    Parameters:
        file_path: Path to the PDF file.

    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    path = Path(file_path)
    if not path.exists():
        return False, f"File not found: {path.name}"
    
    if path.stat().st_size == 0:
        return False, f"File is empty (0 bytes): {path.name}"
        
    try:
        # Check standard PDF file header (%PDF-)
        with open(path, "rb") as f:
            header = f.read(5)
            if not header.startswith(b"%PDF"):
                return False, f"Invalid file format: '{path.name}' does not appear to be a valid PDF."
    except Exception as e:
        return False, f"Failed to read file '{path.name}': {str(e)}"

    return True, ""


def save_uploaded_file(uploaded_file, target_dir: Path) -> Path:
    """
    Saves an uploaded Streamlit file to the target directory.

    Parameters:
        uploaded_file: Streamlit UploadedFile object.
        target_dir: Path where file should be saved.

    Returns:
        Path to the saved file.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    destination_path = target_dir / uploaded_file.name

    with open(destination_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return destination_path


def format_sources(documents: List[Document]) -> List[Dict[str, Any]]:
    """
    Extracts, deduplicates, and formats source metadata from retrieved document chunks.

    Parameters:
        documents: List of LangChain Document objects retrieved from FAISS.

    Returns:
        List of unique citation dictionaries, each containing:
        - 'filename': e.g., 'Python Notes.pdf'
        - 'page': int (1-based page number)
        - 'display_text': formatted string e.g. '📄 Python Notes.pdf — Page 3'
        - 'snippet': preview of chunk text
    """
    seen_citations = set()
    formatted_sources = []

    for doc in documents:
        # Extract source metadata safely
        source_path = doc.metadata.get("source", "Unknown Document")
        # Extract just the filename (basename) rather than the full filesystem path
        filename = Path(source_path).name if source_path != "Unknown Document" else "Unknown Document"
        
        # In PyPDFLoader, pages are 0-indexed. We convert to 1-indexed for human readability.
        raw_page = doc.metadata.get("page", 0)
        page_number = int(raw_page) + 1 if isinstance(raw_page, (int, str)) and str(raw_page).isdigit() else 1
        
        citation_key = (filename, page_number)
        if citation_key not in seen_citations:
            seen_citations.add(citation_key)
            snippet = doc.page_content.strip().replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."
                
            formatted_sources.append({
                "filename": filename,
                "page": page_number,
                "display_text": f"📄 {filename} — Page {page_number}",
                "snippet": snippet
            })

    # Sort sources logically by filename, then page number
    formatted_sources.sort(key=lambda item: (item["filename"], item["page"]))
    return formatted_sources


def get_sources_markdown(sources: List[Dict[str, Any]]) -> str:
    """
    Renders formatted sources into clean Markdown for the chatbot interface.
    """
    if not sources:
        return ""

    lines = ["\n\n**Sources:**"]
    for source in sources:
        lines.append(f"- {source['display_text']}")
    return "\n".join(lines)
