"""
Configuration Module for DocuMind AI.

This module centralizes all system settings, paths, and environment variables.
It uses python-dotenv to load local settings (Ollama model, embeddings, chunk sizes)
from a .env file rather than hardcoding them in source code.
No paid API keys or cloud credentials required!
"""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Base Directories
# -----------------------------------------------------------------------------
# ROOT_DIR points to the top-level project folder (e.g., c:\Users\DELL\Desktop\RAG)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file located at project root
ENV_FILE = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE)

# Data and Vector Store Directories
DATA_DIR = ROOT_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTORSTORE_DIR = ROOT_DIR / "vectorstore" / "faiss_index"

# Ensure essential directories exist on disk
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Application Configuration Class
# -----------------------------------------------------------------------------
@dataclass
class AppConfig:
    """
    Holds all configurable parameters for DocuMind AI (100% Free / Local RAG).
    
    Why use a class?
    1. Keeps parameters organized in one place.
    2. Allows Streamlit to dynamically override settings (e.g., Ollama model, sliders)
       without changing global environment variables.
    """
    # Local LLM (Ollama) Settings
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Local Embedding Model (Hugging Face / Sentence Transformers)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Document Splitting Settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Retrieval Settings
    top_k: int = int(os.getenv("TOP_K", "4"))
    
    # LangGraph Workflow Settings
    max_retrieval_retries: int = 2
    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.3"))
    
    # Directory paths
    documents_dir: Path = DOCUMENTS_DIR
    vectorstore_dir: Path = VECTORSTORE_DIR


# Default singleton configuration instance
default_config = AppConfig()
