"""
Embeddings Module for DocuMind AI.

This module provides local, open-source embeddings using Hugging Face / Sentence Transformers.
No external API calls, paid credits, or internet connection required after initial model download.

WHAT ARE LOCAL EMBEDDINGS?
---------------------------
An embedding model reads a piece of text (e.g., "Python is a programming language")
and translates it into a dense numerical vector (e.g., [0.012, -0.045, ...]).
Semantically similar texts produce vectors that point in nearly identical directions
in vector space (measured via Cosine Similarity or Euclidean Distance in FAISS).

PREFERRED MODEL:
----------------
- sentence-transformers/all-MiniLM-L6-v2
  - Fast, lightweight (runs efficiently on standard laptop CPUs).
  - 384 dimensions.
  - ~80MB in size.
"""

import logging
from typing import Optional
from langchain_core.embeddings import Embeddings
from src.config import default_config

logger = logging.getLogger("documind.embeddings")


def get_embedding_model(
    model_name: Optional[str] = None,
    provider: str = "huggingface",
) -> Embeddings:
    """
    Factory function to initialize and return a local embedding model instance.

    Supported Providers:
    - 'huggingface' (default): Uses local Hugging Face / Sentence Transformers.
    - 'ollama': Uses local Ollama embeddings (e.g. 'nomic-embed-text').
    - 'fake': Uses a deterministic dummy embedding model for unit testing.

    Parameters:
        model_name: Name of the model. Defaults to config embedding_model.
        provider: Provider identifier ('huggingface', 'ollama', or 'fake').

    Returns:
        A LangChain Embeddings instance.
    """
    chosen_provider = (provider or "huggingface").lower().strip()
    target_model = model_name or default_config.embedding_model

    if chosen_provider in ("huggingface", "local"):
        logger.info(f"Initializing local HuggingFace embeddings: {target_model}")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=target_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except ImportError:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(
                    model_name=target_model,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
            except Exception as e:
                logger.error(f"Failed to initialize HuggingFaceEmbeddings: {e}")
                raise RuntimeError(
                    f"Could not load Hugging Face embeddings ({target_model}). "
                    f"Ensure 'sentence-transformers' and 'langchain-huggingface' are installed. Error: {e}"
                )

    elif chosen_provider == "ollama":
        logger.info(f"Initializing local Ollama embeddings: {target_model}")
        try:
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(
                model=target_model,
                base_url=default_config.ollama_base_url,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ollama embeddings: {e}")
            raise RuntimeError(
                f"Could not connect to Ollama for embeddings. "
                f"Make sure Ollama is running at {default_config.ollama_base_url}. Error: {e}"
            )

    elif chosen_provider == "fake":
        logger.info("Initializing FakeEmbeddings for offline unit testing.")
        from langchain_community.embeddings import FakeEmbeddings
        # 384 matches all-MiniLM-L6-v2 vector dimension
        return FakeEmbeddings(size=384)

    else:
        raise ValueError(
            f"Unsupported embedding provider: '{provider}'. "
            "Supported local options are 'huggingface', 'ollama', and 'fake'."
        )
