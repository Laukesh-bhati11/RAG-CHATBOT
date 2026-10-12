"""
Chains Module for DocuMind AI.

This module builds modern LCEL (LangChain Expression Language) pipelines for:
1. Local Ollama LLM initialization (e.g., llama3.2:3b, qwen2.5:3b).
2. Document context formatting with page citations.
3. Conversational follow-up question contextualization.
4. Grounded answer generation strictly from context.
5. Relevance evaluation to guard against hallucinations.
"""

import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from src.config import default_config
from src.prompts import (
    RAG_PROMPT_TEMPLATE,
    CONTEXTUALIZE_PROMPT_TEMPLATE,
    RELEVANCE_PROMPT_TEMPLATE,
)

logger = logging.getLogger("documind.chains")


def get_llm_model(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    base_url: Optional[str] = None,
    provider: str = "ollama",
) -> BaseChatModel:
    """
    Factory function to initialize and configure a local LLM instance via Ollama.

    Parameters:
        model_name: Name of the Ollama model (e.g. 'llama3.2:3b', 'qwen2.5:3b').
                    Defaults to config.ollama_model.
        temperature: Sampling temperature (0.0 for deterministic factual responses).
        base_url: URL where Ollama is running (defaults to 'http://localhost:11434').
        provider: Provider identifier ('ollama' or 'fake').

    Returns:
        BaseChatModel instance.
    """
    chosen_provider = (provider or "ollama").lower().strip()
    target_model = model_name or default_config.ollama_model
    target_base_url = base_url or default_config.ollama_base_url

    if chosen_provider in ("ollama", "local"):
        logger.info(f"Initializing ChatOllama model: '{target_model}' at '{target_base_url}' (temp={temperature})")
        try:
            return ChatOllama(
                model=target_model,
                temperature=temperature,
                base_url=target_base_url,
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatOllama: {e}")
            raise RuntimeError(
                f"Could not connect to Ollama at {target_base_url}.\n"
                f"Please ensure Ollama is running and '{target_model}' is installed.\n"
                f"Run: ollama run {target_model}"
            )

    elif chosen_provider == "fake":
        logger.info("Initializing FakeListChatModel for offline unit testing.")
        from langchain_community.chat_models.fake import FakeListChatModel
        return FakeListChatModel(responses=["This is a grounded test response."])

    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'. Supported options: 'ollama', 'fake'.")


def format_documents_context(documents: List[Document]) -> str:
    """
    Combines retrieved Document chunks into a structured, labeled text block for the LLM prompt.

    Parameters:
        documents: List of retrieved Document chunks.

    Returns:
        Formatted context string with clear document & page labels.
    """
    if not documents:
        return "No relevant document context found."

    formatted_chunks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown Document")
        raw_page = doc.metadata.get("page", 0)
        page_num = int(raw_page) + 1 if isinstance(raw_page, (int, str)) and str(raw_page).isdigit() else 1
        
        header = f"--- [Document {i}: {source} | Page {page_num}] ---"
        body = doc.page_content.strip()
        formatted_chunks.append(f"{header}\n{body}")

    return "\n\n".join(formatted_chunks)


def create_qa_chain(llm: BaseChatModel):
    """
    Builds the question-answering LCEL chain.
    Flow: Prompt Template -> LLM -> String Output Parser.
    """
    return RAG_PROMPT_TEMPLATE | llm | StrOutputParser()


def create_contextualize_chain(llm: BaseChatModel):
    """
    Builds the question contextualization chain for multi-turn conversations.
    Takes recent chat history and rewrites the user's latest query into a standalone question.
    """
    return CONTEXTUALIZE_PROMPT_TEMPLATE | llm | StrOutputParser()


def create_relevance_grader_chain(llm: BaseChatModel):
    """
    Builds a lightweight relevance grading chain to verify if retrieved context matches the question.
    """
    return RELEVANCE_PROMPT_TEMPLATE | llm | StrOutputParser()
