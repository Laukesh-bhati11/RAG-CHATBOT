"""
DocuMind AI: Intelligent Document Q&A System
Streamlit Web Application (100% Free / Local RAG Architecture)

This file implements the complete user interface and frontend orchestration:
1. Multi-PDF file upload and management.
2. Step-by-step document indexing pipeline with visual progress indicators.
3. Conversational chatbot with session memory.
4. LangGraph RAG workflow execution with local Ollama LLM and Hugging Face embeddings.
5. Accurate source citations (document name + page number).
6. Interactive Developer/Debug mode showing retrieval details and execution traces.
7. Clear, actionable error messages for Ollama and document states.
"""

import sys
import os
import urllib.request
from pathlib import Path
from typing import Tuple

# Ensure project root is in sys.path so imports work from anywhere
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from src.config import default_config
from src.document_loader import load_multiple_pdfs
from src.chunker import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import (
    create_vector_store,
    load_vector_store,
    vector_store_exists,
    clear_vector_store,
)
from src.chains import get_llm_model
from src.retriever import format_retrieved_chunks_debug
from src.graph import run_rag_pipeline
from src.utils import is_valid_pdf, save_uploaded_file, format_sources


# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DocuMind AI — Intelligent Document Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern blue/purple theme, rounded cards, and clean typography
st.markdown("""
<style>
    /* Main container and font settings */
    .main {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header card styling */
    .header-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 50%, #6366F1 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.2);
    }
    .header-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .header-subtitle {
        font-size: 15px;
        opacity: 0.92;
        font-weight: 400;
    }
    
    /* Source citation cards */
    .source-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 10px;
        font-size: 13px;
        color: #334155;
    }
    
    /* Debug badge */
    .debug-badge {
        background-color: #EEF2FF;
        border: 1px solid #C7D2FE;
        color: #3730A3;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-family: monospace;
        display: inline-block;
        margin-bottom: 8px;
    }
    
    /* Status pills */
    .status-pill {
        display: block;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .status-ready {
        background-color: #DEF7EC;
        color: #03543F;
        border: 1px solid #BCF0DA;
    }
    .status-empty {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .status-offline {
        background-color: #FDE8E8;
        color: #9B1C1C;
        border: 1px solid #F8B4B4;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Helper: Service Health Check
# -----------------------------------------------------------------------------
def check_ollama_service(base_url: str) -> Tuple[bool, str]:
    """
    Checks whether the local Ollama service is active and responsive.
    """
    try:
        req = urllib.request.Request(base_url, headers={"User-Agent": "DocuMind-AI"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                return True, "Running"
    except Exception:
        pass
    return False, "Not Running"


# -----------------------------------------------------------------------------
# 3. Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "total_chunks_indexed" not in st.session_state:
    st.session_state.total_chunks_indexed = 0

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False


# -----------------------------------------------------------------------------
# 4. Sidebar: Document Management & Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Control Panel")
    
    # --- Document Upload Section ---
    st.markdown("### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple PDF documents to build your local knowledge base.",
    )

    if uploaded_files:
        st.caption(f"Selected {len(uploaded_files)} file(s):")
        for f in uploaded_files:
            st.markdown(f"- 📄 `{f.name}` ({round(f.size / 1024, 1)} KB)")

    col_proc, col_clear = st.columns(2)
    process_clicked = col_proc.button("⚡ Process", type="primary", use_container_width=True)
    clear_clicked = col_clear.button("🗑️ Clear", use_container_width=True)

    st.divider()

    # --- Local Settings ---
    st.markdown("### ⚙️ Settings")
    
    available_models = [
        "llama3.2:3b",
        "qwen2.5:3b",
        "llama3.2:1b",
        "mistral:7b",
        "phi3:latest",
    ]
    current_default = default_config.ollama_model
    selected_idx = available_models.index(current_default) if current_default in available_models else 0

    ollama_model_name = st.selectbox(
        "Ollama Model",
        options=available_models,
        index=selected_idx,
        help="Local LLM model running via Ollama on your machine.",
    )

    top_k = st.slider(
        "Top-K Retrieval",
        min_value=1,
        max_value=8,
        value=default_config.top_k,
        step=1,
        help="Number of relevant document chunks retrieved per question.",
    )

    chunk_size = st.slider(
        "Chunk Size (characters)",
        min_value=300,
        max_value=2000,
        value=default_config.chunk_size,
        step=100,
        help="Target length of each text chunk.",
    )

    chunk_overlap = st.slider(
        "Chunk Overlap (characters)",
        min_value=50,
        max_value=500,
        value=default_config.chunk_overlap,
        step=25,
        help="Characters shared between adjacent chunks to maintain context continuity.",
    )

    st.divider()

    # --- Debug Mode Toggle ---
    debug_mode = st.toggle("🔍 Developer / Debug Mode", value=st.session_state.debug_mode)
    st.session_state.debug_mode = debug_mode

    st.divider()

    # --- System Status ---
    st.markdown("### 📊 System Status")
    
    # 1. Ollama connection status
    is_ollama_online, _ = check_ollama_service(default_config.ollama_base_url)
    if is_ollama_online:
        st.markdown(
            f'<div class="status-pill status-ready">🟢 Ollama: Connected ({ollama_model_name})</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill status-offline">🔴 Ollama: Not Running (localhost:11434)</div>',
            unsafe_allow_html=True,
        )
        st.caption("Start Ollama in terminal:\n`ollama run llama3.2:3b`")

    # 2. Vector Store status
    is_index_ready = vector_store_exists()
    if is_index_ready:
        st.markdown(
            f'<div class="status-pill status-ready">🟢 Vector Store: Active ({st.session_state.total_chunks_indexed} chunks)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill status-empty">🟡 Vector Store: Empty</div>',
            unsafe_allow_html=True,
        )

    # 3. Embedding model info
    st.caption(f"🧠 Embeddings: `{default_config.embedding_model}` (Local CPU)")


# -----------------------------------------------------------------------------
# 5. Action Handlers: Process & Clear Documents
# -----------------------------------------------------------------------------
# Handle "Clear Vector Store"
if clear_clicked:
    clear_vector_store()
    st.session_state.processed_files.clear()
    st.session_state.total_chunks_indexed = 0
    st.session_state.messages.clear()
    st.sidebar.success("Vector store and chat history cleared successfully.")
    st.rerun()

# Handle "Process Documents"
if process_clicked:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF file first.")
    else:
        progress_container = st.sidebar.container()
        progress_bar = progress_container.progress(0, text="Starting processing...")
        
        try:
            # Step 1: Save uploaded files safely to disk
            progress_bar.progress(15, text="1/5 Saving files to disk...")
            saved_paths = []
            for uploaded_file in uploaded_files:
                saved_path = save_uploaded_file(uploaded_file, default_config.documents_dir)
                is_valid, err_msg = is_valid_pdf(saved_path)
                if not is_valid:
                    st.sidebar.error(f"Invalid PDF: {err_msg}")
                    raise ValueError(err_msg)
                saved_paths.append(saved_path)

            # Step 2: Read PDF documents and extract text
            progress_bar.progress(35, text="2/5 Reading PDF documents...")
            loaded_docs, failed = load_multiple_pdfs(saved_paths)
            if failed:
                for f_err in failed:
                    st.sidebar.warning(f_err)

            if not loaded_docs:
                raise ValueError("No readable text found in the provided PDF files.")

            # Step 3: Split into chunks
            progress_bar.progress(55, text=f"3/5 Splitting into chunks (size={chunk_size})...")
            chunks = split_documents(
                documents=loaded_docs,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            # Step 4: Initialize local Hugging Face embeddings
            progress_bar.progress(75, text=f"4/5 Initializing local embeddings ({default_config.embedding_model})...")
            embedding_model = get_embedding_model(
                model_name=default_config.embedding_model,
                provider="huggingface",
            )

            # Step 5: Build FAISS vector database
            progress_bar.progress(90, text="5/5 Building FAISS vector database...")
            create_vector_store(
                chunks=chunks,
                embedding_model=embedding_model,
                save_path=default_config.vectorstore_dir,
            )

            # Update session state
            st.session_state.total_chunks_indexed = len(chunks)
            for f in uploaded_files:
                st.session_state.processed_files.add(f.name)

            progress_bar.progress(100, text="Done!")
            st.sidebar.success(f"Successfully indexed {len(chunks)} chunks from {len(loaded_docs)} pages!")
            st.rerun()

        except Exception as e:
            st.sidebar.error(f"Processing failed: {str(e)}")


# -----------------------------------------------------------------------------
# 6. Main Area: Header & Onboarding
# -----------------------------------------------------------------------------
st.markdown("""
<div class="header-card">
    <div class="header-title">🤖 DocuMind AI</div>
    <div class="header-subtitle">Intelligent Document Q&A System — 100% Free & Local RAG Architecture</div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.messages:
    st.info(
        "👋 **Welcome to DocuMind AI!**\n\n"
        "This project runs **completely locally** on your machine with **zero API costs**, complete privacy, and no expiration.\n\n"
        "**How to use:**\n"
        "1. **Upload PDF**: Select one or multiple PDF documents via the sidebar on the left.\n"
        "2. **Process Documents**: Click `⚡ Process` to extract text, compute local Hugging Face embeddings, and store them in FAISS.\n"
        "3. **Ask Questions**: Ask any question in the chat below. The local Ollama LLM will synthesize an answer grounded strictly in your documents with exact page citations!\n\n"
        "*Tip: Make sure Ollama is running in your terminal (`ollama run llama3.2:3b`).*"
    )


# -----------------------------------------------------------------------------
# 7. Chat History Display
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Render source citations if available
        if msg.get("sources"):
            with st.expander("📚 View Source Citations", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"**{src['display_text']}**")
                    if src.get("snippet"):
                        st.caption(f"*Excerpt:* \"{src['snippet']}\"")
        
        # Render developer debug info if enabled
        if debug_mode and msg.get("debug"):
            with st.expander("🛠️ Developer / Debug Information", expanded=False):
                debug_data = msg["debug"]
                st.markdown(f"**Workflow Trace:** `{' ➔ '.join(debug_data.get('workflow_steps', []))}`")
                st.markdown(f"**Standalone Query (Rewritten):** `{debug_data.get('standalone_question', 'N/A')}`")
                
                retrieved_chunks = debug_data.get("chunks", [])
                st.markdown(f"**Retrieved Chunks ({len(retrieved_chunks)}):**")
                for item in retrieved_chunks:
                    st.markdown(f"- **Rank {item['rank']}** (Score: `{item['score']}` | {item['source']} — Page {item['page']})")
                    st.caption(f"\"{item['content_preview']}\"")


# -----------------------------------------------------------------------------
# 8. User Input & Query Processing
# -----------------------------------------------------------------------------
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    cleaned_query = user_query.strip()
    
    if not cleaned_query:
        st.warning("Please enter a valid, non-empty question.")
    elif not vector_store_exists():
        st.error(
            "Vector database has not been created yet.\n\n"
            "Please upload and process at least one document before asking a question."
        )
    elif not is_ollama_online:
        st.error(
            "Ollama is not running.\n\n"
            "Please start Ollama and make sure the selected model is installed.\n\n"
            f"Example in terminal:\n```bash\nollama run {ollama_model_name}\n```"
        )
    else:
        # Display user message immediately
        st.session_state.messages.append({"role": "user", "content": cleaned_query})
        with st.chat_message("user"):
            st.markdown(cleaned_query)

        # Run RAG Pipeline with LangGraph
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context and synthesizing answer locally..."):
                try:
                    # 1. Load active vector store with local Hugging Face embeddings
                    embedding_model = get_embedding_model(
                        model_name=default_config.embedding_model,
                        provider="huggingface",
                    )
                    vector_store = load_vector_store(
                        embedding_model=embedding_model,
                        save_path=default_config.vectorstore_dir,
                    )
                    
                    if vector_store is None:
                        raise RuntimeError("Failed to load FAISS vector store from disk. Please re-process your documents.")

                    # 2. Initialize local Ollama LLM
                    llm = get_llm_model(
                        model_name=ollama_model_name,
                        temperature=0.0,
                        base_url=default_config.ollama_base_url,
                        provider="ollama",
                    )

                    # 3. Format recent conversation history for memory
                    chat_history = []
                    for past_msg in st.session_state.messages[:-1]:  # Exclude current question
                        if past_msg["role"] == "user":
                            chat_history.append(HumanMessage(content=past_msg["content"]))
                        elif past_msg["role"] == "assistant":
                            chat_history.append(AIMessage(content=past_msg["content"]))
                    
                    # Limit memory to last 6 turns to avoid context overflow
                    chat_history = chat_history[-6:]

                    # 4. Run LangGraph Workflow
                    result = run_rag_pipeline(
                        question=cleaned_query,
                        vector_store=vector_store,
                        llm=llm,
                        chat_history=chat_history,
                        top_k=top_k,
                        max_retries=2,
                    )

                    answer = result["answer"]
                    sources = result["sources"]
                    workflow_steps = result["workflow_steps"]
                    standalone_query = result["standalone_question"]
                    
                    # Extract debug data
                    raw_docs = result.get("documents", [])
                    raw_scores = result.get("scores", [])
                    doc_score_pairs = list(zip(raw_docs, raw_scores))
                    debug_chunks = format_retrieved_chunks_debug(doc_score_pairs)

                    debug_payload = {
                        "workflow_steps": workflow_steps,
                        "standalone_question": standalone_query,
                        "chunks": debug_chunks,
                    }

                    # Display Answer
                    st.markdown(answer)

                    # Display Sources
                    if sources:
                        st.markdown("**Sources:**")
                        for src in sources:
                            st.markdown(f"- {src['display_text']}")
                        
                        with st.expander("📖 View Excerpts from Sources", expanded=False):
                            for src in sources:
                                st.markdown(f"**{src['display_text']}**")
                                if src.get("snippet"):
                                    st.caption(f"*Excerpt:* \"{src['snippet']}\"")

                    # Display Debug Information if enabled
                    if debug_mode:
                        with st.expander("🛠️ Developer / Debug Information", expanded=True):
                            st.markdown(f"**Workflow Trace:** `{' ➔ '.join(workflow_steps)}`")
                            st.markdown(f"**Standalone Query (Rewritten):** `{standalone_query}`")
                            st.markdown(f"**Retrieved Chunks ({len(debug_chunks)}):**")
                            for item in debug_chunks:
                                st.markdown(f"- **Rank {item['rank']}** (Score: `{item['score']}` | {item['source']} — Page {item['page']})")
                                st.caption(f"\"{item['content_preview']}\"")

                    # Save to session history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "debug": debug_payload,
                    })

                except Exception as e:
                    error_message = (
                        f"⚠️ An error occurred while generating the answer: {str(e)}\n\n"
                        f"If using Ollama, ensure the model is installed:\n"
                        f"`ollama pull {ollama_model_name}`"
                    )
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                        "debug": {},
                    })
