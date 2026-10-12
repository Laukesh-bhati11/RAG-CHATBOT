"""
LangGraph Workflow Module for DocuMind AI.

WHAT IS LANGGRAPH?
------------------
LangGraph is a library for building stateful, multi-actor applications with LLMs.
Instead of rigid linear chains (Input -> Prompt -> LLM -> Output), LangGraph models
workflows as cyclic or branching graphs:

1. STATE: A central shared data dictionary passed from node to node.
2. NODES: Python functions that receive the current state, perform a unit of work
   (e.g., retrieval, evaluation, generation), and return an updated state.
3. EDGES: Connections defining the execution path from one node to the next.
4. CONDITIONAL EDGES: Decision routers that choose the next node dynamically
   based on the current state (e.g., if retrieved docs are relevant -> generate;
   otherwise -> fallback).

OUR WORKFLOW GRAPH:
-------------------
      [ START ]
          │
          ▼
     [ RETRIEVE ]  <─── (Retry if query reformulated and retry_count < MAX)
          │
          ▼
   [ CHECK_RELEVANCE ]
          │
    Is Relevant?
    ├── YES ─────────────────► [ GENERATE ] ──► [ END ]
    └── NO (Retries exceeded) ► [ FALLBACK ] ──► [ END ]
"""

import logging
from typing import List, Dict, Any, TypedDict, Optional
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END

from src.retriever import retrieve_relevant_chunks
from src.chains import (
    format_documents_context,
    create_qa_chain,
    create_contextualize_chain,
    create_relevance_grader_chain,
)
from src.utils import format_sources

logger = logging.getLogger("documind.graph")


# -----------------------------------------------------------------------------
# 1. Graph State Definition
# -----------------------------------------------------------------------------
class RAGState(TypedDict):
    """
    The shared state dictionary carried across all nodes in the LangGraph workflow.
    """
    question: str                         # The active user question
    standalone_question: str              # Rewritten question taking chat history into account
    chat_history: List[BaseMessage]       # Prior conversation turns for contextual understanding
    documents: List[Document]             # Retrieved document chunks from FAISS
    scores: List[float]                   # Similarity distances for retrieved chunks
    answer: str                           # Final synthesized answer
    sources: List[Dict[str, Any]]         # Extracted, deduplicated citations
    retrieval_status: str                 # 'success', 'empty', or 'irrelevant'
    retry_count: int                      # Current number of retrieval retries
    workflow_steps: List[str]             # Trace of visited nodes for debugging


# -----------------------------------------------------------------------------
# 2. Graph Builder Function
# -----------------------------------------------------------------------------
def build_rag_graph(
    vector_store: FAISS,
    llm: BaseChatModel,
    top_k: int = 4,
    max_retries: int = 2,
):
    """
    Constructs and compiles the LangGraph RAG workflow.

    Parameters:
        vector_store: The active FAISS vector database.
        llm: Configured BaseChatModel instance.
        top_k: Number of chunks to retrieve per search.
        max_retries: Maximum number of retrieval retry attempts.

    Returns:
        A compiled LangGraph runnable application.
    """
    # Initialize component chains
    qa_chain = create_qa_chain(llm)
    contextualize_chain = create_contextualize_chain(llm)
    relevance_grader = create_relevance_grader_chain(llm)

    # -------------------------------------------------------------------------
    # Node 1: Contextualize Query (Resolves multi-turn references)
    # -------------------------------------------------------------------------
    def contextualize_node(state: RAGState) -> Dict[str, Any]:
        """
        If there is conversation history, rewrites ambiguous follow-up questions
        (e.g., "Why is it important?") into a standalone question.
        """
        history = state.get("chat_history", [])
        original_question = state["question"]
        steps = list(state.get("workflow_steps", []))
        steps.append("Contextualize Query")

        if history:
            logger.info("Chat history detected. Contextualizing user query...")
            try:
                standalone = contextualize_chain.invoke({
                    "chat_history": history,
                    "question": original_question,
                })
                standalone = standalone.strip()
                logger.info(f"Contextualized Query: '{standalone}'")
                return {
                    "standalone_question": standalone,
                    "workflow_steps": steps,
                }
            except Exception as e:
                logger.warning(f"Contextualization failed, fallback to original query: {e}")
                return {
                    "standalone_question": original_question,
                    "workflow_steps": steps,
                }

        # If no chat history, the original question is already standalone
        return {
            "standalone_question": original_question,
            "workflow_steps": steps,
        }

    # -------------------------------------------------------------------------
    # Node 2: Retrieve Documents
    # -------------------------------------------------------------------------
    def retrieve_node(state: RAGState) -> Dict[str, Any]:
        """
        Queries the FAISS vector database to retrieve the top-K chunks
        matching the standalone question.
        """
        query = state.get("standalone_question") or state["question"]
        steps = list(state.get("workflow_steps", []))
        steps.append(f"Retrieve Documents (Top {top_k})")
        logger.info(f"[LangGraph Node: Retrieve] Searching for: '{query}'")

        results = retrieve_relevant_chunks(vector_store, query=query, top_k=top_k)

        if not results:
            logger.warning("[LangGraph Node: Retrieve] No documents found in vector store.")
            return {
                "documents": [],
                "scores": [],
                "retrieval_status": "empty",
                "workflow_steps": steps,
            }

        docs = [doc for doc, score in results]
        scores = [score for doc, score in results]

        return {
            "documents": docs,
            "scores": scores,
            "retrieval_status": "success",
            "workflow_steps": steps,
        }

    # -------------------------------------------------------------------------
    # Node 3: Check Relevance
    # -------------------------------------------------------------------------
    def check_relevance_node(state: RAGState) -> Dict[str, Any]:
        """
        Evaluates whether the retrieved chunks appear relevant to the question.
        Uses a lightweight check to prevent hallucinating over unrelated text.
        """
        docs = state.get("documents", [])
        question = state.get("standalone_question") or state["question"]
        steps = list(state.get("workflow_steps", []))
        steps.append("Check Relevance")

        if not docs:
            return {
                "retrieval_status": "empty",
                "workflow_steps": steps,
            }

        context_preview = format_documents_context(docs[:2])
        try:
            grade = relevance_grader.invoke({
                "context": context_preview,
                "question": question,
            }).strip().lower()

            logger.info(f"[LangGraph Node: Check Relevance] Grader decision: '{grade}'")
            if "yes" in grade:
                return {
                    "retrieval_status": "relevant",
                    "workflow_steps": steps,
                }
            else:
                return {
                    "retrieval_status": "irrelevant",
                    "retry_count": state.get("retry_count", 0) + 1,
                    "workflow_steps": steps,
                }
        except Exception as e:
            logger.warning(f"Relevance grading error ({e}). Defaulting to 'relevant' to proceed with grounded generation.")
            return {
                "retrieval_status": "relevant",
                "workflow_steps": steps,
            }

    # -------------------------------------------------------------------------
    # Node 4: Generate Grounded Answer
    # -------------------------------------------------------------------------
    def generate_node(state: RAGState) -> Dict[str, Any]:
        """
        Synthesizes an answer using the grounded prompt and the retrieved context chunks.
        Extracts document names and page numbers for source citations.
        """
        question = state["question"]
        docs = state.get("documents", [])
        steps = list(state.get("workflow_steps", []))
        steps.append("Generate Grounded Answer")
        logger.info("[LangGraph Node: Generate] Generating answer...")

        formatted_context = format_documents_context(docs)
        try:
            answer = qa_chain.invoke({
                "context": formatted_context,
                "question": question,
            })
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            answer = f"An error occurred while generating the response: {str(e)}"

        # Extract clean source metadata from retrieved documents
        sources = format_sources(docs)

        return {
            "answer": answer.strip(),
            "sources": sources,
            "workflow_steps": steps,
        }

    # -------------------------------------------------------------------------
    # Node 5: Fallback Response (No Hallucination)
    # -------------------------------------------------------------------------
    def fallback_node(state: RAGState) -> Dict[str, Any]:
        """
        Executes when no relevant documents were found or all retries failed.
        Explicitly informs the user rather than inventing facts.
        """
        steps = list(state.get("workflow_steps", []))
        steps.append("Fallback (Information Not Found)")
        logger.info("[LangGraph Node: Fallback] Returning grounded not-found message.")

        fallback_message = "I could not find this information in the uploaded documents."
        return {
            "answer": fallback_message,
            "sources": [],
            "workflow_steps": steps,
        }

    # -------------------------------------------------------------------------
    # Routing Logic (Conditional Edges)
    # -------------------------------------------------------------------------
    def route_after_relevance(state: RAGState) -> str:
        """
        Determines the next step after the relevance evaluation.
        Routes to 'generate' if relevant, retries 'retrieve' if irrelevant
        and retries remaining, or 'fallback' if retries exhausted.
        """
        status = state.get("retrieval_status", "")
        retry_count = state.get("retry_count", 0)

        if status == "relevant":
            return "generate"
        
        # If context is irrelevant and we haven't reached max_retries, retry retrieval
        if status == "irrelevant" and retry_count <= max_retries:
            logger.info(f"[LangGraph Router] Irrelevant context detected. Retrying retrieval (attempt {retry_count}/{max_retries})...")
            return "retrieve"

        logger.info("[LangGraph Router] Max retries exceeded or empty context. Routing to fallback.")
        return "fallback"

    # -------------------------------------------------------------------------
    # Build StateGraph
    # -------------------------------------------------------------------------
    workflow = StateGraph(RAGState)

    # Register Nodes
    workflow.add_node("contextualize", contextualize_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("check_relevance", check_relevance_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("fallback", fallback_node)

    # Register Edges
    workflow.add_edge(START, "contextualize")
    workflow.add_edge("contextualize", "retrieve")
    workflow.add_edge("retrieve", "check_relevance")

    # Conditional Routing
    workflow.add_conditional_edges(
        "check_relevance",
        route_after_relevance,
        {
            "generate": "generate",
            "retrieve": "retrieve",
            "fallback": "fallback",
        },
    )

    # Terminate workflow
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    # Compile the graph
    app = workflow.compile()
    logger.info("LangGraph RAG workflow successfully compiled.")
    return app


def run_rag_pipeline(
    question: str,
    vector_store: FAISS,
    llm: BaseChatModel,
    chat_history: Optional[List[BaseMessage]] = None,
    top_k: int = 4,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Convenience function to execute the LangGraph RAG pipeline.

    Parameters:
        question: User query string.
        vector_store: Active FAISS vector store.
        llm: Configured BaseChatModel.
        chat_history: Optional list of past BaseMessage objects.
        top_k: Number of retrieved chunks.
        max_retries: Max retries allowed for relevance checking.

    Returns:
        Dictionary containing answer, sources, workflow_steps, and retrieved documents.
    """
    graph = build_rag_graph(
        vector_store=vector_store,
        llm=llm,
        top_k=top_k,
        max_retries=max_retries,
    )

    initial_state: RAGState = {
        "question": question,
        "standalone_question": question,
        "chat_history": chat_history or [],
        "documents": [],
        "scores": [],
        "answer": "",
        "sources": [],
        "retrieval_status": "initialized",
        "retry_count": 0,
        "workflow_steps": [],
    }

    final_state = graph.invoke(initial_state)
    return {
        "answer": final_state.get("answer", ""),
        "sources": final_state.get("sources", []),
        "documents": final_state.get("documents", []),
        "scores": final_state.get("scores", []),
        "workflow_steps": final_state.get("workflow_steps", []),
        "standalone_question": final_state.get("standalone_question", question),
    }
