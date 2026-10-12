"""
Prompt Templates for DocuMind AI.

GROUNDING AND HALLUCINATION PREVENTION:
---------------------------------------
A key danger in standard LLM applications is "hallucination"—where the model
generates plausible-sounding but factually fabricated statements.

In Retrieval-Augmented Generation (RAG), we mitigate hallucinations by:
1. Providing a strict System Prompt setting rigid behavioral guardrails.
2. Directing the model to rely SOLELY on the supplied context chunks.
3. Supplying an explicit fallback phrase:
   "I could not find this information in the uploaded documents."
4. Instructing the model never to guess or synthesize outside facts.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# -----------------------------------------------------------------------------
# 1. Grounded RAG Question-Answering Prompt
# -----------------------------------------------------------------------------
RAG_SYSTEM_PROMPT = """You are DocuMind AI, a document question-answering assistant.

Answer the user's question using ONLY the provided document context.

If the answer is present in the context:
- Give a clear and concise answer.
- Explain the answer in simple language when appropriate.

If the answer is not present in the provided context:
- Clearly say that the information was not found in the uploaded documents.
- Do not make up an answer.
- Do not use outside knowledge as if it came from the documents.

Always prefer factual information from the retrieved context.

Context:
{context}
"""

RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", "{question}"),
])


# -----------------------------------------------------------------------------
# 2. Conversation Memory & Question Contextualization Prompt
# -----------------------------------------------------------------------------
# When a user asks a follow-up question like "Why is it important?", the retriever
# cannot know what "it" refers to without examining previous chat turns.
# This prompt rewrites the follow-up question into a clear, standalone search query.
CONTEXTUALIZE_QUESTION_SYSTEM = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history.

DO NOT answer the question. Just reformulate it if needed, or return the original question as is if it is already standalone."""

CONTEXTUALIZE_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", CONTEXTUALIZE_QUESTION_SYSTEM),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])


# -----------------------------------------------------------------------------
# 3. Document Relevance Grader Prompt (Used in LangGraph Workflow)
# -----------------------------------------------------------------------------
RELEVANCE_CHECK_SYSTEM = """You are a document relevance evaluator.
Assess whether the retrieved document context contains information relevant to answering the user question.

Retrieved Context:
{context}

User Question:
{question}

Respond with only one word:
- 'yes' if the context is relevant or contains clues to answer the question.
- 'no' if the context is completely unrelated.
"""

RELEVANCE_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", RELEVANCE_CHECK_SYSTEM),
    ("human", "Is the context relevant? Respond with 'yes' or 'no'."),
])
