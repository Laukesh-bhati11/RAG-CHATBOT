"""
Script to generate a structured, multi-page demo PDF document for DocuMind AI.

Content covers:
- Page 1: Python Programming
- Page 2: Machine Learning Fundamentals
- Page 3: Retrieval-Augmented Generation (RAG)
- Page 4: LangChain Framework
- Page 5: LangGraph Orchestration
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def create_sample_pdf(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=14,
    )
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#2563EB'),
        spaceAfter=10,
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontSize=10,
        leading=15,
        textColor=colors.HexColor('#1F2937'),
        spaceAfter=10,
    )

    story = []

    # -------------------------------------------------------------------------
    # PAGE 1: Python Programming
    # -------------------------------------------------------------------------
    story.append(Paragraph("Chapter 1: Python Programming Fundamentals", title_style))
    story.append(Paragraph("Overview of Python", heading_style))
    story.append(Paragraph(
        "Python is a high-level, general-purpose, interpreted programming language created by Guido van Rossum "
        "and first released in 1991. Python emphasizes code readability with its distinctive use of significant indentation. "
        "Its dynamic typing and garbage collection make it one of the most accessible and versatile languages in the world.",
        body_style
    ))
    story.append(Paragraph("Core Language Features", heading_style))
    story.append(Paragraph(
        "Key aspects of Python include extensive standard libraries, interactive REPL environments, list comprehensions, "
        "first-class functions, and support for object-oriented, procedural, and functional programming paradigms. "
        "In artificial intelligence and data science, Python is the undisputed industry standard due to foundational libraries "
        "like NumPy, Pandas, Scikit-Learn, PyTorch, and TensorFlow.",
        body_style
    ))
    story.append(Paragraph("Execution Model and Virtual Environments", heading_style))
    story.append(Paragraph(
        "Python source code is compiled into bytecode (.pyc) and executed by the Python Virtual Machine (PVM). "
        "Virtual environments (such as venv or conda) isolate package dependencies on a per-project basis, avoiding "
        "system-wide dependency collisions.",
        body_style
    ))
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # PAGE 2: Machine Learning Fundamentals
    # -------------------------------------------------------------------------
    story.append(Paragraph("Chapter 2: Machine Learning Fundamentals", title_style))
    story.append(Paragraph("What is Machine Learning?", heading_style))
    story.append(Paragraph(
        "Machine Learning (ML) is a branch of artificial intelligence focused on building systems that learn from data "
        "and improve their performance over time without being explicitly programmed with deterministic hardcoded rules. "
        "ML models discover patterns, correlations, and representations directly from training observations.",
        body_style
    ))
    story.append(Paragraph("Core Learning Paradigms", heading_style))
    story.append(Paragraph(
        "1. Supervised Learning: Models are trained on labeled datasets containing both input features (X) and ground-truth targets (y). "
        "Tasks include classification (predicting discrete labels) and regression (predicting continuous numerical quantities).<br/>"
        "2. Unsupervised Learning: The algorithm identifies latent patterns, groupings, and clusters in unlabeled data without target guidance. "
        "Examples include K-Means clustering, PCA (Principal Component Analysis), and density estimation.<br/>"
        "3. Reinforcement Learning: Agents learn optimal decision policies by taking actions in an environment to maximize cumulative rewards.",
        body_style
    ))
    story.append(Paragraph("Generalization and Overfitting", heading_style))
    story.append(Paragraph(
        "A critical challenge in machine learning is overfitting, where a model memorizes the training data but fails to generalize "
        "to unseen evaluation data. Techniques like cross-validation, regularization (L1/L2), dropout, and early stopping help achieve "
        "robust generalization performance.",
        body_style
    ))
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # PAGE 3: Retrieval-Augmented Generation (RAG)
    # -------------------------------------------------------------------------
    story.append(Paragraph("Chapter 3: Retrieval-Augmented Generation (RAG)", title_style))
    story.append(Paragraph("The Purpose of RAG", heading_style))
    story.append(Paragraph(
        "Retrieval-Augmented Generation (RAG) is an architectural pattern that enhances Large Language Models (LLMs) by coupling "
        "them with an external retrieval mechanism. While standard LLMs rely only on parametric knowledge acquired during training, "
        "RAG provides real-time access to authoritative, private, or up-to-date document databases.",
        body_style
    ))
    story.append(Paragraph("The RAG Pipeline Steps", heading_style))
    story.append(Paragraph(
        "The standard RAG pipeline operates in six distinct phases:<br/>"
        "1. Ingestion: Ingesting raw documents (such as PDF files, Word documents, or web pages).<br/>"
        "2. Chunking: Breaking text into small, overlapping chunks (e.g., 1000 characters with 200 character overlap) to preserve local semantic context.<br/>"
        "3. Embedding: Converting text chunks into dense mathematical vector representations using an embedding model.<br/>"
        "4. Vector Indexing: Storing vectors and metadata in a specialized index like FAISS.<br/>"
        "5. Retrieval: Converting user questions into query vectors and retrieving top-K nearest neighbors via similarity search.<br/>"
        "6. Synthesis: Inserting the retrieved context into a prompt template and instructing the LLM to formulate a grounded answer.",
        body_style
    ))
    story.append(Paragraph("Mitigating Hallucination", heading_style))
    story.append(Paragraph(
        "A primary advantage of RAG is the elimination of factual hallucinations. By grounding responses strictly in verified document "
        "chunks and instructing the model to declare when information is absent ('I could not find this information in the uploaded documents'), "
        "RAG systems deliver auditable, cited, and trustworthy answers.",
        body_style
    ))
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # PAGE 4: LangChain Framework
    # -------------------------------------------------------------------------
    story.append(Paragraph("Chapter 4: The LangChain Framework", title_style))
    story.append(Paragraph("Overview and Architecture", heading_style))
    story.append(Paragraph(
        "LangChain is a widely adopted open-source framework designed to simplify the development of applications powered by Large Language Models. "
        "It provides modular abstractions and standardized interfaces for connecting language models to external data sources, computation engines, and tools.",
        body_style
    ))
    story.append(Paragraph("Key Components of LangChain", heading_style))
    story.append(Paragraph(
        "LangChain consists of several core building blocks:<br/>"
        "• Document Loaders: Extract text and metadata from diverse formats (PyPDFLoader, CSVLoader, Unstructured).<br/>"
        "• Text Splitters: Intelligent segmentation algorithms like RecursiveCharacterTextSplitter.<br/>"
        "• Vector Stores: Wrappers for databases like FAISS, Chroma, and Pinecone that support similarity search.<br/>"
        "• Embeddings: Standardized wrappers for HuggingFaceEmbeddings, OllamaEmbeddings, and local models.<br/>"
        "• Prompt Templates: Parameterized templates that structure prompts consistently for LLMs.<br/>"
        "• LCEL (LangChain Expression Language): A declarative syntax using the pipe operator (|) to compose runnables with built-in streaming, batching, and async support.",
        body_style
    ))
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # PAGE 5: LangGraph Workflow Orchestration
    # -------------------------------------------------------------------------
    story.append(Paragraph("Chapter 5: LangGraph Workflow Orchestration", title_style))
    story.append(Paragraph("Why LangGraph?", heading_style))
    story.append(Paragraph(
        "While linear chains (like standard LangChain pipelines) work well for straightforward sequential tasks, real-world AI applications "
        "require loops, condition-based branching, state persistence, and multi-step reasoning. LangGraph extends LangChain by representing "
        "LLM workflows as cyclic, stateful computational graphs.",
        body_style
    ))
    story.append(Paragraph("Core Concepts of LangGraph", heading_style))
    story.append(Paragraph(
        "1. State: A centralized schema (typically a TypedDict or Pydantic model) that maintains application state across all steps.<br/>"
        "2. Nodes: Standalone Python functions that receive the current state, perform a unit of work (such as document retrieval or LLM inference), "
        "and return state updates.<br/>"
        "3. Edges: Direct pathways that connect one node to another.<br/>"
        "4. Conditional Edges: Decision routers that examine the current state and dynamically decide which node to visit next.<br/>"
        "5. START and END: Special sentinel nodes denoting the entry and termination points of the workflow execution.",
        body_style
    ))
    story.append(Paragraph("Self-Correcting RAG with LangGraph", heading_style))
    story.append(Paragraph(
        "In DocuMind AI, LangGraph enables an advanced RAG loop: after retrieving document chunks from FAISS, a relevance evaluation node inspects "
        "whether the retrieved content genuinely answers the query. If relevant, execution proceeds to generation. If not relevant, the workflow "
        "routes to a grounded fallback branch or retries retrieval, preventing false answers and ensuring strict fidelity to source documents.",
        body_style
    ))

    doc.build(story)
    print(f"Sample PDF successfully generated at: {output_path}")

if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "data" / "documents" / "sample_ai_notes.pdf"
    create_sample_pdf(target)
