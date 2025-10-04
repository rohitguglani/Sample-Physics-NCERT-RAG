"""Top-level package marker for the Physics Book RAG codebase.

Importing :mod:`physics_rag` makes the submodules listed in ``__all__``
available, which keeps interactive sessions and notebooks concise. The
``__all__`` variable works like a curated index, telling Python and human
readers which modules are the main building blocks worth exploring first.
"""

# Re-export the key modules so ``from physics_rag import ingestion`` feels natural.
__all__ = [
    "config",  # Configuration helpers (reading YAML, ensuring folders exist).
    "ingestion",  # Scripts that build the FAISS index from textbooks.
    "retrieval",  # Vector store loader + similarity search utilities.
    "llm",  # Ollama-friendly language model wrapper used by the chatbot.
]
