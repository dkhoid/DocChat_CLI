"""DocChat — Document Q&A from terminal.

Backward-compatible re-exports so existing `from docchat.xxx import ...`
still works after the folder restructure.
"""

# ── Core ──────────────────────────────────────────────────────────────────────
from docchat.core.chunker import Chunk, chunk_document, chunk_documents
from docchat.core.loader import SUPPORTED_EXTENSIONS, Document, load_directory, load_file
from docchat.core.prompt_manager import PromptManager, get_prompt_manager

# ── Embeddings ────────────────────────────────────────────────────────────────
from docchat.embeddings.embedder import BaseEmbedder, EmbedderFactory

# ── Storage ───────────────────────────────────────────────────────────────────
from docchat.storage.store import BaseStore, ChromaVectorStore, SearchResult, SimpleVectorStore

# ── LLM ───────────────────────────────────────────────────────────────────────
from docchat.llm.session import LLMConfig, LLMSession, SessionStats, ask

# ── Infrastructure ────────────────────────────────────────────────────────────
from docchat.infrastructure.logger import get_logger

__all__ = [
    "Chunk",
    "chunk_document",
    "chunk_documents",
    "Document",
    "load_directory",
    "load_file",
    "SUPPORTED_EXTENSIONS",
    "PromptManager",
    "get_prompt_manager",
    "BaseEmbedder",
    "EmbedderFactory",
    "BaseStore",
    "ChromaVectorStore",
    "SearchResult",
    "SimpleVectorStore",
    "LLMConfig",
    "LLMSession",
    "SessionStats",
    "ask",
    "get_logger",
]
