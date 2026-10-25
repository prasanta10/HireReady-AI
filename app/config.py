"""Central configuration for HireReady-AI.

Everything is overridable through environment variables (or a .env file) so the
app can run fully offline with sensible defaults.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# --- Models -----------------------------------------------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
# nomic-embed-text is trained with task prefixes and degrades without them.
_NOMIC = "nomic" in EMBED_MODEL.lower()
EMBED_QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX", "search_query: " if _NOMIC else "")
EMBED_DOCUMENT_PREFIX = os.getenv("EMBED_DOCUMENT_PREFIX", "search_document: " if _NOMIC else "")
# Cross-encoder used to re-rank fused candidates. "none" disables re-ranking
# and keeps the Reciprocal Rank Fusion order.
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

# --- Vector store -----------------------------------------------------------
# auto -> chroma, then faiss, then the built-in numpy index.
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "auto").lower()
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "./vector_store")
VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION", "repo_chunks")
# Dimension of the offline fallback embedder (only used when Ollama has no
# embedding model available).
FALLBACK_EMBED_DIM = _int("FALLBACK_EMBED_DIM", 512)

# --- Retrieval --------------------------------------------------------------
CHUNK_SIZE = _int("CHUNK_SIZE", 900)          # characters per chunk
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)
MAX_CHUNKS_PER_REPO = _int("MAX_CHUNKS_PER_REPO", 8)
RETRIEVAL_TOP_K = _int("RETRIEVAL_TOP_K", 24)  # chunks pulled per JD query (dense)
LEXICAL_TOP_K = _int("LEXICAL_TOP_K", 24)      # chunks pulled per JD query (BM25)
RRF_K = _int("RRF_K", 60)                      # Reciprocal Rank Fusion damping constant
RERANK_CANDIDATES = _int("RERANK_CANDIDATES", 30)  # fused chunks scored by the cross-encoder
RERANK_MAX_LENGTH = _int("RERANK_MAX_LENGTH", 512)  # cross-encoder token window (query + chunk)
RERANK_QUERY_CHARS = _int("RERANK_QUERY_CHARS", 1000)  # condensed JD budget inside that window
SHORTLIST_SIZE = _int("SHORTLIST_SIZE", 6)     # repos returned by retrieval
FINAL_PROJECT_COUNT = _int("FINAL_PROJECT_COUNT", 3)
# Bump when the way chunks are embedded changes; forces a one-time re-index.
INDEX_VERSION = "2"

# --- Pipeline ---------------------------------------------------------------
MAX_REPOS = _int("MAX_REPOS", 10)              # repos summarised per request
LLM_COOLDOWN_SECONDS = float(os.getenv("LLM_COOLDOWN_SECONDS", "1"))
