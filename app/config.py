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
RETRIEVAL_TOP_K = _int("RETRIEVAL_TOP_K", 24)  # chunks pulled per JD query
SHORTLIST_SIZE = _int("SHORTLIST_SIZE", 6)     # repos handed to the LLM re-ranker
FINAL_PROJECT_COUNT = _int("FINAL_PROJECT_COUNT", 3)

# --- Pipeline ---------------------------------------------------------------
MAX_REPOS = _int("MAX_REPOS", 10)              # repos summarised per request
LLM_COOLDOWN_SECONDS = float(os.getenv("LLM_COOLDOWN_SECONDS", "1"))
