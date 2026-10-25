"""Cross-encoder re-ranking of retrieved chunks.

A cross-encoder reads the (query, passage) pair jointly, which makes it far more
precise than comparing two independently computed embeddings, and unlike an LLM
re-ranker it is deterministic, fast on CPU and returns a usable score.

The model (``RERANK_MODEL``, default ``BAAI/bge-reranker-base``) is downloaded
from Hugging Face on first use and cached locally. If sentence-transformers or
the model is unavailable, ``score`` returns None and callers keep their
fusion order instead.
"""
import math
import threading
from typing import List, Optional, Sequence

from app.config import RERANK_MAX_LENGTH, RERANK_MODEL

_DISABLED = {"", "none", "off", "false", "0"}

_model = None
_available: Optional[bool] = None
_load_lock = threading.Lock()
_predict_lock = threading.Lock()


def _load():
    global _model, _available
    if _available is not None:
        return _model
    with _load_lock:
        if _available is not None:
            return _model
        if RERANK_MODEL.strip().lower() in _DISABLED:
            _available = False
            return None
        try:
            from sentence_transformers import CrossEncoder

            _model = CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_LENGTH)
            _available = True
            print(f"DEBUG: Cross-encoder re-ranker = {RERANK_MODEL}")
        except Exception as exc:  # noqa: BLE001 - re-ranking is an enhancement, never a requirement
            print(
                f"WARN: cross-encoder '{RERANK_MODEL}' unavailable ({exc}). "
                "Falling back to Reciprocal Rank Fusion order. "
                "Run `pip install sentence-transformers` to enable re-ranking."
            )
            _available = False
    return _model


def warm_up() -> None:
    """Load (and on first run download) the model before the first request."""
    _load()


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def score(query: str, passages: Sequence[str]) -> Optional[List[float]]:
    """Relevance of each passage to ``query`` in [0, 1], or None if unavailable."""
    if not passages:
        return []
    model = _load()
    if model is None:
        return None
    try:
        with _predict_lock:
            raw = model.predict([(query, passage) for passage in passages], show_progress_bar=False)
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: cross-encoder scoring failed ({exc}); keeping fusion order.")
        return None

    values = [float(value) for value in raw]
    # Single-label rerankers usually come back sigmoid-activated already; squash
    # raw logits from models that do not, so scores stay comparable as a %.
    if any(value < 0.0 or value > 1.0 for value in values):
        values = [_sigmoid(value) for value in values]
    return values


def reranker_name() -> str:
    """Which re-ranker the last request used (for diagnostics)."""
    if _available:
        return f"cross-encoder:{RERANK_MODEL}"
    if _available is False:
        return "none (RRF order)"
    return "uninitialised"
