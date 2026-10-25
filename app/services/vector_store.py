"""Vector storage + embeddings for semantic search over repository content.

Backends, in order of preference (`VECTOR_BACKEND=auto`):
  1. ChromaDB  - persistent, cosine HNSW index.
  2. FAISS     - flat inner-product index over normalised vectors.
  3. numpy     - built-in brute-force index so the app still works with no extras.

Embeddings come from Ollama (`EMBED_MODEL`, default ``nomic-embed-text``). If no
embedding model is reachable we fall back to a deterministic hashed bag-of-words
vector: lower quality, but it keeps the pipeline offline and functional.
"""
import hashlib
import json
import math
import os
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from app.config import (
    EMBED_DOCUMENT_PREFIX,
    EMBED_MODEL,
    EMBED_QUERY_PREFIX,
    FALLBACK_EMBED_DIM,
    VECTOR_BACKEND,
    VECTOR_COLLECTION,
    VECTOR_STORE_DIR,
)
from app.services.text_utils import tokenize

_embed_lock = threading.Lock()
_ollama_embeddings_available: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def _hashed_embedding(text: str, dim: int = FALLBACK_EMBED_DIM) -> List[float]:
    """Deterministic hashed bag-of-words vector (offline fallback)."""
    vector = np.zeros(dim, dtype=np.float32)
    counts = Counter(tokenize(text))
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign * (1.0 + math.log(count))
    norm = float(np.linalg.norm(vector))
    if norm:
        vector /= norm
    return vector.tolist()


def _ollama_embed(texts: Sequence[str]) -> Optional[List[List[float]]]:
    """Embed via Ollama, or return None if unavailable."""
    global _ollama_embeddings_available
    if _ollama_embeddings_available is False:
        return None
    try:
        import ollama
    except ImportError:
        _ollama_embeddings_available = False
        return None

    try:
        if hasattr(ollama, "embed"):
            response = ollama.embed(model=EMBED_MODEL, input=list(texts))
            vectors = getattr(response, "embeddings", None)
            if vectors is None:
                vectors = response["embeddings"]
        else:  # older ollama clients
            vectors = [
                ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]
                for text in texts
            ]
        parsed = [[float(value) for value in vector] for vector in vectors]
        if not parsed or not parsed[0]:
            raise ValueError("empty embedding response")
        _ollama_embeddings_available = True
        return parsed
    except Exception as exc:  # noqa: BLE001 - any failure means "use the fallback"
        if _ollama_embeddings_available is None:
            print(
                f"WARN: Ollama embeddings unavailable ({exc}). "
                f"Falling back to hashed embeddings. "
                f"Run `ollama pull {EMBED_MODEL}` for semantic quality."
            )
        _ollama_embeddings_available = False
        return None


def embed_texts(texts: Sequence[str], prefix: str = "") -> List[List[float]]:
    """Embed a batch of texts, always returning unit-length vectors.

    ``prefix`` is the model's task instruction (e.g. ``search_query: ``); the
    hashed fallback ignores it so it never pollutes the bag of words.
    """
    texts = [text or "" for text in texts]
    if not texts:
        return []
    with _embed_lock:
        vectors = _ollama_embed([prefix + text for text in texts])
    if vectors is None:
        return [_hashed_embedding(text) for text in texts]
    return [_unit(vector) for vector in vectors]


def embed_text(text: str, prefix: str = "") -> List[float]:
    return embed_texts([text], prefix)[0]


def embed_documents(texts: Sequence[str]) -> List[List[float]]:
    return embed_texts(texts, EMBED_DOCUMENT_PREFIX)


def embed_query(text: str) -> List[float]:
    return embed_text(text, EMBED_QUERY_PREFIX)


def embedding_backend_name() -> str:
    """Which embedder produced the most recent vectors."""
    if _ollama_embeddings_available:
        return f"ollama:{EMBED_MODEL}"
    if _ollama_embeddings_available is False:
        return "hashed-bow (fallback)"
    return "uninitialised"


def _unit(vector: Sequence[float]) -> List[float]:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm:
        array = array / norm
    return array.tolist()


# --------------------------------------------------------------------------- #
# Records / results
# --------------------------------------------------------------------------- #
@dataclass
class VectorRecord:
    id: str
    text: str
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class SearchHit:
    id: str
    text: str
    metadata: Dict[str, object]
    score: float  # cosine similarity in [-1, 1]


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class NumpyVectorStore:
    """Brute-force cosine index persisted as .npy + .json. Zero extra deps."""

    backend = "numpy"

    def __init__(self, directory: str = VECTOR_STORE_DIR, collection: str = VECTOR_COLLECTION):
        self.directory = directory
        self.collection = collection
        os.makedirs(self.directory, exist_ok=True)
        self._vectors_path = os.path.join(self.directory, f"{collection}.npy")
        self._records_path = os.path.join(self.directory, f"{collection}.json")
        self._records: List[VectorRecord] = []
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        self._lock = threading.Lock()
        self._load()

    # -- persistence --
    def _load(self) -> None:
        try:
            if os.path.exists(self._records_path) and os.path.exists(self._vectors_path):
                with open(self._records_path, "r", encoding="utf-8") as handle:
                    raw = json.load(handle)
                self._records = [VectorRecord(**item) for item in raw]
                self._vectors = np.load(self._vectors_path).astype(np.float32)
                if len(self._records) != self._vectors.shape[0]:
                    raise ValueError("index/metadata length mismatch")
        except Exception as exc:  # noqa: BLE001 - a corrupt cache should not be fatal
            print(f"WARN: could not load vector index ({exc}); starting empty.")
            self._records, self._vectors = [], np.zeros((0, 0), dtype=np.float32)

    def _persist(self) -> None:
        np.save(self._vectors_path, self._vectors)
        with open(self._records_path, "w", encoding="utf-8") as handle:
            json.dump(
                [{"id": r.id, "text": r.text, "metadata": r.metadata} for r in self._records],
                handle,
            )

    # -- api --
    def add(self, records: Sequence[VectorRecord], embeddings: Sequence[Sequence[float]]) -> None:
        if not records:
            return
        matrix = np.asarray(embeddings, dtype=np.float32)
        with self._lock:
            if self._vectors.size and self._vectors.shape[1] != matrix.shape[1]:
                # Embedding model changed underneath us - rebuild from scratch.
                print("WARN: embedding dimension changed; resetting vector index.")
                self._records, self._vectors = [], np.zeros((0, matrix.shape[1]), dtype=np.float32)
            keep = [i for i, r in enumerate(self._records) if r.id not in {n.id for n in records}]
            if len(keep) != len(self._records):
                self._records = [self._records[i] for i in keep]
                self._vectors = self._vectors[keep] if self._vectors.size else self._vectors
            self._records.extend(records)
            self._vectors = matrix if not self._vectors.size else np.vstack([self._vectors, matrix])
            self._persist()
            self._on_index_changed()

    def delete(self, where: Dict[str, object]) -> int:
        with self._lock:
            keep = [i for i, r in enumerate(self._records) if not _matches(r.metadata, where)]
            removed = len(self._records) - len(keep)
            if removed:
                self._records = [self._records[i] for i in keep]
                self._vectors = self._vectors[keep] if self._vectors.size else self._vectors
                self._persist()
                self._on_index_changed()
            return removed

    @property
    def dimension(self) -> Optional[int]:
        return int(self._vectors.shape[1]) if self._vectors.size else None

    def clear(self) -> None:
        with self._lock:
            self._records = []
            self._vectors = np.zeros((0, 0), dtype=np.float32)
            self._persist()
            self._on_index_changed()

    def ensure_dimension(self, dim: int) -> bool:
        """Drop the index if it was built with a different embedding model."""
        current = self.dimension
        if current is None or current == dim:
            return False
        self.clear()
        return True

    def search(self, embedding: Sequence[float], k: int, where: Optional[Dict] = None) -> List[SearchHit]:
        with self._lock:
            if not self._records:
                return []
            if self._vectors.size and self._vectors.shape[1] != len(embedding):
                # Stale index from a different embedding model; ensure_dimension()
                # clears it on the next indexing pass.
                print("WARN: query/index embedding dimensions differ; skipping search.")
                return []
            candidates = [i for i, r in enumerate(self._records) if _matches(r.metadata, where or {})]
            if not candidates:
                return []
            query = np.asarray(_unit(embedding), dtype=np.float32)
            scores = self._similarities(query, candidates)
            order = np.argsort(-scores)[:k]
            return [
                SearchHit(
                    id=self._records[candidates[i]].id,
                    text=self._records[candidates[i]].text,
                    metadata=self._records[candidates[i]].metadata,
                    score=float(scores[i]),
                )
                for i in order
            ]

    def count(self, where: Optional[Dict] = None) -> int:
        if not where:
            return len(self._records)
        return sum(1 for r in self._records if _matches(r.metadata, where))

    def get(self, where: Optional[Dict] = None) -> List[VectorRecord]:
        """Every stored chunk matching the filter (the corpus for lexical search)."""
        with self._lock:
            return [r for r in self._records if _matches(r.metadata, where or {})]

    # -- hooks for subclasses --
    def _similarities(self, query: np.ndarray, candidates: List[int]) -> np.ndarray:
        return self._vectors[candidates] @ query

    def _on_index_changed(self) -> None:
        pass


class FaissVectorStore(NumpyVectorStore):
    """Same storage as the numpy store, but queries run through a FAISS index."""

    backend = "faiss"

    def __init__(self, *args, **kwargs):
        import faiss  # noqa: F401 - probed by the factory before construction

        self._faiss = faiss
        self._index = None
        super().__init__(*args, **kwargs)
        self._on_index_changed()

    def _on_index_changed(self) -> None:
        if getattr(self, "_faiss", None) is None or not self._vectors.size:
            self._index = None
            return
        index = self._faiss.IndexFlatIP(self._vectors.shape[1])
        index.add(np.ascontiguousarray(self._vectors, dtype=np.float32))
        self._index = index

    def _similarities(self, query: np.ndarray, candidates: List[int]) -> np.ndarray:
        if self._index is None:
            return super()._similarities(query, candidates)
        # Search the whole index, then keep the rows that pass the metadata filter.
        # Exhaustive on purpose: filtering a truncated result set would silently
        # drop matching chunks that rank low globally.
        scores, ids = self._index.search(
            np.ascontiguousarray([query], dtype=np.float32), len(self._records)
        )
        lookup = {int(row): float(score) for row, score in zip(ids[0], scores[0]) if row >= 0}
        return np.asarray([lookup.get(i, -1.0) for i in candidates], dtype=np.float32)


class ChromaVectorStore:
    """Persistent ChromaDB collection (cosine space)."""

    backend = "chroma"

    def __init__(self, directory: str = VECTOR_STORE_DIR, collection: str = VECTOR_COLLECTION):
        import chromadb
        from chromadb.config import Settings

        os.makedirs(directory, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=directory, settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, records: Sequence[VectorRecord], embeddings: Sequence[Sequence[float]]) -> None:
        if not records:
            return
        self._collection.upsert(
            ids=[r.id for r in records],
            documents=[r.text for r in records],
            metadatas=[_scalarise(r.metadata) for r in records],
            embeddings=[list(map(float, e)) for e in embeddings],
        )

    def delete(self, where: Dict[str, object]) -> int:
        if not where:
            return 0
        before = self.count(where)
        self._collection.delete(where=_chroma_where(where))
        return before

    def search(self, embedding: Sequence[float], k: int, where: Optional[Dict] = None) -> List[SearchHit]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(map(float, embedding))],
            n_results=max(1, k),
            where=_chroma_where(where) if where else None,
            include=["documents", "metadatas", "distances"],
        )
        hits: List[SearchHit] = []
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            hits.append(
                SearchHit(
                    id=doc_id,
                    text=document or "",
                    metadata=dict(metadata or {}),
                    score=1.0 - float(distance),  # cosine distance -> similarity
                )
            )
        return hits

    def count(self, where: Optional[Dict] = None) -> int:
        if not where:
            return int(self._collection.count())
        found = self._collection.get(where=_chroma_where(where), include=[])
        return len(found.get("ids") or [])

    def get(self, where: Optional[Dict] = None) -> List[VectorRecord]:
        """Every stored chunk matching the filter (the corpus for lexical search)."""
        found = self._collection.get(
            where=_chroma_where(where), include=["documents", "metadatas"]
        )
        return [
            VectorRecord(id=doc_id, text=document or "", metadata=dict(metadata or {}))
            for doc_id, document, metadata in zip(
                found.get("ids") or [],
                found.get("documents") or [],
                found.get("metadatas") or [],
            )
        ]

    @property
    def dimension(self) -> Optional[int]:
        if self.count() == 0:
            return None
        peek = self._collection.get(limit=1, include=["embeddings"])
        embeddings = peek.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return None
        return len(embeddings[0])

    def clear(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def ensure_dimension(self, dim: int) -> bool:
        current = self.dimension
        if current is None or current == dim:
            return False
        self.clear()
        return True


# --------------------------------------------------------------------------- #
# Helpers + factory
# --------------------------------------------------------------------------- #
def _matches(metadata: Dict, where: Dict) -> bool:
    return all(metadata.get(key) == value for key, value in (where or {}).items())


def _scalarise(metadata: Dict) -> Dict:
    """Chroma only accepts str/int/float/bool metadata values."""
    clean = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        clean[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return clean or {"_": ""}


def _chroma_where(where: Optional[Dict]) -> Optional[Dict]:
    if not where:
        return None
    if len(where) == 1:
        return dict(where)
    return {"$and": [{key: value} for key, value in where.items()]}


_store = None
_store_lock = threading.Lock()


def _build_store():
    preference = {
        "auto": ["chroma", "faiss", "numpy"],
        "chroma": ["chroma", "numpy"],
        "faiss": ["faiss", "numpy"],
        "numpy": ["numpy"],
    }.get(VECTOR_BACKEND, ["chroma", "faiss", "numpy"])

    for name in preference:
        try:
            if name == "chroma":
                return ChromaVectorStore()
            if name == "faiss":
                return FaissVectorStore()
            return NumpyVectorStore()
        except ImportError:
            continue
        except Exception as exc:  # noqa: BLE001 - fall through to the next backend
            print(f"WARN: vector backend '{name}' unavailable ({exc}); trying next.")
    return NumpyVectorStore()


def get_vector_store():
    """Process-wide singleton vector store."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = _build_store()
                print(f"DEBUG: Vector backend = {_store.backend} ({VECTOR_STORE_DIR})")
    return _store


def reset_vector_store() -> None:
    """Drop the cached singleton (used by tests)."""
    global _store
    with _store_lock:
        _store = None
