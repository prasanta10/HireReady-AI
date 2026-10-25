"""Hybrid retrieval over indexed GitHub repositories.

Replaces the old "cache a summary and let the LLM read all of them" approach:
repository READMEs are chunked, embedded and stored in a vector index, and the
job description is used as the query to shortlist the most relevant projects.

Ranking runs in three stages:
  1. Recall    - dense vector search and BM25 keyword search, each over chunks.
  2. Fusion    - Reciprocal Rank Fusion merges the two ranked lists.
  3. Precision - a cross-encoder re-scores the top fused chunks against a
                 condensed form of the job description.
Chunk scores are then aggregated per repository.
"""
import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_DOCUMENT_PREFIX,
    INDEX_VERSION,
    LEXICAL_TOP_K,
    MAX_CHUNKS_PER_REPO,
    RERANK_CANDIDATES,
    RERANK_QUERY_CHARS,
    RETRIEVAL_TOP_K,
    RRF_K,
    SHORTLIST_SIZE,
)
from app.services import reranker
from app.services.ats_service import SKILL_PHRASES, SKILL_TOKENS, extract_jd_keywords
from app.services.hybrid_search import bm25_rank, reciprocal_rank_fusion
from app.services.text_utils import chunk_text, normalize_whitespace, tokenize
from app.services.vector_store import (
    VectorRecord,
    embed_documents,
    embed_query,
    embed_text,
    embedding_backend_name,
    get_vector_store,
)


def content_hash(*parts: Optional[str]) -> str:
    """Stable fingerprint of a repo's indexed content, used to skip re-embedding.

    Salted with the index version and document prefix so a change to how
    chunks are embedded invalidates every stored fingerprint exactly once.
    """
    digest = hashlib.sha256(f"v{INDEX_VERSION}|{EMBED_DOCUMENT_PREFIX}\x00".encode("utf-8"))
    for part in parts:
        digest.update((part or "").encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def _repo_documents(repo_name: str, language: Optional[str], summary: str, readme: Optional[str]) -> List[str]:
    """Build the text chunks that represent one repository."""
    header = f"Repository: {repo_name}"
    if language:
        header += f" (primary language: {language})"

    documents: List[str] = []
    if summary:
        documents.append(f"{header}\nResume summary: {normalize_whitespace(summary)}")

    readme = normalize_whitespace(readme or "")
    if readme:
        budget = max(1, MAX_CHUNKS_PER_REPO - len(documents))
        for chunk in chunk_text(readme, CHUNK_SIZE, CHUNK_OVERLAP, max_chunks=budget):
            documents.append(f"{header}\n{chunk}")

    if not documents:
        documents.append(header)
    return documents[:MAX_CHUNKS_PER_REPO]


def index_repository(
    user_id: str,
    repo_name: str,
    url: str,
    language: Optional[str],
    summary: str,
    readme: Optional[str],
) -> int:
    """(Re)index one repository. Returns the number of chunks written."""
    store = get_vector_store()
    documents = _repo_documents(repo_name, language, summary, readme)
    fingerprint = content_hash(summary, readme)

    records = [
        VectorRecord(
            id=f"{user_id}::{repo_name}::{index}",
            text=document,
            metadata={
                "user_id": user_id,
                "repo_name": repo_name,
                "url": url or "",
                "language": language or "",
                "summary": summary or "",
                "content_hash": fingerprint,
                "chunk_index": index,
            },
        )
        for index, document in enumerate(documents)
    ]

    store.delete({"user_id": user_id, "repo_name": repo_name})
    store.add(records, embed_documents([record.text for record in records]))
    return len(records)


_dimension_checked = False


def ensure_embedding_compatibility() -> bool:
    """Clear the index when the active embedding model no longer matches it.

    Swapping EMBED_MODEL (or losing Ollama and falling back to hashed vectors)
    changes the vector width, which would otherwise make every search silently
    return nothing. Runs once per process; repos are re-embedded on demand.
    """
    global _dimension_checked
    if _dimension_checked:
        return False
    _dimension_checked = True
    try:
        store = get_vector_store()
        if store.ensure_dimension(len(embed_text("embedding dimension probe"))):
            print("WARN: embedding model changed; vector index cleared and will rebuild.")
            return True
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: embedding compatibility check failed ({exc}).")
    return False


def is_indexed(user_id: str, repo_name: str, fingerprint: str) -> bool:
    """True when this exact content is already in the index."""
    try:
        return get_vector_store().count(
            {"user_id": user_id, "repo_name": repo_name, "content_hash": fingerprint}
        ) > 0
    except Exception as exc:  # noqa: BLE001 - never let the index block the pipeline
        print(f"WARN: vector index lookup failed ({exc}).")
        return False


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
@dataclass
class _Candidate:
    id: str
    text: str
    repo_name: str
    fused: float                       # Reciprocal Rank Fusion score
    cosine: float = 0.0                # dense similarity; 0 when only BM25 found it
    relevance: Optional[float] = None  # cross-encoder score in [0, 1]


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")


def _mentions_skill(sentence: str) -> bool:
    lowered = sentence.lower()
    if any(token in SKILL_TOKENS for token in tokenize(lowered)):
        return True
    return any(phrase in lowered for phrase in SKILL_PHRASES)


def build_rerank_query(job_description: str, max_chars: int = RERANK_QUERY_CHARS) -> str:
    """Condense a JD into what fits a cross-encoder's 512-token window.

    Query and passage share that window, and raw JDs usually open with company
    boilerplate that would crowd the actual requirements out. Keep the top
    keywords plus the sentences that name concrete skills, in original order.
    """
    keywords = [
        term
        for term, _ in extract_jd_keywords(job_description, limit=40)
        if term in SKILL_TOKENS or term in SKILL_PHRASES
    ][:15]
    parts = [f"Key skills: {', '.join(keywords)}."] if keywords else []
    used = len(parts[0]) if parts else 0

    for sentence in _SENTENCE_SPLIT.split(job_description or ""):
        sentence = sentence.strip(" -*•\t")
        if not sentence or not _mentions_skill(sentence):
            continue
        if used + len(sentence) + 1 > max_chars:
            continue
        parts.append(sentence)
        used += len(sentence) + 1

    return " ".join(parts) or normalize_whitespace(job_description)[:max_chars]


def _hybrid_candidates(user_id: str, job_description: str, repo_names: set) -> List[_Candidate]:
    """Dense + BM25 recall over the user's chunks, fused with RRF."""
    store = get_vector_store()
    where = {"user_id": user_id}

    try:
        dense_hits = store.search(embed_query(job_description), k=RETRIEVAL_TOP_K, where=where)
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: semantic search failed ({exc}); continuing with keyword search.")
        dense_hits = []

    try:
        corpus = store.get(where)
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: keyword search failed ({exc}); continuing with semantic search.")
        corpus = []

    lexical_ids = bm25_rank(
        job_description, [r.id for r in corpus], [r.text for r in corpus], LEXICAL_TOP_K
    )
    dense_ids = [hit.id for hit in dense_hits]

    texts = {r.id: (r.text, r.metadata) for r in corpus}
    texts.update({hit.id: (hit.text, hit.metadata) for hit in dense_hits})
    cosine = {hit.id: hit.score for hit in dense_hits}

    candidates: List[_Candidate] = []
    for doc_id, fused in reciprocal_rank_fusion([dense_ids, lexical_ids], k=RRF_K):
        text, metadata = texts[doc_id]
        name = str(metadata.get("repo_name", ""))
        if name in repo_names:
            candidates.append(
                _Candidate(id=doc_id, text=text, repo_name=name, fused=fused, cosine=cosine.get(doc_id, 0.0))
            )
    return candidates


def _aggregate(scores: Sequence[float]) -> float:
    """Weight the single strongest chunk, but reward repos that match broadly."""
    ordered = sorted(scores, reverse=True)
    supporting = sum(ordered[:3]) / len(ordered[:3])
    return 0.7 * ordered[0] + 0.3 * supporting


def rank_repositories(
    user_id: str,
    job_description: str,
    projects: Sequence[Dict],
    top_n: int = SHORTLIST_SIZE,
) -> List[Dict]:
    """Rank the repositories whose content best matches the job description.

    Returns the ``projects`` dicts enriched with ``match_score`` (0-100),
    ``evidence`` (the best matching chunk) and ``rank_source``, ordered most
    relevant first. ``match_score`` is the cross-encoder relevance when the
    re-ranker is available, otherwise the dense cosine similarity.
    """
    by_name = {project["name"]: project for project in projects}
    if not by_name:
        return []

    candidates = _hybrid_candidates(user_id, job_description, set(by_name))

    head = candidates[:RERANK_CANDIDATES]
    relevance = reranker.score(build_rerank_query(job_description), [c.text for c in head]) if head else None
    if relevance is not None:
        for candidate, value in zip(head, relevance):
            candidate.relevance = value

    grouped: Dict[str, List[_Candidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.repo_name, []).append(candidate)

    ranked = []
    for name, chunks in grouped.items():
        reranked = [c for c in chunks if c.relevance is not None]
        fused = _aggregate([c.fused for c in chunks])
        if reranked:
            score = _aggregate([c.relevance for c in reranked])
            evidence = max(reranked, key=lambda c: c.relevance)
            source = "cross-encoder"
        else:
            score = max(0.0, min(1.0, _aggregate([c.cosine for c in chunks])))
            evidence = max(chunks, key=lambda c: c.fused)
            source = "rrf"
        # Re-ranked repos first (by cross-encoder score), then the rest by fusion.
        sort_key = (True, score) if reranked else (False, fused)
        ranked.append(
            (
                sort_key,
                {
                    **by_name[name],
                    "match_score": round(score * 100, 1),
                    "matched_chunks": len(chunks),
                    "evidence": evidence.text[:400],
                    "rank_source": source,
                },
            )
        )

    ranked.sort(key=lambda item: item[0], reverse=True)
    ranked = [project for _, project in ranked]

    # Repos with no retrieval hits still belong in the pool, just below the matches.
    unranked = [
        {**project, "match_score": 0.0, "matched_chunks": 0, "evidence": "", "rank_source": "none"}
        for name, project in by_name.items()
        if name not in grouped
    ]
    return (ranked + unranked)[:top_n]


def index_stats(user_id: Optional[str] = None) -> Dict:
    """Small diagnostic payload for the API/UI."""
    store = get_vector_store()
    try:
        total = store.count()
        mine = store.count({"user_id": user_id}) if user_id else total
    except Exception as exc:  # noqa: BLE001
        return {"backend": "unavailable", "error": str(exc)}
    return {
        "backend": store.backend,
        "embedder": embedding_backend_name(),
        "retrieval": "hybrid (dense + BM25, RRF)",
        "reranker": reranker.reranker_name(),
        "total_chunks": total,
        "user_chunks": mine,
    }
