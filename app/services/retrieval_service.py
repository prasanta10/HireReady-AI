"""Semantic retrieval over indexed GitHub repositories.

Replaces the old "cache a summary and let the LLM read all of them" approach:
repository READMEs are chunked, embedded and stored in a vector index, and the
job description is used as the query to shortlist the most relevant projects.
"""
import hashlib
from typing import Dict, List, Optional, Sequence

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_CHUNKS_PER_REPO,
    RETRIEVAL_TOP_K,
    SHORTLIST_SIZE,
)
from app.services.text_utils import chunk_text, normalize_whitespace
from app.services.vector_store import (
    VectorRecord,
    embed_text,
    embed_texts,
    embedding_backend_name,
    get_vector_store,
)


def content_hash(*parts: Optional[str]) -> str:
    """Stable fingerprint of a repo's indexed content, used to skip re-embedding."""
    digest = hashlib.sha256()
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
    store.add(records, embed_texts([record.text for record in records]))
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


def rank_repositories(
    user_id: str,
    job_description: str,
    projects: Sequence[Dict],
    top_n: int = SHORTLIST_SIZE,
) -> List[Dict]:
    """Shortlist the repositories whose content best matches the job description.

    Returns the ``projects`` dicts enriched with ``match_score`` (0-100) and
    ``evidence`` (the best matching chunk), ordered most relevant first.
    """
    by_name = {project["name"]: project for project in projects}
    if not by_name:
        return []

    try:
        hits = get_vector_store().search(
            embed_text(job_description),
            k=max(RETRIEVAL_TOP_K, top_n * 3),
            where={"user_id": user_id},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: semantic search failed ({exc}); falling back to listing order.")
        hits = []

    grouped: Dict[str, List] = {}
    for hit in hits:
        name = str(hit.metadata.get("repo_name", ""))
        if name in by_name:
            grouped.setdefault(name, []).append(hit)

    ranked: List[Dict] = []
    for name, repo_hits in grouped.items():
        scores = sorted((hit.score for hit in repo_hits), reverse=True)
        best = scores[0]
        supporting = sum(scores[:3]) / len(scores[:3])
        # Weight the single strongest chunk, but reward repos that match broadly.
        combined = 0.7 * best + 0.3 * supporting
        evidence = max(repo_hits, key=lambda hit: hit.score)
        ranked.append(
            {
                **by_name[name],
                "match_score": round(max(0.0, min(1.0, combined)) * 100, 1),
                "matched_chunks": len(repo_hits),
                "evidence": evidence.text[:400],
            }
        )

    ranked.sort(key=lambda project: project["match_score"], reverse=True)

    # Repos with no retrieval hits still belong in the pool, just below the matches.
    unranked = [
        {**project, "match_score": 0.0, "matched_chunks": 0, "evidence": ""}
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
        "total_chunks": total,
        "user_chunks": mine,
    }
