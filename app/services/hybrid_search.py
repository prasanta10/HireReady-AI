"""Lexical search (BM25) and Reciprocal Rank Fusion for hybrid retrieval.

Dense embeddings capture meaning but blur exact technology names ("gRPC",
"Kafka", "k8s"); BM25 matches those tokens exactly. RRF merges the two ranked
lists using rank positions only, so cosine similarities and BM25 scores never
have to be put on the same scale.
"""
import math
from collections import Counter
from typing import Dict, Hashable, List, Sequence, Tuple

from app.services.text_utils import tokenize


def bm25_scores(query: str, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    """Okapi BM25 score of every document for ``query`` (same order as ``documents``)."""
    corpus = [tokenize(document) for document in documents]
    if not corpus:
        return []

    total = len(corpus)
    avg_length = (sum(len(doc) for doc in corpus) / total) or 1.0
    doc_freq = Counter(term for doc in corpus for term in set(doc))
    # A JD that repeats "python" five times cares about it more, but sublinearly.
    query_terms = {term: 1.0 + math.log(count) for term, count in Counter(tokenize(query)).items()}

    scores: List[float] = []
    for doc in corpus:
        term_freq = Counter(doc)
        norm = k1 * (1 - b + b * len(doc) / avg_length)
        score = 0.0
        for term, query_weight in query_terms.items():
            freq = term_freq.get(term)
            if not freq:
                continue
            idf = math.log(1 + (total - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            score += query_weight * idf * freq * (k1 + 1) / (freq + norm)
        scores.append(score)
    return scores


def bm25_rank(query: str, ids: Sequence[Hashable], documents: Sequence[str], k: int) -> List[Hashable]:
    """Ids of the ``k`` best BM25 matches; documents sharing no query term are dropped."""
    scored = [(score, doc_id) for doc_id, score in zip(ids, bm25_scores(query, documents)) if score > 0]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc_id for _, doc_id in scored[:k]]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hashable]], k: int = 60) -> List[Tuple[Hashable, float]]:
    """Fuse ranked id lists: score(d) = sum over lists of 1 / (k + rank_d).

    Items found by several retrievers rise to the top; ``k`` damps the weight of
    the very first ranks so one retriever cannot dominate on its own.
    """
    fused: Dict[Hashable, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)
