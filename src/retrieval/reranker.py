"""Deterministic, metadata-only reranking for retrieved SOMAIA examples."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.models.question import RetrievedExample

METHOD_VERSION = "distance-operation-v1"
SIMILARITY_WEIGHT = 0.8
OPERATION_WEIGHT = 0.2

_OPERATION_TERMS = {
    "soma": ("soma", "adição", "adicao", "mais", "total", "juntar"),
    "subtração": ("subtração", "subtracao", "menos", "diferença", "diminui", "restam"),
    "multiplicação": ("multiplicação", "multiplicacao", "vezes", "grupos", "cada"),
    "divisão": ("divisão", "divisao", "dividir", "repartir", "por pessoa"),
}


@dataclass(frozen=True)
class RerankTrace:
    candidate_id: int | None
    original_rank: int
    reranked_rank: int
    original_distance: float | None
    normalized_similarity: float
    operation_match: float
    rerank_score: float


def infer_operation(query: str) -> str | None:
    normalized = re.sub(r"\s+", " ", query.lower())
    hits = [
        (operation, sum(term in normalized for term in terms))
        for operation, terms in _OPERATION_TERMS.items()
    ]
    operation, count = max(hits, key=lambda item: item[1])
    return operation if count else None


def rerank(
    query: str, candidates: list[RetrievedExample], final_k: int
) -> tuple[list[RetrievedExample], list[RerankTrace]]:
    """Reorder existing candidates only, with stable ties and audit traces."""
    if final_k <= 0:
        raise ValueError("final_k deve ser maior que zero")
    if not candidates:
        return [], []
    unique: list[tuple[int, RetrievedExample]] = []
    seen: set[int | None] = set()
    for rank, candidate in enumerate(candidates, 1):
        candidate_id = candidate.question.id
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        unique.append((rank, candidate))
    distances = [float(candidate.distance) for _, candidate in unique]
    low, high = min(distances), max(distances)
    inferred = infer_operation(query)
    scored: list[tuple[float, int, int, RetrievedExample, float, float]] = []
    for original_rank, candidate in unique:
        distance = float(candidate.distance)
        similarity = 1.0 if high == low else (high - distance) / (high - low)
        operation = (candidate.question.operation or "").lower()
        match = float(inferred is not None and inferred in operation)
        score = SIMILARITY_WEIGHT * similarity + OPERATION_WEIGHT * match
        candidate_id = candidate.question.id if candidate.question.id is not None else -1
        scored.append((score, original_rank, candidate_id, candidate, similarity, match))
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    traces = [
        RerankTrace(
            candidate_id=candidate.question.id,
            original_rank=original_rank,
            reranked_rank=rank,
            original_distance=float(candidate.distance),
            normalized_similarity=similarity,
            operation_match=match,
            rerank_score=score,
        )
        for rank, (score, original_rank, _, candidate, similarity, match) in enumerate(scored, 1)
    ]
    return [row[3] for row in scored[:final_k]], traces
