"""Retrieval schema v2: corpus-based recall/IDCG and explicit relevance provenance.

Pseudo-relevance (Level + Type, then operation) measures metadata agreement,
not human judgments. Legacy recall/nDCG must not be compared with schema v2.
"""

from dataclasses import dataclass
from typing import Any

from src.evaluation.ranking import (
    Qrels,
    aggregate_rankings,
    corpus_fingerprint,
    evaluate_ranking,
    normalize_id,
)
from src.models.question import Question, RetrievedExample


def is_relevant(query: Question, candidate: Question, strong: bool = False) -> bool:
    base = query.level == candidate.level and query.question_type == candidate.question_type
    return base and (not strong or query.operation == candidate.operation)


def evaluate_retrieval(
    query: Question,
    retrieved: list[RetrievedExample],
    k: int,
    *,
    corpus: list[Question],
    qrels: Qrels | None = None,
) -> dict[str, Any]:
    """Require the entire eligible retriever corpus, never infer it from top-k.

    The caller supplies all questions with non-null/nonempty answers, matching
    HybridRetriever's BM25 universe. The query itself is excluded here and there.
    """
    corpus_ids = [normalize_id(q.id) for q in corpus]
    query_id = normalize_id(query.id)
    grades: dict[str, int] | None
    if qrels is None:
        grades = {
            normalize_id(q.id): (
                2 if is_relevant(query, q, strong=True) else 1 if is_relevant(query, q) else 0
            )
            for q in corpus
        }
        source = "pseudo_relevance_level_type_operation"
        version = "1"
        corpus_id = corpus_fingerprint(corpus_ids)
    else:
        qrels.validate_corpus(corpus_ids)
        grades = qrels.judgments.get(query_id)
        source, version, corpus_id = qrels.source, qrels.version, qrels.corpus_id
    result = evaluate_ranking(
        [normalize_id(ex.question.id) for ex in retrieved],
        k,
        corpus_ids=corpus_ids,
        grades=grades,
        source=source,
        relevance_version=version,
        corpus_id=corpus_id,
        excluded_ids=[query_id] if query_id in corpus_ids else [],
    )
    distances = [ex.distance for ex in retrieved[:k]]
    # Hybrid retrieval can return negative BM25 scores here: neither cosine nor RRF.
    result["retriever_distances"] = distances
    result["query_id"] = query_id
    return result


@dataclass
class RetrievalMetrics:
    """Aggregated metrics and denominators, including unjudged exclusions."""

    payload: dict[str, Any]

    @property
    def n_queries(self) -> int:
        return int(self.payload["n_queries"])

    def summary(self) -> str:
        lines = ["\n=== Métricas de Retrieval (schema v2) ===", f"Queries: {self.n_queries}"]
        for key, value in self.payload["metrics"].items():
            rendered = "N/A" if value is None else f"{value:.4f}"
            count = self.payload["valid_counts"][key]
            lines.append(f"{key}: {rendered} (n={count})")
        lines.append(f"Status: {self.payload['status_counts']}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload["metrics"])


def aggregate(retrievals: list[dict[str, Any]]) -> RetrievalMetrics:
    return RetrievalMetrics(aggregate_rankings(retrievals))
