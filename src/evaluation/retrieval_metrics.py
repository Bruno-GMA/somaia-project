"""Métricas de qualidade de retrieval baseadas na literatura de IR (BEIR, TREC).

Como o dataset não possui rótulos explícitos de relevância entre questões (``qrels``),
utilizamos uma definição de **pseudo-relevância**: um vizinho recuperado ``v`` é
considerado relevante para a query ``q`` quando ``v.level == q.level`` e
``v.question_type == q.question_type``. A ``operation`` atua como critério
secundário (relevância "forte" quando também coincide).

Referências:
    - Thakur et al., *BEIR* (2021).
    - Järvelin & Kekäläinen, *Cumulated gain-based evaluation of IR techniques* (2002)
      — nDCG.
    - Voorhees, *The TREC-8 Question Answering Track Report* (1999) — MRR.

Limitação: sem ``qrels`` reais, as métricas medem a concordância do retrieval com os
metadados (Level/Type), não a utilidade factual dos exemplos para a resposta.
"""

from dataclasses import dataclass
from typing import Any

from src.models.question import Question, RetrievedExample


def is_relevant(query: Question, candidate: Question, strong: bool = False) -> bool:
    """Determina pseudo-relevância por metadados (Level + Type).

    Quando ``strong=True``, a ``operation`` também deve coincidir (relevância "forte").
    """
    base = (
        query.level == candidate.level
        and query.question_type == candidate.question_type
    )
    if not base:
        return False
    if strong:
        return query.operation == candidate.operation
    return True


def _relevance_grades(
    query: Question, retrieved: list[RetrievedExample]
) -> list[float]:
    """Graus de relevância (2 = forte, 1 = fraca, 0 = irrelevante) para nDCG."""
    grades: list[float] = []
    for ex in retrieved:
        cand = ex.question
        if is_relevant(query, cand, strong=True):
            grades.append(2.0)
        elif is_relevant(query, cand, strong=False):
            grades.append(1.0)
        else:
            grades.append(0.0)
    return grades


def evaluate_retrieval(
    query: Question,
    retrieved: list[RetrievedExample],
    k: int,
) -> dict[str, Any]:
    """Calcula as métricas de ranking para uma única query.

    Args:
        query: Questão de referência.
        retrieved: Resultados ordenados retornados pelo retriever (top-k).
        k: Valor de corte (``@k``).

    Retorna um dicionário com as métricas e as estatísticas de similaridade.
    """
    top = retrieved[:k]
    grades = _relevance_grades(query, top)

    rel_binary = [1.0 if g > 0 else 0.0 for g in grades]

    hit = 1.0 if any(rel_binary) else 0.0

    # MRR: posição do primeiro relevante (1-indexada)
    mrr = 0.0
    for rank, rel in enumerate(rel_binary, start=1):
        if rel > 0:
            mrr = 1.0 / rank
            break

    # Total de documentos relevantes (considerando o próprio corpus recuperado)
    n_relevant = sum(rel_binary)
    recall = n_relevant / k if k > 0 else 0.0
    precision = n_relevant / k if k > 0 else 0.0

    # DCG com relevância graded (2/1/0)
    dcg = sum(
        (2**g - 1) / _log2(rank + 1)
        for rank, g in enumerate(grades, start=1)
    )

    # DCG ideal: ordena os graus em ordem decrescente
    ideal = sorted(grades, reverse=True)
    idcg = sum(
        (2**g - 1) / _log2(rank + 1)
        for rank, g in enumerate(ideal, start=1)
    )
    ndcg = dcg / idcg if idcg > 0 else 0.0

    distances = [ex.distance for ex in top]
    avg_cosine = sum(distances) / len(distances) if distances else 0.0

    # Proxy de score de fusão (RRF): quanto menor a distância, maior o score.
    # O RetrievedExample não expõe o score RRF original, então derivamos um
    # score decrescendo com a distância (similaridade = 1 - distância).
    rrf_proxies = [max(0.0, 1.0 - d) for d in distances]
    avg_rrf = sum(rrf_proxies) / len(rrf_proxies) if rrf_proxies else 0.0

    return {
        "hit_at_k": hit,
        "mrr_at_k": mrr,
        "recall_at_k": recall,
        "precision_at_k": precision,
        "ndcg_at_k": ndcg,
        "avg_cosine_distance": avg_cosine,
        "avg_rrf_score": avg_rrf,
    }


@dataclass
class RetrievalMetrics:
    """Métricas agregadas em todas as queries."""

    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    hit_at_k: float = 0.0
    avg_cosine_distance: float = 0.0
    avg_rrf_score: float = 0.0
    n_queries: int = 0

    def summary(self) -> str:
        return (
            "\n=== Métricas de Retrieval ===\n"
            f"Queries avaliadas: {self.n_queries}\n"
            f"Recall@k: {self.recall_at_k:.4f}\n"
            f"Precision@k: {self.precision_at_k:.4f}\n"
            f"MRR@k: {self.mrr_at_k:.4f}\n"
            f"nDCG@k: {self.ndcg_at_k:.4f}\n"
            f"Hit@k: {self.hit_at_k:.4f}\n"
            f"Distância cosseno média: {self.avg_cosine_distance:.4f}\n"
            f"Score de similaridade médio (proxy RRF): {self.avg_rrf_score:.4f}\n"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "mrr_at_k": self.mrr_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "hit_at_k": self.hit_at_k,
            "avg_cosine_distance": self.avg_cosine_distance,
            "avg_rrf_score": self.avg_rrf_score,
        }


def _log2(x: float) -> float:
    import math

    return math.log2(x) if x > 1 else 1.0


def aggregate(retrievals: list[dict[str, Any]]) -> RetrievalMetrics:
    """Agrega (média) os resultados por-query em um único objeto de métricas."""
    if not retrievals:
        return RetrievalMetrics()

    n = len(retrievals)
    keys = [
        "recall_at_k",
        "precision_at_k",
        "mrr_at_k",
        "ndcg_at_k",
        "hit_at_k",
        "avg_cosine_distance",
        "avg_rrf_score",
    ]
    agg = {key: sum(r[key] for r in retrievals) / n for key in keys}
    return RetrievalMetrics(
        recall_at_k=agg["recall_at_k"],
        precision_at_k=agg["precision_at_k"],
        mrr_at_k=agg["mrr_at_k"],
        ndcg_at_k=agg["ndcg_at_k"],
        hit_at_k=agg["hit_at_k"],
        avg_cosine_distance=agg["avg_cosine_distance"],
        avg_rrf_score=agg["avg_rrf_score"],
        n_queries=n,
    )
