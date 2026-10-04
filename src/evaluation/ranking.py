"""Pure ranking evaluation, schema v2; no model, storage or network dependency."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

METRIC_KEYS = ("hit_at_k", "precision_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k")
SCHEMA_VERSION = 2


def normalize_id(value: Any) -> str:
    """Accept nonempty string IDs or nonnegative integers without silently trimming."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"ID inválido: {value!r}")
    result = str(value)
    if not result.strip() or result != result.strip() or isinstance(value, int) and value < 0:
        raise ValueError(f"ID inválido: {value!r}")
    return result


def validate_ids(values: list[Any], label: str) -> list[str]:
    ids = [normalize_id(value) for value in values]
    if len(set(ids)) != len(ids):
        raise ValueError(f"IDs duplicados em {label}")
    return ids


def validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k deve ser inteiro positivo")


def corpus_fingerprint(ids: list[str]) -> str:
    data = json.dumps(sorted(ids), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _validate_grades(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise ValueError("Julgamentos devem mapear document_id para grau inteiro >= 0")
    grades: dict[str, int] = {}
    for doc, grade in raw.items():
        doc_id = normalize_id(doc)
        if doc_id in grades:
            raise ValueError(f"ID duplicado nos julgamentos: {doc_id}")
        if isinstance(grade, bool) or not isinstance(grade, int) or grade < 0:
            raise ValueError(f"Grau inválido para {doc_id}: {grade!r}")
        grades[doc_id] = grade
    return grades


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Chave JSON duplicada: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class Qrels:
    source: str
    version: str
    corpus_id: str
    judgments: dict[str, dict[str, int]]

    @classmethod
    def from_json(cls, text: str) -> Qrels:
        """Parse without reading a file; {} means judged with no known relevant docs."""
        obj = json.loads(text, object_pairs_hook=_unique_object)
        if not isinstance(obj, dict) or type(obj.get("schema_version")) is not int:
            raise ValueError("qrels: schema_version deve ser 1")
        if obj["schema_version"] != 1:
            raise ValueError("qrels: schema_version deve ser 1")
        for key in ("source", "version", "corpus_id"):
            if not isinstance(obj.get(key), str) or not obj[key].strip():
                raise ValueError(f"qrels: campo obrigatório {key}")
        raw = obj.get("qrels")
        if not isinstance(raw, dict):
            raise ValueError("qrels: mapa por query_id obrigatório")
        judgments = {normalize_id(q): _validate_grades(g) for q, g in raw.items()}
        return cls(obj["source"], obj["version"], obj["corpus_id"], judgments)

    def validate_corpus(self, corpus_ids: list[str]) -> None:
        corpus = set(validate_ids(corpus_ids, "corpus"))
        unknown = {doc for grades in self.judgments.values() for doc in grades} - corpus
        if unknown:
            raise ValueError(f"qrels contém documentos fora do corpus: {sorted(unknown)}")


def evaluate_ranking(
    retrieved_ids: list[str],
    k: int,
    *,
    corpus_ids: list[str],
    grades: dict[str, int] | None,
    source: str,
    relevance_version: str,
    corpus_id: str,
    excluded_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a ranking; unknown judgments contribute no gain, with coverage exposed.

    Missing query judgments (None) are excluded from averages; an explicit empty
    map is evaluated as no_relevant. IDCG/recall use all judged eligible documents.
    For incomplete qrels these are metrics over KNOWN relevance, not true recall.
    """
    validate_k(k)
    ranking = validate_ids(retrieved_ids, "ranking")
    corpus = set(validate_ids(corpus_ids, "corpus"))
    excluded = set(validate_ids(excluded_ids or [], "exclusões"))
    if not excluded <= corpus:
        raise ValueError("IDs excluídos devem pertencer ao corpus")
    eligible = corpus - excluded
    if not set(ranking) <= eligible:
        raise ValueError(f"Ranking contém IDs inelegíveis: {sorted(set(ranking) - eligible)}")
    all_grades = _validate_grades(grades) if grades is not None else None
    if all_grades is not None and not set(all_grades) <= corpus:
        raise ValueError("Julgamentos contêm IDs fora do corpus")
    known = {doc: g for doc, g in (all_grades or {}).items() if doc in eligible}
    top = ranking[:k]
    top_grades = [known.get(doc) for doc in top]
    relevant = sum(g > 0 for g in known.values())
    found = sum(g is not None and g > 0 for g in top_grades)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "k": k,
        "source": source,
        "relevance_version": relevance_version,
        "corpus_id": corpus_id,
        "eligible_ids_sha256": corpus_fingerprint(sorted(eligible)),
        "excluded_ids": sorted(excluded),
        "n_eligible": len(eligible),
        "n_judged": len(known),
        "n_relevant": relevant if all_grades is not None else None,
        "judgment_coverage": len(known) / len(eligible) if eligible else None,
        "retrieved_ids": top,
        "relevance_grades": top_grades,
        "n_unjudged_retrieved": sum(g is None for g in top_grades),
        "unjudged_policy": "zero_gain_with_coverage; missing_query_excluded",
        "status": "unjudged" if all_grades is None else "ok" if relevant else "no_relevant",
    }
    if all_grades is None:
        result.update(dict.fromkeys(METRIC_KEYS))
        return result
    # Scale all gains by the same power of two, avoiding overflow for large grades.
    max_grade = max(known.values(), default=0)

    def dcg(values: list[int]) -> float:
        return sum(
            (2.0 ** (g - max_grade) - 2.0 ** (-max_grade)) / math.log2(rank + 1)
            for rank, g in enumerate(values, 1)
        )

    idcg = dcg(sorted(known.values(), reverse=True)[:k])
    result.update(
        {
            "hit_at_k": float(found > 0),
            "precision_at_k": found / k,
            "recall_at_k": found / relevant if relevant else 0.0,
            "mrr_at_k": next((1 / rank for rank, g in enumerate(top_grades, 1) if g), 0.0),
            "ndcg_at_k": dcg([g or 0 for g in top_grades]) / idcg if idcg else 0.0,
        }
    )
    return result


def aggregate_rankings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Macro average; do not turn absent evaluations into zero-valued observations."""
    metrics: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for key in METRIC_KEYS:
        values = [r[key] for r in rows if r[key] is not None]
        metrics[key] = sum(values) / len(values) if values else None
        counts[key] = len(values)
    return {
        "schema_version": SCHEMA_VERSION,
        "metrics": metrics,
        "valid_counts": counts,
        "n_queries": len(rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "sources": sorted({r["source"] for r in rows}),
        "relevance_versions": sorted({r["relevance_version"] for r in rows}),
        "corpus_ids": sorted({r["corpus_id"] for r in rows}),
        "k_values": sorted({r["k"] for r in rows}),
    }
