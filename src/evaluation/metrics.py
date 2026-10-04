from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.evaluation.validator import DEFAULT_ATOL, DEFAULT_RTOL, validate_answer


@dataclass
class EvaluationMetrics:
    """Independent correctness checks plus evaluator coverage and latency distribution."""

    total: int = 0
    evaluable: int = 0
    correct: int = 0
    exact_match: int = 0
    numeric_match: int = 0
    symbolic_match: int = 0
    not_evaluable: int = 0
    by_level: dict[str, dict[str, int]] = field(default_factory=dict)
    by_type: dict[str, dict[str, int]] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.evaluable if self.evaluable else None

    @property
    def coverage(self) -> float | None:
        return self.evaluable / self.total if self.total else None

    @property
    def exact_match_rate(self) -> float | None:
        return self.exact_match / self.evaluable if self.evaluable else None

    def latency_summary(self) -> dict[str, float | None]:
        if not self.latencies:
            return {"mean_s": None, "median_s": None, "p95_s": None, "stddev_s": None}
        return {
            "mean_s": float(np.mean(self.latencies)),
            "median_s": float(np.median(self.latencies)),
            "p95_s": float(np.percentile(self.latencies, 95)),
            "stddev_s": float(np.std(self.latencies)),
        }

    def add_result(
        self,
        question: Any,
        predicted: str,
        expected: str,
        latency: float,
        tolerance: float = DEFAULT_RTOL,
        *,
        atol: float = DEFAULT_ATOL,
        rtol: float | None = None,
    ) -> dict[str, Any]:
        self.total += 1
        self.latencies.append(latency)
        level = question.level.value if question.level else "Unknown"
        question_type = question.question_type or "Unknown"
        for key, container in ((level, self.by_level), (question_type, self.by_type)):
            container.setdefault(key, {"total": 0, "evaluable": 0, "correct": 0})
            container[key]["total"] += 1

        result = validate_answer(predicted, expected, tolerance, atol=atol, rtol=rtol)
        if result["status"] == "not_evaluable":
            self.not_evaluable += 1
            self.reasons.append(f"Q{question.id}: {result['reason']}")
            return result

        self.evaluable += 1
        self.by_level[level]["evaluable"] += 1
        self.by_type[question_type]["evaluable"] += 1
        if result["status"] == "correct":
            self.correct += 1
            self.by_level[level]["correct"] += 1
            self.by_type[question_type]["correct"] += 1
        for key in ("exact_match", "numeric_match", "symbolic_match"):
            if result[key]:
                setattr(self, key, getattr(self, key) + 1)
        return result

    def summary(self) -> str:
        latency = self.latency_summary()
        accuracy = "N/A" if self.accuracy is None else f"{self.accuracy:.2%}"
        coverage = "N/A" if self.coverage is None else f"{self.coverage:.2%}"
        return (
            "\n=== Avaliação Matemática ===\n"
            f"Total: {self.total}; avaliáveis: {self.evaluable} ({coverage})\n"
            f"Corretas: {self.correct} ({accuracy})\n"
            "Exact/Numeric/Symbolic: "
            f"{self.exact_match}/{self.numeric_match}/{self.symbolic_match}\n"
            f"Não avaliáveis: {self.not_evaluable}\n"
            f"Latência média/mediana/P95/desvio: {latency['mean_s']}/{latency['median_s']}/"
            f"{latency['p95_s']}/{latency['stddev_s']}\n"
        )


def evaluate_predictions(
    questions: list[Any],
    predictions: list[str],
    expected: list[str],
    latencies: list[float],
    tolerance: float = DEFAULT_RTOL,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float | None = None,
) -> EvaluationMetrics:
    if not (len(questions) == len(predictions) == len(expected) == len(latencies)):
        raise ValueError("questions, predictions, expected e latencies devem ter o mesmo tamanho")
    metrics = EvaluationMetrics()
    for question, predicted, reference, latency in zip(questions, predictions, expected, latencies):
        metrics.add_result(question, predicted, reference, latency, tolerance, atol=atol, rtol=rtol)
    return metrics
