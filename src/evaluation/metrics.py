from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.evaluation.validator import validate_batch


@dataclass
class EvaluationMetrics:
    total: int = 0
    correct: int = 0
    exact_match: int = 0
    numeric_match: int = 0
    symbolic_match: int = 0
    by_level: dict[str, dict[str, int]] = field(default_factory=dict)
    by_type: dict[str, dict[str, int]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def exact_match_rate(self) -> float:
        return self.exact_match / self.total if self.total > 0 else 0.0

    @property
    def avg_latency(self) -> float:
        return float(np.mean(self.latencies)) if self.latencies else 0.0

    @property
    def p95_latency(self) -> float:
        return float(np.percentile(self.latencies, 95)) if self.latencies else 0.0

    def add_result(
        self,
        question: Any,  # Question object
        predicted: str,
        expected: str,
        latency: float,
        tolerance: float = 0.01,
    ) -> dict[str, Any]:
        self.total += 1
        self.latencies.append(latency)

        level = question.level.value if question.level else "Unknown"
        qtype = question.question_type or "Unknown"

        # Initialize level/type counters
        for key, container in [(level, self.by_level), (qtype, self.by_type)]:
            if key not in container:
                container[key] = {"total": 0, "correct": 0}

        res = validate_batch([predicted], [expected], tolerance)["details"][0]

        if res["valid"]:
            self.correct += 1
            self.by_level[level]["correct"] += 1
            self.by_type[qtype]["correct"] += 1
            if res["exact_match"]:
                self.exact_match += 1
            if res["numeric_match"]:
                self.numeric_match += 1
            if res["symbolic_match"]:
                self.symbolic_match += 1
        else:
            self.errors.append(f"Q{question.id}: {res.get('error', 'Unknown error')}")

        self.by_level[level]["total"] += 1
        self.by_type[qtype]["total"] += 1

        return dict(res)
        self.total += 1
        self.latencies.append(latency)

        level = question.level.value if question.level else "Unknown"
        qtype = question.question_type or "Unknown"

        # Initialize level/type counters
        for key, container in [(level, self.by_level), (qtype, self.by_type)]:
            if key not in container:
                container[key] = {"total": 0, "correct": 0}

        res = validate_batch([predicted], [expected], tolerance)["details"][0]

        if res["valid"]:
            self.correct += 1
            self.by_level[level]["correct"] += 1
            self.by_type[qtype]["correct"] += 1
            if res["exact_match"]:
                self.exact_match += 1
            if res["numeric_match"]:
                self.numeric_match += 1
            if res["symbolic_match"]:
                self.symbolic_match += 1
        else:
            self.errors.append(f"Q{question.id}: {res.get('error', 'Unknown error')}")

        self.by_level[level]["total"] += 1
        self.by_type[qtype]["total"] += 1

        return res

    def summary(self) -> str:
        lines = [
            "\n=== Avaliação Detalhada ===",
            f"Total: {self.total}",
            f"Corretas: {self.correct} ({self.accuracy:.2%})",
            f"Exact Match: {self.exact_match} ({self.exact_match_rate:.2%})",
            f"Numeric Match: {self.numeric_match}",
            f"Symbolic Match: {self.symbolic_match}",
            f"Latência média: {self.avg_latency:.2f}s",
            f"Latência P95: {self.p95_latency:.2f}s",
            "\n--- Por Nível ---",
        ]
        for level, stats in sorted(self.by_level.items()):
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            lines.append(f"  {level}: {stats['correct']}/{stats['total']} ({acc:.2%})")

        lines.append("\n--- Por Tipo ---")
        for qtype, stats in sorted(self.by_type.items()):
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            lines.append(f"  {qtype}: {stats['correct']}/{stats['total']} ({acc:.2%})")

        if self.errors:
            lines.append(f"\nErros ({len(self.errors)}):")
            for err in self.errors[:10]:
                lines.append(f"  - {err}")
            if len(self.errors) > 10:
                lines.append(f"  ... e mais {len(self.errors) - 10}")

        return "\n".join(lines)


def evaluate_predictions(
    questions: list[Any],
    predictions: list[str],
    expected: list[str],
    latencies: list[float],
    tolerance: float = 0.01,
) -> EvaluationMetrics:
    """Evaluate a batch of predictions."""
    metrics = EvaluationMetrics()
    for q, pred, exp, lat in zip(questions, predictions, expected, latencies):
        metrics.add_result(q, pred, exp, lat, tolerance)
    return metrics
