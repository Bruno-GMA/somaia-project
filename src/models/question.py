from dataclasses import dataclass, field
from typing import Any

from src.models.enums import DifficultyLevel


@dataclass(slots=True)
class Question:
    id: int | None
    text: str
    level: DifficultyLevel
    operation: str | None = None
    question_type: str | None = None
    answer: str | None = None
    ground_truth: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_answer(self) -> bool:
        return self.answer is not None and self.answer.strip() != ""

    @property
    def has_ground_truth(self) -> bool:
        return self.ground_truth is not None and self.ground_truth.strip() != ""


@dataclass(slots=True)
class Chunk:
    id: int | None
    question_id: int
    index: int
    text: str
    embedding: list[float] | None = None


@dataclass(slots=True)
class RetrievedExample:
    question: Question
    distance: float
    chunk_text: str | None = None
