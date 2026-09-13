import csv
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from src.models.question import Question


class StreamingCSVWriter:
    def __init__(self, path: Path, fieldnames: list[str] | None = None):
        self.path = path
        self.fieldnames = fieldnames or [
            "ID", "Type", "Level", "Question", "Operation", "Comparação", "resposta"
        ]
        self._file: TextIO | None = None
        self._writer: csv.DictWriter[str] | None = None

    def __enter__(self) -> "StreamingCSVWriter":
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames, extrasaction="ignore")
        self._writer.writeheader()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        if self._file:
            self._file.close()

    def write_question(self, q: Question) -> None:
        if not self._writer:
            raise RuntimeError("Writer not initialized. Use context manager.")

        row = {
            "ID": q.id,
            "Type": q.question_type,
            "Level": q.level.value,
            "Question": q.text,
            "Operation": q.operation,
            "Comparação": q.ground_truth,
            "resposta": q.answer,
        }
        self._writer.writerow(row)

    @contextmanager
    def append_mode(self) -> "Generator[StreamingCSVWriter, None, None]":
        """Context manager for appending to existing file."""
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter[str](
            self._file, fieldnames=self.fieldnames, extrasaction="ignore"
        )
        try:
            yield self
        finally:
            if self._file:
                self._file.close()
