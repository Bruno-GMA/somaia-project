import csv
from collections.abc import Generator
from pathlib import Path

from src.models.enums import DifficultyLevel
from src.models.question import Question


class StreamingCSVReader:
    def __init__(self, path: Path, chunk_size: int = 500):
        self.path = path
        self.chunk_size = chunk_size

    def detect_question_column(self, headers: list[str]) -> str:
        candidates = ["question", "pergunta", "problema", "enunciado", "operation", "operação"]
        lower = [h.lower().strip() for h in headers]
        for c in candidates:
            if c in lower:
                return headers[lower.index(c)]
        return headers[0]

    def iter_questions(self) -> Generator[Question, None, None]:
        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = list(reader.fieldnames or [])
            if not headers:
                return

            qcol = self.detect_question_column(headers)

            for row in reader:
                qid = None
                if "ID" in row and row["ID"]:
                    try:
                        qid = int(row["ID"])
                    except ValueError:
                        pass

                yield Question(
                    id=qid,
                    text=row[qcol].strip(),
                    operation=row.get("Operation") or row.get("Operação"),
                    level=DifficultyLevel.from_str(row.get("Level") or row.get("Nível") or ""),
                    question_type=row.get("Type") or row.get("Tipo"),
                    answer=None,
                    ground_truth=(
                        row.get("Comparação")
                        or row.get("Comparacao")
                        or row.get("Resposta")
                        or None
                    ),
                )

    def iter_batches(self) -> Generator[list[Question], None, None]:
        batch = []
        for q in self.iter_questions():
            batch.append(q)
            if len(batch) >= self.chunk_size:
                yield batch
                batch = []
        if batch:
            yield batch
