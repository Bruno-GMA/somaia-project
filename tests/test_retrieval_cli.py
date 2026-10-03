"""SQLite + HybridRetriever integration; only the Ollama embedding call is mocked."""

import json

from typer.testing import CliRunner

from src.cli.main import app
from src.config import settings
from src.db.connection import ConnectionManager
from src.db.repository import EmbeddingRepository, QuestionRepository
from src.db.schema import init_database
from src.embedding.provider import OllamaEmbeddingProvider
from src.models.enums import DifficultyLevel
from src.models.question import Question


def test_cli_real_sqlite_corpus_qrels_and_no_overwrite(tmp_path, monkeypatch):
    path = tmp_path / "retrieval.db"
    init_database(str(path))
    cm = ConnectionManager(str(path))
    repo = QuestionRepository(cm)
    try:
        for i in range(0, 5):
            repo.upsert(
                Question(
                    id=i,
                    text=f"problema numero {i}",
                    level=DifficultyLevel.SIMPLE,
                    question_type="math",
                    operation="+",
                    answer="1" if i < 4 else "",
                )
            )
            EmbeddingRepository(cm).store_question_embedding(i, [1.0] + [0.0] * 767)
        assert [q.id for q in repo.get_retrieval_corpus()] == [0, 1, 2, 3]
    finally:
        cm.close_all()
    calls = []

    async def embed(self, text):
        calls.append(text)
        return [1.0] + [0.0] * 767

    monkeypatch.setattr(OllamaEmbeddingProvider, "embed", embed)
    monkeypatch.setattr(settings, "db_path", str(path))
    qrels = tmp_path / "qrels.json"
    qrels.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "test-annotated",
                "version": "1",
                "corpus_id": "test",
                "qrels": {"1": {"0": 0, "1": 3, "2": 1, "3": 1}, "2": {}},
            }
        )
    )
    output = tmp_path / "metrics.json"
    args = [
        "evaluate-retrieval",
        str(path),
        "--top-k",
        "3",
        "--qrels",
        str(qrels),
        "--output",
        str(output),
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output + repr(result.exception)
    data = json.loads(output.read_text())
    assert data["schema_version"] == 2
    assert data["n_queries"] == 4
    assert data["valid_counts"]["recall_at_k"] == 2
    assert data["status_counts"] == {"ok": 1, "no_relevant": 1, "unjudged": 2}
    first = next(row for row in data["per_query"] if row["query_id"] == "1")
    zero = next(row for row in data["per_query"] if row["query_id"] == "0")
    assert "0" not in zero["retrieved_ids"]
    assert first["n_relevant"] == 2
    assert first["excluded_ids"] == ["1"]
    assert "1" not in first["retrieved_ids"]
    assert first["recall_at_k"] == 1
    original = output.read_bytes()
    assert CliRunner().invoke(app, args).exit_code != 0
    assert output.read_bytes() == original
    assert len(calls) == 4
    assert CliRunner().invoke(app, ["evaluate-retrieval", str(path), "--top-k", "0"]).exit_code != 0

    pseudo_output = tmp_path / "pseudo.json"
    result = CliRunner().invoke(
        app, ["evaluate-retrieval", str(path), "-k", "2", "-o", str(pseudo_output)]
    )
    assert result.exit_code == 0, repr(result.exception)
    pseudo = json.loads(pseudo_output.read_text())
    assert pseudo["sources"] == ["pseudo_relevance_level_type_operation"]
    assert pseudo["valid_counts"]["recall_at_k"] == 4
