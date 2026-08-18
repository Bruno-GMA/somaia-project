import pytest
from pathlib import Path
from src.io.csv_reader import StreamingCSVReader
from src.io.csv_writer import StreamingCSVWriter
from src.models.question import Question
from src.models.enums import DifficultyLevel


def test_streaming_reader_detect_column():
    reader = StreamingCSVReader(Path("dummy.csv"))
    assert reader.detect_question_column(["Question", "Answer"]) == "Question"
    assert reader.detect_question_column(["Pergunta", "Resposta"]) == "Pergunta"
    assert reader.detect_question_column(["Operation", "Result"]) == "Operation"
    assert reader.detect_question_column(["Coluna1", "Coluna2"]) == "Coluna1"


def test_streaming_reader_iter_questions(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""ID,Question,Comparação
1,2 + 2 = ?,4
2,3 * 3 = ?,9
""", encoding="utf-8")

    reader = StreamingCSVReader(csv_file)
    questions = list(reader.iter_questions())

    assert len(questions) == 2
    assert questions[0].id == 1
    assert questions[0].text == "2 + 2 = ?"
    assert questions[0].answer is None
    assert questions[0].ground_truth == "4"
    assert questions[1].id == 2
    assert questions[1].text == "3 * 3 = ?"
    assert questions[1].answer is None
    assert questions[1].ground_truth == "9"


def test_streaming_reader_iter_batches(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""Question
2 + 2 = ?
3 * 3 = ?
4 / 2 = ?
5 - 1 = ?
""", encoding="utf-8")

    reader = StreamingCSVReader(csv_file, chunk_size=2)
    batches = list(reader.iter_batches())

    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2


def test_streaming_writer(tmp_path):
    csv_file = tmp_path / "out.csv"
    questions = [
        Question(id=1, text="2 + 2 = ?", level=DifficultyLevel.SIMPLE, answer="4", ground_truth="4"),
        Question(id=2, text="3 * 3 = ?", level=DifficultyLevel.SIMPLE, answer="9", ground_truth="9"),
    ]

    with StreamingCSVWriter(csv_file) as writer:
        for q in questions:
            writer.write_question(q)

    content = csv_file.read_text(encoding="utf-8")
    assert "2 + 2 = ?" in content
    assert "4" in content
    assert "3 * 3 = ?" in content
    assert "9" in content