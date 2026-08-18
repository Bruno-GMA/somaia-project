import sys
from pathlib import Path
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import Settings
from src.models.question import Question
from src.models.enums import DifficultyLevel, QuestionType


@pytest.fixture
def settings():
    return Settings(
        ollama_base_url="http://localhost:11434",
        chat_model="gemma2:2b",
        embed_model="nomic-embed-text",
        db_path=":memory:",
    )


@pytest.fixture
def sample_questions():
    return [
        Question(id=1, text="2 + 2 = ?", level=DifficultyLevel.SIMPLE, operation="2 + 2"),
        Question(id=2, text="Integral de x^2", level=DifficultyLevel.COMPLEX, operation="∫x²dx"),
        Question(id=3, text="3 * 4 = ?", level=DifficultyLevel.SIMPLE, operation="3 * 4", answer="12"),
    ]


@pytest.fixture
def temp_csv(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""ID,Type,Level,Question,Operation,Comparação
1,Non-contextualized,Simples,2 + 2 = ?,2 + 2,4
2,Non-contextualized,Intermediário,3 * 4 = ?,3 * 4,12
3,Non-contextualized,Complexo,Integral de x^2,∫x²dx,
""", encoding="utf-8")
    return csv_file