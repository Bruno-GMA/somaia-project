import pytest
from src.models.question import Question
from src.models.enums import DifficultyLevel, QuestionType


def test_difficulty_level_from_str():
    assert DifficultyLevel.from_str("Simples") == DifficultyLevel.SIMPLE
    assert DifficultyLevel.from_str("simples") == DifficultyLevel.SIMPLE
    assert DifficultyLevel.from_str("Intermediário") == DifficultyLevel.INTERMEDIATE
    assert DifficultyLevel.from_str("intermediario") == DifficultyLevel.INTERMEDIATE
    assert DifficultyLevel.from_str("Complexo") == DifficultyLevel.COMPLEX
    assert DifficultyLevel.from_str("unknown") == DifficultyLevel.UNKNOWN


def test_question_type_from_str():
    assert QuestionType.from_str("Non-contextualized") == QuestionType.NON_CONTEXTUALIZED
    assert QuestionType.from_str("Contextualized") == QuestionType.CONTEXTUALIZED


def test_question_has_answer():
    q1 = Question(id=1, text="2+2", level=DifficultyLevel.SIMPLE, answer="4")
    q2 = Question(id=2, text="3+3", level=DifficultyLevel.SIMPLE, answer="")
    q3 = Question(id=3, text="4+4", level=DifficultyLevel.SIMPLE, answer=None)

    assert q1.has_answer is True
    assert q2.has_answer is False
    assert q3.has_answer is False