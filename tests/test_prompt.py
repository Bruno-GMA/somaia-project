import pytest
from pathlib import Path
from src.pipeline.prompt import build_prompt
from src.models.question import Question, RetrievedExample
from src.models.enums import DifficultyLevel


def test_build_prompt_no_examples():
    prompt = build_prompt("2 + 2 = ?", [])
    assert "Você é um especialista em matemática" in prompt
    assert "Questão: 2 + 2 = ?" in prompt
    assert "Resposta:" in prompt
    assert "Exemplo" not in prompt


def test_build_prompt_with_examples():
    ex1 = RetrievedExample(
        question=Question(id=1, text="1 + 1 = ?", level=DifficultyLevel.SIMPLE, answer="2"),
        distance=0.1,
    )
    ex2 = RetrievedExample(
        question=Question(id=2, text="2 + 2 = ?", level=DifficultyLevel.SIMPLE, answer="4"),
        distance=0.2,
    )

    prompt = build_prompt("3 + 3 = ?", [ex1, ex2])

    assert "Exemplo 1" in prompt
    assert "Exemplo 2" in prompt
    assert "1 + 1 = ?" in prompt
    assert "Resposta: 2" in prompt
    assert "2 + 2 = ?" in prompt
    assert "Resposta: 4" in prompt
    assert "3 + 3 = ?" in prompt