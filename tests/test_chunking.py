import pytest
from pathlib import Path
from src.pipeline.chunking import chunk_question
from src.config import settings


def test_chunk_question_short():
    text = "2 + 2 = ?"
    chunks = chunk_question(text, max_words=10, overlap=2)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_question_long():
    text = " ".join([f"word{i}" for i in range(100)])
    chunks = chunk_question(text, max_words=30, overlap=5)

    assert len(chunks) > 1
    # Check overlap
    for i in range(len(chunks) - 1):
        words_i = set(chunks[i].split()[-5:])
        words_next = set(chunks[i + 1].split()[:5])
        assert len(words_i & words_next) >= 3  # At least some overlap


def test_chunk_question_exact_boundary():
    text = " ".join([f"w{i}" for i in range(30)])
    chunks = chunk_question(text, max_words=30, overlap=5)
    assert len(chunks) == 1