from src.evaluation.retrieval_metrics import (
    aggregate,
    evaluate_retrieval,
    is_relevant,
)
from src.models.enums import DifficultyLevel
from src.models.question import Question, RetrievedExample


def _make_question(qid: int, level: DifficultyLevel, qtype: str, op: str) -> Question:
    return Question(
        id=qid,
        text=f"questao {qid}",
        level=level,
        question_type=qtype,
        operation=op,
        answer="1",
    )


def test_is_relevant_level_and_type_match():
    q = _make_question(1, DifficultyLevel.SIMPLE, "aritmetica", "2 + 2")
    c = _make_question(2, DifficultyLevel.SIMPLE, "aritmetica", "3 + 3")
    assert is_relevant(q, c) is True
    assert is_relevant(q, c, strong=True) is False  # operation difere


def test_is_relevant_mismatch():
    q = _make_question(1, DifficultyLevel.SIMPLE, "aritmetica", "2 + 2")
    c = _make_question(2, DifficultyLevel.COMPLEX, "algebra", "x + 1")
    assert is_relevant(q, c) is False


def test_evaluate_retrieval_perfect_ranking():
    query = _make_question(0, DifficultyLevel.SIMPLE, "aritmetica", "2 + 2")
    retrieved = [
        RetrievedExample(
            question=_make_question(1, DifficultyLevel.SIMPLE, "aritmetica", "2 + 2"),
            distance=0.1,
        ),
        RetrievedExample(
            question=_make_question(2, DifficultyLevel.SIMPLE, "aritmetica", "3 + 3"),
            distance=0.2,
        ),
        RetrievedExample(
            question=_make_question(3, DifficultyLevel.SIMPLE, "aritmetica", "4 + 4"),
            distance=0.3,
        ),
    ]

    res = evaluate_retrieval(query, retrieved, k=3)

    assert res["hit_at_k"] == 1.0
    assert res["mrr_at_k"] == 1.0  # primeiro relevante na posição 1
    assert res["recall_at_k"] == 1.0
    assert res["precision_at_k"] == 1.0
    assert res["ndcg_at_k"] == 1.0


def test_evaluate_retrieval_empty():
    query = _make_question(0, DifficultyLevel.SIMPLE, "aritmetica", "2 + 2")
    res = evaluate_retrieval(query, [], k=3)

    assert res["hit_at_k"] == 0.0
    assert res["mrr_at_k"] == 0.0
    assert res["recall_at_k"] == 0.0
    assert res["precision_at_k"] == 0.0
    assert res["ndcg_at_k"] == 0.0
    assert res["avg_cosine_distance"] == 0.0
    assert res["avg_rrf_score"] == 0.0


def test_evaluate_retrieval_imperfect_ranking():
    query = _make_question(0, DifficultyLevel.SIMPLE, "aritmetica", "2 + 2")
    retrieved = [
        # primeiro irrelevante (level difere)
        RetrievedExample(
            question=_make_question(1, DifficultyLevel.COMPLEX, "algebra", "x"),
            distance=0.1,
        ),
        # segundo relevante
        RetrievedExample(
            question=_make_question(2, DifficultyLevel.SIMPLE, "aritmetica", "3 + 3"),
            distance=0.2,
        ),
        # terceiro relevante
        RetrievedExample(
            question=_make_question(3, DifficultyLevel.SIMPLE, "aritmetica", "4 + 4"),
            distance=0.3,
        ),
    ]

    res = evaluate_retrieval(query, retrieved, k=3)

    assert res["hit_at_k"] == 1.0
    # primeiro relevante na posição 2 => mrr = 1/2
    assert res["mrr_at_k"] == 0.5
    assert res["recall_at_k"] == 2 / 3
    assert res["precision_at_k"] == 2 / 3
    assert 0.0 < res["ndcg_at_k"] < 1.0


def test_aggregate_averages():
    per_query = [
        {"recall_at_k": 1.0, "precision_at_k": 1.0, "mrr_at_k": 1.0,
         "ndcg_at_k": 1.0, "hit_at_k": 1.0, "avg_cosine_distance": 0.1,
         "avg_rrf_score": 0.9},
        {"recall_at_k": 0.0, "precision_at_k": 0.0, "mrr_at_k": 0.0,
         "ndcg_at_k": 0.0, "hit_at_k": 0.0, "avg_cosine_distance": 0.3,
         "avg_rrf_score": 0.7},
    ]

    agg = aggregate(per_query)

    assert agg.n_queries == 2
    assert agg.recall_at_k == 0.5
    assert agg.precision_at_k == 0.5
    assert agg.mrr_at_k == 0.5
    assert agg.hit_at_k == 0.5
    assert agg.avg_cosine_distance == 0.2
    assert agg.avg_rrf_score == 0.8