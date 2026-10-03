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

    res = evaluate_retrieval(query, retrieved, k=3, corpus=[ex.question for ex in retrieved])

    assert res["hit_at_k"] == 1.0
    assert res["mrr_at_k"] == 1.0  # primeiro relevante na posição 1
    assert res["recall_at_k"] == 1.0
    assert res["precision_at_k"] == 1.0
    assert res["ndcg_at_k"] == 1.0


def test_evaluate_retrieval_empty():
    query = _make_question(0, DifficultyLevel.SIMPLE, "aritmetica", "2 + 2")
    res = evaluate_retrieval(query, [], k=3, corpus=[])

    assert res["hit_at_k"] == 0.0
    assert res["mrr_at_k"] == 0.0
    assert res["recall_at_k"] == 0.0
    assert res["precision_at_k"] == 0.0
    assert res["ndcg_at_k"] == 0.0
    assert res["retriever_distances"] == []


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

    res = evaluate_retrieval(query, retrieved, k=3, corpus=[ex.question for ex in retrieved])

    assert res["hit_at_k"] == 1.0
    # primeiro relevante na posição 2 => mrr = 1/2
    assert res["mrr_at_k"] == 0.5
    assert res["recall_at_k"] == 1.0
    assert res["precision_at_k"] == 2 / 3
    assert 0.0 < res["ndcg_at_k"] < 1.0


def test_aggregate_averages():
    query = _make_question(1, DifficultyLevel.SIMPLE, "aritmetica", "+")
    other = _make_question(2, DifficultyLevel.SIMPLE, "aritmetica", "+")
    corpus = [query, other]
    hit = evaluate_retrieval(query, [RetrievedExample(other, 0.1)], 2, corpus=corpus)
    miss = evaluate_retrieval(query, [], 2, corpus=corpus)
    agg = aggregate([hit, miss])
    assert agg.n_queries == 2
    assert agg.to_dict()["recall_at_k"] == 0.5
    assert agg.to_dict()["precision_at_k"] == 0.25
    assert agg.payload["valid_counts"]["recall_at_k"] == 2
    assert "schema v2" in agg.summary()


def test_full_corpus_required_and_self_excluded():
    import pytest

    query = _make_question(1, DifficultyLevel.SIMPLE, "aritmetica", "+")
    relevant = [_make_question(i, query.level, "aritmetica", "+") for i in range(2, 7)]
    irrelevant = _make_question(7, DifficultyLevel.COMPLEX, "algebra", "x")
    corpus = [query, *relevant, irrelevant]
    result = evaluate_retrieval(
        query,
        [RetrievedExample(relevant[0], 0.1), RetrievedExample(irrelevant, 0.2)],
        2,
        corpus=corpus,
    )
    assert result["recall_at_k"] == 0.2
    assert result["precision_at_k"] == 0.5
    assert result["excluded_ids"] == ["1"]
    assert result["source"].startswith("pseudo_relevance")
    with pytest.raises(TypeError, match="corpus"):
        evaluate_retrieval(query, [], 2)


def test_qrels_missing_query_is_not_zero():
    from src.evaluation.ranking import Qrels

    query = _make_question(1, DifficultyLevel.SIMPLE, "aritmetica", "+")
    qrels = Qrels("human", "v1", "fixture", {})
    result = evaluate_retrieval(query, [], 2, corpus=[query], qrels=qrels)
    assert result["recall_at_k"] is None
    assert aggregate([result]).payload["valid_counts"]["recall_at_k"] == 0
