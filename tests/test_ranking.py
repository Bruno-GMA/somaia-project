"""Contract examples calculated by hand, independent of model/retriever behavior."""

import json
import math

import pytest

from src.evaluation.ranking import (
    METRIC_KEYS,
    Qrels,
    aggregate_rankings,
    evaluate_ranking,
)


def score(ranking, grades, k=2, corpus=None, excluded=None):
    return evaluate_ranking(
        ranking,
        k,
        corpus_ids=corpus or list("abcdef"),
        grades=grades,
        source="fixture",
        relevance_version="1",
        corpus_id="six-documents",
        excluded_ids=excluded,
    )


def test_precision_and_recall_have_different_denominators():
    result = score(["a", "f"], {**dict.fromkeys("abcde", 1), "f": 0})
    assert result["precision_at_k"] == 0.5
    assert result["recall_at_k"] == 0.2
    assert result["mrr_at_k"] == 1
    assert result["ndcg_at_k"] == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert result["n_unjudged_retrieved"] == 0
    assert result["relevance_grades"] == [1, 0]
    assert result["judgment_coverage"] == 1


def test_incomplete_qrels_expose_unknown_documents():
    result = score(["a", "f"], dict.fromkeys("abcde", 1))
    assert result["n_unjudged_retrieved"] == 1
    assert result["relevance_grades"] == [1, None]
    assert result["judgment_coverage"] == 5 / 6


def test_idcg_includes_unretrieved_best_document_and_rank_matters():
    grades = {"a": 1, "b": 2, "c": 0}
    assert score(["a"], grades, k=1)["ndcg_at_k"] == pytest.approx(1 / 3)
    ideal = 3 + 1 / math.log2(3)
    imperfect = score(["a", "b"], grades)
    assert imperfect["ndcg_at_k"] == pytest.approx((1 + 3 / math.log2(3)) / ideal)
    assert score(["b", "a"], grades)["ndcg_at_k"] == 1


def test_single_expected_id_short_list_and_missing_hit():
    result = score(["a"], {"a": 1}, k=3)
    assert result["precision_at_k"] == 1 / 3
    assert result["recall_at_k"] == result["hit_at_k"] == 1
    assert result["ndcg_at_k"] == 1
    missed = score(["f", "e"], {"a": 1})
    assert all(missed[key] == 0 for key in METRIC_KEYS)
    assert score(["f", "a"], {"a": 1})["mrr_at_k"] == 0.5


def test_excluded_self_removed_from_recall_and_idcg():
    result = score(["a"], {"a": 1, "b": 3}, k=1, excluded=["b"])
    assert result["recall_at_k"] == result["ndcg_at_k"] == 1
    assert result["n_relevant"] == 1
    assert result["excluded_ids"] == ["b"]
    assert result["n_eligible"] == 5


def test_empty_ranking_no_relevant_and_unjudged_differ():
    empty = score([], {"a": 1})
    none_relevant = score(["a"], {})
    unknown = score(["a"], None)
    assert all(empty[key] == none_relevant[key] == 0 for key in METRIC_KEYS)
    assert none_relevant["status"] == "no_relevant"
    assert all(unknown[key] is None for key in METRIC_KEYS)
    aggregate = aggregate_rankings([score(["a"], {"a": 1}), none_relevant, unknown])
    assert aggregate["metrics"]["recall_at_k"] == 0.5
    assert aggregate["valid_counts"]["recall_at_k"] == 2
    assert aggregate["status_counts"]["unjudged"] == 1
    assert all(value is None for value in aggregate_rankings([])["metrics"].values())
    assert all(value is None for value in aggregate_rankings([unknown])["metrics"].values())


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_k(k):
    with pytest.raises(ValueError, match="inteiro positivo"):
        score([], {}, k=k)


@pytest.mark.parametrize("ranking", [["a", "a"], [""], [None], [True], [" a"], ["outside"]])
def test_invalid_ranking(ranking):
    with pytest.raises(ValueError):
        score(ranking, {})


@pytest.mark.parametrize("grades", [{"a": -1}, {"a": 0.5}, {"a": True}, {"z": 1}])
def test_invalid_judgments(grades):
    with pytest.raises(ValueError):
        score([], grades)


def test_duplicate_corpus_and_excluded_rank_rejected():
    with pytest.raises(ValueError, match="duplicados"):
        score([], {}, corpus=["a", "a"])
    with pytest.raises(ValueError, match="inelegíveis"):
        score(["a"], {"a": 1}, excluded=["a"])
    with pytest.raises(ValueError, match="pertencer"):
        score([], {}, excluded=["outside"])


def test_large_grades_do_not_overflow():
    assert score(["a", "b"], {"a": 2000, "b": 1999})["ndcg_at_k"] == 1


def qrels_json():
    return {
        "schema_version": 1,
        "source": "human",
        "version": "v1",
        "corpus_id": "corpus-v1",
        "qrels": {"query": {"a": 2, "b": 0}, "judged_empty": {}},
    }


def test_qrels_version_source_and_missing_query():
    data = Qrels.from_json(json.dumps(qrels_json()))
    data.validate_corpus(["a", "b", "c"])
    assert data.judgments.get("absent") is None
    assert data.judgments["judged_empty"] == {}
    assert data.source == "human" and data.version == "v1"
    with pytest.raises(ValueError, match="fora do corpus"):
        data.validate_corpus(["a"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", ""),
        ("version", None),
        ("corpus_id", ""),
        ("schema_version", 2),
        ("schema_version", True),
    ],
)
def test_invalid_qrels_metadata(field, value):
    data = qrels_json()
    data[field] = value
    with pytest.raises(ValueError):
        Qrels.from_json(json.dumps(data))


def test_duplicate_json_keys_rejected():
    with pytest.raises(ValueError, match="duplicada"):
        Qrels.from_json('{"schema_version": 1, "schema_version": 1}')
