from src.models.enums import DifficultyLevel
from src.models.question import Question, RetrievedExample
from src.retrieval.reranker import rerank


def _candidate(identifier: int, distance: float, operation: str) -> RetrievedExample:
    return RetrievedExample(
        Question(identifier, f"questão {identifier}", DifficultyLevel.SIMPLE, operation=operation),
        distance,
    )


def test_reranker_is_deterministic_and_does_not_create_candidates():
    candidates = [_candidate(1, 0.1, "subtração"), _candidate(2, 0.2, "soma")]
    selected, trace = rerank("Qual é o total da soma?", candidates, final_k=2)
    assert [item.question.id for item in selected] == [1, 2]
    assert {item.candidate_id for item in trace} == {1, 2}
    assert trace[0].original_rank == 1


def test_reranker_stably_deduplicates_and_limits_final_k():
    first = _candidate(1, 0.1, "soma")
    selected, trace = rerank("soma", [first, first, _candidate(2, 0.2, "soma")], final_k=1)
    assert [item.question.id for item in selected] == [1]
    assert len(trace) == 2
