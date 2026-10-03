import pytest

from src.evaluation.validator import validate_answer, validate_batch


@pytest.mark.parametrize(
    ("predicted", "expected", "status"),
    [
        ("0", "0", "correct"),
        ("-2,5", "-2.5", "correct"),
        ("1/2", "0.5", "correct"),
        ("x + x", "2*x", "correct"),
        ("4", "5", "incorrect"),
    ],
)
def test_validation_handles_supported_math(predicted, expected, status):
    result = validate_answer(predicted, expected)
    assert result["status"] == status
    assert result["symbolic_match"] is True or status == "incorrect"


def test_explicit_final_answer_avoids_last_number_in_prose():
    result = validate_answer("Há 2 passos. Resultado final: 4", "4")
    assert result["status"] == "correct"
    ambiguous = validate_answer("Há 2 passos e o valor pode ser 4.", "4")
    assert ambiguous["status"] == "not_evaluable"
    assert ambiguous["reason"] == "unsupported_or_ambiguous_expression"


def test_numeric_tolerance_and_independent_checks():
    result = validate_answer("100.5", "100", atol=0.1, rtol=0.01)
    assert result["numeric_match"] is True
    assert result["exact_match"] is False
    assert result["symbolic_match"] is False
    assert validate_answer("100.5", "100", atol=0.1, rtol=0.001)["status"] == "incorrect"


def test_unsupported_and_batch_coverage():
    batch = validate_batch(["4", "resposta: quatro"], ["4", "4"])
    assert batch["evaluable"] == 1
    assert batch["not_evaluable"] == 1
    assert batch["coverage"] == 0.5
    with pytest.raises(ValueError):
        validate_batch(["1"], [])
