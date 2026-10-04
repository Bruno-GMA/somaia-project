"""Conservative mathematical answer validation.

Only short arithmetic expressions over ``x``, ``y`` and ``z`` are parsed. A
free-form explanation without an explicit final answer is reported as
``not_evaluable`` instead of guessing from the last number in the text.
"""

from __future__ import annotations

import math
import re
from typing import Any

from sympy import Symbol, simplify, sympify

DEFAULT_ATOL = 1e-9
DEFAULT_RTOL = 0.01
MAX_EXPRESSION_CHARS = 200
_NUMBER = r"[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?"
_NUMBER_RE = re.compile(rf"^{_NUMBER}$")
_ALLOWED_EXPRESSION_RE = re.compile(r"^[0-9xyzXYZ+\-*/^().,\s]+$")
_ANSWER_PREFIX_RE = re.compile(
    r"(?:^|[.;]\s*)(?:resposta|resultado|answer)(?:\s+final)?\s*[:=]\s*(.+)$",
    re.IGNORECASE,
)
_SYMBOLS = {name: Symbol(name) for name in ("x", "y", "z")}


def extract_expression(text: str) -> str | None:
    """Return one unambiguous candidate, never a number guessed from prose."""
    lines = [line.strip().strip("`") for line in text.splitlines() if line.strip()]
    candidates = [
        match.group(1).strip() for line in lines if (match := _ANSWER_PREFIX_RE.search(line))
    ]
    if candidates:
        if len(candidates) != 1:
            return None
        candidate = candidates[0]
    elif len(lines) == 1:
        candidate = lines[0]
    else:
        return None
    return candidate if _is_supported_expression(candidate) else None


def _is_supported_expression(expression: str) -> bool:
    return (
        0 < len(expression) <= MAX_EXPRESSION_CHARS
        and "__" not in expression
        and _ALLOWED_EXPRESSION_RE.fullmatch(expression) is not None
    )


def normalize_number(expression: str) -> float | None:
    """Parse only a complete decimal/scientific number, never a number in prose."""
    candidate = expression.strip().replace(",", ".")
    if _NUMBER_RE.fullmatch(candidate) is None:
        return None
    try:
        value = float(candidate)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def safe_sympify(expression: str) -> Any | None:
    """Parse the deliberately small grammar used by this project."""
    if not _is_supported_expression(expression):
        return None
    normalized = expression.replace("^", "**").replace(",", ".").lower()
    try:
        return sympify(normalized, locals=_SYMBOLS, evaluate=True)
    except (SyntaxError, TypeError, ValueError):
        return None


def expressions_equal(expr1: str, expr2: str, atol: float = DEFAULT_ATOL) -> bool:
    first = safe_sympify(expr1)
    second = safe_sympify(expr2)
    if first is None or second is None:
        return False
    try:
        difference = simplify(first - second)
        if difference == 0:
            return True
        value = float(difference)
        return math.isfinite(value) and abs(value) <= atol
    except (TypeError, ValueError):
        return False


def validate_answer(
    predicted: str,
    expected: str,
    tolerance: float = DEFAULT_RTOL,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float | None = None,
) -> dict[str, Any]:
    """Evaluate exact, numeric and symbolic checks independently.

    ``tolerance`` remains the compatibility alias for relative tolerance. The
    final status is ``correct``, ``incorrect`` or ``not_evaluable``.
    """
    effective_rtol = tolerance if rtol is None else rtol
    if atol < 0 or effective_rtol < 0:
        raise ValueError("atol e rtol devem ser não negativos")
    predicted_expression = extract_expression(predicted)
    expected_expression = extract_expression(expected)
    result: dict[str, Any] = {
        "status": "not_evaluable",
        "valid": False,
        "exact_match": False,
        "numeric_match": False,
        "symbolic_match": False,
        "predicted_expression": predicted_expression,
        "expected_expression": expected_expression,
        "predicted_value": None,
        "expected_value": None,
        "absolute_error": None,
        "relative_error": None,
        "atol": atol,
        "rtol": effective_rtol,
        "reason": None,
    }
    if predicted_expression is None or expected_expression is None:
        result["reason"] = "unsupported_or_ambiguous_expression"
        return result

    result["exact_match"] = predicted_expression == expected_expression
    predicted_number = normalize_number(predicted_expression)
    expected_number = normalize_number(expected_expression)
    result["predicted_value"] = predicted_number
    result["expected_value"] = expected_number
    if predicted_number is not None and expected_number is not None:
        absolute_error = abs(predicted_number - expected_number)
        result["absolute_error"] = absolute_error
        result["relative_error"] = absolute_error / max(abs(expected_number), atol)
        result["numeric_match"] = absolute_error <= atol + effective_rtol * abs(expected_number)

    result["symbolic_match"] = expressions_equal(predicted_expression, expected_expression, atol)
    result["valid"] = any(result[key] for key in ("exact_match", "numeric_match", "symbolic_match"))
    result["status"] = "correct" if result["valid"] else "incorrect"
    return result


def validate_batch(
    predictions: list[str],
    expected: list[str],
    tolerance: float = DEFAULT_RTOL,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float | None = None,
) -> dict[str, Any]:
    """Validate pairs and report coverage without treating unavailable data as wrong."""
    if len(predictions) != len(expected):
        raise ValueError("predictions e expected devem ter o mesmo tamanho")
    details = [
        validate_answer(prediction, reference, tolerance, atol=atol, rtol=rtol)
        for prediction, reference in zip(predictions, expected)
    ]
    evaluable = [detail for detail in details if detail["status"] != "not_evaluable"]
    correct = sum(detail["status"] == "correct" for detail in evaluable)
    return {
        "total": len(details),
        "evaluable": len(evaluable),
        "not_evaluable": len(details) - len(evaluable),
        "coverage": len(evaluable) / len(details) if details else None,
        "correct": correct,
        "accuracy": correct / len(evaluable) if evaluable else None,
        "details": details,
    }
