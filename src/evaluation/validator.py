import re
from typing import Any

from sympy import N, simplify
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


def normalize_number(text: str) -> float | None:
    """Extract and normalize a number from text."""
    # Handle comma as decimal separator
    text = text.replace(",", ".")
    # Find numbers (integers, decimals, scientific notation)
    matches = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    if matches:
        try:
            return float(matches[-1])  # Take the last number found
        except ValueError:
            pass
    return None


def extract_expression(text: str) -> str | None:
    """Extract mathematical expression from text."""
    # Remove common prefixes
    text = text.strip()
    # Remove "Resposta:" or similar
    for prefix in ["resposta:", "resposta", "answer:", "answer", "resultado:", "resultado"]:
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
            break
    # Take first line if multi-line
    text = text.split("\n")[0].strip()
    return text if text else None


def safe_sympify(expr_str: str) -> Any | None:
    """Safely parse expression with sympy."""
    transformations = standard_transformations + (implicit_multiplication_application,)
    try:
        return parse_expr(expr_str, transformations=transformations)
    except Exception:
        return None


def expressions_equal(expr1: str, expr2: str, tolerance: float = 1e-9) -> bool:
    """Check if two mathematical expressions are equal."""
    try:
        e1 = safe_sympify(expr1)
        e2 = safe_sympify(expr2)
        if e1 is None or e2 is None:
            return False
        # Simplify and compare numerically
        diff = simplify(e1 - e2)
        if diff == 0:
            return True
        # Numerical evaluation
        val: float = float(N(diff))
        return abs(val) < tolerance
    except Exception:
        return False


def validate_answer(predicted: str, expected: str, tolerance: float = 0.01) -> dict[str, Any]:
    """
    Validate predicted answer against expected.
    Returns dict with validation result and details.
    """
    result: dict[str, Any] = {
        "valid": False,
        "exact_match": False,
        "numeric_match": False,
        "symbolic_match": False,
        "predicted_value": None,
        "expected_value": None,
        "error": None,
    }

    pred_expr = extract_expression(predicted)
    exp_expr = extract_expression(expected)

    if not pred_expr or not exp_expr:
        result["error"] = "Could not extract expressions"
        return result

    # Exact string match (normalized)
    if pred_expr.strip() == exp_expr.strip():
        result["valid"] = True
        result["exact_match"] = True
        return result

    # Try numeric comparison
    pred_num = normalize_number(pred_expr)
    exp_num = normalize_number(exp_expr)

    if pred_num is not None and exp_num is not None:
        result["predicted_value"] = pred_num
        result["expected_value"] = exp_num
        rel_error = abs(pred_num - exp_num) / max(abs(exp_num), 1e-9)
        if rel_error <= tolerance:
            result["valid"] = True
            result["numeric_match"] = True
            return result

    # Try symbolic comparison
    if expressions_equal(pred_expr, exp_expr, tolerance):
        result["valid"] = True
        result["symbolic_match"] = True
        return result

    result["error"] = f"Values differ: predicted={pred_expr}, expected={exp_expr}"
    return result


def validate_batch(
    predictions: list[str], expected: list[str], tolerance: float = 0.01
) -> dict[str, Any]:
    """Validate a batch of predictions."""
    results = []
    correct = 0
    for pred, exp in zip(predictions, expected):
        res = validate_answer(pred, exp, tolerance)
        results.append(res)
        if res["valid"]:
            correct += 1

    return {
        "total": len(predictions),
        "correct": correct,
        "accuracy": correct / len(predictions) if predictions else 0.0,
        "details": results,
    }
