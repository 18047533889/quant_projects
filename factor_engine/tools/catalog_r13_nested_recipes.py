"""Narrow reviewed repairs for the round-13 nested-formula residuals."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class R13NestedRecipeResult:
    formula: str
    changes: tuple[str, ...] = ()


_MAX_FORMULA_BYTES = 65_536


def _supports_reviewed_transport_split(logic: str) -> bool:
    """Require the review prose to state both canonical window assignments."""
    compact = re.sub(r"\s+", "", logic).lower()
    return "recent_window=20" in compact and "old_window=40" in compact


def _offset(lines: list[str], lineno: int, byte_col: int) -> int:
    prefix = lines[lineno - 1].encode("utf-8")[:byte_col]
    return sum(len(line) for line in lines[: lineno - 1]) + len(prefix.decode("utf-8"))


def _span(lines: list[str], node: ast.AST) -> tuple[int, int]:
    if node.end_lineno is None or node.end_col_offset is None:
        raise ValueError("AST node has no complete source location")
    return (
        _offset(lines, node.lineno, node.col_offset),
        _offset(lines, node.end_lineno, node.end_col_offset),
    )


def _keyword_map(node: ast.Call) -> dict[str, ast.keyword] | None:
    if any(keyword.arg is None for keyword in node.keywords):
        return None
    result = {keyword.arg: keyword for keyword in node.keywords}
    return result if len(result) == len(node.keywords) else None


def migrate_r13_nested_formula(
    formula: str, *, logic: str = "", enabled: bool = False,
) -> R13NestedRecipeResult:
    """Rewrite only explicitly reviewed quantile-transport ``window=60`` calls.

    A scalar 60-row horizon does not determine the canonical two-window split.
    The recipe therefore requires the review prose to explicitly state both
    ``recent_window=20`` and ``old_window=40``.  Merely naming quantile
    transport is not evidence for that semantic redesign.
    """
    if not isinstance(formula, str):
        raise TypeError("formula must be a string")
    if not isinstance(logic, str):
        raise TypeError("logic must be a string")
    if len(formula.encode("utf-8")) > _MAX_FORMULA_BYTES:
        raise ValueError("input budget exceeded")
    if not enabled:
        return R13NestedRecipeResult(formula)
    tree = ast.parse(formula, mode="eval")
    if not _supports_reviewed_transport_split(logic):
        return R13NestedRecipeResult(formula)
    lines = formula.splitlines(keepends=True)
    edits: list[tuple[int, int, str]] = []
    changes: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name not in {"ts_quantile_transport_slope", "ts_quantile_transport_curvature"}:
            continue
        keywords = _keyword_map(node)
        if keywords is None:
            continue
        window = keywords.get("window")
        if (
            len(node.args) != 1
            or set(keywords) != {"window"}
            or not isinstance(window.value, ast.Constant)
            or type(window.value.value) is not int
            or window.value.value != 60
        ):
            continue
        arg_start, arg_end = _span(lines, node.args[0])
        replacement = (
            f"{name}({formula[arg_start:arg_end]}, "
            "recent_window=20, old_window=40)"
        )
        start, end = _span(lines, node)
        edits.append((start, end, replacement))
        changes.append(
            f"SEMANTIC_REDESIGN {name}: window=60 -> recent_window=20, "
            "old_window=40; split explicitly approved in review logic"
        )
    ordered = sorted(edits)
    if any(start < previous_end for (_, previous_end, _), (start, _, _) in zip(ordered, ordered[1:])):
        return R13NestedRecipeResult(formula)
    for start, end, replacement in reversed(ordered):
        formula = formula[:start] + replacement + formula[end:]
    return R13NestedRecipeResult(formula, tuple(changes))
