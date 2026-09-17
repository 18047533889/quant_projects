"""Opt-in semantic redesigns for additional reviewed catalog failures."""
from __future__ import annotations

import ast

from factor_engine.tools.catalog_recipe_migration import RecipeMigration

_MAX_FORMULA_BYTES = 65_536
_CANONICAL_EVENT_CALLS = frozenset({
    "ashare_limit_up_touch", "ashare_limit_down_touch",
    "ashare_limit_failed", "index_entry_exit_event",
})


def _offset(lines: list[str], lineno: int, byte_col: int) -> int:
    prefix = lines[lineno - 1].encode("utf-8")[:byte_col]
    return sum(len(line) for line in lines[:lineno - 1]) + len(prefix.decode("utf-8"))


def _span(lines: list[str], node: ast.AST) -> tuple[int, int]:
    if node.end_lineno is None or node.end_col_offset is None:
        raise ValueError("AST node has no complete source location")
    return (_offset(lines, node.lineno, node.col_offset),
            _offset(lines, node.end_lineno, node.end_col_offset))


def _literal(node: ast.AST, value: object) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) is type(value) and node.value == value


def _has_logic(logic: str, *needles: str) -> bool:
    normalized = logic.casefold()
    return any(needle.casefold() in normalized for needle in needles)


def _source(formula: str, lines: list[str], node: ast.AST) -> str:
    start, end = _span(lines, node)
    return formula[start:end]


def _is_ret_response(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in {"ret", "ret_1d"}
    return (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "field" and bool(node.args)
        and _literal(node.args[0], "ret")
    )


def migrate_catalog_additional_failure_formula(
    formula: str, *, logic: str = "", enabled: bool = False
) -> RecipeMigration:
    """Rewrite only exact user-reviewed shapes; otherwise fail closed."""
    if not isinstance(formula, str):
        raise TypeError("formula must be a string")
    if not isinstance(logic, str):
        raise TypeError("logic must be a string")
    if len(formula.encode("utf-8")) > _MAX_FORMULA_BYTES:
        raise ValueError("formula exceeds migration input budget")
    if not enabled:
        return RecipeMigration(formula, ())

    tree = ast.parse(formula, mode="eval")
    lines = formula.splitlines(keepends=True) or [""]
    edits: list[tuple[int, int, str]] = []
    changes: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        replacement = None
        change = None
        if (
            name == "ts_transition_count" and not node.keywords and len(node.args) == 3
            and isinstance(node.args[0], ast.Name) and node.args[0].id == "ret"
            and _literal(node.args[1], 20) and _literal(node.args[2], 0.0)
            and _has_logic(logic, "方向切换", "state transition", "transition frequency")
        ):
            replacement = "ts_transition_count(gt(ret, 0.0), 20, 'break')"
            change = (
                "SEMANTIC_REDESIGN ts_transition_count: legacy direction threshold 0.0 "
                "becomes gt(ret, 0.0); window=20; missing_policy='break'"
            )
        elif (
            name == "ts_gap_fill_ratio" and not node.keywords and len(node.args) == 4
            and all(isinstance(arg, ast.Name) and arg.id == expected
                    for arg, expected in zip(node.args[:3], ("close", "open", "pre_close")))
            and _literal(node.args[3], 0.01)
            and _has_logic(logic, "缺口", "gap fill", "gap repair")
        ):
            replacement = "ts_gap_fill_ratio(close, open, pre_close, 60)"
            change = (
                "SEMANTIC_REDESIGN ts_gap_fill_ratio: legacy 0.01 cannot be interpreted "
                "as the canonical integer window; use canonical default window=60"
            )
        elif (
            name == "event_historical_response_mean" and not node.keywords
            and len(node.args) == 2 and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Name)
            and node.args[0].func.id in _CANONICAL_EVENT_CALLS
            and _is_ret_response(node.args[1])
            and _has_logic(logic, "事件", "event", "响应", "response", "触板", "炸板")
        ):
            event = _source(formula, lines, node.args[0])
            response = _source(formula, lines, node.args[1])
            replacement = f"event_historical_response_mean({response}, {event})"
            # This recipe repairs argument order only; Analyzer remains the
            # authority for whether the expression has a legal event type.
            change = (
                "SEMANTIC_REDESIGN event_historical_response_mean: event expression first "
                "argument moved to event position; return response moved to response position; "
                "argument reordering does not certify the event input type"
            )
        if replacement is not None and change is not None:
            start, end = _span(lines, node)
            edits.append((start, end, replacement))
            changes.append(change)

    ordered = sorted(edits)
    if any(next_start < end for (_, end, _), (next_start, _, _) in zip(ordered, ordered[1:])):
        return RecipeMigration(formula, ())
    migrated = formula
    for start, end, replacement in reversed(ordered):
        migrated = migrated[:start] + replacement + migrated[end:]
    return RecipeMigration(migrated, tuple(changes))
