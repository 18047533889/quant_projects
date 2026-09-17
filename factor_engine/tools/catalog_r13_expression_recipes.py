"""Opt-in, exact-shape expression recipes reviewed for catalog R13.

These recipes repair syntax/arity only. A successful rewrite or compile is
not evidence that a factor executed successfully or has economic validity.
"""
from __future__ import annotations

import ast

from factor_engine.tools.catalog_recipe_migration import RecipeMigration

_MAX_FORMULA_BYTES = 65_536
_DAILY_TABLE = "StockDailyBarAdj"
_OHLCV = ("open", "high", "low", "close", "volume")


def _offset(lines: list[str], lineno: int, byte_col: int) -> int:
    prefix = lines[lineno - 1].encode("utf-8")[:byte_col]
    return sum(len(line) for line in lines[:lineno - 1]) + len(prefix.decode("utf-8"))


def _span(lines: list[str], node: ast.AST) -> tuple[int, int]:
    if node.end_lineno is None or node.end_col_offset is None:
        raise ValueError("AST node has no complete source location")
    return (
        _offset(lines, node.lineno, node.col_offset),
        _offset(lines, node.end_lineno, node.end_col_offset),
    )


def _literal(node: ast.AST, value: object) -> bool:
    return (
        isinstance(node, ast.Constant)
        and type(node.value) is type(value)
        and node.value == value
    )


def _exact_daily_field(node: ast.AST, field_name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "field"
        and len(node.args) == 1
        and _literal(node.args[0], field_name)
        and len(node.keywords) == 1
        and node.keywords[0].arg == "table"
        and _literal(node.keywords[0].value, _DAILY_TABLE)
    )


def _source(formula: str, lines: list[str], node: ast.AST) -> str:
    start, end = _span(lines, node)
    return formula[start:end]


def migrate_catalog_r13_expression_formula(
    formula: str, *, logic: str = "", enabled: bool = False
) -> RecipeMigration:
    """Rewrite only the reviewed five-field OBV bundle; otherwise fail closed."""
    if not isinstance(formula, str):
        raise TypeError("formula must be a string")
    if not isinstance(logic, str):
        raise TypeError("logic must be a string")
    if len(formula.encode("utf-8")) > _MAX_FORMULA_BYTES:
        raise ValueError("formula exceeds migration input budget")
    if not enabled:
        return RecipeMigration(formula, ())

    tree = ast.parse(formula, mode="eval")
    if "obv" not in logic.casefold():
        return RecipeMigration(formula, ())

    lines = formula.splitlines(keepends=True) or [""]
    edits: list[tuple[int, int, str]] = []
    changes: list[str] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "OBV"
            and not node.keywords
            and len(node.args) == 5
            and all(
                _exact_daily_field(argument, field_name)
                for argument, field_name in zip(node.args, _OHLCV)
            )
        ):
            continue
        close = _source(formula, lines, node.args[3])
        volume = _source(formula, lines, node.args[4])
        start, end = _span(lines, node)
        edits.append((start, end, f"OBV({close}, {volume})"))
        changes.append(
            "EXACT_SHAPE_RECIPE OBV: reviewed five-field StockDailyBarAdj OHLCV "
            "bundle mapped to canonical price=close and volume=volume; open/high/low "
            "are discarded; compile-only migration does not certify execution"
        )

    ordered = sorted(edits)
    if any(
        next_start < end
        for (_, end, _), (next_start, _, _) in zip(ordered, ordered[1:])
    ):
        return RecipeMigration(formula, ())
    migrated = formula
    for start, end, replacement in reversed(ordered):
        migrated = migrated[:start] + replacement + migrated[end:]
    return RecipeMigration(migrated, tuple(changes))
