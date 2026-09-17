"""Opt-in exact-shape technical recipes reviewed after catalog R13."""
from __future__ import annotations

import ast

from factor_engine.tools.catalog_recipe_migration import RecipeMigration

_MAX_FORMULA_BYTES = 65_536
_TABLE = "StockDailyBarAdj"
_OHLCV = ("open", "high", "low", "close", "volume")


def _offset(lines: list[str], lineno: int, byte_col: int) -> int:
    prefix = lines[lineno - 1].encode("utf-8")[:byte_col]
    return sum(len(line) for line in lines[:lineno - 1]) + len(prefix.decode("utf-8"))


def _span(lines: list[str], node: ast.AST) -> tuple[int, int]:
    if node.end_lineno is None or node.end_col_offset is None:
        raise ValueError("AST node has no complete source location")
    return _offset(lines, node.lineno, node.col_offset), _offset(
        lines, node.end_lineno, node.end_col_offset
    )


def _literal(node: ast.AST, value: object) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) is type(value) and node.value == value


def _qualified_field(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "field"
        and len(node.args) == 1
        and _literal(node.args[0], name)
        and len(node.keywords) == 1
        and node.keywords[0].arg == "table"
        and _literal(node.keywords[0].value, _TABLE)
    )


def _close(node: ast.AST) -> bool:
    return (isinstance(node, ast.Name) and node.id == "close") or _qualified_field(node, "close")


def _source(formula: str, lines: list[str], node: ast.AST) -> str:
    start, end = _span(lines, node)
    return formula[start:end]


def _window(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int and node.value > 0:
        return node.value
    return None


def migrate_catalog_r13_technical_formula(
    formula: str, *, logic: str = "", enabled: bool = False
) -> RecipeMigration:
    """Expand only reviewed DPO shapes into equivalent causal primitives."""
    if not isinstance(formula, str):
        raise TypeError("formula must be a string")
    if not isinstance(logic, str):
        raise TypeError("logic must be a string")
    if len(formula.encode("utf-8")) > _MAX_FORMULA_BYTES:
        raise ValueError("formula exceeds migration input budget")
    if not enabled:
        return RecipeMigration(formula, ())
    corroborated = any(
        needle in logic.casefold()
        for needle in ("dpo", "detrended_cycle_position", "去趋势价格振荡")
    )
    if not corroborated:
        return RecipeMigration(formula, ())

    tree = ast.parse(formula, mode="eval")
    lines = formula.splitlines(keepends=True) or [""]
    edits: list[tuple[int, int, str]] = []
    changes: list[str] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "DPO"
            and not node.keywords
        ):
            continue
        close_node: ast.AST | None = None
        window: int | None = None
        shape = ""
        if len(node.args) in (1, 2) and _close(node.args[0]):
            close_node = node.args[0]
            window = 20 if len(node.args) == 1 else _window(node.args[1])
            shape = "close"
        elif (
            len(node.args) == 5
            and all(_qualified_field(arg, name) for arg, name in zip(node.args, _OHLCV))
        ):
            close_node = node.args[3]
            window = 20
            shape = "reviewed StockDailyBarAdj OHLCV bundle"
        if close_node is None or window is None:
            continue
        close = _source(formula, lines, close_node)
        lag = window // 2 + 1
        replacement = (
            f"subtract({close}, ts_delay(ts_mean({close}, {window}, 1), {lag}))"
        )
        start, end = _span(lines, node)
        edits.append((start, end, replacement))
        changes.append(
            f"EXACT_SHAPE_RECIPE DPO: {shape}; window={window}; "
            f"rolling min_periods=1; causal lag=window//2+1={lag}; "
            "compile success does not certify execution"
        )

    ordered = sorted(edits)
    if any(beg < end for (_, end, _), (beg, _, _) in zip(ordered, ordered[1:])):
        return RecipeMigration(formula, ())
    migrated = formula
    for start, end, replacement in reversed(ordered):
        migrated = migrated[:start] + replacement + migrated[end:]
    return RecipeMigration(migrated, tuple(changes))
