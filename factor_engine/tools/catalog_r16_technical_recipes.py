"""Opt-in exact CCI lowering for catalog R16."""
from __future__ import annotations

import ast
from factor_engine.tools.catalog_recipe_migration import RecipeMigration

_TABLE = "StockDailyBarAdj"
_MAX_FORMULA_BYTES = 65_536


def _field(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "field" and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant) and node.args[0].value == name
        and len(node.keywords) == 1 and node.keywords[0].arg == "table"
        and isinstance(node.keywords[0].value, ast.Constant)
        and node.keywords[0].value.value == _TABLE
    )


def _span(lines: list[str], node: ast.AST) -> tuple[int, int]:
    def off(line: int, col: int) -> int:
        prefix = lines[line - 1].encode()[:col].decode()
        return sum(map(len, lines[: line - 1])) + len(prefix)
    return off(node.lineno, node.col_offset), off(node.end_lineno, node.end_col_offset)


def _src(formula: str, lines: list[str], node: ast.AST) -> str:
    a, b = _span(lines, node)
    return formula[a:b]


def _expanded(high: str, low: str, close: str, window: int) -> str:
    tp = f"divide(add(add({high}, {low}), {close}), 3.0)"
    mad = f"ts_mean_abs_deviation_strict({tp}, {window})"
    numerator = f"subtract({tp}, ts_mean({tp}, {window}))"
    denominator = f"multiply(0.015, {mad})"
    return (
        f"where(eq({mad}, 0.0), safe_div_null(0.0, 0.0), "
        f"divide({numerator}, {denominator}))"
    )


def migrate_catalog_r16_technical_formula(
    formula: str, *, logic: str = "", enabled: bool = False
) -> RecipeMigration:
    if not isinstance(formula, str):
        raise TypeError("formula must be a string")
    if not isinstance(logic, str):
        raise TypeError("logic must be a string")
    if len(formula.encode()) > _MAX_FORMULA_BYTES:
        raise ValueError("formula exceeds migration input budget")
    if not enabled:
        return RecipeMigration(formula, ())
    tree = ast.parse(formula, mode="eval")
    lines = formula.splitlines(keepends=True) or [""]
    edits, changes = [], []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "CCI" and not node.keywords):
            continue
        args = node.args
        if (len(args) == 4 and all(_field(arg, name) for arg, name in zip(args[:3], ("high", "low", "close")))
                and isinstance(args[3], ast.Constant) and type(args[3].value) is int and args[3].value > 0):
            high, low, close = (_src(formula, lines, arg) for arg in args[:3])
            replacement = _expanded(high, low, close, args[3].value)
            kind = "EXACT_SHAPE_RECIPE"
        elif (len(args) == 5 and "cci" in logic.casefold()
              and all(_field(arg, name) for arg, name in zip(args, ("open", "high", "low", "close", "volume")))):
            high, low, close = (_src(formula, lines, arg) for arg in args[1:4])
            replacement = _expanded(high, low, close, 20)
            kind = "EXPLICIT_CONTRACT_REPAIR"
        else:
            continue
        start, end = _span(lines, node)
        edits.append((start, end, replacement))
        if kind == "EXPLICIT_CONTRACT_REPAIR":
            changes.append(
                "EXPLICIT_CONTRACT_REPAIR CCI: malformed StockDailyBarAdj OHLCV "
                "bundle restored to high/low/close; assumed canonical default window=20; "
                "open/volume discarded; strict window mean absolute deviation; "
                "exact-zero MAD maps to null"
            )
        else:
            changes.append(
                f"EXACT_SHAPE_RECIPE CCI: preserved explicit window={args[3].value}; "
                "strict window mean absolute deviation; exact-zero MAD maps to null"
            )
    migrated = formula
    for start, end, replacement in sorted(edits, reverse=True):
        migrated = migrated[:start] + replacement + migrated[end:]
    return RecipeMigration(migrated, tuple(changes))
