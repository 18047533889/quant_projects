"""Exact-shape TRIX and ADXR lowering for catalog R18."""
from __future__ import annotations

import ast

from factor_engine.tools.catalog_recipe_migration import RecipeMigration


_TABLE = "StockDailyBarAdj"
_MAX_FORMULA_BYTES = 65_536


def _field(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "field"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == name
        and len(node.keywords) == 1
        and node.keywords[0].arg == "table"
        and isinstance(node.keywords[0].value, ast.Constant)
        and node.keywords[0].value.value == _TABLE
    )


def migrate_catalog_r18_technical_formula(
    formula: str, *, enabled: bool = False
) -> RecipeMigration:
    if not isinstance(formula, str):
        raise TypeError("formula must be a string")
    if len(formula.encode("utf-8")) > _MAX_FORMULA_BYTES:
        raise ValueError("formula exceeds migration input budget")
    if not enabled:
        return RecipeMigration(formula, ())

    tree = ast.parse(formula, mode="eval")
    changes: list[str] = []

    class Lower(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call):
            node = self.generic_visit(node)
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "TRIX"
                and not node.keywords
                and len(node.args) == 2
                and _field(node.args[0], "close")
                and isinstance(node.args[1], ast.Constant)
                and type(node.args[1].value) is int
                and node.args[1].value > 0
            ):
                close = ast.unparse(node.args[0])
                window = node.args[1].value
                ema3 = f"ts_ema(ts_ema(ts_ema({close}, {window}), {window}), {window})"
                previous = f"ts_delay({ema3}, 1)"
                replacement = (
                    f"where(eq({previous}, 0.0), safe_div_null(0.0, 0.0), "
                    f"multiply(100.0, divide(subtract({ema3}, {previous}), {previous})))"
                )
                changes.append(
                    f"EXACT_SHAPE_RECIPE TRIX: preserved close/window={window}; "
                    "triple adjust=False span EMA, exact-zero prior mapped to null, "
                    "then raw one-row percent change"
                )
                return ast.parse(replacement, mode="eval").body

            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "ADXR"
                and not node.keywords
                and len(node.args) == 4
                and all(
                    _field(arg, name)
                    for arg, name in zip(node.args[:3], ("high", "low", "close"))
                )
                and isinstance(node.args[3], ast.Constant)
                and type(node.args[3].value) is int
                and node.args[3].value > 0
            ):
                high, low, close = (ast.unparse(arg) for arg in node.args[:3])
                window = node.args[3].value
                adx = f"ADX({high}, {low}, {close}, {window})"
                replacement = f"multiply(0.5, add({adx}, ts_delay({adx}, {window})))"
                changes.append(
                    f"EXACT_SHAPE_RECIPE ADXR: preserved high/low/close/window={window}; "
                    "active canonical ADX plus its exact per-instrument window lag"
                )
                return ast.parse(replacement, mode="eval").body

            return node

    lowered = ast.fix_missing_locations(Lower().visit(tree))
    if not changes:
        return RecipeMigration(formula, ())
    return RecipeMigration(ast.unparse(lowered), tuple(changes))
