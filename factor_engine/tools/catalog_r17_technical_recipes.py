"""Exact lowering for the reviewed R17 StochasticD and AROON outer calls."""
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


def _window(node: ast.AST) -> int | None:
    return node.value if isinstance(node, ast.Constant) and type(node.value) is int and node.value > 0 else None


def _stochastic_d(high: str, low: str, close: str, window: int) -> str:
    denominator = f"subtract(ts_max({high}, {window}), ts_min({low}, {window}))"
    k = (
        f"where(eq({denominator}, 0.0), safe_div_null(0.0, 0.0), "
        f"multiply(100.0, divide(subtract({close}, ts_min({low}, {window})), {denominator})))"
    )
    return f"ts_mean({k}, 3)"


def _aroon(close: str, window: int) -> str:
    full = window + 1
    return (
        f"multiply(100.0, divide(subtract(ts_argmin({close}, {full}, {full}), "
        f"ts_argmax({close}, {full}, {full})), {window}))"
    )


def migrate_catalog_r17_technical_formula(formula: str, *, enabled: bool = False) -> RecipeMigration:
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
            if not isinstance(node.func, ast.Name) or node.keywords:
                return node
            if (node.func.id == "StochasticD" and len(node.args) == 4
                    and all(_field(arg, name) for arg, name in zip(node.args[:3], ("high", "low", "close")))):
                window = _window(node.args[3])
                if window is not None:
                    changes.append(f"EXACT_SHAPE_RECIPE StochasticD: window={window}; exact StochasticK then fixed 3-row min1 mean")
                    return ast.parse(_stochastic_d(*(ast.unparse(x) for x in node.args[:3]), window), mode="eval").body
            if (node.func.id == "AROON" and len(node.args) == 2 and _field(node.args[0], "close")):
                window = _window(node.args[1])
                if window is not None:
                    changes.append(f"EXACT_SHAPE_RECIPE AROON: window={window}; full support={window + 1}; age tie=latest")
                    return ast.parse(_aroon(ast.unparse(node.args[0]), window), mode="eval").body
            return node

    lowered = ast.fix_missing_locations(Lower().visit(tree))
    if not changes:
        return RecipeMigration(formula, ())
    return RecipeMigration(ast.unparse(lowered), tuple(changes))
