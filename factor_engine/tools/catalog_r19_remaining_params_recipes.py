"""Reviewed semantic repairs for the remaining R19 parameter failures."""
from __future__ import annotations

import ast

_MAX_BYTES = 65_536


def _literal(node: ast.AST, value: object) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) is type(value) and node.value == value


def migrate_formula(formula: str, logic: str = "") -> tuple[str, list[str]]:
    """Rewrite only exact, reviewed shapes; ambiguous calls remain unchanged."""
    if not isinstance(formula, str) or not isinstance(logic, str):
        raise TypeError("formula and logic must be strings")
    if len(formula.encode("utf-8")) > _MAX_BYTES:
        raise ValueError("formula exceeds migration input budget")
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return formula, []
    changes: list[str] = []

    class Repair(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call):
            node = self.generic_visit(node)
            if not isinstance(node.func, ast.Name):
                return node
            name = node.func.id
            kws = {item.arg: item for item in node.keywords if item.arg is not None}
            if len(kws) != len(node.keywords):
                return node

            if name == "ts_extreme_cluster_ratio" and not node.keywords and len(node.args) == 6:
                if all(_literal(arg, value) for arg, value in zip(node.args[1:], (60, 2.0, 0.0, 5, 5))):
                    node.args[3] = ast.Constant(0.9)
                    node.args[4] = ast.Constant("absolute")
                    changes.append(
                        "SEMANTIC_REDESIGN ts_extreme_cluster_ratio: legacy "
                        "(window=60, threshold=2.0, q=0.0, side=5, min_periods=5) "
                        "-> numeric absolute threshold=2.0, canonical inactive q=0.9, "
                        "side='absolute', min_periods=5"
                    )
                return node

            if name == "ts_first_passage_bias":
                if "min_periods" in kws and "min_anchors" not in kws:
                    kws["min_periods"].arg = "min_anchors"
                    changes.append(
                        "PARAMETER_REBIND ts_first_passage_bias: obsolete min_periods "
                        "-> min_anchors; value and historical-anchor support count preserved"
                    )
                if len(node.args) == 1 and "scale" not in kws:
                    x = node.args[0]
                    node.args.append(ast.Call(ast.Name("ts_std", ast.Load()), [x, ast.Constant(20)], []))
                    changes.append(
                        "SEMANTIC_REDESIGN ts_first_passage_bias: missing same-unit scale "
                        "-> trailing 20-row standard deviation of x"
                    )
                return node

            if name in {"ts_crossing_speed", "ts_crossing_acceleration"}:
                if len(node.args) == 1 and "y" not in kws:
                    x = node.args[0]
                    node.args.append(ast.Call(ast.Name("ts_mean", ast.Load()), [x, ast.Constant(20)], []))
                    changes.append(
                        f"SEMANTIC_REDESIGN {name}: missing same-unit crossing baseline "
                        "-> trailing 20-row mean of x"
                    )
                return node

            structure = {"ts_support_level", "ts_resistance_level", "ts_support_slope", "ts_resistance_slope", "ts_distance_to_support", "ts_distance_to_resistance", "ts_support_break", "ts_resistance_break"}
            if name in structure and not node.keywords and len(node.args) in {5, 6}:
                if all(_literal(arg, 3) for arg in node.args[-4:]):
                    node.args[-2] = ast.Constant(60)
                    changes.append(f"SEMANTIC_REDESIGN {name}: history_window=3 cannot contain three confirmed pivots with left_window=right_window=3; explicit history_window=60, points=3 retained; horizon changed non-equivalently")
                return node

            if name == "ts_current_drawdown_area" and len(node.args) == 1 and not node.keywords:
                x = node.args[0]
                is_ret = (
                    isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
                    and x.func.id == "field" and len(x.args) == 1
                    and _literal(x.args[0], "ret") and len(x.keywords) == 1
                    and x.keywords[0].arg == "table"
                    and _literal(x.keywords[0].value, "StockDailyBarAdj")
                )
                if is_ret:
                    if isinstance(x, ast.Call):
                        x.args[0] = ast.Constant("close")
                        node.args = [x, ast.Constant(60)]
                    else:
                        return node
                    changes.append(
                        "SEMANTIC_REDESIGN ts_current_drawdown_area: return input -> "
                        "corresponding adjusted close level; drawdown window=60"
                    )
                return node
            return node

    repaired = Repair().visit(tree)
    ast.fix_missing_locations(repaired)
    if not changes:
        return formula, []
    return ast.unparse(repaired), changes
