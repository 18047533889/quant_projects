"""GTJA DSL → FactorEngine canonical 算子命名、字段语义与安全常量折叠。"""
from __future__ import annotations

import ast
import math
import re

_FN_RENAMES: dict[str, str] = {
    "delay": "ts_delay",
    "shift": "ts_delay",
    "EMA": "ts_ema",
    "ema": "ts_ema",
    "if_else": "where",
    "IIF": "where",
    "m_argmax": "ts_argmax",
    "m_argmin": "ts_argmin",
    "Slope": "ts_time_slope",
    "slope": "ts_time_slope",
}


def _is_int_literal(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def _literal_number(node: ast.AST) -> float | int | None:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _is_rolling_max_min(call: ast.Call) -> bool:
    if len(call.args) != 2:
        return False
    n = _is_int_literal(call.args[1])
    if n is None or n < 2:
        return False
    a0 = call.args[0]
    if isinstance(a0, ast.Constant) and isinstance(a0.value, (int, float)):
        return False
    return True


def _flatten_add_names(node: ast.AST) -> list[str] | None:
    names: list[str] = []

    def walk(cur: ast.AST) -> bool:
        if isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Add):
            return walk(cur.left) and walk(cur.right)
        if isinstance(cur, ast.Name):
            names.append(cur.id)
            return True
        return False

    return names if walk(node) else None


def _is_legacy_vwap_proxy(node: ast.AST) -> bool:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
        return False
    if _literal_number(node.right) not in (3, 3.0):
        return False
    names = _flatten_add_names(node.left)
    return names is not None and sorted(names) == ["close", "high", "low"]


class _CanonicalRenamer(ast.NodeTransformer):
    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if _is_legacy_vwap_proxy(node):
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="col", ctx=ast.Load()),
                    args=[ast.Constant(value="vwap")],
                    keywords=[],
                ),
                node,
            )
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name):
            return node
        name = node.func.id
        if name in ("max", "min") and not _is_rolling_max_min(node):
            node.func.id = "flex_max" if name == "max" else "flex_min"
            return node
        if name in _FN_RENAMES:
            node.func.id = _FN_RENAMES[name]
            name = node.func.id
        # 纯标量 power 不应进入 cleaned Series operator；在 DSL 层确定性折叠。
        if name == "power" and len(node.args) == 2 and not node.keywords:
            base = _literal_number(node.args[0])
            exponent = _literal_number(node.args[1])
            if base is not None and exponent is not None:
                try:
                    value = float(base) ** float(exponent)
                except (OverflowError, ValueError, ZeroDivisionError):
                    return node
                if math.isfinite(value):
                    return ast.copy_location(ast.Constant(value=value), node)
        return node


def normalize_operator_names(formula: str) -> str:
    text = str(formula or "").strip()
    if not text:
        return text
    tree = ast.parse(text, mode="eval")
    new_tree = _CanonicalRenamer().visit(tree)
    ast.fix_missing_locations(new_tree)
    return re.sub(r"\s+", " ", ast.unparse(new_tree)).strip()
