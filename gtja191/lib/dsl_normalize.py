"""GTJA DSL → factor_engine 最新 canonical 算子命名与字段语义。"""
from __future__ import annotations

import ast
import re

# 函数名直替（旧 DSL 别名 → 当前 canonical）
# 注意：GTJA SMA(x,n,m) 由 convert 脚本译为 ts_ema(x, span)，勿在此把 SMA→ts_mean。
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


def _is_rolling_max_min(call: ast.Call) -> bool:
    if len(call.args) != 2:
        return False
    n = _is_int_literal(call.args[1])
    if n is None or n < 2:
        return False
    a0 = call.args[0]
    if isinstance(a0, ast.Constant) and a0.value == 0:
        return False
    if isinstance(a0, ast.Constant) and isinstance(a0.value, (int, float)):
        return False
    return True


def _flatten_add_names(node: ast.AST) -> list[str] | None:
    """识别 ``high + low + close``，忽略括号但不忽略系数。"""
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
    """识别旧转换器生成的 ``(high + low + close) / 3``。"""
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
        return False
    if not isinstance(node.right, ast.Constant) or node.right.value not in (3, 3.0):
        return False
    names = _flatten_add_names(node.left)
    return names is not None and sorted(names) == ["close", "high", "low"]


class _CanonicalRenamer(ast.NodeTransformer):
    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if _is_legacy_vwap_proxy(node):
            # vwap 同时是算子名，必须用显式 col() 引用数据列。
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
        return node


def normalize_operator_names(formula: str) -> str:
    """将 DSL 字符串规范为当前 FactorEngine canonical 语义。"""
    text = str(formula or "").strip()
    if not text:
        return text
    tree = ast.parse(text, mode="eval")
    new_tree = _CanonicalRenamer().visit(tree)
    ast.fix_missing_locations(new_tree)
    out = ast.unparse(new_tree)
    return re.sub(r"\s+", " ", out).strip()
