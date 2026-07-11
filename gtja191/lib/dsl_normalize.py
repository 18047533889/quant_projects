"""GTJA DSL → factor_engine 最新 canonical 算子命名。"""
from __future__ import annotations

import ast
import re

# 函数名直替（旧 DSL 别名 → 当前 canonical）
# 注意：GTJA SMA(x,n,m) 由 convert 脚本译为 ts_ema(x, span)，勿在此把 SMA→ts_mean
# （factor_engine 的 SMA 别名语义是简单 ts_mean，与 GTJA 指数平滑不同）
_FN_RENAMES: dict[str, str] = {
    "delay": "ts_delay",
    "shift": "ts_delay",
    "EMA": "ts_ema",
    "ema": "ts_ema",
    "if_else": "where",
    "IIF": "where",
    "m_argmax": "ts_argmax",
    "m_argmin": "ts_argmin",
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


class _CanonicalRenamer(ast.NodeTransformer):
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
    """将 DSL 字符串中的算子名规范为 factor_engine 最新 canonical。"""
    text = str(formula or "").strip()
    if not text:
        return text
    tree = ast.parse(text, mode="eval")
    new_tree = _CanonicalRenamer().visit(tree)
    ast.fix_missing_locations(new_tree)
    out = ast.unparse(new_tree)
    return re.sub(r"\s+", " ", out).strip()
