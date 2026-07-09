"""检测 GTJA DSL 中是否仍含已废弃的算子别名。"""
from __future__ import annotations

import ast

# factor_engine _dedupe / _aliases 中已合并的旧 DSL 名（投递应写 canonical）
LEGACY_OPERATOR_NAMES: frozenset[str] = frozenset({
    "delay",
    "shift",
    "EMA",
    "ema",
    "SMA",
    "if_else",
    "m_argmax",
    "m_argmin",
    "max",  # 逐元素应写 flex_max
    "min",  # 逐元素应写 flex_min
})


def _is_rolling_max_min(call: ast.Call) -> bool:
    if len(call.args) != 2:
        return False
    a1 = call.args[1]
    if not isinstance(a1, ast.Constant) or not isinstance(a1.value, int) or a1.value < 2:
        return False
    a0 = call.args[0]
    if isinstance(a0, ast.Constant) and a0.value == 0:
        return False
    if isinstance(a0, ast.Constant) and isinstance(a0.value, (int, float)):
        return False
    return True


def find_legacy_operators(formula: str) -> list[str]:
    """返回公式中仍使用的旧算子名（去重保序）。"""
    tree = ast.parse(formula, mode="eval")
    found: list[str] = []
    seen: set[str] = set()

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name in LEGACY_OPERATOR_NAMES:
                    if name in ("max", "min") and _is_rolling_max_min(node):
                        pass
                    elif name not in seen:
                        seen.add(name)
                        found.append(name)
            self.generic_visit(node)

    V().visit(tree)
    return found
