# -*- coding: utf-8 -*-
"""AST 安全化简（§7 规则）：括号壳、unary+、x*1/1*x/x/1/x+0/0+x/x-0/0-x→-x、
x*(-1)→-x、-(-x)→x、pow(x,1)→x、常量折叠。

**禁止危险化简**（§8）：x/x 不得变 1，log(exp(x)) 不得变 x 等——宁可少去重不可错合并。
"""

from __future__ import annotations

import math
from typing import Tuple

from factor_engine.expr.cleaned_call import CleanedCall
from factor_engine.expr.field import FieldRef
from factor_engine.expr.literal import Literal
from factor_engine.expr.column import ColumnRef
from factor_engine.expr.base import Expr

from .operator_meta import get_operator_meta


def canonicalize(node: Expr) -> Expr:
    """递归 AST 安全化简。

    规则：
    1. 括号壳：AST 天然解决，序列化时幂等保证。
    2. unary + 去除。
    3. 简化：x*1/1*x/x/1/x+0/0+x/x-0 → x；0-x → -x；x*(-1)/(-1)*x → -x；-(-x) → x。
    4. pow(x,1) → x。
    5. 常量折叠：纯常量表达式（叶子全为 Literal）。
    6. 交换律排序：commutative 算子按子节点 hash 排序。
    7. 禁止化简 x/x → 1 等危险变换。
    """
    return _canonicalize_recursive(node)


def _canonicalize_recursive(node: Expr) -> Expr:
    if isinstance(node, Literal):
        return node
    if isinstance(node, (FieldRef, ColumnRef)):
        return node
    if isinstance(node, CleanedCall):
        return _canonicalize_call(node)
    # 未知类型原样返回
    return node


def _canonicalize_call(node: CleanedCall) -> Expr:
    op = node.op
    args = tuple(_canonicalize_recursive(a) for a in node.args)
    kwargs = node.kwargs

    # 应用代数化简规则
    result: Expr = CleanedCall(op=op, args=args, kwargs=kwargs)

    # --- §7 规则 ---
    result = _apply_idempotency(result)
    result = _apply_constant_folding(result)

    # 最后刷新参数（常量折叠等可能改变 args）
    if isinstance(result, CleanedCall):
        result = _apply_commutative_sort(result)

    return result


def _is_numeric_literal(node: Expr, value: float | None = None) -> bool:
    """检查节点是否为数值 Literal，可选匹配特定值。"""
    if not isinstance(node, Literal):
        return False
    v = node.value
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    if value is not None and v != value:
        return False
    return True


def _is_zero(node: Expr) -> bool:
    return _is_numeric_literal(node, 0.0) or _is_numeric_literal(node, 0)


def _is_one(node: Expr) -> bool:
    return _is_numeric_literal(node, 1.0) or _is_numeric_literal(node, 1)


def _is_neg_one(node: Expr) -> bool:
    return _is_numeric_literal(node, -1.0) or _is_numeric_literal(node, -1)


def _apply_idempotency(node: Expr) -> Expr:
    """应用代数恒等化简（§7）。"""
    if not isinstance(node, CleanedCall):
        return node

    op = node.op
    args = node.args

    # --- unary + 去除 ---
    if op == "add" and len(args) == 1:
        return args[0]

    # 特判：neg 是单目算子
    # -(-x) → x
    if op == "neg" and len(args) == 1:
        inner = args[0]
        if isinstance(inner, CleanedCall) and inner.op == "neg" and len(inner.args) == 1:
            return _canonicalize_recursive(inner.args[0])

    # --- 二元恒等 ---
    if len(args) != 2:
        return node

    left, right = args

    # x+0 / 0+x → x
    if op == "add":
        if _is_zero(right):
            return left
        if _is_zero(left):
            return right

    # x-0 → x
    if op == "subtract":
        if _is_zero(right):
            return left
        # 0-x → -x
        if _is_zero(left):
            return CleanedCall(op="neg", args=(right,))

    # x*1 / 1*x → x
    if op == "multiply":
        if _is_one(right):
            return left
        if _is_one(left):
            return right
        # x*(-1) → -x
        if _is_neg_one(right):
            return CleanedCall(op="neg", args=(left,))
        if _is_neg_one(left):
            return CleanedCall(op="neg", args=(right,))

    # x/1 → x
    if op == "divide":
        if _is_one(right):
            return left

    # pow(x,1) → x
    if op == "power":
        if _is_one(right):
            return left

    return node


def _is_pure_constant(node: Expr) -> bool:
    """检查子树是否全部由 Literal 组成（无数据依赖）。"""
    if isinstance(node, Literal):
        return True
    if isinstance(node, (FieldRef, ColumnRef)):
        return False
    if isinstance(node, CleanedCall):
        return all(_is_pure_constant(a) for a in node.args)
    return False


def _eval_constant(node: Expr) -> float:
    """在纯常量树上执行数值计算（常量折叠）。"""
    if isinstance(node, Literal):
        v = node.value
        if isinstance(v, bool):
            return float(v)
        return float(v)
    if isinstance(node, CleanedCall):
        op = node.op
        vals = [_eval_constant(a) for a in node.args]
        if op == "add":
            return vals[0] + vals[1] if len(vals) == 2 else vals[0]
        if op == "subtract":
            return vals[0] - vals[1]
        if op == "multiply":
            return vals[0] * vals[1]
        if op == "divide":
            # 禁止危险化简：常量折叠时如果除零，保留原表达式
            if vals[1] == 0.0:
                return float("nan")
            return vals[0] / vals[1]
        if op == "power":
            try:
                return vals[0] ** vals[1]
            except (OverflowError, ValueError):
                return float("nan")
        if op == "neg":
            return -vals[0]
        # 其他算子：不折叠
        raise ValueError(f"折叠不支持算子: {op}")
    raise TypeError(f"不可折叠类型: {type(node).__name__}")


def _apply_constant_folding(node: Expr) -> Expr:
    """常量折叠：仅对纯常量表达式求值。"""
    if not isinstance(node, CleanedCall):
        return node

    # 只对纯常量运算折叠
    if _is_pure_constant(node):
        try:
            result = _eval_constant(node)
            # 非有限值（NaN/Inf）不折叠（保留原表达式）
            if math.isfinite(result):
                return Literal(value=result)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            pass

    # 递归折叠子节点
    args = tuple(_apply_constant_folding(a) for a in node.args)
    if args != node.args:
        return CleanedCall(op=node.op, args=args, kwargs=node.kwargs)
    return node


def _apply_commutative_sort(node: CleanedCall) -> CleanedCall:
    """交换律排序：对 commutative 算子按子节点 hash 排序。"""
    meta = get_operator_meta(node.op)
    if not meta or not meta.commutative:
        return node

    # 对 args 排序：按 hash 值稳定排序
    sorted_args = tuple(sorted(node.args, key=_expr_sort_key))
    if sorted_args == node.args:
        return node
    return CleanedCall(op=node.op, args=sorted_args, kwargs=node.kwargs)


def _expr_sort_key(node: Expr) -> str:
    """为排序提供稳定、可比较的 key。"""
    if isinstance(node, Literal):
        return f"L:{repr(node.value)}"
    if isinstance(node, FieldRef):
        return f"F:{node.field_id}"
    if isinstance(node, ColumnRef):
        return f"C:{node.name}"
    if isinstance(node, CleanedCall):
        children_keys = ",".join(_expr_sort_key(a) for a in node.args)
        return f"K:{node.op}({children_keys})"
    return f"X:{repr(node)}"