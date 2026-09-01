# -*- coding: utf-8 -*-
"""符号归一化器（§16/§17）。

只识别明确全局负号：
- -f / 0-f / (-1)*f / f*(-1) → 同一 signal_equivalence_id，orientation 分别 -1/-1/-1/-1，
  f 为 +1。

不做激进代数证明（A-B 不判等 -(B-A)）。

signal_equivalence_id = hash(namespace + sign_normalized_ast_text)。
"""

from __future__ import annotations

from factor_engine.expr.cleaned_call import CleanedCall
from factor_engine.expr.field import FieldRef
from factor_engine.expr.literal import Literal
from factor_engine.expr.column import ColumnRef
from factor_engine.expr.base import Expr

from . import hasher
from . import serializer


def detect_orientation(node: Expr) -> int:
    """检测公式的全局符号方向。

    返回 +1（正）或 -1（负）。只识别明确全局负号模式。

    识别模式：
    - neg(f)            → -1
    - subtract(0, f)    → -1
    - multiply(-1, f)   → -1
    - multiply(f, -1)   → -1
    - 其他              → +1
    """
    if not isinstance(node, CleanedCall):
        return 1

    op = node.op
    args = node.args

    # neg(f) → -1
    if op == "neg" and len(args) == 1:
        return -1

    # subtract(0, f) → -1
    if op == "subtract" and len(args) == 2:
        if _is_zero(args[0]) and not _is_zero(args[1]):
            return -1

    # multiply(-1, f) or multiply(f, -1) → -1
    if op == "multiply" and len(args) == 2:
        if _is_neg_one(args[0]) or _is_neg_one(args[1]):
            return -1

    return 1


def _is_zero(node: Expr) -> bool:
    if isinstance(node, Literal):
        v = node.value
        if isinstance(v, bool):
            return False
        if isinstance(v, (int, float)):
            return v == 0
    return False


def _is_neg_one(node: Expr) -> bool:
    if isinstance(node, Literal):
        v = node.value
        if isinstance(v, bool):
            return False
        if isinstance(v, (int, float)):
            return v == -1
    return False


def sign_normalize(node: Expr) -> Expr:
    """去掉顶层全局负号，返回 sign-normalized 的 AST。

    只去掉最外层的 **明确全局负号**。
    - neg(f) → f
    - subtract(0, f) → f
    - multiply(-1, f) → f
    - multiply(f, -1) → f
    - 其余 → node 原样
    """
    if not isinstance(node, CleanedCall):
        return node

    op = node.op
    args = node.args

    # neg(f) → f
    if op == "neg" and len(args) == 1:
        return args[0]

    # subtract(0, f) → f
    if op == "subtract" and len(args) == 2:
        if _is_zero(args[0]) and not _is_zero(args[1]):
            return args[1]

    # multiply(-1, f) → f
    if op == "multiply" and len(args) == 2:
        if _is_neg_one(args[0]):
            return args[1]
        if _is_neg_one(args[1]):
            return args[0]

    return node


def compute_signal_equivalence_id(node: Expr) -> str:
    """计算 signal_equivalence_id。

    先检测 orientation，然后 sign-normalize，再对 normalization 后的 AST
    计算 hash(namespace + sign_normalized_ast_text)。
    """
    normalized = sign_normalize(node)
    text = serializer.sign_normalized_ast_text(normalized)
    return hasher.compute_hash(text)