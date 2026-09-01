# -*- coding: utf-8 -*-
"""稳定 AST 序列化（基于/兼容 factor_engine.expr.canonical）。

提供 ``canonical_ast_text`` 返回规范化的 AST 文本（JSON，sort_keys），
作为 hasher 的输入。基于 ``factor_engine.expr.canonical`` 的
``canonical_expression`` 并扩展了 ``sign_normalized_payload`` 用于
sign-normalized 的 AST 表示。
"""

from __future__ import annotations

import json
from typing import Any, Tuple

from factor_engine.expr.cleaned_call import CleanedCall
from factor_engine.expr.column import ColumnRef
from factor_engine.expr.field import FieldRef
from factor_engine.expr.literal import Literal
from factor_engine.expr.base import Expr


def _value(v: Any) -> Any:
    """递归将值标准化为可 JSON 序列化形式。"""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_value(item) for item in v]
    if isinstance(v, dict):
        return {str(k): _value(item) for k, item in sorted(v.items())}
    return repr(v)


def canonical_ast_payload(node: Expr) -> dict[str, Any]:
    """返回规范化的 AST payload（与 expr.canonical.expression_payload 一致）。"""
    if isinstance(node, FieldRef):
        return {
            "kind": "field",
            "field_id": node.field_id,
            "canonical_name": node.canonical_name,
            "table": node.table,
            "source_name": node.source_name,
        }
    if isinstance(node, ColumnRef):
        return {"kind": "column", "name": node.name}
    if isinstance(node, Literal):
        return {"kind": "literal", "value": _value(node.value)}
    if isinstance(node, CleanedCall):
        return {
            "kind": "call",
            "op": node.op,
            "args": [canonical_ast_payload(child) for child in node.args],
            "kwargs": {str(k): _value(v) for k, v in sorted(node.kwargs)},
        }
    raise TypeError(f"unsupported expression node: {type(node).__name__}")


def canonical_ast_text(node: Expr) -> str:
    """返回规范化的 AST 文本（JSON，sort_keys，紧凑格式）。

    与 ``factor_engine.expr.canonical.canonical_expression`` 输出一致。
    """
    return json.dumps(
        canonical_ast_payload(node),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def sign_normalized_payload(node: Expr) -> dict[str, Any]:
    """返回 sign-normalized 的 AST payload。

    sign-normalized = 去掉顶层全局负号后的 AST 的 payload。
    用于 sign_normalizer 的 signal_equivalence_id 计算。
    """
    # 递归去掉顶层 neg 包装
    inner = _strip_outer_neg(node)
    return canonical_ast_payload(inner)


def _strip_outer_neg(node: Expr) -> Expr:
    """递归去掉外层 neg 包装（sign_normalizer 用）。"""
    if isinstance(node, CleanedCall) and node.op == "neg" and len(node.args) == 1:
        return node.args[0]
    return node


def sign_normalized_ast_text(node: Expr) -> str:
    """返回 sign-normalized 的 AST 文本。"""
    return json.dumps(
        sign_normalized_payload(node),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )