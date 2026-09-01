# -*- coding: utf-8 -*-
"""参数族归一化（§18-§19）。

按 operator role 对数字参数做 placeholder 化：

- WINDOW / LAG 数字参数 → placeholder ``$WINDOW``（或 ``$LAG``）后 hash；
- THRESHOLD 可配置（默认 family 化，配置项给出）；
- POWER 保守不 family 化。

``ts_mean(close,19/20/21)`` 必须同 family、canonical_ast_hash 不同。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from factor_engine.expr.cleaned_call import CleanedCall
from factor_engine.expr.field import FieldRef
from factor_engine.expr.literal import Literal
from factor_engine.expr.column import ColumnRef
from factor_engine.expr.base import Expr

from . import hasher
from . import serializer
from .operator_meta import get_operator_meta

# 可 family 化的角色
_FAMILY_ROLES = frozenset({"WINDOW", "LAG", "THRESHOLD"})

# placeholder 名称映射
_ROLE_PLACEHOLDER = {
    "WINDOW": "$WINDOW",
    "LAG": "$LAG",
    "THRESHOLD": "$THRESHOLD",
}


@dataclass(frozen=True)
class ParameterFamilyConfig:
    """参数族化配置。

    Attributes
    ----------
    family_threshold : bool
        是否将 THRESHOLD 角色参数 family 化（默认 True）。
    family_window : bool
        是否将 WINDOW 角色参数 family 化（默认 True）。
    family_lag : bool
        是否将 LAG 角色参数 family 化（默认 True）。
    """

    family_threshold: bool = True
    family_window: bool = True
    family_lag: bool = True


def _is_numeric_literal(node: Expr) -> bool:
    if not isinstance(node, Literal):
        return False
    v = node.value
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _family_placeholder(role: str) -> str:
    return _ROLE_PLACEHOLDER.get(role, "$PARAM")


def _should_family(role: str, config: ParameterFamilyConfig) -> bool:
    """是否对指定角色做 family 化（POWER 保守不 family 化）。"""
    if role == "POWER":
        return False
    if role == "WINDOW" and config.family_window:
        return True
    if role == "LAG" and config.family_lag:
        return True
    if role == "THRESHOLD" and config.family_threshold:
        return True
    return False


def parameter_family_payload(
    node: Expr,
    *,
    config: ParameterFamilyConfig | None = None,
) -> dict[str, Any]:
    """返回 parameter-family 化的 AST payload。

    根据算子的声明 role，将数字字面量参数替换为 placeholder：
    - WINDOW / LAG / THRESHOLD（可配置）→ ``$WINDOW`` / ``$LAG`` / ``$THRESHOLD``
    - POWER 保守不 family 化（角色 POWER 始终保留原值）
    - 未知角色保留原值
    """
    config = config or ParameterFamilyConfig()

    if isinstance(node, (FieldRef, ColumnRef, Literal)):
        return serializer.canonical_ast_payload(node)
    if isinstance(node, CleanedCall):
        meta = get_operator_meta(node.op)
        role = meta.role if meta else "OTHER"
        family = _should_family(role, config)
        placeholder = _family_placeholder(role)
        args = []
        for arg in node.args:
            if family and _is_numeric_literal(arg):
                args.append({"kind": "literal", "value": placeholder})
            else:
                args.append(parameter_family_payload(arg, config=config))
        kwargs = {}
        for k, v in sorted(node.kwargs):
            if family and isinstance(v, (int, float)) and not isinstance(v, bool):
                kwargs[str(k)] = placeholder
            else:
                kwargs[str(k)] = serializer._value(v)
        return {
            "kind": "call",
            "op": node.op,
            "args": args,
            "kwargs": kwargs,
        }
    raise TypeError(f"unsupported expression node: {type(node).__name__}")


def parameter_family_text(node: Expr, *, config: ParameterFamilyConfig | None = None) -> str:
    """返回 parameter-family 化的 AST 文本（JSON，sort_keys）。"""
    payload = parameter_family_payload(node, config=config)
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def parameter_family_id(node: Expr, *, config: ParameterFamilyConfig | None = None) -> str:
    """计算 parameter_family_id。

    对 parameter-family 化的 AST 文本做 hash(namespace + text)。
    """
    text = parameter_family_text(node, config=config)
    return hasher.compute_hash(text)