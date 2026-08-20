# -*- coding: utf-8
"""Scalar broadcasting 契约（第一阶段）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ScalarKind = Literal["float", "int"]


@dataclass(frozen=True)
class ScalarBroadcastSpec:
    allows_scalar: bool = True
    scalar_kind: ScalarKind = "float"
    scalar_null_allowed: bool = False
    scalar_nan_allowed: bool = False


# 已认证支持 scalar 的 production 算子子集
SCALAR_BROADCAST_OPS: dict[str, ScalarBroadcastSpec] = {
    "add": ScalarBroadcastSpec(),
    "subtract": ScalarBroadcastSpec(),
    "multiply": ScalarBroadcastSpec(),
    "maximum": ScalarBroadcastSpec(),
    "minimum": ScalarBroadcastSpec(),
    "gt": ScalarBroadcastSpec(),
    "lt": ScalarBroadcastSpec(),
    "eq": ScalarBroadcastSpec(),
    "ge": ScalarBroadcastSpec(),
    "le": ScalarBroadcastSpec(),
    "ne": ScalarBroadcastSpec(),
    "where": ScalarBroadcastSpec(),
    "clip": ScalarBroadcastSpec(),
}


def scalar_broadcast_spec_for(canon: str) -> ScalarBroadcastSpec | None:
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return SCALAR_BROADCAST_OPS.get(name)
