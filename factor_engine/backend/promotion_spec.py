# -*- coding: utf-8
"""算子入口数值类型提升规则（不完全依赖 backend 自动 cast）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DtypeTarget = Literal["float64", "int64", "decimal128", "boolean"]


@dataclass(frozen=True)
class PromotionRule:
    left: str
    right: str
    result: DtypeTarget
    overflow_to: DtypeTarget | None = "float64"


PROMOTION_RULES: tuple[PromotionRule, ...] = (
    PromotionRule("int64", "float64", "float64"),
    PromotionRule("float64", "float64", "float64"),
    PromotionRule("int64", "int64", "float64", overflow_to="float64"),
    PromotionRule("decimal", "float64", "float64"),
    PromotionRule("boolean", "float64", "float64"),
)

STATISTICAL_OPS_TARGET: DtypeTarget = "float64"
COUNT_OPS_TARGET: DtypeTarget = "int64"

OVERFLOW_SENSITIVE_OPS: frozenset[str] = frozenset(
    {"multiply", "cum_sum", "cum_prod", "power", "cs_sum", "ts_sum", "vwap"}
)


def promote_binary(left_dtype: str, right_dtype: str) -> DtypeTarget:
    for rule in PROMOTION_RULES:
        if rule.left == left_dtype and rule.right == right_dtype:
            return rule.result
    return "float64"


def statistical_ops_use_float64() -> bool:
    return STATISTICAL_OPS_TARGET == "float64"
