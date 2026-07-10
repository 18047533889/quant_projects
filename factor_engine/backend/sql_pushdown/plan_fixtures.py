# -*- coding: utf-8
"""SQL capability / emitter 测试用最小 PlanNode 构造。

为 registry 中每个 canonical 生成可编译的最小逻辑计划，供能力探测与
emitter 回归测试复用。
"""
from __future__ import annotations

from planner.logical_plan import PlanNode


def column(name: str) -> PlanNode:
    """构造列引用节点 ``column(name)``。"""
    return PlanNode(op="column", attrs={"name": name})


def literal(value) -> PlanNode:
    """构造字面量节点 ``literal(value)``。"""
    return PlanNode(op="literal", attrs={"value": value})


def minimal_plan(op: str) -> PlanNode:
    """为 registry 中指定 canonical 构造可编译的最小逻辑计划。"""
    close, volume, industry = column("close"), column("volume"), column("industry")
    high, low = column("high"), column("low")
    if op in {
        "add",
        "subtract",
        "multiply",
        "divide",
        "gt",
        "lt",
        "eq",
        "ge",
        "le",
        "ne",
        "and_",
        "or_",
        "ts_corr",
        "ts_cov",
        "ts_beta",
        "ewm_corr",
        "ewm_cov",
        "protected_div",
        "safe_div_null",
        "ts_regression",
        "Slope",
        "power",
        "rolling_beta",
        "maximum",
        "minimum",
    }:
        return PlanNode(op=op, inputs=[close, volume], attrs={"d": 3, "window": 3, "span": 3})
    if op in {"cs_resid", "cs_regression"}:
        return PlanNode(op=op, inputs=[close, volume], attrs={"d": 3})
    if op.startswith("group_"):
        return PlanNode(op=op, inputs=[close, industry], attrs={"d": 3, "window": 3, "p": 0.5})
    if op in {"where", "if_else", "coalesce"}:
        return PlanNode(op=op, inputs=[close, volume, literal(0.0)], attrs={})
    if op == "clip":
        return PlanNode(op=op, inputs=[close], attrs={"min": 0.0, "max": 1.0})
    if op in {"winsorize"}:
        return PlanNode(op=op, inputs=[close], attrs={"lower": 0.01, "upper": 0.99, "a": 0.05})
    if op in {"rank_pct", "cs_pct_rank", "log_returns"}:
        return PlanNode(op=op, inputs=[close], attrs={})
    if op in {"cs_quantile", "c_percentile"}:
        return PlanNode(op=op, inputs=[close], attrs={"p": 0.5})
    if op == "volatility":
        return PlanNode(op=op, inputs=[close], attrs={"d": 20, "window": 20})
    if op == "vwap":
        return PlanNode(op=op, inputs=[close, volume], attrs={"d": 20, "window": 20})
    if op in {"fillna", "fillna_const", "nan_to_num", "bfill", "ffill"}:
        return PlanNode(op=op, inputs=[close], attrs={"value": 0, "limit": 1})
    if op == "ts_quantile":
        return PlanNode(op=op, inputs=[close], attrs={"d": 3, "q": 0.5})
    if op in {"ts_sharpe", "ts_autocorr"}:
        return PlanNode(op=op, inputs=[close], attrs={"d": 20, "window": 20})
    if op in {"ewm_mean", "ewm_std", "ewm_var", "WMA", "ts_decay_linear"}:
        return PlanNode(op=op, inputs=[close], attrs={"span": 5, "window": 5, "d": 5})
    if op == "ATR_WILDER":
        return PlanNode(op=op, inputs=[high, low, close], attrs={"d": 14, "window": 14})
    if op == "RSI_WILDER":
        return PlanNode(op=op, inputs=[close], attrs={"d": 14, "window": 14})
    if op == "not_":
        return PlanNode(op=op, inputs=[close], attrs={})
    return PlanNode(op=op, inputs=[close], attrs={"d": 3, "window": 3, "span": 3})
