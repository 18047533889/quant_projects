# -*- coding: utf-8
"""Composite reference parity 辅助：直接调用 Pandas operator（不经 optimizer）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from cleaned_operators.registry import OperatorRegistry
from planner.logical_plan import PlanNode
from planner.optimizer import Optimizer


@dataclass(frozen=True)
class CompositeReferenceCase:
    """单个 composite reference parity 用例。"""

    canon: str
    columns: tuple[str, ...]
    window: int = 3
    extra_attrs: dict[str, Any] = field(default_factory=dict)
    calc_kwargs: dict[str, Any] = field(default_factory=dict)


# 全部 16 个已注册 composite lowering
COMPOSITE_REFERENCE_CASES: tuple[CompositeReferenceCase, ...] = (
    CompositeReferenceCase("MOM", ("close",), window=3),
    CompositeReferenceCase("ROC", ("close",), window=3),
    CompositeReferenceCase("BollingerBands", ("close",), window=3),
    CompositeReferenceCase("BollingerUpper", ("close",), window=3, calc_kwargs={"std_dev": 2.0}),
    CompositeReferenceCase("BollingerLower", ("close",), window=3, calc_kwargs={"std_dev": 2.0}),
    CompositeReferenceCase("DPO", ("close",), window=4),
    CompositeReferenceCase("WilliamsR", ("high", "low", "close"), window=3),
    CompositeReferenceCase("StochasticK", ("high", "low", "close"), window=3),
    CompositeReferenceCase("StochasticD", ("high", "low", "close"), window=3),
    CompositeReferenceCase("OBV", ("close", "volume")),
    CompositeReferenceCase("operating_margin", ("operating_income", "revenue")),
    CompositeReferenceCase("current_ratio", ("current_assets", "current_liabilities")),
    CompositeReferenceCase("quick_ratio", ("current_assets", "inventory", "current_liabilities")),
    CompositeReferenceCase("debt_to_equity", ("total_debt", "total_equity")),
    CompositeReferenceCase("real_turnover_rate", ("volume", "float_shares")),
    CompositeReferenceCase("micro_spread", ("high", "low", "close")),
    CompositeReferenceCase("ts_ratio", ("close",), window=1),
)


def series_to_panel(series: pd.Series) -> pd.DataFrame:
    """MultiIndex series → 宽表 panel（index=时间, columns=标的）。"""
    if isinstance(series.index, pd.MultiIndex):
        return series.unstack(level="instrument")
    return pd.DataFrame(series)


def panel_to_series(panel: pd.DataFrame, index: pd.MultiIndex) -> pd.Series:
    """宽表 panel → MultiIndex series。"""
    stacked = panel.stack(future_stack=True)
    if isinstance(stacked, pd.Series):
        return stacked.reindex(index)
    return pd.Series(stacked.values, index=index)


def reference_pandas_calculate(
    case: CompositeReferenceCase,
    panels: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """直接调用 ``OperatorRegistry`` Pandas 实现（原始高级算子语义）。"""
    op = OperatorRegistry.get(case.canon, backend="pandas_numpy")
    if op is None:
        raise RuntimeError(f"no pandas_numpy operator for {case.canon!r}")
    meta = getattr(op, "metadata", None)
    param_names = list(getattr(meta, "param_names", None) or [])
    work = dict(panels)
    if "x" not in work and "close" in work:
        work["x"] = work["close"]
    if "price" not in work and "close" in work:
        work["price"] = work["close"]
    args = [work[name] for name in param_names if name in work]
    kwargs = dict(case.calc_kwargs)
    if "window" in param_names and case.window is not None:
        kwargs.setdefault("window", case.window)
    return op.calculate(*args, **kwargs)


def build_composite_plan(case: CompositeReferenceCase) -> PlanNode:
    """构造 composite 算子 PlanNode。"""
    inputs = [PlanNode(op="column", attrs={"name": c}, inputs=[]) for c in case.columns]
    attrs: dict[str, object] = {"window": case.window, "d": case.window}
    attrs.update(case.extra_attrs)
    if "std_dev" in case.calc_kwargs:
        attrs["std_dev"] = case.calc_kwargs["std_dev"]
    return PlanNode(op=case.canon, inputs=inputs, attrs=attrs)


def lowered_plan_for(case: CompositeReferenceCase) -> PlanNode:
    """composite lowering + 常量折叠（不含 fastpath rewrite）。"""
    return Optimizer().lower_only(build_composite_plan(case))


def execute_lowered_pandas(plan: PlanNode, ctx) -> pd.Series:
    """在 Pandas backend 上执行已 lowered 的 plan，返回 MultiIndex Series。"""
    from backend.pandas_backend import PandasBackend

    return PandasBackend().execute(plan, ctx)


def build_reference_panels(source, index: pd.MultiIndex) -> dict[str, pd.DataFrame]:
    """从 InMemorySeriesSource 构建 reference 用 panel 字典。"""
    panels: dict[str, pd.DataFrame] = {}
    for name in source.data:
        panels[name] = series_to_panel(source.data[name])
    return panels
