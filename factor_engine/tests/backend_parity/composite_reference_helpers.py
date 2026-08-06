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
    CompositeReferenceCase("MACD_line", ("close",), window=3, extra_attrs={"fast": 2, "slow": 3}, calc_kwargs={"fast": 2, "slow": 3}),
    CompositeReferenceCase("MACD_signal", ("close",), window=3, extra_attrs={"fast": 2, "slow": 3, "signal": 2}, calc_kwargs={"fast": 2, "slow": 3, "signal": 2}),
    CompositeReferenceCase("MACD_hist", ("close",), window=3, extra_attrs={"fast": 2, "slow": 3, "signal": 2}, calc_kwargs={"fast": 2, "slow": 3, "signal": 2}),
    CompositeReferenceCase("ts_ratio", ("close",), window=1),
    # A 股 composite lowerings
    CompositeReferenceCase("earnings_yield", ("pe",)),
    CompositeReferenceCase("book_to_price", ("pb",)),
    CompositeReferenceCase("float_share_ratio", ("float_shares", "total_shares")),
    CompositeReferenceCase("free_float_share_ratio", ("free_float_shares", "total_shares")),
    CompositeReferenceCase("true_turnover_rate", ("volume", "free_float_shares")),
    CompositeReferenceCase("limit_up_state", ("close", "upper_limit")),
    CompositeReferenceCase("limit_down_state", ("close", "lower_limit")),
    CompositeReferenceCase("limit_up_close", ("close", "upper_limit")),
    CompositeReferenceCase("limit_down_close", ("close", "lower_limit")),
    CompositeReferenceCase("benchmark_excess_return", ("ret", "benchmark_ret")),
    CompositeReferenceCase("benchmark_relative_price", ("price", "benchmark_price")),
    CompositeReferenceCase("holder_concentration", ("top_holder_shares", "total_shares")),
    # Next-stage composite lowerings (2026-08).
    CompositeReferenceCase("a_share_cap_ratio", ("a_cap", "capitalization")),
    CompositeReferenceCase("capital_change_magnitude", ("total_capital",), window=1),
    CompositeReferenceCase("fin_acquisition_cash_intensity", ("net_cash_from_subcompany", "avg_assets", "period_id")),
    CompositeReferenceCase("fin_borrowing_intensity", ("cash_from_borrowing", "avg_assets", "period_id")),
    CompositeReferenceCase("fin_capex_intensity", ("capex_cash", "avg_assets", "period_id")),
    CompositeReferenceCase("fin_contract_asset_intensity", ("contract_assets", "total_assets", "period_id")),
    CompositeReferenceCase("fin_contract_asset_liability_gap", ("contract_assets", "contract_liability", "total_assets", "period_id")),
    CompositeReferenceCase("fin_contract_liability_intensity", ("contract_liability", "total_assets", "period_id")),
    CompositeReferenceCase("fin_debt_repayment_intensity", ("borrowing_repayment", "avg_assets", "period_id")),
    CompositeReferenceCase("fin_debt_service_coverage_proxy", ("ocf", "borrowing_repayment", "interest_cost", "period_id")),
    CompositeReferenceCase("fin_deferred_tax_gap", ("deferred_tax_assets", "deferred_tax_liability", "total_assets", "period_id")),
    CompositeReferenceCase("fin_discontinued_operation_ratio", ("discontinued_operation_profit", "net_profit", "period_id")),
    CompositeReferenceCase("fin_fair_value_income_dependence", ("fair_value_income", "total_profit", "period_id")),
    CompositeReferenceCase("fin_goodwill_intensity", ("goodwill", "total_assets", "period_id")),
    CompositeReferenceCase("fin_impairment_intensity", ("asset_impairment_loss", "credit_impairment_loss", "operating_revenue", "period_id")),
    CompositeReferenceCase("fin_interest_coverage_proxy", ("operating_profit", "interest_cost", "period_id")),
    CompositeReferenceCase("fin_investment_income_dependence", ("investment_income", "total_profit", "period_id")),
    CompositeReferenceCase("fin_lease_intensity", ("usufruct_assets", "lease_liability", "total_assets", "period_id")),
    CompositeReferenceCase("fin_minority_profit_share", ("minority_profit", "net_profit", "period_id")),
    CompositeReferenceCase("fin_net_borrowing_cashflow", ("cash_from_borrowing", "cash_from_bonds_issue", "borrowing_repayment", "avg_assets", "period_id")),
    CompositeReferenceCase("fin_oci_to_equity", ("other_comprehensive_income", "avg_equity", "period_id")),
    CompositeReferenceCase("fin_other_earnings_dependence", ("other_earnings", "total_profit", "period_id")),
    CompositeReferenceCase("fin_rd_capitalization_ratio", ("capitalized_dev_increase", "rd_expense", "period_id")),
    CompositeReferenceCase("fin_rd_total_intensity", ("rd_expense", "capitalized_dev_increase", "operating_revenue", "period_id")),
    CompositeReferenceCase("fin_roe_cash_gap", ("net_profit", "ocf", "avg_equity", "period_id")),
    CompositeReferenceCase("free_float_ratio", ("free_cap", "capitalization")),
    CompositeReferenceCase("free_float_turnover", ("volume", "free_cap")),
    CompositeReferenceCase("free_to_circulating_ratio", ("free_cap", "circulating_cap")),
    CompositeReferenceCase("holder_float_concentration_gap", ("top10_concentration", "top10_float_concentration")),
    CompositeReferenceCase("holder_freeze_ratio", ("freeze_shares", "total_capital")),
    CompositeReferenceCase("holder_locked_share_ratio", ("locked_shares", "total_capital")),
    CompositeReferenceCase("holder_pledge_ratio", ("pledge_shares", "total_capital")),
    CompositeReferenceCase("holder_shareholder_network_centrality", ("degree", "total")),
    CompositeReferenceCase("holder_shareholder_overlap_ratio", ("shared_holders", "total_holders")),
    CompositeReferenceCase("index_weight_gap_to_free_float", ("index_weight", "free_float_weight")),
    CompositeReferenceCase("market_cap_free_cap_gap", ("market_cap", "free_market_cap")),
    CompositeReferenceCase("valuation_pcf_definition_gap", ("pcf_ratio", "pcf_ratio2")),
    CompositeReferenceCase("valuation_pe_ttm_lyr_gap", ("pe_ratio", "pe_ratio_lyr")),
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
