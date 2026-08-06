# -*- coding: utf-8 -*-
"""Earnings quality, accruals and core / non-core income operators (P0).

All inputs are PIT-aligned daily panels produced by the data layer
(``PubDate`` as-of).  Period-change and multi-period-statistic kernels follow
the same ``period_id`` + visible-period semantics as ``transforms_v2``:
results advance only when a new report period becomes visible.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.fundamental.transforms_v2 import (
    _lag_value,
    _period_insert,
    _period_key,
    _safe_div,
    _values,
    _walk_periods,
)

_EPS = 1e-12
_CANONICALS: list[str] = []


def _meta(name: str, description: str, params: list[str]) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="fundamental_period",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "fundamental", "period_aware", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:fundamental",
            "unit:ratio", "cost:3",
        ],
    )


def _mk(name: str, description: str, params: list[str], fn: Callable[..., Any]):
    metadata = _meta(name, description, params)

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"FundamentalQuality_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="fundamental_period",
        business_category="fundamental",
        canonical=name,
        source="fundamental.quality_v2",
        backend="pandas_numpy",
        status="experimental",
    )(cls)
    _CANONICALS.append(name)
    # Register the canonical on the reviewed extended surface immediately so
    # modules that reuse ``_mk`` (e.g. ``accruals_scores``) stay surfaced.
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | {name}
    )
    return cls


def _walk_two(
    primary: pd.DataFrame,
    period_id: pd.DataFrame,
    secondary: pd.DataFrame,
    fn: Callable[[list[object], OrderedDict, OrderedDict, object], float],
) -> pd.DataFrame:
    period_id = period_id.reindex(index=primary.index, columns=primary.columns)
    secondary = secondary.reindex(index=primary.index, columns=primary.columns)
    out = pd.DataFrame(np.nan, index=primary.index, columns=primary.columns, dtype=float)
    for col in primary.columns:
        order: list[object] = []
        v1: OrderedDict[object, float] = OrderedDict()
        v2: OrderedDict[object, float] = OrderedDict()
        a = pd.to_numeric(primary[col], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(secondary[col], errors="coerce").to_numpy(dtype=float)
        pv = period_id[col].to_numpy()
        arr = np.full(len(a), np.nan, dtype=float)
        for i, (va, vb, raw_period) in enumerate(zip(a, b, pv)):
            key = _period_key(raw_period)
            if key is not None:
                if key not in order:
                    # Fiscal-ordinal insertion, not first-appearance order: a
                    # late-disclosed / back-filled older report period must not
                    # reorder the walked sequence (review §5.3).
                    _period_insert(order, key)
                if np.isfinite(va):
                    v1[key] = float(va)
                if np.isfinite(vb):
                    v2[key] = float(vb)
            if key is None:
                continue
            try:
                arr[i] = fn(order, v1, v2, key)
            except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
                arr[i] = np.nan
        out[col] = arr
    return out


def _ar1_slope(vals: np.ndarray) -> float:
    if len(vals) < 4 or np.std(vals[:-1]) <= _EPS:
        return np.nan
    return float(np.cov(vals[1:], vals[:-1])[0, 1] / np.var(vals[:-1]))


# ---------------------------------------------------------------------------
# § Accruals
# ---------------------------------------------------------------------------

def _fin_working_capital_accruals(ca, cash, cl, std, tp, avg_assets, period_id):
    d_ca = _walk_periods(ca, period_id, lambda o, v, c: float(v[c]) - _lag_value(o, v, c, 1))
    d_cash = _walk_periods(cash, period_id, lambda o, v, c: float(v[c]) - _lag_value(o, v, c, 1))
    d_cl = _walk_periods(cl, period_id, lambda o, v, c: float(v[c]) - _lag_value(o, v, c, 1))
    d_std = _walk_periods(std, period_id, lambda o, v, c: float(v[c]) - _lag_value(o, v, c, 1))
    d_tp = _walk_periods(tp, period_id, lambda o, v, c: float(v[c]) - _lag_value(o, v, c, 1))
    wc = (d_ca - d_cash) - (d_cl - d_std - d_tp)
    return wc / avg_assets.replace(0, np.nan)


_mk(
    "fin_working_capital_accruals",
    "Sloan 营运资本应计：[Δ(CA-Cash) - Δ(CL-STD-TP)] / AvgAssets。",
    ["current_assets", "cash", "current_liabilities", "short_term_debt", "tax_payable", "avg_assets", "period_id"],
    _fin_working_capital_accruals,
)


def _fin_total_operating_accruals(ta, cash, cl, std, tp, depreciation, avg_assets, period_id):
    d_ta = _walk_periods(ta, period_id, lambda o, v, c: float(v[c]) - _lag_value(o, v, c, 1))
    d_cash = _walk_periods(cash, period_id, lambda o, v, c: float(v[c]) - _lag_value(o, v, c, 1))
    d_cl = _walk_periods(cl, period_id, lambda o, v, c: float(v[c]) - _lag_value(o, v, c, 1))
    d_std = _walk_periods(std, period_id, lambda o, v, c: float(v[c]) - _lag_value(o, v, c, 1))
    d_tp = _walk_periods(tp, period_id, lambda o, v, c: float(v[c]) - _lag_value(o, v, c, 1))
    op_acc = (d_ta - d_cash) - (d_cl - d_std - d_tp) - depreciation
    return op_acc / avg_assets.replace(0, np.nan)


_mk(
    "fin_total_operating_accruals",
    "总经营应计：[Δ(TA-Cash)-Δ(CL-STD-TP)-Dep] / AvgAssets。",
    ["total_assets", "cash", "current_liabilities", "short_term_debt", "tax_payable", "depreciation", "avg_assets", "period_id"],
    _fin_total_operating_accruals,
)


def _fin_delta_noa(ta, cash, tl, std, ltd, avg_assets, period_id):
    noa = (
        _walk_periods(ta, period_id, lambda o, v, c: float(v[c]))
        - _walk_periods(cash, period_id, lambda o, v, c: float(v[c]))
        - (
            _walk_periods(tl, period_id, lambda o, v, c: float(v[c]))
            - _walk_periods(std, period_id, lambda o, v, c: float(v[c]))
            - _walk_periods(ltd, period_id, lambda o, v, c: float(v[c]))
        )
    )
    noa_prev = _walk_periods(noa, period_id, lambda o, v, c: _lag_value(o, v, c, 1))
    return (noa - noa_prev) / avg_assets.replace(0, np.nan)


_mk(
    "fin_delta_noa",
    "净经营资产变动 ΔNOA/AvgAssets，NOA=(TA-Cash)-(TL-STD-LTD)。",
    ["total_assets", "cash", "total_liabilities", "short_term_debt", "long_term_debt", "avg_assets", "period_id"],
    _fin_delta_noa,
)


# ---------------------------------------------------------------------------
# § Earnings vs cash
# ---------------------------------------------------------------------------

def _fin_roe_cash_gap(net_profit, ocf, avg_equity, period_id):
    return (net_profit - ocf) / avg_equity.replace(0, np.nan)


_mk(
    "fin_roe_cash_gap",
    "会计ROE与现金ROE之差：(NetProfit-OCF)/AvgEquity。",
    ["net_profit", "ocf", "avg_equity", "period_id"],
    _fin_roe_cash_gap,
)


def _fin_earnings_cash_gap_volatility(net_profit, ocf, avg_assets, period_id, periods=8):
    n = max(2, int(periods))
    gap = net_profit - ocf

    def _calc(o, v, c):
        vals = np.asarray(_values(o, v, c, n), dtype=float)
        return float(np.std(vals)) if len(vals) >= 3 else np.nan

    std_gap = _walk_periods(gap, period_id, _calc)
    return std_gap / avg_assets.abs().replace(0, np.nan)


_mk(
    "fin_earnings_cash_gap_volatility",
    "过去报告期 std(NetProfit-OCF)/AvgAssets。",
    ["net_profit", "ocf", "avg_assets", "period_id", "periods"],
    _fin_earnings_cash_gap_volatility,
)


def _fin_earnings_smoothness(net_profit, ocf, period_id, periods=8):
    n = max(2, int(periods))

    def _calc(o, v1, v2, c):
        keys = o[-n:]
        e = [float(v1[k]) for k in keys if k in v1 and np.isfinite(v1[k])]
        f = [float(v2[k]) for k in keys if k in v2 and np.isfinite(v2[k])]
        if len(e) < 3 or len(f) < 3 or np.std(f) <= _EPS:
            return np.nan
        return float(np.std(e) / np.std(f))

    return _walk_two(net_profit, period_id, ocf, _calc)


_mk(
    "fin_earnings_smoothness",
    "利润平滑度 std(NetProfit)/std(OCF)，越低越平滑。",
    ["net_profit", "ocf", "period_id", "periods"],
    _fin_earnings_smoothness,
)


def _fin_persistence(x, period_id, periods=8):
    n = max(3, int(periods))

    def _calc(o, v, c):
        vals = np.asarray(_values(o, v, c, n), dtype=float)
        return _ar1_slope(vals) if len(vals) >= 4 else np.nan

    return _walk_periods(x, period_id, _calc)


_mk(
    "fin_earnings_persistence",
    "报告期利润 AR(1) 系数。",
    ["net_profit", "period_id", "periods"],
    lambda x, period_id, periods=8: _fin_persistence(x, period_id, periods),
)
_mk(
    "fin_cashflow_persistence",
    "报告期经营现金流 AR(1) 系数。",
    ["ocf", "period_id", "periods"],
    lambda x, period_id, periods=8: _fin_persistence(x, period_id, periods),
)
_mk(
    "fin_margin_persistence",
    "利润率 AR(1) 系数。",
    ["margin", "period_id", "periods"],
    lambda x, period_id, periods=8: _fin_persistence(x, period_id, periods),
)


# ---------------------------------------------------------------------------
# § Core vs non-core income
# ---------------------------------------------------------------------------

def _fin_core_earnings_ratio(op, inv_income, fv_income, asset_deal, other_earnings, revenue, period_id, scale="revenue"):
    core = op - inv_income - fv_income - asset_deal - other_earnings
    return core / revenue.replace(0, np.nan)


_mk(
    "fin_core_earnings_ratio",
    "核心利润占比：(OperatingProfit-投资收益-公允价值变动-资产处置-其他收益)/营业收入。",
    ["operating_profit", "investment_income", "fair_value_income", "asset_deal_income", "other_earnings", "revenue", "period_id", "scale"],
    _fin_core_earnings_ratio,
)


def _fin_noncore_income_ratio(inv_income, fv_income, asset_deal, other_earnings, nonop_rev, nonop_exp, total_profit, period_id):
    noncore = inv_income + fv_income + asset_deal + other_earnings + nonop_rev - nonop_exp
    return noncore / total_profit.abs().replace(0, np.nan)


_mk(
    "fin_noncore_income_ratio",
    "非核心收益占比：(投资收益+公允价值变动+资产处置+其他收益+营业外收入-营业外支出)/|利润总额|。",
    ["investment_income", "fair_value_income", "asset_deal_income", "other_earnings", "non_operating_revenue", "non_operating_expense", "total_profit", "period_id"],
    _fin_noncore_income_ratio,
)


def _mk_dependence(name: str, description: str, num_idx: int, params: list[str]):
    def _calc(*args):
        num, total_profit = args[num_idx], args[-2]
        return num / total_profit.abs().replace(0, np.nan)

    _mk(name, description, params, _calc)


_mk_dependence(
    "fin_fair_value_income_dependence",
    "公允价值变动收益依赖度：FairValueVariableIncome/|利润总额|。",
    0, ["fair_value_income", "total_profit", "period_id"],
)
_mk_dependence(
    "fin_investment_income_dependence",
    "投资收益依赖度：InvestmentIncome/|利润总额|。",
    0, ["investment_income", "total_profit", "period_id"],
)
_mk_dependence(
    "fin_other_earnings_dependence",
    "其他收益依赖度：OtherEarnings/|利润总额|。",
    0, ["other_earnings", "total_profit", "period_id"],
)


def _fin_comprehensive_income_gap(total_composite_income, net_profit, avg_equity, period_id):
    return (total_composite_income - net_profit) / avg_equity.replace(0, np.nan)


_mk(
    "fin_comprehensive_income_gap",
    "综合收益与净利润之差 / 平均权益。",
    ["total_composite_income", "net_profit", "avg_equity", "period_id"],
    _fin_comprehensive_income_gap,
)


def _fin_oci_to_equity(oci, avg_equity, period_id):
    return oci / avg_equity.replace(0, np.nan)


_mk(
    "fin_oci_to_equity",
    "其他综合收益 / 平均权益。",
    ["other_comprehensive_income", "avg_equity", "period_id"],
    _fin_oci_to_equity,
)


def _fin_discontinued_operation_ratio(discon_profit, net_profit, period_id):
    return discon_profit / net_profit.abs().replace(0, np.nan)


_mk(
    "fin_discontinued_operation_ratio",
    "终止经营损益 / |净利润|。",
    ["discontinued_operation_profit", "net_profit", "period_id"],
    _fin_discontinued_operation_ratio,
)


def _fin_minority_profit_share(minority_profit, net_profit, period_id):
    return minority_profit / net_profit.abs().replace(0, np.nan)


_mk(
    "fin_minority_profit_share",
    "少数股东损益 / |净利润|。",
    ["minority_profit", "net_profit", "period_id"],
    _fin_minority_profit_share,
)

# (Surface registration is performed inline in ``_mk`` so reused registrars
# surface correctly; no trailing block needed.)
