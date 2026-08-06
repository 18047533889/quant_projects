# -*- coding: utf-8 -*-
"""Polars backends for next-stage fundamental operators (genuine expressions).

Elementwise financial ratios / dependencies / intensities expressed directly
with ``pl.Expr``.  The ``period_id`` argument is accepted for signature
compatibility with the pandas reference and is not needed by these kernels.
"""
from __future__ import annotations

from typing import Any

import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.common._polars_bridge import align_cols

_EPS = 1e-12


def _safe_div_expr(num, den):
    return num / den.fill_null(0.0).replace(0, None)


def _meta(name: str, description: str, params: list[str]) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="fundamental_period",
        description=description,
        param_names=params,
        return_type="series",
        tags=["fundamental", "polars", "native", "typed_v2"],
    )


def _register(name: str, description: str, params: list[str], fn):
    @register_operator(
        name=name,
        category="fundamental_period",
        business_category="fundamental",
        canonical=name,
        source="fundamental.polars_quality_v2",
    )
    class _FundamentalPolars(SeriesOperator):
        metadata = _meta(name, description, params)

        def _calculate_series(self, *args, **kwargs):
            return fn(*args, **kwargs)

    return _FundamentalPolars


def _binary(a, b, expr_fn):
    cols = align_cols(a, b)
    return a.with_columns([expr_fn(a[c], b[c]).alias(c) for c in cols])


_register(
    "fin_roe_cash_gap", "会计ROE与现金ROE之差（Polars）。", ["net_profit", "ocf", "avg_equity", "period_id"],
    lambda np_, ocf, ae, period_id=None: _roe_cash_gap(np_, ocf, ae),
)
_register(
    "fin_fair_value_income_dependence", "公允价值变动依赖度（Polars）。", ["fair_value_income", "total_profit", "period_id"],
    lambda fv, tp, period_id=None: _binary(fv, tp, lambda a, b: _safe_div_expr(a, b.abs())),
)
_register(
    "fin_investment_income_dependence", "投资收益依赖度（Polars）。", ["investment_income", "total_profit", "period_id"],
    lambda inv, tp, period_id=None: _binary(inv, tp, lambda a, b: _safe_div_expr(a, b.abs())),
)
_register(
    "fin_other_earnings_dependence", "其他收益依赖度（Polars）。", ["other_earnings", "total_profit", "period_id"],
    lambda oe, tp, period_id=None: _binary(oe, tp, lambda a, b: _safe_div_expr(a, b.abs())),
)
_register(
    "fin_comprehensive_income_gap", "综合收益与净利润之差/权益（Polars）。", ["total_composite_income", "net_profit", "avg_equity", "period_id"],
    lambda tci, np_, ae, period_id=None: _comp_gap(tci, np_, ae),
)
_register(
    "fin_oci_to_equity", "其他综合收益/权益（Polars）。", ["other_comprehensive_income", "avg_equity", "period_id"],
    lambda oci, ae, period_id=None: _binary(oci, ae, lambda a, e: _safe_div_expr(a, e)),
)
_register(
    "fin_discontinued_operation_ratio", "终止经营损益/|净利润|（Polars）。", ["discontinued_operation_profit", "net_profit", "period_id"],
    lambda d, np_, period_id=None: _binary(d, np_, lambda a, b: _safe_div_expr(a, b.abs())),
)
_register(
    "fin_minority_profit_share", "少数股东损益/|净利润|（Polars）。", ["minority_profit", "net_profit", "period_id"],
    lambda m, np_, period_id=None: _binary(m, np_, lambda a, b: _safe_div_expr(a, b.abs())),
)
_register(
    "fin_contract_asset_intensity", "合同资产/总资产（Polars）。", ["contract_assets", "total_assets", "period_id"],
    lambda a, ta, period_id=None: _binary(a, ta, lambda x, y: _safe_div_expr(x, y)),
)
_register(
    "fin_contract_liability_intensity", "合同负债/总资产（Polars）。", ["contract_liability", "total_assets", "period_id"],
    lambda l, ta, period_id=None: _binary(l, ta, lambda x, y: _safe_div_expr(x, y)),
)
_register(
    "fin_contract_asset_liability_gap", "(合同资产-合同负债)/总资产（Polars）。", ["contract_assets", "contract_liability", "total_assets", "period_id"],
    lambda a, l, ta, period_id=None: _three(a, l, ta, lambda x, y, z: _safe_div_expr(x - y, z)),
)
_register(
    "fin_lease_intensity", "(使用权+租赁负债)/总资产（Polars）。", ["usufruct_assets", "lease_liability", "total_assets", "period_id"],
    lambda u, l, ta, period_id=None: _three(u, l, ta, lambda x, y, z: _safe_div_expr(x + y, z)),
)
_register(
    "fin_lease_asset_liability_gap", "(使用权-租赁负债)/总资产（Polars）。", ["usufruct_assets", "lease_liability", "total_assets", "period_id"],
    lambda u, l, ta, period_id=None: _three(u, l, ta, lambda x, y, z: _safe_div_expr(x - y, z)),
)
_register(
    "fin_goodwill_intensity", "商誉/总资产（Polars）。", ["goodwill", "total_assets", "period_id"],
    lambda g, ta, period_id=None: _binary(g, ta, lambda x, y: _safe_div_expr(x, y)),
)
_register(
    "fin_deferred_tax_gap", "(递延资产-递延负债)/总资产（Polars）。", ["deferred_tax_assets", "deferred_tax_liability", "total_assets", "period_id"],
    lambda a, l, ta, period_id=None: _three(a, l, ta, lambda x, y, z: _safe_div_expr(x - y, z)),
)
_register(
    "fin_impairment_intensity", "(资产+信用减值)/营收（Polars）。", ["asset_impairment_loss", "credit_impairment_loss", "operating_revenue", "period_id"],
    lambda ai, ci, rev, period_id=None: _three(ai, ci, rev, lambda x, y, z: _safe_div_expr(x + y, z)),
)
_register(
    "fin_borrowing_intensity", "取得借款现金/平均资产（Polars）。", ["cash_from_borrowing", "avg_assets", "period_id"],
    lambda c, aa, period_id=None: _binary(c, aa, lambda x, y: _safe_div_expr(x, y)),
)
_register(
    "fin_debt_repayment_intensity", "偿债支付/平均资产（Polars）。", ["borrowing_repayment", "avg_assets", "period_id"],
    lambda c, aa, period_id=None: _binary(c, aa, lambda x, y: _safe_div_expr(x, y)),
)
_register(
    "fin_net_borrowing_cashflow", "(借款+发债-偿债)/平均资产（Polars）。", ["cash_from_borrowing", "cash_from_bonds_issue", "borrowing_repayment", "avg_assets", "period_id"],
    lambda cb, bo, rp, aa, period_id=None: _four(cb, bo, rp, aa, lambda a, b, c, d: _safe_div_expr(a + b - c, d)),
)
_register(
    "fin_interest_coverage_proxy", "营业利润/|利息|（Polars）。", ["operating_profit", "interest_cost", "period_id"],
    lambda op, ic, period_id=None: _binary(op, ic, lambda x, y: _safe_div_expr(x, y.abs())),
)
_register(
    "fin_debt_service_coverage_proxy", "OCF/(偿债+|利息|)（Polars）。", ["ocf", "borrowing_repayment", "interest_cost", "period_id"],
    lambda o, rp, ic, period_id=None: _three(o, rp, ic, lambda x, y, z: _safe_div_expr(x, y + z.abs())),
)
_register(
    "fin_capex_intensity", "购建长期资产支付/平均资产（Polars）。", ["capex_cash", "avg_assets", "period_id"],
    lambda c, aa, period_id=None: _binary(c, aa, lambda x, y: _safe_div_expr(x, y)),
)
_register(
    "fin_acquisition_cash_intensity", "取得子公司现金/平均资产（Polars）。", ["net_cash_from_subcompany", "avg_assets", "period_id"],
    lambda c, aa, period_id=None: _binary(c, aa, lambda x, y: _safe_div_expr(x, y)),
)
_register(
    "fin_rd_total_intensity", "(研发+资本化)/营收（Polars）。", ["rd_expense", "capitalized_dev_increase", "operating_revenue", "period_id"],
    lambda r, cd, rev, period_id=None: _three(r, cd, rev, lambda x, y, z: _safe_div_expr(x + y, z)),
)
_register(
    "fin_rd_capitalization_ratio", "开发支出资本化率（Polars）。", ["capitalized_dev_increase", "rd_expense", "period_id"],
    lambda cd, r, period_id=None: _binary(cd, r, lambda x, y: _safe_div_expr(x, x + y)),
)
_register(
    "fin_cash_burn_runway", "现金/|年化负OCF|（Polars）。", ["cash_equivalents", "ocf", "period_id"],
    lambda c, o, period_id=None: _binary(c, o, lambda x, y: _safe_div_expr(x, pl.min_horizontal(y, pl.lit(0.0)).abs())),
)
_register(
    "fin_financing_gap", "(资本开支+偿债+分派-OCF)/平均资产（Polars）。", ["capex", "debt_repayment", "dividend_interest_payment", "ocf", "avg_assets", "period_id"],
    lambda cx, dr, di, o, aa, period_id=None: _financing_gap(cx, dr, di, o, aa),
)


def _roe_cash_gap(np_, ocf, ae):
    cols = align_cols(np_, ocf, ae)
    out = []
    for c in cols:
        num = np_[c] - (ocf[c] if c in ocf.columns else pl.lit(float("nan")))
        out.append(_safe_div_expr(num, ae[c]).alias(c))
    return np_.with_columns(out)


def _comp_gap(tci, np_, ae):
    cols = align_cols(tci, np_, ae)
    out = []
    for c in cols:
        num = tci[c] - (np_[c] if c in np_.columns else pl.lit(float("nan")))
        out.append(_safe_div_expr(num, ae[c]).alias(c))
    return tci.with_columns(out)


def _three(a, b, c, expr_fn):
    cols = align_cols(a, b, c)
    return a.with_columns([expr_fn(a[col], b[col], c[col]).alias(col) for col in cols])


def _four(a, b, c, d, expr_fn):
    cols = align_cols(a, b, c, d)
    return a.with_columns([expr_fn(a[col], b[col], c[col], d[col]).alias(col) for col in cols])


def _financing_gap(cx, dr, di, o, aa):
    cols = align_cols(cx, dr, di, o, aa)
    out = []
    for c in cols:
        num = cx[c] + dr[c] + di[c] - o[c]
        out.append(_safe_div_expr(num, aa[c]).alias(c))
    return cx.with_columns(out)
