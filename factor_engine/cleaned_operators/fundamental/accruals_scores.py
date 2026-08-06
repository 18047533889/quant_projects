# -*- coding: utf-8 -*-
"""Asset growth, revenue quality, financing / solvency and composite scores (P0).

All inputs are PIT-aligned daily panels (``PubDate`` as-of).  Growth / change
terms use the ``period_id`` visible-period semantics from ``transforms_v2``.
Composite scores accept an optional ``mask`` panel (financial-industry
applicability) so ratios stay NaN outside the valid universe.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.fundamental.quality_v2 import _mk
from cleaned_operators.fundamental.transforms_v2 import _lag_value, _walk_periods

_EPS = 1e-12
_CANONICALS: list[str] = []


def _growth(x: pd.DataFrame, period_id: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    p = max(1, int(periods))
    return _walk_periods(x, period_id, lambda o, v, c: _pct_change(float(v[c]), _lag_value(o, v, c, p)))


def _pct_change(cur: float, old: float) -> float:
    if not np.isfinite(old) or abs(old) <= _EPS:
        return np.nan
    return float((cur - old) / abs(old))


def _delta(x: pd.DataFrame, period_id: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    p = max(1, int(periods))
    return _walk_periods(x, period_id, lambda o, v, c: float(v[c]) - _lag_value(o, v, c, p))


def _safe_ratio(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
    return num / den.replace(0, np.nan)


# ---------------------------------------------------------------------------
# § Revenue / asset quality divergence
# ---------------------------------------------------------------------------

def _divergence(a: pd.DataFrame, b: pd.DataFrame, period_id: pd.DataFrame) -> pd.DataFrame:
    return _growth(a, period_id) - _growth(b, period_id)


_mk(
    "fin_receivable_sales_divergence",
    "应收增长 - 营收增长。",
    ["account_receivable", "operating_revenue", "period_id"],
    lambda a, b, period_id: _divergence(a, b, period_id),
)
_mk(
    "fin_inventory_sales_divergence",
    "存货增长 - 营收增长。",
    ["inventories", "operating_revenue", "period_id"],
    lambda a, b, period_id: _divergence(a, b, period_id),
)
_mk(
    "fin_cash_sales_divergence",
    "销售收现增长 - 营收增长。",
    ["goods_sale_cash", "operating_revenue", "period_id"],
    lambda a, b, period_id: _divergence(a, b, period_id),
)
_mk(
    "fin_expense_sales_divergence",
    "期间费用增长 - 营收增长。",
    ["period_expense", "operating_revenue", "period_id"],
    lambda a, b, period_id: _divergence(a, b, period_id),
)


# ---------------------------------------------------------------------------
# § Contract assets / liabilities (new revenue standard)
# ---------------------------------------------------------------------------

_mk(
    "fin_contract_asset_intensity",
    "合同资产 / 总资产。",
    ["contract_assets", "total_assets", "period_id"],
    lambda a, ta, period_id: _safe_ratio(a, ta),
)
_mk(
    "fin_contract_asset_growth",
    "合同资产增长。",
    ["contract_assets", "period_id"],
    lambda a, period_id: _growth(a, period_id),
)
_mk(
    "fin_contract_liability_intensity",
    "合同负债 / 总资产。",
    ["contract_liability", "total_assets", "period_id"],
    lambda l, ta, period_id: _safe_ratio(l, ta),
)
_mk(
    "fin_contract_liability_growth",
    "合同负债增长。",
    ["contract_liability", "period_id"],
    lambda l, period_id: _growth(l, period_id),
)
_mk(
    "fin_contract_asset_liability_gap",
    "(合同资产 - 合同负债) / 总资产。",
    ["contract_assets", "contract_liability", "total_assets", "period_id"],
    lambda a, l, ta, period_id: _safe_ratio(a - l, ta),
)


# ---------------------------------------------------------------------------
# § Leases / goodwill / deferred tax / impairments
# ---------------------------------------------------------------------------

_mk(
    "fin_lease_intensity",
    "(使用权资产 + 租赁负债) / 总资产。",
    ["usufruct_assets", "lease_liability", "total_assets", "period_id"],
    lambda u, l, ta, period_id: _safe_ratio(u + l, ta),
)
_mk(
    "fin_lease_asset_liability_gap",
    "(使用权资产 - 租赁负债) / 总资产。",
    ["usufruct_assets", "lease_liability", "total_assets", "period_id"],
    lambda u, l, ta, period_id: _safe_ratio(u - l, ta),
)
_mk(
    "fin_goodwill_intensity",
    "商誉 / 总资产。",
    ["goodwill", "total_assets", "period_id"],
    lambda g, ta, period_id: _safe_ratio(g, ta),
)
_mk(
    "fin_deferred_tax_gap",
    "(递延所得税资产 - 递延所得税负债) / 总资产。",
    ["deferred_tax_assets", "deferred_tax_liability", "total_assets", "period_id"],
    lambda a, l, ta, period_id: _safe_ratio(a - l, ta),
)
_mk(
    "fin_impairment_intensity",
    "(资产减值损失 + 信用减值损失) / 营业收入。",
    ["asset_impairment_loss", "credit_impairment_loss", "operating_revenue", "period_id"],
    lambda ai, ci, rev, period_id: _safe_ratio(ai + ci, rev),
)


def _fin_goodwill_risk_score(goodwill, goodwill_prev, asset_impairment, credit_impairment, total_assets, period_id):
    intensity = _safe_ratio(goodwill, total_assets)
    gw_growth = _safe_ratio(_delta(goodwill, period_id), goodwill_prev.replace(0, np.nan))
    imp = _safe_ratio(asset_impairment + credit_impairment, total_assets)
    return intensity + gw_growth + imp


_mk(
    "fin_goodwill_risk_score",
    "商誉风险：GoodWill/Assets + GoodWill增长 + 减值/Assets。",
    ["goodwill", "goodwill_prev", "asset_impairment_loss", "credit_impairment_loss", "total_assets", "period_id"],
    _fin_goodwill_risk_score,
)


# ---------------------------------------------------------------------------
# § Financing / solvency / cash burn
# ---------------------------------------------------------------------------

_mk(
    "fin_net_debt_issuance",
    "Δ(短贷+长贷+应付债券)/平均资产。",
    ["short_term_loan", "long_term_loan", "bonds_payable", "avg_assets", "period_id"],
    lambda s, l, b, aa, period_id: _safe_ratio(_delta(s + l + b, period_id), aa),
)
_mk(
    "fin_borrowing_intensity",
    "取得借款收到的现金 / 平均资产。",
    ["cash_from_borrowing", "avg_assets", "period_id"],
    lambda c, aa, period_id: _safe_ratio(c, aa),
)
_mk(
    "fin_debt_repayment_intensity",
    "偿还债务支付的现金 / 平均资产。",
    ["borrowing_repayment", "avg_assets", "period_id"],
    lambda c, aa, period_id: _safe_ratio(c, aa),
)
_mk(
    "fin_net_borrowing_cashflow",
    "(借款 + 发债 - 偿债) / 平均资产。",
    ["cash_from_borrowing", "cash_from_bonds_issue", "borrowing_repayment", "avg_assets", "period_id"],
    lambda cb, bo, rp, aa, period_id: _safe_ratio(cb + bo - rp, aa),
)
_mk(
    "fin_equity_capital_growth",
    "实收资本+资本公积增长率（股权融资代理）。",
    ["paidin_capital", "capital_reserve", "period_id"],
    lambda p, c, period_id: _growth(p + c, period_id),
)
_mk(
    "fin_financing_gap",
    "(资本开支 + 偿债 + 分派股利利息 - OCF) / 平均资产。",
    ["capex", "debt_repayment", "dividend_interest_payment", "ocf", "avg_assets", "period_id"],
    lambda c, d, di, o, aa, period_id: _safe_ratio(c + d + di - o, aa),
)
_mk(
    "fin_interest_coverage_proxy",
    "营业利润 / |利息费用|。",
    ["operating_profit", "interest_cost", "period_id"],
    lambda op, ic, period_id: _safe_ratio(op, ic.abs()),
)
_mk(
    "fin_debt_service_coverage_proxy",
    "OCF / (偿债 + |利息费用|)。",
    ["ocf", "borrowing_repayment", "interest_cost", "period_id"],
    lambda o, rp, ic, period_id: _safe_ratio(o, rp + ic.abs()),
)
_mk(
    "fin_cash_burn_runway",
    "现金及等价物 / |年化负OCF|（OCF为负时）。",
    ["cash_equivalents", "ocf", "period_id"],
    lambda c, o, period_id: _burn_runway(c, o),
)


def _burn_runway(cash_equivalents: pd.DataFrame, ocf: pd.DataFrame) -> pd.DataFrame:
    neg_ocf = ocf.where(ocf < 0, np.nan).abs()
    return cash_equivalents / neg_ocf.replace(0, np.nan)


_mk(
    "fin_capex_intensity",
    "购建固定资产无形资产其他长期资产支付 / 平均资产。",
    ["capex_cash", "avg_assets", "period_id"],
    lambda c, aa, period_id: _safe_ratio(c, aa),
)
_mk(
    "fin_capex_growth",
    "资本开支增长率。",
    ["capex_cash", "period_id"],
    lambda c, period_id: _growth(c, period_id),
)
_mk(
    "fin_acquisition_cash_intensity",
    "取得子公司支付现金 / 平均资产。",
    ["net_cash_from_subcompany", "avg_assets", "period_id"],
    lambda c, aa, period_id: _safe_ratio(c, aa),
)
_mk(
    "fin_rd_total_intensity",
    "(研发费用 + 开发支出资本化增加) / 营业收入（资本化部分为代理值）。",
    ["rd_expense", "capitalized_dev_increase", "operating_revenue", "period_id"],
    lambda r, cd, rev, period_id: _safe_ratio(r + cd, rev),
)
_mk(
    "fin_rd_capitalization_ratio",
    "开发支出资本化增加 / (研发费用 + 资本化增加)（代理值）。",
    ["capitalized_dev_increase", "rd_expense", "period_id"],
    lambda cd, r, period_id: _safe_ratio(cd, cd + r),
)


# ---------------------------------------------------------------------------
# § Composite scores
# ---------------------------------------------------------------------------

def _masked(score: pd.DataFrame, mask: pd.DataFrame | None) -> pd.DataFrame:
    if mask is None:
        return score
    return score.where(mask.astype(bool), np.nan)


def _fin_piotroski_f_score(roa, ocf, net_profit, leverage, current_ratio, total_capital,
                           gross_margin, asset_turnover, period_id, mask=None):
    roa_up = roa > roa.shift(1)
    lev_down = leverage < leverage.shift(1)
    cr_up = current_ratio > current_ratio.shift(1)
    cap_flat = total_capital.abs() <= total_capital.shift(1).abs() * 1.05
    gm_up = gross_margin > gross_margin.shift(1)
    at_up = asset_turnover > asset_turnover.shift(1)
    score = (
        (roa > 0).astype(float) + (ocf > 0).astype(float) + roa_up.astype(float)
        + (ocf > net_profit).astype(float) + lev_down.astype(float) + cr_up.astype(float)
        + cap_flat.astype(float) + gm_up.astype(float) + at_up.astype(float)
    )
    return _masked(score, mask)


_mk(
    "piotroski_f_score",
    "Piotroski F-score（9 项布尔加总，需外部适用性掩码）。",
    ["roa", "ocf", "net_profit", "leverage", "current_ratio", "total_capital",
     "gross_margin", "asset_turnover", "period_id", "mask"],
    _fin_piotroski_f_score,
)


def _fin_altman_z_score(working_capital, retained_earnings, operating_profit, total_assets,
                        market_cap, total_liabilities, revenue, period_id, mask=None):
    z = (
        1.2 * _safe_ratio(working_capital, total_assets)
        + 1.4 * _safe_ratio(retained_earnings, total_assets)
        + 3.3 * _safe_ratio(operating_profit, total_assets)
        + 0.6 * _safe_ratio(market_cap, total_liabilities)
        + 1.0 * _safe_ratio(revenue, total_assets)
    )
    return _masked(z, mask)


_mk(
    "altman_z_score",
    "Altman Z-score（非金融企业版）。",
    ["working_capital", "retained_earnings", "operating_profit", "total_assets",
     "market_cap", "total_liabilities", "revenue", "period_id", "mask"],
    _fin_altman_z_score,
)


def _fin_zmijewski_score(net_profit, total_assets, total_liabilities, current_assets,
                         current_liabilities, period_id, mask=None):
    x = (
        -4.336
        - 4.513 * _safe_ratio(net_profit, total_assets)
        + 5.679 * _safe_ratio(total_liabilities, total_assets)
        + 0.004 * _safe_ratio(current_assets, current_liabilities)
    )
    return _masked(x, mask)


_mk(
    "zmijewski_score",
    "Zmijewski 破产概率得分。",
    ["net_profit", "total_assets", "total_liabilities", "current_assets", "current_liabilities", "period_id", "mask"],
    _fin_zmijewski_score,
)


def _fin_fundamental_strength_score(roa, ocf, gross_margin, asset_turnover, leverage,
                                    receivable_turnover, inventory_turnover, revenue_growth,
                                    period_id, mask=None):
    """基本面强度：方向数组求和（正=越强越好，负=越高越差）。"""
    score = (
        roa + _safe_ratio(ocf, roa.abs().replace(0, np.nan)) + gross_margin
        + asset_turnover - leverage + receivable_turnover + inventory_turnover + revenue_growth
    )
    return _masked(score, mask)


_mk(
    "fin_fundamental_strength_score",
    "基本面强度：多组件方向求和。",
    ["roa", "ocf", "gross_margin", "asset_turnover", "leverage",
     "receivable_turnover", "inventory_turnover", "revenue_growth", "period_id", "mask"],
    _fin_fundamental_strength_score,
)

import cleaned_operators.operator_surface as _surface  # noqa: E402

_surface.EXTENDED_ONLY_CANONICALS = frozenset(
    set(_surface.EXTENDED_ONLY_CANONICALS) | set(_CANONICALS)
)
