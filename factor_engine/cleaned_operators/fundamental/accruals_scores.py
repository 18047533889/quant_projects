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
    "合同资产 / 总资产。period_id 仅对齐契约，不参与计算。",
    ["contract_assets", "total_assets"],
    lambda a, ta, period_id=None: _safe_ratio(a, ta),
)
_mk(
    "fin_contract_asset_growth",
    "合同资产增长。",
    ["contract_assets", "period_id"],
    lambda a, period_id: _growth(a, period_id),
)
_mk(
    "fin_contract_liability_intensity",
    "合同负债 / 总资产。period_id 仅对齐契约，不参与计算。",
    ["contract_liability", "total_assets"],
    lambda l, ta, period_id=None: _safe_ratio(l, ta),
)
_mk(
    "fin_contract_liability_growth",
    "合同负债增长。",
    ["contract_liability", "period_id"],
    lambda l, period_id: _growth(l, period_id),
)
_mk(
    "fin_contract_asset_liability_gap",
    "(合同资产 - 合同负债) / 总资产。period_id 仅对齐契约，不参与计算。",
    ["contract_assets", "contract_liability", "total_assets"],
    lambda a, l, ta, period_id=None: _safe_ratio(a - l, ta),
)


# ---------------------------------------------------------------------------
# § Leases / goodwill / deferred tax / impairments
# ---------------------------------------------------------------------------

_mk(
    "fin_lease_intensity",
    "(使用权资产 + 租赁负债) / 总资产。period_id 仅对齐契约，不参与计算。",
    ["usufruct_assets", "lease_liability", "total_assets"],
    lambda u, l, ta, period_id=None: _safe_ratio(u + l, ta),
)
_mk(
    "fin_lease_asset_liability_gap",
    "(使用权资产 - 租赁负债) / 总资产。period_id 仅对齐契约，不参与计算。",
    ["usufruct_assets", "lease_liability", "total_assets"],
    lambda u, l, ta, period_id=None: _safe_ratio(u - l, ta),
)
_mk(
    "fin_goodwill_intensity",
    "商誉 / 总资产。period_id 仅对齐契约，不参与计算。",
    ["goodwill", "total_assets"],
    lambda g, ta, period_id=None: _safe_ratio(g, ta),
)
_mk(
    "fin_deferred_tax_gap",
    "(递延所得税资产 - 递延所得税负债) / 总资产。period_id 仅对齐契约，不参与计算。",
    ["deferred_tax_assets", "deferred_tax_liability", "total_assets"],
    lambda a, l, ta, period_id=None: _safe_ratio(a - l, ta),
)
_mk(
    "fin_impairment_intensity",
    "(资产减值损失 + 信用减值损失) / 营业收入。period_id 仅对齐契约，不参与计算。",
    ["asset_impairment_loss", "credit_impairment_loss", "operating_revenue"],
    lambda ai, ci, rev, period_id=None: _safe_ratio(ai + ci, rev),
)


def _fin_goodwill_risk_score(goodwill, asset_impairment, credit_impairment, total_assets, period_id):
    # P1-134: the caller-supplied ``goodwill_prev`` was a second "prior period"
    # source that could disagree with the fiscal-ordinal prior computed from
    # ``period_id``.  Only the fiscal-ordinal prior is retained — growth is
    # ``_growth`` (current - prior)/|prior| walked over visible report periods.
    intensity = _safe_ratio(goodwill, total_assets)
    gw_growth = _growth(goodwill, period_id)
    imp = _safe_ratio(asset_impairment + credit_impairment, total_assets)
    return intensity + gw_growth + imp


_mk(
    "fin_goodwill_risk_score",
    "商誉风险：GoodWill/Assets + GoodWill增长(fiscal-ordinal) + 减值/Assets。",
    ["goodwill", "asset_impairment_loss", "credit_impairment_loss", "total_assets", "period_id"],
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
    "取得借款收到的现金 / 平均资产。period_id 仅对齐契约，不参与计算。",
    ["cash_from_borrowing", "avg_assets"],
    lambda c, aa, period_id=None: _safe_ratio(c, aa),
)
_mk(
    "fin_debt_repayment_intensity",
    "偿还债务支付的现金 / 平均资产。period_id 仅对齐契约，不参与计算。",
    ["borrowing_repayment", "avg_assets"],
    lambda c, aa, period_id=None: _safe_ratio(c, aa),
)
_mk(
    "fin_net_borrowing_cashflow",
    "(借款 + 发债 - 偿债) / 平均资产。period_id 仅对齐契约，不参与计算。",
    ["cash_from_borrowing", "cash_from_bonds_issue", "borrowing_repayment", "avg_assets"],
    lambda cb, bo, rp, aa, period_id=None: _safe_ratio(cb + bo - rp, aa),
)
_mk(
    "fin_equity_capital_growth",
    "实收资本+资本公积增长率（股权融资代理）。",
    ["paidin_capital", "capital_reserve", "period_id"],
    lambda p, c, period_id: _growth(p + c, period_id),
)
_mk(
    "fin_financing_gap",
    "(资本开支 + 偿债 + 分派股利利息 - OCF) / 平均资产。period_id 仅对齐契约，不参与计算。",
    ["capex", "debt_repayment", "dividend_interest_payment", "ocf", "avg_assets"],
    lambda c, d, di, o, aa, period_id=None: _safe_ratio(c + d + di - o, aa),
)
_mk(
    "fin_interest_coverage_proxy",
    "营业利润 / |利息费用|。period_id 仅对齐契约，不参与计算。",
    ["operating_profit", "interest_cost"],
    lambda op, ic, period_id=None: _safe_ratio(op, ic.abs()),
)
_mk(
    "fin_debt_service_coverage_proxy",
    "OCF / (偿债 + |利息费用|)。period_id 仅对齐契约，不参与计算。",
    ["ocf", "borrowing_repayment", "interest_cost"],
    lambda o, rp, ic, period_id=None: _safe_ratio(o, rp + ic.abs()),
)
_mk(
    "fin_cash_burn_runway",
    "现金及等价物 / |负OCF|（输入 OCF 须为 TTM/年化口径；单季 OCF 会低估 runway 年化月数）。",
    ["cash_equivalents", "annualized_ocf"],
    lambda c, o, period_id=None: _burn_runway(c, o),
)


def _burn_runway(cash_equivalents: pd.DataFrame, ocf: pd.DataFrame) -> pd.DataFrame:
    # P1-133: the doc used to claim "现金/|年化负OCF|" but the code did not
    # annualize — cash/|OCF| on a quarterly/semi-annual/TTM input each mean
    # different things.  The operator now documents the input as TTM/annualized
    # OCF; no hidden annualization is performed.
    neg_ocf = ocf.where(ocf < 0, np.nan).abs()
    return cash_equivalents / neg_ocf.replace(0, np.nan)


_mk(
    "fin_capex_intensity",
    "购建固定资产无形资产其他长期资产支付 / 平均资产。period_id 仅对齐契约，不参与计算。",
    ["capex_cash", "avg_assets"],
    lambda c, aa, period_id=None: _safe_ratio(c, aa),
)
_mk(
    "fin_capex_growth",
    "资本开支增长率。",
    ["capex_cash", "period_id"],
    lambda c, period_id: _growth(c, period_id),
)
_mk(
    "fin_acquisition_cash_intensity",
    "取得子公司支付现金 / 平均资产。period_id 仅对齐契约，不参与计算。",
    ["net_cash_from_subcompany", "avg_assets"],
    lambda c, aa, period_id=None: _safe_ratio(c, aa),
)
_mk(
    "fin_rd_total_intensity",
    "(研发费用 + 开发支出资本化增加) / 营业收入（资本化部分为代理值）。period_id 仅对齐契约，不参与计算。",
    ["rd_expense", "capitalized_dev_increase", "operating_revenue"],
    lambda r, cd, rev, period_id=None: _safe_ratio(r + cd, rev),
)
_mk(
    "fin_rd_capitalization_ratio",
    "开发支出资本化增加 / (研发费用 + 资本化增加)（代理值）。period_id 仅对齐契约，不参与计算。",
    ["capitalized_dev_increase", "rd_expense"],
    lambda cd, r, period_id=None: _safe_ratio(cd, cd + r),
)


# ---------------------------------------------------------------------------
# § Composite scores
# ---------------------------------------------------------------------------

def _masked(score: pd.DataFrame, mask: pd.DataFrame | None) -> pd.DataFrame:
    if mask is None:
        return score
    # An *unknown* applicability (NaN) must stay NaN — never coerce to truthy.
    # ``NaN.astype(bool)`` is True in NumPy/Pandas, which silently kept unknown
    # names in the factor — 3rd-round audit P0-09.
    valid_mask = mask.notna() & mask.ne(0)
    return score.where(valid_mask, np.nan)


def _fin_piotroski_f_score(roa, ocf, net_profit, leverage, current_ratio, total_capital,
                           gross_margin, asset_turnover, period_id, mask=None):
    # Period-over-period comparisons must use fiscal ordinals, never trading-day
    # shifts: with daily PubDate as-of ffill, ``roa.shift(1)`` is almost always
    # the SAME report period (both Q1), so every improvement test was pinned
    # False.  ``_delta``/``_growth`` walk the visible report-period sequence —
    # 3rd-round audit P0-08.
    roa_up = _delta(roa, period_id) > 0
    lev_down = _delta(leverage, period_id) < 0
    cr_up = _delta(current_ratio, period_id) > 0
    gm_up = _delta(gross_margin, period_id) > 0
    at_up = _delta(asset_turnover, period_id) > 0
    # Equity issuance: paid-in+reserves grew >5% from the prior fiscal period.
    cap_flat = _growth(total_capital.abs(), period_id) <= 0.05
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
