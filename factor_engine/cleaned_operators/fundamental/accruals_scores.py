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
from cleaned_operators.fundamental.quality_v2 import (
    _mk,
    _reject_ytd_growth,
    _require_same_flow_grain,
)
from cleaned_operators.fundamental.transforms_v2 import _lag_value, _pos_int, _walk_periods
from cleaned_operators.fiscal_strict import GROWTH_FORBIDDEN_FLOW_TYPES, flow_types as _flow_types

_EPS = 1e-12
_CANONICALS: list[str] = []

# R11: the fundamental-strength score is only comparable across stocks with
# comparable field coverage.  Fewer than this many observed components -> NaN
# (a stock with 3/8 fields must not be scored against one with 8/8).
MIN_FUNDAMENTAL_COMPONENTS = 6

# Round-3 item 30: input-semantic declarations for threshold-relative score
# operators.  Each signal tests ``field > 0`` / ``field_delta > 0`` — meaningful
# only for a Return/Rate/flow field, never a raw Price/Volume (``Price > 0`` /
# ``Volume > 0`` are almost always True).  Exposed as ``input_units`` metadata.
_PIOTROSKI_INPUT_UNITS = {
    "roa": "rate",
    "ocf": "rate",
    "net_profit": "rate",
    "leverage": "rate",
    "current_ratio": "rate",
    "total_capital": "stock",
    "gross_margin": "rate",
    "asset_turnover": "rate",
    "period_id": "fiscal_period",
}
_ALTZMI_INPUT_UNITS = {
    "working_capital": "stock",
    "retained_earnings": "stock",
    "operating_profit": "flow",
    "total_assets": "stock",
    "market_cap": "stock",
    "total_liabilities": "stock",
    "revenue": "flow",
    "net_profit": "flow",
    "current_assets": "stock",
    "current_liabilities": "stock",
    "period_id": "fiscal_period",
}
_STRENGTH_INPUT_UNITS = {
    "roa": "rate",
    "ocf": "rate",
    "gross_margin": "rate",
    "asset_turnover": "rate",
    "leverage": "rate",
    "receivable_turnover": "rate",
    "inventory_turnover": "rate",
    "avg_assets": "stock",
    "revenue_growth": "rate",
    "period_id": "fiscal_period",
}


def _growth(x: pd.DataFrame, period_id: pd.DataFrame, periods: int = 1, flow_type=None) -> pd.DataFrame:
    # Finding #52: growth over a cumulative-YTD input is not a period growth
    # rate; a caller that declares the YTD grain is rejected at the boundary.
    _reject_ytd_growth("fin_growth", flow_type)
    # R23-295: strict integer authority — reject True/3.7/nan/inf/"3.7", never
    # coerce (the old ``max(1, int(periods))`` silently truncated 3.7 -> 3).
    p = _pos_int(periods, "periods")
    return _walk_periods(x, period_id, lambda o, v, c: _pct_change(float(v[c]), _lag_value(o, v, c, p)))


def _pct_change(cur: float, old: float) -> float:
    if not np.isfinite(old) or abs(old) <= _EPS:
        return np.nan
    return np.where(abs(old)) != 0, float((cur - old) / abs(old)), np.nan)


def _delta(x: pd.DataFrame, period_id: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    # R23-295: strict integer authority (see ``_growth``).
    p = _pos_int(periods, "periods")
    return _walk_periods(x, period_id, lambda o, v, c: float(v[c]) - _lag_value(o, v, c, p))


def _safe_ratio(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
    return np.where(den.replace(0, np.nan) != 0, num / den.replace(0, np.nan), np.nan)


def _net_borrowing_cashflow(cb, bo, rp, aa, flow_type=None):
    # #51: the three financing cash-flow inputs must share one reporting grain.
    _require_same_flow_grain("fin_net_borrowing_cashflow", flow_type, 3)
    return _safe_ratio(cb + bo - rp, aa)


def _financing_gap(c, d, di, o, aa, flow_type=None):
    # #51: capex / debt repayment / distributions / OCF must share one grain.
    _require_same_flow_grain("fin_financing_gap", flow_type, 4)
    return _safe_ratio(c + d + di - o, aa)


# ---------------------------------------------------------------------------
# § Revenue / asset quality divergence
# ---------------------------------------------------------------------------

def _divergence(a: pd.DataFrame, b: pd.DataFrame, period_id: pd.DataFrame, flow_type=None) -> pd.DataFrame:
    # R23-296 (per-slot grain contract): ``a`` is a balance-sheet Stock (e.g.
    # receivables / inventories) and ``b`` is a Flow (e.g. revenue).  A single
    # "same flow grain" rule over-rejects the valid (Stock, Flow) pair and a
    # ``flow_type=None`` skips all checking — so the contract is per-slot:
    #   slot A: Stock | SinglePeriodFlow | TTMFlow
    #   slot B: SinglePeriodFlow | TTMFlow
    #   comparison interval: SAME fiscal interval (both walks share period_id)
    # A CumulativeYTDFlow on either slot is rejected — its growth is not a
    # period growth rate.
    if flow_type is not None:
        types = _flow_types(flow_type, 2)
        for slot_label, t in zip(("a", "b"), types):
            if t in GROWTH_FORBIDDEN_FLOW_TYPES:
                raise ValueError(
                    f"fin_divergence slot {slot_label}: growth over a "
                    f"CumulativeYTDFlow input is not a period growth rate; "
                    "convert with fin_quarter_from_cumulative first."
                )
    # Pass flow_type=None to _growth: the per-slot YTD rejection is already
    # enforced at the divergence boundary above, and the same-grain rule must
    # not fire on a legitimate (Stock, Flow) pair.
    return _growth(a, period_id) - _growth(b, period_id)


_mk(
    "fin_receivable_sales_divergence",
    "应收增长 - 营收增长（per-slot：slot_a=Stock，slot_b=SinglePeriodFlow，同一 fiscal interval；R23-296）。",
    ["account_receivable", "operating_revenue", "period_id", "flow_type"],
    lambda a, b, period_id, flow_type=None: _divergence(a, b, period_id, flow_type),
    extra_tags=["flow_slot_a:Stock", "flow_slot_b:SinglePeriodFlow", "comparison_interval:same_fiscal_interval"],
)
_mk(
    "fin_inventory_sales_divergence",
    "存货增长 - 营收增长（per-slot：slot_a=Stock，slot_b=SinglePeriodFlow；R23-296）。",
    ["inventories", "operating_revenue", "period_id", "flow_type"],
    lambda a, b, period_id, flow_type=None: _divergence(a, b, period_id, flow_type),
    extra_tags=["flow_slot_a:Stock", "flow_slot_b:SinglePeriodFlow", "comparison_interval:same_fiscal_interval"],
)
_mk(
    "fin_cash_sales_divergence",
    "销售收现增长 - 营收增长（per-slot：slot_a=Stock，slot_b=SinglePeriodFlow；R23-296）。",
    ["goods_sale_cash", "operating_revenue", "period_id", "flow_type"],
    lambda a, b, period_id, flow_type=None: _divergence(a, b, period_id, flow_type),
    extra_tags=["flow_slot_a:Stock", "flow_slot_b:SinglePeriodFlow", "comparison_interval:same_fiscal_interval"],
)
_mk(
    "fin_expense_sales_divergence",
    "期间费用增长 - 营收增长（per-slot：slot_a=Stock，slot_b=SinglePeriodFlow；R23-296）。",
    ["period_expense", "operating_revenue", "period_id", "flow_type"],
    lambda a, b, period_id, flow_type=None: _divergence(a, b, period_id, flow_type),
    extra_tags=["flow_slot_a:Stock", "flow_slot_b:SinglePeriodFlow", "comparison_interval:same_fiscal_interval"],
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
    ["contract_assets", "period_id", "flow_type"],
    lambda a, period_id, flow_type=None: _growth(a, period_id, flow_type=flow_type),
    extra_tags=["flow_type:Stock"],
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
    ["contract_liability", "period_id", "flow_type"],
    lambda l, period_id, flow_type=None: _growth(l, period_id, flow_type=flow_type),
    extra_tags=["flow_type:Stock"],
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


def _fin_goodwill_risk_score(goodwill, asset_impairment, credit_impairment, total_assets, period_id, flow_type=None):
    # P1-134: the caller-supplied ``goodwill_prev`` was a second "prior period"
    # source that could disagree with the fiscal-ordinal prior computed from
    # ``period_id``.  Only the fiscal-ordinal prior is retained — growth is
    # ``_growth`` (current - prior)/|prior| walked over visible report periods.
    intensity = _safe_ratio(goodwill, total_assets)
    gw_growth = _growth(goodwill, period_id, flow_type=flow_type)
    imp = _safe_ratio(asset_impairment + credit_impairment, total_assets)
    return intensity + gw_growth + imp


_mk(
    "fin_goodwill_risk_score",
    "商誉风险：GoodWill/Assets + GoodWill增长(fiscal-ordinal) + 减值/Assets。",
    ["goodwill", "asset_impairment_loss", "credit_impairment_loss", "total_assets", "period_id", "flow_type"],
    _fin_goodwill_risk_score,
    extra_tags=["flow_type:Stock"],
)


# ---------------------------------------------------------------------------
# § Financing / solvency / cash burn
# ---------------------------------------------------------------------------

_mk(
    "fin_net_debt_issuance",
    "Δ(短贷+长贷+应付债券)/平均资产。",
    ["short_term_loan", "long_term_loan", "bonds_payable", "avg_assets", "period_id", "flow_type"],
    lambda s, l, b, aa, period_id, flow_type=None: _safe_ratio(_delta(s + l + b, period_id), aa),
    extra_tags=["flow_type:Stock", "flow_grain:period_delta"],
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
    ["cash_from_borrowing", "cash_from_bonds_issue", "borrowing_repayment", "avg_assets", "period_id", "flow_type"],
    lambda cb, bo, rp, aa, period_id=None, flow_type=None: _net_borrowing_cashflow(cb, bo, rp, aa, flow_type),
    extra_tags=["flow_type:SinglePeriodFlow"],
)
_mk(
    "fin_equity_capital_growth",
    "实收资本+资本公积增长率（股权融资代理）。",
    ["paidin_capital", "capital_reserve", "period_id", "flow_type"],
    lambda p, c, period_id, flow_type=None: _growth(p + c, period_id, flow_type=flow_type),
    extra_tags=["flow_type:Stock"],
)
_mk(
    "fin_financing_gap",
    "(资本开支 + 偿债 + 分派股利利息 - OCF) / 平均资产。period_id 仅对齐契约，不参与计算。",
    ["capex", "debt_repayment", "dividend_interest_payment", "ocf", "avg_assets", "period_id", "flow_type"],
    lambda c, d, di, o, aa, period_id=None, flow_type=None: _financing_gap(c, d, di, o, aa, flow_type),
    extra_tags=["flow_type:SinglePeriodFlow"],
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
    return np.where(neg_ocf.replace(0, np.nan) != 0, cash_equivalents / neg_ocf.replace(0, np.nan), np.nan)


_mk(
    "fin_capex_intensity",
    "购建固定资产无形资产其他长期资产支付 / 平均资产。period_id 仅对齐契约，不参与计算。",
    ["capex_cash", "avg_assets"],
    lambda c, aa, period_id=None: _safe_ratio(c, aa),
)
_mk(
    "fin_capex_growth",
    "资本开支增长率。",
    ["capex_cash", "period_id", "flow_type"],
    lambda c, period_id, flow_type=None: _growth(c, period_id, flow_type=flow_type),
    extra_tags=["flow_type:SinglePeriodFlow"],
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
    # ApplicabilityBool contract (R11 round-2, review section 十五): an
    # applicability mask is a STRICT boolean panel — the only valid cell values
    # are {0, 1, NaN}.  A finite value that is neither 0 nor 1 is a contract
    # violation and fails closed (raises).  NaN and 0 both mean "not applicable"
    # (masked out); 1 means "applicable".  The old ``mask.notna() & mask.ne(0)``
    # treated -1/0.4/2 all as "applicable", silently scoring names the caller
    # never meant to include.  An *unknown* applicability (NaN) must stay NaN —
    # never coerce to truthy: ``NaN.astype(bool)`` is True in NumPy/Pandas, which
    # silently kept unknown names in the factor (3rd-round audit P0-09).
    m = mask.reindex(index=score.index, columns=score.columns)
    try:
        arr = m.to_numpy(dtype=float)
    except (TypeError, ValueError):  # object / non-numeric mask — coerce leniently
        arr = m.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(arr)
    invalid = finite & (arr != 0.0) & (arr != 1.0)
    if invalid.any():
        bad = np.unique(arr[invalid])
        raise ValueError(
            "applicability mask must be bool-valued: only 0/1/NaN are valid; "
            f"got finite value(s) {bad.tolist()!r}"
        )
    applicable = pd.DataFrame(arr == 1.0, index=score.index, columns=score.columns)
    return score.where(applicable, np.nan)


def _piotroski_components(roa, ocf, net_profit, leverage, current_ratio, total_capital,
                          gross_margin, asset_turnover, period_id, issuance_threshold=0.05):
    """Return ``(passed, observed, normalized_partial)`` for the 9 F-score signals.

    Finding #50: a NaN comparison must never be silently converted to a "failed
    signal" (``NaN > 0 -> False -> 0``).  Each of the 9 signals is scored ONLY
    where its inputs are finite; the primary output is the count of PASSED
    signals among the OBSERVED components, the observed count is exposed for
    callers that need completeness, and ``normalized_partial`` maps the partial
    score to [0, 1] (NaN when nothing is observed).

    ``issuance_threshold`` is the maximum share-capital + capital-reserve growth
    (``cap_g``) at which the equity-issuance signal still passes.  R11 round-2
    split: the strict full score passes ``issuance_threshold=0.0`` (no issuance
    at all), while the tolerant canonical and the partial / observed-count paths
    retain the historical ``0.05`` (5%) tolerance.  The observed count is
    threshold-independent.
    """
    # Period-over-period comparisons must use fiscal ordinals, never trading-day
    # shifts: with daily PubDate as-of ffill, ``roa.shift(1)`` is almost always
    # the SAME report period (both Q1), so every improvement test was pinned
    # False.  ``_delta``/``_growth`` walk the visible report-period sequence —
    # 3rd-round audit P0-08.
    roa_d = _delta(roa, period_id)
    lev_d = _delta(leverage, period_id)
    cr_d = _delta(current_ratio, period_id)
    gm_d = _delta(gross_margin, period_id)
    at_d = _delta(asset_turnover, period_id)
    cap_g = _growth(total_capital.abs(), period_id)

    passed = (
        ((roa > 0) & roa.notna()).astype(float)
        + ((ocf > 0) & ocf.notna()).astype(float)
        + ((roa_d > 0) & roa_d.notna()).astype(float)
        + ((ocf > net_profit) & ocf.notna() & net_profit.notna()).astype(float)
        + ((lev_d < 0) & lev_d.notna()).astype(float)
        + ((cr_d > 0) & cr_d.notna()).astype(float)
        + ((cap_g <= issuance_threshold) & cap_g.notna()).astype(float)
        + ((gm_d > 0) & gm_d.notna()).astype(float)
        + ((at_d > 0) & at_d.notna()).astype(float)
    )
    observed = (
        roa.notna().astype(float)
        + ocf.notna().astype(float)
        + roa_d.notna().astype(float)
        + (ocf.notna() & net_profit.notna()).astype(float)
        + lev_d.notna().astype(float)
        + cr_d.notna().astype(float)
        + cap_g.notna().astype(float)
        + gm_d.notna().astype(float)
        + at_d.notna().astype(float)
    )
    normalized = passed / observed.where(observed > 0, np.nan)
    return passed, observed, normalized


def _piotroski_observed_count(roa, ocf, net_profit, leverage, current_ratio, total_capital,
                              gross_margin, asset_turnover, period_id) -> pd.DataFrame:
    """Number of F-score signals with all inputs observed (finding #50)."""
    _, observed, _ = _piotroski_components(roa, ocf, net_profit, leverage, current_ratio,
                                           total_capital, gross_margin, asset_turnover, period_id)
    return observed


def _piotroski_normalized_partial_score(roa, ocf, net_profit, leverage, current_ratio,
                                        total_capital, gross_margin, asset_turnover,
                                        period_id) -> pd.DataFrame:
    """Partial F-score normalised to [0, 1] by the observed-component count."""
    _, _, normalized = _piotroski_components(roa, ocf, net_profit, leverage, current_ratio,
                                             total_capital, gross_margin, asset_turnover, period_id)
    return normalized


def _fin_piotroski_f_score(roa, ocf, net_profit, leverage, current_ratio, total_capital,
                           gross_margin, asset_turnover, period_id, mask=None):
    """Strict genuine full F-score: the passed count only when all 9 signals are observed.

    R11 round-2 definition split (P0, review section 十四): the strict
    ``piotroski_f_score`` uses the original no-share-issuance criterion — the
    equity-issuance signal passes ONLY when share capital + capital reserve did
    not grow at all (``cap_g <= 0``; a buyback / return of capital also passes).
    Any positive share-capital growth fails the signal.  The 5%-tolerant proxy
    is the separate ``piotroski_f_score_tolerant`` canonical (cap_g <= 5%).
    """
    passed, observed, _ = _piotroski_components(roa, ocf, net_profit, leverage, current_ratio,
                                                total_capital, gross_margin, asset_turnover,
                                                period_id, issuance_threshold=0.0)
    # R11 completeness: a partial F-score (observed < 9) is NOT comparable with a
    # genuine full F-score — fail closed to NaN instead of under-reporting (a
    # "4 of 4 observed, all passed -> 4" row must not equal "9 of 9, 4 passed").
    full = passed.where(observed == 9)
    return _masked(full, mask)


def _fin_piotroski_f_score_tolerant(roa, ocf, net_profit, leverage, current_ratio, total_capital,
                                    gross_margin, asset_turnover, period_id, mask=None):
    """5%-tolerant full F-score: same 9-component completeness as ``piotroski_f_score``,
    but the equity-issuance signal passes under the relaxed ``cap_g <= 5%`` tolerance
    (the historical ``piotroski_f_score`` behavior kept as its own economic definition).
    """
    passed, observed, _ = _piotroski_components(roa, ocf, net_profit, leverage, current_ratio,
                                                total_capital, gross_margin, asset_turnover,
                                                period_id, issuance_threshold=0.05)
    full = passed.where(observed == 9)
    return _masked(full, mask)


def _fin_piotroski_partial_score(roa, ocf, net_profit, leverage, current_ratio, total_capital,
                                 gross_margin, asset_turnover, period_id, mask=None):
    """Partial F-score: passed / observed fraction in [0, 1]; NaN when observed == 0."""
    _, _, normalized = _piotroski_components(roa, ocf, net_profit, leverage, current_ratio,
                                             total_capital, gross_margin, asset_turnover, period_id)
    return _masked(normalized, mask)


def _fin_piotroski_observed_count(roa, ocf, net_profit, leverage, current_ratio, total_capital,
                                  gross_margin, asset_turnover, period_id, mask=None):
    """Number of F-score signals whose inputs are all observed (0..9)."""
    _, observed, _ = _piotroski_components(roa, ocf, net_profit, leverage, current_ratio,
                                           total_capital, gross_margin, asset_turnover, period_id)
    return _masked(observed, mask)


# R11 round-2 definition split (P0, review section 十四): the old
# ``piotroski_f_score`` branded itself a "genuine full F-score" while its
# equity-issuance signal passed at cap_g <= 5% — a tolerant proxy, not the strict
# "no share issuance" criterion.  Split into two genuinely different economic
# definitions (not a parameter tweak):
#   * ``piotroski_f_score``           — strict: issuance passes only at cap_g <= 0.
#   * ``piotroski_f_score_tolerant``  — 5% tolerance (cap_g <= 0.05).
# The strict canonical keeps the historical ``piotroski_f_score`` spelling, so
# the old name resolves to the strict definition (no rename/alias migration is
# required — the canonical key itself is unchanged).
_mk(
    "piotroski_f_score",
    "Piotroski F-score（严格股权发行判定）：仅当 9 项信号全部观测到时返回通过数（0..9），"
    "否则 NaN（#50）。股权发行信号仅在股本+资本公积无增长（cap_g<=0）时通过（R11 round-2 拆分）。"
    "输入语义=Return/Rate 型财务字段，非原始价格/成交量（round-3 item 30）。",
    ["roa", "ocf", "net_profit", "leverage", "current_ratio", "total_capital",
     "gross_margin", "asset_turnover", "period_id", "mask"],
    _fin_piotroski_f_score,
    extra_tags=["flow_type:SinglePeriodFlow", "applicable_universe:non_financial", "input_semantics:rate"],
    input_units=_PIOTROSKI_INPUT_UNITS,
)
_mk(
    "piotroski_f_score_tolerant",
    "Piotroski F-score（5% 股权发行容忍）：与 piotroski_f_score 同构（9 项全观测才打分），"
    "仅股权发行信号放宽为 cap_g<=5%（历史 piotroski_f_score 行为的独立经济定义）。"
    "输入语义=Return/Rate 型财务字段（round-3 item 30）。",
    ["roa", "ocf", "net_profit", "leverage", "current_ratio", "total_capital",
     "gross_margin", "asset_turnover", "period_id", "mask"],
    _fin_piotroski_f_score_tolerant,
    extra_tags=["flow_type:SinglePeriodFlow", "applicable_universe:non_financial", "input_semantics:rate"],
    input_units=_PIOTROSKI_INPUT_UNITS,
)
_mk(
    "piotroski_partial_score",
    "Piotroski 部分得分：passed / observed ∈ [0,1]，observed==0 时为 NaN（#50）。"
    "输入语义=Return/Rate 型财务字段（round-3 item 30）。",
    ["roa", "ocf", "net_profit", "leverage", "current_ratio", "total_capital",
     "gross_margin", "asset_turnover", "period_id", "mask"],
    _fin_piotroski_partial_score,
    extra_tags=["flow_type:SinglePeriodFlow", "applicable_universe:non_financial", "input_semantics:rate"],
    input_units=_PIOTROSKI_INPUT_UNITS,
)
_mk(
    "piotroski_observed_count",
    "Piotroski 观测分量数（0..9），用于完整度诊断（#50）。"
    "输入语义=Return/Rate 型财务字段（round-3 item 30）。",
    ["roa", "ocf", "net_profit", "leverage", "current_ratio", "total_capital",
     "gross_margin", "asset_turnover", "period_id", "mask"],
    _fin_piotroski_observed_count,
    extra_tags=["flow_type:SinglePeriodFlow", "applicable_universe:non_financial", "input_semantics:rate"],
    input_units=_PIOTROSKI_INPUT_UNITS,
)


# Finding #55: Altman Z and Zmijewski X are calibrated on non-financial
# industrial firms.  A direct interpretation on financial/insurance names (and
# other capital-structure-distorted industries) is not valid.  The operators
# DECLARE this applicability contract and fail closed (NaN) wherever the caller
# supplies an applicability ``mask`` that marks the universe inapplicable.
ApplicableUniverse = "non_financial"
_APPLICABLE_TAG = f"applicable_universe:{ApplicableUniverse}"


def _fin_altman_z_score(working_capital, retained_earnings, operating_profit, total_assets,
                        market_cap, total_liabilities, revenue, period_id, mask=None):
    # R11 applicability is a HARD semantic: the calibration universe is
    # non-financial, so the caller MUST supply an applicability mask.  Omitting
    # it would silently score financials with a formula that does not apply.
    if mask is None:
        raise ValueError("altman_z_score requires an applicability mask (applicable_universe=non_financial)")
    z = (
        1.2 * _safe_ratio(working_capital, total_assets)
        + 1.4 * _safe_ratio(retained_earnings, total_assets)
        + 3.3 * _safe_ratio(operating_profit, total_assets)
        + 0.6 * _safe_ratio(market_cap, total_liabilities)
        + 1.0 * _safe_ratio(revenue, total_assets)
    )
    # #55: financials are outside the calibration universe — ``mask`` is the
    # fail-closed gate; an all-inapplicable universe yields an all-NaN panel.
    return _masked(z, mask)


_mk(
    "altman_z_score",
    "Altman Z-score（非金融企业版；适用域=非金融，适用性掩码为必填，#55）。"
    "输入=资产负债表存量/损益流量字段（round-3 item 30）。",
    ["working_capital", "retained_earnings", "operating_profit", "total_assets",
     "market_cap", "total_liabilities", "revenue", "period_id", "mask"],
    _fin_altman_z_score,
    extra_tags=[_APPLICABLE_TAG],
    input_units=_ALTZMI_INPUT_UNITS,
)


def _fin_zmijewski_score(net_profit, total_assets, total_liabilities, current_assets,
                         current_liabilities, period_id, mask=None):
    # R11 applicability is a HARD semantic: the calibration universe is
    # non-financial, so the caller MUST supply an applicability mask.  Omitting
    # it would silently score financials with a formula that does not apply.
    if mask is None:
        raise ValueError("zmijewski_score requires an applicability mask (applicable_universe=non_financial)")
    x = (
        -4.336
        - 4.513 * _safe_ratio(net_profit, total_assets)
        + 5.679 * _safe_ratio(total_liabilities, total_assets)
        + 0.004 * _safe_ratio(current_assets, current_liabilities)
    )
    return _masked(x, mask)


_mk(
    "zmijewski_score",
    "Zmijewski 破产概率得分（适用域=非金融，适用性掩码为必填，#55）。"
    "输入=资产负债表存量/损益流量字段（round-3 item 30）。",
    ["net_profit", "total_assets", "total_liabilities", "current_assets", "current_liabilities", "period_id", "mask"],
    _fin_zmijewski_score,
    extra_tags=[_APPLICABLE_TAG],
    input_units=_ALTZMI_INPUT_UNITS,
)


def _cs_rank_normalize(x: pd.DataFrame, direction: float) -> pd.DataFrame:
    """Cross-sectional percentile rank mapped to [-1, 1] per date.

    Each date row ranks the instrument columns; ``direction`` = +1 means
    higher-is-better, -1 higher-is-worse.  A component with no finite values on
    a row stays NaN.
    """
    ranks = x.rank(axis=1, pct=True)
    return direction * (2.0 * ranks - 1.0)


def _fundamental_strength_components(roa, ocf, gross_margin, asset_turnover, leverage,
                                     receivable_turnover, inventory_turnover, avg_assets,
                                     revenue_growth):
    """Return ``(normalized, stacked, valid, counts)`` for the 8 strength components."""
    # R11 semantic change: cash_yield := OCF / AverageAssets.  The old OCF/|ROA|
    # mixed a flow AMOUNT with a dimensionless RATIO and was therefore still
    # amount-dimensioned (dominated by company size) rather than a genuine cash
    # yield per unit of assets.
    cash_yield = _safe_ratio(ocf, avg_assets)
    normalized = [
        _cs_rank_normalize(roa, +1.0),
        _cs_rank_normalize(cash_yield, +1.0),
        _cs_rank_normalize(gross_margin, +1.0),
        _cs_rank_normalize(asset_turnover, +1.0),
        _cs_rank_normalize(leverage, -1.0),
        _cs_rank_normalize(receivable_turnover, +1.0),
        _cs_rank_normalize(inventory_turnover, +1.0),
        _cs_rank_normalize(revenue_growth, +1.0),
    ]
    stacked = np.stack([df.to_numpy(dtype=float) for df in normalized])
    valid = np.isfinite(stacked)
    counts = valid.sum(axis=0)
    return normalized, stacked, valid, counts


def _fin_fundamental_strength_score(roa, ocf, gross_margin, asset_turnover, leverage,
                                    receivable_turnover, inventory_turnover, avg_assets,
                                    revenue_growth, period_id, mask=None):
    """基本面强度：各分量先做截面 rank 标准化（[-1,1] 同向）再对观测分量取均值。

    Finding #54: adding heterogeneous RAW metrics lets the component with the
    largest units/scale decide the weight (ROA ~0.05 vs asset_turnover ~1.0 vs
    revenue_growth ~0.1).  Each component is cross-sectionally rank-normalised
    per date with its direction applied BEFORE combining.  R11 (#50): the score
    is the MEAN over the OBSERVED components (not the sum), so a stock with 3/8
    field coverage is not directly compared against a stock with 8/8; and a cell
    with fewer than ``MIN_FUNDAMENTAL_COMPONENTS`` observed components is NaN.
    """
    _, stacked, valid, counts = _fundamental_strength_components(
        roa, ocf, gross_margin, asset_turnover, leverage,
        receivable_turnover, inventory_turnover, avg_assets, revenue_growth,
    )
    mean = np.where(valid, stacked, 0.0).sum(axis=0) / np.where(counts > 0, counts, 1)
    mean[counts < MIN_FUNDAMENTAL_COMPONENTS] = np.nan
    score = pd.DataFrame(mean, index=roa.index, columns=roa.columns)
    return _masked(score, mask)


def _fin_fundamental_strength_coverage(roa, ocf, gross_margin, asset_turnover, leverage,
                                       receivable_turnover, inventory_turnover, avg_assets,
                                       revenue_growth, period_id, mask=None):
    """基本面强度覆盖度：每个 cell 观测到的分量数（0..8）。"""
    _, _, valid, counts = _fundamental_strength_components(
        roa, ocf, gross_margin, asset_turnover, leverage,
        receivable_turnover, inventory_turnover, avg_assets, revenue_growth,
    )
    cov = pd.DataFrame(counts.astype(float), index=roa.index, columns=roa.columns)
    return _masked(cov, mask)


_mk(
    "fin_fundamental_strength_score",
    "基本面强度：多分量截面 rank 标准化后对观测分量取均值；<6 个观测分量为 NaN（#54，#50）。"
    "分量输入语义=Return/Rate 型财务字段（round-3 item 30）。",
    ["roa", "ocf", "gross_margin", "asset_turnover", "leverage",
     "receivable_turnover", "inventory_turnover", "avg_assets", "revenue_growth", "period_id", "mask"],
    _fin_fundamental_strength_score,
    extra_tags=["flow_type:SinglePeriodFlow", "input_semantics:rate"],
    input_units=_STRENGTH_INPUT_UNITS,
)
_mk(
    "fin_fundamental_strength_coverage",
    "基本面强度覆盖度：观测到的分量数（0..8），用于覆盖度诊断。"
    "分量输入语义=Return/Rate 型财务字段（round-3 item 30）。",
    ["roa", "ocf", "gross_margin", "asset_turnover", "leverage",
     "receivable_turnover", "inventory_turnover", "avg_assets", "revenue_growth", "period_id", "mask"],
    _fin_fundamental_strength_coverage,
    extra_tags=["flow_type:SinglePeriodFlow", "input_semantics:rate"],
    input_units=_STRENGTH_INPUT_UNITS,
)

import cleaned_operators.operator_surface as _surface  # noqa: E402

_surface.extend_extended_only(set(_CANONICALS))
