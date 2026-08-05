# -*- coding: utf-8
"""Audited rewrites for the single-default fundamental library.

Rewrites are metadata only until their required field/source contracts and
runtime backend evidence exist.  No unavailable formula is replaced by a stub.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FormulaRewrite:
    factor_id: str
    status: str
    expression: str | None
    reason: str
    required_preprocessing: tuple[str, ...] = ()


REWRITES: dict[str, FormulaRewrite] = {
    "PROF044": FormulaRewrite(
        "PROF044", "requires_source_preprocessing",
        "safe_div(ttm_from_cumulative(inc_operating_revenue, inc_report_period_end_date, inc_fiscal_quarter, revision_policy='latest_available'), employee_count)",
        "TTM operating revenue divided by employee count; cumulative flow requires PIT fiscal-quarter conversion.",
        ("employee_count",),
    ),
    "OWN010": FormulaRewrite(
        "OWN010", "parseable",
        "period_change(safe_div(shagg_top10_share_ratio, 100.0), sh_report_period_end_date, periods=1, mode='absolute', require_consecutive=False, revision_policy='latest_available')",
        "holder concentration change uses strict fiscal/event period change.",
    ),
    "OWN011": FormulaRewrite(
        "OWN011", "requires_source_preprocessing",
        "period_change(shagg_top1_share_ratio, sh_report_period_end_date, periods=1, mode='absolute', require_consecutive=False, revision_policy='latest_available')",
        "relation snapshot must be materialized before period change.",
    ),
    "OWN012": FormulaRewrite(
        "OWN012", "requires_source_preprocessing",
        "period_change(shagg_top10_share_ratio, sh_report_period_end_date, periods=1, mode='absolute', require_consecutive=False, revision_policy='latest_available')",
        "relation snapshot must be materialized before period change.",
    ),
    "OWN015": FormulaRewrite(
        "OWN015", "requires_source_preprocessing", None,
        "relation_jaccard requires PIT-visible relation snapshots and cannot use an ordinary numeric panel.",
    ),
    "OWN029": FormulaRewrite(
        "OWN029", "requires_source_preprocessing", None,
        "relation_entropy requires entity, weight, and PIT snapshot_id; source-side aggregation is required.",
    ),
    "SUP_FCON001": FormulaRewrite(
        "SUP_FCON001", "parseable",
        "add(add(multiply(-0.737, ln(safe_div(val_market_cap, 1000000.0))), multiply(0.043, power(ln(safe_div(val_market_cap, 1000000.0)), 2.0))), multiply(-0.04, safe_div(date_diff_days(trade_date, list_start_date), 365.25)))",
        "Years since listing is computed from explicit decision trade_date and list_start_date; no wall-clock date is read.",
    ),
    "ACC037": FormulaRewrite(
        "ACC037", "unavailable", None,
        "group_multi_resid is not implemented; group_cs_resid/list-child semantics are rejected.",
    ),
    "T3_REM008": FormulaRewrite(
        "T3_REM008", "unavailable", None,
        "group_multi_resid with strict fiscal inputs is not yet certified.",
    ),
    "T3_REM009": FormulaRewrite(
        "T3_REM009", "unavailable", None,
        "group_multi_resid with strict fiscal inputs is not yet certified.",
    ),
}


def rewrite_for(factor_id: str) -> FormulaRewrite | None:
    return REWRITES.get(str(factor_id))
