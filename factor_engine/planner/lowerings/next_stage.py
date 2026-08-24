# -*- coding: utf-8
"""Next-stage elementwise composite lowerings.

Expands the next-stage elementwise operators into core primitives that already
have SQL / DuckDB pushdown, so the lowered plans become SQL-capable without
per-operator emitter code.  Only elementwise / single-shift mappings are
registered here; panel-aggregation and stateful kernels are excluded.
"""
from __future__ import annotations

from factor_engine.planner.composite_lowering import register_lowering
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.lowerings import _helpers as H


def _abs(node: PlanNode) -> PlanNode:
    return PlanNode(op="abs", inputs=[node], attrs={})


def _log_abs(node: PlanNode) -> PlanNode:
    return PlanNode(op="log", inputs=[_abs(node)], attrs={})


def _safe_div(a: PlanNode, b: PlanNode) -> PlanNode:
    return H.safe_div(a, b)


def _sub(a: PlanNode, b: PlanNode) -> PlanNode:
    return H.binop("subtract", a, b)


def _add(a: PlanNode, b: PlanNode) -> PlanNode:
    return H.binop("add", a, b)


def _ratio(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 2:
        return node
    return _safe_div(node.inputs[0], node.inputs[1])


def _ratio_abs_den(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 2:
        return node
    return _safe_div(node.inputs[0], _abs(node.inputs[1]))


def _gap(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 2:
        return node
    return _sub(_log_abs(node.inputs[0]), _log_abs(node.inputs[1]))


def _three(node: PlanNode, op, combine) -> PlanNode:
    if len(node.inputs) < 3:
        return node
    return combine(node.inputs[0], node.inputs[1], node.inputs[2])


# -- valuation -------------------------------------------------------------
for _name in ("free_float_ratio", "free_to_circulating_ratio", "a_share_cap_ratio", "free_float_turnover"):
    register_lowering(_name)(_ratio)

register_lowering("valuation_pe_ttm_lyr_gap")(_gap)
register_lowering("valuation_pcf_definition_gap")(_gap)
register_lowering("market_cap_free_cap_gap")(_gap)


@register_lowering("capital_change_magnitude")
def _lower_capital_change_magnitude(node: PlanNode) -> PlanNode:
    if not node.inputs:
        return node
    tc = node.inputs[0]
    prev = H.delay(tc, 1)
    return _sub(_safe_div(tc, prev), H.literal(1.0))


# -- financial elementwise ------------------------------------------------
register_lowering("fin_oci_to_equity")(_ratio)
register_lowering("fin_contract_asset_intensity")(_ratio)
register_lowering("fin_contract_liability_intensity")(_ratio)
register_lowering("fin_goodwill_intensity")(_ratio)
register_lowering("fin_borrowing_intensity")(_ratio)
register_lowering("fin_debt_repayment_intensity")(_ratio)
register_lowering("fin_capex_intensity")(_ratio)
register_lowering("fin_acquisition_cash_intensity")(_ratio)
register_lowering("fin_interest_coverage_proxy")(_ratio_abs_den)

for _name in (
    "fin_fair_value_income_dependence",
    "fin_investment_income_dependence",
    "fin_other_earnings_dependence",
    "fin_minority_profit_share",
    "fin_discontinued_operation_ratio",
):
    register_lowering(_name)(_ratio_abs_den)


@register_lowering("fin_roe_cash_gap")
def _lower_roe_cash_gap(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 3:
        return node
    return _safe_div(_sub(node.inputs[0], node.inputs[1]), node.inputs[2])


@register_lowering("fin_contract_asset_liability_gap")
def _lower_contract_gap(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 3:
        return node
    return _safe_div(_sub(node.inputs[0], node.inputs[1]), node.inputs[2])


@register_lowering("fin_lease_intensity")
def _lower_lease_intensity(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 3:
        return node
    return _safe_div(_add(node.inputs[0], node.inputs[1]), node.inputs[2])


@register_lowering("fin_deferred_tax_gap")
def _lower_deferred_tax_gap(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 3:
        return node
    return _safe_div(_sub(node.inputs[0], node.inputs[1]), node.inputs[2])


@register_lowering("fin_impairment_intensity")
def _lower_impairment(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 3:
        return node
    return _safe_div(_add(node.inputs[0], node.inputs[1]), node.inputs[2])


@register_lowering("fin_net_borrowing_cashflow")
def _lower_net_borrowing(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 4:
        return node
    return _safe_div(_sub(_add(node.inputs[0], node.inputs[1]), node.inputs[2]), node.inputs[3])


@register_lowering("fin_rd_total_intensity")
def _lower_rd_total(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 3:
        return node
    return _safe_div(_add(node.inputs[0], node.inputs[1]), node.inputs[2])


@register_lowering("fin_rd_capitalization_ratio")
def _lower_rd_capitalization(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 2:
        return node
    return _safe_div(node.inputs[0], _add(node.inputs[0], node.inputs[1]))


@register_lowering("fin_debt_service_coverage_proxy")
def _lower_debt_service(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 3:
        return node
    return _safe_div(node.inputs[0], _add(node.inputs[1], _abs(node.inputs[2])))


# -- shareholder elementwise ----------------------------------------------
for _name in (
    "holder_pledge_ratio",
    "holder_freeze_ratio",
    "holder_locked_share_ratio",
    "holder_shareholder_network_centrality",
    "holder_shareholder_overlap_ratio",
):
    register_lowering(_name)(_ratio)


@register_lowering("holder_float_concentration_gap")
def _lower_float_gap(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 2:
        return node
    return _sub(node.inputs[0], node.inputs[1])


# -- index / listing ------------------------------------------------------
@register_lowering("index_weight_gap_to_free_float")
def _lower_index_weight_gap(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 2:
        return node
    return _safe_div(_sub(node.inputs[0], node.inputs[1]), _abs(node.inputs[1]))
