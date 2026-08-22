# -*- coding: utf-8 -*-
"""DuckDB / SQL capability of next-stage elementwise operators via composite lowering."""
from __future__ import annotations

import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from backend.sql_pushdown.sql_registry import is_sql_capable
from planner.logical_plan import PlanNode
from planner.optimizer import Optimizer

ensure_cleaned_loaded()

_LOWERED = {
    "free_float_ratio": (("free_cap",), ("capitalization",), "safe_div_null"),
    "free_to_circulating_ratio": (("free_cap",), ("circulating_cap",), "safe_div_null"),
    "a_share_cap_ratio": (("a_cap",), ("capitalization",), "safe_div_null"),
    "free_float_turnover": (("volume",), ("free_cap",), "safe_div_null"),
    "valuation_pe_ttm_lyr_gap": (("pe_ratio",), ("pe_ratio_lyr",), "subtract"),
    "valuation_pcf_definition_gap": (("pcf_ratio",), ("pcf_ratio2",), "subtract"),
    "market_cap_free_cap_gap": (("market_cap",), ("free_market_cap",), "subtract"),
    "capital_change_magnitude": (("total_capital",), (), "subtract"),
    "fin_roe_cash_gap": (("net_profit", "ocf"), ("avg_equity",), "safe_div_null"),
    "fin_oci_to_equity": (("oci",), ("avg_equity",), "safe_div_null"),
    "fin_fair_value_income_dependence": (("fv",), ("total_profit",), "safe_div_null"),
    "fin_investment_income_dependence": (("inv",), ("total_profit",), "safe_div_null"),
    "fin_other_earnings_dependence": (("other",), ("total_profit",), "safe_div_null"),
    "fin_minority_profit_share": (("minority",), ("net_profit",), "safe_div_null"),
    "fin_discontinued_operation_ratio": (("discon",), ("net_profit",), "safe_div_null"),
    "fin_contract_asset_intensity": (("contract_assets",), ("total_assets",), "safe_div_null"),
    "fin_contract_liability_intensity": (("contract_liability",), ("total_assets",), "safe_div_null"),
    "fin_contract_asset_liability_gap": (("contract_assets", "contract_liability"), ("total_assets",), "safe_div_null"),
    "fin_lease_intensity": (("usufruct", "lease_liability"), ("total_assets",), "safe_div_null"),
    "fin_goodwill_intensity": (("goodwill",), ("total_assets",), "safe_div_null"),
    "fin_deferred_tax_gap": (("deferred_assets", "deferred_liability"), ("total_assets",), "safe_div_null"),
    "fin_impairment_intensity": (("asset_impairment", "credit_impairment"), ("revenue",), "safe_div_null"),
    "fin_borrowing_intensity": (("borrowing",), ("avg_assets",), "safe_div_null"),
    "fin_debt_repayment_intensity": (("repayment",), ("avg_assets",), "safe_div_null"),
    "fin_net_borrowing_cashflow": (("borrowing", "bonds"), ("repayment", "avg_assets"), "safe_div_null"),
    "fin_interest_coverage_proxy": (("op",), ("interest",), "safe_div_null"),
    "fin_debt_service_coverage_proxy": (("ocf", "repayment"), ("interest",), "safe_div_null"),
    "fin_capex_intensity": (("capex",), ("avg_assets",), "safe_div_null"),
    "fin_acquisition_cash_intensity": (("acq",), ("avg_assets",), "safe_div_null"),
    "fin_rd_total_intensity": (("rd", "capitalized"), ("revenue",), "safe_div_null"),
    "fin_rd_capitalization_ratio": (("capitalized",), ("rd",), "safe_div_null"),
    "holder_pledge_ratio": (("pledge",), ("total_capital",), "safe_div_null"),
    "holder_freeze_ratio": (("freeze",), ("total_capital",), "safe_div_null"),
    "holder_locked_share_ratio": (("locked",), ("total_capital",), "safe_div_null"),
    "holder_float_concentration_gap": (("a",), ("b",), "subtract"),
    "holder_shareholder_network_centrality": (("degree",), ("total",), "safe_div_null"),
    "holder_shareholder_overlap_ratio": (("shared",), ("total",), "safe_div_null"),
    "index_weight_gap_to_free_float": (("iw",), ("fw",), "safe_div_null"),
}


def _col(name: str) -> PlanNode:
    return PlanNode(op="column", attrs={"name": name}, inputs=[])


@pytest.mark.parametrize("name", sorted(_LOWERED))
def test_elementwise_operator_is_sql_capable(name: str) -> None:
    nums, dens, lowered_op = _LOWERED[name]
    inputs = [_col(c) for c in nums] + [_col(c) for c in dens]
    plan = PlanNode(op=name, inputs=inputs, attrs={})
    optimized = Optimizer().optimize(plan)
    assert is_sql_capable(optimized), name
    assert optimized.op == lowered_op, f"{name}: expected {lowered_op}, got {optimized.op}"


def test_stateful_operators_not_sql_lowered() -> None:
    """Stateful / aggregation kernels are intentionally not SQL-lowered."""
    for name in ("ts_kalman_level", "ts_garch_vol_forecast", "ts_permutation_entropy",
                 "intra_realized_skewness", "panel_rolling_pca_loading"):
        plan = PlanNode(op=name, inputs=[_col("x")], attrs={})
        optimized = Optimizer().optimize(plan)
        assert optimized.op == name, f"{name} should not lower"
