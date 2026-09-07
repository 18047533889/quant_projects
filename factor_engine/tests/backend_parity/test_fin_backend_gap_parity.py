# -*- coding: utf-8
"""fin_* elementwise algebraic family: pandas / polars-long / DuckDB parity.

These operators are pure elementwise algebra (ratio / abs-ratio / sum-diff-ratio
over up to seven operands).  The trailing ``period_id`` input is structural
PIT-alignment only and does not participate in the computation, so the long
polars and DuckDB backends compile only the formula operands.

The PIT/period-walk fin_* operators (fin_lag / fin_pct_change / fin_ttm /
fin_revision_* / fin_days_since_update / fin_seasonal_* / fin_trend_* /
fin_growth_* / fin_earnings_persistence / fin_surprise_event_* / fin_beat_streak
/ fin_miss_streak / fin_period_restated / fin_period_revision_* / fin_staleness
/ fin_announcement_lag / fin_applicability_mask / fin_fundamental_strength_* /
fin_component_score / fin_working_capital_* / fin_delta_noa /
fin_total_operating_accruals / fin_earnings_cash_gap_volatility /
fin_earnings_smoothness / fin_net_debt_issuance / fin_equity_capital_growth /
fin_capex_growth / fin_contract_*_growth / fin_cash_sales_divergence /
fin_expense_sales_divergence / fin_inventory_sales_divergence /
fin_receivable_sales_divergence / fin_divergence / fin_growth_acceleration /
fin_growth_change / fin_growth_volatility / fin_growth_stability /
fin_growth_persistence / fin_stability / fin_zscore_history /
fin_percentile_history / fin_zscore_vs_prior_history /
fin_percentile_vs_prior_history / fin_average_balance / fin_ttm_quarterly /
fin_ttm_cumulative / fin_quarter_from_cumulative / fin_expectation_revision* /
fin_days_since_expectation_revision / period_stability) require a fiscal-ordinal
period walk and are intentionally NOT single-column native SQL — they stay on the
existing polars registry bridge / pandas reference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource

load_all()


# ---------------------------------------------------------------------------
# Fixtures: a daily panel with the operand columns each fin_* op needs.
# ---------------------------------------------------------------------------

def _panel(n_days: int = 30, seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    instruments = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product(
        [dates, instruments], names=["timestamp", "instrument"]
    )
    n = len(idx)

    def _series(base, scale, *, zero_frac=0.0, nan_frac=0.0, neg_frac=0.0):
        s = pd.Series(rng.normal(base, scale, n), index=idx)
        if zero_frac:
            z = rng.choice(n, int(n * zero_frac), replace=False)
            s.iloc[z] = 0.0
        if nan_frac:
            z = rng.choice(n, int(n * nan_frac), replace=False)
            s.iloc[z] = np.nan
        if neg_frac:
            z = rng.choice(n, int(n * neg_frac), replace=False)
            s.iloc[z] = -abs(s.iloc[z])
        return s

    return InMemorySeriesSource(data={
        "earnings": _series(100, 20, zero_frac=0.05, nan_frac=0.05),
        "cashflow": _series(80, 15, zero_frac=0.05, nan_frac=0.05, neg_frac=0.1),
        "assets": _series(1000, 200, zero_frac=0.03, nan_frac=0.03),
        "revenue": _series(500, 100, zero_frac=0.03, nan_frac=0.03),
        "op": _series(60, 10, zero_frac=0.03, nan_frac=0.03),
        "inv_inc": _series(5, 2, zero_frac=0.1, nan_frac=0.05, neg_frac=0.2),
        "fv_inc": _series(3, 1, zero_frac=0.1, nan_frac=0.05, neg_frac=0.2),
        "asset_deal": _series(2, 1, zero_frac=0.1, nan_frac=0.05, neg_frac=0.2),
        "other_earn": _series(4, 1, zero_frac=0.1, nan_frac=0.05, neg_frac=0.2),
        "nonop_rev": _series(3, 1, zero_frac=0.1, nan_frac=0.05),
        "nonop_exp": _series(2, 1, zero_frac=0.1, nan_frac=0.05),
        "total_profit": _series(50, 10, zero_frac=0.03, nan_frac=0.03, neg_frac=0.1),
        "net_profit": _series(40, 8, zero_frac=0.03, nan_frac=0.03, neg_frac=0.1),
        "avg_equity": _series(800, 100, zero_frac=0.03, nan_frac=0.03),
        "avg_assets": _series(1000, 200, zero_frac=0.03, nan_frac=0.03),
        "goodwill": _series(50, 10, zero_frac=0.1, nan_frac=0.05),
        "impair": _series(3, 1, zero_frac=0.1, nan_frac=0.05, neg_frac=0.2),
        "credit_impair": _series(2, 1, zero_frac=0.1, nan_frac=0.05, neg_frac=0.2),
        "lease_liab": _series(30, 5, zero_frac=0.1, nan_frac=0.05),
        "usufruct": _series(25, 5, zero_frac=0.1, nan_frac=0.05),
        "contract_assets": _series(20, 4, zero_frac=0.1, nan_frac=0.05),
        "contract_liab": _series(15, 3, zero_frac=0.1, nan_frac=0.05),
        "total_assets": _series(1000, 200, zero_frac=0.03, nan_frac=0.03),
        "dta": _series(10, 2, zero_frac=0.1, nan_frac=0.05),
        "dtl": _series(8, 2, zero_frac=0.1, nan_frac=0.05),
        "rd_exp": _series(20, 4, zero_frac=0.1, nan_frac=0.05),
        "cap_dev": _series(5, 1, zero_frac=0.1, nan_frac=0.05),
        "borrow": _series(30, 6, zero_frac=0.1, nan_frac=0.05),
        "bonds": _series(10, 2, zero_frac=0.1, nan_frac=0.05),
        "repay": _series(15, 3, zero_frac=0.1, nan_frac=0.05),
        "capex": _series(40, 8, zero_frac=0.1, nan_frac=0.05),
        "div_int": _series(8, 2, zero_frac=0.1, nan_frac=0.05),
        "ocf": _series(70, 12, zero_frac=0.05, nan_frac=0.05, neg_frac=0.1),
        "interest_cost": _series(5, 1, zero_frac=0.1, nan_frac=0.05),
        "cash_eq": _series(200, 30, zero_frac=0.05, nan_frac=0.05),
        "annual_ocf": _series(280, 40, zero_frac=0.05, nan_frac=0.05, neg_frac=0.2),
        "exp_std": _series(2, 0.5, zero_frac=0.1, nan_frac=0.05),
        "exp_mean": _series(10, 2, zero_frac=0.1, nan_frac=0.05),
        "actual": _series(12, 2, zero_frac=0.05, nan_frac=0.05),
        "expected": _series(10, 2, zero_frac=0.05, nan_frac=0.05),
        "scale_base": _series(5, 1, zero_frac=0.1, nan_frac=0.05),
        "discon": _series(2, 1, zero_frac=0.1, nan_frac=0.05, neg_frac=0.2),
        "minority": _series(3, 1, zero_frac=0.1, nan_frac=0.05, neg_frac=0.2),
        "oci": _series(4, 1, zero_frac=0.1, nan_frac=0.05, neg_frac=0.2),
        "total_income": _series(50, 10, zero_frac=0.03, nan_frac=0.03, neg_frac=0.1),
    })


@pytest.fixture(scope="module")
def source():
    return _panel()


def _run(source, expr, backend_name: str):
    return FactorEngine(
        backend=build_backend(backend_name), data_source=source, run_mode="research"
    ).run(Factor(name="t", expr=expr))


def _result(run_out) -> pd.Series:
    return run_out["result"].sort_index()


def _assert_parity(source, expr, rtol=1e-9, atol=1e-12):
    pandas_out = _result(_run(source, expr, "pandas"))
    polars_out = _result(_run(source, expr, "polars_long"))
    pd.testing.assert_series_equal(
        pandas_out, polars_out, check_names=False, rtol=rtol, atol=atol
    )
    duckdb_out = _result(_run(source, expr, "duckdb_sql"))
    pd.testing.assert_series_equal(
        pandas_out, duckdb_out, check_names=False, rtol=rtol, atol=atol
    )


# ---------------------------------------------------------------------------
# Elementwise algebraic family (pure Expr / pure SQL).
# ---------------------------------------------------------------------------

ELEMENTWISE_CASES = [
    ("fin_common_size", lambda c: make_cleaned_call_factory("fin_common_size")(c("op"), c("revenue"))),
    ("fin_cash_conversion", lambda c: make_cleaned_call_factory("fin_cash_conversion")(c("cashflow"), c("earnings"))),
    ("fin_acquisition_cash_intensity", lambda c: make_cleaned_call_factory("fin_acquisition_cash_intensity")(c("capex"), c("avg_assets"))),
    ("fin_borrowing_intensity", lambda c: make_cleaned_call_factory("fin_borrowing_intensity")(c("borrow"), c("avg_assets"))),
    ("fin_capex_intensity", lambda c: make_cleaned_call_factory("fin_capex_intensity")(c("capex"), c("avg_assets"))),
    ("fin_goodwill_intensity", lambda c: make_cleaned_call_factory("fin_goodwill_intensity")(c("goodwill"), c("total_assets"))),
    ("fin_debt_repayment_intensity", lambda c: make_cleaned_call_factory("fin_debt_repayment_intensity")(c("repay"), c("avg_assets"))),
    ("fin_contract_asset_intensity", lambda c: make_cleaned_call_factory("fin_contract_asset_intensity")(c("contract_assets"), c("total_assets"))),
    ("fin_contract_liability_intensity", lambda c: make_cleaned_call_factory("fin_contract_liability_intensity")(c("contract_liab"), c("total_assets"))),
    ("fin_oci_to_equity", lambda c: make_cleaned_call_factory("fin_oci_to_equity")(c("oci"), c("avg_equity"))),
    ("fin_interest_coverage_proxy", lambda c: make_cleaned_call_factory("fin_interest_coverage_proxy")(c("op"), c("interest_cost"))),
    ("fin_discontinued_operation_ratio", lambda c: make_cleaned_call_factory("fin_discontinued_operation_ratio")(c("discon"), c("net_profit"))),
    ("fin_minority_profit_share", lambda c: make_cleaned_call_factory("fin_minority_profit_share")(c("minority"), c("net_profit"))),
    ("fin_fair_value_income_dependence", lambda c: make_cleaned_call_factory("fin_fair_value_income_dependence")(c("fv_inc"), c("total_profit"))),
    ("fin_investment_income_dependence", lambda c: make_cleaned_call_factory("fin_investment_income_dependence")(c("inv_inc"), c("total_profit"))),
    ("fin_other_earnings_dependence", lambda c: make_cleaned_call_factory("fin_other_earnings_dependence")(c("other_earn"), c("total_profit"))),
    ("fin_rd_capitalization_ratio", lambda c: make_cleaned_call_factory("fin_rd_capitalization_ratio")(c("cap_dev"), c("rd_exp"))),
    ("fin_expectation_dispersion", lambda c: make_cleaned_call_factory("fin_expectation_dispersion")(c("exp_std"), c("exp_mean"))),
    ("fin_cash_burn_runway", lambda c: make_cleaned_call_factory("fin_cash_burn_runway")(c("cash_eq"), c("annual_ocf"))),
    ("fin_accrual_ratio", lambda c: make_cleaned_call_factory("fin_accrual_ratio")(c("earnings"), c("cashflow"), c("assets"))),
    ("fin_cash_earnings_gap", lambda c: make_cleaned_call_factory("fin_cash_earnings_gap")(c("earnings"), c("cashflow"), c("scale_base"))),
    ("fin_impairment_intensity", lambda c: make_cleaned_call_factory("fin_impairment_intensity")(c("impair"), c("credit_impair"), c("revenue"))),
    ("fin_lease_intensity", lambda c: make_cleaned_call_factory("fin_lease_intensity")(c("usufruct"), c("lease_liab"), c("total_assets"))),
    ("fin_rd_total_intensity", lambda c: make_cleaned_call_factory("fin_rd_total_intensity")(c("rd_exp"), c("cap_dev"), c("revenue"))),
    ("fin_contract_asset_liability_gap", lambda c: make_cleaned_call_factory("fin_contract_asset_liability_gap")(c("contract_assets"), c("contract_liab"), c("total_assets"))),
    ("fin_lease_asset_liability_gap", lambda c: make_cleaned_call_factory("fin_lease_asset_liability_gap")(c("usufruct"), c("lease_liab"), c("total_assets"))),
    ("fin_deferred_tax_gap", lambda c: make_cleaned_call_factory("fin_deferred_tax_gap")(c("dta"), c("dtl"), c("total_assets"))),
    ("fin_comprehensive_income_gap", lambda c: make_cleaned_call_factory("fin_comprehensive_income_gap")(c("total_income"), c("net_profit"), c("avg_equity"))),
    ("fin_roe_cash_gap", lambda c: make_cleaned_call_factory("fin_roe_cash_gap")(c("net_profit"), c("ocf"), c("avg_equity"))),
    ("fin_debt_service_coverage_proxy", lambda c: make_cleaned_call_factory("fin_debt_service_coverage_proxy")(c("ocf"), c("repay"), c("interest_cost"))),
    ("fin_actual_expectation_divergence", lambda c: make_cleaned_call_factory("fin_actual_expectation_divergence")(c("actual"), c("expected"), c("scale_base"))),
    ("fin_surprise", lambda c: make_cleaned_call_factory("fin_surprise")(c("actual"), c("expected"), c("scale_base"))),
    ("fin_net_borrowing_cashflow", lambda c: make_cleaned_call_factory("fin_net_borrowing_cashflow")(c("borrow"), c("bonds"), c("repay"), c("avg_assets"))),
    ("fin_financing_gap", lambda c: make_cleaned_call_factory("fin_financing_gap")(c("capex"), c("repay"), c("div_int"), c("ocf"), c("avg_assets"))),
    ("fin_core_earnings_ratio", lambda c: make_cleaned_call_factory("fin_core_earnings_ratio")(c("op"), c("inv_inc"), c("fv_inc"), c("asset_deal"), c("other_earn"), c("revenue"))),
    ("fin_noncore_income_ratio", lambda c: make_cleaned_call_factory("fin_noncore_income_ratio")(c("inv_inc"), c("fv_inc"), c("asset_deal"), c("other_earn"), c("nonop_rev"), c("nonop_exp"), c("total_profit"))),
]


@pytest.mark.parametrize(
    ("name", "expr_builder"),
    ELEMENTWISE_CASES,
    ids=[name for name, _ in ELEMENTWISE_CASES],
)
def test_fin_elementwise_three_backend_parity(source, name, expr_builder):
    _assert_parity(source, expr_builder(col))


def test_fin_elementwise_ops_registered_in_sql_and_polars_native():
    from factor_engine.backend.polars_long_policy import POLARS_LONG_NATIVE
    from factor_engine.backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    for name, _ in ELEMENTWISE_CASES:
        assert name in SQL_IMPLEMENTED_CANONICALS, f"{name} missing from SQL_IMPLEMENTED"
        assert name in POLARS_LONG_NATIVE, f"{name} missing from POLARS_LONG_NATIVE"


def test_fin_component_score_stays_out_of_sql_implemented():
    """fin_component_score is multi-panel (each component its own panel); the
    long-table SQL model has a single _v column, so it must NOT be SQL-implemented."""
    from factor_engine.backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    assert "fin_component_score" not in SQL_IMPLEMENTED_CANONICALS
