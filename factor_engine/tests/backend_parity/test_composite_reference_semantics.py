# -*- coding: utf-8
"""Composite reference parity：原始 Pandas operator vs lowered-plan Pandas。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.context import ExecutionContext
from cleaned_operators import load_all
from planner.composite_lowering import list_composite_lowerings
from tests.backend_parity.composite_reference_helpers import (
    COMPOSITE_REFERENCE_CASES,
    build_reference_panels,
    execute_lowered_pandas,
    lowered_plan_for,
    panel_to_series,
    reference_pandas_calculate,
)
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module")
def ref_source():
    load_all()
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
            (pd.Timestamp("2024-01-05"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-03"), "B"),
            (pd.Timestamp("2024-01-04"), "B"),
            (pd.Timestamp("2024-01-05"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0, 10.5, 12.0, 20.0, 19.0, 21.0, 22.0], index=idx)
    high = close + 0.5
    low = close - 0.5
    volume = pd.Series([100.0, 110.0, 0.0, 120.0, 200.0, 210.0, 190.0, 220.0], index=idx)
    return InMemorySeriesSource(
        data={
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "price": close,
            "operating_income": pd.Series([100.0, 110.0, 0.0, 120.0, 200.0, 180.0, 210.0, 220.0], index=idx),
            "revenue": pd.Series([200.0, 220.0, 110.0, 240.0, 400.0, 0.0, 420.0, 440.0], index=idx),
            "current_assets": pd.Series([50.0, 55.0, 52.0, 60.0, 80.0, 78.0, 82.0, 85.0], index=idx),
            "current_liabilities": pd.Series([25.0, 0.0, 26.0, 30.0, 40.0, 41.0, 0.0, 42.0], index=idx),
            "inventory": pd.Series([5.0, 5.5, 5.2, 6.0, 8.0, 7.8, 8.2, 8.5], index=idx),
            "debt": pd.Series([30.0, 31.0, 32.0, 33.0, 60.0, 61.0, 62.0, 63.0], index=idx),
            "equity": pd.Series([70.0, 0.0, 72.0, 73.0, 140.0, 141.0, 142.0, 143.0], index=idx),
            "total_debt": pd.Series([30.0, 31.0, 32.0, 33.0, 60.0, 61.0, 62.0, 63.0], index=idx),
            "total_equity": pd.Series([70.0, 0.0, 72.0, 73.0, 140.0, 141.0, 142.0, 143.0], index=idx),
            "float_shares": pd.Series([1e6, 1e6, 0.0, 1e6, 2e6, 2e6, 2e6, 2e6], index=idx),
            # A 股 composite reference data
            "one": pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], index=idx),
            "pe": pd.Series([20.0, 22.0, 18.0, 25.0, 30.0, 28.0, 32.0, 35.0], index=idx),
            "pb": pd.Series([2.0, 2.2, 1.8, 2.5, 3.0, 2.8, 3.2, 3.5], index=idx),
            "total_shares": pd.Series([5e6, 5e6, 5e6, 5e6, 1e7, 1e7, 1e7, 1e7], index=idx),
            "free_float_shares": pd.Series([3e6, 3e6, 3e6, 3e6, 6e6, 6e6, 6e6, 6e6], index=idx),
            "upper_limit": pd.Series([11.0, 12.1, 11.55, 13.2, 22.0, 20.9, 23.1, 24.2], index=idx),
            "lower_limit": pd.Series([9.0, 9.9, 9.45, 10.8, 18.0, 17.1, 18.9, 19.8], index=idx),
            "ret": pd.Series([0.01, 0.02, -0.01, 0.03, 0.02, -0.02, 0.05, 0.02], index=idx),
            "benchmark_ret": pd.Series([0.005, 0.01, -0.005, 0.02, 0.005, -0.01, 0.03, 0.01], index=idx),
            "benchmark_price": pd.Series([100.0, 101.0, 100.5, 102.0, 100.0, 101.0, 100.5, 102.0], index=idx),
            "top_holder_shares": pd.Series([1e6, 1e6, 1e6, 1e6, 2e6, 2e6, 2e6, 2e6], index=idx),
            # Next-stage composite reference data (2026-08).
            "a_cap": pd.Series([1e6, 1.1e6, 0.9e6, 1.2e6, 2e6, 1.9e6, 2.1e6, 2.2e6], index=idx),
            "capitalization": pd.Series([2e6, 2.2e6, 1.8e6, 2.5e6, 4e6, 3.8e6, 4.2e6, 4.4e6], index=idx),
            "circulating_cap": pd.Series([1.5e6, 1.6e6, 1.4e6, 1.7e6, 3e6, 2.9e6, 3.1e6, 3.2e6], index=idx),
            "free_cap": pd.Series([0.9e6, 1.0e6, 0.8e6, 1.1e6, 1.8e6, 1.7e6, 1.9e6, 2.0e6], index=idx),
            "free_market_cap": pd.Series([1.8e7, 1.9e7, 1.7e7, 2.0e7, 3.6e7, 3.5e7, 3.7e7, 3.8e7], index=idx),
            "market_cap": pd.Series([4e7, 4.2e7, 3.8e7, 4.5e7, 8e7, 7.8e7, 8.2e7, 8.5e7], index=idx),
            "total_capital": pd.Series([5e6, 5e6, 5e6, 5e6, 1e7, 1e7, 1e7, 1e7], index=idx),
            "total_assets": pd.Series([200.0, 210.0, 190.0, 220.0, 400.0, 390.0, 410.0, 420.0], index=idx),
            "avg_assets": pd.Series([180.0, 190.0, 175.0, 200.0, 360.0, 350.0, 370.0, 380.0], index=idx),
            "avg_equity": pd.Series([65.0, 70.0, 60.0, 72.0, 130.0, 128.0, 132.0, 135.0], index=idx),
            "net_profit": pd.Series([15.0, 16.0, 14.0, 18.0, 30.0, 28.0, 32.0, 33.0], index=idx),
            "ocf": pd.Series([20.0, 0.0, 18.0, 22.0, 40.0, 38.0, 0.0, 42.0], index=idx),
            "operating_profit": pd.Series([18.0, 19.0, 17.0, 21.0, 36.0, 34.0, 38.0, 39.0], index=idx),
            "operating_revenue": pd.Series([200.0, 220.0, 190.0, 240.0, 400.0, 380.0, 420.0, 440.0], index=idx),
            "net_cash_from_subcompany": pd.Series([5.0, 6.0, 4.0, 7.0, 10.0, 9.0, 11.0, 12.0], index=idx),
            "cash_from_borrowing": pd.Series([8.0, 9.0, 7.0, 10.0, 16.0, 15.0, 17.0, 18.0], index=idx),
            "cash_from_bonds_issue": pd.Series([3.0, 4.0, 2.0, 5.0, 6.0, 5.0, 7.0, 8.0], index=idx),
            "borrowing_repayment": pd.Series([6.0, 7.0, 5.0, 8.0, 12.0, 11.0, 13.0, 14.0], index=idx),
            "capex_cash": pd.Series([4.0, 5.0, 3.0, 6.0, 8.0, 7.0, 9.0, 10.0], index=idx),
            "contract_assets": pd.Series([10.0, 11.0, 9.0, 12.0, 20.0, 19.0, 21.0, 22.0], index=idx),
            "contract_liability": pd.Series([8.0, 9.0, 7.0, 10.0, 16.0, 15.0, 17.0, 18.0], index=idx),
            "deferred_tax_assets": pd.Series([3.0, 4.0, 2.0, 5.0, 6.0, 5.0, 7.0, 8.0], index=idx),
            "deferred_tax_liability": pd.Series([2.0, 3.0, 1.0, 4.0, 4.0, 3.0, 5.0, 6.0], index=idx),
            "discontinued_operation_profit": pd.Series([1.0, 2.0, 0.0, 3.0, 2.0, 1.0, 3.0, 4.0], index=idx),
            "fair_value_income": pd.Series([2.0, 3.0, 1.0, 4.0, 4.0, 3.0, 5.0, 6.0], index=idx),
            "goodwill": pd.Series([12.0, 13.0, 11.0, 14.0, 24.0, 23.0, 25.0, 26.0], index=idx),
            "asset_impairment_loss": pd.Series([1.0, 2.0, 0.0, 3.0, 2.0, 1.0, 3.0, 4.0], index=idx),
            "credit_impairment_loss": pd.Series([1.0, 2.0, 0.0, 3.0, 2.0, 1.0, 3.0, 4.0], index=idx),
            "interest_cost": pd.Series([2.0, 3.0, 1.0, 4.0, 4.0, 3.0, 5.0, 6.0], index=idx),
            "investment_income": pd.Series([3.0, 4.0, 2.0, 5.0, 6.0, 5.0, 7.0, 8.0], index=idx),
            "lease_liability": pd.Series([5.0, 6.0, 4.0, 7.0, 10.0, 9.0, 11.0, 12.0], index=idx),
            "usufruct_assets": pd.Series([4.0, 5.0, 3.0, 6.0, 8.0, 7.0, 9.0, 10.0], index=idx),
            "minority_profit": pd.Series([2.0, 3.0, 1.0, 4.0, 4.0, 3.0, 5.0, 6.0], index=idx),
            "other_comprehensive_income": pd.Series([1.0, 2.0, 0.0, 3.0, 2.0, 1.0, 3.0, 4.0], index=idx),
            "other_earnings": pd.Series([1.0, 2.0, 0.0, 3.0, 2.0, 1.0, 3.0, 4.0], index=idx),
            "capitalized_dev_increase": pd.Series([2.0, 3.0, 1.0, 4.0, 4.0, 3.0, 5.0, 6.0], index=idx),
            "rd_expense": pd.Series([6.0, 7.0, 5.0, 8.0, 12.0, 11.0, 13.0, 14.0], index=idx),
            "total_profit": pd.Series([20.0, 22.0, 18.0, 25.0, 40.0, 38.0, 42.0, 44.0], index=idx),
            "pe_ratio": pd.Series([20.0, 22.0, 18.0, 25.0, 30.0, 28.0, 32.0, 35.0], index=idx),
            "pe_ratio_lyr": pd.Series([19.0, 21.0, 17.0, 24.0, 29.0, 27.0, 31.0, 34.0], index=idx),
            "pcf_ratio": pd.Series([10.0, 11.0, 9.0, 12.0, 15.0, 14.0, 16.0, 17.0], index=idx),
            "pcf_ratio2": pd.Series([9.0, 10.0, 8.0, 11.0, 14.0, 13.0, 15.0, 16.0], index=idx),
            "index_weight": pd.Series([0.05, 0.06, 0.04, 0.07, 0.08, 0.07, 0.09, 0.10], index=idx),
            "free_float_weight": pd.Series([0.04, 0.05, 0.03, 0.06, 0.07, 0.06, 0.08, 0.09], index=idx),
            "freeze_shares": pd.Series([1e5, 1e5, 1e5, 1e5, 2e5, 2e5, 2e5, 2e5], index=idx),
            "locked_shares": pd.Series([2e5, 2e5, 2e5, 2e5, 4e5, 4e5, 4e5, 4e5], index=idx),
            "pledge_shares": pd.Series([1.5e5, 1.5e5, 1.5e5, 1.5e5, 3e5, 3e5, 3e5, 3e5], index=idx),
            "degree": pd.Series([3.0, 4.0, 2.0, 5.0, 6.0, 5.0, 7.0, 8.0], index=idx),
            "total": pd.Series([10.0, 11.0, 9.0, 12.0, 20.0, 19.0, 21.0, 22.0], index=idx),
            "shared_holders": pd.Series([4.0, 5.0, 3.0, 6.0, 8.0, 7.0, 9.0, 10.0], index=idx),
            "total_holders": pd.Series([10.0, 11.0, 9.0, 12.0, 20.0, 19.0, 21.0, 22.0], index=idx),
            "top10_concentration": pd.Series([0.4, 0.42, 0.38, 0.45, 0.5, 0.48, 0.52, 0.53], index=idx),
            "top10_float_concentration": pd.Series([0.35, 0.37, 0.33, 0.40, 0.45, 0.43, 0.47, 0.48], index=idx),
            "period_id": pd.Series([4, 5, 6, 7, 4, 5, 6, 7], index=idx, dtype=float),
        }
    )


def test_composite_reference_cases_cover_all_registered_lowerings():
    registered = list_composite_lowerings()
    covered = {c.canon for c in COMPOSITE_REFERENCE_CASES}
    missing = sorted(registered - covered)
    extra = sorted(covered - registered)
    assert not missing, f"reference cases 缺少: {missing}"
    assert not extra, f"reference cases 多余: {extra}"
    assert len(covered) >= 27


@pytest.mark.parametrize("case", COMPOSITE_REFERENCE_CASES, ids=lambda c: c.canon)
def test_composite_reference_matches_lowered_pandas(ref_source, case):
    from cleaned_operators.registry import OperatorRegistry
    if case.canon not in OperatorRegistry._operators:
        pytest.skip("migrated to recipe or research layer")
    panels = build_reference_panels(ref_source, ref_source.data["close"].index)
    ref_panel = reference_pandas_calculate(case, panels)
    ref_series = panel_to_series(ref_panel, ref_source.data["close"].index)

    ctx = ExecutionContext(data_source=ref_source)
    lowered = lowered_plan_for(case)
    lowered_series = execute_lowered_pandas(lowered, ctx)

    pd.testing.assert_series_equal(
        ref_series,
        lowered_series,
        check_names=False,
        rtol=1e-5,
        atol=1e-5,
    )


def test_obv_first_row_zero_reference_and_lowered(ref_source):
    from tests.backend_parity.composite_reference_helpers import CompositeReferenceCase
    from cleaned_operators.registry import OperatorRegistry

    if "OBV" not in OperatorRegistry._operators:
        pytest.skip("OBV is a recipe, not a primitive composite")

    case = CompositeReferenceCase("OBV", ("close", "volume"))
    panels = build_reference_panels(ref_source, ref_source.data["close"].index)
    ref = panel_to_series(reference_pandas_calculate(case, panels), ref_source.data["close"].index)
    lowered = execute_lowered_pandas(
        lowered_plan_for(case),
        ExecutionContext(data_source=ref_source),
    )
    for inst in ("A", "B"):
        ts0 = pd.Timestamp("2024-01-02")
        assert ref.loc[(ts0, inst)] == 0.0
        assert lowered.loc[(ts0, inst)] == 0.0


def test_safe_div_null_ratio_zero_denominator(ref_source):
    from cleaned_operators.registry import OperatorRegistry
    if "operating_margin" not in OperatorRegistry._operators:
        pytest.skip("operating_margin is a recipe, not a primitive composite")
    case = next(c for c in COMPOSITE_REFERENCE_CASES if c.canon == "operating_margin")
    panels = build_reference_panels(ref_source, ref_source.data["close"].index)
    ref = reference_pandas_calculate(case, panels)
    lowered = execute_lowered_pandas(
        lowered_plan_for(case),
        ExecutionContext(data_source=ref_source),
    )
    ref_s = panel_to_series(ref, ref_source.data["close"].index)
    # revenue=0 行（B, 2024-01-03）应为 NaN
    key = (pd.Timestamp("2024-01-03"), "B")
    assert np.isnan(ref_s.loc[key])
    assert np.isnan(lowered.loc[key])
