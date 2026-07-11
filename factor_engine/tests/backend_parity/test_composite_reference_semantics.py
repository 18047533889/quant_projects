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
        }
    )


def test_composite_reference_cases_cover_all_registered_lowerings():
    registered = list_composite_lowerings()
    covered = {c.canon for c in COMPOSITE_REFERENCE_CASES}
    missing = sorted(registered - covered)
    extra = sorted(covered - registered)
    assert not missing, f"reference cases 缺少: {missing}"
    assert not extra, f"reference cases 多余: {extra}"
    assert len(covered) >= 17


@pytest.mark.parametrize("case", COMPOSITE_REFERENCE_CASES, ids=lambda c: c.canon)
def test_composite_reference_matches_lowered_pandas(ref_source, case):
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
