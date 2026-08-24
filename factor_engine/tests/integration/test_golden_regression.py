"""Golden panel 与截面隔离回归测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from factor_engine.api import rank, ts_delay, ts_mean
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from tests.fixtures.golden_panel import (
    build_golden_close_panel,
    expected_ts_delay,
    expected_ts_mean,
)
from tests.helpers import InMemorySeriesSource


def test_golden_ts_mean_matches_hand_calculation():
    close = build_golden_close_panel()
    factor = Factor(name="mean2", expr=ts_mean(col("close"), 2))
    out = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close}),
    ).run(factor)["result"]
    expected = expected_ts_mean(close, 2)
    overlap = out.index.intersection(expected.index)
    pd.testing.assert_series_equal(
        out.loc[overlap],
        expected.loc[overlap],
        check_names=False,
    )


def test_golden_ts_delay_matches_hand_calculation():
    close = build_golden_close_panel()
    factor = Factor(name="delay1", expr=ts_delay(col("close"), 1))
    out = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close}),
    ).run(factor)["result"]
    expected = expected_ts_delay(close, 1)
    overlap = out.index.intersection(expected.index)
    pd.testing.assert_series_equal(
        out.loc[overlap],
        expected.loc[overlap],
        check_names=False,
    )


def test_cross_section_rank_invariant_to_future_truncation():
    """截断未来日期后，历史日截面 rank 不变。"""
    close = build_golden_close_panel()
    factor = Factor(name="rank_close", expr=rank(col("close")))
    engine = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close}),
    )
    full = engine.run(factor)["result"]
    cutoff = pd.Timestamp("2024-01-04")
    truncated_close = close[close.index.get_level_values(0) <= cutoff]
    trunc = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": truncated_close}),
    ).run(factor)["result"]

    hist_idx = trunc.index
    pd.testing.assert_series_equal(
        full.loc[hist_idx],
        trunc.loc[hist_idx],
        check_names=False,
    )
