"""双市场 golden 回归：A 股 / 美股面板公式输出基准。"""

from __future__ import annotations

import pandas as pd

from factor_engine.api import rank, ts_mean
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from tests.fixtures.golden.market_panels import (
    build_ashare_golden_close_panel,
    build_us_golden_close_panel,
    expected_rank,
)
from tests.helpers import InMemorySeriesSource


def test_ashare_golden_rank_matches_expected():
    close = build_ashare_golden_close_panel()
    factor = Factor(name="ashare_rank_close", expr=rank(col("close")))
    out = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close}),
    ).run(factor)["result"]
    expected = expected_rank(close)
    pd.testing.assert_series_equal(
        out,
        expected,
        check_names=False,
    )


def test_us_golden_rank_matches_expected():
    close = build_us_golden_close_panel()
    factor = Factor(name="us_rank_close", expr=rank(col("close")))
    out = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close}),
    ).run(factor)["result"]
    expected = expected_rank(close)
    pd.testing.assert_series_equal(
        out,
        expected,
        check_names=False,
    )


def test_dual_market_ts_mean_consistency():
    """双市场 ts_mean(2) 在各自面板上与 pandas rolling 一致。"""
    for close in (build_ashare_golden_close_panel(), build_us_golden_close_panel()):
        factor = Factor(name="mean2", expr=ts_mean(col("close"), 2))
        out = FactorEngine(
            backend=PandasBackend(),
            data_source=InMemorySeriesSource(data={"close": close}),
        ).run(factor)["result"]
        panel = close.unstack("instrument")
        expected = panel.rolling(2, min_periods=1).mean().stack(future_stack=True)
        pd.testing.assert_series_equal(
            out,
            expected,
            check_names=False,
        )
