# -*- coding: utf-8 -*-
"""Parity tests for native Polars PIT-safe fundamental period operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _load_registry():
    load_all()


def _fiscal_panel(n_days: int = 240, seed: int = 11):
    """Daily panel whose period_id switches quarterly, mimicking PIT filings."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-01", periods=n_days, freq="D")
    cols = ["A", "B"]
    x = pd.DataFrame(np.nan, index=index, columns=cols, dtype=float)
    period_id = pd.DataFrame(np.nan, index=index, columns=cols, dtype=object)
    fiscal_q = pd.DataFrame(np.nan, index=index, columns=cols, dtype=object)

    # anchor to quarter boundaries
    for col in cols:
        raw = rng.normal(100.0, 5.0, 20)
        for k in range(20):
            # report period k visible starting at a staggered offset
            start = min(n_days - 1, k * 12 + (k % 4))
            end = min(n_days, (k + 1) * 12 + (k % 4))
            if start >= n_days:
                break
            year = 2022 + k // 4
            qtr = (k % 4) + 1
            period_id.iloc[start:end, x.columns.get_loc(col)] = f"{year}Q{qtr}"
            fiscal_q.iloc[start:end, x.columns.get_loc(col)] = qtr
            x.iloc[start:end, x.columns.get_loc(col)] = raw[k]
    # add some intra-period revision noise on x (same period, new value)
    for col in cols:
        for _ in range(5):
            t = int(rng.integers(5, n_days - 1))
            x.iloc[t, x.columns.get_loc(col)] += float(rng.normal(0.3, 0.1))
    return x, period_id, fiscal_q


@pytest.fixture(scope="module")
def fiscal():
    x, period_id, fiscal_q = _fiscal_panel()
    rng = np.random.default_rng(21)
    index = x.index
    cols = x.columns
    expected = x + rng.normal(0, 0.5, x.shape)
    scale = pd.DataFrame(np.full(x.shape, 10.0), index=index, columns=cols)
    # a second panel for two-input ops
    y = x * 1.1 + 3.0
    flow = pd.DataFrame(rng.uniform(5, 50, x.shape), index=index, columns=cols)
    balance = pd.DataFrame(rng.uniform(100, 500, x.shape), index=index, columns=cols)
    assets = pd.DataFrame(rng.uniform(500, 5000, x.shape), index=index, columns=cols)
    return x, period_id, fiscal_q, expected, scale, y, flow, balance, assets


def _polars(frame: pd.DataFrame) -> pl.DataFrame:
    data = {}
    for c in frame.columns:
        if frame[c].dtype == object:
            values = frame[c].tolist()
            cleaned = [
                None
                if value is None or (isinstance(value, float) and np.isnan(value))
                else value
                for value in values
            ]
            data[c] = pl.Series(cleaned, dtype=pl.Object) if any(not isinstance(v, str) for v in cleaned) else cleaned
        else:
            data[c] = frame[c].to_numpy()
    return pl.DataFrame(data)


def _assert_parity(name, args, kwargs, rtol=1e-8, atol=1e-8):
    pandas_op = OperatorRegistry.get(name, backend="pandas_numpy")
    polars_op = OperatorRegistry.get(name, backend="polars")
    assert pandas_op is not None, f"{name} missing pandas"
    assert polars_op is not None, f"{name} missing polars"
    pandas_out = pandas_op.calculate(*args, **kwargs)
    polars_out = polars_op.calculate(*[_polars(arg) for arg in args], **kwargs)
    assert list(pandas_out.columns) == list(polars_out.columns)
    for column in pandas_out.columns:
        np.testing.assert_allclose(
            pandas_out[column].to_numpy(),
            polars_out[column].to_numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )


@pytest.mark.parametrize(
    ("name", "inputs", "kwargs"),
    [
        ("fin_lag", (0, 1), {"periods": 1}),
        ("fin_lag", (0, 1), {"periods": 4}),
        ("fin_diff", (0, 1), {"periods": 1}),
        ("fin_pct_change", (0, 1), {"periods": 1}),
        ("fin_log_change", (0, 1), {"periods": 1}),
        ("fin_qoq", (0, 1), {}),
        ("fin_yoy", (0, 1), {"periods_per_year": 4}),
        ("fin_ttm", (0, 1), {"periods_per_year": 4}),
        ("fin_average_balance", (0, 1), {"periods": 2}),
        ("fin_growth", (0, 1), {"periods": 1}),
        ("fin_cagr", (0, 1), {"periods": 4, "periods_per_year": 4}),
        ("fin_growth_acceleration", (0, 1), {"short_periods": 1, "long_periods": 4}),
        ("fin_growth_change", (0, 1), {"growth_periods": 4, "compare_periods": 1}),
        ("fin_std", (0, 1), {"periods": 8}),
        ("fin_mad", (0, 1), {"periods": 8}),
        ("fin_cv", (0, 1), {"periods": 8}),
        ("fin_stability", (0, 1), {"periods": 8}),
        ("fin_range", (0, 1), {"periods": 8}),
        ("fin_zscore_history", (0, 1), {"periods": 8}),
        ("fin_percentile_history", (0, 1), {"periods": 8}),
        ("fin_trend_slope", (0, 1), {"periods": 8}),
        ("fin_trend_r2", (0, 1), {"periods": 8}),
        ("fin_trend_tstat", (0, 1), {"periods": 8}),
        ("fin_trend_acceleration", (0, 1), {"short_periods": 4, "long_periods": 8}),
        ("fin_monotonicity", (0, 1), {"periods": 8}),
        ("fin_positive_streak", (0, 1), {"max_periods": 8}),
        ("fin_negative_streak", (0, 1), {"max_periods": 8}),
        ("fin_sign_change_count", (0, 1), {"periods": 8}),
        ("fin_growth_volatility", (0, 1), {"growth_periods": 1, "window_periods": 8}),
        ("fin_growth_stability", (0, 1), {"growth_periods": 1, "window_periods": 8}),
        ("fin_growth_persistence", (0, 1), {"growth_periods": 1, "window_periods": 8}),
        ("fin_ratio", (0, 8), {}),
        ("fin_common_size", (0, 8), {}),
        ("fin_turnover", (6, 7, 1), {"average_periods": 2}),
        ("fin_divergence", (0, 5, 1), {"periods": 4}),
        ("fin_cash_earnings_gap", (0, 6, 4), {}),
        ("fin_accrual_ratio", (0, 6, 8), {}),
        ("fin_cash_conversion", (6, 0), {}),
        ("fin_working_capital_change", (0, 1), {"periods": 1}),
        ("fin_surprise", (0, 3, 4), {}),
        ("fin_surprise_zscore", (0, 3, 4), {"window_days": 60}),
        ("fin_surprise_event_zscore", (0, 3, 4, 1), {"periods": 8}),
        ("fin_surprise_event_percentile", (0, 3, 4, 1), {"periods": 8}),
        ("fin_expectation_revision", (3, 1), {}),
        ("fin_expectation_revision_pct", (3, 1), {}),
        ("fin_expectation_revision_speed", (3, 1), {"window_days": 60}),
        ("fin_expectation_revision_count", (3, 1), {"window_days": 60}),
        ("fin_expectation_revision_magnitude", (3, 1), {"window_days": 60}),
        ("fin_days_since_expectation_revision", (3, 1), {"max_days": 252}),
        ("fin_expectation_dispersion", (4, 4), {}),
        ("fin_actual_expectation_divergence", (0, 3, 4), {}),
        ("fin_beat_streak", (0, 3, 1), {"max_periods": 8}),
        ("fin_miss_streak", (0, 3, 1), {"max_periods": 8}),
        ("fin_revision_delta", (0, 1), {}),
        ("fin_revision_pct", (0, 1), {}),
        ("fin_revision_direction", (0, 1), {}),
        ("fin_revision_count", (0, 1), {"window_days": 60}),
        ("fin_revision_magnitude", (0, 1), {"window_days": 60}),
        ("fin_restated_flag", (0, 1), {"window_days": 60}),
        ("fin_days_since_update", (0, 1), {"max_days": 100}),
        ("fin_staleness", (0, 1), {"max_days": 100}),
        ("fin_ttm_quarterly", (0, 1), {"periods_per_year": 4}),
        ("fin_quarter_from_cumulative", (0, 1, 2), {}),
        ("fin_ttm_cumulative", (0, 1, 2), {"periods_per_year": 4}),
        ("fin_seasonal_zscore", (0, 1, 2), {"years": 4, "min_history": 2}),
        ("fin_seasonal_percentile", (0, 1, 2), {"years": 4, "min_history": 2}),
    ],
)
def test_native_polars_fundamental_matches_pandas(fiscal, name, inputs, kwargs):
    _assert_parity(name, tuple(fiscal[index] for index in inputs), kwargs)
