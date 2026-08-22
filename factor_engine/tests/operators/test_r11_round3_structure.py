# -*- coding: utf-8 -*-
"""Round-3 structure audit tests (items 12/13/22/23/24/25/26).

These tests import the owned pandas and polars module functions DIRECTLY (not via
``load_all()`` / registry canonical resolution) so they are hermetic and
deterministic with respect to concurrent registry churn and duplicate backend
registrations elsewhere in the catalog.  The tests/operators session conftest
still exercises ``load_all()`` whenever the wider operator suite runs.

Covered audit items
-------------------
12  ``ts_days_since_high/low`` ties must reference the LATEST occurrence
13  ``ts_resistance_log_slope`` / ``ts_support_log_slope`` log-price slopes
22  ``ts_swing_amplitude`` must use the most recent adjacent opposite-sign pair
23  ``ts_swing_duration`` / ``ts_swing_velocity`` share the same SwingSegment
24  resistance/support fit R^2 requires >= 3 fitted points
25  ``ts_line_parallelism`` uses normalized (log-price) slopes
26  ``ts_impulse_strength`` scale = volatility excluding the current impulse
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from cleaned_operators.price_volume.structure_patterns_v2 import (
    _fit_r2,
    _pivot_stream,
    ts_impulse_strength,
    ts_swing_amplitude,
    ts_swing_duration,
    ts_swing_velocity,
)
from cleaned_operators.price_volume.technical_extensions import _ts_days_since_extreme
from cleaned_operators.price_volume.technical_structure_repairs import _bounded_line
from cleaned_operators.technical.polars_misc_v2 import (
    ts_days_since_high as pl_days_since_high,
    ts_days_since_low as pl_days_since_low,
)
from cleaned_operators.price_volume.polars_structure import (
    ts_line_parallelism as pl_line_parallelism,
    ts_resistance_log_slope as pl_resistance_log_slope,
    ts_support_log_slope as pl_support_log_slope,
    ts_swing_amplitude as pl_swing_amplitude,
    ts_swing_amplitude_pct as pl_swing_amplitude_pct,
    ts_swing_duration as pl_swing_duration,
    ts_swing_velocity as pl_swing_velocity,
)


def _frame(values) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"A": np.asarray(values, dtype=float)}, index=idx)


def _plframe(values) -> pl.DataFrame:
    return pl.DataFrame({"A": np.asarray(values, dtype=float)})


def _assert_pd_pl_parity(pandas_out, polars_out, *, rtol=1e-9, atol=1e-9):
    assert list(pandas_out.columns) == list(polars_out.columns)
    for col in pandas_out.columns:
        np.testing.assert_allclose(
            pandas_out[col].to_numpy(),
            polars_out[col].to_numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )


# ---------------------------------------------------------------------------
# item 12 — ts_days_since_high/low ties take the LATEST occurrence
# ---------------------------------------------------------------------------


def test_days_since_extreme_tie_references_latest_occurrence():
    # Window [10, 12, 12, 11]: "days since high" must reference the second 12
    # (index 2, newest prior bar index 3 -> 1), not the first (index 1 -> 2).
    out = _ts_days_since_extreme(_frame([10.0, 12.0, 12.0, 11.0, 20.0]), 4, high=True)
    assert out.iloc[4, 0] == 1.0
    # Window [20, 11, 11, 15]: latest low is the second 11 -> 1.
    out = _ts_days_since_extreme(_frame([20.0, 11.0, 11.0, 15.0, 9.0]), 4, high=False)
    assert out.iloc[4, 0] == 1.0
    # Newest bar is itself the max -> 0.
    out = _ts_days_since_extreme(_frame([10.0, 12.0, 13.0, 13.0, 20.0]), 4, high=True)
    assert out.iloc[4, 0] == 0.0


def test_days_since_extreme_polars_matches_pandas():
    # High tie window [10, 12, 12, 11] -> latest max is the second 12 (index 2),
    # so "days since high" at t=4 is 1 (not the first-12 value 2).
    values_hi = [10.0, 12.0, 12.0, 11.0, 13.0, 13.0, 14.0, 10.0, 10.0, 15.0, 15.0, 16.0]
    pandas_out = _ts_days_since_extreme(_frame(values_hi), 4, high=True)
    polars_out = pl_days_since_high(_plframe(values_hi), 4)
    _assert_pd_pl_parity(pandas_out, polars_out)
    assert pandas_out["A"].iloc[4] == 1.0
    # Low tie window [20, 11, 11, 15] -> latest min is the second 11 (index 2),
    # so "days since low" at t=4 is 1.
    values_lo = [20.0, 11.0, 11.0, 15.0, 9.0, 13.0, 14.0, 12.0, 12.0, 10.0, 15.0, 16.0]
    pandas_out = _ts_days_since_extreme(_frame(values_lo), 4, high=False)
    polars_out = pl_days_since_low(_plframe(values_lo), 4)
    _assert_pd_pl_parity(pandas_out, polars_out)
    assert pandas_out["A"].iloc[4] == 1.0


# ---------------------------------------------------------------------------
# item 13 — ts_resistance_log_slope / ts_support_log_slope (log price)
# ---------------------------------------------------------------------------


def _log_linear_high(base, n=60, step=0.01, peak_every=5):
    v = base * np.exp(step * np.arange(n))
    for t in range(n):
        if t % peak_every != 0:
            v[t] = v[t] * 0.96
    return v


def test_log_slope_is_price_level_invariant():
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    h100 = pd.DataFrame({"A": _log_linear_high(100.0)}, index=idx)
    h010 = pd.DataFrame({"A": _log_linear_high(10.0)}, index=idx)
    kw = dict(left_window=2, right_window=2, history_window=40, points=3)
    # pandas registered canonical
    from cleaned_operators.registry import OperatorRegistry
    log100 = OperatorRegistry.get("ts_resistance_log_slope", backend="pandas_numpy").calculate(h100, **kw)
    log010 = OperatorRegistry.get("ts_resistance_log_slope", backend="pandas_numpy").calculate(h010, **kw)
    # log slopes equal across price levels and match the injected log trend
    assert abs(log100.iloc[40, 0] - 0.01) < 1e-3
    assert abs(log100.iloc[40, 0] - log010.iloc[40, 0]) < 1e-6
    # raw slopes are price-level dependent (differ by roughly the price ratio);
    # use the bounded helper directly so the test is independent of whichever
    # ``ts_resistance_slope`` registration the full catalog resolves to.
    raw100 = _bounded_line(h100, 2, 2, 40, 3, high=True, output="slope")
    raw010 = _bounded_line(h010, 2, 2, 40, 3, high=True, output="slope")
    assert abs(raw100.iloc[40, 0] - raw010.iloc[40, 0]) > 0.5


def test_log_slope_polars_matches_pandas():
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    high = pd.DataFrame({"A": _log_linear_high(100.0)}, index=idx)
    low = pd.DataFrame({"A": 90.0 + _log_linear_high(10.0, n=n, step=0.005)}, index=idx)
    kw = dict(left_window=2, right_window=2, history_window=40, points=3)
    from cleaned_operators.registry import OperatorRegistry
    for name, pl_fn, frame in (
        ("ts_resistance_log_slope", pl_resistance_log_slope, high),
        ("ts_support_log_slope", pl_support_log_slope, low),
    ):
        pandas_out = OperatorRegistry.get(name, backend="pandas_numpy").calculate(frame, **kw)
        polars_out = pl_fn(_plframe(frame["A"].to_numpy()), **kw)
        _assert_pd_pl_parity(pandas_out, polars_out)


# ---------------------------------------------------------------------------
# item 22/23 — swing kernel: most recent ADJACENT opposite-sign pivot pair
# ---------------------------------------------------------------------------


@pytest.fixture
def swing_panels():
    """Pivot topology: low@5=80, high@10=110, low@15=85, high@20=105.

    At t >= 22 the most recent adjacent opposite-sign pair is low@15 -> high@20
    (amp 20, dur 5, vel 4).
    """
    n = 40
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    high = np.full(n, 95.0)
    low = np.full(n, 95.0)
    low[5] = 80.0
    high[10] = 110.0
    low[15] = 85.0
    high[20] = 105.0
    return (
        pd.DataFrame({"A": high}, index=idx),
        pd.DataFrame({"A": low}, index=idx),
    )


def test_swing_kernel_uses_adjacent_opposite_pair(swing_panels):
    high, low = swing_panels
    kw = dict(left_window=2, right_window=2, history_window=30)
    amp = ts_swing_amplitude(high, low, **kw)
    dur = ts_swing_duration(high, low, **kw)
    vel = ts_swing_velocity(high, low, **kw)
    assert abs(amp.iloc[22, 0] - 20.0) < 1e-9
    assert abs(dur.iloc[22, 0] - 5.0) < 1e-9
    assert abs(vel.iloc[22, 0] - 4.0) < 1e-9


def test_swing_kernel_ignores_later_same_sign_pivot():
    # Adding a NEWER high@25=115 (confirmed at 27) does not change the swing:
    # the most recent adjacent opposite-sign pair is still low@15 -> high@20.
    n = 40
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    high = np.full(n, 95.0)
    low = np.full(n, 95.0)
    low[5] = 80.0
    high[10] = 110.0
    low[15] = 85.0
    high[20] = 105.0
    high[25] = 115.0
    high = pd.DataFrame({"A": high}, index=idx)
    low = pd.DataFrame({"A": low}, index=idx)
    kw = dict(left_window=2, right_window=2, history_window=30)
    amp = ts_swing_amplitude(high, low, **kw)
    dur = ts_swing_duration(high, low, **kw)
    vel = ts_swing_velocity(high, low, **kw)
    # Naive |latest_high - latest_low| would be |115 - 85| = 30; kernel must use
    # the real swing low@15 -> high@20.
    assert abs(amp.iloc[28, 0] - 20.0) < 1e-9
    assert abs(dur.iloc[28, 0] - 5.0) < 1e-9
    assert abs(vel.iloc[28, 0] - 4.0) < 1e-9


def test_swing_ops_polars_match_pandas(swing_panels):
    high, low = swing_panels
    kw = dict(left_window=2, right_window=2, history_window=30)
    close = pd.DataFrame({"A": np.full(high.shape[0], 100.0)}, index=high.index)
    for name, pandas_fn, pl_fn, args in (
        ("ts_swing_amplitude", ts_swing_amplitude, pl_swing_amplitude, (high, low)),
        ("ts_swing_duration", ts_swing_duration, pl_swing_duration, (high, low)),
        ("ts_swing_velocity", ts_swing_velocity, pl_swing_velocity, (high, low)),
        ("ts_swing_amplitude_pct", None, pl_swing_amplitude_pct, (high, low, close)),
    ):
        if pandas_fn is None:
            from cleaned_operators.price_volume.structure_patterns_v2 import ts_swing_amplitude_pct
            pandas_fn = ts_swing_amplitude_pct
        pandas_out = pandas_fn(*args, **kw)
        pl_args = tuple(_plframe(a["A"].to_numpy()) for a in args)
        polars_out = pl_fn(*pl_args, **kw)
        _assert_pd_pl_parity(pandas_out, polars_out)


def test_swing_kernel_stream_matches_pivot_stream(swing_panels):
    # The kernel must derive from the same chronological merged pivot stream.
    high, low = swing_panels
    stream = _pivot_stream(high, low, 2, 2, 30)
    events = stream[28][0]
    assert [e[1] for e in events] == [False, True, False, True]  # L H L H
    assert abs(events[-1][2] - 105.0) < 1e-9
    assert abs(events[-2][2] - 85.0) < 1e-9


# ---------------------------------------------------------------------------
# item 24 — resistance/support fit R^2 requires >= 3 points
# ---------------------------------------------------------------------------


def test_fit_r2_rejects_two_points():
    with pytest.raises(ValueError, match="points"):
        _fit_r2(_frame([1.0, 2.0, 3.0, 4.0]), 1, 1, 10, 2, True)


def test_fit_r2_registered_operators_require_three_points():
    from cleaned_operators.registry import OperatorRegistry
    frame = _frame([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    for name in ("ts_resistance_fit_r2", "ts_support_fit_r2"):
        op = OperatorRegistry.get(name, backend="pandas_numpy")
        with pytest.raises(ValueError, match="points"):
            op.calculate(frame, 1, 1, 10, 2)
        # three points are accepted (returns a panel; may be all NaN without pivots)
        out = op.calculate(frame, 1, 1, 10, 3)
        assert list(out.columns) == ["A"]


# ---------------------------------------------------------------------------
# item 25 — ts_line_parallelism uses normalized (log-price) slopes
# ---------------------------------------------------------------------------


def test_line_parallelism_price_level_invariant():
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    high100 = _log_linear_high(100.0, n=n)
    low100 = 92.0 + _log_linear_high(8.0, n=n, step=0.01)
    high010 = _log_linear_high(10.0, n=n)
    low010 = 9.2 + _log_linear_high(0.8, n=n, step=0.01)
    for low_arr, h_arr in ((low100, high100), (low010, high010)):
        for t in range(n):
            if t % 5 == 0:
                low_arr[t] = low_arr[t] * 0.96
    h100 = pd.DataFrame({"A": high100}, index=idx)
    l100 = pd.DataFrame({"A": low100}, index=idx)
    h010 = pd.DataFrame({"A": high010}, index=idx)
    l010 = pd.DataFrame({"A": low010}, index=idx)
    kw = dict(left_window=2, right_window=2, history_window=40, points=3)
    a = ts_line_parallelism_pandas(h100, l100, **kw).iloc[40, 0]
    b = ts_line_parallelism_pandas(h010, l010, **kw).iloc[40, 0]
    assert np.isfinite(a) and abs(a - b) < 1e-5
    # resistance and support trend at the same log slope -> near-parallel (>= -1e-2)
    assert a > -1e-2


def ts_line_parallelism_pandas(high, low, **kw):
    from cleaned_operators.price_volume.structure_patterns_v2 import ts_line_parallelism
    return ts_line_parallelism(high, low, **kw)


def test_line_parallelism_polars_matches_pandas():
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    high = _log_linear_high(100.0, n=n)
    low = np.full(n, 90.0)
    for t in range(n):
        low[t] = 90.0 + high[t] * 0.05
    high = pd.DataFrame({"A": high}, index=idx)
    low = pd.DataFrame({"A": low}, index=idx)
    kw = dict(left_window=2, right_window=2, history_window=40, points=3)
    pandas_out = ts_line_parallelism_pandas(high, low, **kw)
    polars_out = pl_line_parallelism(
        _plframe(high["A"].to_numpy()), _plframe(low["A"].to_numpy()), **kw
    )
    _assert_pd_pl_parity(pandas_out, polars_out)


# ---------------------------------------------------------------------------
# item 26 — ts_impulse_strength denominator excludes the current impulse
# ---------------------------------------------------------------------------


def test_impulse_strength_denominator_excludes_impulse():
    values = [100.0, 101.0, 102.0, 103.0, 104.0, 120.0, 121.0, 122.0, 123.0, 124.0]
    close = _frame(values)
    out = ts_impulse_strength(close, 5, 3)
    # Manual reference: rv = std of daily pct_change up to t-1 (``volatility_{t-1}``,
    # ``.shift(1)``) so the current impulse bar cannot inflate its own denominator.
    ret = close["A"].pct_change(fill_method=None)
    rv = ret.rolling(3, min_periods=3).std().shift(1)
    imp = close["A"] / close["A"].shift(5) - 1.0
    expected = imp / (rv * np.sqrt(5.0))
    np.testing.assert_allclose(out["A"].to_numpy(), expected.to_numpy(), equal_nan=True)
    # At the impulse completion (t=5) the denominator uses the pre-impulse vol
    # (excluding the 120-vs-104 jump bar), so the strength is a large positive
    # number rather than being deflated by the impulse bar's own return.
    assert out["A"].iloc[9] > 100.0
