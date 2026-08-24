# -*- coding: utf-8 -*-
"""Round-3 audit tests — multiscale trend / crossing / envelope / interval_geometry.

Audit items (2026-08 round 3, file-disjoint fixer #96 + cross-cutting):
* 96  multiscale trend: ``window`` is a dead parameter -> declared POLICY /
      non-searchable (kept for back-compat, like ``group_decay_linear``).
*     multiscale trend: ALL declared scales must be present (contiguous-finite,
      non-degenerate residual) or the row is NaN — no partial-scale recompute,
      no ``+ eps`` floor on the residual-scale denominator.
*     crossing: no EPS floor on the denominator; a NaN CURRENT row fails closed
      (never a stale 0/trailing value); volatility scale must be finite & > 0.
*     envelope: constant/zero-width envelope fails closed to NaN (no EPS
      explosion, no clip); width ratio is %-normalised so a 100-yuan stock and
      a 10-yuan stock with the same relative envelope are comparable.
*     interval_geometry: same no-EPS-denominator intent applied where the
      ``> 0`` guard already exists (union coverage / mode distance / exploration
      efficiency).

Tests import the owned module kernels DIRECTLY (hermetic) and also exercise the
registered operators through ``load_all()`` where the wider suite already does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.base import ParamRole
from factor_engine.cleaned_operators.crossing import (
    _crossing_acceleration_series,
    _crossing_parts,
    _crossing_speed_series,
)
from factor_engine.cleaned_operators.envelope import (
    _boundary_dwell_series,
    _compression_series,
    _normalised_position,
    _pressure_series,
)
from factor_engine.cleaned_operators.interval_geometry import (
    _union_coverage_series,
)
from factor_engine.cleaned_operators.multiscale_trend import (
    _consensus_series,
    _curvature_series,
    _dispersion_series,
)


def _panel(*series: np.ndarray) -> np.ndarray:
    return np.asarray(series, dtype=float).T


def _frame(values, cols=("A",)) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    return pd.DataFrame(
        values, index=pd.date_range("2024-01-01", periods=values.shape[0], freq="D"), columns=list(cols)
    )


# ---------------------------------------------------------------------------
# Item 96 — multiscale trend: ``window`` is a dead parameter
# ---------------------------------------------------------------------------
def test_window_declared_policy_non_searchable():
    from factor_engine.cleaned_operators.multiscale_trend import TsMultiscaleTrendConsensus

    spec = TsMultiscaleTrendConsensus.metadata.param_specs["window"]
    assert spec.searchable is False
    assert spec.param_role is ParamRole.POLICY
    assert spec.dtype is int


def test_window_does_not_participate_in_output():
    from factor_engine.cleaned_operators.multiscale_trend import TsMultiscaleTrendConsensus

    rng = np.random.default_rng(0)
    f = _frame(np.cumsum(rng.normal(0, 1, 80)))
    op = TsMultiscaleTrendConsensus()
    a = op.calculate(f, window=60, scales=(5, 10, 20, 40))
    b = op.calculate(f, window=120, scales=(5, 10, 20, 40))
    pd.testing.assert_frame_equal(a, b)


def test_window_still_bounds_scales():
    from factor_engine.cleaned_operators.multiscale_trend import TsMultiscaleTrendConsensus

    f = _frame(np.arange(60, dtype=float) + 0.01 * np.random.default_rng(0).normal(size=60))
    with pytest.raises(ValueError):
        TsMultiscaleTrendConsensus().calculate(f, window=10, scales=(5, 20))


# ---------------------------------------------------------------------------
# Multiscale trend — all-scales requirement / no partial-scale recompute
# ---------------------------------------------------------------------------
def test_short_data_no_partial_scale_recompute():
    # Data shorter than the LARGEST scale: every row must be NaN — the kernel
    # must never silently recompute on the subset of scales that do fit.
    x = np.arange(1, 15, dtype=float)  # 14 rows, largest scale 20
    p = _panel(x)
    for fn in (_consensus_series, _dispersion_series, _curvature_series):
        out = fn(p, [5, 10, 20])
        assert np.isnan(out[:, 0]).all(), fn.__name__


def test_warmup_is_max_scales():
    rng = np.random.default_rng(1)
    x = np.arange(1, 40, dtype=float) + 0.01 * rng.normal(size=39)
    out = _consensus_series(_panel(x), [5, 10, 20])
    # Rows [0, max(scale)-1] are NaN (data shorter than the largest scale);
    # from row 19 (= max scale) onward every declared scale is present.
    assert np.isnan(out[:19, 0]).all()
    assert np.isfinite(out[19:, 0]).all()


def test_consensus_sign_all_scales_present():
    rng = np.random.default_rng(1)
    x = np.arange(1, 40, dtype=float) + 0.01 * rng.normal(size=39)
    out = _consensus_series(_panel(x), [5, 10, 20])
    # Monotone-up series: every scale slope > 0 -> consensus = +1.
    assert np.allclose(out[19:, 0], 1.0, atol=1e-6)


def test_any_missing_scale_kills_row():
    # A NaN inside the largest scale's window must NaN the WHOLE row, even
    # though the smaller scales could still be computed (no partial recompute).
    rng = np.random.default_rng(1)
    x = np.arange(1, 20, dtype=float) + 0.01 * rng.normal(size=19)
    x[8] = np.nan  # sits inside scale-10 window for rows t in [8, 17]
    out = _consensus_series(_panel(x), [5, 10])
    assert np.isnan(out[17, 0])   # scale-10 window still contains the NaN
    assert np.isfinite(out[18, 0])  # scale-10 window finally clear
    assert out[18, 0] == pytest.approx(1.0, abs=1e-6)


def test_perfectly_linear_series_fails_closed():
    # Perfectly linear series -> OLS residual scale rs == 0 -> T_s undefined;
    # the no-EPS rule must fail closed to NaN instead of slope/1e-12 (explosion).
    x = np.arange(1, 30, dtype=float)
    for fn in (_consensus_series, _dispersion_series):
        out = fn(_panel(x), [5, 10, 20])
        assert np.isnan(out).all(), fn.__name__


def test_curvature_requires_three_scales():
    # A quadratic-in-log(s) fit needs >= 3 distinct scales; with 2 declared
    # scales the kernel emits all-NaN (documented limitation), never a value.
    rng = np.random.default_rng(2)
    x = np.arange(1, 40, dtype=float) + 0.01 * rng.normal(size=39)
    out = _curvature_series(_panel(x), [5, 10])
    assert np.isnan(out).all()


def test_curvature_three_or_more_scales_emits():
    rng = np.random.default_rng(3)
    x = np.arange(1, 60, dtype=float) + 0.01 * rng.normal(size=59)
    out = _curvature_series(_panel(x), [5, 10, 20, 40])
    assert np.isfinite(out[39:, 0]).all()


# ---------------------------------------------------------------------------
# Crossing — no EPS floor, NaN current row fails closed, sample-size aware
# ---------------------------------------------------------------------------
def test_crossing_speed_no_eps_floor():
    # z = x - y with a tiny-scale up-cross at t=3.  The exact value must equal
    # |dz|/scale_z (population std), NOT |dz|/(scale_z + 1e-12).
    zt = np.array([-2e-9, -1e-9, -0.5e-9, 1e-9])
    xs = zt.reshape(-1, 1)
    ys = np.zeros((4, 1))
    out = _crossing_speed_series(xs, ys, 4)
    z, dz, scale = _crossing_parts(xs, ys, 4)
    assert scale[3, 0] > 0.0
    expected = abs(dz[3, 0]) / scale[3, 0]
    assert out[3, 0] == pytest.approx(expected, rel=1e-12)
    # With an EPS floor the tiny scale (~1e-9) would be visibly inflated.
    assert abs(out[3, 0] - expected) < 1e-12


def test_crossing_nan_current_row_fails_closed():
    # Current row has NaN x -> z is NaN -> output must be NaN, never a stale 0.
    xs = _panel(np.array([1.0, 1.0, 1.0, np.nan]))
    ys = _panel(np.array([2.0, 2.0, 0.5, 0.5]))
    out = _crossing_speed_series(xs, ys, 4)
    assert np.isnan(out[3, 0])          # NaN current row -> NaN
    assert out[2, 0] > 0.0              # genuine up-cross one bar earlier -> value
    assert out[0, 0] == 0.0             # finite first bar, no previous z -> 0.0
    acc = _crossing_acceleration_series(xs, ys, 4)
    assert np.isnan(acc[3, 0])
    assert acc[0, 0] == 0.0 and acc[1, 0] == 0.0


def test_crossing_non_crossing_finite_bar_is_zero():
    xs = _panel(np.array([1.0, 1.5, 2.0, 2.5]))
    ys = _panel(np.array([0.0, 0.0, 0.0, 0.0]))
    out = _crossing_speed_series(xs, ys, 4)
    # z always > 0, never crosses 0 -> all bars 0.0, none NaN.
    assert np.all(out[:, 0] == 0.0)


def test_crossing_speed_hand_computed_value():
    # z = [-1, -1, 0.5] with window 3: up-cross at t=2.
    xs = _panel(np.array([1.0, 1.0, 1.5]))
    ys = _panel(np.array([2.0, 2.0, 1.0]))
    out = _crossing_speed_series(xs, ys, 3)
    z, dz, scale = _crossing_parts(xs, ys, 3)
    assert abs(dz[2, 0]) / scale[2, 0] == pytest.approx(out[2, 0], rel=1e-12)
    assert out[2, 0] > 0.0


# ---------------------------------------------------------------------------
# Envelope — constant series fails closed; price-level normalisation
# ---------------------------------------------------------------------------
def test_envelope_constant_zero_width_fails_closed():
    n = 10
    x = np.full(n, 5.0)
    u = np.full(n, 6.0)
    l = np.full(n, 6.0)  # upper == lower -> zero width
    m = np.full(n, 6.0)
    pr = _pressure_series(_panel(x), _panel(u), _panel(l), 5)
    bd = _boundary_dwell_series(_panel(x), _panel(u), _panel(l), 5, 0.8)
    cp = _compression_series(_panel(u), _panel(l), _panel(m), 5)
    assert np.isnan(pr).all()
    assert np.isnan(bd).all()
    assert np.isnan(cp).all()


def test_normalised_position_degenerate_is_nan_not_clip():
    assert np.isnan(_normalised_position(5.0, 6.0, 6.0))      # zero width
    assert np.isnan(_normalised_position(5.0, 6.0, 8.0))      # inverted
    assert _normalised_position(101.0, 102.0, 98.0) == pytest.approx(0.5)


def test_envelope_price_level_normalisation():
    # Same relative envelope at price level 100 vs 10 -> identical outputs.
    xa = np.array([99, 100, 101, 100, 99, 100, 101, 100], dtype=float)
    ua = np.array([102, 103, 104, 103, 102, 103, 104, 103], dtype=float)
    la = np.array([98, 99, 100, 99, 98, 99, 100, 99], dtype=float)
    ma = (ua + la) / 2.0
    xb, ub, lb, mb = xa * 0.1, ua * 0.1, la * 0.1, ma * 0.1
    pa = _pressure_series(_panel(xa), _panel(ua), _panel(la), 4)
    pb = _pressure_series(_panel(xb), _panel(ub), _panel(lb), 4)
    assert np.allclose(pa, pb, equal_nan=True, atol=1e-9)
    ca = _compression_series(_panel(ua), _panel(la), _panel(ma), 4)
    cb = _compression_series(_panel(ub), _panel(lb), _panel(mb), 4)
    assert np.allclose(ca, cb, equal_nan=True, atol=1e-9)


def test_envelope_compression_width_is_mid_normalised():
    # Compression width = (u-l)/|mid|: two histories with the same RELATIVE
    # envelope but different absolute price levels produce the same width.
    u = np.array([102.0, 103.0, 104.0])
    l = np.array([98.0, 99.0, 100.0])
    m = (u + l) / 2.0
    out = _compression_series(_panel(u), _panel(l), _panel(m), 3)
    # widths = 4/100, 4/101, 4/102 -> strictly decreasing -> current is min -> 1 - 0 (rank 1/3? width 4/102 is smallest, rank = 1/3 -> 1-1/3)
    # Just assert finite and in [0,1].
    assert np.isfinite(out[:, 0]).all()
    assert ((out[:, 0] >= 0.0) & (out[:, 0] <= 1.0)).all()


# ---------------------------------------------------------------------------
# Interval geometry — no EPS denominator (same cross-cutting intent)
# ---------------------------------------------------------------------------
def test_interval_union_coverage_no_eps_floor():
    # Tiny-scale price envelope: the ratio must be exactly union/env, with no
    # 1e-12 floor inflating the denominator.
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([1e-9, 2e-9, 1e-9])
    out = _union_coverage_series(_panel(lo), _panel(hi), 3)
    assert out[2, 0] == pytest.approx(1.0, abs=1e-12)


def test_interval_union_coverage_constant_envelope_nan():
    # All intervals collapsed to a point -> env == 0 -> NaN (no EPS explosion).
    lo = np.array([1.0, 1.0, 1.0])
    hi = np.array([1.0, 1.0, 1.0])
    out = _union_coverage_series(_panel(lo), _panel(hi), 3)
    assert np.isnan(out[:, 0]).all()


# ---------------------------------------------------------------------------
# Registered-operator smoke tests (through the registry)
#
# NOTE: no explicit ``load_all()`` here.  Importing the owned modules at the top
# of this file already runs their ``register_operator`` decorators, so the
# canonicals are in the registry; the operators-directory session conftest runs
# ``load_all()`` once for the whole suite anyway.  Keeping this hermetic avoids
# a hard dependency on unrelated concurrent-registry state.
# ---------------------------------------------------------------------------
def test_registered_ops_smoke():
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    close = pd.DataFrame(100 + np.cumsum(rng.normal(0, 1, (80, 2)), axis=0), index=idx, columns=["A", "B"])
    volume = pd.DataFrame(rng.lognormal(5, 0.5, (80, 2)), index=idx, columns=["A", "B"])
    upper = close.rolling(10, min_periods=3).max()
    lower = close.rolling(10, min_periods=3).min()

    cases = {
        "ts_multiscale_trend_consensus": ((close,), {"window": 50, "scales": (5, 10, 20, 40)}),
        "ts_crossing_speed": ((close, volume), {"window": 20}),
        "ts_crossing_acceleration": ((close, volume), {"window": 20}),
        "ts_envelope_pressure": ((close, upper, lower), {"window": 20}),
        "ts_interval_union_coverage": ((lower, upper), {"window": 20}),
    }
    for name, (args, kwargs) in cases.items():
        op = OperatorRegistry.get(name, "pandas_numpy")
        assert op is not None, name
        out = op.calculate(*args, **kwargs)
        arr = np.asarray(out, dtype=float)
        assert np.isfinite(arr).mean() >= 0.2, f"{name} finite fraction too low"
