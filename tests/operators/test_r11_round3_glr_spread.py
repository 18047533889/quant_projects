# -*- coding: utf-8 -*-
"""R11 round-3 operator audit: P0-L GLR calibration + P0-M OHLC-spread gates.

Covers the five audit items owned by this file-disjoint fixer:

* P0-L-69 — GLR perfect split is capped by a documented LLR cap, never decided
  by the program EPS floor (``_EPS * full_ss`` is no longer a denominator).
* P0-L-70 — max-over-breakpoint GLR is null-calibrated with a ``ln(#candidates)``
  penalty so the null is not a free function of the window.
* P0-M-71 — EDGE exposes ``min_valid_pairs`` / ``min_valid_ratio`` output gates.
* P0-M-72 — Abdi-Ranaldo returns NaN on a negative squared-spread estimate
  instead of silently clipping it to 0 (a clipped 0 is read as "spread = 0").
* P0-M-73 — Pastor-Stambaugh takes an explicit ``flow_scale`` unit parameter
  instead of a hidden hard-coded ``1e6`` in the kernel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.base import ParamRole
from factor_engine.cleaned_operators.glr_change import (
    _GLR_MAX_LLR,
    _mean_shift_score,
    _null_calibrate_llr,
    _ss_noise_floor,
    _variance_shift_score,
)
from factor_engine.cleaned_operators.ohlc_spread import _abdi_ranaldo_window, _edge_window
from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module")
def loaded():
    from factor_engine.cleaned_operators import load_all

    load_all()


# ---------------------------------------------------------------------------
# P0-L-69: GLR perfect split must not be decided by an EPS floor
# ---------------------------------------------------------------------------
def test_mean_shift_perfect_split_is_capped_not_eps_dominated():
    # constant A then constant B -> residual within-segment SS = 0, true LLR
    # unbounded.  The old ``max(pooled, EPS*full_ss)`` denominator gave
    # sqrt((n/2)*ln(1/1e-12)) ~ 34; the documented cap must be sqrt(40) ~ 6.32.
    v = np.concatenate([np.full(40, 0.0), np.full(40, 1.0)])
    s = _mean_shift_score(v, 6)
    assert np.isfinite(s)
    assert s == pytest.approx(np.sqrt(_GLR_MAX_LLR), abs=1e-12)
    assert s < 7.0  # capped, not the old EPS-dominated ~34


def test_mean_shift_near_perfect_split_also_caps():
    # a near-perfect split (residual SS just above the noise floor) must land on
    # the same documented cap as a perfect split — the cap is applied after the
    # scan, so the max score is continuous in the change size.
    v = np.concatenate([np.full(40, 0.0), np.full(40, 1.0)])
    v[40] = 1e-6  # break the perfect split by a hair
    s = _mean_shift_score(v, 6)
    assert np.isfinite(s)
    assert s == pytest.approx(np.sqrt(_GLR_MAX_LLR), abs=1e-12)


def test_variance_shift_one_side_constant_caps():
    # one segment perfectly constant, the other noisy -> infinite variance ratio.
    rng = np.random.default_rng(2)
    v = np.concatenate([np.full(40, 5.0), rng.normal(0, 0.1, 40)])
    s = _variance_shift_score(v, 6)
    assert np.isfinite(s)
    assert s == pytest.approx(np.sqrt(_GLR_MAX_LLR), abs=1e-12)


def test_noise_floor_is_measurement_based_not_arbitrary():
    # the perfect-split classifier floor must scale with the data (n * eps),
    # not be the old absolute-ish 1e-12 * full_ss.
    full_ss = 1.0
    floor_n20 = _ss_noise_floor(full_ss, 20)
    floor_n80 = _ss_noise_floor(full_ss, 80)
    assert floor_n20 < floor_n80  # grows with window length (cancellation)
    assert floor_n80 < 1e-12      # still far below the old program EPS


# ---------------------------------------------------------------------------
# P0-L-70: max-over-breakpoint GLR scan-selection bias (null calibration)
# ---------------------------------------------------------------------------
def test_null_calibration_centres_lnm_at_zero():
    # m = n - 2*min_segment + 1 candidates; a raw max LLR of ln(m) (the
    # approximate expected max under the null) must calibrate to 0.
    n, ms = 80, 6
    m = n - 2 * ms + 1
    assert _null_calibrate_llr(np.log(m), n, ms) == 0.0
    # a raw LLR above ln(m) survives; below collapses to 0
    assert _null_calibrate_llr(np.log(m) + 5.0, n, ms) == pytest.approx(5.0, abs=1e-12)
    assert _null_calibrate_llr(np.log(m) - 5.0, n, ms) == 0.0


def test_flat_null_score_is_window_independent():
    # the whole point of the calibration: a no-change series must not score
    # higher just because a bigger window scans more breakpoints.
    rng = np.random.default_rng(4)
    data = rng.normal(0.0, 0.05, 160)
    scores = [_mean_shift_score(data[:n], 6) for n in (40, 80, 120)]
    for s in scores:
        assert np.isfinite(s)
    # raw max LLR under the null grows ~ ln m; after the ln(m) penalty the score
    # is centred near 0 for every window.
    assert all(s < 1.0 for s in scores)
    assert max(scores) - min(scores) < 0.5


def test_real_break_still_detected_after_calibration():
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.normal(0.0, 0.05, 40), rng.normal(1.0, 0.05, 40)])
    flat = rng.normal(0.0, 0.05, 80)
    brk = _mean_shift_score(x, 6)
    base = _mean_shift_score(flat, 6)
    assert np.isfinite(brk) and np.isfinite(base)
    assert brk > base + 2.0


# ---------------------------------------------------------------------------
# P0-M-71: EDGE valid-pair coverage output gates
# ---------------------------------------------------------------------------
def _edge_ohlc(n=60, seed=7):
    rng = np.random.default_rng(seed)
    p = 100.0 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))
    o = p * np.exp(rng.normal(0, 0.002, n))
    c = p * np.exp(rng.normal(0, 0.002, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    return o, h, l, c


def test_edge_valid_pair_gate_kernel_level():
    o, h, l, c = _edge_ohlc()
    # clean data: every adjacent pair usable -> default and ratio=1.0 agree
    base = _edge_window(o, h, l, c)
    assert np.isfinite(base)
    assert _edge_window(o, h, l, c, min_valid_pairs=2, min_valid_ratio=1.0) == pytest.approx(base, abs=1e-12)

    # corrupt ONE OHLC row (h < max(o, c)) -> that row masks, pairs around it
    # drop out.  A strict ratio gate must fail the window closed...
    o2, h2, l2, c2 = o.copy(), h.copy(), l.copy(), c.copy()
    h2[30] = max(o2[30], c2[30]) * 0.999
    assert np.isfinite(_edge_window(o2, h2, l2, c2))                 # default stays on
    assert np.isnan(_edge_window(o2, h2, l2, c2, min_valid_ratio=1.0))  # strict gate NaN
    # ...and a strict absolute-count gate must too
    assert np.isnan(_edge_window(o2, h2, l2, c2, min_valid_pairs=60))


def test_edge_valid_pair_gate_registered_and_searchable_false(loaded):
    op = OperatorRegistry.get("ts_edge_effective_spread", "pandas_numpy")
    specs = op.metadata.param_specs
    for name in ("min_valid_pairs", "min_valid_ratio"):
        assert name in specs, name
        assert specs[name].searchable is False, name
        assert specs[name].param_role == ParamRole.NUMERICAL, name

    o, h, l, c = _edge_ohlc()
    f = lambda a: pd.DataFrame({"s0": a})
    out_default = op.calculate(f(o), f(h), f(l), f(c), window=60).to_numpy()[-1, 0]
    out_gated = op.calculate(f(o), f(h), f(l), f(c), window=60, min_valid_ratio=1.0).to_numpy()[-1, 0]
    assert np.isfinite(out_default)
    assert out_gated == pytest.approx(out_default, abs=1e-12)

    o2, h2, l2, c2 = o.copy(), h.copy(), l.copy(), c.copy()
    h2[30] = max(o2[30], c2[30]) * 0.999
    assert np.isfinite(op.calculate(f(o2), f(h2), f(l2), f(c2), window=60).to_numpy()[-1, 0])
    assert np.isnan(op.calculate(f(o2), f(h2), f(l2), f(c2), window=60, min_valid_ratio=1.0).to_numpy()[-1, 0])


# ---------------------------------------------------------------------------
# P0-M-72: Abdi-Ranaldo negative squared spread -> NaN, not clipped 0
# ---------------------------------------------------------------------------
def test_abdi_negative_s2_returns_nan_not_zero():
    # (c - eta) alternates sign -> mean product negative -> s2 < 0
    c = np.array([1.0, 2.0, 3.0, 4.0])
    eta = np.array([3.0, 1.0, 5.0, 2.0])
    assert np.isnan(_abdi_ranaldo_window(c, eta, 4))


def test_abdi_positive_s2_finite_and_zero_is_zero():
    # falling series -> s2 > 0, a real positive spread
    close = np.log(np.array([10.0, 9.8, 9.6, 9.4, 9.2]))
    hi = np.log(np.array([10.0, 9.8, 9.6, 9.4, 9.2]) * 1.001)
    lo = np.log(np.array([10.0, 9.8, 9.6, 9.4, 9.2]) * 0.999)
    eta = (hi + lo) / 2.0
    val = _abdi_ranaldo_window(close, eta, 5)
    assert np.isfinite(val) and val > 0.0
    # exactly constant path -> s2 == 0 is a legitimate zero spread
    c2 = np.log(np.full(5, 10.0))
    assert _abdi_ranaldo_window(c2, np.log(np.full(5, 10.0)), 5) == 0.0


def test_abdi_negative_s2_operator_fail_closed(loaded):
    # the same panel the old reference-formula test fed, which produces a
    # negative squared spread: the operator must yield NaN, not a silent 0.
    close = pd.DataFrame({"a": [10.0, 10.5, 11.0, 10.8, 11.2]})
    high = close * 1.03
    low = close * 0.97
    op = OperatorRegistry.get("ts_abdi_ranaldo_spread", "pandas_numpy")
    out = op.calculate(close, high, low, window=5).to_numpy()[-1, 0]
    assert np.isnan(out)


# ---------------------------------------------------------------------------
# P0-M-73: Pastor-Stambaugh flow_scale is an explicit unit parameter
# ---------------------------------------------------------------------------
def test_ps_flow_scale_scales_gamma_linearly(loaded):
    rng = np.random.default_rng(0)
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    ret = pd.DataFrame(rng.normal(0, 0.01, (n, 2)), index=idx, columns=list("AB"))
    amt = pd.DataFrame(np.abs(rng.normal(1e6, 3e5, (n, 2))), index=idx, columns=list("AB"))
    bench = pd.DataFrame(np.zeros((n, 2)), index=idx, columns=list("AB"))

    op = OperatorRegistry.get("ts_pastor_stambaugh_liquidity_gamma", "pandas_numpy")
    out_m = op.calculate(ret, bench, amt, window=40, min_periods=12).to_numpy()[-1, :]           # flow_scale=1e6
    out_1 = op.calculate(ret, bench, amt, window=40, min_periods=12, flow_scale=1.0).to_numpy()[-1, :]
    mask = np.isfinite(out_m) & np.isfinite(out_1)
    assert mask.sum() > 0
    # gamma is the regression coefficient on flow = amount/flow_scale, so
    # gamma(1e6) == 1e6 * gamma(1).  The old hard-coded 1e6 is now explicit.
    np.testing.assert_allclose(out_m[mask], 1e6 * out_1[mask], rtol=1e-6)


def test_ps_flow_scale_registered_and_searchable_false(loaded):
    op = OperatorRegistry.get("ts_pastor_stambaugh_liquidity_gamma", "pandas_numpy")
    spec = op.metadata.param_specs["flow_scale"]
    assert spec.searchable is False
    assert spec.param_role == ParamRole.NUMERICAL
    assert spec.default == 1e6  # the reference "per million" convention
