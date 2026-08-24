# -*- coding: utf-8 -*-
"""R20 spectral/wavelet excess DirectUse rehabilitation oracles.

Slice: R20-SPECTRAL-EXCESS-DIRECTUSE.

Duplicate-audit outcome (documented skip / distinction):
* ``ts_spectral_lowpass_trailing`` / ``ts_wavelet_shrinkage_trailing``
  (frequency_layer / denoise_filter) are FILTERS — they return a filtered
  price-scale series.  No existing canonical returns a dimensionless
  spectral/wavelet ENERGY RATIO -> the three canonicals below are genuinely
  new math (no scipy / new dependency; pure numpy rfft + Haar).

Landed canonicals (all rolling-only, trailing window ENDING at t,
min_periods=window -> a single NaN inside the window makes the output NaN
(fail-closed, never zero-filled); DC rejected explicitly by de-meaning the
window before rfft and excluding bin 0 from the energy denominator; NOT in
``_RECURSIVE_EWM``; constant window -> 0.0 by contract):
* ``spectral_energy_ratio``      — top-k rfft-bin energy share, [0,1]
* ``spectral_trend_share``       — lowest-frequency-bin energy share, [0,1]
* ``wavelet_detail_energy_ratio``— Haar multilevel detail-energy share, [0,1]
  (window must be a power of 2 — ValueError at the call boundary).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

def _ensure_technical_chain() -> None:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    if OperatorRegistry.lifecycle() == "frozen":
        return
    if OperatorRegistry.get("spectral_energy_ratio", "pandas_numpy") is not None:
        return
    from factor_engine.cleaned_operators.technical import signal  # noqa: F401
    from factor_engine.cleaned_operators.technical import polars_signal  # noqa: F401
    from factor_engine.cleaned_operators import composite_fastpath  # noqa: F401
    from factor_engine.cleaned_operators.technical import indicators_v2  # noqa: F401

@pytest.fixture(scope="module", autouse=True)
def _bootstrap():
    _ensure_technical_chain()

def _op(name: str):
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get(name, "pandas_numpy") or OperatorRegistry.get(name)
    assert op is not None, f"{name} not registered"
    return op

def _trending_panel(n=40, seed=7):
    rng = np.random.default_rng(seed)
    close = pd.DataFrame({"A": 100.0 + np.cumsum(rng.normal(0.05, 0.6, n))})
    return close

def _sine_panel(n=64, periods=4.0, amp=2.0, seed=None):
    t = np.arange(n)
    x = 100.0 + amp * np.sin(2.0 * np.pi * periods * t / n)
    if seed is not None:
        x = x + np.random.default_rng(seed).normal(0.0, 1e-6, n)
    return pd.DataFrame({"A": x})

def _noise_panel(n=64, seed=11):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"A": 100.0 + rng.normal(0.0, 1.0, n)})

# ---------------------------------------------------------------------------
# manual oracles (independent of the operator helpers — recompute from raw
# numpy rfft / Haar on the raw window)
# ---------------------------------------------------------------------------
def _manual_oracle(close, window, kind, k=2):
    w = window
    vals = []
    arr = close.to_numpy(float).ravel()
    for t in range(len(arr)):
        if t < w - 1:
            vals.append(np.nan)
            continue
        a = arr[t - w + 1 : t + 1]
        if not np.isfinite(a).all():
            vals.append(np.nan)
            continue
        x = a - a.mean()
        if kind in ("energy", "trend"):
            spec = np.fft.rfft(x)
            power = spec.real ** 2 + spec.imag ** 2
            power = power[1:]
            total = power.sum()
            if total <= 1e-12:
                vals.append(0.0)
                continue
            if kind == "energy":
                kk = min(k, power.size)
                vals.append(float(np.sort(power)[::-1][:kk].sum() / total))
            else:
                kk = min(k, power.size)
                vals.append(float(power[:kk].sum() / total))
        else:  # haar detail (levels 1..J-1; coarsest detail grouped with DC)
            total = float(np.dot(x, x))
            if total <= 1e-12:
                vals.append(0.0)
                continue
            de = 0.0
            y = x
            n = y.size
            while n >= 4:
                even, odd = y[0::2], y[1::2]
                de += 0.5 * float(np.dot(even - odd, even - odd))
                y = (even + odd) / np.sqrt(2.0)
                n = y.size
            vals.append(float(min(de / total, 1.0)))
    return pd.DataFrame(vals, index=close.index, columns=close.columns)

def test_spectral_energy_ratio_matches_manual_oracle():
    close = _trending_panel(n=48, seed=7)
    out = _op("spectral_energy_ratio").calculate(close, window=16, top_k=2)
    expected = _manual_oracle(close, 16, "energy", k=2)
    np.testing.assert_allclose(
        out.to_numpy(), expected.to_numpy(), rtol=1e-10, atol=1e-12, equal_nan=True
    )
    # warmup: first w-1 bars NaN
    assert out.iloc[:15].isna().all().all()
    assert out.iloc[15:].notna().all().all()

def test_spectral_trend_share_matches_manual_oracle():
    close = _trending_panel(n=48, seed=7)
    out = _op("spectral_trend_share").calculate(close, window=16, trend_bins=2)
    expected = _manual_oracle(close, 16, "trend", k=2)
    np.testing.assert_allclose(
        out.to_numpy(), expected.to_numpy(), rtol=1e-10, atol=1e-12, equal_nan=True
    )

def test_wavelet_detail_energy_ratio_matches_manual_oracle():
    close = _trending_panel(n=48, seed=7)
    out = _op("wavelet_detail_energy_ratio").calculate(close, window=16)
    expected = _manual_oracle(close, 16, "haar")
    np.testing.assert_allclose(
        out.to_numpy(), expected.to_numpy(), rtol=1e-10, atol=1e-12, equal_nan=True
    )

# ---------------------------------------------------------------------------
# regime discrimination: pure sine vs white noise
# ---------------------------------------------------------------------------
def test_sine_panel_energy_ratio_near_one():
    # 4 full periods over the 64-bar panel: a 16-bar window holds exactly one
    # period -> ~all non-DC energy in one rfft bin -> ratio ~ 1.
    close = _sine_panel(n=64, periods=4.0, amp=2.0, seed=3)
    out = _op("spectral_energy_ratio").calculate(close, window=16, top_k=1)
    tail = out.iloc[15:]
    assert (tail > 0.99).all().all()
    # single dominant frequency bin: the sine has period 16 = the window, so
    # all non-DC energy sits in rfft bin 1 (one cycle per window)
    arr = close.to_numpy(float).ravel()[-16:]
    spec = np.fft.rfft(arr - arr.mean())
    power = spec.real ** 2 + spec.imag ** 2
    assert int(np.argmax(power[1:])) + 1 == 1
    assert power[1] / power[1:].sum() > 0.99

def test_noise_panel_energy_ratio_materially_lower():
    sine = _op("spectral_energy_ratio").calculate(_sine_panel(n=64, periods=4.0, seed=3), window=16, top_k=1)
    noise = _op("spectral_energy_ratio").calculate(_noise_panel(n=64, seed=11), window=16, top_k=1)
    assert sine.iloc[-1, 0] - noise.iloc[-1, 0] > 0.5
    assert noise.iloc[-1, 0] < 0.5

def test_trend_share_separates_trend_from_noise():
    # A smooth ramp concentrates non-DC energy in the lowest bins; white noise
    # does not.
    ramp = pd.DataFrame({"A": 100.0 + np.linspace(0.0, 10.0, 64)})
    out_ramp = _op("spectral_trend_share").calculate(ramp, window=16, trend_bins=2)
    out_noise = _op("spectral_trend_share").calculate(_noise_panel(n=64, seed=5), window=16, trend_bins=2)
    assert out_ramp.iloc[-1, 0] > 0.7  # linear ramp: two slowest bins dominate
    assert out_ramp.iloc[-1, 0] - out_noise.iloc[-1, 0] > 0.3

def test_haar_detail_separates_smooth_from_choppy():
    ramp = pd.DataFrame({"A": 100.0 + np.linspace(0.0, 10.0, 64)})
    smooth = _op("wavelet_detail_energy_ratio").calculate(ramp, window=16)
    choppy = _op("wavelet_detail_energy_ratio").calculate(_noise_panel(n=64, seed=5), window=16)
    assert smooth.iloc[-1, 0] < 0.3
    assert choppy.iloc[-1, 0] > smooth.iloc[-1, 0] + 0.4

# ---------------------------------------------------------------------------
# edge values: constant window -> 0.0 (documented)
# ---------------------------------------------------------------------------
def test_constant_window_is_zero():
    close = pd.DataFrame({"A": [10.0] * 24})
    for name, kwargs in (
        ("spectral_energy_ratio", {"top_k": 2}),
        ("spectral_trend_share", {"trend_bins": 2}),
        ("wavelet_detail_energy_ratio", {}),
    ):
        out = _op(name).calculate(close, window=8, **kwargs)
        assert out.iloc[:7].isna().all().all()
        np.testing.assert_allclose(out.iloc[7:].to_numpy(), 0.0)

def test_outputs_bounded_unit_interval():
    panels = [_trending_panel(n=48, seed=1), _noise_panel(n=48, seed=2)]
    for p in panels:
        for name, kwargs in (
            ("spectral_energy_ratio", {"top_k": 2}),
            ("spectral_trend_share", {"trend_bins": 3}),
            ("wavelet_detail_energy_ratio", {}),
        ):
            out = _op(name).calculate(p, window=8, **kwargs).dropna()
            assert ((out >= 0.0) & (out <= 1.0)).all().all()

def test_dc_offset_invariance():
    # DC is rejected via de-meaning: adding a large constant leaves the ratio
    # unchanged (dimensionless, level-free).
    base = _trending_panel(n=48, seed=9)
    for name, kwargs in (
        ("spectral_energy_ratio", {"top_k": 2}),
        ("spectral_trend_share", {"trend_bins": 2}),
        ("wavelet_detail_energy_ratio", {}),
    ):
        a = _op(name).calculate(base, window=16, **kwargs)
        b = _op(name).calculate(base + 1000.0, window=16, **kwargs)
        np.testing.assert_allclose(
            a.to_numpy(), b.to_numpy(), rtol=1e-8, atol=1e-10, equal_nan=True
        )

# ---------------------------------------------------------------------------
# bad-close masking (R5-38) + NaN-in-window fail-closed
# ---------------------------------------------------------------------------
def test_masks_non_positive_close():
    close = _trending_panel(n=40, seed=5)
    close.iloc[20, 0] = -1.0
    for name, kwargs in (
        ("spectral_energy_ratio", {"top_k": 2}),
        ("spectral_trend_share", {"trend_bins": 2}),
        ("wavelet_detail_energy_ratio", {}),
    ):
        out = _op(name).calculate(close, window=8, **kwargs)
        # strict-positive masking: the bad bar itself and every window whose
        # FIRST bars include it (t in 20..27) -> NaN; windows whose bad bar
        # would be dropped from the head (t > 27) recover with w valid obs.
        assert np.isnan(out.iloc[20, 0]), name
        assert out.iloc[20:28].isna().all().all(), name
        assert out.iloc[28:].notna().all().all(), name
        # no fabricated values: windows before the bad bar are untouched
        assert out.iloc[8:20].notna().all().all(), name

def test_nan_in_window_fail_closed():
    close = _trending_panel(n=40, seed=5)
    close.iloc[20, 0] = np.nan
    for name, kwargs in (
        ("spectral_energy_ratio", {"top_k": 2}),
        ("spectral_trend_share", {"trend_bins": 2}),
        ("wavelet_detail_energy_ratio", {}),
    ):
        out = _op(name).calculate(close, window=8, **kwargs)
        # a true NaN is NEVER dropped by the head-count: every window that
        # contains bar 20 (t in 20..27) -> NaN (fail-closed, no zero-fill)
        assert out.iloc[20:28].isna().all().all(), name
        assert out.iloc[28:].notna().all().all(), name
        assert out.iloc[8:20].notna().all().all(), name

# ---------------------------------------------------------------------------
# wavelet power-of-2 window enforcement
# ---------------------------------------------------------------------------
def test_wavelet_window_must_be_power_of_two():
    close = _trending_panel(n=40, seed=5)
    with pytest.raises(ValueError):
        _op("wavelet_detail_energy_ratio").calculate(close, window=12)
    # window=2 is rejected by the shared family minimum (>= 4) inside
    # _pow2 — it can no longer pass _pow2 and die later on a misleading
    # message from _spectral_energy (review P1).
    with pytest.raises(ValueError):
        _op("wavelet_detail_energy_ratio").calculate(close, window=2)
    # power-of-2 windows work
    out = _op("wavelet_detail_energy_ratio").calculate(close, window=16)
    assert out.iloc[15:].notna().all().all()

# ---------------------------------------------------------------------------
# causality / prefix invariance
# ---------------------------------------------------------------------------
def test_spectral_family_is_prefix_invariant_under_truncation():
    close = _trending_panel(n=48, seed=13)
    for name, kwargs in (
        ("spectral_energy_ratio", {"top_k": 2}),
        ("spectral_trend_share", {"trend_bins": 2}),
        ("wavelet_detail_energy_ratio", {}),
    ):
        full = _op(name).calculate(close, window=8, **kwargs)
        head = _op(name).calculate(close.iloc[:30], window=8, **kwargs)
        pd.testing.assert_frame_equal(full.iloc[:30], head)

def test_spectral_family_is_scale_invariant():
    close = _trending_panel(n=48, seed=13)
    for name, kwargs in (
        ("spectral_energy_ratio", {"top_k": 2}),
        ("spectral_trend_share", {"trend_bins": 2}),
        ("wavelet_detail_energy_ratio", {}),
    ):
        a = _op(name).calculate(close, window=8, **kwargs)
        b = _op(name).calculate(close * 100.0, window=8, **kwargs)
        np.testing.assert_allclose(
            a.to_numpy(), b.to_numpy(), rtol=1e-8, atol=1e-10, equal_nan=True
        )

# ---------------------------------------------------------------------------
# governance / promotion
# ---------------------------------------------------------------------------
def test_spectral_family_param_specs_and_governance():
    specs = _op("spectral_energy_ratio").metadata.param_specs
    assert set(specs) == {"window", "top_k"}
    # min=4 mirrors the runtime _pi(window,"window",4) guard in
    # _spectral_energy — windows < 4 are pruned by the spec up front
    # (review P1: spec/runtime minimum mismatch caused dead search nodes).
    assert specs["window"].dtype is int and specs["window"].min == 4
    assert specs["top_k"].dtype is int and specs["top_k"].min == 1
    assert specs["top_k"].searchable is False  # structural, not a tuned knob
    assert not _op("spectral_energy_ratio").metadata.relational_specs

    specs = _op("spectral_trend_share").metadata.param_specs
    assert set(specs) == {"window", "trend_bins"}
    assert specs["trend_bins"].searchable is False
    assert not _op("spectral_trend_share").metadata.relational_specs

    specs = _op("wavelet_detail_energy_ratio").metadata.param_specs
    assert set(specs) == {"window"}

    from factor_engine.cleaned_operators.technical.indicators_v2 import _RECURSIVE_EWM

    for name in ("spectral_energy_ratio", "spectral_trend_share", "wavelet_detail_energy_ratio"):
        assert name not in _RECURSIVE_EWM
        tags = _op(name).metadata.tags
        assert "stateful" not in tags
        assert "causal" in tags and "pit_safe" in tags

def test_spectral_family_promotion_membership():
    promoted = {"spectral_energy_ratio", "spectral_trend_share", "wavelet_detail_energy_ratio"}
    from factor_engine.mining.direct_use import _RELATIVE_ALPHA_OPS

    assert promoted <= _RELATIVE_ALPHA_OPS

    from factor_engine.cleaned_operators.operator_surface import (
        _DAILY_SPECTRAL_PACK_2026_08,
        classify_canonical,
        daily_factor_migrated,
    )

    assert promoted <= _DAILY_SPECTRAL_PACK_2026_08
    assert _DAILY_SPECTRAL_PACK_2026_08 <= daily_factor_migrated()
    for name in promoted:
        assert classify_canonical(name) == "daily"

    # dimensionless alphas, never price-scale intermediates
    from factor_engine.mining.direct_use import _PRICE_LEVEL_INTERMEDIATE_OPS

    for name in promoted:
        assert name not in _PRICE_LEVEL_INTERMEDIATE_OPS
