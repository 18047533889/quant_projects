# -*- coding: utf-8 -*-
"""R11 round-3 audit fixes — spectral (wavelet / FFT) + memory operators.

Covers audit items 55-62 on the file-disjoint owned set:

* ``ts_model/wavelet_spectral.py`` — 55 (missing current value -> NaN, never
  stale), 56 (fixed power-of-two window anchor {32,64,128,256} instead of the
  unstable "largest power-of-two suffix"), 57 (single-band wavelet entropy = 0,
  not NaN), 58 (spectral factor no partial warmup).
* ``spectral_ext.py``             — 58 (full-window entropy / dominant cycle),
  59 (dominant-cycle ``window`` is NOT a free search parameter — the peak-share
  gate ``max(P)/sum(P)`` has mechanical N dependence).
* ``research_spectral.py``        — 58 (bicoherence full window), 59
  (bicoherence ``window`` gets the coarse ESTIMATOR_RESOLUTION role).
* ``memory_ext.py``               — 60 (standard finite-N ACF denominator
  ``gamma_k = (1/(N-k)) sum ...``), 61 (Geyer initial-positive /
  initial-monotone sequence estimators), 62 (fractional-difference
  ``discarded_weight_mass`` output + fail-closed gate).

Modules are imported directly (their registration is import-triggered) so this
file stays self-contained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factor_engine.cleaned_operators.spectral_ext  # noqa: F401
import factor_engine.cleaned_operators.memory_ext  # noqa: F401
import factor_engine.cleaned_operators.research_spectral  # noqa: F401
import factor_engine.cleaned_operators.ts_model.wavelet_spectral as _ws  # noqa: F401

from factor_engine.cleaned_operators.base import ParamRole
from factor_engine.cleaned_operators.memory_ext import (
    _fd_discarded_weight_mass,
    _geyer_ims_tau,
    _geyer_ips_tau,
    _sample_acf,
)
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _frame(values: np.ndarray, start: str = "2024-01-01") -> pd.DataFrame:
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    return pd.DataFrame(
        v, index=pd.date_range(start, periods=v.shape[0], freq="B"), columns=["A"]
    )


def _op(name: str):
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None, name
    return op


def _calc(name: str, values: np.ndarray, **params) -> np.ndarray:
    return _op(name).calculate(_frame(values), **params)["A"].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# item 55  wavelet: a missing current value must output NaN, never stale
# ---------------------------------------------------------------------------
def test_wavelet_current_nan_is_nan_not_stale():
    rng = np.random.default_rng(0)
    seq = rng.normal(10.0, 1.0, 200)
    seq_nan = seq.copy()
    seq_nan[-1] = np.nan

    good = _calc("ts_wavelet_low_frequency_ratio", seq, window=128)
    bad = _calc("ts_wavelet_low_frequency_ratio", seq_nan, window=128)

    # the current (last) row is NaN -> the trailing factor must be NaN, not the
    # previous day's value computed on the pre-NaN run
    assert np.isnan(bad[-1])
    # the row before the missing value is unaffected (no re-pairing of history)
    assert np.isfinite(good[-2])
    assert np.isclose(good[-2], bad[-2], equal_nan=True)


def test_wavelet_entropy_current_nan_is_nan():
    rng = np.random.default_rng(1)
    seq = rng.normal(10.0, 1.0, 200)
    seq_nan = seq.copy()
    seq_nan[-1] = np.nan
    out = _calc("ts_wavelet_entropy", seq_nan, window=128)
    assert np.isnan(out[-1])


# ---------------------------------------------------------------------------
# item 56  wavelet: fixed power-of-two window anchor (no horizon jumps)
# ---------------------------------------------------------------------------
def test_wavelet_fixed_window_anchor_domain():
    assert _ws._fixed_window_anchor(32) == 32
    assert _ws._fixed_window_anchor(64) == 64
    assert _ws._fixed_window_anchor(128) == 128
    assert _ws._fixed_window_anchor(256) == 256
    for unsupported in (60, 100, 127):
        with pytest.raises(ValueError, match="window must be one of"):
            _ws._fixed_window_anchor(unsupported)


def test_wavelet_haar_requires_full_fixed_anchor():
    rng = np.random.default_rng(2)
    v127 = rng.normal(size=127)
    v128 = np.concatenate([v127, [1.0]])
    # 127 observations cannot fill a 128-anchor window -> NaN (no silent 64-bar
    # Haar that would make the definition jump when the 128th bar arrives)
    assert _ws._haar_energy(v127, 128) == []
    assert len(_ws._haar_energy(v128, 128)) == 7  # exactly 128 samples -> 7 levels


def test_wavelet_transform_uses_exactly_the_fixed_anchor():
    # Two series with DIFFERENT histories but the SAME trailing 128-sample block
    # must produce the identical trailing factor: the transform always uses the
    # newest fixed-anchor samples and never a length that depends on how many
    # observations happened to be available.
    rng = np.random.default_rng(11)
    block = rng.normal(size=128)
    a = np.concatenate([rng.normal(size=50), block])
    b = np.concatenate([rng.normal(size=100), block])
    va = _calc("ts_wavelet_low_frequency_ratio", a, window=128)[-1]
    vb = _calc("ts_wavelet_low_frequency_ratio", b, window=128)[-1]
    assert np.isclose(va, vb)
    # and the same fixed anchor holds for the other wavelet statistics
    ea = _calc("ts_wavelet_entropy", a, window=128)[-1]
    eb = _calc("ts_wavelet_entropy", b, window=128)[-1]
    assert np.isclose(ea, eb)


# ---------------------------------------------------------------------------
# item 57  wavelet: a single positive energy band has entropy 0, not NaN
# ---------------------------------------------------------------------------
def test_wavelet_entropy_single_band_is_zero():
    # 128 samples: first half = 1.0, second half = 2.0 -> only the coarsest Haar
    # band carries energy -> p = (1, 0, 0, ...) -> H = 0 theoretically.
    x = np.concatenate([np.full(64, 1.0), np.full(64, 2.0)])
    val = _ws._wavelet_stats(x, 128, "entropy")
    assert np.isfinite(val)
    assert val == 0.0

    # via the operator on a panel whose trailing 128-block is the same shape
    out = _calc("ts_wavelet_entropy",
                np.concatenate([np.full(64, 1.0), np.full(64, 2.0), np.full(64, 1.0)]),
                window=128)
    assert np.isfinite(out[-1])
    assert out[-1] == 0.0


# ---------------------------------------------------------------------------
# item 58  spectral factors must NOT partial-warmup (full window required)
# ---------------------------------------------------------------------------
def test_spectral_entropy_no_partial_warmup():
    rng = np.random.default_rng(3)
    ret = rng.normal(0.0, 1.0, 100)
    out = _calc("ts_return_spectral_entropy", ret, window=60)
    assert np.isnan(out[:59]).all(), "startup rows must be NaN (no partial spectrum)"
    assert np.isfinite(out[59])


def test_spectral_low_frequency_ratio_no_partial_warmup():
    rng = np.random.default_rng(4)
    x = rng.normal(0.0, 1.0, 140)
    out = _calc("ts_spectral_low_frequency_ratio", x, window=128)
    assert np.isnan(out[:127]).all()
    assert np.isfinite(out[127])


def test_dominant_cycle_period_no_partial_warmup():
    t = np.arange(100.0)
    sine = np.sin(2.0 * np.pi * t / 8.0)  # strong peak -> period survives the gate
    out = _calc("ts_dominant_cycle_period", sine, window=60)
    assert np.isnan(out[:59]).all()
    assert np.isfinite(out[59])


def test_bicoherence_no_partial_warmup():
    rng = np.random.default_rng(6)
    ret = rng.normal(0.0, 1.0, 80)
    out = _calc("ts_bicoherence_top_decile_mean", ret, window=48, n_segments=2)
    assert np.isnan(out[:47]).all()
    assert np.isfinite(out[47])


# ---------------------------------------------------------------------------
# item 59  spectral peak concentration: window is not a free parameter
# ---------------------------------------------------------------------------
def test_dominant_cycle_window_not_searchable():
    spec = _op("ts_dominant_cycle_period").metadata.param_specs["window"]
    assert spec.searchable is False
    assert spec.param_role == ParamRole.ESTIMATOR_RESOLUTION


def test_bicoherence_window_coarse_resolution_role():
    for name in ("ts_bicoherence_top_decile_mean", "ts_bicoherence_top_decile_excess"):
        spec = _op(name).metadata.param_specs["window"]
        assert spec.param_role == ParamRole.ESTIMATOR_RESOLUTION


def test_spectral_entropy_window_stays_searchable():
    # the normalized entropy canonicals are N-invariant and keep a searchable
    # window — only the peak-concentration statistics lose the free window
    for name in ("ts_return_spectral_entropy", "ts_spectral_entropy",
                 "ts_detrended_level_spectral_entropy"):
        assert _op(name).metadata.param_specs["window"].searchable is True


# ---------------------------------------------------------------------------
# item 60  ACF-time standard finite-N estimator
# ---------------------------------------------------------------------------
def test_sample_acf_uses_standard_n_minus_k_denominator():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rho = _sample_acf(x, 2)
    mu = 3.0
    gamma0 = float(np.sum((x - mu) ** 2) / 5.0)  # == 2.0
    assert gamma0 == pytest.approx(2.0)
    # gamma_k = (1/(N-k)) * sum_{t=1}^{N-k} (x_t - xbar)(x_{t+k} - xbar)
    for k in (1, 2):
        num = float(np.sum((x[: 5 - k] - mu) * (x[k:] - mu)))
        assert rho[k] == pytest.approx((num / (5.0 - k)) / gamma0)
    # lag-1: 4/4 / 2 = 0.5.  The old full-sample denominator (num / (5*gamma0))
    # would have given 4/10 = 0.4 — the finite-N bias the audit targets.
    assert rho[0] == pytest.approx(1.0)
    assert rho[1] == pytest.approx(0.5)


def test_autocorrelation_time_uses_corrected_denominator():
    # A series whose lag-1 correlation is 0.5 by the standard estimator must
    # return the IMS tau based on the corrected rho, not the biased one.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 2.0, 4.0, 5.0, 6.0])
    rho = _sample_acf(x, 3)
    assert rho[1] > 0.0
    out = _op("ts_autocorrelation_time").calculate(_frame(x), window=10, max_lag=3)["A"].to_numpy()[-1]
    assert out == pytest.approx(_geyer_ims_tau(rho), abs=1e-12)


# ---------------------------------------------------------------------------
# item 61  Geyer initial-positive / initial-monotone sequence estimators
# ---------------------------------------------------------------------------
def test_geyer_ims_and_ips_white_noise_and_ar1():
    rng = np.random.default_rng(7)
    # white noise -> tau ~ 1 bar regardless of max_lag
    rho_wn = _sample_acf(rng.normal(size=2000), 30)
    assert 0.5 < _geyer_ims_tau(rho_wn) < 2.0
    assert 0.5 < _geyer_ips_tau(rho_wn) < 2.0

    # strongly autocorrelated AR(1) -> tau well above 1
    n = 4000
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.95 * y[t - 1] + rng.normal()
    rho_ar = _sample_acf(y, 40)
    assert _geyer_ims_tau(rho_ar) > 5.0
    assert np.isfinite(_geyer_ips_tau(rho_ar))


def test_geyer_ims_handles_negative_rho1():
    # a single non-positive lag-1 autocorrelation must still give a FINITE tau
    # (the monotone-sequence recursion never crashes)
    rho_neg = np.array([1.0, -0.1, 0.3])
    val = _geyer_ims_tau(rho_neg)
    assert np.isfinite(val)
    assert val == pytest.approx(1.0 + 2.0 * (-0.1))
    assert np.isfinite(_geyer_ips_tau(rho_neg))


def test_autocorrelation_time_canonical_uses_ims():
    rng = np.random.default_rng(8)
    x = rng.normal(size=300)
    out = _calc("ts_autocorrelation_time", x, window=200, max_lag=30)
    assert np.isfinite(out[-1])
    assert out[-1] == pytest.approx(_geyer_ims_tau(_sample_acf(x[-200:], 30)), abs=1e-12)


def test_autocorrelation_time_ips_canonical_matches_ips():
    rng = np.random.default_rng(9)
    x = rng.normal(size=300)
    out = _calc("ts_autocorrelation_time_initial_positive_sequence", x, window=200, max_lag=30)
    assert np.isfinite(out[-1])
    assert out[-1] == pytest.approx(_geyer_ips_tau(_sample_acf(x[-200:], 30)), abs=1e-12)


def test_integrated_autocorrelation_time_alias_still_resolves():
    # the precise spelling keeps resolving to the (now Geyer IMS) canonical
    from factor_engine.cleaned_operators.registry import OperatorRegistry as _reg

    target = _reg.resolve_canonical("ts_integrated_autocorrelation_time")
    assert target == "ts_autocorrelation_time"


# ---------------------------------------------------------------------------
# item 62  fractional-difference cutoff must follow d (discarded-mass gate)
# ---------------------------------------------------------------------------
def test_fractional_difference_discarded_mass_gate():
    x = np.linspace(1.0, 80.0, 80)
    # d < 0 (fractional integration): weights decay like k^{-(1+d)}, a divergent
    # p-series -> discarded mass = 1 -> the gate ALWAYS fails closed (all NaN).
    out_neg = _calc("ts_fractional_difference", x, fd=-0.4, cutoff=20)
    assert np.isnan(out_neg).all()
    # d > 0 with the default tolerance (0.5): D(0.4,20) ~ 0.10 passes, startup
    # region (r < cutoff) is NaN as before.
    out_pos = _calc("ts_fractional_difference", x, fd=0.4, cutoff=20)
    assert np.isnan(out_pos[:20]).all()
    assert np.isfinite(out_pos[20:]).all()
    # a tight tolerance fails closed too
    out_tight = _calc("ts_fractional_difference", x, fd=0.4, cutoff=20,
                      max_discarded_weight_mass=0.05)
    assert np.isnan(out_tight).all()
    # invalid tolerance rejected
    with pytest.raises(ValueError, match="max_discarded_weight_mass"):
        _op("ts_fractional_difference").calculate(
            _frame(x), fd=0.4, cutoff=20, max_discarded_weight_mass=1.5
        )


def test_fractional_difference_discarded_weight_mass_output():
    x = np.linspace(1.0, 80.0, 80)
    # diagnostic op exposes the discarded mass (constant per fd/cutoff)
    out_neg = _calc("ts_fractional_difference_discarded_weight_mass", x, fd=-0.4, cutoff=20)
    assert out_neg[-1] == pytest.approx(1.0)
    out_pos = _calc("ts_fractional_difference_discarded_weight_mass", x, fd=0.4, cutoff=20)
    assert out_pos[-1] == pytest.approx(_fd_discarded_weight_mass(0.4, 20), abs=1e-9)
    assert 0.0 < out_pos[-1] < 1.0
    # the discarded mass drops as the cutoff grows (for d > 0)
    small = _fd_discarded_weight_mass(0.4, 20)
    large = _fd_discarded_weight_mass(0.4, 200)
    assert large < small
