# -*- coding: utf-8 -*-
"""R50 parameter-injectivity audit — focused mutation tests.

The R50 matrix flagged 421 operators ``PARAMETER_NOT_WIRED``.  A robust
re-scan on a non-degenerate panel (200+ rows, regime-switching series, proper
per-panel fixtures, 3 clearly-distinct valid values respecting each
ParamSpec domain) showed the overwhelming majority are FALSE POSITIVES from
the tiny synthetic probe: their searchable parameters genuinely change the
output.  Only a handful are genuinely dead (rank-scale-invariant knobs whose
scale a downstream rank normalises away).

This module pins the real outcomes so a regression (a param dropped, or a
rank-removal that would make a knob injective) is caught.

Categories documented here:
  * FIXTURE_INSENSITIVE — param active but the original probe panel could not
    exercise it; verified active on a proper fixture.
  * CONDITIONALLY_ACTIVE — param only matters under a non-default output;
    ParamSpec.active_when governs it.
  * TRUE_DEAD / rank-scale-invariant — param feeds the math but is washed out
    by a downstream normalisation (not a code bug, no fake wiring).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _op(name: str):
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get(name, "pandas_numpy") or OperatorRegistry.get(name)
    assert op is not None, f"{name} not registered"
    return op


def _hash(out) -> bytes:
    arr = np.asarray(out, dtype=float)
    finite = np.nan_to_num(arr, nan=-1e30, posinf=1e30, neginf=-1e30)
    return np.ascontiguousarray(finite).tobytes()


def _mk(n=200, seed=0, col="x"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    trend = np.concatenate([np.linspace(0, 40, n // 2), np.linspace(40, -25, n - n // 2)])
    return pd.DataFrame({col: trend + rng.normal(0, 4, n)}, index=idx, dtype=float)


@pytest.fixture(scope="module", autouse=True)
def _bootstrap():
    from factor_engine.cleaned_operators import load_all

    load_all()


# --------------------------------------------------------------------------- #
# FIXTURE_INSENSITIVE — verified ACTIVE on a proper fixture
# --------------------------------------------------------------------------- #
def _rv_frame(n=400, seed=1):
    """A positive realized-variance panel (the HAR family's required input)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    rv = np.abs(rng.normal(0.01, 0.008, n)) + 0.001
    return pd.DataFrame({"x": rv}, index=idx, dtype=float)


@pytest.mark.parametrize(
    "canonical,panel,base,probe",
    [
        # rolling window genuinely re-fits the model
        ("ts_ar_forecast", ["x"], dict(window=20), ("window", 60)),
        ("ts_generalized_hurst_exponent", ["x"], dict(window=60, q=1.0), ("q", 0.5)),
        ("ts_spectral_lowpass_trailing", ["x"], dict(window=64, cutoff_freq=5), ("cutoff_freq", 20)),
        ("ts_bessel_lowpass_causal", ["x"], dict(order=4, cutoff=0.05), ("cutoff", 0.3)),
        ("ts_fir_lowpass_causal", ["x"], dict(ntaps=21, cutoff=0.1, window="hamming"), ("cutoff", 0.4)),
        ("ts_rolling_sr_gaussian_mean_shift_score", ["x"],
         dict(window=60, baseline_window=30), ("baseline_window", 50)),
        ("ts_bds_statistic", ["x"], dict(window=60, embedding_dim=2), ("embedding_dim", 5)),
        ("HMA", ["x"], dict(window=15, rounding="floor"), ("rounding", "round")),
    ],
)
def test_window_param_is_active(canonical, panel, base, probe):
    op = _op(canonical)
    x = _mk(col=panel[0])
    base_out = op.calculate(x, **base)
    base_hash = _hash(base_out)
    assert np.isfinite(np.asarray(base_out, dtype=float)).any(), f"{canonical} all-NaN fixture"
    pname, pval = probe
    out = op.calculate(x, **{**base, pname: pval})
    assert _hash(out) != base_hash, f"{canonical}.{pname} did not change output"


def test_har_rv_window_is_active():
    """``ts_har_rv_*`` forecast window re-fits the HAR model over realized
    variance.  The generic price fixture is all-NaN for it (HAR requires an RV
    input); on a positive-RV panel the window genuinely changes the fit."""
    for c, base, probe in [
        ("ts_har_rv_next_vol_forecast", dict(window=30), ("window", 90)),
        ("ts_har_rv_forecast_error_z", dict(window=30), ("window", 90)),
        ("ts_har_rv_next_var_forecast", dict(window=30), ("window", 90)),
    ]:
        op = _op(c)
        x = _rv_frame()
        base_out = op.calculate(x, **base)
        base_hash = _hash(base_out)
        # HAR's double sample-coverage gate can leave an all-NaN column on a
        # short fixture; the window still re-fits the design, so a window change
        # must alter the output (NaN mask and/or values).
        pname, pval = probe
        out = op.calculate(x, **{**base, pname: pval})
        assert _hash(out) != base_hash, f"{c}.{pname} did not change output"


def test_expectile_q_is_active():
    """ts_expectile_regression_* / ts_quantile_regression_* ``q`` genuinely re-
    weights the regression — the tiny probe's single-panel fixture could not
    exercise it (misaligned multi-input), but on an aligned 2-panel it does."""
    idx = pd.date_range("2023-01-02", periods=200, freq="B")
    rng = np.random.default_rng(1)
    y = pd.DataFrame({"A": np.cumsum(rng.normal(0, 1, 200)) + 100}, index=idx, dtype=float)
    x = pd.DataFrame({"A": np.cumsum(rng.normal(0, 1, 200)) * 0.5 + 50}, index=idx, dtype=float)
    for c in (
        "ts_expectile_regression_coeff",
        "ts_expectile_regression_coeff_prior",
        "ts_expectile_regression_forecast_error",
        "ts_expectile_regression_resid",
        "ts_quantile_regression_coeff_prior",
    ):
        op = _op(c)
        base = _hash(op.calculate(y, x, window=60, q=0.5))
        assert _hash(op.calculate(y, x, window=60, q=0.1)) != base, f"{c}.q dead"


def test_hvg_min_coverage_fraction_active():
    """ts_hvg_* ``min_coverage_fraction`` gates a NaN-gapped window to NaN.  A
    fully-finite probe panel cannot exercise the gate; a gapped one can."""
    idx = pd.date_range("2023-01-02", periods=200, freq="B")
    rng = np.random.default_rng(9)
    s = np.cumsum(rng.normal(0, 1, 200)) + 100
    f = pd.DataFrame({"A": s}, index=idx, dtype=float)
    f.iloc[100:120, 0] = np.nan  # coverage gap
    for c in (
        "ts_hvg_assortativity",
        "ts_hvg_clustering_coefficient",
        "ts_hvg_degree_entropy",
        "ts_hvg_forward_backward_asymmetry",
        "ts_hvg_motif_entropy",
    ):
        op = _op(c)
        base = op.calculate(f, window=60, min_periods=10, min_coverage_fraction=0.5)
        base_hash = _hash(base)
        loose = op.calculate(f, window=60, min_periods=10, min_coverage_fraction=0.0)
        assert _hash(loose) != base_hash, f"{c}.min_coverage_fraction dead"


def test_state_episode_age_capped_max_cap_active():
    """``max_cap`` caps the episode age — active when episodes actually occur."""
    idx = pd.date_range("2023-01-02", periods=200, freq="B")
    state = np.zeros(200)
    state[10:40] = 1
    state[40:80] = 2
    state[80:100] = 0
    state[100:160] = 3
    state[160:180] = 1
    st = pd.DataFrame({"state": state}, index=idx, dtype=float)
    op = _op("state_episode_age_capped")
    base = op.calculate(st, window=60, max_cap=5)
    assert np.isfinite(np.asarray(base, dtype=float)).any()
    assert _hash(op.calculate(st, window=60, max_cap=30)) != _hash(base)


def test_cond_param_active_under_nondefault_output():
    """CONDITIONALLY_ACTIVE: ``signal_smooth`` (FisherTransform) and ``factor``
    (QQE) only enter under the non-default ``signal``/``trend`` output paths.
    The default ``value``/``line`` output does not depend on them, so a probe at
    the default output falsely reports them DEAD."""
    idx = pd.date_range("2023-01-02", periods=160, freq="B")
    rng = np.random.default_rng(3)
    s = np.cumsum(rng.normal(0, 1, 160)) + 100
    hi = pd.DataFrame({"A": s + 1}, index=idx, dtype=float)
    lo = pd.DataFrame({"A": s - 1}, index=idx, dtype=float)

    ft = _op("FisherTransform")
    base = _hash(ft.calculate(hi, lo, window=10, smooth=0.5, signal_smooth=0.5, output="signal"))
    assert _hash(ft.calculate(hi, lo, window=10, smooth=0.5, signal_smooth=0.2, output="signal")) != base

    qqe = _op("QQE")
    base = _hash(qqe.calculate(hi, length=14, smooth=5, factor=4.236, output="trend"))
    assert _hash(qqe.calculate(hi, length=14, smooth=5, factor=2.0, output="trend")) != base


# --------------------------------------------------------------------------- #
# TRUE_DEAD / rank-scale-invariant — documented, no fake wiring
# --------------------------------------------------------------------------- #
def test_keltner_compression_multiplier_is_rank_scale_invariant():
    """``multiplier`` feeds the Keltner band width, but ``keltner_compression``
    then ranks that width, so any positive multiplier is a monotone rescale the
    rank washes out.  This is a genuine non-injective knob on THIS transform —
    not a dead code path and not a bug — so it is not fake-wired.  It is pinned
    so a future rank-removal that would make it injective is caught."""
    n = 60
    vals = np.linspace(10.0, 15.0, n) + np.sin(np.arange(n) * 0.9) * 0.4
    close = pd.DataFrame({"A": vals}, dtype=float)
    high = close + 0.4
    low = close - 0.4
    op = _op("keltner_compression")
    base = op.calculate(high, low, close, ema_window=10, atr_window=10, multiplier=2.0, score_window=20)
    assert np.isfinite(base.to_numpy(dtype=float)).any()
    for m in (0.5, 5.0, 20.0):
        out = op.calculate(high, low, close, ema_window=10, atr_window=10, multiplier=m, score_window=20)
        np.testing.assert_array_equal(out.to_numpy(dtype=float), base.to_numpy(dtype=float),
                                      err_msg=f"multiplier={m} changed output")
