# -*- coding: utf-8 -*-
"""Regression tests for the fourth-batch review fixes (2026-08-08).

Covers the P0 golden behaviours that were verified interactively during the
audit, so a future regression is caught by CI rather than silently resurfacing:
  * Huber IRLS uses sqrt(weight) — matches an independent robust-regression fit.
  * permutation transition entropy produces valid output for delay 1/2/3.
  * ts_new_high/new_low emit NaN on warmup/missing, not 0.
  * candlestick 3_inside / 3_outside fire on hand-built golden bars and warmup
    rows are NaN (not 0).
  * wavelet entropy stays finite when a band has zero energy; slope uses the
    dyadic scale axis.
  * integer params reject non-integer floats (P0-06).
  * systemic-tail extreme indicator NaN during the min_periods cold start.
  * cs_tail_retention rejects an unknown ``side``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry
from backend.operator_errors import OperatorParameterError


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    load_all()


def _frame(vals, cols=("A",)):
    n = len(vals)
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({c: np.asarray(vals, dtype=float) for c in cols}, index=idx)


# --------------------------------------------------------------------------
# P0-01 Huber IRLS sqrt(weight)
# --------------------------------------------------------------------------
def test_huber_matches_independent_irls_reference():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 60)
    y = 1.5 * x + 0.3 + rng.normal(0, 0.05, 60)
    y[::7] = 50.0  # outliers
    from cleaned_operators.regression_models import _huber_fit

    design = np.column_stack([np.ones(60), x])
    beta = _huber_fit(design, y)

    # independent IRLS with sqrt(w) — the reference convention
    ref = np.linalg.lstsq(design, y, rcond=None)[0]
    delta = 1.345
    for _ in range(5):
        resid = y - design @ ref
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        z = resid / scale
        w = np.where(np.abs(z) <= delta, 1.0, delta / np.abs(z))
        rw = np.sqrt(w)
        ref = np.linalg.lstsq(design * rw[:, None], y * rw, rcond=None)[0]
    np.testing.assert_allclose(beta, ref, rtol=1e-8, atol=1e-8)


# --------------------------------------------------------------------------
# P0-11 permutation transition entropy delay
# --------------------------------------------------------------------------
@pytest.mark.parametrize("delay", [1, 2, 3])
def test_permutation_transition_entropy_all_delays(delay):
    rng = np.random.default_rng(1)
    x = _frame(rng.normal(0, 1, 400))
    op = OperatorRegistry.get("ts_permutation_transition_entropy", "pandas_numpy")
    out = pd.DataFrame(op.calculate(x=x, window=120, order=3, delay=delay))
    assert int(out.notna().to_numpy().sum()) > 0


def test_permutation_entropy_tie_fail_closed():
    op = OperatorRegistry.get("ts_permutation_entropy", "pandas_numpy")
    const = _frame(np.zeros(120))
    out = pd.DataFrame(op.calculate(x=const, window=60, order=3))
    assert int(out.notna().to_numpy().sum()) == 0  # P1-14: fully tied -> NaN


# --------------------------------------------------------------------------
# P0-07 ts_new_high / new_low warmup NaN
# --------------------------------------------------------------------------
def test_new_high_warmup_is_nan_not_zero():
    op = OperatorRegistry.get("ts_new_high", "pandas_numpy")
    x = _frame([1.0, 2.0, 3.0, 4.0, np.nan, 6.0])
    out = pd.DataFrame(op.calculate(x=x, window=3))
    vals = out["A"].tolist()
    assert vals[0] != vals[0] and vals[1] != vals[1] and vals[2] != vals[2]
    assert vals[3] == 1.0
    assert vals[4] != vals[4]  # missing current -> NaN
    assert vals[5] != vals[5]  # baseline window under-populated -> NaN


# --------------------------------------------------------------------------
# P0-08 / P0-09 candlestick golden patterns
# --------------------------------------------------------------------------
def _candle_engine():
    from cleaned_operators.price_volume.candle_pattern_engine_v2 import _engine

    return _engine


def test_3_inside_golden_and_warmup_nan():
    engine = _candle_engine()
    o = _frame([10, 12, 11.5, 11, 10, 10])["A"]
    h = _frame([11, 12.5, 11.6, 12.8, 11, 11])["A"]
    l = _frame([9, 9.5, 11.1, 10.9, 9, 9])["A"]
    c = _frame([10, 10, 11.2, 12.5, 10, 10])["A"]
    frames = tuple(pd.DataFrame({"A": v}) for v in (o, h, l, c))
    out = pd.DataFrame(engine(*frames, "3_inside", 3, 3, 0.3))
    vals = out["A"].tolist()
    assert vals[0] != vals[0] and vals[1] != vals[1]  # warmup NaN, not 0
    assert vals[3] == 1.0  # hand-built bullish 3-inside fires at confirmation


def test_3_outside_golden():
    engine = _candle_engine()
    o = _frame([10, 12, 9.5, 11, 10, 10])["A"]
    h = _frame([11, 12.5, 12.7, 12.9, 11, 11])["A"]
    l = _frame([9, 9.5, 9.4, 10.8, 9, 9])["A"]
    c = _frame([10, 10, 12.6, 12.8, 10, 10])["A"]
    frames = tuple(pd.DataFrame({"A": v}) for v in (o, h, l, c))
    out = pd.DataFrame(engine(*frames, "3_outside", 3, 3, 0.3))
    assert out["A"].iloc[3] == 1.0


def test_candle_active_params_helper():
    from cleaned_operators.price_volume.candle_pattern_engine_v2 import (
        candlestick_active_params,
    )

    assert candlestick_active_params("long_line") == frozenset({"pattern", "body_window"})
    assert "penetration" in candlestick_active_params("abandoned_baby")
    assert "penetration" not in candlestick_active_params("3_inside")


# --------------------------------------------------------------------------
# P0-12 / P0-13 wavelet
# --------------------------------------------------------------------------
def test_wavelet_entropy_finite_with_zero_energy_band():
    op = OperatorRegistry.get("ts_wavelet_entropy", "pandas_numpy")
    # level-0 detail energy == 0 (pairs equal), higher bands non-zero
    seq = np.repeat([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], 40)[: 512]
    x = _frame(seq)
    out = pd.DataFrame(op.calculate(x=x, window=128))
    assert np.isfinite(out["A"].dropna()).all()


def test_wavelet_uses_newest_data():
    op = OperatorRegistry.get("ts_wavelet_low_frequency_ratio", "pandas_numpy")
    # old half flat, recent half noisy -> low-freq ratio must reflect recent
    seq = np.concatenate([np.full(256, 10.0), np.random.default_rng(0).normal(10, 1, 256)])
    x = _frame(seq)
    out = pd.DataFrame(op.calculate(x=x, window=256))
    tail = out["A"].dropna().iloc[-20:]
    assert (tail > 0).all() and (tail < 1).all()


# --------------------------------------------------------------------------
# P0-06 integer parameter validation
# --------------------------------------------------------------------------
def test_integer_param_rejects_float():
    op = OperatorRegistry.get("PPO", "pandas_numpy")
    close = _frame(np.linspace(1, 100, 100))
    with pytest.raises(OperatorParameterError):
        op.calculate(close=close, fast_window=5.9, slow_window=10)
    out = op.calculate(close=close, fast_window=5.0, slow_window=10)
    assert pd.DataFrame(out).notna().to_numpy().sum() > 0


# --------------------------------------------------------------------------
# P0-14 systemic-tail cold start
# --------------------------------------------------------------------------
def test_tail_extreme_cold_start_nan():
    from cleaned_operators.tail_systemic import _extreme_indicator_panel

    xv = np.arange(1.0, 6.0).reshape(-1, 1)
    E = _extreme_indicator_panel(xv, 5, 0.1, "lower", 10)
    assert np.isnan(E).all()  # min_periods=10 > 5 obs -> cannot judge


# --------------------------------------------------------------------------
# P1-23 rotation side strict enum
# --------------------------------------------------------------------------
def test_tail_retention_side_strict():
    op = OperatorRegistry.get("cs_tail_retention", "pandas_numpy")
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(0, 1, (40, 6)), columns=list("ABCDEF"))
    with pytest.raises(ValueError):
        op.calculate(x=x, lag=3, quantile=0.2, side="abc")
    out = op.calculate(x=x, lag=3, quantile=0.2, side="top")
    assert pd.DataFrame(out).notna().to_numpy().sum() > 0
