# -*- coding: utf-8 -*-
"""R57: the vectorized spectral kernels must stay equal to the scalar ones.

``spectral._periodogram_batch`` replaces the per-window ``polyfit`` + ``rfft``
with a closed-form detrend and one batched transform.  It is only acceptable if
it reproduces ``_periodogram`` row by row, so this module keeps a VERBATIM copy
of the scalar kernel and the two sliding-window loops and pins the contracts:

* identical finite values (to floating-point tolerance) and identical NaN mask;
* rows before ``w - 1`` stay NaN (no partial warmup);
* a window that is not fully finite emits NaN -- never zero-padded;
* entropy is a ratio in ``[0, 1]``; the dominant period is ``>= 2``.
"""
from __future__ import annotations

import numpy as np
import pytest

from factor_engine.cleaned_operators import spectral_ext
from factor_engine.cleaned_operators.spectral import _periodogram, _periodogram_batch

_EPS = 1e-12
_MIN_FINITE = 16


# ----------------------------------------------------------- scalar reference
def _periodogram_scalar(chunk: np.ndarray):
    n = chunk.size
    if n < _MIN_FINITE:
        return None
    v = chunk.astype(float)
    if not np.all(np.isfinite(v)):
        return None
    t = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(t, v, 1)
    resid = v - (slope * t + intercept)
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / (n - 1.0))) if n > 1 else np.ones(n)
    spectrum = np.fft.rfft(resid * hann)
    power = (np.abs(spectrum) ** 2) / float(n)
    i_max = n // 2
    p_pos = power[1 : i_max + 1]
    if p_pos.size == 0:
        return None
    return p_pos, i_max


def _entropy_scalar(x2d: np.ndarray, window: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            if r < w - 1:
                continue
            pg = _periodogram_scalar(col[r - w + 1 : r + 1])
            if pg is None:
                continue
            p, i_max = pg
            total = float(p.sum())
            if total <= _EPS or not np.isfinite(total):
                continue
            pn = p / total
            pn = pn[pn > 0.0]
            if pn.size < 2 or i_max < 2:
                continue
            out[r, c] = float(-np.sum(pn * np.log(pn)) / np.log(float(i_max)))
    return out


def _dominant_scalar(x2d: np.ndarray, window: int, min_peak_share: float) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    gate = float(min_peak_share)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            if r < w - 1:
                continue
            chunk = col[r - w + 1 : r + 1]
            pg = _periodogram_scalar(chunk)
            if pg is None:
                continue
            p, _ = pg
            total = float(p.sum())
            if total <= _EPS or not np.isfinite(total):
                continue
            if float(p.max()) / total < gate:
                continue
            j = int(np.argmax(p))
            period = float(chunk.size) / float(j + 1)
            if period < 2.0 or not np.isfinite(period):
                continue
            out[r, c] = period
    return out


def _panels() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260919)
    out: dict[str, np.ndarray] = {}
    base = np.abs(rng.normal(size=(220, 6)).cumsum(axis=0)) + 1.0
    out["clean"] = base
    gapped = base.copy()
    gapped[rng.random(size=gapped.shape) < 0.05] = np.nan
    out["gaps"] = gapped
    front = base.copy()
    front[:20, :] = np.nan
    out["gaps_front"] = front
    # a column that is entirely NaN, plus a flat (degenerate) column
    degenerate = base.copy()
    degenerate[:, 0] = np.nan
    degenerate[:, 1] = 5.0
    out["degenerate_columns"] = degenerate
    return out


@pytest.mark.parametrize("window", [16, 27, 60])
def test_entropy_matches_scalar_reference(window: int) -> None:
    for name, x in _panels().items():
        want = _entropy_scalar(x, window)
        got = spectral_ext._spectral_entropy_series(x, window)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(want), err_msg=name)
        fin = ~np.isnan(want)
        if fin.any():
            np.testing.assert_allclose(got[fin], want[fin], rtol=1e-9, atol=1e-12)
            assert got[fin].min() >= -1e-9, "entropy below 0"
            assert got[fin].max() <= 1.0 + 1e-9, "entropy above 1"
        # no partial warmup: everything above the first full window is untouched
        if x.shape[0] >= window:
            assert np.isnan(got[: window - 1]).all(), "partial warmup leaked"


@pytest.mark.parametrize("gate", [0.0, 0.1, 0.5])
def test_dominant_period_matches_scalar_reference(gate: float) -> None:
    window = 30
    for name, x in _panels().items():
        want = _dominant_scalar(x, window, gate)
        got = spectral_ext._dominant_cycle_period_series(x, window, gate)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(want), err_msg=name)
        fin = ~np.isnan(want)
        if fin.any():
            np.testing.assert_allclose(got[fin], want[fin], rtol=1e-9, atol=1e-12)
            assert got[fin].min() >= 2.0 - 1e-9, "period below 2 bars"


def test_periodogram_batch_matches_scalar_row_by_row() -> None:
    rng = np.random.default_rng(7)
    wins = rng.normal(size=(41, 48)).cumsum(axis=1)
    batch, i_max = _periodogram_batch(wins)
    assert i_max == 48 // 2
    for k in range(wins.shape[0]):
        want, want_i = _periodogram_scalar(wins[k])
        assert want_i == i_max
        np.testing.assert_allclose(batch[k], want, rtol=1e-9, atol=1e-12)


def test_periodogram_batch_nan_window_emits_nan() -> None:
    rng = np.random.default_rng(11)
    wins = rng.normal(size=(9, 40)).cumsum(axis=1)
    wins[3, :] = np.nan
    got, _ = _periodogram_batch(wins)
    assert np.isnan(got[3]).all(), "a non-finite window leaked a finite spectrum"
    for k in (0, 1, 2, 4, 8):
        assert np.isfinite(got[k]).all()


def test_batch_kernel_rejects_short_windows() -> None:
    assert _periodogram_batch(np.zeros((4, _MIN_FINITE - 1))) is None
    # the scalar kernel agrees: a 15-bar window has no spectrum
    assert _periodogram(np.zeros(_MIN_FINITE - 1)) is None
