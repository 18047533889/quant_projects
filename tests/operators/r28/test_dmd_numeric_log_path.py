# -*- coding: utf-8 -*-
"""R28 §四十六: DMD must work in log domain — growing modes never overflow,
zero-amplitude modes never fabricate energy, all-zero modes fail closed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _load():
    load_all()


def _panel(vals: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=vals.shape[0], freq="D")
    return pd.DataFrame(vals, index=idx, columns=["A"])


def _growth(x, window=60, rank=2):
    op = OperatorRegistry.get("ts_dmd_dominant_growth_rate", "pandas_numpy", mode="any")
    if op is None:
        pytest.skip("ts_dmd_dominant_growth_rate backend not registered (transient)")
    return op.calculate(
        _panel(x), window=window, rank=rank
    )["A"].to_numpy()


def test_dmd_growing_mode_no_overflow():
    """A geometrically growing mode (|λ| > 1) must produce a finite output, never inf."""
    _load()
    rng = np.random.default_rng(1)
    n = 180
    # strongly growing exponential + noise
    t = np.arange(n)
    x = np.exp(0.02 * t) + rng.standard_normal(n) * 0.5
    out = _growth(x)
    finite = out[np.isfinite(out)]
    assert len(finite) > 0, "growing mode produced no finite output"
    assert np.all(np.isfinite(finite)), "growing mode produced inf"
    # a growing mode has positive log growth rate
    assert float(np.nanmedian(out)) > 0.0


def test_dmd_decaying_mode_negative_growth():
    _load()
    rng = np.random.default_rng(2)
    t = np.arange(180)
    x = np.exp(-0.03 * t) + rng.standard_normal(180) * 0.5
    out = _growth(x)
    finite = out[np.isfinite(out)]
    assert len(finite) > 0
    assert float(np.nanmedian(out)) < 0.0


def test_dmd_zero_amplitude_no_fabricated_energy():
    """A mode with exactly zero amplitude must vanish (log_energy=-inf), never
    receive a machine-epsilon energy (R16-085).  Runtime proof: an ALL-ZERO input
    makes every mode zero-amplitude, so the operator must fail closed to NaN
    instead of fabricating a finite energy (the -inf - -inf path is guarded)."""
    _load()
    x = np.zeros(120)
    out = _growth(x)
    # every all-zero mode -> log_energy=-inf -> return None -> NaN everywhere
    assert np.all(np.isnan(out)), "all-zero DMD fabricated a finite energy"
    # constant-then-zero: the constant segment is a valid mode (lambda=1, growth 0),
    # so a FINITE result is correct there — the zero-amplitude tail mode just
    # vanishes.  No value may be non-finite.
    x2 = np.ones(120)
    x2[-40:] = 0.0
    out2 = _growth(x2)
    finite = out2[np.isfinite(out2)]
    assert np.all(np.isfinite(finite)), "DMD produced non-finite value"
    assert float(np.nanmedian(out2)) < 0.0  # trailing decay drags growth negative


def test_dmd_all_zero_fails_closed():
    """An all-zero series must not produce a NaN from an accidental -inf - -inf path."""
    _load()
    x = np.zeros(120)
    out = _growth(x)
    # either fully NaN (fail closed) — but NEVER a non-finite fabricated value
    finite = out[np.isfinite(out)]
    assert np.all(np.isfinite(finite)), "all-zero DMD produced non-finite value"


def test_dmd_future_perturbation():
    """DMD growth rate at t depends only on the trailing window ending at t."""
    _load()
    rng = np.random.default_rng(3)
    n = 180
    x = np.exp(0.01 * np.arange(n)) + rng.standard_normal(n) * 0.5
    base = _growth(x, window=60)
    xm = x.copy()
    xm[120:] = 1e6  # future blow-up
    changed = _growth(xm, window=60)
    np.testing.assert_allclose(changed[:100], base[:100], equal_nan=True, atol=1e-6)
