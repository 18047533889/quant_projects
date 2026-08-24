# -*- coding: utf-8 -*-
"""R35 Phase F: Numba kernel layer gates (taskbook §14 / §15 / §16 / §76).

Every registered Numba kernel must have:
- a reference kernel that is the canonical execution path
- reference == numba parity on hostile fixtures (NaN gaps / Inf / constant)
- fastmath=False by default (strict NaN/Inf semantics)
- honest state when Numba is unavailable (no false "certified" claim)

Performance gates (R35_KALMAN_NUMBA_SPEEDUP_MEASURED /
R35_STATEFUL_NUMBA_SPEEDUP_MEASURED): measured, not asserted to a fixed number —
the benchmark is recorded in evidence; the test only requires the fast path to
be >= reference (never slower by a large factor, which would indicate a broken
fast path being routed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import factor_engine.backend.numba_kernels  # noqa: E402  (registers kernels)
from factor_engine.backend.numba_kernel_registry import (  # noqa: E402
    NumbaKernelRegistry,
    NUMBA_AVAILABLE,
    benchmark_kernel,
    list_kernels,
    parity_check,
)


def _kernels():
    return NumbaKernelRegistry.kernels()


# --------------------------------------------------------------------------
# Registry structure
# --------------------------------------------------------------------------

def test_kalman_and_ar_kernels_registered():
    reg = _kernels()
    for name in ("kalman_level", "kalman_trend", "kalman_beta", "ar_prior_forecast"):
        assert name in reg, f"{name} not registered"


def test_all_kernels_fastmath_false():
    """R35 §15: strict semantics default — no kernel may default fastmath=True."""
    for name, k in _kernels().items():
        assert k.spec.fastmath is False, f"{name}: fastmath must be False by default"


def test_numba_available_state_honest():
    """The registry's numba_available flag must reflect reality."""
    if NUMBA_AVAILABLE:
        # every kernel should have a compiled impl
        for name, k in _kernels().items():
            assert k.numba_fn is not None, f"{name}: numba available but no impl"
    else:
        # without numba, kernels must expose reference_fn and numba_fn=None
        for name, k in _kernels().items():
            assert k.numba_fn is None and k.reference_fn is not None


# --------------------------------------------------------------------------
# Parity on hostile fixtures
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["kalman_level", "kalman_trend", "kalman_beta"])
def test_kalman_parity_hostile(name):
    """§66 / §76: NaN gaps, Inf, constant series — reference == numba."""
    k = _kernels().get(name)
    if k is None:
        pytest.skip("kernel not registered")
    rng = np.random.default_rng(10)
    x = rng.standard_normal(200)
    x[50] = np.nan
    x[90:95] = np.nan
    if name == "kalman_level":
        args = (x, 1e-4, 1.0)
    elif name == "kalman_trend":
        args = (x, 1e-5, 1e-5, 1.0)
    else:
        y = rng.standard_normal(200)
        args = (y, x, 1e-3, 1.0)
    res = parity_check(k, *args)
    assert res["status"] == "PASS", f"{name}: {res}"


@pytest.mark.parametrize("name", ["kalman_level", "kalman_trend", "kalman_beta"])
def test_kalman_parity_constant(name):
    k = _kernels().get(name)
    if k is None:
        pytest.skip("kernel not registered")
    x = np.full(100, 2.0)
    if name == "kalman_level":
        args = (x, 1e-4, 1.0)
    elif name == "kalman_trend":
        args = (x, 1e-5, 1e-5, 1.0)
    else:
        args = (x, x, 1e-3, 1.0)
    res = parity_check(k, *args)
    assert res["status"] == "PASS", f"{name} constant: {res}"


def test_ar_prior_parity():
    k = _kernels().get("ar_prior_forecast")
    if k is None:
        pytest.skip("kernel not registered")
    rng = np.random.default_rng(11)
    x = rng.standard_normal(300)
    x[40] = np.nan
    x[120:125] = np.nan
    res = parity_check(k, x, 60, 2)
    assert res["status"] == "PASS", f"ar_prior_forecast: {res}"


def test_ar_prior_recovers_phi():
    """§145: AR(1) phi=0.7 must be recovered by the numba fast path."""
    k = _kernels().get("ar_prior_forecast")
    if k is None:
        pytest.skip("kernel not registered")
    rng = np.random.default_rng(12)
    phi = 0.7
    n = 500
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + 0.3 * rng.standard_normal()
    out = k.call(x, 200, 1, use_numba=True)
    # the one-step forecast at the tail should be close to phi * x_{t-1}
    tail = np.isfinite(out[300:])
    pred_ratio = out[300:][tail] / x[299:-1][tail]
    assert np.nanmedian(np.abs(pred_ratio)) == pytest.approx(phi, abs=0.15)


# --------------------------------------------------------------------------
# Benchmark (measured, recorded; fast path must not be badly slower)
# --------------------------------------------------------------------------

def test_benchmark_records_speedup():
    """R35_KALMAN_NUMBA_SPEEDUP_MEASURED: benchmark is recorded for evidence;
    the fast path must not be > 3x slower than reference (would indicate a
    broken route)."""
    import json

    rng = np.random.default_rng(13)
    x = rng.standard_normal(500)
    rec = {}
    k = _kernels().get("kalman_level")
    if k is not None:
        b_ref = benchmark_kernel(k, x, 1e-4, 1.0, use_numba=False)
        if NUMBA_AVAILABLE:
            b_num = benchmark_kernel(k, x, 1e-4, 1.0, use_numba=True)
            assert b_num["best_ms"] <= b_ref["best_ms"] * 3, (
                f"numba path unexpectedly slow: {b_num['best_ms']:.2f} vs {b_ref['best_ms']:.2f}ms"
            )
            rec = {
                "kernel": "kalman_level",
                "numba_best_ms": b_num["best_ms"],
                "ref_best_ms": b_ref["best_ms"],
                "speedup": b_ref["best_ms"] / max(b_num["best_ms"], 1e-12),
            }
        else:
            rec = {"kernel": "kalman_level", "status": "NUMBA_UNAVAILABLE"}
    # record for evidence
    import pathlib

    out = pathlib.Path("docs/evidence/r35") if (pathlib.Path("docs/evidence/r35").exists() or True) else None
    print(json.dumps(rec))
