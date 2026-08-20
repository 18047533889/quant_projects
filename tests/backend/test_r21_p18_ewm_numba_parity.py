# -*- coding: utf-8 -*-
"""R21-P18-EWM-NUMBA-KERNEL: Regression tests for Numba EWM pairwise corr/cov.

Tests parity between Numba kernel, reference implementation, and pandas ewm.
Covers: basic parity, NaN holes, Inf handling, edge cases, performance.
"""
import math

import numpy as np
import polars as pl
import pytest

from backend.numba_kernel_registry import NUMBA_AVAILABLE, NumbaKernelRegistry

# Force registration of EWM kernels
import backend.numba_kernels.ewm_pairwise  # noqa: F401


def _get_ewm_kernel(kernel_name: str):
    """Get the registered EWM kernel from registry."""
    kernel = NumbaKernelRegistry.get(kernel_name)
    assert kernel is not None, f"Kernel {kernel_name} not registered"
    return kernel


def _ewm_pandas_reference(x_values, y_values, window, *, corr, bias=False):
    """Pandas ewm reference for parity comparison."""
    import numpy as np
    import pandas as pd

    def _finite(value):
        if value is None:
            return math.nan
        try:
            value = float(value)
        except (TypeError, ValueError):
            return math.nan
        return value if math.isfinite(value) else math.nan

    xs = pd.Series([_finite(v) for v in x_values], dtype=float)
    ys = pd.Series([_finite(v) for v in y_values], dtype=float)
    pair = xs.notna() & ys.notna()
    xs = xs.where(pair, np.nan)
    ys = ys.where(pair, np.nan)
    ewm = xs.ewm(span=window, adjust=False, ignore_na=False, min_periods=2)
    out = ewm.corr(ys) if corr else ewm.cov(ys, bias=bias)
    return [None if pd.isna(v) else float(v) for v in out.tolist()]


def _assert_ewm(actual, expected, *, rel_tol=1e-10, abs_tol=1e-12):
    """Assert EWM results match with tolerance."""
    assert len(actual) == len(expected)
    for i, (got, want) in enumerate(zip(actual, expected)):
        if want is None:
            # Handle both None and NaN as "null"
            assert got is None or (isinstance(got, (float, np.floating)) and np.isnan(got)), \
                f"Index {i}: expected None, got {got!r}"
        else:
            assert got is not None, f"Index {i}: expected {want!r}, got None"
            # Convert NaN to None for comparison
            if isinstance(got, (float, np.floating)) and np.isnan(got):
                got = None
            assert got is not None, f"Index {i}: expected {want!r}, got None"
            assert got == pytest.approx(want, rel=rel_tol, abs=abs_tol), \
                f"Index {i}: got {got}, expected {want}"


# --------------------------------------------------------------------------
# Test 1: Parity between reference and numba kernel
# --------------------------------------------------------------------------

def test_ewm_parity_reference_vs_numba():
    """Reference and numba kernels produce identical output."""
    if not NUMBA_AVAILABLE:
        pytest.skip("Numba not available")

    kernel = _get_ewm_kernel("ts_ewm_corr")
    assert kernel.numba_fn is not None

    rng = np.random.default_rng(42)
    x = rng.normal(0, 2, 20)
    y = 0.5 * x + rng.normal(0, 1, 20)
    alpha = 2.0 / 7.0  # span=6

    ref_out = kernel.reference_fn(x, y, alpha, corr=True)
    numba_out = kernel.numba_fn(x, y, alpha, corr=True)

    assert len(ref_out) == len(numba_out)
    for i, (ref, num) in enumerate(zip(ref_out, numba_out)):
        if np.isnan(ref):
            assert np.isnan(num), f"Index {i}: ref NaN, num not NaN"
        else:
            assert num == pytest.approx(ref, rel=1e-12, abs=1e-15), \
                f"Index {i}: ref={ref}, num={num}"


import numpy as np


# --------------------------------------------------------------------------
# Test 2: Parity with pandas ewm on random series
# --------------------------------------------------------------------------

def test_ewm_parity_with_pandas_on_random_series():
    """Numba kernel matches pandas ewm.corr/cov on random data."""
    if not NUMBA_AVAILABLE:
        pytest.skip("Numba not available")

    kernel = _get_ewm_kernel("ts_ewm_corr")
    rng = np.random.default_rng(11)
    xs = rng.normal(0, 2, 12).tolist()
    ys = (0.5 * np.asarray(xs) + rng.normal(0, 1, 12)).tolist()
    alpha = 2.0 / 7.0  # span=6

    # Test corr
    numba_corr = kernel.numba_fn(np.array(xs), np.array(ys), alpha, corr=True)
    pandas_corr = _ewm_pandas_reference(xs, ys, 6, corr=True)
    _assert_ewm(numba_corr, pandas_corr)

    # Test cov
    kernel_cov = _get_ewm_kernel("ts_ewm_cov")
    numba_cov = kernel_cov.numba_fn(np.array(xs), np.array(ys), alpha, corr=False)
    pandas_cov = _ewm_pandas_reference(xs, ys, 6, corr=False)
    _assert_ewm(numba_cov, pandas_cov)


# --------------------------------------------------------------------------
# Test 3: NaN holes - invalid pairs don't update state
# --------------------------------------------------------------------------

def test_ewm_nan_holes_match_pandas():
    """Rows with invalid pairs are skipped, matching pandas behavior."""
    if not NUMBA_AVAILABLE:
        pytest.skip("Numba not available")

    kernel = _get_ewm_kernel("ts_ewm_corr")
    rng = np.random.default_rng(5)
    xs = rng.normal(0, 2, 12).tolist()
    ys = (0.5 * np.asarray(xs) + rng.normal(0, 1, 12)).tolist()

    # Create holes
    holed_x = [None if i in (4, 9) else v for i, v in enumerate(xs)]
    holed_y = [None if i == 7 else v for i, v in enumerate(ys)]

    alpha = 2.0 / 6.0  # span=5
    numba_corr = kernel.numba_fn(
        np.array([x if x is not None else np.nan for x in holed_x]),
        np.array([y if y is not None else np.nan for y in holed_y]),
        alpha, corr=True
    )
    pandas_corr = _ewm_pandas_reference(holed_x, holed_y, 5, corr=True)
    _assert_ewm(numba_corr, pandas_corr)


# --------------------------------------------------------------------------
# Test 4: Inf handling - Inf pre-masked to NaN
# --------------------------------------------------------------------------

def test_ewm_inf_pre_masked_to_nan():
    """Inf values are treated as invalid and don't propagate."""
    if not NUMBA_AVAILABLE:
        pytest.skip("Numba not available")

    kernel = _get_ewm_kernel("ts_ewm_corr")
    x = np.array([1.0, np.inf, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    alpha = 2.0 / 5.0  # span=4

    numba_corr = kernel.numba_fn(x, y, alpha, corr=True)
    # Row 1 is dropped (Inf), so warmup until row 2, then perfect correlation
    expected = [None, None, 1.0, 1.0, 1.0]
    _assert_ewm(numba_corr, expected)


# --------------------------------------------------------------------------
# Test 5: Edge cases - constant series, exact linear correlation
# --------------------------------------------------------------------------

def test_ewm_constant_x_corr_null_cov_zero():
    """Zero-variance stream: corr is null, cov is zero."""
    if not NUMBA_AVAILABLE:
        pytest.skip("Numba not available")

    kernel_corr = _get_ewm_kernel("ts_ewm_corr")
    kernel_cov = _get_ewm_kernel("ts_ewm_cov")

    x = np.array([2.0, 2.0, 2.0, 2.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    alpha = 2.0 / 5.0  # span=4

    numba_corr = kernel_corr.numba_fn(x, y, alpha, corr=True)
    numba_cov = kernel_cov.numba_fn(x, y, alpha, corr=False)

    # corr should be None (zero variance)
    for i, val in enumerate(numba_corr):
        assert val is None or np.isnan(val), f"Index {i}: expected None, got {val}"

    # cov should be 0 (constant x)
    expected_cov = [None, 0.0, 0.0, 0.0]
    _assert_ewm(numba_cov, expected_cov, abs_tol=1e-12)


def test_ewm_exact_linear_correlation():
    """Perfect linear correlation gives corr = +/-1."""
    if not NUMBA_AVAILABLE:
        pytest.skip("Numba not available")

    kernel = _get_ewm_kernel("ts_ewm_corr")

    x = np.array([1.0, 2.0, 3.0, 4.0])
    y_pos = np.array([2.0, 4.0, 6.0, 8.0])
    y_neg = np.array([-2.0, -4.0, -6.0, -8.0])
    alpha = 2.0 / 5.0  # span=4

    # Positive correlation
    corr_pos = kernel.numba_fn(x, y_pos, alpha, corr=True)
    expected_pos = [None, 1.0, 1.0, 1.0]
    _assert_ewm(corr_pos, expected_pos)

    # Negative correlation
    corr_neg = kernel.numba_fn(x, y_neg, alpha, corr=True)
    expected_neg = [None, -1.0, -1.0, -1.0]
    _assert_ewm(corr_neg, expected_neg)


def test_ewm_warmup_requires_two_valid_pairs():
    """First output is always None (need 2 valid pairs for bias correction)."""
    if not NUMBA_AVAILABLE:
        pytest.skip("Numba not available")

    kernel = _get_ewm_kernel("ts_ewm_corr")
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 7.0, 8.0, 11.0])
    alpha = 2.0 / 4.0  # span=3

    numba_corr = kernel.numba_fn(x, y, alpha, corr=True)
    assert numba_corr[0] is None or np.isnan(numba_corr[0])


# --------------------------------------------------------------------------
# Test 6: Performance benchmark (informational, not assertion)
# --------------------------------------------------------------------------

def test_ewm_numba_performance_benchmark():
    """Benchmark numba vs reference (informational, no assertion)."""
    if not NUMBA_AVAILABLE:
        pytest.skip("Numba not available")

    import time

    kernel = _get_ewm_kernel("ts_ewm_corr")
    rng = np.random.default_rng(99)
    x = rng.normal(0, 1, 10000)
    y = rng.normal(0, 1, 10000)
    alpha = 0.1

    # Warmup
    kernel.reference_fn(x, y, alpha, corr=True)
    kernel.numba_fn(x, y, alpha, corr=True)

    # Benchmark reference
    t0 = time.perf_counter()
    for _ in range(10):
        kernel.reference_fn(x, y, alpha, corr=True)
    ref_time = (time.perf_counter() - t0) / 10

    # Benchmark numba
    t0 = time.perf_counter()
    for _ in range(10):
        kernel.numba_fn(x, y, alpha, corr=True)
    numba_time = (time.perf_counter() - t0) / 10

    speedup = ref_time / numba_time if numba_time > 0 else float('inf')
    print(f"\nEWM Performance: reference={ref_time*1000:.2f}ms, "
          f"numba={numba_time*1000:.2f}ms, speedup={speedup:.2f}x")

    # Just ensure it ran without error (no assertion on speedup)
    assert speedup > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
