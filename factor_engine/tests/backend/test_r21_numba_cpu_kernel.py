# -*- coding: utf-8 -*-
"""R21 §32 P10: Numba CPU kernel regression tests.

Tests verify:
- ExecutionKind.NUMBA_CPU_KERNEL is present in the unified enum
- 5 rolling kernels (ts_mean, ts_std, ts_sum, ts_min, ts_max) are registered
- Reference vs numba parity on hostile fixtures (NaN, Inf, -Inf, constant)
- Reference vs pandas parity for each kernel
- fastmath=False policy (strict NaN/Inf semantics)
- Numba available state honesty

Thread env vars: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.contracts import ExecutionKind
from factor_engine.backend.numba_kernel_registry import (
    NUMBA_AVAILABLE,
    NumbaKernelRegistry,
    parity_check,
)

# Ensure ts_rolling kernels are registered
import factor_engine.backend.numba_kernels.ts_rolling  # noqa: F401  (registers kernels)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kernels():
    return NumbaKernelRegistry.kernels()


# Hostile fixture: NaN gaps, Inf, -Inf, constant
_RNG = np.random.default_rng(99)
_HOSTILE_1D = np.concatenate([
    _RNG.standard_normal(50),
    [np.nan, np.nan],
    [np.inf],
    [-np.inf],
    [0.0, 0.0, 0.0, 0.0],  # constant region
    _RNG.standard_normal(20),
])
# Insert NaN gaps in the middle
_HOSTILE_1D[30] = np.nan
_HOSTILE_1D[40:45] = np.nan

_WINDOW = 5
_MIN_COUNT = 1


# ===========================================================================
# 1. Enum presence and registry structure
# ===========================================================================

class TestNumbaKernelRegistry:
    """Registry structure and contract tests."""

    def test_execution_kind_numba_cpu_kernel_exists(self):
        """NUMBA_CPU_KERNEL is present in the unified ExecutionKind enum."""
        assert hasattr(ExecutionKind, "NUMBA_CPU_KERNEL")
        assert ExecutionKind.NUMBA_CPU_KERNEL.value == "numba_cpu_kernel"

    def test_five_ts_rolling_kernels_registered(self):
        """All 5 core rolling kernels are registered in NumbaKernelRegistry."""
        reg = _kernels()
        for name in ("ts_sum", "ts_mean", "ts_min", "ts_max", "ts_std"):
            assert name in reg, f"{name} not registered"

    def test_all_ts_rolling_kernels_fastmath_false(self):
        """R35 §15: strict semantics default — no kernel may default fastmath=True."""
        for name in ("ts_sum", "ts_mean", "ts_min", "ts_max", "ts_std"):
            k = _kernels().get(name)
            if k is None:
                pytest.skip(f"{name} not registered")
            assert k.spec.fastmath is False, f"{name}: fastmath must be False by default"
            assert k.spec.nogil is True, f"{name}: nogil must be True"
            assert k.spec.canonical_family == "ts_rolling"


# ===========================================================================
# 2. Reference vs Numba parity on hostile fixtures
# ===========================================================================

class TestNumbaKernelParity:
    """Parity between reference and numba implementations."""

    @pytest.mark.parametrize("kernel_name", ["ts_sum", "ts_mean", "ts_min", "ts_max", "ts_std"])
    def test_parity_hostile_fixture(self, kernel_name):
        """Reference == numba on hostile fixture (NaN gaps, Inf, -Inf, constant)."""
        k = _kernels().get(kernel_name)
        if k is None:
            pytest.skip(f"{kernel_name} not registered")
        res = parity_check(k, _HOSTILE_1D, _WINDOW, _MIN_COUNT)
        assert res["status"] == "PASS", f"{kernel_name}: {res}"

    def test_std_single_element_is_nan(self):
        """ts_std with window=1 should return NaN (ddof=1 requires >=2 elements)."""
        k = _kernels().get("ts_std")
        if k is None:
            pytest.skip("ts_std not registered")
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ref = k.reference_fn(arr, 1, 1)
        assert all(np.isnan(ref)), f"window=1 should be all NaN with ddof=1: {ref}"

    def test_std_constant_region_zero(self):
        """ts_std on constant region should return 0.0 (not NaN)."""
        k = _kernels().get("ts_std")
        if k is None:
            pytest.skip("ts_std not registered")
        arr = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        ref = k.reference_fn(arr, 3, 1)
        assert np.allclose(ref[2:], 0.0), f"constant region should be 0.0: {ref}"

    def test_min_max_all_nan_window_returns_nan(self):
        """ts_min/ts_max on all-NaN window should return NaN."""
        for name in ("ts_min", "ts_max"):
            k = _kernels().get(name)
            if k is None:
                pytest.skip(f"{name} not registered")
            arr = np.array([np.nan, np.nan, np.nan, 1.0, 2.0])
            ref = k.reference_fn(arr, 3, 1)
            assert np.isnan(ref[0]), f"{name}: all-NaN window should be NaN"
            assert np.isnan(ref[1]), f"{name}: all-NaN window should be NaN"
            assert np.isnan(ref[2]), f"{name}: all-NaN window should be NaN"


# ===========================================================================
# 3. Reference vs Pandas parity
# ===========================================================================

class TestNumbaKernelPandasParity:
    """Reference implementation matches pandas rolling semantics."""

    @pytest.mark.parametrize("kernel_name", ["ts_sum", "ts_mean", "ts_min", "ts_max", "ts_std"])
    def test_pandas_parity(self, kernel_name):
        """Reference implementation matches pandas rolling() for each kernel."""
        k = _kernels().get(kernel_name)
        if k is None:
            pytest.skip(f"{kernel_name} not registered")

        s = pd.Series(_HOSTILE_1D)
        if kernel_name == "ts_sum":
            pd_out = s.rolling(_WINDOW, min_periods=_MIN_COUNT).sum().values
        elif kernel_name == "ts_mean":
            pd_out = s.rolling(_WINDOW, min_periods=_MIN_COUNT).mean().values
        elif kernel_name == "ts_min":
            pd_out = s.rolling(_WINDOW, min_periods=_MIN_COUNT).min().values
        elif kernel_name == "ts_max":
            pd_out = s.rolling(_WINDOW, min_periods=_MIN_COUNT).max().values
        elif kernel_name == "ts_std":
            pd_out = s.rolling(_WINDOW, min_periods=_MIN_COUNT).std().values
        else:
            pytest.skip(f"unknown kernel: {kernel_name}")

        ref_out = k.reference_fn(_HOSTILE_1D, _WINDOW, _MIN_COUNT)
        np.testing.assert_allclose(
            pd_out, ref_out, rtol=1e-10, atol=1e-10,
            err_msg=f"{kernel_name}: reference does not match pandas"
        )


# ===========================================================================
# 4. Numba availability honesty
# ===========================================================================

class TestNumbaAvailabilityHonesty:
    """Registry's numba_available flag must reflect reality."""

    def test_numba_available_state_honest(self):
        if NUMBA_AVAILABLE:
            for name in ("ts_sum", "ts_mean", "ts_min", "ts_max", "ts_std"):
                k = _kernels().get(name)
                if k is not None:
                    assert k.numba_fn is not None, f"{name}: numba available but no impl"
        else:
            for name in ("ts_sum", "ts_mean", "ts_min", "ts_max", "ts_std"):
                k = _kernels().get(name)
                if k is not None:
                    assert k.numba_fn is None and k.reference_fn is not None
