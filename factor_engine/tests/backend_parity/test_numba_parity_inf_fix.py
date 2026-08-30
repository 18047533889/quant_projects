# -*- coding: utf-8 -*-
"""R21-NUMBA-PARITY-INF-FIX: Regression tests for Numba parity Inf-mask correctness.

Four regression tests ensuring the parity checker correctly compares:
 1. +Inf mask (isposinf)
 2. -Inf mask (isneginf)
 3. Finite mask (isfinite)
 4. dtype and shape identity

Previously, parity_check used allclose on a combined isfinite mask,
which silently accepted +Inf vs NaN mismatches when the NumPy ref and
the Numba kernel both happened to produce NaN at different Inf-tainted
positions. The fix decomposes the comparison into explicit sub-masks.

Serial pytest only. Thread env vars set by parent harness.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("FACTOR_ENGINE_DISABLE_NUMBA", "0")
os.environ.setdefault("FACTOR_ENGINE_NUMBA_THREADS", "1")

from factor_engine.backend.numba_kernel_registry import (  # noqa: E402
    NUMBA_AVAILABLE,
    DuplicateKernelRegistrationError,
    DuplicateSpecMismatchError,
    NumbaKernel,
    NumbaKernelRegistry,
    NumbaKernelSpec,
    _compute_numba_kernel_implementation_id,
    parity_check,
)


def _compute_impl_id_for_test() -> str:
    """Return the implementation_id the registry WILL compute for (_trivial_ref,_trivial_numba)."""
    return _compute_numba_kernel_implementation_id(_trivial_ref, _trivial_numba)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trivial_ref(arr: np.ndarray) -> np.ndarray:
    """Identity reference kernel — returns arr unchanged."""
    return arr.copy()


def _trivial_numba(arr: np.ndarray) -> np.ndarray:
    """Identity numba kernel — same as reference."""
    return arr.copy()


def _nan_injecting_ref(arr: np.ndarray) -> np.ndarray:
    """Reference that turns +Inf into NaN (correct behaviour for e.g. rolling mean)."""
    out = arr.copy()
    out[np.isposinf(out)] = np.nan
    return out


def _inf_passthrough_numba(arr: np.ndarray) -> np.ndarray:
    """Buggy numba kernel that leaves +Inf as-is (parity should FAIL)."""
    return arr.copy()


def _dtypes_aware_ref(arr: np.ndarray) -> np.ndarray:
    """Reference preserving dtype."""
    return arr.copy()


def _dtypes_mismatch_numba(arr: np.ndarray) -> np.ndarray:
    """Numba kernel returning wrong dtype (float32 instead of float64)."""
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Test 1: +Inf mask comparison — parity must detect Inf-vs-NaN mismatch
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
class TestInfMaskPositiveInf:
    """When ref converts +Inf to NaN but numba keeps +Inf, parity must FAIL."""

    def test_posinf_mismatch_detected(self):
        kernel_name = "_test_posinf_parity"
        spec = NumbaKernelSpec(
            canonical_family="test",
            kernel_name=kernel_name,
            semantic_version="0.0",
        )
        k = NumbaKernel(
            spec=spec,
            reference_fn=_nan_injecting_ref,
            numba_fn=_inf_passthrough_numba,
        )
        fixture = np.array([1.0, np.inf, 2.0, -np.inf, 3.0], dtype=np.float64)
        result = parity_check(k, fixture)
        assert result["status"] == "FAIL", (
            f"parity_check should detect +Inf mask mismatch but got {result['status']}"
        )
        assert result["posinf_ok"] is False, "+Inf mask must mismatch"

    def test_posinf_both_nan_passes(self):
        """Both ref and numba turn +Inf to NaN -> parity PASS."""
        kernel_name = "_test_posinf_both_nan"
        spec = NumbaKernelSpec(
            canonical_family="test",
            kernel_name=kernel_name,
            semantic_version="0.0",
        )
        k = NumbaKernel(
            spec=spec,
            reference_fn=_nan_injecting_ref,
            numba_fn=_nan_injecting_ref,
        )
        fixture = np.array([1.0, np.inf, 2.0, -np.inf, 3.0], dtype=np.float64)
        result = parity_check(k, fixture)
        assert result["status"] == "PASS"
        assert result["posinf_ok"] is True


# ---------------------------------------------------------------------------
# Test 2: -Inf mask comparison
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
class TestInfMaskNegativeInf:
    """When ref converts -Inf to NaN but numba keeps -Inf, parity must FAIL."""

    def _neg_inf_ref(a: np.ndarray) -> np.ndarray:
        out = a.copy()
        out[np.isneginf(out)] = np.nan
        return out

    def test_neginf_mismatch_detected(self):
        kernel_name = "_test_neginf_parity"
        spec = NumbaKernelSpec(
            canonical_family="test",
            kernel_name=kernel_name,
            semantic_version="0.0",
        )
        def neg_inf_ref(a: np.ndarray) -> np.ndarray:
            return TestInfMaskNegativeInf._neg_inf_ref(a)

        k = NumbaKernel(
            spec=spec,
            reference_fn=neg_inf_ref,
            numba_fn=_inf_passthrough_numba,
        )
        fixture = np.array([1.0, np.inf, 2.0, -np.inf, 3.0], dtype=np.float64)
        result = parity_check(k, fixture)
        assert result["status"] == "FAIL"
        assert result["neginf_ok"] is False, "-Inf mask must mismatch"


# ---------------------------------------------------------------------------
# Test 3: Finite mask comparison — Inf vs non-Inf at same position
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
class TestFiniteMask:
    """Finite mask mismatch must be detected independently of allclose."""

    def test_finite_mask_mismatch_detected(self):
        """Ref produces NaN at pos0; numba produces finite 1.0 at pos0 -> finite masks differ."""
        def ref_nans_first(arr: np.ndarray) -> np.ndarray:
            """Reference: NaN at index 0."""
            out = arr.copy()
            out[0] = np.nan
            return out

        def numba_identity(arr: np.ndarray) -> np.ndarray:
            """Numba: passes everything through."""
            return arr.copy()

        kernel_name = "_test_finite_mask_parity"
        spec = NumbaKernelSpec(
            canonical_family="test",
            kernel_name=kernel_name,
            semantic_version="0.0",
        )
        k = NumbaKernel(
            spec=spec,
            reference_fn=ref_nans_first,
            numba_fn=numba_identity,
        )
        fixture = np.array([1.0, np.inf, 2.0, -np.inf, 3.0], dtype=np.float64)
        result = parity_check(k, fixture)
        assert result["status"] == "FAIL"
        assert result["finite_mask_ok"] is False, "finite mask must mismatch"

    def test_finite_mask_both_agree_passes(self):
        def finite_only_ref(a: np.ndarray) -> np.ndarray:
            out = a.copy()
            out[~np.isfinite(out)] = np.nan
            return out

        kernel_name = "_test_finite_mask_pass"
        spec = NumbaKernelSpec(
            canonical_family="test",
            kernel_name=kernel_name,
            semantic_version="0.0",
        )
        k = NumbaKernel(
            spec=spec,
            reference_fn=finite_only_ref,
            numba_fn=finite_only_ref,
        )
        fixture = np.array([1.0, np.inf, 2.0, -np.inf, 3.0], dtype=np.float64)
        result = parity_check(k, fixture)
        assert result["status"] == "PASS"
        assert result["finite_mask_ok"] is True


# ---------------------------------------------------------------------------
# Test 4: dtype and shape mismatch detection
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
class TestDtypeShape:
    """dtype and shape mismatches must be caught even when allclose values agree."""

    def test_dtype_mismatch_detected(self):
        kernel_name = "_test_dtype_parity"
        spec = NumbaKernelSpec(
            canonical_family="test",
            kernel_name=kernel_name,
            semantic_version="0.0",
        )
        k = NumbaKernel(
            spec=spec,
            reference_fn=_dtypes_aware_ref,
            numba_fn=_dtypes_mismatch_numba,
        )
        fixture = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = parity_check(k, fixture)
        assert result["status"] == "FAIL"
        assert result["dtype_ok"] is False, "dtype must mismatch (float64 vs float32)"

    def test_shape_mismatch_detected(self):
        def _longer_numba(arr: np.ndarray) -> np.ndarray:
            return np.append(arr, 999.0)

        kernel_name = "_test_shape_parity"
        spec = NumbaKernelSpec(
            canonical_family="test",
            kernel_name=kernel_name,
            semantic_version="0.0",
        )
        k = NumbaKernel(
            spec=spec,
            reference_fn=_trivial_ref,
            numba_fn=_longer_numba,
        )
        fixture = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = parity_check(k, fixture)
        assert result["status"] == "FAIL"
        assert result["shape_ok"] is False, "shape must mismatch"

    def test_exact_parity_passes(self):
        """Identical ref and numba -> PASS with all sub-checks True."""
        kernel_name = "_test_exact_parity"
        spec = NumbaKernelSpec(
            canonical_family="test",
            kernel_name=kernel_name,
            semantic_version="0.0",
        )
        k = NumbaKernel(
            spec=spec,
            reference_fn=_trivial_ref,
            numba_fn=_trivial_numba,
        )
        fixture = np.array([1.0, np.inf, -np.inf, np.nan, 42.0], dtype=np.float64)
        result = parity_check(k, fixture)
        assert result["status"] == "PASS"
        assert result["posinf_ok"] is True
        assert result["neginf_ok"] is True
        assert result["finite_mask_ok"] is True
        assert result["nan_ok"] is True
        assert result["dtype_ok"] is True
        assert result["shape_ok"] is True
        assert result["finite_match"] is True


# ---------------------------------------------------------------------------
# Test 5: Registry dedup — same name, different spec -> error
# ---------------------------------------------------------------------------

class TestRegistryDedup:
    """DuplicateKernelRegistrationError when kernel_name re-registered with different spec."""

    def test_duplicate_different_spec_raises(self):
        k1 = NumbaKernelSpec(
            canonical_family="kalman",
            kernel_name="_dedup_test_kernel",
            semantic_version="1.0",
        )
        k2 = NumbaKernelSpec(
            canonical_family="kalman",
            kernel_name="_dedup_test_kernel",
            semantic_version="2.0",  # different
        )
        kernel_a = NumbaKernel(spec=k1, reference_fn=_trivial_ref)
        kernel_b = NumbaKernel(spec=k2, reference_fn=_trivial_ref)

        NumbaKernelRegistry._kernels["_dedup_test_kernel"] = kernel_a
        try:
            with pytest.raises(DuplicateSpecMismatchError, match="_dedup_test_kernel"):
                NumbaKernelRegistry.register(
                    "kalman", "_dedup_test_kernel", _trivial_ref, _trivial_numba,
                    semantic_version="2.0",
                )
        finally:
            NumbaKernelRegistry._kernels.pop("_dedup_test_kernel", None)

    def test_idempotent_same_spec_allows(self):
        """Re-registering the exact same spec is idempotent (no error)."""
        k = NumbaKernelSpec(
            canonical_family="kalman",
            kernel_name="_dedup_idempotent",
            semantic_version="1.0",
        )
        kernel_a = NumbaKernel(
            spec=k,
            reference_fn=_trivial_ref,
            implementation_id=_compute_impl_id_for_test(),
        )
        NumbaKernelRegistry._kernels["_dedup_idempotent"] = kernel_a
        try:
            # Same spec AND same reference_fn/numba_fn → same implementation_id
            # → idempotent allow (registry's stated dedup policy).
            NumbaKernelRegistry.register(
                "kalman", "_dedup_idempotent", _trivial_ref, _trivial_numba,
                semantic_version="1.0",
            )
        finally:
            NumbaKernelRegistry._kernels.pop("_dedup_idempotent", None)
