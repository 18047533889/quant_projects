# -*- coding: utf-8 -*-
"""R21-NUMBA-FAIL-CLOSED: Regression tests for NumbaKernel production mode enforcement.

Tests verify:
- Production mode: NumbaKernel.call() raises KernelUnavailableError when numba unavailable
- Research mode: NumbaKernel.call() falls back to reference with UserWarning
- Both modes: NumbaKernel.call() works correctly when numba is available

Thread env vars: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1
"""
from __future__ import annotations

import warnings

import pytest

from backend.numba_kernel_registry import (
    NumbaKernel,
    NumbaKernelSpec,
    KernelUnavailableError,
    NUMBA_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_numba_unavailable_kernel():
    """Create a NumbaKernel with numba_available=False for testing fallback/error."""
    spec = NumbaKernelSpec(
        canonical_family="test",
        kernel_name="test_kernel",
        semantic_version="1.0",
    )
    return NumbaKernel(
        spec=spec,
        reference_fn=lambda x: x * 2,
        numba_fn=None,
        numba_available=False,
    )


def _make_numba_available_kernel():
    """Create a NumbaKernel with numba_available=True for testing normal operation."""
    spec = NumbaKernelSpec(
        canonical_family="test",
        kernel_name="test_kernel",
        semantic_version="1.0",
    )
    return NumbaKernel(
        spec=spec,
        reference_fn=lambda x: x * 2,
        numba_fn=lambda x: x * 2,  # Same implementation for parity
        numba_available=True,
    )


# ---------------------------------------------------------------------------
# Test Class: Production Mode Raises
# ---------------------------------------------------------------------------

class TestNumbaProductionModeFailClosed:
    """Production mode must raise KernelUnavailableError when numba unavailable."""

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Test requires numba available")
    def test_production_mode_raises_kernel_unavailable(self, monkeypatch):
        """NumbaKernel.call() raises KernelUnavailableError in production mode."""
        monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", "production")
        kernel = _make_numba_unavailable_kernel()

        with pytest.raises(KernelUnavailableError, match="test_kernel.*production"):
            kernel.call(5, use_numba=True)

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Test requires numba available")
    def test_production_mode_use_numba_false_works(self, monkeypatch):
        """NumbaKernel.call(use_numba=False) still works in production mode."""
        monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", "production")
        kernel = _make_numba_unavailable_kernel()

        # Should not raise when use_numba=False
        result = kernel.call(5, use_numba=False)
        assert result == 10


# ---------------------------------------------------------------------------
# Test Class: Research Mode Warns
# ---------------------------------------------------------------------------

class TestNumbaResearchModeWarns:
    """Research mode must fall back to reference with UserWarning when numba unavailable."""

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Test requires numba available")
    def test_research_mode_warns_and_falls_back(self, monkeypatch):
        """NumbaKernel.call() warns and falls back to reference in research mode."""
        monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", "research")
        kernel = _make_numba_unavailable_kernel()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = kernel.call(5, use_numba=True)

            assert result == 10  # Reference function: x * 2
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "test_kernel" in str(w[0].message)
            assert "falling back" in str(w[0].message)


# ---------------------------------------------------------------------------
# Test Class: Numba Available Works
# ---------------------------------------------------------------------------

class TestNumbaAvailableWorks:
    """Both modes must work correctly when numba is available."""

    def test_production_mode_with_numba(self, monkeypatch):
        """Production mode uses numba when available."""
        monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", "production")
        kernel = _make_numba_available_kernel()

        result = kernel.call(5, use_numba=True)
        assert result == 10

    def test_research_mode_with_numba(self, monkeypatch):
        """Research mode uses numba when available."""
        monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", "research")
        kernel = _make_numba_available_kernel()

        result = kernel.call(5, use_numba=True)
        assert result == 10

    def test_both_modes_use_reference_when_use_numba_false(self, monkeypatch):
        """Both modes use reference when use_numba=False."""
        for mode in ("production", "research"):
            monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", mode)
            kernel = _make_numba_available_kernel()

            result = kernel.call(5, use_numba=False)
            assert result == 10
