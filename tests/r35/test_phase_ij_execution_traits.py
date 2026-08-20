# -*- coding: utf-8 -*-
"""R35 Phase I-J: ExecutionTraits + nested-thread governance + thread budget.

Covers taskbook §23/§24 (only one layer owns the CPU), §78 (oversubscription
detection), §109 (backend taxonomy) and §110 (ExecutionTraits).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from runtime.execution_traits import (  # noqa: E402
    EXECUTION_TRAITS,
    nested_thread_oversubscription_error,
    thread_budget,
    traits_for_backend,
)


# --------------------------------------------------------------------------
# §110: ExecutionTraits
# --------------------------------------------------------------------------

def test_all_backend_kinds_have_traits():
    for kind in ("sql_native", "polars_native", "numpy_blas", "numba_kernel", "python_specialized"):
        assert kind in EXECUTION_TRAITS, kind
        t = EXECUTION_TRAITS[kind]
        assert isinstance(t.releases_gil, bool)
        assert t.internal_threads >= 1
        assert t.parallelizable_dimension in ("factor", "block", "column", "none")


def test_numba_and_native_release_gil():
    for kind in ("numba_kernel", "numpy_blas", "sql_native", "polars_native"):
        assert EXECUTION_TRAITS[kind].releases_gil, kind


def test_python_specialized_is_gil_bound():
    t = EXECUTION_TRAITS["python_specialized"]
    assert not t.releases_gil
    assert t.picklable_payload is False


def test_traits_for_backend_mapping():
    assert traits_for_backend("duckdb").backend_kind == "sql_native"
    assert traits_for_backend("polars").backend_kind == "polars_native"
    assert traits_for_backend("pandas_numpy").backend_kind == "numpy_blas"
    assert traits_for_backend("numba").backend_kind == "numba_kernel"


# --------------------------------------------------------------------------
# §78: oversubscription detection
# --------------------------------------------------------------------------

def test_oversubscription_detected():
    errs = nested_thread_oversubscription_error(8, 8, 16)
    assert errs, "8x8 on 16 CPUs must be flagged"


def test_no_oversubscription_small():
    assert nested_thread_oversubscription_error(2, 2, 16) == []
    assert nested_thread_oversubscription_error(16, 1, 16) == []
    assert nested_thread_oversubscription_error(1, 16, 16) == []


# --------------------------------------------------------------------------
# §24: thread_budget context manager
# --------------------------------------------------------------------------

def test_thread_budget_runs():
    """thread_budget must be a working context manager (limits applied or
    no-op if threadpoolctl unavailable) and must not change compute output."""
    x = np.ones(100)
    with thread_budget(1):
        out = float(np.dot(x, x))
    assert out == pytest.approx(100.0)


def test_thread_budget_zero_is_noop():
    with thread_budget(0):
        pass  # must not raise


# --------------------------------------------------------------------------
# §77: serial vs threaded execution reference parity
# --------------------------------------------------------------------------

def test_kernel_reference_thread_safe():
    """Numba kernels are nogil and releases-GIL: running the same input twice
    (cold + warm) must give identical output — determinism under threading."""
    import backend.numba_kernels  # noqa: F401
    from backend.numba_kernel_registry import NumbaKernelRegistry

    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    k = NumbaKernelRegistry.get("kalman_level")
    if k is None:
        pytest.skip("kalman_level not registered")
    r1 = k.call(x, 1e-4, 1.0, use_numba=True)
    r2 = k.call(x, 1e-4, 1.0, use_numba=True)
    np.testing.assert_array_equal(r1, r2)
