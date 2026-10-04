"""Stored scalar precision must not be erased by endpoint certification."""
import numpy as np
import pytest
from quant_evaluator.metrics.correlation_endpoint import certify_correlation_endpoint


def test_nonaffine_large_int64_is_not_snapped_after_lossy_conversion():
    base = 2**53
    x = np.array([base, base+2, base+4, base+6, base+8], dtype=np.int64)
    y = x.copy()
    y[-1] += 1
    candidate = np.nextafter(1., 0.)
    assert certify_correlation_endpoint(candidate, x, y) == candidate


def test_mixed_integer_float_identity_is_not_proved_by_coercing_equality():
    x = np.array([2**53, 2**53+2, 2**53+5], dtype=np.int64)
    y = x.astype(np.float64)
    candidate = np.nextafter(1., 0.)
    assert certify_correlation_endpoint(candidate, x, y) == candidate


def test_unsigned_modular_negation_is_not_an_affinity_proof():
    y = np.array([0, 1, 2], dtype=np.uint64)
    x = np.array([0, 2**64-1, 2**64-2], dtype=np.uint64)
    candidate = -np.nextafter(1., 0.)
    assert certify_correlation_endpoint(candidate, x, y) == candidate


def test_extended_precision_nonaffinity_is_preserved():
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("no extended floating precision on this platform")
    x = np.arange(5, dtype=np.longdouble) + np.longdouble(2**53)
    y = x.copy()
    y[-1] = np.nextafter(y[-1], np.longdouble(np.inf))
    candidate = np.nextafter(1., 0.)
    assert certify_correlation_endpoint(candidate, x, y) == candidate
