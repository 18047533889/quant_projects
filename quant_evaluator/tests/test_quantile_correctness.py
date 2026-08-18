"""
Property tests for quantile assignment correctness.

QE-Q-P0-001: Prove tie policy parameter affects output
QE-Q-P0-002: Prove NumPy-Numba boundary parity
QE-Q-P0-004: Prove NaN/Inf handling correctness
"""

import numpy as np
import pytest

from quant_evaluator.metrics.quantile import (
    assign_quantiles,
    assign_quantiles_fast,
    assign_quantiles_batch,
    compute_quantile_returns,
)
from quant_evaluator.metrics.quantile_optimized import (
    compute_quantile_returns_ultra_fast,
)
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.quantile_optimized import assign_quantiles_vectorized_v2
from quant_evaluator.contracts.quantile_policy import QuantileTiePolicy

try:
    from quant_evaluator.metrics.quantile_numba import (
        assign_quantiles_numba,
        is_numba_available,
    )
    NUMBA_AVAILABLE = is_numba_available()
except ImportError:
    NUMBA_AVAILABLE = False


def test_quantile_returns_respect_factor_validity():
    """Finite factors marked invalid do not define quantile observations."""
    T, N, F = 3, 10, 1
    values = np.arange(T * N, dtype=float).reshape(T, N, F)
    validity = np.ones_like(values, dtype=bool)
    validity[1, 0, 0] = False
    batch = FactorBatch(
        factor_ids=("factor",),
        time_axis=AxisRef(name="time", dtype="int64", size=T),
        asset_axis=AxisRef(name="asset", dtype="int64", size=N),
        values=values,
        validity=validity,
    )
    bundle = LabelBundle(
        target_id="return",
        values=np.arange(T * N, dtype=float).reshape(T, N),
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
    )

    implementations = [
        lambda: compute_quantile_returns(batch, bundle, n_quantiles=5, min_assets=1),
        lambda: compute_quantile_returns_ultra_fast(batch, bundle, n_quantiles=5, min_assets=1),
    ]
    if NUMBA_AVAILABLE:
        from quant_evaluator.metrics.quantile_numba import compute_quantile_returns_numba
        implementations.append(
            lambda: compute_quantile_returns_numba(
                batch, bundle, n_quantiles=5, min_assets=1
            )
        )
    from quant_evaluator.metrics.quantile import compute_quantile_returns_fast
    implementations.append(
        lambda: compute_quantile_returns_fast(
            batch, bundle, n_quantiles=5, min_assets=1
        )
    )

    for implementation in implementations:
        returns, counts = implementation()
        assert counts[1, :, :].sum() == N - 1
        assert np.all(np.isfinite(returns[1, :, :]))


def test_quantile_returns_respect_one_dimensional_label_validity():
    """Invalid scalar-label periods contribute no quantile observations."""
    T, N, F = 3, 10, 1
    values = np.arange(T * N, dtype=float).reshape(T, N, F)
    batch = FactorBatch(
        factor_ids=("factor",),
        time_axis=AxisRef(name="time", dtype="int64", size=T),
        asset_axis=AxisRef(name="asset", dtype="int64", size=N),
        values=values,
    )
    bundle = LabelBundle(
        target_id="return",
        values=np.array([1.0, 2.0, 3.0]),
        validity=np.array([True, False, True]),
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
    )

    implementations = [
        lambda: compute_quantile_returns(batch, bundle, n_quantiles=5, min_assets=1),
        lambda: compute_quantile_returns_ultra_fast(batch, bundle, n_quantiles=5, min_assets=1),
    ]
    if NUMBA_AVAILABLE:
        from quant_evaluator.metrics.quantile_numba import compute_quantile_returns_numba
        implementations.append(
            lambda: compute_quantile_returns_numba(
                batch, bundle, n_quantiles=5, min_assets=1
            )
        )
    from quant_evaluator.metrics.quantile import compute_quantile_returns_fast
    implementations.append(
        lambda: compute_quantile_returns_fast(
            batch, bundle, n_quantiles=5, min_assets=1
        )
    )

    for implementation in implementations:
        returns, counts = implementation()
        assert np.all(counts[1, :, :] == 0)
        assert np.all(np.isnan(returns[1, :, :]))
        assert np.all(counts[[0, 2], :, :] > 0)
        assert np.all(np.isfinite(returns[[0, 2], :, :]))



def test_tie_policy_min_vs_max_different_results():
    """
    QE-Q-P0-001: Prove that method parameter actually affects output.

    Create data where many values exactly equal the computed boundaries.
    """
    np.random.seed(42)

    # Create data where values exactly match percentile boundaries
    # For 5 quantiles: boundaries at 20%, 40%, 60%, 80%
    # With 101 values (0-100), boundaries will be at indices 20, 40, 60, 80
    n = 101
    values = np.arange(n, dtype=float).reshape(1, n)  # 0, 1, 2, ..., 100

    # Now make many values equal to boundary values
    # Set values 15-25 to exactly 20.0
    # Set values 35-45 to exactly 40.0
    # Set values 55-65 to exactly 60.0
    # Set values 75-85 to exactly 80.0
    values[0, 15:26] = 20.0
    values[0, 35:46] = 40.0
    values[0, 55:66] = 60.0
    values[0, 75:86] = 80.0

    # Test MIN policy
    q_min = assign_quantiles(values, n_quantiles=5, method="min")

    # Test MAX policy
    q_max = assign_quantiles(values, n_quantiles=5, method="max")

    # They must differ for tied boundary values
    assert not np.array_equal(q_min, q_max), (
        "QE-Q-P0-001 FAIL: min and max policies produce identical results"
    )

    # Count differences
    n_different = np.sum(q_min != q_max)
    assert n_different > 10, (
        f"QE-Q-P0-001 FAIL: Only {n_different} differences, expected many more with heavy ties"
    )

    # Verify tie behavior: values equal to boundary should differ by policy
    # Find which values are exactly at boundaries
    percentiles = np.linspace(0, 100, 6)[1:-1]
    boundaries = np.percentile(values[0, :], percentiles)

    for i, val in enumerate(values[0, :]):
        # Check if this value is very close to a boundary
        for boundary in boundaries:
            if abs(val - boundary) < 1e-10:
                # This value is at a boundary, policies should differ
                if q_min[0, i] != q_max[0, i]:
                    # At least one tie was handled differently
                    print(f"✓ QE-Q-P0-001: Tie policy affects output ({n_different} differences)")
                    return

    print(f"✓ QE-Q-P0-001: Tie policy affects output ({n_different} differences)")



def test_tie_policy_invalid_method_raises():
    """QE-Q-P0-001: Invalid method should raise ValueError."""
    values = np.array([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError, match="Invalid tie policy"):
        assign_quantiles(values, method="invalid")


def test_tie_policy_average_not_implemented():
    """QE-Q-P0-001: 'average' method should raise NotImplementedError."""
    values = np.array([[1.0, 2.0, 3.0]])

    with pytest.raises(NotImplementedError, match="not implemented"):
        assign_quantiles(values, method="average")


def test_tie_policy_first_not_implemented():
    """QE-Q-P0-001: 'first' method should raise NotImplementedError."""
    values = np.array([[1.0, 2.0, 3.0]])

    with pytest.raises(NotImplementedError, match="not implemented"):
        assign_quantiles(values, method="first")


# =============================================================================
# QE-Q-P0-002: NumPy-Numba boundary parity
# =============================================================================

@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_numpy_numba_parity_heavy_ties():
    """
    QE-Q-P0-002: NumPy and Numba must produce identical results with heavy ties.

    80/100 values are identical to test tie-breaking at boundaries.
    """
    np.random.seed(123)
    T, N, F = 10, 100, 3

    values = np.random.randn(T, N, F)

    # Create heavy ties in each time slice
    for t in range(T):
        for f in range(F):
            # Set 80% of values to the same number
            tie_value = values[t, 0, f]
            values[t, :80, f] = tie_value

    # Test both policies
    for method in ["min", "max"]:
        q_numpy = assign_quantiles_batch(values, n_quantiles=5, method=method)
        q_numba = assign_quantiles_numba(values, n_quantiles=5, method=method)

        assert np.array_equal(q_numpy, q_numba), (
            f"QE-Q-P0-002 FAIL: NumPy-Numba mismatch with method='{method}' and heavy ties"
        )

    print("✓ QE-Q-P0-002: NumPy-Numba parity verified with heavy ties")


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_numpy_numba_parity_boundary_exact_match():
    """
    QE-Q-P0-002: Test exact boundary value matches.

    Create values that exactly equal computed boundaries.
    """
    np.random.seed(456)
    T, N, F = 5, 50, 2

    # Create values with exact quantile boundaries
    values = np.zeros((T, N, F))
    for t in range(T):
        for f in range(F):
            # Values from 0 to 49, guaranteeing exact boundary matches
            values[t, :, f] = np.arange(N, dtype=float)

    for method in ["min", "max"]:
        q_numpy = assign_quantiles_batch(values, n_quantiles=5, method=method)
        q_numba = assign_quantiles_numba(values, n_quantiles=5, method=method)

        assert np.array_equal(q_numpy, q_numba), (
            f"QE-Q-P0-002 FAIL: NumPy-Numba mismatch with method='{method}' and boundary exact matches"
        )

    print("✓ QE-Q-P0-002: NumPy-Numba parity verified with boundary exact matches")


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_numpy_numba_parity_all_implementations():
    """
    QE-Q-P0-002: All NumPy implementations must agree with Numba.
    """
    np.random.seed(789)
    T, N = 20, 100

    values_2d = np.random.randn(T, N)

    # Add ties
    for t in range(T):
        tie_value = values_2d[t, 0]
        values_2d[t, :30] = tie_value

    for method in ["min", "max"]:
        q_base = assign_quantiles(values_2d, n_quantiles=5, method=method)
        q_fast = assign_quantiles_fast(values_2d, n_quantiles=5, method=method)
        q_batch = assign_quantiles_batch(values_2d, n_quantiles=5, method=method)
        q_numba = assign_quantiles_numba(values_2d, n_quantiles=5, method=method)

        assert np.array_equal(q_base, q_fast), (
            f"assign_quantiles vs assign_quantiles_fast mismatch (method={method})"
        )
        assert np.array_equal(q_base, q_batch), (
            f"assign_quantiles vs assign_quantiles_batch mismatch (method={method})"
        )
        assert np.array_equal(q_base, q_numba), (
            f"assign_quantiles vs assign_quantiles_numba mismatch (method={method})"
        )

    print("✓ QE-Q-P0-002: All implementations agree")


# =============================================================================
# QE-Q-P0-004: NaN/Inf correctness
# =============================================================================

@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_nan_propagation_numpy_numba_parity():
    """
    QE-Q-P0-004: NaN propagation must be identical in NumPy and Numba.

    Without fastmath, Numba must handle NaN exactly like NumPy.
    """
    np.random.seed(111)
    T, N, F = 10, 50, 2

    values = np.random.randn(T, N, F)

    # Insert NaNs in various positions
    values[0, 0, 0] = np.nan
    values[2, 10:20, 1] = np.nan
    values[5, :, 0] = np.nan  # All NaN slice

    q_numpy = assign_quantiles_batch(values, n_quantiles=5, method="max")
    q_numba = assign_quantiles_numba(values, n_quantiles=5, method="max")

    # Check that NaN positions match
    nan_mask_numpy = (q_numpy == -1)
    nan_mask_numba = (q_numba == -1)

    assert np.array_equal(nan_mask_numpy, nan_mask_numba), (
        "QE-Q-P0-004 FAIL: NaN masking differs between NumPy and Numba"
    )

    # Check that finite values match
    finite_mask = (q_numpy != -1)
    assert np.array_equal(q_numpy[finite_mask], q_numba[finite_mask]), (
        "QE-Q-P0-004 FAIL: Finite values differ between NumPy and Numba"
    )

    print("✓ QE-Q-P0-004: NaN propagation parity verified")


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_inf_handling_numpy_numba_parity():
    """
    QE-Q-P0-004: +Inf/-Inf handling must be identical.
    """
    np.random.seed(222)
    T, N, F = 10, 50, 2

    values = np.random.randn(T, N, F)

    # Insert infinities
    values[0, 0, 0] = np.inf
    values[1, 5, 1] = -np.inf
    values[3, 10:15, 0] = np.inf
    values[4, 20:25, 1] = -np.inf

    q_numpy = assign_quantiles_batch(values, n_quantiles=5, method="max")
    q_numba = assign_quantiles_numba(values, n_quantiles=5, method="max")

    # Inf values should be marked as invalid (-1)
    inf_mask = np.isinf(values)
    assert np.all(q_numpy[inf_mask] == -1), "NumPy: Inf should be -1"
    assert np.all(q_numba[inf_mask] == -1), "Numba: Inf should be -1"

    # Non-inf finite values should match
    finite_mask = np.isfinite(values)
    assert np.array_equal(q_numpy[finite_mask], q_numba[finite_mask]), (
        "QE-Q-P0-004 FAIL: Finite values differ with Inf present"
    )

    print("✓ QE-Q-P0-004: Inf handling parity verified")


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_mixed_nan_inf_numpy_numba_parity():
    """
    QE-Q-P0-004: Mixed NaN and Inf handling.
    """
    np.random.seed(333)
    T, N, F = 8, 40, 3

    values = np.random.randn(T, N, F)

    # Mix of NaN, +Inf, -Inf
    values[0, 0:5, 0] = np.nan
    values[0, 5:10, 0] = np.inf
    values[0, 10:15, 0] = -np.inf
    values[2, :, 1] = np.nan
    values[3, ::2, 2] = np.inf
    values[4, 1::2, 2] = -np.inf

    q_numpy = assign_quantiles_batch(values, n_quantiles=5, method="min")
    q_numba = assign_quantiles_numba(values, n_quantiles=5, method="min")

    assert np.array_equal(q_numpy, q_numba), (
        "QE-Q-P0-004 FAIL: Mixed NaN/Inf handling differs"
    )

    print("✓ QE-Q-P0-004: Mixed NaN/Inf parity verified")


def test_all_nan_slice():
    """QE-Q-P0-004: All-NaN slice should produce all -1."""
    values = np.full((5, 10), np.nan)

    q = assign_quantiles(values, n_quantiles=5, method="max")

    assert np.all(q == -1), "All-NaN slice should produce all -1"


def test_all_inf_slice():
    """QE-Q-P0-004: All-Inf slice should produce all -1."""
    values = np.full((5, 10), np.inf)

    q = assign_quantiles(values, n_quantiles=5, method="max")

    assert np.all(q == -1), "All-Inf slice should produce all -1"


# =============================================================================
# Edge cases
# =============================================================================

@pytest.mark.parametrize(
    ("shape", "n_quantiles", "expected_shape"),
    [
        ((1, 5, 1), 5, (1, 5)),
        ((5, 1, 1), 1, (5, 1)),
        ((1, 1, 1), 1, (1, 1)),
        ((3, 5, 1), 5, (3, 5)),
        ((1, 5, 3), 5, (1, 5, 3)),
    ],
)
def test_quantile_assignment_preserves_time_and_asset_axes(
    shape, n_quantiles, expected_shape
):
    """Only the singleton factor axis may be removed from 3D input."""
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)

    q_batch = assign_quantiles_batch(values, n_quantiles=n_quantiles)
    q_vectorized = assign_quantiles_vectorized_v2(
        values, n_quantiles=n_quantiles
    )

    assert q_batch.shape == expected_shape
    assert q_vectorized.shape == expected_shape
    assert np.array_equal(q_batch, q_vectorized)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
@pytest.mark.parametrize(
    ("shape", "n_quantiles", "expected_shape"),
    [
        ((1, 5, 1), 5, (1, 5)),
        ((5, 1, 1), 1, (5, 1)),
        ((1, 1, 1), 1, (1, 1)),
        ((3, 5, 1), 5, (3, 5)),
        ((1, 5, 3), 5, (1, 5, 3)),
    ],
)
def test_quantile_assignment_shape_numpy_numba_parity(
    shape, n_quantiles, expected_shape
):
    """NumPy and Numba preserve the same shape ABI for singleton axes."""
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)

    q_numpy = assign_quantiles_batch(values, n_quantiles=n_quantiles)
    q_numba = assign_quantiles_numba(values, n_quantiles=n_quantiles)

    assert q_numpy.shape == expected_shape
    assert q_numba.shape == expected_shape
    assert np.array_equal(q_numpy, q_numba)


def test_insufficient_finite_values():
    """Fewer finite values than quantiles should produce all -1."""
    values = np.array([[1.0, 2.0, np.nan, np.nan, np.nan]])

    q = assign_quantiles(values, n_quantiles=5, method="max")

    assert np.all(q == -1), "Insufficient values should produce all -1"


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_single_finite_value():
    """Single finite value should produce -1 (insufficient for quantiles)."""
    values = np.array([[1.0, np.nan, np.nan]])

    q_numpy = assign_quantiles(values, n_quantiles=5, method="max")
    q_numba = assign_quantiles_numba(values, n_quantiles=5, method="max")

    assert np.all(q_numpy == -1)
    assert np.all(q_numba == -1)

    assert q_numpy.shape == (1, 3)
    assert q_numba.shape == (1, 3)
    assert np.array_equal(q_numpy, q_numba)


if __name__ == "__main__":
    print("Running QE-Q-P0 quantile correctness tests...\n")

    # QE-Q-P0-001: Tie policy
    test_tie_policy_min_vs_max_different_results()
    test_tie_policy_invalid_method_raises()
    test_tie_policy_average_not_implemented()
    test_tie_policy_first_not_implemented()

    # QE-Q-P0-002: Boundary parity
    if NUMBA_AVAILABLE:
        test_numpy_numba_parity_heavy_ties()
        test_numpy_numba_parity_boundary_exact_match()
        test_numpy_numba_parity_all_implementations()
    else:
        print("⚠ Skipping NumPy-Numba parity tests (Numba not available)")

    # QE-Q-P0-004: NaN/Inf correctness
    if NUMBA_AVAILABLE:
        test_nan_propagation_numpy_numba_parity()
        test_inf_handling_numpy_numba_parity()
        test_mixed_nan_inf_numpy_numba_parity()
    else:
        print("⚠ Skipping NaN/Inf parity tests (Numba not available)")

    test_all_nan_slice()
    test_all_inf_slice()
    test_insufficient_finite_values()

    if NUMBA_AVAILABLE:
        test_single_finite_value()

    print("\n✅ All QE-Q-P0 tests passed")
