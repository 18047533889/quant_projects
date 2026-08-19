"""
Test exposure neutralization behavior with singular matrices.

Verifies that LinAlgError is handled properly by returning NaN
instead of silently returning un-neutralized values.
"""
import numpy as np
import pytest
from modeling_adapters.exposure import _cross_sectional_residual, compute_exposure_residual


def test_ols_singular_matrix_returns_nan():
    """Verify OLS returns NaN when matrix is singular."""
    # Create singular matrix: col2 = 2 * col1
    X = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
    ])
    y = np.array([1.0, 2.0, 3.0])

    result = _cross_sectional_residual(y, X, method="ols")

    # Should return NaN array to signal computation failure
    assert np.all(np.isnan(result)), "Expected NaN array for singular matrix in OLS"


def test_weighted_ols_singular_matrix():
    """Verify weighted OLS handles singular matrix."""
    # Perfectly collinear columns
    X = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    y = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 1.0, 1.0])

    result = _cross_sectional_residual(y, X, method="weighted_ols", weights=weights)
    assert np.all(np.isnan(result)), "Expected NaN array for singular matrix in weighted OLS"


def test_ridge_singular_matrix_may_succeed():
    """Ridge adds regularization, may handle near-singular matrices."""
    # Nearly collinear (ridge might handle this)
    X = np.array([[1.0, 2.0], [1.0, 2.001], [1.0, 2.002]])
    y = np.array([1.0, 2.0, 3.0])

    result = _cross_sectional_residual(y, X, method="ridge")
    # Ridge might succeed or return NaN, but should NOT return original y unchanged
    # (unless all values happen to align perfectly, which is unlikely)
    assert not np.allclose(result, y) or np.all(np.isnan(result)), \
        "Ridge should either compute residuals or return NaN, not original values"


def test_ols_rank_deficient_matrix():
    """Test with explicitly rank-deficient matrix."""
    # Matrix with rank 1 (all rows are multiples of first row)
    X = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
    ])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    result = _cross_sectional_residual(y, X, method="ols")
    assert np.all(np.isnan(result)), "Expected NaN for rank-deficient matrix"


def test_compute_exposure_residual_propagates_nan():
    """Verify that NaN from singular matrix propagates through time dimension."""
    # Create 3D exposure with singular cross-section
    T, N, K = 2, 3, 2
    factor_values = np.array([
        [1.0, 2.0, 3.0],  # t=0
        [4.0, 5.0, 6.0],  # t=1
    ])

    # Both time slices have singular exposure matrix
    exposures = np.array([
        [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],  # t=0: singular
        [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],  # t=1: singular
    ])

    result = compute_exposure_residual(factor_values, exposures, method="ols")

    # Both time slices should be NaN
    assert result.shape == (T, N)
    assert np.all(np.isnan(result)), "Expected all NaN when all time slices are singular"


def test_ols_valid_then_singular():
    """Test mixed case: one valid time slice, one singular."""
    T, N, K = 2, 3, 2
    factor_values = np.array([
        [1.0, 2.0, 3.0],  # t=0
        [4.0, 5.0, 6.0],  # t=1
    ])

    exposures = np.array([
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],  # t=0: valid (identity-like)
        [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],  # t=1: singular
    ])

    result = compute_exposure_residual(factor_values, exposures, method="ols")

    # t=0 should have residuals (not all original values)
    # t=1 should be all NaN
    assert not np.all(np.isnan(result[0, :])), "First time slice should compute residuals"
    assert np.all(np.isnan(result[1, :])), "Second time slice should be NaN (singular)"


def test_weighted_ols_zero_weights_becomes_singular():
    """Verify that zero weights can create singular weighted system."""
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.0, 0.0, 1.0])  # Only one non-zero weight

    # With only one effective observation, system is underdetermined
    result = _cross_sectional_residual(y, X, method="weighted_ols", weights=weights)
    # This might return NaN or handle gracefully depending on lstsq behavior
    # We just verify it doesn't silently return original y without warning
    assert np.all(np.isnan(result)) or not np.allclose(result, y)


def test_sparse_cross_section_returns_nan_not_raw_factor():
    """Pre-fix: a slice with < K+1 valid rows silently returned raw y."""
    import numpy as np
    from modeling_adapters.exposure import compute_exposure_residual

    rng = np.random.default_rng(0)
    T, N, K = 2, 6, 2
    factor_values = rng.normal(size=(T, N))
    exposures = rng.normal(size=(T, N, K))
    # t=1: only 2 valid rows for K=2 exposures (needs >= 3)
    factor_values[1, :] = [1.0, np.nan, np.nan, np.nan, np.nan, 2.0]

    result = compute_exposure_residual(factor_values, exposures, method="ols")
    assert not np.all(np.isnan(result[0, :])), "Dense slice must compute residuals"
    assert np.all(np.isnan(result[1, :])), "Sparse slice must signal NaN, not pass raw y"


def test_weighted_ols_without_weights_is_a_clear_error():
    """Pre-fix: fell through to the misleading 'Unknown method' message."""
    import numpy as np
    import pytest
    from modeling_adapters.exposure import compute_exposure_residual

    factor_values = np.ones((1, 4))
    exposures = np.ones((1, 4, 1))
    with pytest.raises(ValueError, match="requires weights"):
        compute_exposure_residual(factor_values, exposures, method="weighted_ols")
