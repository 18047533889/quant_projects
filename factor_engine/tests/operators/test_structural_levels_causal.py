import numpy as np
import pandas as pd
import pytest
from factor_engine.cleaned_operators.structural_levels import (
    TsStructuralLevelDensity,
    TsNearestStructuralLevelDistance,
    TsStructuralLevelStrength,
)
from factor_engine.cleaned_operators.common._pivot_ledger import (
    StreamingConfirmedPivotLedger,
    confirmed_pivot_events,
)

# Test for Causality Mutation
def test_structural_levels_causality_mutation():
    """
    Verify that changing future data does not affect the current row's output.
    This ensures the operators are prefix-causal and PIT-safe.
    """
    np.random.seed(42)
    price_data = np.random.lognormal(mean=0, sigma=0.02, size=(100, 1))
    price_df = pd.DataFrame(price_data, columns=["price"])

    # Initialize operators
    density_op = TsStructuralLevelDensity()
    distance_op = TsNearestStructuralLevelDistance()
    strength_op = TsStructuralLevelStrength()

    # Calculate output for the original data
    out1_density = density_op._calculate_series(price_df, window=20, prominence=0.02, confirmation=3)
    out1_distance = distance_op._calculate_series(price_df, window=20, prominence=0.02, confirmation=3)
    out1_strength = strength_op._calculate_series(price_df, window=20, prominence=0.02, confirmation=3, cutoff=0.05)

    # Mutate the future (rows > 50)
    price_df_mutated = price_df.copy()
    price_df_mutated.iloc[51:, :] = np.random.lognormal(mean=1, sigma=0.1, size=(49, 1))

    # Calculate output for the mutated data
    out2_density = density_op._calculate_series(price_df_mutated, window=20, prominence=0.02, confirmation=3)
    out2_distance = distance_op._calculate_series(price_df_mutated, window=20, prominence=0.02, confirmation=3)
    out2_strength = strength_op._calculate_series(price_df_mutated, window=20, prominence=0.02, confirmation=3, cutoff=0.05)

    # Check that outputs for the first 50 rows are identical
    # Use a helper that handles NaNs correctly
    def _assert_equal_nan(a, b):
        mask = np.isnan(a) & np.isnan(b)
        np.testing.assert_array_equal(a[~mask], b[~mask])
        np.testing.assert_array_equal(a[mask], b[mask]) # Both should be NaN

    _assert_equal_nan(out1_density.iloc[:51].values, out2_density.iloc[:51].values)
    _assert_equal_nan(out1_distance.iloc[:51].values, out2_distance.iloc[:51].values)
    _assert_equal_nan(out1_strength.iloc[:51].values, out2_strength.iloc[:51].values)

# Test for Staleness Cap
def test_structural_levels_staleness_cap():
    """
    Verify that max_pivot_age correctly filters out old pivots.
    If a pivot is older than max_pivot_age, it should not contribute to the output.
    """
    # Use random data that generates confirmed pivots
    np.random.seed(42)
    price_data = np.random.lognormal(mean=0, sigma=0.02, size=200)
    price_df = pd.DataFrame(price_data, columns=["price"])

    # With no max_pivot_age (or very large), pivots remain active
    density_op = TsStructuralLevelDensity()
    out_no_cap = density_op._calculate_series(price_df, window=20, prominence=0.02, confirmation=3, max_pivot_age=None, min_periods=5)

    # With a small max_pivot_age (e.g., 5), pivots are dropped once their age exceeds 5
    out_with_cap = density_op._calculate_series(price_df, window=20, prominence=0.02, confirmation=3, max_pivot_age=5, min_periods=5)

    # The values should differ after the staleness cap kicks in
    # We expect NaN or lower density if the only active pivot is dropped
    # Use a helper that handles NaNs correctly
    def _assert_not_equal_nan(a, b):
        # If both are NaN, they are considered equal in allclose, but we want to check if values differ
        mask_a = np.isnan(a)
        mask_b = np.isnan(b)
        if not np.array_equal(mask_a, mask_b):
            return True # One is NaN, other is not
        return not np.allclose(a[~mask_a], b[~mask_b])

    assert _assert_not_equal_nan(out_no_cap.values, out_with_cap.values)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
