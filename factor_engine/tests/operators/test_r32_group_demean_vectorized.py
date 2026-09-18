"""Independent oracle for scaled group accumulation."""
import math
import numpy as np
import pytest
from factor_engine.cleaned_operators._numpy_kernels import group_demean_panel_


@pytest.mark.parametrize("magnitude", [1.0, 1e308, 1e-308])
def test_group_demean_scaled_accumulation(magnitude):
    rng = np.random.default_rng(32)
    x = rng.uniform(-0.8, 0.8, (6, 97)) * magnitude
    groups = rng.integers(0, 17, x.shape).astype(float)
    x[0] = np.nan
    x[1, :10] = np.inf
    groups[2, :12] = np.nan
    x[3, :20] = 0.0
    groups[3, :20] = 99
    expected = np.full_like(x, np.nan)
    for r in range(len(x)):
        for g in np.unique(groups[r, np.isfinite(groups[r])]):
            mask = (groups[r] == g) & np.isfinite(x[r])
            if mask.any():
                values = x[r, mask]
                # Sum divided terms, avoiding overflowing the oracle sum itself.
                mean = math.fsum(float(v) / len(values) for v in values)
                expected[r, mask] = values - mean
    got = group_demean_panel_(x, groups)
    # Compare after restoring scale so tiny-number mismatches cannot hide
    # beneath a large absolute tolerance.
    np.testing.assert_allclose(got / magnitude, expected / magnitude,
                               rtol=1e-12, atol=1e-14, equal_nan=True)


def test_group_demean_empty_and_missing_members():
    for shape in [(0, 3), (3, 0), (0, 0)]:
        got = group_demean_panel_(np.empty(shape), np.empty(shape))
        assert got.shape == shape
    got = group_demean_panel_(np.array([[np.nan, 1.0]]), np.array([[1.0, np.nan]]))
    assert np.isnan(got).all()
