"""
Tests for stateless transforms.
"""
import pytest
import numpy as np

from modeling_adapters.preprocess.stateless import rank_transform, zscore_transform, winsorize
from modeling_adapters.errors import InsufficientDataError


class TestRankTransform:
    def test_cross_sectional_rank(self):
        X = np.array([[1.0, 3.0, 2.0], [5.0, 1.0, 3.0]])
        ranked = rank_transform(X, axis=1, pct=True)
        assert ranked.shape == X.shape

    def test_time_series_rank(self):
        X = np.array([[1.0, 5.0], [3.0, 3.0], [2.0, 1.0]])
        ranked = rank_transform(X, axis=0, pct=True)
        assert ranked.shape == X.shape

    def test_rank_with_nan(self):
        X = np.array([[1.0, np.nan, 3.0], [2.0, 4.0, np.nan]])
        ranked = rank_transform(X, axis=1, pct=True)
        assert np.isnan(ranked[0, 1])

    def test_rank_ties_use_average_percentile(self):
        X = np.array([[1.0, 2.0, 2.0, 4.0]])
        ranked = rank_transform(X, axis=1, pct=True)
        np.testing.assert_allclose(ranked, [[0.0, 0.5, 0.5, 1.0]])

    def test_all_nan_row(self):
        X = np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]])
        ranked = rank_transform(X, axis=1, pct=True)
        assert np.all(np.isnan(ranked[1, :]))

    def test_empty_array(self):
        X = np.array([]).reshape(0, 5)
        with pytest.raises(InsufficientDataError):
            rank_transform(X, axis=1)


class TestZscoreTransform:
    def test_cross_sectional_zscore(self):
        np.random.seed(42)
        X = np.random.randn(100, 5) * 2 + 5
        z = zscore_transform(X, axis=1)
        row_means = np.nanmean(z, axis=1)
        assert np.allclose(row_means, 0, atol=1e-10)

    def test_time_series_zscore(self):
        np.random.seed(42)
        X = np.random.randn(100, 5) * 2 + 5
        z = zscore_transform(X, axis=0)
        col_means = np.nanmean(z, axis=0)
        assert np.allclose(col_means, 0, atol=1e-10)

    def test_zscore_with_clipping(self):
        X = np.array([[1.0, 2.0, 3.0, 100.0]])
        z = zscore_transform(X, axis=1, clip=3.0)
        assert np.all(z >= -3.0)
        assert np.all(z <= 3.0)

    def test_zscore_with_nan(self):
        X = np.array([[1.0, np.nan, 3.0, 4.0], [2.0, 3.0, np.nan, 5.0]])
        z = zscore_transform(X, axis=1)
        assert np.isnan(z[0, 1])

    def test_constant_values(self):
        X = np.array([[5.0, 5.0, 5.0, 5.0]])
        z = zscore_transform(X, axis=1)
        assert np.all(np.isnan(z))

    def test_empty_array(self):
        X = np.array([]).reshape(0, 5)
        with pytest.raises(InsufficientDataError):
            zscore_transform(X, axis=1)


class TestWinsorize:
    def test_winsorize_basic(self):
        X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=float)
        w = winsorize(X, lower=0.1, upper=0.9, axis=1)
        assert w[0, 0] > 1

    def test_winsorize_cross_sectional(self):
        np.random.seed(42)
        X = np.random.randn(100, 50)
        w = winsorize(X, lower=0.01, upper=0.99, axis=1)
        assert w.shape == X.shape

    def test_winsorize_with_nan(self):
        X = np.array([[1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        w = winsorize(X, lower=0.1, upper=0.9, axis=1)
        assert np.isnan(w[0, 1])

    def test_invalid_quantiles(self):
        X = np.random.randn(10, 5)
        with pytest.raises(ValueError):
            winsorize(X, lower=0.5, upper=0.3)

    def test_empty_array(self):
        X = np.array([]).reshape(0, 5)
        with pytest.raises(InsufficientDataError):
            winsorize(X, lower=0.01, upper=0.99)


class TestTransformsWithRealData:
    def test_pipeline_rank_then_zscore(self):
        np.random.seed(42)
        X = np.random.randn(252, 500)
        X_ranked = rank_transform(X, axis=1, pct=True)
        X_z = zscore_transform(X_ranked, axis=1)
        assert X_z.shape == X.shape

    def test_pipeline_winsorize_then_zscore(self):
        np.random.seed(42)
        X = np.random.randn(252, 500)
        X_w = winsorize(X, lower=0.01, upper=0.99, axis=1)
        X_z = zscore_transform(X_w, axis=1)
        assert X_z.shape == X.shape


def test_rank_transform_rejects_boolean_mask():
    """Pre-fix: True/False silently ranked as 1.0/0.0 pseudo-factor."""
    import numpy as np
    import pytest
    from modeling_adapters.preprocess.stateless import rank_transform

    with pytest.raises(TypeError, match="boolean mask"):
        rank_transform(np.array([True, False, True]))
