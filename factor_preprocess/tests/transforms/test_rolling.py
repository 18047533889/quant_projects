"""Tests for grouped causal rolling transforms."""
import numpy as np
import pandas as pd
import pytest

from factor_preprocess.transforms.rolling import ewma, rolling_mean, rolling_std, rolling_zscore


def _two_assets():
    return pd.DataFrame({
        "asset_id": ["A", "A", "A", "B", "B", "B"],
        "date": pd.date_range("2020-01-01", periods=3).tolist() * 2,
        "value": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
    })


class TestRollingTransforms:
    def test_rolling_mean_is_per_asset(self):
        result = rolling_mean(_two_assets(), window=2)
        assert pd.isna(result.iloc[0]) and pd.isna(result.iloc[1])
        assert result.iloc[2] == 1.5
        assert pd.isna(result.iloc[3]) and pd.isna(result.iloc[4])
        assert result.iloc[5] == 150.0

    def test_explicit_min_periods_does_not_cross_boundary(self):
        result = rolling_mean(_two_assets(), window=3, min_periods=1)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == 1.0
        assert result.iloc[2] == 1.5
        assert pd.isna(result.iloc[3])
        assert result.iloc[4] == 100.0
        assert result.iloc[5] == 150.0

    @pytest.mark.parametrize("transform", [
        lambda df: rolling_std(df, window=3, min_periods=1),
        lambda df: rolling_zscore(df, window=3, min_periods=2),
        lambda df: ewma(df, halflife=2.0, min_periods=1),
    ])
    def test_grouped_transform_prefix_is_invariant_to_other_asset_suffix(self, transform):
        clean = _two_assets()
        poisoned = clean.copy()
        poisoned.loc[poisoned.asset_id.eq("B"), "value"] = [1e12, -1e12, 1e12]
        baseline = transform(clean)
        observed = transform(poisoned)
        np.testing.assert_allclose(
            observed.iloc[:3], baseline.iloc[:3], equal_nan=True,
        )

    def test_ewma_explicit_min_periods_does_not_cross_boundary(self):
        result = ewma(_two_assets(), halflife=2.0, min_periods=2)
        assert result.iloc[:2].isna().all()
        assert result.iloc[2] == pytest.approx(1.2928932188)
        assert result.iloc[3:5].isna().all()
        assert result.iloc[5] == pytest.approx(129.28932188)

    def test_suffix_poison_does_not_change_prefix(self):
        clean = pd.DataFrame({
            "asset_id": ["A"] * 6,
            "date": pd.date_range("2020-01-01", periods=6),
            "value": [1., 2., 3., 4., 5., 6.],
        })
        poisoned = clean.copy()
        poisoned.loc[4:, "value"] = [1e12, -1e12]
        np.testing.assert_allclose(
            rolling_mean(poisoned, window=3, min_periods=1).iloc[:4],
            rolling_mean(clean, window=3, min_periods=1).iloc[:4],
            equal_nan=True,
        )

    def test_unsorted_fails(self):
        df = _two_assets().iloc[[1, 0, 2, 3, 4, 5]]
        with pytest.raises(ValueError, match="sorted"):
            rolling_mean(df, window=2)
