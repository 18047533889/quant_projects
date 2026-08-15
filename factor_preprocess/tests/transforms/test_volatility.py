"""
Test suite for volatility transforms.
"""
import pytest
import pandas as pd
import numpy as np
from factor_preprocess.transforms.volatility import (
    volatility_scale,
    volatility_scale_returns,
    realized_volatility,
    ewma_volatility,
    garch_inspired_volatility,
)


class TestVolatilityScale:
    """Test volatility scaling transform."""

    def test_basic_volatility_scale(self):
        """Test basic volatility scaling."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 10,
            "date": pd.date_range("2020-01-01", periods=10),
            "value": [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0],
        })

        result = volatility_scale(df, window=3, target_vol=1.0)

        # First 3 should be NaN (warmup period)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])

        # After warmup, should have values
        assert pd.notna(result.iloc[3])

    def test_target_vol_applied(self):
        """Test that target volatility is correctly applied."""
        # Create data with known volatility
        df = pd.DataFrame({
            "asset_id": ["A"] * 10,
            "date": pd.date_range("2020-01-01", periods=10),
            "value": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        })

        result = volatility_scale(df, window=3, target_vol=2.0)

        # After warmup, compute what the scaling should produce
        # At index 3: lagged values are [1.0, 2.0, 1.0], std ~0.577
        # scaled value = 2.0 * (2.0 / 0.577) ~ 6.93
        lagged_std = np.std([1.0, 2.0, 1.0], ddof=1)
        expected = 2.0 * (2.0 / lagged_std)
        np.testing.assert_allclose(result.iloc[3], expected, rtol=1e-5)

    def test_zero_volatility_produces_nan(self):
        """Zero historical volatility follows the documented NaN contract."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [5.0, 5.0, 5.0, 5.0, 10.0],  # Constant then spike
        })

        result = volatility_scale(df, window=3)

        # At index 3, lagged volatility is 0 (all 5.0).
        assert pd.isna(result.iloc[3])

    def test_per_asset_isolation(self):
        """Test that assets are processed independently."""
        df = pd.DataFrame({
            "asset_id": ["A", "A", "A", "B", "B", "B"],
            "date": pd.date_range("2020-01-01", periods=3).tolist() * 2,
            "value": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        })

        result = volatility_scale(df, window=2)

        # Asset A's scaling should not be affected by Asset B's values
        # Each asset should have independent volatility calculation
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.notna(result.iloc[2])

        assert pd.isna(result.iloc[3])
        assert pd.isna(result.iloc[4])
        assert pd.notna(result.iloc[5])

    def test_invalid_window(self):
        """Test that invalid window is rejected."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 3,
            "date": pd.date_range("2020-01-01", periods=3),
            "value": [1.0, 2.0, 3.0],
        })

        with pytest.raises(ValueError, match="window must be >= 1"):
            volatility_scale(df, window=0)

    def test_invalid_target_vol(self):
        """Test that invalid target_vol is rejected."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 3,
            "date": pd.date_range("2020-01-01", periods=3),
            "value": [1.0, 2.0, 3.0],
        })

        with pytest.raises(ValueError, match="target_vol must be > 0"):
            volatility_scale(df, window=2, target_vol=-1.0)

    def test_unsorted_raises_error(self):
        """Test that unsorted data raises error."""
        df = pd.DataFrame({
            "asset_id": ["A", "A", "A"],
            "date": pd.date_range("2020-01-03", periods=3)[::-1],  # Reverse order
            "value": [1.0, 2.0, 3.0],
        })

        with pytest.raises(ValueError, match="must be sorted"):
            volatility_scale(df, window=2)


class TestVolatilityScaleReturns:
    """Test return-specific volatility scaling."""

    def test_returns_scaling(self):
        """Test scaling of return series."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 10,
            "date": pd.date_range("2020-01-01", periods=10),
            "return": [0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02, -0.02, 0.01, -0.01],
        })

        result = volatility_scale_returns(df, window=3, target_vol=0.05)

        # Should scale returns to target vol
        assert pd.isna(result.iloc[:3]).all()
        assert pd.notna(result.iloc[3:]).all()


class TestRealizedVolatility:
    """Test realized volatility computation."""

    def test_basic_realized_vol(self):
        """Test basic realized volatility."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 10,
            "date": pd.date_range("2020-01-01", periods=10),
            "value": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        })

        result = realized_volatility(df, window=3)

        # First 3 should be NaN
        assert pd.isna(result.iloc[:3]).all()

        # After warmup, should have volatility values
        assert pd.notna(result.iloc[3:]).all()

    def test_annualization_factor(self):
        """Test annualization factor is applied."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 10,
            "date": pd.date_range("2020-01-01", periods=10),
            "value": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        })

        result_no_annualization = realized_volatility(df, window=3, annualization_factor=1.0)
        result_with_annualization = realized_volatility(df, window=3, annualization_factor=np.sqrt(252))

        # With annualization should be sqrt(252) times larger
        np.testing.assert_allclose(
            result_with_annualization.iloc[3:],
            result_no_annualization.iloc[3:] * np.sqrt(252),
            rtol=1e-10,
        )

    def test_lagged_computation(self):
        """Test that realized vol is computed from lagged values."""
        # Create data with spike at end
        df = pd.DataFrame({
            "asset_id": ["A"] * 6,
            "date": pd.date_range("2020-01-01", periods=6),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],  # Spike at end
        })

        result = realized_volatility(df, window=3)

        # At index 5, realized vol should be from [3.0, 4.0, 5.0], not include 100.0
        lagged_std = np.std([3.0, 4.0, 5.0], ddof=1)
        np.testing.assert_allclose(result.iloc[5], lagged_std, rtol=1e-5)


@pytest.mark.future_poison
class TestVolatilityFuturePoison:
    """Test that volatility transforms exclude future information."""

    def test_volatility_scale_excludes_current(self):
        """Test that volatility_scale excludes current observation."""
        # Create data with spike at end
        df = pd.DataFrame({
            "asset_id": ["A"] * 6,
            "date": pd.date_range("2020-01-01", periods=6),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],  # Spike
        })

        result = volatility_scale(df, window=3, target_vol=1.0)

        # At index 5, volatility should be from [3.0, 4.0, 5.0], not include 100.0
        lagged_std = np.std([3.0, 4.0, 5.0], ddof=1)
        expected = 100.0 * (1.0 / lagged_std)
        np.testing.assert_allclose(result.iloc[5], expected, rtol=1e-5)

    def test_no_forward_looking_volatility(self):
        """Test that future changes do not affect past volatility."""
        df1 = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        df2 = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1.0, 2.0, 3.0, 4.0, 100.0],  # Different last value
        })

        result1 = realized_volatility(df1, window=3)
        result2 = realized_volatility(df2, window=3)

        # Volatility at indices 0-3 should be identical
        np.testing.assert_array_equal(result1.iloc[:4].values, result2.iloc[:4].values)

    def test_warmup_period_prevents_leakage(self):
        """Test that warmup period prevents using current observation."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 4,
            "date": pd.date_range("2020-01-01", periods=4),
            "value": [100.0, 1.0, 2.0, 3.0],  # Spike at beginning
        })

        result = realized_volatility(df, window=3)

        # First 3 observations should be NaN (no peeking forward)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])


class TestEWMAVolatility:
    """Test EWMA volatility computation."""

    def test_basic_ewma_volatility(self):
        """Test basic EWMA volatility."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 10,
            "date": pd.date_range("2020-01-01", periods=10),
            "value": [0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02, -0.02, 0.01, -0.01],
        })

        result = ewma_volatility(df, halflife=3.0, min_periods=2)

        # First two observations should be NaN (shift excludes current, need min 2)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])

        # After warmup, should have volatility values
        assert pd.notna(result.iloc[2:]).all()

    def test_annualization_factor(self):
        """Test annualization factor is applied."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 10,
            "date": pd.date_range("2020-01-01", periods=10),
            "value": [0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02, -0.02, 0.01, -0.01],
        })

        result_no_annualization = ewma_volatility(df, halflife=3.0, annualization_factor=1.0)
        result_with_annualization = ewma_volatility(df, halflife=3.0, annualization_factor=np.sqrt(252))

        # With annualization should be sqrt(252) times larger
        np.testing.assert_allclose(
            result_with_annualization.iloc[1:],
            result_no_annualization.iloc[1:] * np.sqrt(252),
            rtol=1e-10,
        )

    def test_excludes_current_observation(self):
        """Test that EWMA volatility excludes current observation."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 6,
            "date": pd.date_range("2020-01-01", periods=6),
            "value": [0.01, 0.01, 0.01, 0.01, 0.01, 10.0],  # Spike at end
        })

        result = ewma_volatility(df, halflife=3.0)

        # At index 5, volatility should not include the 10.0 spike
        # It should be based only on lagged observations
        assert result.iloc[5] < 1.0  # Should be small, not influenced by spike

    def test_per_asset_isolation(self):
        """Test that assets are processed independently."""
        df = pd.DataFrame({
            "asset_id": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "date": list(pd.date_range("2020-01-01", periods=4)) + list(pd.date_range("2020-01-01", periods=4)),
            "value": [0.01, 0.02, 0.01, 0.02, 0.10, 0.20, 0.10, 0.20],
        })
        # Sort to ensure proper order
        df = df.sort_values(['asset_id', 'date']).reset_index(drop=True)

        result = ewma_volatility(df, halflife=2.0, min_periods=2)

        # Each asset should have independent calculations
        # Asset A: indices 0-3
        assert pd.isna(result.iloc[0])  # Asset A, first obs
        assert pd.isna(result.iloc[1])  # Asset A, second obs (only 1 lagged value)
        assert pd.notna(result.iloc[2])  # Asset A, third obs (2 lagged values)

        # Asset B: indices 4-7, should start fresh
        assert pd.isna(result.iloc[4])  # Asset B, first obs
        assert pd.isna(result.iloc[5])  # Asset B, second obs
        assert pd.notna(result.iloc[6])  # Asset B, third obs

        # Asset B should have higher volatility than Asset A (10x larger values)
        assert result.iloc[6] > result.iloc[2]

    def test_invalid_halflife(self):
        """Test that invalid halflife is rejected."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 3,
            "date": pd.date_range("2020-01-01", periods=3),
            "value": [1.0, 2.0, 3.0],
        })

        with pytest.raises(ValueError, match="halflife must be > 0"):
            ewma_volatility(df, halflife=0.0)


class TestGARCHInspiredVolatility:
    """Test GARCH-inspired volatility computation."""

    def test_basic_garch_volatility(self):
        """Test basic GARCH-inspired volatility."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 25,
            "date": pd.date_range("2020-01-01", periods=25),
            "value": np.random.randn(25) * 0.01,
        })

        result = garch_inspired_volatility(df, short_window=5, long_window=20)

        # First observations should be NaN (warmup period)
        assert pd.isna(result.iloc[:5]).any()

        # After warmup, should have volatility values
        assert pd.notna(result.iloc[20:]).all()

    def test_short_and_long_combination(self):
        """Test that short and long volatilities are combined."""
        # Create data with recent spike
        values = [0.01] * 20 + [0.10, 0.10, 0.10, 0.10, 0.10]
        df = pd.DataFrame({
            "asset_id": ["A"] * 25,
            "date": pd.date_range("2020-01-01", periods=25),
            "value": values,
        })

        result = garch_inspired_volatility(df, short_window=5, long_window=20)

        # The combined volatility should reflect both short-term spike and long-term baseline
        # Short-term component (70%) should dominate after spike
        assert pd.notna(result.iloc[-1])

    def test_annualization_factor(self):
        """Test annualization factor is applied."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 25,
            "date": pd.date_range("2020-01-01", periods=25),
            "value": np.random.randn(25) * 0.01,
        })

        result_no_annualization = garch_inspired_volatility(
            df, short_window=5, long_window=20, annualization_factor=1.0
        )
        result_with_annualization = garch_inspired_volatility(
            df, short_window=5, long_window=20, annualization_factor=np.sqrt(252)
        )

        # With annualization should be sqrt(252) times larger
        valid_mask = pd.notna(result_no_annualization) & pd.notna(result_with_annualization)
        np.testing.assert_allclose(
            result_with_annualization[valid_mask],
            result_no_annualization[valid_mask] * np.sqrt(252),
            rtol=1e-10,
        )

    def test_excludes_current_observation(self):
        """Test that GARCH volatility excludes current observation."""
        # Create data with spike at end
        values = [0.01] * 24 + [10.0]
        df = pd.DataFrame({
            "asset_id": ["A"] * 25,
            "date": pd.date_range("2020-01-01", periods=25),
            "value": values,
        })

        result = garch_inspired_volatility(df, short_window=5, long_window=20)

        # At last index, volatility should not include the 10.0 spike
        assert result.iloc[-1] < 0.1  # Should be small, not influenced by spike

    def test_invalid_windows(self):
        """Test that invalid window parameters are rejected."""
        df = pd.DataFrame({
            "asset_id": ["A"] * 25,
            "date": pd.date_range("2020-01-01", periods=25),
            "value": [0.01] * 25,
        })

        with pytest.raises(ValueError, match="short_window must be >= 1"):
            garch_inspired_volatility(df, short_window=0, long_window=20)

        with pytest.raises(ValueError, match="long_window must be >= short_window"):
            garch_inspired_volatility(df, short_window=10, long_window=5)

    def test_per_asset_isolation(self):
        """Test that assets are processed independently."""
        np.random.seed(42)
        df = pd.DataFrame({
            "asset_id": ["A"] * 25 + ["B"] * 25,
            "date": pd.date_range("2020-01-01", periods=25).tolist() * 2,
            "value": np.concatenate([np.random.randn(25) * 0.01, np.random.randn(25) * 0.10]),
        })

        result = garch_inspired_volatility(df, short_window=5, long_window=20)

        # Each asset should have independent calculations
        # Asset B should have ~10x higher volatility than Asset A
        asset_a_vol = result.iloc[24]
        asset_b_vol = result.iloc[49]
        assert pd.notna(asset_a_vol)
        assert pd.notna(asset_b_vol)
        assert asset_b_vol > asset_a_vol * 5  # At least 5x higher


class TestVolatilityAssetLayoutMetamorphics:
    """Asset layout must not alter per-asset temporal results."""

    @staticmethod
    def _frame(layout: str) -> pd.DataFrame:
        dates = pd.date_range("2022-01-01", periods=8)
        rows = []
        series = {
            "A": [1.0, 2.0, 4.0, 3.0, 7.0, 5.0, 9.0, 6.0],
            "B": [20.0, 18.0, 25.0, 21.0, 30.0, 24.0, 33.0, 27.0],
        }
        if layout == "interleaved":
            for i, date in enumerate(dates):
                for asset in ("A", "B"):
                    rows.append((asset, date, series[asset][i]))
        else:
            asset_order = ("B", "A") if layout == "blocks_reordered" else ("A", "B")
            for asset in asset_order:
                rows.extend((asset, date, series[asset][i]) for i, date in enumerate(dates))
        frame = pd.DataFrame(rows, columns=["asset_id", "date", "value"])
        frame.index = pd.Index(np.arange(100, 100 + 3 * len(frame), 3), name="row_id")
        return frame

    @staticmethod
    def _keyed(frame: pd.DataFrame, result: pd.Series) -> pd.Series:
        aligned = frame[["asset_id", "date"]].copy()
        aligned["result"] = result
        return aligned.set_index(["asset_id", "date"])["result"].sort_index()

    @pytest.mark.parametrize(
        "transform,kwargs",
        [
            (volatility_scale, {"window": 3}),
            (realized_volatility, {"window": 3}),
            (ewma_volatility, {"halflife": 2.0, "min_periods": 2}),
            (
                garch_inspired_volatility,
                {"short_window": 2, "long_window": 4, "min_periods": 2},
            ),
        ],
    )
    def test_interleaved_assets_match_isolated_execution(self, transform, kwargs):
        interleaved = self._frame("interleaved")
        actual = transform(interleaved, **kwargs)

        assert actual.index.equals(interleaved.index)
        expected = pd.Series(index=interleaved.index, dtype=float)
        for _, group in interleaved.groupby("asset_id", sort=False):
            expected.loc[group.index] = transform(group, **kwargs)

        expected.name = actual.name
        pd.testing.assert_series_equal(actual, expected)

    @pytest.mark.parametrize(
        "transform,kwargs",
        [
            (volatility_scale, {"window": 3}),
            (realized_volatility, {"window": 3}),
            (ewma_volatility, {"halflife": 2.0, "min_periods": 2}),
            (
                garch_inspired_volatility,
                {"short_window": 2, "long_window": 4, "min_periods": 2},
            ),
        ],
    )
    def test_block_reorder_preserves_key_alignment(self, transform, kwargs):
        original = self._frame("blocks")
        reordered = self._frame("blocks_reordered")

        expected = self._keyed(original, transform(original, **kwargs))
        actual_result = transform(reordered, **kwargs)
        assert actual_result.index.equals(reordered.index)
        actual = self._keyed(reordered, actual_result)

        pd.testing.assert_series_equal(actual, expected)
