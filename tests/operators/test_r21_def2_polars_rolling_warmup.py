# -*- coding: utf-8 -*-
"""R21-DEF2: Polars rolling warmup min_samples divergence regression tests.

Tests verify that polars rolling operations respect min_periods=window (hold NaN
until window has w finite observations), matching pandas fail-closed semantics.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest


class TestEwmMeanWarmup:
    """Test ewm_mean with min_periods=window masking."""

    def test_dema_warmup_with_leading_nulls(self):
        """DEMA should publish NaN until window finite observations."""
        data = [None, None, 1.0, 2.0, 3.0, 4.0, 5.0]
        df = pl.DataFrame({"x": data})

        # Calculate DEMA manually with min_periods masking
        w = 3
        alpha = 2.0 / (w + 1)

        # Calculate ema1 with min_periods masking
        ema1_raw = pl.col('x').ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
        ema1_count = pl.col('x').is_not_null().cast(pl.Int64).cum_sum()
        ema1 = pl.when(ema1_count >= w).then(ema1_raw).otherwise(None)

        # Calculate ema2 with min_periods masking
        ema2_raw = ema1.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
        ema2_count = ema1.is_not_null().cast(pl.Int64).cum_sum()
        ema2 = pl.when(ema2_count >= w).then(ema2_raw).otherwise(None)

        # Calculate DEMA
        result = df.with_columns([
            (2 * ema1 - ema2).alias('dema')
        ])['dema'].to_numpy()

        # First 6 rows should be NaN (window=3, ema1 needs 3 finite obs, ema2 needs 3 more)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.isnan(result[2])
        assert np.isnan(result[3])
        assert np.isnan(result[4])
        assert np.isnan(result[5])
        # Row 6 should have a value (7 finite observations for ema2)
        assert np.isfinite(result[6])

    def test_tema_warmup_with_leading_nulls(self):
        """TEMA should publish NaN until window finite observations."""
        data = [None, None, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
        df = pl.DataFrame({"x": data})

        # Calculate TEMA manually with min_periods masking
        w = 3
        alpha = 2.0 / (w + 1)

        # Calculate ema1 with min_periods masking
        ema1_raw = pl.col('x').ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
        ema1_count = pl.col('x').is_not_null().cast(pl.Int64).cum_sum()
        ema1 = pl.when(ema1_count >= w).then(ema1_raw).otherwise(None)

        # Calculate ema2 with min_periods masking
        ema2_raw = ema1.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
        ema2_count = ema1.is_not_null().cast(pl.Int64).cum_sum()
        ema2 = pl.when(ema2_count >= w).then(ema2_raw).otherwise(None)

        # Calculate ema3 with min_periods masking
        ema3_raw = ema2.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
        ema3_count = ema2.is_not_null().cast(pl.Int64).cum_sum()
        ema3 = pl.when(ema3_count >= w).then(ema3_raw).otherwise(None)

        # Calculate TEMA
        result = df.with_columns([
            (3 * ema1 - 3 * ema2 + ema3).alias('tema')
        ])['tema'].to_numpy()

        # First 8 rows should be NaN (window=3, ema1 needs 3, ema2 needs 3, ema3 needs 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.isnan(result[2])
        assert np.isnan(result[3])
        assert np.isnan(result[4])
        assert np.isnan(result[5])
        assert np.isnan(result[6])
        assert np.isnan(result[7])
        # Row 8 should have a value (13 finite observations for ema3)
        assert np.isfinite(result[8])


class TestRollingMeanWarmup:
    """Test rolling_mean with min_samples=window."""

    def test_bollinger_pct_b_warmup(self):
        """Bollinger %B should publish NaN until window finite observations."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        df = pl.DataFrame({"x": data})

        # Calculate Bollinger %B manually with min_samples=window
        w = 3
        result = df.with_columns([
            pl.col('x').rolling_mean(window_size=w, min_samples=w).alias('ma'),
            pl.col('x').rolling_std(window_size=w, min_samples=w).alias('std')
        ]).with_columns([
            (pl.col('ma') + 2.0 * pl.col('std')).alias('upper'),
            (pl.col('ma') - 2.0 * pl.col('std')).alias('lower')
        ]).with_columns([
            (pl.col('upper') - pl.col('lower')).alias('bandwidth')
        ]).with_columns([
            pl.when(pl.col('bandwidth') != 0)
            .then((pl.col('x') - pl.col('lower')) / pl.col('bandwidth'))
            .otherwise(None)
            .alias('pct_b')
        ])['pct_b'].to_numpy()

        # First 2 rows should be NaN (window=3, only 1-2 observations)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # Row 2 should have a value (3 observations)
        assert np.isfinite(result[2])


class TestRollingStdWarmup:
    """Test rolling_std with min_samples=window."""

    def test_bollinger_width_warmup(self):
        """Bollinger Width should publish NaN until window finite observations."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        df = pl.DataFrame({"x": data})

        # Calculate Bollinger Width manually with min_samples=window
        w = 3
        result = df.with_columns([
            pl.col('x').rolling_mean(window_size=w, min_samples=w).alias('ma'),
            pl.col('x').rolling_std(window_size=w, min_samples=w).alias('std')
        ]).with_columns([
            pl.when(pl.col('ma') == 0)
            .then(None)
            .otherwise(2 * 2.0 * pl.col('std') / pl.col('ma'))
            .alias('width')
        ])['width'].to_numpy()

        # First 2 rows should be NaN (window=3, only 1-2 observations)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # Row 2 should have a value (3 observations)
        assert np.isfinite(result[2])


class TestRollingSumWarmup:
    """Test rolling_sum with min_samples=window."""

    def test_donchian_lower_warmup(self):
        """Donchian Lower should publish NaN until window finite observations."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        df = pl.DataFrame({"x": data})

        # Calculate Donchian Lower manually with min_samples=window
        w = 3
        result = df.with_columns([
            pl.col('x').rolling_min(window_size=w, min_samples=w).alias('donchian_lower')
        ])['donchian_lower'].to_numpy()

        # First 2 rows should be NaN (window=3, only 1-2 observations)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # Row 2 should have a value (3 observations)
        assert np.isfinite(result[2])
