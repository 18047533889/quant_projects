"""Inactive alpha tools kept for audit and reproducibility.

This module is intentionally empty in the initial seed version. Tools are moved
here only after they were previously active and later replaced or deprecated.
"""

# --- inactive_alpha_tool: robust_volume_zscore deactivated_at=2026-07-07T13:57:13.965030+00:00
def robust_volume_zscore(volume, window=20, clip=3.0):
    """Compute robust z-score of volume relative to trailing median and MAD.

    Economic meaning:
        Measures how many MADs the current volume deviates from the trailing median,
        using a robust normalization that resists outliers. High absolute values 
        indicate unusual volume, which may signal informed trading or regime shifts.
    Args:
        volume: pd.Series of daily trading volume for one stock.
        window: Trailing lookback window for median and MAD calculation.
        clip: Max absolute value for clipping; values beyond are capped.
    Returns:
        pd.Series of robust z-scores, aligned to the input index.
    Required columns:
        volume.
    Leakage notes:
        Uses only past observations via rolling median and MAD; no future data.
    """
    rolling_median = volume.rolling(window, min_periods=window).median()
    mad = (volume - rolling_median).abs().rolling(window, min_periods=window).median()
    mad = mad.replace(0, np.nan)
    z = (volume - rolling_median) / mad
    z = z.clip(-clip, clip)
    return z

