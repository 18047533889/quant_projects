"""FactorEngine-native, lagged smoothing on sparse long panels."""
from __future__ import annotations
import pandas as pd

def execute_lagged_sma(frame: pd.DataFrame, *, window: int) -> pd.Series:
    """Run the complete-window FE lagged-SMA recipe on a sparse long panel."""
    from factor_engine.backend.long_smoothing import lagged_mean
    return lagged_mean(frame, window=window, min_periods=window)
