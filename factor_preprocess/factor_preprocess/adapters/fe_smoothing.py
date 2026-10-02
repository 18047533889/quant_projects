"""FactorEngine-backed long-panel smoothing adapter (lazy FE dependency)."""
from __future__ import annotations
import pandas as pd
from factor_preprocess.errors import GovernanceError

TRAILING_SMA_RECIPE = "FE_COMPOSITE:long_smoothing.lagged_mean:v1"
ROLLING_MEAN_RECIPE = TRAILING_SMA_RECIPE
TRAILING_MEDIAN_RECIPE = "FE_COMPOSITE:long_smoothing.lagged_median:v1"
ROLLING_STD_RECIPE = "FE_COMPOSITE:long_smoothing.lagged_std:v1"
ROLLING_ZSCORE_RECIPE = "FE_COMPOSITE:long_smoothing.lagged_zscore:v1"
EWMA_RECIPE = "FE_COMPOSITE:long_smoothing.lagged_ewma:v1"
IIR_RECIPE = "FE_COMPOSITE:long_ewm.lagged_iir_lowpass:v1"
ROBUST_EWMA_RECIPE = "FE_COMPOSITE:long_robust_ewm.lagged_robust_ewma:v1"


def get_fe_composite_executor(name: str, recipe_identity: str | None):
    """Resolve known semantics, not a content hash of the FE implementation."""
    recipes = {
        "trailing_sma": TRAILING_SMA_RECIPE,
        "rolling_mean": ROLLING_MEAN_RECIPE,
        "trailing_median": TRAILING_MEDIAN_RECIPE,
        "rolling_std": ROLLING_STD_RECIPE,
        "rolling_zscore": ROLLING_ZSCORE_RECIPE,
        "ewma": EWMA_RECIPE,
        "one_sided_iir_lowpass": IIR_RECIPE,
        "robust_ewma": ROBUST_EWMA_RECIPE,
    }
    if name not in recipes or recipe_identity != recipes[name]:
        raise GovernanceError(f"FE composite recipe is not registered for {name!r}")
    # Recursive EWMA is native FE math and does not need the expression
    # emitter used by windowed smoothing composites.
    if name == "ewma":
        try:
            import polars  # noqa: F401
            from factor_engine.backend.long_smoothing import lagged_ewma  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            return None
        return execute_ewma
    if name == "one_sided_iir_lowpass":
        try:
            import polars  # noqa: F401
            from factor_engine.backend.long_ewm import lagged_iir_lowpass  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            return None
        return execute_iir_lowpass
    if name == "robust_ewma":
        try:
            import polars  # noqa: F401
            from factor_engine.backend.long_robust_ewm import lagged_robust_ewma  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            return None
        return execute_robust_ewma
    try:
        import polars  # noqa: F401
        from factor_engine.backend.polars_expr_emitter import compile_plan_to_polars  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return None
    if name in {"trailing_sma", "rolling_mean"}:
        from factor_engine.backend.long_smoothing import lagged_mean  # noqa: F401
        return execute_trailing_sma
    if name == "rolling_std":
        from factor_engine.backend.long_smoothing import lagged_std  # noqa: F401
        return execute_rolling_std
    if name == "rolling_zscore":
        from factor_engine.backend.long_smoothing import lagged_zscore  # noqa: F401
        return execute_rolling_zscore
    from factor_engine.backend.long_smoothing import lagged_median  # noqa: F401
    return execute_trailing_median

def execute_trailing_sma(values: pd.DataFrame, window: int, min_periods: int | None = None,
                         asset_col: str = "asset_id", time_col: str = "date",
                         value_col: str = "value") -> pd.Series:
    """Delegate trailing SMA to FE's compositional long-panel recipe."""
    from factor_engine.backend.long_smoothing import lagged_mean
    return lagged_mean(values, window=window, min_periods=min_periods,
                       asset_col=asset_col, time_col=time_col, value_col=value_col).rename(None)

def execute_trailing_median(values: pd.DataFrame, window: int, min_periods: int | None = None,
                            asset_col: str = "asset_id", time_col: str = "date",
                            value_col: str = "value") -> pd.Series:
    """Delegate trailing median to FE's lagged median recipe."""
    from factor_engine.backend.long_smoothing import lagged_median
    return lagged_median(values, window=window, min_periods=min_periods,
                         asset_col=asset_col, time_col=time_col,
                         value_col=value_col).rename(None)


def execute_rolling_std(values: pd.DataFrame, window: int, min_periods: int | None = None,
                        ddof: float = 1, asset_col: str = "asset_id",
                        time_col: str = "date", value_col: str = "value") -> pd.Series:
    """Delegate causal rolling std to the FE lagged long-panel recipe."""
    from factor_engine.backend.long_smoothing import lagged_std
    return lagged_std(values, window=window, min_periods=min_periods, ddof=ddof,
                      asset_col=asset_col, time_col=time_col, value_col=value_col).rename(None)


def execute_rolling_zscore(values: pd.DataFrame, window: int, min_periods: int | None = None,
                           ddof: float = 1, asset_col: str = "asset_id",
                           time_col: str = "date", value_col: str = "value") -> pd.Series:
    """Delegate causal rolling z-score to FE's lagged long-panel recipe."""
    from factor_engine.backend.long_smoothing import lagged_zscore
    return lagged_zscore(values, window=window, min_periods=min_periods, ddof=ddof,
                         asset_col=asset_col, time_col=time_col, value_col=value_col).rename(None)


def execute_ewma(values: pd.DataFrame, halflife: float, min_periods: int = 1,
                 asset_col: str = "asset_id", time_col: str = "date",
                 value_col: str = "value") -> pd.Series:
    """Delegate causal EWMA to FE's native lagged long-panel recipe."""
    from factor_engine.backend.long_smoothing import lagged_ewma
    return lagged_ewma(
        values, halflife=halflife, min_periods=min_periods,
        asset_col=asset_col, time_col=time_col, value_col=value_col,
    ).rename(None)


def execute_iir_lowpass(values: pd.DataFrame, alpha: float,
                        asset_col: str = "asset_id", time_col: str = "date",
                        value_col: str = "value") -> pd.Series:
    """Delegate exact-alpha, gap-reset IIR execution to FactorEngine."""
    from factor_engine.backend.long_ewm import lagged_iir_lowpass
    return lagged_iir_lowpass(
        values, alpha=alpha, asset_col=asset_col, time_col=time_col,
        value_col=value_col,
    ).rename(None)


def execute_robust_ewma(values: pd.DataFrame, halflife: float,
                        winsor_std: float = 4.0, min_periods: int = 1,
                        asset_col: str = "asset_id", time_col: str = "date",
                        value_col: str = "value") -> pd.Series:
    """Delegate lagged rolling-winsorized EWMA to the distinct FE recipe."""
    from factor_engine.backend.long_robust_ewm import lagged_robust_ewma
    return lagged_robust_ewma(
        values, halflife=halflife, winsor_std=winsor_std,
        min_periods=min_periods, asset_col=asset_col,
        time_col=time_col, value_col=value_col,
    ).rename(None)
