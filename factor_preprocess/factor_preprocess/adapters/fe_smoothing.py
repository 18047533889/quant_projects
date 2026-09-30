"""FactorEngine-backed long-panel smoothing adapter (lazy FE dependency)."""
from __future__ import annotations
import pandas as pd
from factor_preprocess.errors import GovernanceError

TRAILING_SMA_RECIPE = "FE_COMPOSITE:long_smoothing.lagged_mean:v1"


def get_fe_composite_executor(name: str, recipe_identity: str | None):
    """Resolve known semantics, not a content hash of the FE implementation."""
    if name != "trailing_sma" or recipe_identity != TRAILING_SMA_RECIPE:
        raise GovernanceError(f"FE composite recipe is not registered for {name!r}")
    try:
        import polars  # noqa: F401
        from factor_engine.backend.polars_expr_emitter import compile_plan_to_polars  # noqa: F401
        from factor_engine.backend.long_smoothing import lagged_mean  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return None
    return execute_trailing_sma

def execute_trailing_sma(values: pd.DataFrame, window: int, min_periods: int | None = None,
                         asset_col: str = "asset_id", time_col: str = "date",
                         value_col: str = "value") -> pd.Series:
    """Delegate trailing SMA to FE's compositional long-panel recipe."""
    from factor_engine.backend.long_smoothing import lagged_mean
    return lagged_mean(values, window=window, min_periods=min_periods,
                       asset_col=asset_col, time_col=time_col, value_col=value_col).rename(None)
