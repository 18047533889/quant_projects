"""FactorEngine-native, lagged smoothing on sparse long panels."""
from __future__ import annotations

import numbers

import numpy as np
import pandas as pd


def execute_lagged_sma(frame: pd.DataFrame, *, window: int) -> pd.Series:
    """Run FE ``ts_mean(ts_delay(x, 1), window, min_periods=window)``.

    FE's long Polars emitter groups by instrument and orders by observation
    time. It therefore rolls over each asset's observed rows without widening
    the panel or inventing values for absent dates.
    """
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, numbers.Integral) or window < 1:
        raise ValueError("window must be a positive integer")
    if frame.empty:
        return pd.Series(index=frame.index, dtype=float, name="value")
    if not frame.groupby("asset_id", sort=False, observed=True)["date"].is_monotonic_increasing.all():
        raise ValueError("per-asset dates must be monotone increasing")

    import polars as pl
    from factor_engine.backend.polars_expr_emitter import compile_plan_to_polars
    from factor_engine.expr.cleaned_call import CleanedCall
    from factor_engine.expr.column import ColumnRef
    from factor_engine.ir.analyzer import Analyzer

    base_frame = frame.loc[:, ["date", "asset_id", "value"]].rename(
        columns={"date": "ts", "asset_id": "inst", "value": "_v"}
    ).reset_index(drop=True)
    delayed = CleanedCall("ts_delay", (ColumnRef("_v"),), (("n", 1),))
    expression = CleanedCall(
        "ts_mean", (delayed,), (("window", int(window)), ("min_periods", int(window)))
    )
    plan = Analyzer(production=False).lower(expression).ir
    base = pl.from_pandas(base_frame, include_index=False, nan_to_null=False).lazy()
    compiled = compile_plan_to_polars(plan, base)
    if compiled is None:
        raise RuntimeError("FactorEngine could not compile lagged SMA on a long Polars panel")
    result = compiled.frame.collect()

    expected_dates = pd.Index(frame["date"])
    actual_dates = pd.Index(result.get_column("ts").to_pandas(date_as_object=True))
    expected_assets = frame["asset_id"].to_numpy(copy=False)
    actual_assets = result.get_column("inst").to_numpy()
    if not actual_dates.equals(expected_dates) or not np.array_equal(actual_assets, expected_assets):
        raise RuntimeError("FactorEngine long SMA changed row identity or order")
    return pd.Series(result.get_column("_v").to_numpy(), index=frame.index, name="value")
