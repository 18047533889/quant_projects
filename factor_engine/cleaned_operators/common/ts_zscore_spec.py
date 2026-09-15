"""Single parameter authority for canonical ``ts_zscore`` implementations."""
from __future__ import annotations

from dataclasses import dataclass

from factor_engine.backend.window_spec import WindowSpec


@dataclass(frozen=True)
class TSZScoreSpec:
    window: WindowSpec
    includes_current_bar: bool = True
    zero_std_policy: str = "zero"

    @classmethod
    def resolve(cls, *, window=20, min_periods=1, null_policy="ignore",
                nan_policy="propagate", includes_current_bar=True, ddof=1,
                zero_std_policy="zero") -> "TSZScoreSpec":
        if type(window) is not int or window < 1:
            raise ValueError("window must be a positive integer")
        if type(min_periods) is not int or not 1 <= min_periods <= window:
            raise ValueError("min_periods must be an integer in [1, window]")
        if type(ddof) is not int or ddof not in {0, 1}:
            raise ValueError("ddof must be 0 or 1")
        if type(includes_current_bar) is not bool:
            raise ValueError("includes_current_bar must be bool")
        if zero_std_policy not in {"zero", "nan"}:
            raise ValueError("zero_std_policy must be 'zero' or 'nan'")
        spec = WindowSpec(
            size=window, min_periods=min_periods, ddof=ddof,
            null_policy=null_policy, nan_policy=nan_policy,
        )
        if spec.null_policy.value not in {"ignore", "propagate"}:
            raise ValueError("ts_zscore null_policy supports only ignore/propagate")
        if spec.nan_policy not in {"ignore", "propagate"}:
            raise ValueError("ts_zscore nan_policy supports only ignore/propagate")
        return cls(spec, includes_current_bar, zero_std_policy)


def polars_zscore(
    x: pl.DataFrame,
    window: int = 20,
    min_periods: int = 1,
    null_policy: str = "ignore",
    nan_policy: str = "propagate",
    includes_current_bar: bool = True,
    ddof: int = 1,
    zero_std_policy: str = "zero",
    **kwargs,
) -> pl.DataFrame:
    import polars as pl

    spec = TSZScoreSpec.resolve(
        window=window, min_periods=min_periods, null_policy=null_policy,
        nan_policy=nan_policy, includes_current_bar=includes_current_bar,
        ddof=ddof, zero_std_policy=zero_std_policy,
    )
    exprs = []
    propagate = (
        spec.window.nan_policy == "propagate"
        or spec.window.null_policy.value == "propagate"
    )
    for c in (c for c in x.columns if c not in {"date", "stock_code"}):
        current = pl.col(c).cast(pl.Float64).fill_nan(None)
        current = pl.when(current.is_finite()).then(current).otherwise(None)
        stats = current if spec.includes_current_bar else current.shift(1)
        mean = stats.rolling_mean(spec.window.size, min_samples=spec.window.min_periods)
        std = stats.rolling_std(
            spec.window.size, min_samples=spec.window.min_periods,
            ddof=spec.window.ddof,
        )
        value = pl.when(current.is_null() | std.is_null()).then(None)
        if propagate:
            bad = stats.is_null().cast(pl.Int64).rolling_sum(
                spec.window.size, min_samples=1
            ) > 0
            value = value.when(bad).then(None)
        zero_value = 0.0 if spec.zero_std_policy == "zero" else None
        value = value.when(std == 0).then(zero_value).otherwise((current - mean) / std)
        exprs.append(value.alias(c))
    return x.with_columns(exprs)
