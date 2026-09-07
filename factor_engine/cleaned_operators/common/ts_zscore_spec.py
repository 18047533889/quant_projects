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
