"""Native membership-transition counting with explicit unknown-state semantics."""
from __future__ import annotations
import polars as pl
from ._polars_bridge import SKIP
from factor_engine.cleaned_operators.parameter_validation import strict_integer


def reconstitution_churn(member: pl.DataFrame, window: int = 60) -> pl.DataFrame:
    window = strict_integer(window, "window", minimum=1)
    exprs = []
    for column in member.columns:
        if column in SKIP:
            continue
        values = member[column]
        valid = values.is_finite().fill_null(False)
        if (valid & ~values.is_in([0, 1])).any():
            raise ValueError("member must be a ConditionBool (values in {0, 1} with nonfinite as unknown)")
        current = pl.col(column)
        known = current.is_finite().fill_null(False)
        both_known = known & known.shift(1).fill_null(False)
        event = pl.when(both_known).then(
            (current != current.shift(1)).cast(pl.Float64)
        ).otherwise(0.)
        exprs.append(event.rolling_sum(window_size=window, min_samples=window).alias(column))
    return member.with_columns(exprs)
