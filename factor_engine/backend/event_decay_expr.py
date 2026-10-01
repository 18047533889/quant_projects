"""Stable pure-Expr event sums shared by panel and long lowering."""
from __future__ import annotations
import math
import polars as pl


def event_decay_sum_expr(contribution, step, *, half_life, groups, order_by=None):
    """Adjusted EWM times geometric weights avoids inverse-weight overflow."""
    weight = 0.5 ** (1.0 / half_life)
    alpha = 1.0 - weight
    window = {"order_by": order_by} if order_by is not None else {}
    if alpha == 0.0:
        total = contribution.cum_sum()
        return total if groups is None else total.over(groups, **window)
    mean = contribution.ewm_mean(alpha=alpha, adjust=True, min_samples=1)
    if groups is not None:
        mean = mean.over(groups, **window)
    exponent = step.cast(pl.Float64) * math.log(weight)
    # Polars lacks Expr.expm1; this branch avoids cancellation near unit decay.
    small = -exponent * (1.0 + exponent * (0.5 + exponent * (
        1.0 / 6.0 + exponent * (1.0 / 24.0 + exponent * (
            1.0 / 120.0 + exponent / 720.0
        ))
    )))
    geometric = pl.when(exponent.abs() < 1e-4).then(small).otherwise(1.0 - exponent.exp()) / alpha
    return mean * geometric
