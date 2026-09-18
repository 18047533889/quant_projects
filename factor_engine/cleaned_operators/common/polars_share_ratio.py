"""Shared Polars kernels for bounded share-count ratios."""
from __future__ import annotations

import polars as pl

from factor_engine.cleaned_operators.common._polars_bridge import (
    numeric_cols,
    verify_frames_share_identity,
)


def bounded_share_ratio(
    numerator: pl.DataFrame, denominator: pl.DataFrame
) -> pl.DataFrame:
    """Return numerator/denominator only inside the valid share-count domain."""
    verify_frames_share_identity(
        "bounded holder share ratio", numerator, denominator
    )
    expressions = []
    for col in numeric_cols(numerator):
        num = numerator[col]
        den = denominator[col]
        valid = (
            num.is_finite()
            & den.is_finite()
            & (num >= 0.0)
            & (den > 0.0)
            & (num <= den)
        )
        expressions.append(
            pl.when(valid).then(num / den).otherwise(float("nan")).alias(col)
        )
    return numerator.with_columns(expressions)
