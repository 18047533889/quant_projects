# -*- coding: utf-8 -*-
"""Numerically deterministic high-moment semantic references."""
from __future__ import annotations

import numpy as np

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator


def _unbiased_excess_kurtosis(values) -> float:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        return np.nan
    count = len(array)
    if count < 4:
        return np.nan
    centered = array - float(np.mean(array))
    second = float(np.sum(centered * centered))
    if second <= 0.0:
        return np.nan
    fourth = float(np.sum(centered ** 4))
    biased_excess = count * fourth / (second * second) - 3.0
    return float(
        (count - 1)
        / ((count - 2) * (count - 3))
        * ((count + 1) * biased_excess + 6.0)
    )


class StableTsKurt(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_kurt",
        category="time_series",
        description=(
            "Prefix-stable unbiased Fisher excess kurtosis over an explicit window."
        ),
        param_names=["x", "window"],
        return_type="series",
        tags=[
            "time_series",
            "pit_safe",
            "causal",
            "bounded_history",
            "numerically_stable",
            "production_repair",
        ],
    )

    def _calculate_series(self, x, window=20):
        length = int(window)
        if length < 4:
            raise ValueError("ts_kurt window must be >= 4")
        return x.rolling(length, min_periods=length).apply(
            _unbiased_excess_kurtosis,
            raw=True,
        )


register_operator(
    name="ts_kurt",
    category="time_series",
    business_category="time_series",
    canonical="ts_kurt",
    source="stable_high_moments_v2",
    backend="pandas_numpy",
    status="production",
)(StableTsKurt)
