# -*- coding: utf-8 -*-
"""Numerically deterministic high-moment semantic references."""
from __future__ import annotations

import numpy as np

from factor_engine.cleaned_operators.base import OperatorMetadata, ParamSpec, ParamRole, SeriesOperator, register_operator


def _unbiased_excess_kurtosis(values) -> float:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        return np.nan
    count = len(array)
    if count < 4:
        return np.nan
    with np.errstate(over="ignore", invalid="ignore"):
        offsets = array - array[0]
    if not np.isfinite(offsets).all():
        scale = float(np.max(np.abs(array)))
        scaled = array / scale
        offsets = scaled - scaled[0]
    offset_scale = float(np.max(np.abs(offsets)))
    if offset_scale == 0.0:
        return np.nan
    normalized_offsets = offsets / offset_scale
    centered = normalized_offsets - float(np.mean(normalized_offsets))
    second = float(np.sum(centered * centered))
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
        panel_params=("x",), scalar_params=("window",),
        param_specs={"window": ParamSpec(dtype=int, min=4, default=20, param_role=ParamRole.HORIZON)},
        window_semantics="exact_rows",
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
        from factor_engine.cleaned_operators.common.strict_params import strict_int
        length = strict_int(window, "window", minimum=4)
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


def kurt_polars(x, window=20):
    """Same full finite-window estimator without a Pandas panel conversion."""
    import polars as pl
    from factor_engine.cleaned_operators.base_polars import PANEL_SKIP_COLUMNS
    from factor_engine.cleaned_operators.common.strict_params import strict_int
    length = strict_int(window, "window", minimum=4)
    series_input = isinstance(x, pl.Series)
    frame = x.to_frame() if series_input else x
    columns = []
    for name in frame.columns:
        if name in PANEL_SKIP_COLUMNS:
            continue
        values = np.asarray(frame[name].to_numpy(), dtype=float)
        out = np.full(len(values), np.nan)
        for end in range(length - 1, len(values)):
            out[end] = _unbiased_excess_kurtosis(values[end-length+1:end+1])
        columns.append(pl.Series(name, out, dtype=pl.Float64))
    result = frame.with_columns(columns)
    return result[x.name] if series_input else result


def kurt_physical_spec():
    import hashlib
    from pathlib import Path
    from factor_engine.backend.contracts import PhysicalImplementationSpec, ExecutionKind
    return PhysicalImplementationSpec(
        canonical="ts_kurt", backend="polars",
        execution_kind=ExecutionKind.POLARS_NUMPY_KERNEL,
        materializes_full_panel=True, requires_sorted=True, supports_nan=True, supports_nulls=True,
        supports_inf=True,
        implementation_source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        kernel_identity="stable_high_moments_v2._unbiased_excess_kurtosis",
        notes="Full finite window, unbiased Fisher excess, constant window undefined (NaN); CPU NumPy.")
