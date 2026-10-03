"""Explicit, non-default FE candidate for finite-anchor cross-sectional z-scores.

This module does not replace the legacy ``zscore`` operator. Its Polars backend
uses native expressions over a wide panel, and registration is explicit. NaN
and null cells stay missing; any Inf in a row causes non-missing cells in that
row to receive ``constant_value``.
"""

from __future__ import annotations
import hashlib
from numbers import Integral
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import polars as pl
from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.base import Operator as PandasOperator
from factor_engine.cleaned_operators.base_polars import Operator, OperatorMetadata
from factor_engine.cleaned_operators.registry import OperatorRegistry

_SOURCE = "factor_engine.cleaned_operators.polars_native.cs_zscore_finite_anchor_v2"
_AXES = frozenset(
    {
        "date",
        "timestamp",
        "trade_date",
        "datetime",
        "stock_code",
        "instrument",
        "symbol",
        "__fe_time__",
        "__fe_instrument__",
        "session",
    }
)


def _validate(ddof: Any, constant_value: Any) -> tuple[int, float]:
    if isinstance(ddof, bool) or not isinstance(ddof, Integral) or ddof < 0:
        raise ValueError("ddof must be a non-negative integer")
    if isinstance(
        constant_value, (bool, complex, np.complexfloating)
    ) or not isinstance(constant_value, (int, float, np.number)):
        raise ValueError("constant_value must be a finite real number")
    try:
        converted = float(constant_value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("constant_value must be a finite real number") from exc
    if not np.isfinite(converted):
        raise ValueError("constant_value must be a finite real number")
    return int(ddof), converted


def _finite_anchor_expr(
    frame: pl.DataFrame, ddof: int, constant_value: float
) -> pl.DataFrame:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("panel must be a Polars DataFrame")
    names = [n for n in frame.columns if n not in _AXES]
    ddof, constant_value = _validate(ddof, constant_value)
    if not names:
        raise ValueError("panel must contain numeric instrument columns")
    if any(
        frame.schema[n] == pl.Boolean
        or (not frame.schema[n].is_numeric() and frame.schema[n] != pl.Null)
        for n in names
    ):
        raise TypeError("instrument columns must be numeric")
    if ddof >= len(names):
        casts = [pl.col(n).cast(pl.Float64, strict=False).alias(n) for n in names]
        frame = frame.with_columns(casts)
        return frame.with_columns(
            [
                pl.when(pl.col(n).is_null() | pl.col(n).is_nan())
                .then(pl.col(n))
                .otherwise(constant_value)
                .alias(n)
                for n in names
            ]
        )
    prefix = "__csa_"
    while any(n.startswith(prefix) for n in frame.columns):
        prefix = "_" + prefix
    frame = frame.with_columns(
        [pl.col(n).cast(pl.Float64, strict=False).alias(n) for n in names]
    )
    xs = [pl.col(n) for n in names]
    valid = [x.is_not_null() & ~x.is_nan() for x in xs]
    finite = [ok & x.is_finite() for x, ok in zip(xs, valid)]
    count_name, inf_name, anchor_name, scale_name, mean_name, ss_name, std_name = [
        prefix + s for s in ("count", "inf", "anchor", "scale", "mean", "ss", "std")
    ]
    lo = pl.min_horizontal(
        [pl.when(f).then(x).otherwise(float("inf")) for x, f in zip(xs, finite)]
    )
    hi = pl.max_horizontal(
        [pl.when(f).then(x).otherwise(float("-inf")) for x, f in zip(xs, finite)]
    )
    frame = frame.with_columns(
        [
            pl.sum_horizontal([f.cast(pl.UInt32) for f in finite]).alias(count_name),
            pl.any_horizontal([ok & x.is_infinite() for x, ok in zip(xs, valid)]).alias(
                inf_name
            ),
            (lo / 2.0 + hi / 2.0).alias(anchor_name),
        ]
    )
    anchor = pl.col(anchor_name)
    delta_names = [prefix + f"q{i}" for i in range(len(names))]
    frame = frame.with_columns(
        [
            pl.when(f).then((x - anchor).abs()).otherwise(None).alias(delta_names[i])
            for i, (x, f) in enumerate(zip(xs, finite))
        ]
    )
    scale = pl.max_horizontal([pl.col(n) for n in delta_names])
    frame = frame.with_columns(scale.alias(scale_name))
    scale_expr = pl.when(pl.col(scale_name) > 0).then(pl.col(scale_name)).otherwise(1.0)
    q_names = [prefix + f"unit{i}" for i in range(len(names))]
    frame = frame.with_columns(
        [
            pl.when(f).then((x - anchor) / scale_expr).otherwise(None).alias(q_names[i])
            for i, (x, f) in enumerate(zip(xs, finite))
        ]
    )
    frame = frame.with_columns(
        (pl.sum_horizontal([pl.col(n) for n in q_names]) / pl.col(count_name)).alias(
            mean_name
        )
    )
    mean = pl.col(mean_name)
    c_names = [prefix + f"center{i}" for i in range(len(names))]
    frame = frame.with_columns(
        [(pl.col(qn) - mean).alias(cn) for qn, cn in zip(q_names, c_names)]
    )
    frame = frame.with_columns(
        pl.sum_horizontal([pl.col(n).pow(2) for n in c_names]).alias(ss_name)
    )
    count_float = pl.col(count_name).cast(pl.Float64)
    frame = frame.with_columns(
        (pl.col(ss_name) / (count_float - float(ddof))).sqrt().alias(std_name)
    )
    std = pl.col(std_name)
    temp_names = [
        count_name,
        inf_name,
        anchor_name,
        scale_name,
        mean_name,
        ss_name,
        std_name,
        *delta_names,
        *q_names,
        *c_names,
    ]
    out = []
    for name, x, cn, ok in zip(names, xs, c_names, valid):
        out.append(
            pl.when(x.is_null() | x.is_nan())
            .then(x)
            .when(
                pl.col(inf_name)
                | (pl.col(count_name) <= ddof)
                | (pl.col(scale_name) == 0)
                | (pl.col(ss_name) == 0)
            )
            .then(constant_value)
            .otherwise(pl.col(cn) / std)
            .alias(name)
        )
    return frame.with_columns(out).drop(temp_names)


def cs_zscore_finite_anchor_v2_numpy(
    panel: pd.DataFrame, ddof: int = 1, constant_value: float = 0.0
) -> pd.DataFrame:
    ddof, constant_value = _validate(ddof, constant_value)
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    result = panel.copy()
    names = [n for n in result.columns if n not in _AXES]
    if not names:
        raise ValueError("panel must contain numeric instrument columns")
    for n in names:
        if (
            not pd.api.types.is_numeric_dtype(result[n])
            or pd.api.types.is_bool_dtype(result[n].dtype)
            or pd.api.types.is_complex_dtype(result[n])
        ):
            raise TypeError("instrument columns must be real numeric values")
    a = result[names].to_numpy(dtype=np.float64, na_value=np.nan)
    out = a.copy()
    for i, row in enumerate(a):
        missing = np.isnan(row)
        inf = np.isinf(row)
        finite = np.isfinite(row)
        vals = row[finite]
        if inf.any():
            out[i, ~missing] = constant_value
            continue
        if len(vals) <= ddof or len(vals) < 2:
            out[i, ~missing] = constant_value
            continue
        anchor = float(vals.min() / 2.0 + vals.max() / 2.0)
        delta = vals - anchor
        scale = float(np.max(np.abs(delta)))
        if scale == 0:
            out[i, ~missing] = constant_value
            continue
        q = delta / scale
        mean = float(q.mean())
        centered = q - mean
        ss = float(np.dot(centered, centered))
        denom = len(q) - ddof
        if ss == 0:
            out[i, ~missing] = constant_value
        else:
            z = centered * np.sqrt(denom / ss)
            out[i, finite] = z
    for column_index, name in enumerate(names):
        result[name] = pd.Series(
            out[:, column_index], index=result.index, dtype=np.float64
        )
    return result


class _Reference(PandasOperator):
    _HANDLES_CALL_CONTRACT = True

    def __init__(self, metadata):
        self.metadata = metadata

    def calculate(self, *args, **kwargs):
        from factor_engine.cleaned_operators.base import validate_operator_call

        args, kwargs = validate_operator_call(self, args, kwargs)
        bound = dict(zip(self.metadata.param_names, args))
        bound.update(kwargs)
        return cs_zscore_finite_anchor_v2_numpy(
            bound["x"],
            ddof=bound.get("ddof", 1),
            constant_value=bound.get("constant_value", 0.0),
        )


class _Native(Operator):
    _HANDLES_CALL_CONTRACT = True

    def __init__(self, metadata, spec):
        self.metadata, self._physical_spec = metadata, spec

    def calculate(self, *args, **kwargs):
        args, kwargs = self._prepare_call(args, kwargs)
        bound = dict(zip(self.metadata.param_names, args))
        bound.update(kwargs)
        if "x" not in bound:
            raise TypeError("expects one panel")
        ddof, constant = _validate(
            bound.get("ddof", 1), bound.get("constant_value", 0.0)
        )
        if not isinstance(bound["x"], pl.DataFrame):
            raise TypeError("panel must be a Polars DataFrame")
        return _finite_anchor_expr(bound["x"], ddof, constant)

    _calculate_series = calculate


def register_cs_zscore_finite_anchor_v2() -> tuple[str, ...]:
    from factor_engine.cleaned_operators.base import ParamSpec

    metadata = OperatorMetadata(
        name="cs_zscore_finite_anchor_v2",
        category="cross_section",
        description="Finite-anchor cross-sectional z-score candidate",
        param_names=["x", "ddof", "constant_value"],
        param_types={"ddof": int, "constant_value": float},
        param_specs={
            "ddof": ParamSpec(dtype=int, min=0, default=1),
            "constant_value": ParamSpec(dtype=float, default=0.0),
        },
        tags=["preserve_panel_time_coordinate"],
        panel_params=("x",),
        panel_arity=1,
        scalar_params=("ddof", "constant_value"),
        total_positional_arity=3,
    )
    from factor_engine.cleaned_operators.base import (
        OperatorMetadata as PandasMetadata,
        ParamSpec as PandasParamSpec,
    )

    pandas_metadata = PandasMetadata(
        name=metadata.name,
        category=metadata.category,
        description=metadata.description,
        param_names=list(metadata.param_names),
        param_types=dict(metadata.param_types),
        param_specs={
            "ddof": PandasParamSpec(dtype=int, min=0, default=1),
            "constant_value": PandasParamSpec(dtype=float, default=0.0),
        },
        tags=list(metadata.tags),
        panel_params=("x",),
        panel_arity=1,
        scalar_params=("ddof", "constant_value"),
        total_positional_arity=3,
    )
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    spec = PhysicalImplementationSpec(
        canonical=metadata.name,
        backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        supports_lazy=False,
        supports_streaming=False,
        materializes_full_panel=True,
        supports_nulls=True,
        supports_nan=True,
        supports_inf=True,
        implementation_source_hash=digest,
        emitter_identity=f"{_SOURCE}:horizontal_scaled_moments:v1",
        kernel_identity=f"{_SOURCE}:finite_anchor",
        parameter_domain_hash=hashlib.sha256(
            repr(metadata.param_specs).encode()
        ).hexdigest(),
        semantic_contract_hash=hashlib.sha256(
            b"finite-anchor-zscore-v2:finite-only;nan-null-preserved;inf-row-constant;integer-ddof"
        ).hexdigest(),
        notes="Candidate only; explicit registration required; NaN/null are preserved and any Inf causes constant fill for non-missing cells.",
    )
    if OperatorRegistry.lifecycle() != "building":
        raise RuntimeError(
            "explicit candidate registration requires registry BUILDING lifecycle"
        )
    OperatorRegistry.register(
        _Reference(pandas_metadata),
        canonical=metadata.name,
        backend="pandas_numpy",
        source=_SOURCE,
        status="implemented",
    )
    OperatorRegistry.register(
        _Native(metadata, spec),
        canonical=metadata.name,
        backend="polars",
        source=_SOURCE,
        status="implemented",
    )
    return (metadata.name,)


__all__ = ["cs_zscore_finite_anchor_v2_numpy", "register_cs_zscore_finite_anchor_v2"]
