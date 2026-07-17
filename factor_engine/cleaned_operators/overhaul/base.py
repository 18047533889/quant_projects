# -*- coding: utf-8 -*-
"""Shared helpers for audited operator overrides."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.base import Operator as PandasOperator
from cleaned_operators.base import OperatorMetadata as PandasMetadata
from cleaned_operators.base_polars import Operator as PolarsOperator
from cleaned_operators.base_polars import OperatorMetadata as PolarsMetadata
from cleaned_operators.registry import OperatorRegistry

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore

SKIP_COLUMNS = frozenset({"date", "stock_code", "ts", "inst", "instrument", "timestamp"})
EPS = 1e-12


def positive_int(value: Any, name: str) -> int:
    out = int(value)
    if out < 1 or float(value) != float(out):
        raise ValueError(f"{name} must be a positive integer")
    return out


def nonnegative_int(value: Any, name: str) -> int:
    out = int(value)
    if out < 0 or float(value) != float(out):
        raise ValueError(f"{name} must be a non-negative integer")
    return out


def window_params(window: Any, min_periods: Any | None, *, default_mp: int = 1) -> tuple[int, int]:
    w = positive_int(window, "window")
    mp = default_mp if min_periods is None else positive_int(min_periods, "min_periods")
    if mp > w:
        raise ValueError("min_periods must satisfy 1 <= min_periods <= window")
    return w, mp


def aligned_pd(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    if not frames:
        return ()
    base = frames[0]
    return tuple([base] + [f.reindex(index=base.index, columns=base.columns) for f in frames[1:]])


def finite_pd(x: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.isfinite(x.to_numpy(dtype=float, copy=False)),
        index=x.index,
        columns=x.columns,
    )


def frame_pd(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def pl_cols(frame: "pl.DataFrame") -> list[str]:
    return [c for c in frame.columns if c not in SKIP_COLUMNS]


def pl_base_with(frame: "pl.DataFrame", replacements: dict[str, "pl.Series"]) -> "pl.DataFrame":
    return frame.with_columns([series.alias(name) for name, series in replacements.items()])


def pl_finite(name: str) -> "pl.Expr":
    return pl.col(name).is_not_null() & pl.col(name).cast(pl.Float64, strict=False).is_finite()


def pl_unary_rolling_map(
    x: "pl.DataFrame",
    window: int,
    min_periods: int,
    func: Callable[[np.ndarray], float],
) -> "pl.DataFrame":
    replacements: dict[str, pl.Series] = {}
    for col in pl_cols(x):
        temp = pl.DataFrame({"v": x[col]})
        replacements[col] = temp.select(
            pl.col("v").rolling_map(
                func,
                window_size=window,
                min_samples=min_periods,
            ).alias("o")
        )["o"]
    return pl_base_with(x, replacements)


class PandasFunctionOperator(PandasOperator):
    def __init__(
        self,
        name: str,
        category: str,
        params: list[str],
        description: str,
        fn: Callable[..., pd.DataFrame],
    ):
        self._fn = fn
        self.metadata = PandasMetadata(
            name=name,
            category=category,
            description=description,
            param_names=params,
            return_type="series",
            tags=["daily", "panel", "pit_safe", "audited", category],
        )

    def calculate(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self._fn(*args, **kwargs)


class PolarsFunctionOperator(PolarsOperator):
    def __init__(
        self,
        name: str,
        category: str,
        params: list[str],
        description: str,
        fn: Callable[..., "pl.DataFrame"],
    ):
        self._fn = fn
        self.metadata = PolarsMetadata(
            name=name,
            category=category,
            description=description,
            param_names=params,
            return_type="series",
            tags=["daily", "panel", "pit_safe", "audited", "polars_native", category],
        )

    def calculate(self, *args: Any, **kwargs: Any) -> "pl.DataFrame":
        if pl is None:
            raise ImportError("polars is required for the Polars backend")
        return self._fn(*args, **kwargs)


@dataclass(frozen=True)
class Spec:
    category: str
    params: list[str]
    description: str
    pandas_fn: Callable[..., pd.DataFrame]
    polars_fn: Callable[..., "pl.DataFrame"] | None = None


def register_specs(specs: dict[str, Spec]) -> None:
    for name, spec in specs.items():
        OperatorRegistry.register(
            PandasFunctionOperator(name, spec.category, spec.params, spec.description, spec.pandas_fn),
            canonical=name,
            backend="pandas_numpy",
            source="operator_overhaul_audited",
            status="production",
            backend_explicit=True,
        )
        if pl is not None and spec.polars_fn is not None:
            OperatorRegistry.register(
                PolarsFunctionOperator(name, spec.category, spec.params, spec.description, spec.polars_fn),
                canonical=name,
                backend="polars",
                source="operator_overhaul_native_polars",
                status="production",
                backend_explicit=True,
            )
