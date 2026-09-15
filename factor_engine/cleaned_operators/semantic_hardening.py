# -*- coding: utf-8 -*-
"""Semantic hardening overrides for high-risk factor operators.

This module is imported after the historical pandas and Polars implementations.
The registry intentionally keeps one implementation per ``canonical/backend``;
these audited implementations therefore become the active runtime definitions.

The overrides are deliberately narrow:
- ``ts_product`` preserves zero and sign and fails closed on overflow;
- ``ts_mad`` is the rolling median absolute deviation;
- ``group_percentile`` has explicit top/bottom semantics and no silent global fallback;
- ``div_or_null`` actually returns null for null/near-zero denominators;
- daily authoring surface is aligned with the production-denied policy.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    ParamSpec, ParamRole, RelationalParamSpec,
    SeriesOperator,
    TwoVarOperator,
    register_operator,
)

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional backend
    pl = None  # type: ignore

_FLOAT_LOG_MAX = math.log(np.finfo(np.float64).max)


def _contract(name):
    if name == "group_percentile":
        return dict(panel_params=("x", "group"), scalar_params=("p", "side", "missing_group_policy"),
            param_specs={
                "p": ParamSpec(dtype=float, min=0, max=1, default=.5, param_role=ParamRole.ECONOMIC),
                "side": ParamSpec(dtype=str, choices=("top", "bottom"), default="top", param_role=ParamRole.POLICY),
                "missing_group_policy": ParamSpec(dtype=str, choices=("raise", "null", "global"),
                    default="raise", searchable=False, param_role=ParamRole.MISSING_POLICY)},
            relational_specs=[RelationalParamSpec("p > 0")])
    control = ("skipna" if name == "ts_product" else "scale")
    specs = {
        "window": ParamSpec(dtype=int, min=1, default=20, param_role=ParamRole.HORIZON),
        "min_periods": ParamSpec(dtype=int, min=1, default=None, searchable=False, param_role=ParamRole.SUPPORT_POLICY)}
    specs[control] = (ParamSpec(dtype=bool, default=True, searchable=False, param_role=ParamRole.MISSING_POLICY)
        if name == "ts_product" else ParamSpec(dtype=float, min=0, default=1.,
            searchable=False, param_role=ParamRole.NUMERICAL))
    return dict(panel_params=("x",), scalar_params=("window", "min_periods", control),
        param_specs=specs, window_semantics="exact_rows",
        relational_specs=[RelationalParamSpec("min_periods is None or min_periods <= window")])


def _strict_skipna(value):
    from factor_engine.cleaned_operators.common.strict_params import strict_bool
    return strict_bool(value, "skipna")


def _strict_scale(value):
    from factor_engine.cleaned_operators.parameter_validation import strict_finite_scalar
    return strict_finite_scalar(value, "scale", minimum=0)


def _validate_window(window: int, min_periods: int | None) -> tuple[int, int]:
    from factor_engine.cleaned_operators.common.strict_params import strict_int
    w = strict_int(window, "window", minimum=1)
    if w < 1:
        raise ValueError("window must be >= 1")
    mp = w if min_periods is None else strict_int(min_periods, "min_periods", minimum=1, maximum=w)
    if mp < 1 or mp > w:
        raise ValueError("min_periods must satisfy 1 <= min_periods <= window")
    return w, mp


def _stable_product(values: np.ndarray, *, skipna: bool = True) -> float:
    """Product with correct zero/sign semantics and overflow-to-null policy."""
    arr = np.asarray(values, dtype=float)
    if not skipna and np.isnan(arr).any():
        return np.nan
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan

    zero_mask = arr == 0.0
    if zero_mask.any():
        # 0 * inf is undefined. Returning null is safer than inventing a value.
        if np.isinf(arr).any():
            return np.nan
        return 0.0

    if np.isinf(arr).any():
        return np.nan
    # Keep a normalized mantissa and an unbounded integer exponent. This
    # avoids both log cancellation and overflow of intermediate products.
    mantissa, exponent = 1.0, 0
    for value in arr:
        part, power = math.frexp(float(value))
        mantissa, carry = math.frexp(mantissa * part)
        exponent += power + carry
    try:
        result = math.ldexp(mantissa, exponent)
    except OverflowError:
        return np.nan
    return result if math.isfinite(result) else np.nan


def _median_abs_deviation(values: np.ndarray, *, scale: float = 1.0) -> float:
    arr = np.asarray(values, dtype=float)
    if np.isinf(arr).any():
        return np.nan
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    arr = arr.astype(np.longdouble)
    median = np.median(arr)
    result = np.longdouble(scale) * np.median(np.abs(arr - median))
    return float(result) if np.isfinite(result) and abs(result) <= np.finfo(float).max else np.nan


def _rolling_numpy_panel(
    arr: np.ndarray,
    *,
    window: int,
    min_periods: int,
    func: Callable[[np.ndarray], float],
) -> np.ndarray:
    rows, cols = arr.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        series = arr[:, col]
        for row in range(rows):
            start = max(0, row - window + 1)
            values = series[start : row + 1]
            if np.count_nonzero(~np.isnan(values)) < min_periods:
                continue
            out[row, col] = func(values)
    return out


@register_operator(
    name="ts_product",
    category="time_series",
    business_category="time_series",
    canonical="ts_product",
    source="semantic_hardening",
    backend="pandas_numpy",
    status="implemented",
    replace=True,
    replacement_reason="overflow-fail-closed product overrides the gtja compatibility layer (round-7 P0 chain pinning)",
    expected_old_source="gtja_compat",
)
class TimeSeriesProductAudited(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_product",
        category="time_series",
        description=(
            "滚动乘积；保留零与负号，NaN 可跳过，溢出返回 NaN。"
        ),
        examples=["ts_product(1 + returns, 20)"],
        param_names=["x", "window", "min_periods", "skipna"],
        **_contract("ts_product"),
        return_type="series",
        tags=["time_series", "product", "pit_safe", "audited"],
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 20,
        min_periods: int | None = None,
        skipna: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        w, mp = _validate_window(window, min_periods)
        skipna = _strict_skipna(skipna)
        out = _rolling_numpy_panel(x.to_numpy(dtype=float), window=w, min_periods=mp,
            func=lambda values: _stable_product(values, skipna=skipna))
        return pd.DataFrame(out, index=x.index, columns=x.columns)


@register_operator(
    name="ts_mad",
    category="time_series",
    business_category="time_series",
    canonical="ts_mad",
    source="semantic_hardening",
    backend="pandas_numpy",
    status="implemented",
)
class TimeSeriesMedianAbsoluteDeviationAudited(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_mad",
        category="time_series",
        description="滚动中位绝对偏差 median(|x-median(x)|)。",
        examples=["ts_mad(returns, 20)", "ts_mad(returns, 20, scale=1.4826)"],
        param_names=["x", "window", "min_periods", "scale"],
        **_contract("ts_mad"),
        return_type="series",
        tags=["time_series", "mad", "robust", "pit_safe", "audited"],
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 20,
        min_periods: int | None = None,
        scale: float = 1.0,
        **kwargs,
    ) -> pd.DataFrame:
        w, mp = _validate_window(window, min_periods)
        scale = _strict_scale(scale)
        out = _rolling_numpy_panel(x.to_numpy(dtype=float), window=w, min_periods=mp,
            func=lambda values: _median_abs_deviation(values, scale=scale))
        return pd.DataFrame(out, index=x.index, columns=x.columns)


def _group_percentile_numpy(
    x: np.ndarray,
    group: np.ndarray | None,
    *,
    p: float,
    side: str,
    missing_group_policy: str,
) -> np.ndarray:
    if not 0.0 < p <= 1.0:
        raise ValueError("p must satisfy 0 < p <= 1")
    if side not in {"top", "bottom"}:
        raise ValueError("side must be 'top' or 'bottom'")
    if missing_group_policy not in {"raise", "null", "global"}:
        raise ValueError("missing_group_policy must be 'raise', 'null', or 'global'")

    out = np.full(x.shape, np.nan, dtype=float)
    for row in range(x.shape[0]):
        xv = pd.Series(x[row], dtype=float)
        gv = None if group is None else pd.Series(group[row])

        if gv is None or gv.isna().all():
            if missing_group_policy == "raise":
                raise ValueError(f"group labels are missing for row {row}")
            if missing_group_policy == "null":
                continue
            gv = pd.Series(np.zeros(len(xv), dtype=int))

        for label in gv.dropna().unique():
            mask = gv.eq(label) & np.isfinite(xv)
            if not mask.any():
                continue
            values = xv[mask]
            ranks = values.rank(
                method="average",
                pct=True,
                ascending=(side == "bottom"),
            )
            out[row, values.index.to_numpy()] = (ranks <= p).astype(float).to_numpy()
    return out


@register_operator(
    name="group_percentile",
    category="cross_sectional",
    business_category="group_neutralization",
    canonical="group_percentile",
    source="semantic_hardening",
    backend="pandas_numpy",
    status="implemented",
)
class GroupPercentileAudited(SeriesOperator):
    metadata = OperatorMetadata(
        name="group_percentile",
        category="cross_sectional",
        description=(
            "组内 top/bottom 分位掩码；默认 top。缺失分组默认报错，不静默退化为全截面。"
        ),
        examples=[
            "group_percentile(ROE, industry, 0.2)",
            "group_percentile(PE, industry, 0.2, side='bottom')",
        ],
        param_names=["x", "group", "p", "side", "missing_group_policy"],
        **_contract("group_percentile"),
        return_type="series",
        tags=["cross_sectional", "group", "percentile", "pit_safe", "audited"],
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        group: pd.DataFrame | None = None,
        p: float = 0.5,
        side: str = "top",
        missing_group_policy: str = "raise",
        **kwargs,
    ) -> pd.DataFrame:
        group_arr = None
        if group is not None:
            if (not x.index.is_unique or not x.columns.is_unique or
                not group.index.is_unique or not group.columns.is_unique or
                not x.index.equals(group.index) or not x.columns.equals(group.columns)):
                raise ValueError("x and group axes must be unique and exactly aligned")
            group_arr = group.to_numpy()
        out = _group_percentile_numpy(
            x.to_numpy(dtype=float),
            group_arr,
            p=float(p),
            side=str(side),
            missing_group_policy=str(missing_group_policy),
        )
        return pd.DataFrame(out, index=x.index, columns=x.columns)


@register_operator(
    name="div_or_null",
    category="data_cleaning",
    business_category="data_cleaning",
    canonical="div_or_null",
    source="semantic_hardening",
    backend="pandas_numpy",
    status="production",
)
class DivOrNullAudited(TwoVarOperator):
    metadata = OperatorMetadata(
        name="div_or_null",
        category="data_cleaning",
        description="安全除法：NULL 或 |denominator|<=epsilon 时返回 NULL。",
        examples=["div_or_null(earnings, assets)"],
        param_names=["x", "y", "epsilon"],
        return_type="series",
        tags=["data_cleaning", "ratio", "pit_safe", "audited"],
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        epsilon: float = 1e-12,
        **kwargs,
    ) -> pd.DataFrame:
        eps = abs(float(epsilon))
        valid = x.notna() & y.notna() & (y.abs() > eps)
        out = (x / y.where(valid)).where(valid)
        return out.replace([np.inf, -np.inf], np.nan)


if pl is not None:

    from factor_engine.cleaned_operators.base_polars import (
        OperatorMetadata as PolarsOperatorMetadata,
        SeriesOperator as PolarsSeriesOperator,
        TwoVarOperator as PolarsTwoVarOperator,
        register_operator as register_polars_operator,
    )

    @register_polars_operator(
        name="ts_product",
        category="time_series",
        business_category="time_series",
        canonical="ts_product",
        source="semantic_hardening",
        status="implemented",
    )
    class TimeSeriesProductAuditedPolars(PolarsSeriesOperator):
        metadata = PolarsOperatorMetadata(
            name="ts_product",
            category="time_series",
            description="滚动乘积；与 pandas 审计实现语义一致。",
            param_names=["x", "window", "min_periods", "skipna"],
            **_contract("ts_product"),
            return_type="series",
            tags=["time_series", "product", "pit_safe", "audited"],
        )

        def _calculate_series(
            self,
            x: pl.DataFrame,
            window: int = 20,
            min_periods: int | None = None,
            skipna: bool = True,
            **kwargs,
        ) -> pl.DataFrame:
            from factor_engine.cleaned_operators.common.semantic_hardening_native import calculate
            return calculate("ts_product", x, window=window, min_periods=min_periods, skipna=skipna)

        def physical_spec(self):
            from factor_engine.cleaned_operators.common.semantic_hardening_native import physical_spec
            return physical_spec("ts_product")

    @register_polars_operator(
        name="ts_mad",
        category="time_series",
        business_category="time_series",
        canonical="ts_mad",
        source="semantic_hardening",
        status="implemented",
    )
    class TimeSeriesMedianAbsoluteDeviationAuditedPolars(PolarsSeriesOperator):
        metadata = PolarsOperatorMetadata(
            name="ts_mad",
            category="time_series",
            description="滚动中位绝对偏差；与 pandas 审计实现语义一致。",
            param_names=["x", "window", "min_periods", "scale"],
            **_contract("ts_mad"),
            return_type="series",
            tags=["time_series", "mad", "robust", "pit_safe", "audited"],
        )

        def _calculate_series(
            self,
            x: pl.DataFrame,
            window: int = 20,
            min_periods: int | None = None,
            scale: float = 1.0,
            **kwargs,
        ) -> pl.DataFrame:
            from factor_engine.cleaned_operators.common.semantic_hardening_native import calculate
            return calculate("ts_mad", x, window=window, min_periods=min_periods, scale=scale)

        def physical_spec(self):
            from factor_engine.cleaned_operators.common.semantic_hardening_native import physical_spec
            return physical_spec("ts_mad")

    @register_polars_operator(
        name="group_percentile",
        category="cross_sectional",
        business_category="group_neutralization",
        canonical="group_percentile",
        source="semantic_hardening",
        status="implemented",
    )
    class GroupPercentileAuditedPolars(PolarsSeriesOperator):
        metadata = PolarsOperatorMetadata(
            name="group_percentile",
            category="cross_sectional",
            description="组内 top/bottom 分位掩码；与 pandas 审计实现语义一致。",
            param_names=["x", "group", "p", "side", "missing_group_policy"],
            **_contract("group_percentile"),
            return_type="series",
            tags=["cross_sectional", "group", "percentile", "pit_safe", "audited"],
        )

        def _calculate_series(
            self,
            x: pl.DataFrame,
            group: pl.DataFrame | None = None,
            p: float = 0.5,
            side: str = "top",
            missing_group_policy: str = "raise",
            **kwargs,
        ) -> pl.DataFrame:
            from factor_engine.cleaned_operators.common.semantic_hardening_native import calculate
            return calculate("group_percentile", x, group=group, p=p, side=side,
                missing_group_policy=missing_group_policy)

        def physical_spec(self):
            from factor_engine.cleaned_operators.common.semantic_hardening_native import physical_spec
            return physical_spec("group_percentile")

    @register_polars_operator(
        name="div_or_null",
        category="data_cleaning",
        business_category="data_cleaning",
        canonical="div_or_null",
        source="semantic_hardening",
        status="production",
    )
    class DivOrNullAuditedPolars(PolarsTwoVarOperator):
        metadata = PolarsOperatorMetadata(
            name="div_or_null",
            category="data_cleaning",
            description="安全除法：NULL 或近零分母返回 NULL。",
            param_names=["x", "y", "epsilon"],
            return_type="series",
            tags=["data_cleaning", "ratio", "pit_safe", "audited"],
        )

        def _calculate_series(
            self,
            x: pl.DataFrame,
            y: pl.DataFrame,
            epsilon: float = 1e-12,
            **kwargs,
        ) -> pl.DataFrame:
            xv = x.to_numpy().astype(float)
            yv = y.to_numpy().astype(float)
            eps = abs(float(epsilon))
            valid = np.isfinite(xv) & np.isfinite(yv) & (np.abs(yv) > eps)
            out = np.full(xv.shape, np.nan, dtype=float)
            np.divide(xv, yv, out=out, where=valid)
            out[~np.isfinite(out)] = np.nan
            return pl.DataFrame(out, schema=x.columns)


def apply_surface_hardening() -> None:
    """Compatibility hook; the static surface module is authoritative."""


apply_surface_hardening()
