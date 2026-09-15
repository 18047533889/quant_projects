# -*- coding: utf-8 -*-
"""Robust time-series statistics operators.

These operators resist single extreme observations: quantile ranges, trimmed
means and median/MAD z-scores.  Every function consumes one or more
``timestamp x instrument`` panels and returns a panel with the same shape,
using only the current row and historical rows (causal).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    SeriesOperator,
    register_operator,
)
from factor_engine.cleaned_operators.common.strict_params import strict_int


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    domain: str,
    unit: str,
    output_unit: str | None = None,
    param_specs: dict | None = None,
) -> OperatorMetadata:
    specs={
        "window":ParamSpec(dtype=int,min=2,default=20,param_role=ParamRole.HORIZON,
                           history_semantics="exact_rows" if name.endswith("_prior") else "max_rows"),
    }
    for key,default in (("q_low",.25),("q_high",.75)):
        if key in params:
            specs[key]=ParamSpec(dtype=float,min=float(np.nextafter(0.,1.)),max=float(np.nextafter(1.,0.)),
                                 default=default,param_role=ParamRole.ECONOMIC)
    if "trim_ratio" in params:
        specs["trim_ratio"]=ParamSpec(dtype=float,min=0.,max=float(np.nextafter(.5,0.)),default=.1,param_role=ParamRole.ECONOMIC)
    if "center" in params:
        specs.update(center=ParamSpec(dtype=str,default="median",param_role=ParamRole.POLICY),
                     scale=ParamSpec(dtype=str,default="mad",param_role=ParamRole.POLICY),
                     clip=ParamSpec(dtype=float,min=float(np.nextafter(0.,1.)),default=None,param_role=ParamRole.POLICY))
    return OperatorMetadata(
        name=name,
        panel_params=("x",),scalar_params=tuple(params[1:]),
        category="robust_statistics",
        description=description,
        param_names=params,
        return_type="series",
        output_unit=output_unit,
        param_specs={**(param_specs or {}),**specs},
        tags=[
            "robust_statistics", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", "cost:1",
        ],
    )


# R11 round-3 #123: a default ``min_periods=1`` lets a 20-day IQR be built from
# 1/2/3 observations — a warm-up that is not a robust statistic.  The searchable
# canonicals default to ``max(5, 0.5 * window)`` finite observations.
def _auto_min_periods(window: int, min_periods: int | None) -> int:
    if min_periods is not None:
        return strict_int(min_periods,"min_periods",minimum=1)
    return max(5, int(0.5 * window))


def _quantile_spread(valid,lo,hi):
    magnitude=float(np.max(np.abs(valid)))
    if magnitude==0.:
        return 0.
    unit=valid/magnitude
    with np.errstate(over="ignore",invalid="ignore"):
        value=float(np.quantile(unit,hi)-np.quantile(unit,lo))*magnitude
    return value if np.isfinite(value) else np.nan

def _robust_score(valid,current,center,scale,clip):
    """Scale coordinates before centering; preserve original mean/MAD policies."""
    if not np.isfinite(current) or valid.size==0:
        return np.nan
    magnitude=float(np.max(np.abs(valid)))
    if magnitude==0.:
        return np.nan
    unit=valid/magnitude
    location=float(np.median(unit)) if center=="median" else float(np.mean(unit))
    spread=float(np.std(unit)) if scale=="std" else float(np.median(np.abs(unit-location)))*1.4826
    if not np.isfinite(spread) or spread<=0.:
        return np.nan
    with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
        value=(current/magnitude-location)/spread
    if clip is not None:
        value=float(np.clip(value,-clip,clip))
    return float(value) if np.isfinite(value) else np.nan


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _rolling_apply_2d(
    values: np.ndarray,
    window: int,
    fn: Any,
    min_periods: int = 1,
) -> np.ndarray:
    rows, cols = values.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - window + 1)
            chunk = values[start : row + 1, col]
            out[row, col] = fn(chunk)
    return out


def _rolling_prior_apply_2d(
    values: np.ndarray,
    window: int,
    fn: Any,
    min_periods: int = 1,
) -> np.ndarray:
    """Prior-only rolling kernel: the baseline is estimated on ``[t-W, t-1]``
    and the current row ``t`` is scored against it.  An extreme ``x_t`` can
    never contaminate its own median/MAD (R11 round-3 #122).

    R16-107: the prior window is ``[t-W, t-1]`` — exactly W prior observations
    (``start = row - window``).  The old ``row - window + 1`` capped at W-1
    observations, so a W=20 prior used only 19.
    """
    rows, cols = values.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - window)
            prior = values[start:row, col]
            x_t = values[row, col]
            out[row, col] = fn(prior, x_t, min_periods)
    return out


@register_operator(
    name="ts_quantile_range",
    category="robust_statistics",
    business_category="robust_statistics",
    canonical="ts_quantile_range",
    source="robust_stats",
    status="experimental")
class TsQuantileRange(SeriesOperator):
    """滚动分位区间：Q(x, q_high) - Q(x, q_low)。q_low=0.25/q_high=0.75 即 IQR。"""

    metadata = _metadata(
        "ts_quantile_range",
        "滚动分位区间 Q(x,q_high)-Q(x,q_low)。",
        ["x", "window", "q_low", "q_high", "min_periods"],
        domain="price_volume",
        # R11 round-3 #120: Q_high(x) - Q_low(x) is a spread of x in x's own
        # unit — ``same_as:x``, not ``ratio``.
        unit="same_as:x",
        output_unit="same_as:x",
        param_specs={
            "window": ParamSpec(dtype=int, min=2),
            "min_periods": ParamSpec(
                dtype=int, min=1, default=None, searchable=False,
                param_role=ParamRole.ESTIMATOR_RESOLUTION,
            ),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 20,
        q_low: float = 0.25,
        q_high: float = 0.75,
        min_periods: int | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        w = strict_int(window,"window",minimum=2)
        lo = float(q_low)
        hi = float(q_high)
        if not (0.0 < lo < hi < 1.0):
            raise ValueError("ts_quantile_range requires 0 < q_low < q_high < 1")
        mp = _auto_min_periods(w, min_periods)

        def _fn(chunk: np.ndarray) -> float:
            valid = chunk[np.isfinite(chunk)]
            if valid.size < mp:
                return np.nan
            return _quantile_spread(valid,lo,hi)

        return _frame_like(x, _rolling_apply_2d(x.to_numpy(dtype=float), w, _fn, mp))


@register_operator(
    name="ts_trimmed_mean",
    category="robust_statistics",
    business_category="robust_statistics",
    canonical="ts_trimmed_mean",
    source="robust_stats",
    status="experimental")
class TsTrimmedMean(SeriesOperator):
    """滚动截尾均值：删除最低/最高 trim_ratio 后对剩余求平均。"""

    metadata = _metadata(
        "ts_trimmed_mean",
        "滚动截尾均值（删除两侧 trim_ratio 后求平均）。",
        ["x", "window", "trim_ratio", "min_periods"],
        domain="price_volume",
        # R11 round-3 #121: a trimmed mean is a location statistic of x — the
        # output carries x's own unit, ``same_as:x``, not ``ratio``.
        unit="same_as:x",
        output_unit="same_as:x",
        param_specs={
            "window": ParamSpec(dtype=int, min=2),
            "min_periods": ParamSpec(
                dtype=int, min=1, default=None, searchable=False,
                param_role=ParamRole.ESTIMATOR_RESOLUTION,
            ),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 20,
        trim_ratio: float = 0.1,
        min_periods: int | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        w = strict_int(window,"window",minimum=2)
        trim = float(trim_ratio)
        if not (0.0 <= trim < 0.5):
            raise ValueError("ts_trimmed_mean requires 0 <= trim_ratio < 0.5")
        mp = _auto_min_periods(w, min_periods)

        def _fn(chunk: np.ndarray) -> float:
            valid = chunk[np.isfinite(chunk)]
            if valid.size < mp:
                return np.nan
            ordered = np.sort(valid)
            cut = int(np.floor(trim * ordered.size))
            if cut * 2 >= ordered.size:
                # Parameter/sample infeasible: trimming would remove everything.
                # Falling back to the plain mean silently changes the operator —
                # return NaN instead (P1-77).
                return np.nan
            selected=ordered[cut : ordered.size - cut]
            magnitude=float(np.max(np.abs(selected)))
            return float(np.mean(selected/magnitude))*magnitude if magnitude else 0.

        return _frame_like(x, _rolling_apply_2d(x.to_numpy(dtype=float), w, _fn, mp))


@register_operator(
    name="ts_robust_zscore_inclusive",
    category="robust_statistics",
    business_category="robust_statistics",
    canonical="ts_robust_zscore_inclusive",
    source="robust_stats",
    status="experimental")
class TsRobustZscore(SeriesOperator):
    """稳健 z-score：(x - center) / (scale)，center 中位数、scale MAD。

    ``center`` 取 ``"median"``/``"mean"``，``scale`` 取 ``"mad"``/``"std"``；
    ``clip`` 非空时将输出截断到 [-clip, clip]。

    R11 round-3 #122: this is the *inclusive* baseline — the trailing window
    ``[t-W+1, t]`` contains ``x_t`` itself, so an extreme current value shifts the
    median/MAD and then scores itself.  The canonical is renamed to
    ``ts_robust_zscore_inclusive`` (with ``ts_robust_zscore`` kept as a
    back-compat alias) by :func:`_register_robust_zscore_split`; the
    non-contaminating variant is ``ts_robust_zscore_prior``.
    """

    metadata = _metadata(
        "ts_robust_zscore_inclusive",
        "稳健 z-score：(x-median)/(1.4826*MAD)，可选 clip。",
        ["x", "window", "center", "scale", "clip"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 20,
        center: Any = "median",
        scale: Any = "mad",
        clip: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        w = strict_int(window,"window",minimum=2)
        center_name = str(center or "median").lower()
        scale_name = str(scale or "mad").lower()
        # Enum validation: an invalid string must fail loudly, never silently
        # fall back to mean/MAD (P1-75).
        if center_name not in ("median", "mean"):
            raise ValueError("center must be 'median' or 'mean'")
        if scale_name not in ("mad", "std"):
            raise ValueError("scale must be 'mad' or 'std'")
        if clip is not None:
            bound = float(clip)
            # A non-positive clip would invert the clamp (P1-76).
            if not np.isfinite(bound) or bound <= 0.0:
                raise ValueError("clip must be None or > 0")

        def _fn(chunk: np.ndarray) -> float:
            valid = chunk[np.isfinite(chunk)]
            if valid.size == 0:
                return np.nan
            return _robust_score(valid,float(chunk[-1]),center_name,scale_name,None if clip is None else bound)

        return _frame_like(x, _rolling_apply_2d(x.to_numpy(dtype=float), w, _fn))


@register_operator(
    name="ts_robust_zscore_prior",
    category="robust_statistics",
    business_category="robust_statistics",
    canonical="ts_robust_zscore_prior",
    source="robust_stats",
    status="experimental")
class TsRobustZscorePrior(SeriesOperator):
    """稳健 z-score，baseline 仅在 ``[t-W, t-1]``。

    The median/MAD centre+scale are estimated on the prior window *excluding*
    the current row, so an extreme ``x_t`` can never contaminate the baseline it
    is scored against (R11 round-3 #122).  NaN while fewer than ``min_periods``
    prior observations are available (default ``max(5, 0.5*window)``).
    """

    metadata = _metadata(
        "ts_robust_zscore_prior",
        "稳健 z-score：(x_t-center)/(scale)，center/scale 仅用 [t-W, t-1]。",
        ["x", "window", "center", "scale", "clip", "min_periods"],
        domain="price_volume",
        unit="ratio",
        param_specs={
            "window": ParamSpec(dtype=int, min=2),
            "min_periods": ParamSpec(
                dtype=int, min=1, default=None, searchable=False,
                param_role=ParamRole.ESTIMATOR_RESOLUTION,
            ),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 20,
        center: Any = "median",
        scale: Any = "mad",
        clip: Any = None,
        min_periods: int | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        w = strict_int(window,"window",minimum=2)
        center_name = str(center or "median").lower()
        scale_name = str(scale or "mad").lower()
        # Enum validation: an invalid string must fail loudly, never silently
        # fall back to mean/MAD (P1-75).
        if center_name not in ("median", "mean"):
            raise ValueError("center must be 'median' or 'mean'")
        if scale_name not in ("mad", "std"):
            raise ValueError("scale must be 'mad' or 'std'")
        if clip is not None:
            bound = float(clip)
            # A non-positive clip would invert the clamp (P1-76).
            if not np.isfinite(bound) or bound <= 0.0:
                raise ValueError("clip must be None or > 0")
        mp = _auto_min_periods(w, min_periods)

        def _fn(prior: np.ndarray, x_t: float, m: int) -> float:
            valid = prior[np.isfinite(prior)]
            if valid.size < m:
                return np.nan
            return _robust_score(valid,float(x_t),center_name,scale_name,None if clip is None else bound)

        return _frame_like(x, _rolling_prior_apply_2d(x.to_numpy(dtype=float), w, _fn, mp))


def _register_robust_zscore_split() -> None:
    """R11 round-3 #122: split ``ts_robust_zscore`` into two canonicals.

    * ``ts_robust_zscore_inclusive`` — the historical behaviour (baseline
      window ``[t-W+1, t]`` includes ``x_t``), and
    * ``ts_robust_zscore_prior`` — the non-contaminating baseline on
      ``[t-W, t-1]``.

    ``ts_robust_zscore`` is kept as a back-compat alias of the inclusive
    kernel.  Idempotent across backends and import orders (mirrors
    ``polars_robust_stats._register_best_lag_corr_split``).
    """
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    if "ts_robust_zscore" in OperatorRegistry._operators:
        OperatorRegistry.rename_canonical("ts_robust_zscore", "ts_robust_zscore_inclusive")
    elif "ts_robust_zscore_inclusive" in OperatorRegistry._operators:
        OperatorRegistry.register_alias("ts_robust_zscore", "ts_robust_zscore_inclusive")
    try:
        from factor_engine.cleaned_operators.operator_surface import (
            extend_extended_only,
            retract_extended_only,
        )

        extend_extended_only(["ts_robust_zscore_inclusive", "ts_robust_zscore_prior"])
    except ImportError:  # pragma: no cover - surface module always present in-tree
        pass


_register_robust_zscore_split()
