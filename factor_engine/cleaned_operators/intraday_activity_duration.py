# -*- coding: utf-8 -*-
"""Intraday activity-duration curvature (2026-08-08 Gemini round).

``intraday_activity_duration_curvature`` — one scalar per (date, symbol): split
the day's non-negative minute ``activity`` (volume / amount) into ``buckets``
equal-activity buckets, record the wall-clock bar index at which each bucket
completes ``D_1..D_B``, and return the standardized second difference of the
duration curve ``mean(Δ²D) / (MAD(D)+eps)``.

* Positive curvature → activity *decelerating* into the close (later buckets
  take longer wall-clock).
* Negative curvature → activity *accelerating* into the close.

This studies how the activity clock itself speeds up / slows down relative to
wall-clock, orthogonal to the existing volume-clock path-geometry operators
(which study the price path *on* the activity clock).  Minute-source input,
strict-PIT within the day, deterministic, NaN fail-closed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str]) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="intraday_microstructure",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "intraday", "minute", "daily", "pit_safe", "causal", "deterministic",
            f"signature:{','.join(params)}->series", "domain:intraday",
            "unit:level", "cost:6",
        ],
    )


def _day_curvature(values: np.ndarray, buckets: int) -> float:
    """Duration-curve second-difference curvature for one day of minute activity."""
    finite = values[np.isfinite(values)]
    finite = finite[finite >= 0.0]
    total = float(finite.sum())
    if total <= _EPS or finite.size < buckets:
        return np.nan
    b = int(buckets)
    if b < 3:
        raise ValueError("intraday_activity_duration_curvature requires buckets >= 3")
    cum = np.cumsum(finite)
    # completion bar index for each fraction b/B (b = 1..B)
    D = np.full(b, np.nan)
    for k in range(1, b + 1):
        target = float(k) / b * total
        pos = np.flatnonzero(cum >= target)
        if pos.size == 0:
            return np.nan
        D[k - 1] = float(pos[0])
    if np.any(np.isnan(D)):
        return np.nan
    d2 = D[2:] - 2.0 * D[1:-1] + D[:-2]
    mad = float(np.median(np.abs(D - np.median(D))))
    if mad <= _EPS:
        return np.nan
    return float(np.mean(d2) / mad)


@register_operator(
    name="intraday_activity_duration_curvature",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_activity_duration_curvature",
    source="intraday_activity_duration",
    status="implemented",
)
class IntradayActivityDurationCurvature(SeriesOperator):
    """日内 activity 等量分桶后的 wall-clock 时长曲线标准化二阶曲率。"""

    metadata = _metadata(
        "intraday_activity_duration_curvature",
        "分钟 activity 等量分桶时长的二阶曲率（mean Δ²D / MAD(D)）。",
        ["activity", "buckets"],
    )

    def _calculate_series(
        self,
        activity: pd.DataFrame,
        buckets: int = 10,
        **_: Any,
    ) -> pd.DataFrame:
        act = activity.copy()
        b = int(buckets)
        if b < 3:
            raise ValueError("intraday_activity_duration_curvature requires buckets >= 3")
        out: dict[str, pd.Series] = {}
        for inst in act.columns:
            col = act[inst]
            arr = col.to_numpy(dtype=float)
            idx = col.index
            per_day: dict[pd.Timestamp, float] = {}
            for day, group_idx in pd.Series(np.arange(len(idx)), index=idx).groupby(idx.normalize()):
                positions = np.asarray(group_idx, dtype=int)
                per_day[day] = _day_curvature(arr[positions], b)
            out[inst] = pd.Series(per_day, dtype=float)
        if not out:
            return pd.DataFrame(dtype=float)
        return pd.DataFrame(out).sort_index()


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | {"intraday_activity_duration_curvature"}
    )
    from cleaned_operators.rolling_pack import register_polars_udf

    register_polars_udf("intraday_activity_duration_curvature")


_register_surface()
