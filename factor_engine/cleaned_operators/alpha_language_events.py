# -*- coding: utf-8 -*-
"""Alpha-language generic event and fundamental report-sequence operators (2026-08).

Generic event language (no domain-specific ``days_since_limit_up``-style names):

* event_frequency: fraction of days a condition held in the trailing window;
* event_cluster_count / event_cluster_mean_size: events within ``max_gap`` days
  form a cluster; count clusters / average events-per-cluster.

Fundamental report-sequence convenience operators (thin wrappers over the
certified ``fin_*`` fiscal kernels):

* report_rolling_mean: mean of the most recent K *report observations* (not a
  forward-filled daily ``ts_mean``);
* report_yoy_lag: same-fiscal-period-last-year value (fin_lag with periods=4).

Both event operators are causal trailing-window transforms.  The report
operators use the audited ``_walk_periods`` ordinal engine from
``fundamental/transforms_v2`` and inherit its PIT/revision semantics.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.fundamental.transforms_v2 import (
    _lag_value,
    _pos_int,
    _values,
    _walk_periods,
)
from cleaned_operators.rolling_pack import (
    check_window,
    frame_like,
    map_rolling,
    register_polars_bridge,
    valid_values,
)


def _metadata(name: str, description: str, params: list[str], *, unit: str) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="time_series_event",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "time_series_event", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:event",
            f"unit:{unit}", "cost:1",
        ],
    )


def _fundamental_metadata(name: str, description: str, params: list[str]) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="fundamental_period",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "fundamental_period", "period_aware", "pit_safe", "causal",
            "production_extension", f"signature:{','.join(params)}->series",
            "domain:fundamental", "unit:level", "cost:1",
        ],
    )


@register_operator(
    name="event_frequency",
    category="time_series_event",
    business_category="time_series_event",
    canonical="event_frequency",
    source="alpha_language_events",
)
class EventFrequency(SeriesOperator):
    """事件频率: 窗口内条件为真的有效观测占比(即 ts_count_if/window 的 fused)。"""

    metadata = _metadata(
        "event_frequency",
        "窗口内事件发生比例。",
        ["condition", "window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(self, condition: pd.DataFrame, window: int = 20, min_periods: int = 1, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        mp = max(1, int(min_periods))
        cv = condition.to_numpy(dtype=float)

        def _fn(chunk: np.ndarray) -> float:
            finite = chunk[np.isfinite(chunk)]
            n = finite.size
            if n < mp:
                return np.nan
            hits = float(np.sum(finite != 0.0))
            return hits / n

        return frame_like(condition, map_rolling(cv, w, _fn))


@register_operator(
    name="event_cluster_count",
    category="time_series_event",
    business_category="time_series_event",
    canonical="event_cluster_count",
    source="alpha_language_events",
)
class EventClusterCount(SeriesOperator):
    """事件簇计数: 窗口内事件按 max_gap 分簇, 返回簇数。无事件 -> 0。"""

    metadata = _metadata(
        "event_cluster_count",
        "窗口内事件簇数量。",
        ["condition", "window", "max_gap"],
        unit="count",
    )

    def _calculate_series(self, condition: pd.DataFrame, window: int = 60, max_gap: int = 3, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        gap = int(max_gap)
        if gap < 0:
            raise ValueError("max_gap must be >= 0")
        cv = condition.to_numpy(dtype=float)

        def _fn(chunk: np.ndarray) -> float:
            truth = np.isfinite(chunk) & (chunk != 0.0)
            positions = np.flatnonzero(truth)
            if positions.size == 0:
                return 0.0
            clusters = 1
            for i in range(1, positions.size):
                if positions[i] - positions[i - 1] > gap:
                    clusters += 1
            return float(clusters)

        return frame_like(condition, map_rolling(cv, w, _fn))


@register_operator(
    name="event_cluster_mean_size",
    category="time_series_event",
    business_category="time_series_event",
    canonical="event_cluster_mean_size",
    source="alpha_language_events",
)
class EventClusterMeanSize(SeriesOperator):
    """事件簇平均规模: 窗口内平均每簇事件数。无事件 -> NaN。"""

    metadata = _metadata(
        "event_cluster_mean_size",
        "窗口内每簇平均事件数。",
        ["condition", "window", "max_gap"],
        unit="count",
    )

    def _calculate_series(self, condition: pd.DataFrame, window: int = 60, max_gap: int = 3, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        gap = int(max_gap)
        if gap < 0:
            raise ValueError("max_gap must be >= 0")
        cv = condition.to_numpy(dtype=float)

        def _fn(chunk: np.ndarray) -> float:
            truth = np.isfinite(chunk) & (chunk != 0.0)
            positions = np.flatnonzero(truth)
            if positions.size == 0:
                return np.nan
            sizes = []
            cluster_size = 1
            for i in range(1, positions.size):
                if positions[i] - positions[i - 1] > gap:
                    sizes.append(cluster_size)
                    cluster_size = 1
                else:
                    cluster_size += 1
            sizes.append(cluster_size)
            return float(np.mean(sizes))

        return frame_like(condition, map_rolling(cv, w, _fn))


def _register_report_operator(name: str, params: Iterable[str], fn: Any, description: str) -> None:
    metadata = _fundamental_metadata(name, description, list(params))

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"AlphaReport_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="fundamental_period",
        business_category="fundamental",
        canonical=name,
        source="alpha_language_events",
        backend="pandas_numpy",
        status="production",
    )(cls)


def _report_rolling_mean(x, period_id, periods=8):
    n = _pos_int(periods, "periods", 2)

    def calc(order, visible, current):
        vals = np.asarray(_values(order, visible, current, n), dtype=float)
        return float(np.mean(vals)) if len(vals) == n else np.nan

    return _walk_periods(x, period_id, calc)


def _report_yoy_lag(x, period_id, periods_per_year=4):
    n = _pos_int(periods_per_year, "periods_per_year")

    def calc(order, visible, current):
        return _lag_value(order, visible, current, n)

    return _walk_periods(x, period_id, calc)


_register_report_operator(
    "report_rolling_mean",
    ("x", "period_id", "periods"),
    _report_rolling_mean,
    "最近 K 个报告观测值的均值(非日频 ffill 的 ts_mean)。",
)
_register_report_operator(
    "report_yoy_lag",
    ("x", "period_id", "periods_per_year"),
    _report_yoy_lag,
    "相同财政期的上一年度报告值(fin_lag periods=4)。",
)


_NEW_CANONICALS = (
    "event_frequency",
    "event_cluster_count",
    "event_cluster_mean_size",
    "report_rolling_mean",
    "report_yoy_lag",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | set(_NEW_CANONICALS)
    )
    # Event ops get the exact-parity polars bridge; report ops are
    # fundamental_period pandas kernels (matching the fin_* pattern).
    for _canon in ("event_frequency", "event_cluster_count", "event_cluster_mean_size"):
        register_polars_bridge(_canon)


_register_surface()
