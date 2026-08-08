# -*- coding: utf-8 -*-
"""Volume-clock intraday path geometry (2026-08 V3, P1, minute → daily).

Natural-clock path statistics (``ts_path_efficiency`` / ``ts_roughness`` /
``intra_path_efficiency``) measure price motion per *minute of wall clock*.
Volume-clock geometry re-prices the day along cumulative *trading activity*:
two stocks can have identical natural-time paths yet very different volume-clock
paths when one moves its price mostly on thin volume.

* ``intraday_volume_clock_path_efficiency`` — ``|p(1)-p(0)| / Σ|Δp_b|`` along the
  equal-activity grid (P1).
* ``intraday_volume_clock_roughness`` — ``Σ(Δ²p_b)² / (Σ(Δp_b)² + eps)`` on the
  same resampled log-price path (P1).

``activity`` is a user-supplied per-minute intensity (volume / amount / trades);
the clock is built on its cumulative sum, so the operator stays activity-agnostic.
Only the day's own bars enter the computation — prefix-causal and deterministic.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, ParamSpec, SeriesOperator, register_operator
from cleaned_operators.microstructure.intraday_agg import _as_panel, _daily_agg_two

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="intraday_microstructure",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "intraday", "daily_agg", "minute", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:price_volume",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _volume_clock_log_path(price: np.ndarray, activity: np.ndarray, buckets: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Resampled log-price path on the equal-activity grid -> (q_grid, log_p).

    R6-141: price must be strictly positive (``log`` domain) — a non-positive
    price is data-invalid, not a valid log-price.  R6-142: negative activity is
    a data error, not "no activity" — it fails the whole path closed rather
    than being filtered out and silently reconnecting the grid around it.
    """
    # R6-142: negative activity invalidates the entire path (a negative
    # contribution to cumulative activity is not a valid clock).  Zero activity
    # is a valid "no trade" minute.  NaN in either series fails the path.
    if np.any(~np.isfinite(price)) or np.any(~np.isfinite(activity)) or np.any(activity < 0.0):
        return None
    # R6-141: non-positive price is invalid for log-price (fail closed, never
    # a silently-produced -inf that then gets filtered by the finite mask).
    if np.any(price <= 0.0):
        return None
    valid = activity > 0.0
    if int(valid.sum()) < 3:
        return None
    act = activity[valid].astype(float)
    logp = np.log(price[valid].astype(float))
    Q = np.cumsum(act)
    Q = Q / Q[-1]
    # Deduplicate ties (zero-activity bars) so np.interp's xp is strictly rising.
    keep = np.concatenate(([True], np.diff(Q) > 0.0))
    Q = Q[keep]
    logp = logp[keep]
    if Q.shape[0] < 2 or Q[0] != Q[0]:
        return None
    B = max(4, int(buckets))
    # R6-144: a smooth B-point path cannot be built from fewer distinct activity
    # points than B+1 — the interpolation would fabricate a path between
    # missing observations (roughness/curvature becomes an interpolation
    # artifact).  Fail closed instead of manufacturing smoothness.
    if Q.shape[0] < B + 1:
        return None
    grid = np.linspace(0.0, 1.0, B + 1)
    path = np.interp(grid, Q, logp)
    return grid, path


def _volume_clock_efficiency(price: np.ndarray, activity: np.ndarray, buckets: int) -> float:
    res = _volume_clock_log_path(price, activity, buckets)
    if res is None:
        return np.nan
    _, path = res
    total_path = float(np.sum(np.abs(np.diff(path))))
    if total_path <= _EPS:
        return 0.0
    return float(abs(path[-1] - path[0]) / total_path)


def _volume_clock_roughness(price: np.ndarray, activity: np.ndarray, buckets: int) -> float:
    res = _volume_clock_log_path(price, activity, buckets)
    if res is None:
        return np.nan
    _, path = res
    delta = np.diff(path)
    delta2 = np.diff(delta)
    denom = float(np.sum(delta * delta))
    if denom <= _EPS:
        return 0.0
    return float(np.sum(delta2 * delta2) / denom)


@register_operator(
    name="intraday_volume_clock_path_efficiency",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_volume_clock_path_efficiency",
    source="volume_clock",
)
class IntradayVolumeClockPathEfficiency(SeriesOperator):
    """成交量时钟路径效率 ``|p(1)-p(0)| / Σ|Δp_b|``（等 activity 网格）。

    按累计 activity 等分 ``buckets`` 档，在每档处对 log-price 线性插值；效率
    接近 1 = 一单位真实成交对应稳定单向位移；接近 0 = 大量成交却在来回震荡。
    与自然时钟 ``intra_path_efficiency`` 信息不同。P1。
    """

    metadata = _metadata(
        "intraday_volume_clock_path_efficiency",
        "成交量时钟路径效率（等 activity 网格，log-price）。",
        ["price", "activity", "buckets"],
        unit="ratio",
        cost=4,
    )
    # R6-141/142: price must be > 0 (log domain), activity must be >= 0
    # (negative activity is a data error, not "no trade").  R6-144: buckets
    # must not exceed the distinct positive-activity points - 1, else the path
    # is interpolated smoothness, not observation.
    metadata.param_specs = {
        "buckets": ParamSpec(dtype=int, min=4),
    }

    def _calculate_series(
        self, price: pd.DataFrame, activity: pd.DataFrame, buckets: int = 16, **_: Any
    ) -> pd.DataFrame:
        b = max(4, int(buckets))
        return _daily_agg_two(price, activity, lambda p, a: _volume_clock_efficiency(p, a, b))


@register_operator(
    name="intraday_volume_clock_roughness",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_volume_clock_roughness",
    source="volume_clock",
)
class IntradayVolumeClockRoughness(SeriesOperator):
    """成交量时钟粗糙度 ``Σ(Δ²p_b)² / (Σ(Δp_b)² + eps)``。

    高 = 同一单位真实成交量下价格不断来回跳动；低 = 成交量推动价格稳定单向
    移动。与自然时钟 ``ts_roughness`` 不重复。P1。
    """

    metadata = _metadata(
        "intraday_volume_clock_roughness",
        "成交量时钟粗糙度（等 activity 网格二阶/一阶差平方比）。",
        ["price", "activity", "buckets"],
        unit="ratio",
        cost=4,
    )
    metadata.param_specs = {
        "buckets": ParamSpec(dtype=int, min=4),
    }

    def _calculate_series(
        self, price: pd.DataFrame, activity: pd.DataFrame, buckets: int = 16, **_: Any
    ) -> pd.DataFrame:
        b = max(4, int(buckets))
        return _daily_agg_two(price, activity, lambda p, a: _volume_clock_roughness(p, a, b))


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({
            "intraday_volume_clock_path_efficiency",
            "intraday_volume_clock_roughness",
        })


_register_surface()
