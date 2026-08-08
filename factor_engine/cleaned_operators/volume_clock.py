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

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
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
    """Resampled log-price path on the equal-activity grid -> (q_grid, log_p)."""
    finite = np.isfinite(price) & np.isfinite(activity) & (activity > 0.0)
    if int(finite.sum()) < 3:
        return None
    act = activity[finite].astype(float)
    logp = np.log(price[finite].astype(float))
    Q = np.cumsum(act)
    Q = Q / Q[-1]
    # Deduplicate ties (zero-activity bars) so np.interp's xp is strictly rising.
    keep = np.concatenate(([True], np.diff(Q) > 0.0))
    Q = Q[keep]
    logp = logp[keep]
    if Q.shape[0] < 2 or Q[0] != Q[0]:
        return None
    B = max(4, int(buckets))
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
