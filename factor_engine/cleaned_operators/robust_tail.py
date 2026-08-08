# -*- coding: utf-8 -*-
"""Robust tail and distribution-shape operators (2026-08 final pack, group 3).

Adds partial moments, expected shortfall, quantile-based skew/kurtosis, a tail
ratio with denominator protection and an extreme-event clustering ratio.  All
are causal daily-panel transforms with fail-closed missing/constant handling.

Missing-value policy: NaN never means 0; degenerate quantiles return NaN
rather than an unbounded number; tail samples below the configured minimum
return NaN (never Inf).  Constant (dispersion=0) windows have a well-defined
partial-moment / expected-shortfall value, so those operators return the
deterministic value (R5 P1-38(a)) instead of NaN; pure *ratio* operators
(quantile skew/kurtosis, tail ratio) still fail-closed to NaN when the
denominator degenerates.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
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
        category="time_series_risk",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "time_series_risk", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:distribution",
            f"unit:{unit}", "cost:1",
        ],
    )


def _has_spread(valid: np.ndarray, *, eps: float = 1e-12, min_count: int = 2) -> bool:
    """False when the finite window is empty, too small or (near-)constant."""
    if valid.size < max(2, int(min_count)):
        return False
    return bool(np.isfinite(valid).all() and np.std(valid, ddof=0) >= eps)


@register_operator(
    name="ts_lower_partial_moment",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_lower_partial_moment",
    source="robust_tail",
)
class TsLowerPartialMoment(SeriesOperator):
    """下偏矩：mean(max(threshold - x, 0)^order)，order=1 下行缺口、order=2 下行方差。"""

    metadata = _metadata(
        "ts_lower_partial_moment",
        "下偏矩 mean(max(threshold-x,0)^order)。",
        ["x", "window", "threshold", "order", "min_periods"],
        unit="level",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, threshold: float = 0.0, order: float = 1.0, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        thr = float(threshold)
        ord_ = float(order)
        if ord_ < 1.0:
            raise ValueError("order must be >= 1")
        mp = max(2, int(min_periods))

        def _fn(chunk: np.ndarray) -> float:
            valid = valid_values(chunk)
            # R5 P1-38(a): a constant window (dispersion=0) still has a
            # well-defined partial moment (max(threshold - c, 0)^order) — only
            # the sample-size floor applies, not a spread requirement.
            if valid.size < mp:
                return np.nan
            below = np.maximum(thr - valid, 0.0)
            return float(np.mean(np.power(below, ord_)))

        return frame_like(x, map_rolling(x.to_numpy(dtype=float), w, _fn))


@register_operator(
    name="ts_upper_partial_moment",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_upper_partial_moment",
    source="robust_tail",
)
class TsUpperPartialMoment(SeriesOperator):
    """上偏矩：mean(max(x - threshold, 0)^order)。"""

    metadata = _metadata(
        "ts_upper_partial_moment",
        "上偏矩 mean(max(x-threshold,0)^order)。",
        ["x", "window", "threshold", "order", "min_periods"],
        unit="level",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, threshold: float = 0.0, order: float = 1.0, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        thr = float(threshold)
        ord_ = float(order)
        if ord_ < 1.0:
            raise ValueError("order must be >= 1")
        mp = max(2, int(min_periods))

        def _fn(chunk: np.ndarray) -> float:
            valid = valid_values(chunk)
            # R5 P1-38(a): constant windows have a well-defined upper partial
            # moment; only the sample-size floor applies.
            if valid.size < mp:
                return np.nan
            above = np.maximum(valid - thr, 0.0)
            return float(np.mean(np.power(above, ord_)))

        return frame_like(x, map_rolling(x.to_numpy(dtype=float), w, _fn))


@register_operator(
    name="ts_expected_shortfall",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_expected_shortfall",
    source="robust_tail",
)
class TsExpectedShortfall(SeriesOperator):
    """历史期望损失（尾部均值）：side='lower' 左尾、side='upper' 右尾，尾部样本不足返回 NaN。"""

    metadata = _metadata(
        "ts_expected_shortfall",
        "历史期望损失 mean(tail) 基于窗口内分位数，尾部样本不足返回 NaN。",
        ["x", "window", "q", "side", "min_tail_count"],
        unit="level",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, q: float = 0.05, side: str = "lower", min_tail_count: int = 5, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        quantile = float(q)
        # Round-7 P0: ES ``q`` is a *tail fraction* — q > 0.5 is not tail-risk
        # semantics (a q=0.8 "lower tail" is the bottom 80% of the distribution).
        if not 0.0 < quantile <= 0.5:
            raise ValueError("q must be in (0, 0.5] for expected-shortfall tail semantics")
        side_kind = str(side).lower()
        if side_kind not in {"lower", "upper"}:
            raise ValueError("side must be 'lower' or 'upper'")
        min_tail = max(2, int(min_tail_count))

        def _fn(chunk: np.ndarray) -> float:
            valid = valid_values(chunk)
            # R5 P1-38(a): a constant window has a well-defined ES (the tail mean
            # is the sample value itself) — only the sample-size floor applies.
            if valid.size < min_tail:
                return np.nan
            if side_kind == "lower":
                thr = float(np.quantile(valid, quantile))
                tail = valid[valid <= thr]
            else:
                thr = float(np.quantile(valid, 1.0 - quantile))
                tail = valid[valid >= thr]
            if tail.size < min_tail:
                return np.nan
            return float(np.mean(tail))

        return frame_like(x, map_rolling(x.to_numpy(dtype=float), w, _fn))


@register_operator(
    name="ts_quantile_skew",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_quantile_skew",
    source="robust_tail",
)
class TsQuantileSkew(SeriesOperator):
    """Bowley 分位数偏度：(Q_high + Q_low - 2 Q_mid) / (Q_high - Q_low)。"""

    metadata = _metadata(
        "ts_quantile_skew",
        "Bowley 分位数偏度，退化区间返回 NaN。",
        ["x", "window", "q_low", "q_mid", "q_high", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, q_low: float = 0.1, q_mid: float = 0.5, q_high: float = 0.9, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        ql, qm, qh = float(q_low), float(q_mid), float(q_high)
        if not 0.0 < ql < qm < qh < 1.0:
            raise ValueError("require 0 < q_low < q_mid < q_high < 1")
        mp = max(5, int(min_periods))

        def _fn(chunk: np.ndarray) -> float:
            valid = valid_values(chunk)
            if not _has_spread(valid, min_count=mp):
                return np.nan
            q_lo = float(np.quantile(valid, ql))
            q_mid_v = float(np.quantile(valid, qm))
            q_hi = float(np.quantile(valid, qh))
            denom = q_hi - q_lo
            if not np.isfinite(denom) or abs(denom) < 1e-12:
                return np.nan
            return float((q_hi + q_lo - 2.0 * q_mid_v) / denom)

        return frame_like(x, map_rolling(x.to_numpy(dtype=float), w, _fn))


@register_operator(
    name="ts_quantile_kurtosis",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_quantile_kurtosis",
    source="robust_tail",
)
class TsQuantileKurtosis(SeriesOperator):
    """分位数峰度（尾部宽度比）：(Q_outer_hi - Q_outer_lo) / (Q_inner_hi - Q_inner_lo)。标准正态约 2.906。"""

    metadata = _metadata(
        "ts_quantile_kurtosis",
        "分位数峰度（尾部宽度相对中部宽度比），退化中部区间返回 NaN。",
        ["x", "window", "outer", "inner", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, outer: Any = (0.025, 0.975), inner: Any = (0.25, 0.75), min_periods: int = 8, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        try:
            o_lo, o_hi = float(outer[0]), float(outer[1])
            i_lo, i_hi = float(inner[0]), float(inner[1])
        except (TypeError, IndexError, ValueError):
            raise ValueError("outer/inner must be 2-tuples of quantiles")
        if not (0.0 < o_lo < o_hi < 1.0 and 0.0 < i_lo < i_hi < 1.0):
            raise ValueError("outer/inner must be strictly ordered quantiles")
        mp = max(8, int(min_periods))

        def _fn(chunk: np.ndarray) -> float:
            valid = valid_values(chunk)
            if not _has_spread(valid, min_count=mp):
                return np.nan
            inner_spread = float(np.quantile(valid, i_hi) - np.quantile(valid, i_lo))
            if not np.isfinite(inner_spread) or abs(inner_spread) < 1e-12:
                return np.nan
            outer_spread = float(np.quantile(valid, o_hi) - np.quantile(valid, o_lo))
            return float(outer_spread / inner_spread)

        return frame_like(x, map_rolling(x.to_numpy(dtype=float), w, _fn))


@register_operator(
    name="ts_tail_ratio",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_tail_ratio",
    source="robust_tail",
)
class TsTailRatio(SeriesOperator):
    """尾部比：abs(Q_high) / abs(Q_low)，Q_low=0 返回 NaN（分母保护）。"""

    metadata = _metadata(
        "ts_tail_ratio",
        "右尾/左尾分位数绝对值比，分母为零返回 NaN。",
        ["x", "window", "q_low", "q_high", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, q_low: float = 0.05, q_high: float = 0.95, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        ql, qh = float(q_low), float(q_high)
        if not 0.0 < ql < qh < 1.0:
            raise ValueError("require 0 < q_low < q_high < 1")
        mp = max(5, int(min_periods))

        def _fn(chunk: np.ndarray) -> float:
            valid = valid_values(chunk)
            if not _has_spread(valid, min_count=mp):
                return np.nan
            q_lo = float(np.quantile(valid, ql))
            q_hi = float(np.quantile(valid, qh))
            if abs(q_lo) < 1e-12:
                return np.nan
            return float(abs(q_hi) / abs(q_lo))

        return frame_like(x, map_rolling(x.to_numpy(dtype=float), w, _fn))


@register_operator(
    name="ts_extreme_cluster_ratio",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_extreme_cluster_ratio",
    source="robust_tail",
)
class TsExtremeClusterRatio(SeriesOperator):
    """极端事件聚集比：相邻极端-极端对 / 极端事件数，衡量极端是否连续出现。"""

    metadata = _metadata(
        "ts_extreme_cluster_ratio",
        "窗口内相邻极端事件占比；极端样本不足返回 NaN。",
        ["x", "window", "threshold", "q", "side", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, threshold: Any = "quantile", q: float = 0.9, side: str = "absolute", min_periods: int = 2, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        side_kind = str(side).lower()
        if side_kind not in {"absolute", "upper", "lower"}:
            raise ValueError("side must be 'absolute', 'upper' or 'lower'")
        quantile = float(q)
        if not 0.0 < quantile < 1.0:
            raise ValueError("q must be in (0, 1)")
        mp = max(2, int(min_periods))
        use_quantile = str(threshold).lower() == "quantile"
        absolute_thr = None if use_quantile else float(threshold)

        def _extreme(chunk: np.ndarray) -> np.ndarray:
            valid = chunk[np.isfinite(chunk)]
            if valid.size == 0:
                return np.zeros_like(chunk, dtype=bool)
            if use_quantile:
                if side_kind == "absolute":
                    thr = float(np.quantile(np.abs(valid), quantile))
                    return np.abs(chunk) >= thr
                if side_kind == "upper":
                    thr = float(np.quantile(valid, quantile))
                    return chunk >= thr
                thr = float(np.quantile(valid, 1.0 - quantile))
                return chunk <= thr
            thr = float(absolute_thr)
            if side_kind == "absolute":
                return np.abs(chunk) >= thr
            if side_kind == "upper":
                return chunk >= thr
            return chunk <= thr

        def _fn(chunk: np.ndarray) -> float:
            finite = np.isfinite(chunk)
            if not _has_spread(chunk[finite], min_count=mp):
                return np.nan
            extreme = _extreme(chunk)
            count = int(extreme.sum())
            if count < mp:
                return np.nan
            # R5 P1-38(c): NaN must not be treated as a non-extreme point that
            # cuts a cluster.  Censor it: skip the position and keep the "previous
            # was extreme" flag, so extremes separated only by missing values
            # still count as adjacent.
            clustered = 0
            prev = False
            for i in range(len(chunk)):
                if not finite[i]:
                    continue
                cur = bool(extreme[i])
                if cur and prev:
                    clustered += 1
                prev = cur
            return float(clustered / count)

        return frame_like(x, map_rolling(x.to_numpy(dtype=float), w, _fn))


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({
            "ts_lower_partial_moment", "ts_upper_partial_moment",
            "ts_expected_shortfall", "ts_quantile_skew", "ts_quantile_kurtosis",
            "ts_tail_ratio", "ts_extreme_cluster_ratio",
        })
    for _canon in (
        "ts_lower_partial_moment", "ts_upper_partial_moment",
        "ts_expected_shortfall", "ts_quantile_skew", "ts_quantile_kurtosis",
        "ts_tail_ratio", "ts_extreme_cluster_ratio",
    ):
        register_polars_bridge(_canon)


_register_surface()
