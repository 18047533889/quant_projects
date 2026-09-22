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

from factor_engine.cleaned_operators.base import OperatorMetadata, ParamRole, ParamSpec, RelationalParamSpec, SeriesOperator, register_operator
from factor_engine.cleaned_operators.rolling_pack import (
    check_window,
    frame_like,
    map_rolling,
    register_polars_bridge,
    valid_values,
)


def _metadata(
    name: str, description: str, params: list[str], *, unit: str,
    param_specs: dict[str, ParamSpec], relational_specs: list[RelationalParamSpec],
) -> OperatorMetadata:
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
        param_specs=param_specs,
        relational_specs=relational_specs,
    )


def _has_spread(valid: np.ndarray, *, eps: float = 1e-12, min_count: int = 2) -> bool:
    """False when the finite window is empty, too small or (near-)constant."""
    if valid.size < max(2, int(min_count)):
        return False
    return bool(np.isfinite(valid).all() and np.std(valid, ddof=0) >= eps)


# ---------------------------------------------------------------------------
# R62 vectorised kernels for the trailing-window risk operators.
#
# Window == the raw positional slice values[lo(r):r+1] with
# lo(r) = max(0, r - w + 1) (exactly map_rolling's slice; no calendar
# arithmetic).  Quantiles use np.quantile's default 'linear' rule on the finite
# values only; non-finite slots are padded with +inf so they sort past every
# finite value and never enter a quantile / tail / cluster decision.
# ---------------------------------------------------------------------------
def _r62_win(rows: int, w: int):
    r = np.arange(rows)[:, None]
    lo = np.maximum(0, r - w + 1)
    return lo, lo + np.arange(w)[None, :]


def _r62_quantile(V: np.ndarray, fin: np.ndarray, q: float) -> np.ndarray:
    rows, W = V.shape
    X = np.sort(np.where(fin, V, np.inf), axis=1)
    cnt = fin.sum(axis=1)
    pos = q * (cnt - 1)
    li = np.clip(np.floor(pos).astype(np.int64), 0, W - 1)
    hi = np.clip(np.ceil(pos).astype(np.int64), 0, W - 1)
    frac = pos - li
    rr = np.arange(rows)
    # Rows with no finite value (cnt == 0) index a padded +inf and would raise a
    # spurious invalid-op warning; they are rejected by the caller's guard.
    with np.errstate(invalid="ignore"):
        return X[rr, li] * (1.0 - frac) + X[rr, hi] * frac


# ---------------------------------------------------------------------------
# R63 batch-5: batched trailing-window quantiles for the quantile-moment
# operators.  ``_r63_window_quantiles`` reproduces ``map_rolling``'s window
# (``xv[max(0, r-w+1) : r+1, c]``, NaN-padded head) for every row at once via
# ``sliding_window_view`` and takes all requested quantiles off ONE shared
# per-window sort, using exactly the ``_r62_quantile`` 'linear' recipe over the
# finite values (non-finite slots padded +inf sort past every finite value).
# The ``_has_spread`` gate statistics (finite-window std ddof=0, NaN-safe, and
# the finite count) come back alongside so the caller applies the identical
# gates.  The per-window ``_fn`` closures above are kept as semantic reference.
# ---------------------------------------------------------------------------
def _r63_window_quantiles(xv: np.ndarray, w: int, qs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = xv.shape
    padded = np.concatenate([np.full((w - 1, cols), np.nan), xv], axis=0)
    V = np.lib.stride_tricks.sliding_window_view(padded, w, axis=0)
    V = V.reshape(rows * cols, w)
    fin = np.isfinite(V)
    cnt = fin.sum(axis=1)
    Xs = np.sort(np.where(fin, V, np.inf), axis=1)
    rr = np.arange(rows * cols)
    out = np.empty((len(qs), rows * cols), dtype=float)
    for k, q in enumerate(qs):
        pos = q * (cnt - 1)
        li = np.clip(np.floor(pos).astype(np.int64), 0, w - 1)
        hi = np.clip(np.ceil(pos).astype(np.int64), 0, w - 1)
        frac = pos - li
        with np.errstate(invalid="ignore"):
            out[k] = Xs[rr, li] * (1.0 - frac) + Xs[rr, hi] * frac
    c = np.maximum(cnt, 1)
    mean = np.where(fin, V, 0.0).sum(axis=1) / c
    dev = np.where(fin, V - mean[:, None], 0.0)
    std = np.sqrt((dev * dev).sum(axis=1) / c)
    return out, std, cnt


@register_operator(
    name="ts_lower_partial_moment",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_lower_partial_moment",
    source="robust_tail")
class TsLowerPartialMoment(SeriesOperator):
    """下偏矩：mean(max(threshold - x, 0)^order)，order=1 下行缺口、order=2 下行方差。"""

    metadata = _metadata(
        "ts_lower_partial_moment",
        "下偏矩 mean(max(threshold-x,0)^order)。",
        ["x", "window", "threshold", "order", "min_periods"],
        unit="level",
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, history_semantics="max_rows", param_role=ParamRole.HORIZON),
            "threshold": ParamSpec(dtype=float, default=0.0, param_role=ParamRole.STATE_THRESHOLD),
            "order": ParamSpec(dtype=float, min=1.0, default=1.0, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "min_periods": ParamSpec(dtype=int, min=2, default=2, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        }, relational_specs=[RelationalParamSpec("min_periods <= window")],
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
    source="robust_tail")
class TsUpperPartialMoment(SeriesOperator):
    """上偏矩：mean(max(x - threshold, 0)^order)。"""

    metadata = _metadata(
        "ts_upper_partial_moment",
        "上偏矩 mean(max(x-threshold,0)^order)。",
        ["x", "window", "threshold", "order", "min_periods"],
        unit="level",
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, history_semantics="max_rows", param_role=ParamRole.HORIZON),
            "threshold": ParamSpec(dtype=float, default=0.0, param_role=ParamRole.STATE_THRESHOLD),
            "order": ParamSpec(dtype=float, min=1.0, default=1.0, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "min_periods": ParamSpec(dtype=int, min=2, default=2, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        }, relational_specs=[RelationalParamSpec("min_periods <= window")],
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
    source="robust_tail")
class TsExpectedShortfall(SeriesOperator):
    """历史期望损失（尾部均值）：side='lower' 左尾、side='upper' 右尾，尾部样本不足返回 NaN。"""

    metadata = _metadata(
        "ts_expected_shortfall",
        "历史期望损失 mean(tail) 基于窗口内分位数，尾部样本不足返回 NaN。",
        ["x", "window", "q", "side", "min_tail_count"],
        unit="level",
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=60, history_semantics="max_rows", param_role=ParamRole.HORIZON),
            "q": ParamSpec(dtype=float, min=1e-12, max=0.5, default=0.05, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "side": ParamSpec(dtype=str, choices=("lower", "upper"), default="lower", searchable=False, param_role=ParamRole.POLICY),
            "min_tail_count": ParamSpec(dtype=int, min=2, default=5, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        }, relational_specs=[RelationalParamSpec("min_tail_count <= window")],
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, q: float = 0.05, side: str = "lower", min_tail_count: int = 5, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        quantile = float(q)
        # Round-7 P0: ES ``q`` is a *tail fraction* -- q > 0.5 is not tail-risk
        # semantics (a q=0.8 "lower tail" is the bottom 80% of the distribution).
        if not 0.0 < quantile <= 0.5:
            raise ValueError("q must be in (0, 0.5] for expected-shortfall tail semantics")
        side_kind = str(side).lower()
        if side_kind not in {"lower", "upper"}:
            raise ValueError("side must be 'lower' or 'upper'")
        min_tail = max(2, int(min_tail_count))

        # R62: threshold is the window quantile itself (np.quantile 'linear'),
        # tail = finite values on the tail side of it, out = mean(tail).
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        r = np.arange(rows)[:, None]
        lo, i = _r62_win(rows, w)
        valid = i <= r
        idx = np.clip(i, 0, rows - 1)
        lower = side_kind == "lower"
        for col in range(cols):
            V = xv[idx, col]
            fin = valid & np.isfinite(V)
            cnt = fin.sum(axis=1)
            if lower:
                thr = _r62_quantile(V, fin, quantile)
                tail = fin & (V <= thr[:, None])
            else:
                thr = _r62_quantile(V, fin, 1.0 - quantile)
                tail = fin & (V >= thr[:, None])
            tcnt = tail.sum(axis=1)
            tsum = np.where(tail, V, 0.0).sum(axis=1)
            ok = (cnt >= min_tail) & (tcnt >= min_tail)
            out[:, col] = np.where(ok, tsum / np.maximum(tcnt, 1), np.nan)
        return frame_like(x, out)


@register_operator(
    name="ts_quantile_skew",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_quantile_skew",
    source="robust_tail")
class TsQuantileSkew(SeriesOperator):
    """Bowley 分位数偏度：(Q_high + Q_low - 2 Q_mid) / (Q_high - Q_low)。"""

    metadata = _metadata(
        "ts_quantile_skew",
        "Bowley 分位数偏度，退化区间返回 NaN。",
        ["x", "window", "q_low", "q_mid", "q_high", "min_periods"],
        unit="ratio",
        param_specs={
            "window": ParamSpec(dtype=int, min=5, default=20, history_semantics="max_rows", param_role=ParamRole.HORIZON),
            "q_low": ParamSpec(dtype=float, min=1e-12, max=1.0 - 1e-12, default=0.1, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "q_mid": ParamSpec(dtype=float, min=1e-12, max=1.0 - 1e-12, default=0.5, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "q_high": ParamSpec(dtype=float, min=1e-12, max=1.0 - 1e-12, default=0.9, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "min_periods": ParamSpec(dtype=int, min=5, default=5, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        }, relational_specs=[RelationalParamSpec("q_low < q_mid and q_mid < q_high"), RelationalParamSpec("min_periods <= window")],
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

        # R63: one shared per-window sort serves every quantile; the _fn above
        # is kept as the semantic reference.
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        Q, std, cnt = _r63_window_quantiles(xv, w, (ql, qm, qh))
        denom = Q[2] - Q[0]
        ok = (cnt >= mp) & (std >= 1e-12) & np.isfinite(denom) & (np.abs(denom) >= 1e-12)
        with np.errstate(invalid="ignore", divide="ignore"):
            val = (Q[2] + Q[0] - 2.0 * Q[1]) / denom
        return frame_like(x, np.where(ok, val, np.nan).reshape(rows, cols))


@register_operator(
    name="ts_quantile_kurtosis",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_quantile_kurtosis",
    source="robust_tail")
class TsQuantileKurtosis(SeriesOperator):
    """分位数峰度（尾部宽度比）：(Q_outer_hi - Q_outer_lo) / (Q_inner_hi - Q_inner_lo)。标准正态约 2.906。"""

    metadata = _metadata(
        "ts_quantile_kurtosis",
        "分位数峰度（尾部宽度相对中部宽度比），退化中部区间返回 NaN。",
        ["x", "window", "outer", "inner", "min_periods"],
        unit="ratio",
        param_specs={
            "window": ParamSpec(dtype=int, min=8, default=60, history_semantics="max_rows", param_role=ParamRole.HORIZON),
            "outer": ParamSpec(dtype=tuple, items=ParamSpec(dtype=float, min=1e-12, max=1.0 - 1e-12), min_items=2, max_items=2, default=(0.025, 0.975), param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "inner": ParamSpec(dtype=tuple, items=ParamSpec(dtype=float, min=1e-12, max=1.0 - 1e-12), min_items=2, max_items=2, default=(0.25, 0.75), param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "min_periods": ParamSpec(dtype=int, min=8, default=8, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        }, relational_specs=[RelationalParamSpec("min_periods <= window")],
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
        if not (o_lo < i_lo < i_hi < o_hi):
            raise ValueError("outer quantiles must strictly enclose inner quantiles")
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

        # R63: one shared per-window sort serves every quantile; the _fn above
        # is kept as the semantic reference.
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        Q, std, cnt = _r63_window_quantiles(xv, w, (i_hi, i_lo, o_hi, o_lo))
        inner_spread = Q[0] - Q[1]
        outer_spread = Q[2] - Q[3]
        ok = (cnt >= mp) & (std >= 1e-12) & np.isfinite(inner_spread) & (np.abs(inner_spread) >= 1e-12)
        with np.errstate(invalid="ignore", divide="ignore"):
            val = outer_spread / inner_spread
        return frame_like(x, np.where(ok, val, np.nan).reshape(rows, cols))


@register_operator(
    name="ts_tail_ratio",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_tail_ratio",
    source="robust_tail")
class TsTailRatio(SeriesOperator):
    """尾部比：abs(Q_high) / abs(Q_low)，Q_low=0 返回 NaN（分母保护）。"""

    metadata = _metadata(
        "ts_tail_ratio",
        "右尾/左尾分位数绝对值比，分母为零返回 NaN。",
        ["x", "window", "q_low", "q_high", "min_periods"],
        unit="ratio",
        param_specs={
            "window": ParamSpec(dtype=int, min=5, default=60, history_semantics="max_rows", param_role=ParamRole.HORIZON),
            "q_low": ParamSpec(dtype=float, min=1e-12, max=1.0 - 1e-12, default=0.05, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "q_high": ParamSpec(dtype=float, min=1e-12, max=1.0 - 1e-12, default=0.95, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "min_periods": ParamSpec(dtype=int, min=5, default=5, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        }, relational_specs=[RelationalParamSpec("q_low < q_high"), RelationalParamSpec("min_periods <= window")],
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
    source="robust_tail")
class TsExtremeClusterRatio(SeriesOperator):
    """极端事件聚集比：相邻极端-极端对 / 极端事件数，衡量极端是否连续出现。"""

    metadata = _metadata(
        "ts_extreme_cluster_ratio",
        "窗口内相邻极端事件占比；极端样本不足返回 NaN。",
        ["x", "window", "threshold", "q", "side", "min_periods"],
        unit="ratio",
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=60, history_semantics="max_rows", param_role=ParamRole.HORIZON),
            "threshold": ParamSpec(alternatives=(ParamSpec(dtype=str), ParamSpec(dtype=float)), default="quantile", param_role=ParamRole.STATE_THRESHOLD),
            "q": ParamSpec(dtype=float, min=1e-12, max=1.0 - 1e-12, default=0.9, active_when=("threshold", ("quantile",)), param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "side": ParamSpec(dtype=str, choices=("absolute", "upper", "lower"), default="absolute", searchable=False, param_role=ParamRole.POLICY),
            "min_periods": ParamSpec(dtype=int, min=2, default=2, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        }, relational_specs=[RelationalParamSpec("min_periods <= window")],
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
        if absolute_thr is not None and not np.isfinite(absolute_thr):
            raise ValueError("threshold must be finite or 'quantile'")
        if side_kind == "absolute" and absolute_thr is not None and absolute_thr < 0.0:
            raise ValueError("absolute threshold must be non-negative")

        # R62: same window, same quantile, same censoring of NaN gaps -- the
        # "clustered" count is the number of adjacent extreme *pairs in the
        # finite subsequence* (a NaN gap keeps the previous extreme flag).
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        r = np.arange(rows)[:, None]
        lo, i = _r62_win(rows, w)
        valid = i <= r
        idx = np.clip(i, 0, rows - 1)
        mincnt = max(2, int(mp))
        for col in range(cols):
            V = xv[idx, col]
            fin = valid & np.isfinite(V)
            cnt = fin.sum(axis=1)
            nn = np.maximum(cnt, 1).astype(float)
            mean = np.where(fin, V, 0.0).sum(axis=1) / nn
            cen = np.where(fin, V - mean[:, None], 0.0)
            with np.errstate(invalid="ignore", over="ignore"):
                std0 = np.sqrt((cen * cen).sum(axis=1) / nn)
            spread = (cnt >= mincnt) & (std0 >= 1e-12)
            if use_quantile:
                if side_kind == "absolute":
                    thr = _r62_quantile(np.abs(V), fin, quantile)
                    ext = fin & (np.abs(V) >= thr[:, None])
                elif side_kind == "upper":
                    thr = _r62_quantile(V, fin, quantile)
                    ext = fin & (V >= thr[:, None])
                else:
                    thr = _r62_quantile(V, fin, 1.0 - quantile)
                    ext = fin & (V <= thr[:, None])
            else:
                if side_kind == "absolute":
                    ext = fin & (np.abs(V) >= absolute_thr)
                elif side_kind == "upper":
                    ext = fin & (V >= absolute_thr)
                else:
                    ext = fin & (V <= absolute_thr)
            count = ext.sum(axis=1)
            pos = np.cumsum(fin, axis=1) - 1
            Ec = np.zeros_like(ext)
            rr, cc = np.nonzero(fin)
            Ec[rr, pos[rr, cc]] = ext[rr, cc]
            clustered = (Ec[:, :-1] & Ec[:, 1:]).sum(axis=1)
            ok = spread & (count >= mp)
            out[:, col] = np.where(ok, clustered / np.maximum(count, 1), np.nan)
        return frame_like(x, out)


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

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
