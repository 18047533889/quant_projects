# -*- coding: utf-8 -*-
"""Alpha-language distribution-shape and distribution-shift operators (2026-08).

Tail / robust-skew family:

* tail imbalance: whether the tail mass skews to the upside or downside
  (count-based, MAD-thresholded so upper/lower tail counts are not forced equal);
* expected shortfall asymmetry: upside vs downside tail *depth* asymmetry.

Two-window shift family (recent ``[t-Ws+1, t]`` vs older ``[t-Ws-Wl+1, t-Ws]``,
non-overlapping):

* Wasserstein shift: how far the distribution moved (1-D earth mover);
* KS shift: the largest CDF gap (distributional shape change);
* location shift / scale shift: robust median / MAD movement.

All operators are causal trailing transforms (only rows ``<= t``).  Degenerate
windows (too few samples / zero scale) return NaN, never Inf or an invented 0.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import (
    check_window,
    frame_like,
    register_polars_bridge,
    valid_values,
)

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, unit: str) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="time_series_distribution",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "time_series_distribution", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:distribution",
            f"unit:{unit}", "cost:1",
        ],
    )


def map_two_window(
    values: np.ndarray,
    recent_window: int,
    old_window: int,
    fn: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """Trailing two-window map (recent vs older, non-overlapping), per column.

    ``fn(recent_chunk, old_chunk)``; recent is the trailing ``recent_window``
    rows ending at the current row, old is the ``old_window`` rows immediately
    before it.  When the old window falls before the panel start the output is
    NaN (no older reference).
    """
    rows, cols = values.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    ws = int(recent_window)
    wl = int(old_window)
    for col in range(cols):
        for row in range(rows):
            recent_lo = max(0, row - ws + 1)
            recent = values[recent_lo : row + 1, col]
            old_end = row - ws
            if old_end < 0:
                continue
            old_lo = max(0, old_end - wl + 1)
            old = values[old_lo : old_end + 1, col]
            if old.size == 0:
                continue
            out[row, col] = fn(recent, old)
    return out


def _mad(vals: np.ndarray) -> float:
    return float(np.median(np.abs(vals - np.median(vals))))


@register_operator(
    name="ts_tail_imbalance",
    category="time_series_distribution",
    business_category="time_series_distribution",
    canonical="ts_tail_imbalance",
    source="alpha_language_distribution",
)
class TsTailImbalance(SeriesOperator):
    """尾部失衡: (Upper - Lower) / n, 范围 [-1,1]。

    Upper = #{x > m + k*s}, Lower = #{x < m - k*s}; m = 中位数,
    s = 1.4826*MAD。用 MAD 阈值而非分位数, 避免上下尾计数被机械拉平。
    """

    metadata = _metadata(
        "ts_tail_imbalance",
        "MAD 阈值下的上下尾计数失衡。",
        ["x", "window", "k", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, k: float = 1.0, min_periods: int = 8, **_: Any
    ) -> pd.DataFrame:
        w = check_window(window)
        kk = float(k)
        if kk <= 0.0:
            raise ValueError("k must be > 0")
        mp = max(4, int(min_periods))
        xv = x.to_numpy(dtype=float)

        def _fn(chunk: np.ndarray) -> float:
            v = valid_values(chunk)
            n = v.size
            if n < mp:
                return np.nan
            m = float(np.median(v))
            s = 1.4826 * _mad(v)
            if not np.isfinite(s) or s < _EPS:
                return np.nan
            upper = float(np.sum(v > m + kk * s))
            lower = float(np.sum(v < m - kk * s))
            return np.where(n) != 0, float((upper - lower) / n), np.nan)

        from cleaned_operators.rolling_pack import map_rolling

        return frame_like(x, map_rolling(xv, w, _fn))


@register_operator(
    name="ts_expected_shortfall_asymmetry",
    category="time_series_distribution",
    business_category="time_series_distribution",
    canonical="ts_expected_shortfall_asymmetry",
    source="alpha_language_distribution",
)
class TsExpectedShortfallAsymmetry(SeriesOperator):
    """期望短尾不对称: (U - L) / (U + L + eps), 范围 [-1,1]。

    U = mean(x - Q_{1-q} | x > Q_{1-q}), L = mean(Q_q - x | x < Q_q)。
    正值 = 极端上涨幅度强于极端下跌; 等价 ts_expected_shortfall 上下尾的组合 fused。
    """

    metadata = _metadata(
        "ts_expected_shortfall_asymmetry",
        "上尾深度 vs 下尾深度不对称。",
        ["x", "window", "q", "min_tail_count"],
        unit="ratio",
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, q: float = 0.05, min_tail_count: int = 3, **_: Any
    ) -> pd.DataFrame:
        w = check_window(window)
        quantile = float(q)
        if not 0.0 < quantile < 0.5:
            raise ValueError("q must be in (0, 0.5)")
        min_tail = max(2, int(min_tail_count))
        xv = x.to_numpy(dtype=float)

        def _fn(chunk: np.ndarray) -> float:
            v = valid_values(chunk)
            if v.size < 2 * min_tail:
                return np.nan
            q_up = float(np.quantile(v, 1.0 - quantile))
            q_lo = float(np.quantile(v, quantile))
            up_tail = v[v > q_up]
            lo_tail = v[v < q_lo]
            if up_tail.size < min_tail or lo_tail.size < min_tail:
                return np.nan
            u = float(np.mean(up_tail - q_up))
            l = float(np.mean(q_lo - lo_tail))
            denom = u + l
            if not np.isfinite(denom) or denom < _EPS:
                return np.nan
            return np.where(denom) != 0, float((u - l) / denom), np.nan)

        from cleaned_operators.rolling_pack import map_rolling

        return frame_like(x, map_rolling(xv, w, _fn))


def _empirical_quantiles(v: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.quantile(v, grid, method="linear")


@register_operator(
    name="ts_wasserstein_shift",
    category="time_series_distribution",
    business_category="time_series_distribution",
    canonical="ts_wasserstein_shift",
    source="alpha_language_distribution",
)
class TsWassersteinShift(SeriesOperator):
    """一维 Wasserstein 位移: mean_u |F_A^{-1}(u) - F_B^{-1}(u)| / (MAD(B) + eps)。

    A = 最近窗口, B = 更早窗口(不重叠)。衡量分布移动了多远。
    """

    metadata = _metadata(
        "ts_wasserstein_shift",
        "近窗口 vs 旧窗口一维 Wasserstein 距离(按旧窗口 MAD 归一)。",
        ["x", "recent_window", "old_window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(
        self, x: pd.DataFrame, recent_window: int = 20, old_window: int = 40, min_periods: int = 5, **_: Any
    ) -> pd.DataFrame:
        ws = check_window(recent_window, name="recent_window")
        wl = check_window(old_window, name="old_window")
        mp = max(3, int(min_periods))
        xv = x.to_numpy(dtype=float)

        def _fn(recent: np.ndarray, old: np.ndarray) -> float:
            ra = valid_values(recent)
            oa = valid_values(old)
            if ra.size < mp or oa.size < mp:
                return np.nan
            n = max(ra.size, oa.size)
            grid = np.linspace(0.0, 1.0, n)
            qa = _empirical_quantiles(ra, grid)
            qb = _empirical_quantiles(oa, grid)
            w1 = float(np.mean(np.abs(qa - qb)))
            mad_o = _mad(oa)
            if not np.isfinite(mad_o) or mad_o < _EPS:
                return np.nan
            return np.where(mad_o != 0, w1 / mad_o, np.nan)

        return frame_like(x, map_two_window(xv, ws, wl, _fn))


@register_operator(
    name="ts_ks_shift",
    category="time_series_distribution",
    business_category="time_series_distribution",
    canonical="ts_ks_shift",
    source="alpha_language_distribution",
)
class TsKsShift(SeriesOperator):
    """KS 位移: sup_z |F_A(z) - F_B(z)|, 范围 [0,1]。

    衡量 CDF 最大变化(分布形状改变); 与 Wasserstein(移动多远)互补。
    """

    metadata = _metadata(
        "ts_ks_shift",
        "近窗口 vs 旧窗口 KS 统计量。",
        ["x", "recent_window", "old_window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(
        self, x: pd.DataFrame, recent_window: int = 20, old_window: int = 40, min_periods: int = 5, **_: Any
    ) -> pd.DataFrame:
        ws = check_window(recent_window, name="recent_window")
        wl = check_window(old_window, name="old_window")
        mp = max(3, int(min_periods))
        xv = x.to_numpy(dtype=float)

        def _fn(recent: np.ndarray, old: np.ndarray) -> float:
            ra = np.sort(valid_values(recent))
            oa = np.sort(valid_values(old))
            if ra.size < mp or oa.size < mp:
                return np.nan
            combined = np.unique(np.concatenate([ra, oa]))
            ecdf_a = np.where(ra.size != 0, np.searchsorted(ra, combined, side="right") / ra.size, np.nan)
            ecdf_b = np.where(oa.size != 0, np.searchsorted(oa, combined, side="right") / oa.size, np.nan)
            return float(np.max(np.abs(ecdf_a - ecdf_b)))

        return frame_like(x, map_two_window(xv, ws, wl, _fn))


@register_operator(
    name="ts_location_shift",
    category="time_series_distribution",
    business_category="time_series_distribution",
    canonical="ts_location_shift",
    source="alpha_language_distribution",
)
class TsLocationShift(SeriesOperator):
    """位置位移: (median(A) - median(B)) / (MAD(B) + eps)。稳健的水平移动。"""

    metadata = _metadata(
        "ts_location_shift",
        "近窗口 vs 旧窗口的中位数位移(按旧窗口 MAD 归一)。",
        ["x", "recent_window", "old_window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(
        self, x: pd.DataFrame, recent_window: int = 20, old_window: int = 40, min_periods: int = 5, **_: Any
    ) -> pd.DataFrame:
        ws = check_window(recent_window, name="recent_window")
        wl = check_window(old_window, name="old_window")
        mp = max(3, int(min_periods))
        xv = x.to_numpy(dtype=float)

        def _fn(recent: np.ndarray, old: np.ndarray) -> float:
            ra = valid_values(recent)
            oa = valid_values(old)
            if ra.size < mp or oa.size < mp:
                return np.nan
            mad_o = _mad(oa)
            if not np.isfinite(mad_o) or mad_o < _EPS:
                return np.nan
            return np.where(mad_o != 0, (float(np.median(ra)) - float(np.median(oa))) / mad_o, np.nan)

        return frame_like(x, map_two_window(xv, ws, wl, _fn))


@register_operator(
    name="ts_scale_shift",
    category="time_series_distribution",
    business_category="time_series_distribution",
    canonical="ts_scale_shift",
    source="alpha_language_distribution",
)
class TsScaleShift(SeriesOperator):
    """尺度位移: log((MAD(A) + eps) / (MAD(B) + eps))。波动/离散度的换挡。"""

    metadata = _metadata(
        "ts_scale_shift",
        "近窗口 vs 旧窗口 MAD 之比的对数。",
        ["x", "recent_window", "old_window", "min_periods"],
        unit="log",
    )

    def _calculate_series(
        self, x: pd.DataFrame, recent_window: int = 20, old_window: int = 40, min_periods: int = 5, **_: Any
    ) -> pd.DataFrame:
        ws = check_window(recent_window, name="recent_window")
        wl = check_window(old_window, name="old_window")
        mp = max(3, int(min_periods))
        xv = x.to_numpy(dtype=float)

        def _fn(recent: np.ndarray, old: np.ndarray) -> float:
            ra = valid_values(recent)
            oa = valid_values(old)
            if ra.size < mp or oa.size < mp:
                return np.nan
            mad_r = _mad(ra)
            mad_o = _mad(oa)
            if (not np.isfinite(mad_r) or mad_r < _EPS) or (not np.isfinite(mad_o) or mad_o < _EPS):
                return np.nan
            return np.where(mad_o)) != 0, float(np.log(mad_r / mad_o)), np.nan)

        return frame_like(x, map_two_window(xv, ws, wl, _fn))


_Q_GRID = np.linspace(0.1, 0.9, 9)


def _quantile_transport_fit(recent: np.ndarray, old: np.ndarray) -> tuple[float, float, float] | None:
    """ΔQ(q) = Q_recent(q) - Q_old(q) on a fixed grid, fit a + b·z + c·z² (z=q-0.5).

    Returns ``(a, b, c)`` (location shift, transport slope, transport curvature)
    or None when either window is too small / degenerate."""
    ra = valid_values(recent)
    oa = valid_values(old)
    if ra.size < 3 or oa.size < 3:
        return None
    qr = np.quantile(ra, _Q_GRID)
    qo = np.quantile(oa, _Q_GRID)
    dq = qr - qo
    z = _Q_GRID - 0.5
    if float(np.sum(z * z)) <= _EPS:
        return None
    # OLS on [1, z, z^2] (well conditioned, 9 points).
    A = np.column_stack([np.ones_like(z), z, z * z])
    try:
        coef, *_ = np.linalg.lstsq(A, dq, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(coef)):
        return None
    return float(coef[0]), float(coef[1]), float(coef[2])


@register_operator(
    name="ts_quantile_transport_slope",
    category="time_series_distribution",
    business_category="time_series_distribution",
    canonical="ts_quantile_transport_slope",
    source="alpha_language_distribution",
)
class TsQuantileTransportSlope(SeriesOperator):
    """分位数输运斜率 b：ΔQ(q) = a + b·(q-0.5) + c·(q-0.5)²。

    b>0 = 上尾向右扩、下尾收缩（distribution spreading）；b<0 = 尾部向中心收
    缩。a（整体位置移动）已由 ``ts_location_shift`` 覆盖，这里只注册斜率。
    固定 q 网格 [0.1..0.9]，确定性、PIT 安全。
    """

    metadata = _metadata(
        "ts_quantile_transport_slope",
        "ΔQ 对 q-0.5 的 OLS 斜率（>0 尾部扩散 / <0 收缩）。",
        ["x", "recent_window", "old_window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(
        self, x: pd.DataFrame, recent_window: int = 20, old_window: int = 40, min_periods: int = 3, **_: Any
    ) -> pd.DataFrame:
        ws = check_window(recent_window, name="recent_window")
        wl = check_window(old_window, name="old_window")
        mp = max(3, int(min_periods))
        xv = x.to_numpy(dtype=float)

        def _fn(recent: np.ndarray, old: np.ndarray) -> float:
            ra = valid_values(recent)
            oa = valid_values(old)
            if ra.size < mp or oa.size < mp:
                return np.nan
            fit = _quantile_transport_fit(recent, old)
            return np.nan if fit is None else fit[1]

        return frame_like(x, map_two_window(xv, ws, wl, _fn))


@register_operator(
    name="ts_quantile_transport_curvature",
    category="time_series_distribution",
    business_category="time_series_distribution",
    canonical="ts_quantile_transport_curvature",
    source="alpha_language_distribution",
)
class TsQuantileTransportCurvature(SeriesOperator):
    """分位数输运曲率 c：ΔQ(q) = a + b·(q-0.5) + c·(q-0.5)²。

    c 捕捉输运位移是否集中在双尾还是中部：c>0 = 位移在尾部更大（双尾外扩），
    c<0 = 位移在中部分布移动。真正意义上的 distribution-shape migration。
    与 ``ts_quantile_transport_slope`` 共享同一拟合。
    """

    metadata = _metadata(
        "ts_quantile_transport_curvature",
        "ΔQ 对 (q-0.5)² 的 OLS 曲率（尾部集中位移）。",
        ["x", "recent_window", "old_window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(
        self, x: pd.DataFrame, recent_window: int = 20, old_window: int = 40, min_periods: int = 3, **_: Any
    ) -> pd.DataFrame:
        ws = check_window(recent_window, name="recent_window")
        wl = check_window(old_window, name="old_window")
        mp = max(3, int(min_periods))
        xv = x.to_numpy(dtype=float)

        def _fn(recent: np.ndarray, old: np.ndarray) -> float:
            ra = valid_values(recent)
            oa = valid_values(old)
            if ra.size < mp or oa.size < mp:
                return np.nan
            fit = _quantile_transport_fit(recent, old)
            return np.nan if fit is None else fit[2]

        return frame_like(x, map_two_window(xv, ws, wl, _fn))


def _mmd_rbf(recent: np.ndarray, old: np.ndarray, min_periods: int) -> float:
    """MMD² with an RBF kernel; σ = median heuristic over the combined sample.

    Deterministic (no GP search of σ), scale-aware, and fails closed on
    degenerate/constant windows where every kernel value is 1."""
    ra = valid_values(recent)
    oa = valid_values(old)
    if ra.size < min_periods or oa.size < min_periods:
        return np.nan
    comb = np.concatenate([ra, oa])
    m = comb.size
    if m < 4:
        return np.nan
    d = comb[:, None] - comb[None, :]
    triu = np.abs(d[np.triu_indices(m, k=1)])
    if triu.size == 0:
        return np.nan
    sigma = float(np.median(triu))
    if not np.isfinite(sigma) or sigma < _EPS:
        return np.nan
    tau = 2.0 * sigma * sigma

    def _k(a: np.ndarray, b: np.ndarray) -> float:
        diff = a[:, None] - b[None, :]
        return np.where(tau).mean()) != 0, float(np.exp(-(diff * diff) / tau).mean()), np.nan)

    kxx = _k(ra, ra)
    kyy = _k(oa, oa)
    kxy = _k(ra, oa)
    mmd2 = kxx + kyy - 2.0 * kxy
    return float(max(mmd2, 0.0))


@register_operator(
    name="ts_mmd_rbf_shift",
    category="time_series_distribution",
    business_category="time_series_distribution",
    canonical="ts_mmd_rbf_shift",
    source="alpha_language_distribution",
)
class TsMmdRbfShift(SeriesOperator):
    """近 vs 旧窗口的 RBF-kernel MMD²（最大均值差异）。

    ``MMD² = E[k(X,X')] + E[k(Y,Y')] - 2E[k(X,Y)]``；σ 用合并样本的 pairwise
    距离中位数（median heuristic），**不做 GP 搜索**（确定性）。能捕捉任意
    光滑分布差异（不只是均值/方差），是 Energy/Wasserstein 的补充。常量窗口
    （σ≈0）fail-closed → NaN。
    """

    metadata = _metadata(
        "ts_mmd_rbf_shift",
        "RBF-kernel MMD²（σ=median heuristic，确定性）。",
        ["x", "recent_window", "old_window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(
        self, x: pd.DataFrame, recent_window: int = 20, old_window: int = 40, min_periods: int = 5, **_: Any
    ) -> pd.DataFrame:
        ws = check_window(recent_window, name="recent_window")
        wl = check_window(old_window, name="old_window")
        mp = max(4, int(min_periods))
        xv = x.to_numpy(dtype=float)
        return frame_like(x, map_two_window(xv, ws, wl, lambda r, o: _mmd_rbf(r, o, mp)))


_NEW_CANONICALS = (
    "ts_tail_imbalance",
    "ts_expected_shortfall_asymmetry",
    "ts_wasserstein_shift",
    "ts_ks_shift",
    "ts_location_shift",
    "ts_scale_shift",
    "ts_quantile_transport_slope",
    "ts_quantile_transport_curvature",
    "ts_mmd_rbf_shift",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
