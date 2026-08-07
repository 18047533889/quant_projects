# -*- coding: utf-8 -*-
"""Advanced minute-frequency operators (2026-08 Gemini round, P1/P2).

Minute panels in, one scalar per ``(TradeDate, Symbol)`` out.  All kernels are
prefix-causal (only the day's own bars plus that day's daily limit prices) and
deterministic (quantile grids, no random projection).

* ``intraday_wasserstein_pair_distance``          — W1 between two intraday
  distributions (e.g. stock vs market minute returns), MAD-standardised (P1).
* ``intraday_barrier_approach_acceleration``      — acceleration of the approach
  to the day's upper/lower limit price, from the second difference of the
  log-headroom (P1).
* ``intraday_wasserstein_quantile_pca_score``     — Wasserstein-geodesic PCA
  score of the daily minute-return quantile curve vs a trailing reference (P2).
* ``intraday_wasserstein_quantile_pca_residual``  — projection residual of the
  same curve (P2).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.microstructure.intraday_agg import _as_panel, _daily_agg_two

_EPS = 1e-12
_QGRID_PAIR = np.linspace(0.01, 0.99, 99)
_QGRID_PCA = np.linspace(0.02, 0.98, 49)


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    cost: int,
    extra_tags: tuple[str, ...] = (),
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="intraday_microstructure",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "intraday", "daily_agg", "minute", "pit_safe", "causal", "typed_v2",
            "deterministic", *extra_tags,
            f"signature:{','.join(params)}->series", f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _pair_w1(a: np.ndarray, b: np.ndarray) -> float:
    av = a[np.isfinite(a)]
    bv = b[np.isfinite(b)]
    if av.size < 5 or bv.size < 5:
        return np.nan
    mad = float(np.median(np.abs(bv - np.median(bv))))
    qa = np.quantile(av, _QGRID_PAIR)
    qb = np.quantile(bv, _QGRID_PAIR)
    return float(np.mean(np.abs(qa - qb))) / (mad + _EPS)


@register_operator(
    name="intraday_wasserstein_pair_distance",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_wasserstein_pair_distance",
    source="advanced_intraday",
)
class IntradayWassersteinPairDistance(SeriesOperator):
    """两个日内分布之间的 Wasserstein-1 距离（MAD 标准化）。

    同日两只序列（如个股 vs 市场分钟收益）的 W1，在 99 点均匀分位网格上计算
    ``mean|Q_X(q) - Q_Y(q)|``，再除以历史 MAD(Y)。市场基准由来源层构造
    （截面中位数分钟收益），算子本身保持序列无关。P1。
    """

    metadata = _metadata(
        "intraday_wasserstein_pair_distance",
        "两日内分布 W1 距离（分位网格 + MAD 标准化）。",
        ["x", "y"],
        unit="ratio",
        cost=4,
    )

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, **_: Any) -> pd.DataFrame:
        return _daily_agg_two(x, y, _pair_w1)


def _barrier_approach(day_vals: np.ndarray, b_up: float, b_dn: float, lookback: int) -> float:
    """Mean approach acceleration to the upper/lower barrier over approach minutes."""
    out: list[float] = []
    for headroom, barrier in ((b_up, True), (b_dn, False)):
        if not np.isfinite(barrier) or barrier <= 0.0:
            continue
        if barrier:
            h = 1.0 - day_vals / barrier
        else:
            h = 1.0 - barrier / day_vals
        h = np.where(np.isfinite(h) & (day_vals > 0.0), h, np.nan)
        v = np.full(h.shape, np.nan, dtype=float)
        v[1:] = h[1:] - h[:-1]
        a = np.full(h.shape, np.nan, dtype=float)
        a[2:] = v[2:] - v[1:-1]
        # Restrict to the most recent ``lookback`` minutes.
        lo = max(0, h.shape[0] - int(lookback))
        v_slice = v[lo:]
        a_slice = a[lo:]
        m = int(v_slice.shape[0])
        if m < 2:
            continue
        # Steady approach = velocity negative on two consecutive minutes; the
        # acceleration during that approach is the second difference at minute+1.
        steady = (v_slice[: m - 1] < 0.0) & (v_slice[1:] < 0.0)
        a_approach = a_slice[1:]
        sel = steady & np.isfinite(a_approach)
        if sel.any():
            out.append(float(-np.mean(a_approach[sel])))
    if not out:
        return np.nan
    return float(max(out))


def _barrier_series(close_day: np.ndarray, hl: float, ll: float, lookback: int) -> float:
    if hl <= 0.0 or ll <= 0.0 or not np.any(close_day > 0.0):
        return np.nan
    return _barrier_approach(close_day, hl, ll, lookback)


@register_operator(
    name="intraday_barrier_approach_acceleration",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_barrier_approach_acceleration",
    source="advanced_intraday",
)
class IntradayBarrierApproachAcceleration(SeriesOperator):
    """涨/跌停接近加速度。

    对当日分钟收盘价相对上/下停板的 log-headroom
    ``h = 1 - close/limit``（下板用 ``1 - limit/close``）求二阶差分，
    ``approach = v[m]<0 且 v[m-1]<0``（连续逼近），输出
    ``-mean(二阶差分)`` 即"接近加速度"，上/下两个方向取较大者。仅用当日
    数据 + 当日涨跌停价，PIT 安全。P1。
    """

    metadata = _metadata(
        "intraday_barrier_approach_acceleration",
        "涨/跌停接近加速度（headroom 二阶差分，连续逼近时段平均）。",
        ["close", "high_limit", "low_limit", "lookback"],
        unit="ratio",
        cost=5,
        # Minute ``close`` mixes with daily limit-price panels: the op aligns
        # them per trading day itself, so panel-broadcast is intentional.
        extra_tags=("allow_panel_broadcast",),
    )

    def _calculate_series(
        self,
        close: pd.DataFrame,
        high_limit: pd.DataFrame,
        low_limit: pd.DataFrame,
        lookback: int = 10,
        **_: Any,
    ) -> pd.DataFrame:
        lb = max(3, int(lookback))
        close = _as_panel(close)
        out: dict[str, pd.Series] = {}
        for inst in close.columns:
            c = close[inst]
            hl = high_limit[inst] if inst in high_limit.columns else None
            ll = low_limit[inst] if inst in low_limit.columns else None
            per_day: dict[pd.Timestamp, float] = {}
            for day, group in c.groupby(c.index.normalize()):
                vals = np.asarray(group, dtype=float)
                if not np.any(np.isfinite(vals)):
                    per_day[day] = np.nan
                    continue
                if hl is None or ll is None:
                    per_day[day] = np.nan
                    continue
                try:
                    hl_day = float(hl.get(day, np.nan))
                    ll_day = float(ll.get(day, np.nan))
                    per_day[day] = _barrier_series(vals, hl_day, ll_day, lb)
                except (ValueError, ZeroDivisionError, OverflowError):
                    per_day[day] = np.nan
            out[inst] = pd.Series(per_day, dtype=float)
        if not out:
            return pd.DataFrame(dtype=float)
        return pd.DataFrame(out).sort_index()


def _quantile_curve(day_returns: np.ndarray) -> np.ndarray | None:
    r = day_returns[np.isfinite(day_returns)]
    if r.size < 5:
        return None
    return np.quantile(r, _QGRID_PCA)


def _pca_score_series(
    day_returns: list[np.ndarray],
    lookback: int,
    k: int,
) -> np.ndarray:
    rows = len(day_returns)
    out = np.full(rows, np.nan, dtype=float)
    curves: list[np.ndarray | None] = [_quantile_curve(r) for r in day_returns]
    for i in range(rows):
        if curves[i] is None:
            continue
        past = [curves[j] for j in range(max(0, i - lookback), i) if curves[j] is not None]
        if len(past) < max(3, k + 1):
            continue
        P = np.stack(past)
        center = P.mean(axis=0)
        Pc = P - center
        _, _, vt = np.linalg.svd(Pc, full_matrices=False)
        if vt.shape[0] < k:
            continue
        vk = vt[k - 1]
        # Deterministic sign orientation: largest-|loading| element positive.
        ax = int(np.argmax(np.abs(vk)))
        if vk[ax] < 0.0:
            vk = -vk
        out[i] = float((curves[i] - center) @ vk)
    return out


def _pca_resid_series(
    day_returns: list[np.ndarray],
    lookback: int,
    k: int,
) -> np.ndarray:
    rows = len(day_returns)
    out = np.full(rows, np.nan, dtype=float)
    curves: list[np.ndarray | None] = [_quantile_curve(r) for r in day_returns]
    for i in range(rows):
        if curves[i] is None:
            continue
        past = [curves[j] for j in range(max(0, i - lookback), i) if curves[j] is not None]
        if len(past) < max(3, k + 1):
            continue
        P = np.stack(past)
        center = P.mean(axis=0)
        Pc = P - center
        _, _, vt = np.linalg.svd(Pc, full_matrices=False)
        K = min(k, vt.shape[0])
        loading = vt[:K]
        for idx in range(K):
            ax = int(np.argmax(np.abs(loading[idx])))
            if loading[idx][ax] < 0.0:
                loading[idx] = -loading[idx]
        score = (curves[i] - center) @ loading.T
        proj = score @ loading
        out[i] = float(np.linalg.norm(curves[i] - center - proj))
    return out


def _per_day_returns(panel: pd.DataFrame) -> tuple[list[pd.Timestamp], list[np.ndarray]]:
    day_list: list[pd.Timestamp] = []
    day_ret: list[np.ndarray] = []
    for day, group in panel.groupby(panel.index.normalize()):
        day_list.append(day)
        day_ret.append(np.asarray(group, dtype=float))
    return day_list, day_ret


def _quantile_pca_panel(
    returns: pd.DataFrame,
    lookback: int,
    k: int,
    residual: bool,
) -> pd.DataFrame:
    returns = _as_panel(returns)
    out: dict[str, pd.Series] = {}
    for inst in returns.columns:
        days, rets = _per_day_returns(returns[inst])
        values = _pca_resid_series(rets, lookback, k) if residual else _pca_score_series(rets, lookback, k)
        out[inst] = pd.Series({d: v for d, v in zip(days, values)}, dtype=float)
    if not out:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(out).sort_index()


@register_operator(
    name="intraday_wasserstein_quantile_pca_score",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_wasserstein_quantile_pca_score",
    source="advanced_intraday",
    status="experimental",
)
class IntradayWassersteinQuantilePcaScore(SeriesOperator):
    """日内收益分位曲线（Wasserstein-geodesic 表示）的滚动 PCA 得分。

    每日取 49 点固定分位网格的分钟收益分位数作为 ``Q_t``，严格只用过去
    ``window`` 日的曲线拟合 PCA，输出当前曲线在第 ``k`` 主成分上的投影得分。
    特征向量符号按最大 |loading| 元素为正做确定性定向。P2 / Research。
    """

    metadata = _metadata(
        "intraday_wasserstein_quantile_pca_score",
        "日内收益分位曲线滚动 PCA 第 k 主成分得分。",
        ["returns", "window", "k"],
        unit="ratio",
        cost=8,
    )

    def _calculate_series(self, returns: pd.DataFrame, window: int = 60, k: int = 1, **_: Any) -> pd.DataFrame:
        w = int(window)
        kk = int(k)
        if w < 2 or kk < 1:
            raise ValueError("intraday_wasserstein_quantile_pca_score requires window >= 2, k >= 1")
        return _quantile_pca_panel(returns, w, kk, residual=False)


@register_operator(
    name="intraday_wasserstein_quantile_pca_residual",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_wasserstein_quantile_pca_residual",
    source="advanced_intraday",
    status="experimental",
)
class IntradayWassersteinQuantilePcaResidual(SeriesOperator):
    """日内收益分位曲线滚动 PCA 投影残差范数。

    与 score 同管线；输出 ``||Q_t - Q̄ - projection_k||_2``，度量当日曲线相对
    历史低维主空间的距离（曲线形状的离群度）。P2 / Research。
    """

    metadata = _metadata(
        "intraday_wasserstein_quantile_pca_residual",
        "日内收益分位曲线滚动 PCA 投影残差范数。",
        ["returns", "window", "k"],
        unit="ratio",
        cost=8,
    )

    def _calculate_series(self, returns: pd.DataFrame, window: int = 60, k: int = 1, **_: Any) -> pd.DataFrame:
        w = int(window)
        kk = int(k)
        if w < 2 or kk < 1:
            raise ValueError("intraday_wasserstein_quantile_pca_residual requires window >= 2, k >= 1")
        return _quantile_pca_panel(returns, w, kk, residual=True)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS)
        | {"intraday_wasserstein_pair_distance", "intraday_barrier_approach_acceleration"}
    )
    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS)
        | {
            "intraday_wasserstein_quantile_pca_score",
            "intraday_wasserstein_quantile_pca_residual",
        }
    )


_register_surface()
