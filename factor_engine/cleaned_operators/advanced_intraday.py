# -*- coding: utf-8 -*-
"""Advanced minute-frequency operators (2026-08 Gemini round, P1/P2).

Minute panels in, one scalar per ``(TradeDate, Symbol)`` out.  All kernels are
prefix-causal (only the day's own bars plus that day's daily limit prices) and
deterministic (quantile grids, no random projection).

* ``intraday_wasserstein_pair_distance``          — W1 between two intraday
  distributions (e.g. stock vs market minute returns), MAD-standardised (P1).
* ``intraday_barrier_approach_acceleration``      — signed acceleration of the
  approach to the day's upper/lower limit price, from the second difference of
  the (linear) headroom (P1).
* ``intraday_quantile_curve_pca_score``           — PCA score of the daily
  minute-return quantile curve vs a trailing reference.  This is Euclidean PCA
  on quantile *curves* (a W2-flavoured coordinate space); it is *not* a
  Wasserstein principal-geodesic analysis, hence the honest name (P2).
* ``intraday_quantile_curve_pca_residual``        — projection residual of the
  same curve (P2).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.microstructure.intraday_agg import _as_panel, _daily_agg_two
from cleaned_operators.rolling_pack import register_polars_udf

_EPS = 1e-12
_EIGEN_GAP_MIN = 1e-2  # below this (lambda_k - lambda_{k+1})/lambda_k the PC is unstable.


def _eigen_gap_unstable(s: np.ndarray, k: int) -> bool:
    """True when ``lambda_k ~ lambda_{k+1}`` makes the k-th PC direction unstable.

    With near-degenerate eigenvalues the PCA basis may rotate/swap (and flip sign
    under sign-fixing), so the *signed* score would jump although the distribution
    structure did not change.  Zero reference variance is not "unstable" — the
    score is then exactly 0.
    """
    if k >= s.size:
        return False  # no (k+1)-th eigenvalue to compare -> guard not applicable.
    sk = float(s[k - 1])
    if sk <= _EPS:
        return False
    return (sk - float(s[k])) / sk < _EIGEN_GAP_MIN
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
    if not np.isfinite(mad) or mad <= _EPS:
        return np.nan  # degenerate (constant) baseline -> undefined scale, fail closed.
    qa = np.quantile(av, _QGRID_PAIR)
    qb = np.quantile(bv, _QGRID_PAIR)
    return float(np.mean(np.abs(qa - qb))) / mad


@register_operator(
    name="intraday_wasserstein_pair_distance",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_wasserstein_pair_distance",
    source="advanced_intraday",
)
class IntradayWassersteinPairDistance(SeriesOperator):
    """两个日内分布之间的 Wasserstein-1 距离（同日 MAD 标准化）。

    同日两只序列（如个股 vs 市场分钟收益）的 W1，在 99 点均匀分位网格上计算
    ``mean|Q_X(q) - Q_Y(q)|``，再除以基准序列的**同日** MAD(Y)（MAD 为 0 时
    fail-closed 返回 NaN）。市场基准由来源层构造（截面中位数分钟收益），算子
    本身保持序列无关。P1。
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
    """Signed approach acceleration to the upper/lower barrier.

    Positive = accelerating toward the *upper* limit (buying pressure), negative
    = accelerating toward the *lower* limit (selling pressure).  The earlier
    implementation swapped the variables (``headroom`` got the limit price,
    ``barrier`` a bool), so the upper branch computed ``1 - price/1`` and the
    lower branch was skipped — the real limit prices were never used.
    """
    upper: float | None = None
    lower: float | None = None
    for limit_price, is_upper in ((b_up, True), (b_dn, False)):
        if not np.isfinite(limit_price) or limit_price <= 0.0:
            continue
        if is_upper:
            h = 1.0 - day_vals / limit_price
        else:
            h = 1.0 - limit_price / day_vals
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
        if not sel.any():
            continue
        acc = float(-np.mean(a_approach[sel]))
        if is_upper:
            upper = acc
        else:
            lower = acc
    if upper is None and lower is None:
        return np.nan
    if upper is None:
        return -lower
    if lower is None:
        return upper
    # Both barriers reachable: keep direction, magnitude of the dominant one.
    return upper if upper >= lower else -lower


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
    """涨/跌停接近加速度（带方向）。

    对当日分钟收盘价相对上/下停板的**线性** headroom
    ``h = 1 - close/limit``（下板用 ``1 - limit/close``）求二阶差分，
    ``approach = v[m]<0 且 v[m-1]<0``（连续逼近），输出 ``-mean(二阶差分)``
    即"接近加速度"。方向保留：加速冲向涨停为正，加速冲向跌停为负（占优方向
    决定符号与大小）。仅用当日数据 + 当日涨跌停价，PIT 安全。P1。
    """

    metadata = _metadata(
        "intraday_barrier_approach_acceleration",
        "涨/跌停接近加速度（线性 headroom 二阶差分，带方向，连续逼近时段平均）。",
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
        _, s, vt = np.linalg.svd(Pc, full_matrices=False)
        if vt.shape[0] < k:
            continue  # rank < k -> the requested PC does not exist (same rule as residual).
        vk = vt[k - 1]
        # Eigen-gap guard: with near-degenerate eigenvalues the PCA basis may
        # rotate/swap and the *signed* score would jump although the structure did
        # not change -> fail closed.
        sk = float(s[k - 1])
        if sk <= _EPS:
            out[i] = 0.0  # zero reference variance -> current curve sits at the center.
            continue
        if _eigen_gap_unstable(s, k):
            continue
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
        if vt.shape[0] < k:
            continue  # rank < k -> projection on k components is undefined (same rule as score).
        loading = vt[:k]
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
    name="intraday_quantile_curve_pca_score",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_quantile_curve_pca_score",
    source="advanced_intraday",
    status="experimental",
)
class IntradayQuantileCurvePcaScore(SeriesOperator):
    """日内收益分位曲线（Euclidean 表示）的滚动 PCA 得分。

    每日取 49 点固定分位网格的分钟收益分位数作为 ``Q_t``，严格只用过去
    ``window`` 日的曲线拟合 PCA，输出当前曲线在第 ``k`` 主成分上的投影得分。
    这是分位曲线空间的普通 Euclidean PCA（近似 1D W2 坐标），**不是**
    Wasserstein 主测地线分析，故不沿用 wasserstein 命名。特征向量符号按最大
    |loading| 元素为正做确定性定向；near-degenerate 特征值（eigen-gap 过小）
    时 fail-closed。P2 / Research。
    """

    metadata = _metadata(
        "intraday_quantile_curve_pca_score",
        "日内收益分位曲线滚动 PCA 第 k 主成分得分（Euclidean，非 Wasserstein PCA）。",
        ["returns", "window", "k"],
        unit="ratio",
        cost=8,
    )

    def _calculate_series(self, returns: pd.DataFrame, window: int = 60, k: int = 1, **_: Any) -> pd.DataFrame:
        w = int(window)
        kk = int(k)
        if w < 2 or kk < 1:
            raise ValueError("intraday_quantile_curve_pca_score requires window >= 2, k >= 1")
        return _quantile_pca_panel(returns, w, kk, residual=False)


@register_operator(
    name="intraday_quantile_curve_pca_residual",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_quantile_curve_pca_residual",
    source="advanced_intraday",
    status="experimental",
)
class IntradayQuantileCurvePcaResidual(SeriesOperator):
    """日内收益分位曲线滚动 PCA 投影残差范数。

    与 score 同管线；输出 ``||Q_t - Q̄ - projection_k||_2``，度量当日曲线相对
    历史低维主空间的距离（曲线形状的离群度）。残差为子空间量，天然
    basis-invariant，不受特征向量旋转/交换影响。P2 / Research。
    """

    metadata = _metadata(
        "intraday_quantile_curve_pca_residual",
        "日内收益分位曲线滚动 PCA 投影残差范数。",
        ["returns", "window", "k"],
        unit="ratio",
        cost=8,
    )

    def _calculate_series(self, returns: pd.DataFrame, window: int = 60, k: int = 1, **_: Any) -> pd.DataFrame:
        w = int(window)
        kk = int(k)
        if w < 2 or kk < 1:
            raise ValueError("intraday_quantile_curve_pca_residual requires window >= 2, k >= 1")
        return _quantile_pca_panel(returns, w, kk, residual=True)


# ---------------------------------------------------------------------------
# Sampling-scale / noise diagnostics and historical-profile surprise
# ---------------------------------------------------------------------------
def _subsampled_rv_dispersion(day_vals: np.ndarray, sampling: int) -> float:
    """Coefficient of variation of realised variances over subsampling offsets."""
    sm = int(sampling)
    m = day_vals.shape[0]
    if m < 2 * sm:
        return np.nan
    rvs = []
    for off in range(sm):
        seg = day_vals[off::sm]
        rv = float(np.sum(seg * seg))
        if np.isfinite(rv):
            rvs.append(rv)
    if len(rvs) < 2:
        return np.nan
    mu = float(np.mean(rvs))
    if mu <= _EPS:
        return np.nan
    return float(np.std(rvs) / mu)


@register_operator(
    name="intraday_subsampled_rv_dispersion",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_subsampled_rv_dispersion",
    source="advanced_intraday",
)
class IntradaySubsampledRvDispersion(SeriesOperator):
    """重采样网格间已实现波动率离散度（Std(RV_k)/Mean(RV_k)）。

    对当日分钟收益按 ``sampling`` 个不同起点偏移构造 RV_k（如 09:30/09:31/...
    起点每 5 分钟一个网格），输出各网格 RV 的变异系数。高 = 波动率估计对
    "从哪一分钟开始采样"极度敏感（jump 集中 / noise / 路径不规则）。P1。
    """

    metadata = _metadata(
        "intraday_subsampled_rv_dispersion",
        "重采样网格已实现波动率变异系数 Std(RV)/Mean(RV)。",
        ["returns", "sampling"],
        unit="ratio",
        cost=4,
    )

    def _calculate_series(self, returns: pd.DataFrame, sampling: int = 5, **_: Any) -> pd.DataFrame:
        sm = int(sampling)
        if sm < 2:
            raise ValueError("intraday_subsampled_rv_dispersion requires sampling >= 2")
        return _daily_agg(returns, lambda v, t: _subsampled_rv_dispersion(v, sm))


def _vol_signature_slope(day_vals: np.ndarray, max_interval: int) -> float:
    """Slope of log(RV(interval)) vs log(interval) over power-of-2 intervals."""
    m = day_vals.shape[0]
    points: list[tuple[float, float]] = []
    iv = 1
    while iv <= int(max_interval) and iv <= m // 2:
        nblocks = m // iv
        if nblocks >= 2:
            blocks = day_vals[: nblocks * iv].reshape(nblocks, iv)
            agg = blocks.sum(axis=1)
            rv = float(np.sum(agg * agg))
            if np.isfinite(rv) and rv > _EPS:
                points.append((float(iv), rv))
        iv *= 2
    if len(points) < 3:
        return np.nan
    xs = np.log(np.array([p[0] for p in points]))
    ys = np.log(np.array([p[1] for p in points]))
    denom = float(np.sum((xs - xs.mean()) ** 2))
    if denom <= _EPS:
        return np.nan
    return float(np.sum((xs - xs.mean()) * (ys - ys.mean())) / denom)


@register_operator(
    name="intraday_volatility_signature_slope",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_volatility_signature_slope",
    source="advanced_intraday",
)
class IntradayVolatilitySignatureSlope(SeriesOperator):
    """波动率签名斜率（log RV vs log 采样区间 回归斜率）。

    RV(interval) 随区间 1,2,4,8,... 变化；noise 主导 → 斜率显著为负（高频被
    微观结构噪声污染）；随机游走 → 斜率≈0；趋势/长记忆 → 可为正。诊断高频
    noise / 流动性结构。P1。
    """

    metadata = _metadata(
        "intraday_volatility_signature_slope",
        "波动率签名斜率（log RV ~ log interval 回归斜率）。",
        ["returns", "max_interval"],
        unit="slope",
        cost=4,
    )

    def _calculate_series(self, returns: pd.DataFrame, max_interval: int = 32, **_: Any) -> pd.DataFrame:
        mi = int(max_interval)
        if mi < 4:
            raise ValueError("intraday_volatility_signature_slope requires max_interval >= 4")
        return _daily_agg(returns, lambda v, t: _vol_signature_slope(v, mi))


def _realized_power_variation(day_vals: np.ndarray, order: float, sampling: int) -> float:
    seg = day_vals[:: int(sampling)]
    fin = seg[np.isfinite(seg)]
    if fin.size < 1:
        return np.nan
    return float(np.sum(np.abs(fin) ** float(order)))


@register_operator(
    name="intraday_realized_power_variation",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_realized_power_variation",
    source="advanced_intraday",
    status="experimental",
)
class IntradayRealizedPowerVariation(SeriesOperator):
    """已实现幂变差 Σ|r|^p（order=p，可子采样）。

    p=2 即 RV；p=4 放大跳跃/大波动（quadricity）；p<1 压缩大波动。比 BV/RV
    更细粒度地刻画波动幅度的尾部分布。P2 / Research。
    """

    metadata = _metadata(
        "intraday_realized_power_variation",
        "已实现幂变差 Σ|r|^p（order=p，子采样）。",
        ["returns", "order", "sampling"],
        unit="power",
        cost=3,
    )

    def _calculate_series(self, returns: pd.DataFrame, order: float = 4.0, sampling: int = 1, **_: Any) -> pd.DataFrame:
        od = float(order)
        sm = int(sampling)
        if od <= 0.0:
            raise ValueError("intraday_realized_power_variation requires order > 0")
        if sm < 1:
            raise ValueError("intraday_realized_power_variation requires sampling >= 1")
        return _daily_agg(returns, lambda v, t: _realized_power_variation(v, od, sm))


def _day_profile(day_vals: np.ndarray, n_slots: int) -> np.ndarray | None:
    """Fixed ``n_slots``-slot profile of a day (equal-count groups, mean per slot)."""
    ns = int(n_slots)
    m = day_vals.shape[0]
    if m < ns:
        return None
    edges = np.linspace(0, m, ns + 1).astype(int)
    prof = np.empty(ns, dtype=float)
    for i in range(ns):
        slot = day_vals[edges[i] : edges[i + 1]]
        fin = slot[np.isfinite(slot)]
        if fin.size == 0:
            return None
        prof[i] = float(fin.mean())
    return prof


def _best_phase(cur: np.ndarray, med: np.ndarray, max_shift: int, n_slots: int) -> float:
    K = min(int(max_shift), int(n_slots) - 1)
    best_k = 0
    best_c = -np.inf
    for k in range(-K, K + 1):
        if k >= 0:
            a = cur[k:]
            b = med[: len(cur) - k]
        else:
            a = cur[: len(cur) + k]
            b = med[-k:]
        ok = np.isfinite(a) & np.isfinite(b)
        if int(ok.sum()) < 3:
            continue
        aa = a[ok]
        bb = b[ok]
        va = float(np.var(aa))
        vb = float(np.var(bb))
        if va <= _EPS or vb <= _EPS:
            continue
        c = float(np.corrcoef(aa, bb)[0, 1])
        if c > best_c:
            best_c = c
            best_k = k
    if best_c <= -np.inf:
        return np.nan
    return float(best_k / int(n_slots))


def _profile_series(day_vals: list[np.ndarray], history_days: int, n_slots: int, cap: float, phase: bool, max_shift: int) -> list[float]:
    n = len(day_vals)
    out: list[float] = [np.nan] * n
    profiles: list[np.ndarray] = []
    for i in range(n):
        if len(profiles) >= int(history_days):
            mat = np.stack(profiles[-int(history_days):])
            med = np.median(mat, axis=0)
            mad = np.median(np.abs(mat - med), axis=0)
            scale = 1.4826 * mad + _EPS
            cur = _day_profile(day_vals[i], n_slots)
            if cur is not None:
                if phase:
                    out[i] = _best_phase(cur, med, int(max_shift), int(n_slots))
                else:
                    z = (cur - med) / scale
                    out[i] = float(np.mean(np.minimum(z * z, float(cap))))
        p = _day_profile(day_vals[i], n_slots)
        if p is not None:
            profiles.append(p)
    return out


def _profile_panel(frame: pd.DataFrame, history_days: int, n_slots: int, cap: float, phase: bool, max_shift: int) -> pd.DataFrame:
    frame = _as_panel(frame)
    out: dict[str, pd.Series] = {}
    for inst in frame.columns:
        days, day_vals = _per_day_returns(frame[inst])
        vals = _profile_series(day_vals, history_days, n_slots, cap, phase, max_shift)
        out[inst] = pd.Series(dict(zip(days, vals)), dtype=float)
    if not out:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(out).sort_index()


@register_operator(
    name="intraday_profile_surprise_energy",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_profile_surprise_energy",
    source="advanced_intraday",
)
class IntradayProfileSurpriseEnergy(SeriesOperator):
    """日内 profile 异常能量（控制时间季节性后的 minute-level 反常度）。

    把每日常模化为固定 ``n_slots`` 个 slot 的 profile，用前 ``history_days`` 天
    逐 slot 的 Median/MAD 构造 z 分，输出 ``mean(min(z², cap))``。与 profile PCA
    residual（当前形状是否脱离历史低维空间）不同：这里度量今天整体有多少
    minute-by-minute 反常。P1。
    """

    metadata = _metadata(
        "intraday_profile_surprise_energy",
        "日内 profile 异常能量 mean(min(z², cap))（z 按历史 slot 中位数/MAD）。",
        ["x", "history_days", "n_slots", "cap"],
        unit="energy",
        cost=6,
    )

    def _calculate_series(self, x: pd.DataFrame, history_days: int = 20, n_slots: int = 32, cap: float = 25.0, **_: Any) -> pd.DataFrame:
        hd = int(history_days)
        ns = int(n_slots)
        cp = float(cap)
        if hd < 3:
            raise ValueError("intraday_profile_surprise_energy requires history_days >= 3")
        if ns < 4:
            raise ValueError("intraday_profile_surprise_energy requires n_slots >= 4")
        if cp <= 0.0:
            raise ValueError("intraday_profile_surprise_energy requires cap > 0")
        return _profile_panel(x, hd, ns, cp, phase=False, max_shift=0)


@register_operator(
    name="intraday_profile_phase_shift",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_profile_phase_shift",
    source="advanced_intraday",
    status="experimental",
)
class IntradayProfilePhaseShift(SeriesOperator):
    """日内 profile 相位偏移（今天成交高峰是否提前/延后）。

    在 ``[-max_shift, +max_shift]`` slot 偏移内最大化当前 profile 与历史中位
    profile 的相关，输出最佳偏移 ``k/n_slots``。正 = 高峰提前到上午；
    负 = 高峰延后。对 volume/amount profile 很有意思。P2 / Research。
    """

    metadata = _metadata(
        "intraday_profile_phase_shift",
        "日内 profile 最佳相位偏移 k/n_slots（高峰提前/延后）。",
        ["x", "history_days", "max_shift", "n_slots"],
        unit="phase",
        cost=6,
    )

    def _calculate_series(self, x: pd.DataFrame, history_days: int = 20, max_shift: int = 4, n_slots: int = 32, **_: Any) -> pd.DataFrame:
        hd = int(history_days)
        ns = int(n_slots)
        ms = int(max_shift)
        if hd < 3:
            raise ValueError("intraday_profile_phase_shift requires history_days >= 3")
        if ns < 4:
            raise ValueError("intraday_profile_phase_shift requires n_slots >= 4")
        if ms < 1:
            raise ValueError("intraday_profile_phase_shift requires max_shift >= 1")
        return _profile_panel(x, hd, ns, 1.0, phase=True, max_shift=ms)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS)
        | {
            "intraday_wasserstein_pair_distance",
            "intraday_barrier_approach_acceleration",
            "intraday_subsampled_rv_dispersion",
            "intraday_volatility_signature_slope",
            "intraday_profile_surprise_energy",
        }
    )
    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS)
        | {
            "intraday_quantile_curve_pca_score",
            "intraday_quantile_curve_pca_residual",
            "intraday_realized_power_variation",
            "intraday_profile_phase_shift",
        }
    )
    for _canon in (
        "intraday_subsampled_rv_dispersion",
        "intraday_volatility_signature_slope",
        "intraday_realized_power_variation",
        "intraday_profile_surprise_energy",
        "intraday_profile_phase_shift",
    ):
        register_polars_udf(_canon)


_register_surface()
