# -*- coding: utf-8 -*-
"""Extreme-value tail shape and quantile relationship operators (2026-08 V3).

* ``ts_hill_tail_index``        — Hill estimator of the tail index ξ (shape, not
  size): how thick the extreme tail is (P1).  Upper/lower via ``side``, both
  defined as explicit peaks-over-threshold exceedances (exact counterparts,
  review #38); a raw nonstationary price level is flagged by a data-quality
  warning, never a hard semantic gate — the typed FieldSpec ``input_units``
  declaration is the authoritative contract (review #39, demoted in R11).
* ``ts_quantile_regression_beta`` — exact quantile-regression slope β_q via a
  linear program (P2 research): the marginal relationship of ``x`` when the
  stock is in its own q-th return state.  Carries ``unit(y)/unit(x)`` (review
  #40), not a uniform ratio.

Deterministic (fixed threshold fraction, LP with HiGHS), prefix-causal.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12

# ISSUE 4 / R11: a numeric price-level heuristic is at most a data-quality
# warning, never a hard semantic gate.  This set deduplicates the warning so a
# given price-level-looking input warns once per process (module-level flag).
_WARNED_PRICE_LEVEL: set[Any] = set()


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    # R4-87: the lower tail is defined against a low threshold (``threshold - x``)
    # so positive price/valuation series have a well-defined lower tail; the input
    # is nevertheless declared as a signed series (returns / centred residual /
    # signed signal) where a tail shape is meaningful.
    return OperatorMetadata(
        name=name,
        category="extreme_tail",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "extreme_tail", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:price_volume",
            f"unit:{unit}", f"cost:{cost}",
        ],
        input_units={"x": "signed_return_or_centred_residual_or_signed_signal"},
    )


def _column_map(xv: np.ndarray, fn) -> np.ndarray:
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        out[:, c] = fn(xv[:, c])
    return out


# ---------------------------------------------------------------------------
# ts_hill_tail_index
# ---------------------------------------------------------------------------

def _reject_price_level(series: np.ndarray) -> None:
    """Data-quality warning for a raw nonstationary price level input (R11).

    ``ts_hill_tail_index`` is a *shape* estimator on exceedance magnitudes;
    feeding it a raw price level (strictly positive and nonstationary, so the
    level itself carries no tail-shape meaning) is a caller bug.  Review #39
    originally failed closed with a hard ValueError, but whether the input is a
    RawPrice / Return / Residual / PositiveMagnitude is a decision owned by the
    typed FieldSpec / IR layer — the operator's own metadata already declares
    ``input_units={"x": "signed_return_or_centred_residual_or_signed_signal"}``
    as the authoritative typed contract.  A numeric heuristic is at most a
    data-quality warning, never a hard semantic gate (demoted in R11): we warn
    once per distinct price-level-looking input (module-level dedup flag) and
    return normally so the estimator runs.
    """
    vals = series[np.isfinite(series)]
    if vals.size < 8:
        return  # too short to judge; the kernel's min-tail-count fails closed
    if not np.all(vals > 0.0):
        return  # signed return / centred residual / signed signal -> allowed
    sd_level = float(np.std(vals))
    sd_diff = float(np.std(np.diff(vals)))
    # sd_diff ~ 0 -> a deterministic smooth trend (pure price level);
    # sd_level / sd_diff large -> random-walk price level.  Both are DQ flags.
    ratio = sd_level / max(sd_diff, _EPS)
    if ratio > 5.0:
        # Dedupe by a coarse fingerprint of the input so repeated calls with the
        # same price-level series warn once, not on every window/call.
        stride = max(1, vals.size // 16)
        key = (vals.size, tuple(np.round(vals[::stride] * 1e3).tolist()))
        if key not in _WARNED_PRICE_LEVEL:
            _WARNED_PRICE_LEVEL.add(key)
            warnings.warn(
                "ts_hill_tail_index received a series that looks like a raw "
                "nonstationary price level (sd(level)/sd(diff) > 5).  The typed "
                "FieldSpec input_units declaration is the authoritative input "
                "contract; treat this as a data-quality warning, not a hard "
                "semantic gate (review #39, demoted in R11).",
                RuntimeWarning,
                stacklevel=2,
            )


def _pot_exceedances(valid: np.ndarray, side: str, frac: float) -> np.ndarray:
    """Explicit peaks-over-threshold exceedance magnitudes (review #38).

    Upper and lower tails are exact counterparts:
    * upper exceedance ``x - u`` for ``x > u`` with ``u = Q(valid, 1-frac)``;
    * lower exceedance ``u - x`` for ``x < u`` with ``u = Q(valid, frac)``.
    """
    if valid.size == 0:
        return valid
    if side == "upper":
        u = float(np.quantile(valid, 1.0 - frac))
        mags = valid[valid > u] - u
    else:
        u = float(np.quantile(valid, frac))
        mags = u - valid[valid < u]
    return mags[mags > 0.0]


def _hill_series(series: np.ndarray, window: int, side: str, tail_fraction: float, min_tail_count: int) -> np.ndarray:
    """TRUE Hill estimator with a SINGLE tail selection (review #38 / R11).

    One threshold ``u`` at the ``tail_fraction`` quantile and ALL exceedances
    beyond it feed the shape estimate directly — the tail fraction is never
    applied a second time (the pre-R11 code POT-thresholded at ``frac`` and then
    sub-sampled ``k = floor(mags.size * frac)`` again, which with
    window=120/frac=0.2/min_tail_count=10 produced ``k=4 < 10`` and long NaN
    stretches under default parameters).

    * upper: ``u = Q(valid, 1-frac)``, exceedances ``x_i > u``;
    * lower: mirror of the upper tail on ``y = -x``, i.e. ``u = Q(valid, frac)``
      and exceedances ``x_i < u`` — ``ξ_lower(x) == ξ_upper(-x)`` exactly.

    Hill shape on the exceedance values scaled by the (positive) threshold:
    ``ξ = (1/k) Σ log(x_i / u)``, ``k = len(exceedances)``.  A threshold that is
    not finite or not strictly positive has no meaningful log-ratio and yields
    NaN for that window (the input units are returns / centred residual / signed
    signal / positive magnitude — see the authoritative ``input_units``).
    ``min_tail_count`` gates the minimum number of exceedances.
    """
    n = series.shape[0]
    w = max(2, int(window))
    # Invalid tail_fraction must fail loudly, never silently clip to a legal
    # value — silent clipping turns different ASTs into the same parameter and
    # corrupts the search space (P1-16).
    frac = float(tail_fraction)
    if not (0.0 < frac <= 0.5):
        raise ValueError("tail_fraction must satisfy 0 < tail_fraction <= 0.5")
    mtc = max(3, int(min_tail_count))
    _reject_price_level(series)
    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - w + 1)
        chunk = series[lo : t + 1]
        valid = chunk[np.isfinite(chunk)]
        if valid.size < mtc:
            continue
        # ONE kernel for both sides: upper operates on x, lower on the mirrored
        # series z = -x.  HillLower(x) == HillUpper(-x) exactly — there is no
        # separate lower-tail code path to drift.  The log-ratio needs every
        # exceedance to share the sign of the threshold and a non-zero
        # threshold, which holds on BOTH mirrors:
        #   * centred signed series, lower: z = -x, u = Q(-x, 1-frac) > 0,
        #     exceedances z > u > 0  (the old ``u <= 0`` guard spuriously NaN'd
        #     this case because it tested Q(x, frac) < 0 instead);
        #   * positive-magnitude series, lower: z = -mag < 0, u = -Q(mag, frac)
        #     < 0, exceedances z > u are likewise < 0 — same sign, ratio
        #     positive, identical to the direct x-space computation.
        # Only an exact zero / non-finite threshold is degenerate (division by
        # zero / no defined ratio); a mixed-sign exceedance set falls out as NaN
        # through the ``np.isfinite(xi)`` guard below.
        z = valid if side == "upper" else -valid
        u = float(np.quantile(z, 1.0 - frac))
        exc = z[z > u]
        if not np.isfinite(u) or u == 0.0:
            continue  # log-ratio is undefined for a zero / non-finite threshold
        if exc.size < mtc:
            continue
        xi = float(np.mean(np.log(exc / u)))
        if np.isfinite(xi):
            out[t] = xi
    return out


@register_operator(
    name="ts_hill_tail_index",
    category="extreme_tail",
    business_category="extreme_tail",
    canonical="ts_hill_tail_index",
    source="extreme_tail",
)
class TsHillTailIndex(SeriesOperator):
    """Hill 尾部指数 ξ（POT 上/下超阈值形状，非大小）。

    显式 peaks-over-threshold 单次尾部选择（review #38 / R11）：上尾阈值
    ``u = Q(1-frac)`` 取全部 ``x > u`` 的超阈值样本，下尾是镜像
    ``u = Q(frac)`` 取 ``x < u``（== 上尾作用于 ``-x``）。单次选择后
    ``k = len(exceedances)``，``ξ = (1/k) Σ log(x_i/u)`` —— 绝不再按 frac
    二次子抽样。ξ 越大尾部越厚。``min_tail_count`` 内样本不足则 NaN。
    输入必须是 return / residual / positive magnitude / tail loss；原始非平稳
    价格水平按 R11 只发 data-quality 警告、不再硬拒（类型契约由 typed
    FieldSpec ``input_units`` 声明为准）。P1。
    """

    metadata = _metadata(
        "ts_hill_tail_index",
        "Hill 尾部指数 ξ（POT 上/下超阈值形状，拒收原始价格水平）。",
        ["x", "window", "side", "tail_fraction", "min_tail_count"],
        unit="ratio",
        cost=5,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 120,
        side: str = "upper",
        tail_fraction: float = 0.2,
        min_tail_count: int = 10,
        **_: Any,
    ) -> pd.DataFrame:
        side_k = str(side).lower()
        if side_k not in {"upper", "lower"}:
            raise ValueError("ts_hill_tail_index requires side in {'upper','lower'}")
        return _frame_like_result(
            x,
            _column_map(
                x.to_numpy(dtype=float),
                lambda s: _hill_series(s, window, side_k, tail_fraction, min_tail_count),
            ),
        )


def _frame_like_result(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return frame_like(template, values)


# ---------------------------------------------------------------------------
# ts_quantile_regression_beta (P2 research)
# ---------------------------------------------------------------------------

def _quantile_beta(x: np.ndarray, y: np.ndarray, q: float) -> float:
    n = x.shape[0]
    if n < 3:
        return np.nan
    xc = x - x.mean()
    if float(np.sum(xc * xc)) <= _EPS:
        return np.nan
    from scipy.optimize import linprog

    # variables: [alpha, beta, u_1..u_n, v_1..v_n];  alpha + beta x_i + u_i - v_i = y_i
    dim = 2 + 2 * n
    c = np.zeros(dim)
    c[2 : 2 + n] = q
    c[2 + n :] = 1.0 - q
    A_eq = np.zeros((n, dim))
    for i in range(n):
        A_eq[i, 0] = 1.0
        A_eq[i, 1] = x[i]
        A_eq[i, 2 + i] = 1.0
        A_eq[i, 2 + n + i] = -1.0
    bounds = [(None, None), (None, None)] + [(0.0, None)] * (2 * n)
    try:
        result = linprog(c, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
    except Exception:
        return np.nan
    if result is None or result.status != 0 or result.x is None:
        return np.nan
    beta = float(result.x[1])
    if not np.isfinite(beta):
        return np.nan
    return beta


def _quantile_beta_series(y: np.ndarray, x: np.ndarray, window: int, q: float) -> np.ndarray:
    rows, cols = y.shape
    w = max(2, int(window))
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for t in range(rows):
            lo = max(0, t - w + 1)
            if t - lo + 1 < w:
                continue
            xw = x[lo : t + 1, c]
            yw = y[lo : t + 1, c]
            finite = np.isfinite(xw) & np.isfinite(yw)
            if int(finite.sum()) < 3:
                continue
            out[t, c] = _quantile_beta(xw[finite], yw[finite], q)
    return out


@register_operator(
    name="ts_quantile_regression_beta",
    category="extreme_tail",
    business_category="extreme_tail",
    canonical="ts_quantile_regression_beta",
    source="extreme_tail",
    status="experimental",
)
class TsQuantileRegressionBeta(SeriesOperator):
    """精确分位数回归斜率 β_q（LP 求解）。

    最小化 ``Σ ρ_q(y - α - βx)``（``ρ_q(u)=u(q-I(u<0))``），exact 线性规划
    (HiGHS)，不用不稳定的 IRLS 近似。q=0.1 回答"股票自己处于最差收益状态时 x
    对它的边际关系"。P2 / Research。
    """

    metadata = _metadata(
        "ts_quantile_regression_beta",
        "分位数回归斜率 β_q（精确 LP，单位 unit(y)/unit(x)）。",
        ["y", "x", "window", "quantile"],
        # A regression slope carries unit(y)/unit(x), NOT a uniform ratio
        # (review #40) — mirroring the ts_expectile_beta convention.
        unit="unit(y)/unit(x)",
        cost=8,
    )

    def _calculate_series(
        self, y: pd.DataFrame, x: pd.DataFrame, window: int = 120, quantile: float = 0.5, **_: Any
    ) -> pd.DataFrame:
        q = float(quantile)
        if not 0.0 < q < 1.0:
            raise ValueError("ts_quantile_regression_beta requires quantile in (0, 1)")
        return _frame_like_result(
            y,
            _quantile_beta_series(y.to_numpy(dtype=float), x.to_numpy(dtype=float), window, q),
        )


# ---------------------------------------------------------------------------
# ts_extremal_index (P1 deepening)
# ---------------------------------------------------------------------------

def _extremal_index_series(
    series: np.ndarray, window: int, side: str, q: float, min_exceed: int, run_length: int = 1
) -> np.ndarray:
    """Leadbetter extremal index θ via the runs estimator.

    Over the trailing window count exceedances above the empirical q-quantile
    and the number of *runs* of exceedances (each run = one cluster).
    ``θ = clusters / exceedances``: ≈1 = extremes arrive as isolated single
    observations (Poisson-like); →0 = extremes cluster strongly (regime-like).

    Cluster semantics (R11):
    * ``run_length`` (default 1): an exceedance only continues the current
      cluster when the gap of non-exceedances since the last exceedance is
      ``< run_length``; otherwise it starts a new cluster.
    * a NaN (unknown-state) gap always BREAKS the current run — an exceedance on
      the far side of a missing bar is never joined to one on the near side,
      regardless of ``run_length`` (no compression across unknown gaps).
    """
    n = series.shape[0]
    w = max(2, int(window))
    quant = float(q)
    if not 0.0 < quant < 1.0:
        raise ValueError("q must be in (0, 1)")
    rl = int(run_length)
    if rl < 1:
        raise ValueError("run_length must be >= 1")
    mex = max(2, int(min_exceed))
    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - w + 1)
        chunk = series[lo : t + 1]
        valid = chunk[np.isfinite(chunk)]
        if valid.size < 3:
            continue
        if side == "upper":
            thr = float(np.quantile(valid, quant))
            is_exceed = lambda v: v > thr  # noqa: E731
        else:
            # R4-87: lower tail = strictly below the (1-quant) quantile, so a
            # positive price/valuation series keeps a well-defined lower tail.
            thr = float(np.quantile(valid, 1.0 - quant))
            is_exceed = lambda v: v < thr  # noqa: E731
        # R11: gap-based clustering with an explicit run_length, and a NaN bar
        # BREAKS the current run (pre-R11 carried ``prev`` across NaN, so
        # ``Extreme, NaN, NaN, NaN, Extreme`` was miscounted as ONE cluster).
        count = 0
        clusters = 0
        seen_exceed = False
        gap = 0  # consecutive non-exceedances since the last exceedance
        for v in chunk:
            if not np.isfinite(v):
                # Unknown state: force a break — set the gap to run_length so
                # the next exceedance necessarily opens a new cluster.
                gap = rl
                continue
            e = bool(is_exceed(v))
            if e:
                count += 1
                if not seen_exceed or gap >= rl:
                    clusters += 1
                seen_exceed = True
                gap = 0
            else:
                gap += 1
        if count < mex:
            continue
        out[t] = float(clusters / count)
    return out


@register_operator(
    name="ts_extremal_index",
    category="extreme_tail",
    business_category="extreme_tail",
    canonical="ts_extremal_index",
    source="extreme_tail",
)
class TsExtremalIndex(SeriesOperator):
    """Leadbetter 极值指数 θ（runs 估计：簇数 / 超阈次数）。

    ≈1 = 极端观测以孤立单点到达（类 Poisson，可独立处理）；→0 = 极端高度成簇
    （regime 型，波动聚集）。``run_length``（默认 1）= 距上次超阈的连续
    non-extreme 计数必须小于该值才续接同一簇；缺失 bar（NaN）永远断开当前簇，
    跨越未知区间的两个极端绝不会被并入同一簇（R11 修正，不跨 gap 压缩）。
    与 ``ts_extreme_cluster_ratio``（相邻极端对数/极端数）数学不同：连续 3 个
    极端前者 2/3、这里 1/3。PIT 安全、确定性。
    """

    metadata = _metadata(
        "ts_extremal_index",
        "极值指数 θ = 簇数/超阈次数（[0,1]，低=成簇）。",
        ["x", "window", "side", "q", "min_exceed", "run_length"],
        unit="ratio",
        cost=4,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 120,
        side: str = "upper",
        q: float = 0.9,
        min_exceed: int = 3,
        run_length: int = 1,
        **_: Any,
    ) -> pd.DataFrame:
        side_k = str(side).lower()
        if side_k not in {"upper", "lower"}:
            raise ValueError("ts_extremal_index requires side in {'upper','lower'}")
        return _frame_like_result(
            x,
            _column_map(
                x.to_numpy(dtype=float),
                lambda s: _extremal_index_series(s, window, side_k, q, min_exceed, run_length),
            ),
        )


# ---------------------------------------------------------------------------
# ts_mean_excess_slope (P2 deepening)
# ---------------------------------------------------------------------------

def _mean_excess_slope_series(
    series: np.ndarray, window: int, side: str, min_tail_count: int
) -> np.ndarray:
    """OLS slope of the mean-excess plot over the upper (or mirrored lower) tail.

    For thresholds at quantiles ``p ∈ {0.55..0.95}``, mean excess
    ``ME(u) = mean(y - u | y > u)``; the slope of ME vs u is dimensionless.
    For a GPD tail it equals ``ξ/(1-ξ)`` (negative = bounded light tail,
    positive = heavy tail, ~0 = exponential).  The lower tail is computed on the
    mirrored series ``y = -x`` reusing the exact upper-tail computation, so
    ``ME_lower(x) == ME_upper(-x)`` and the slope sign is interpreted
    identically on both sides (positive = heavy tail).
    """
    n = series.shape[0]
    w = max(2, int(window))
    mtc = max(2, int(min_tail_count))
    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - w + 1)
        chunk = series[lo : t + 1]
        valid = chunk[np.isfinite(chunk)]
        if valid.size < mtc + 3:
            continue
        # R11: the lower tail is the exact mirror of the upper tail on ``y = -x``
        # (ME_lower(x) == ME_upper(-x)); reusing one code path keeps the slope
        # direction consistent with the documented interpretation on both sides.
        work = -valid if side == "lower" else valid
        us: list[float] = []
        mes: list[float] = []
        for p in np.linspace(0.55, 0.95, 9):
            u = float(np.quantile(work, p))
            exc = work[work > u] - u
            if exc.size < mtc:
                continue
            us.append(u)
            mes.append(float(exc.mean()))
        if len(us) < 3:
            continue
        ua = np.asarray(us)
        ma = np.asarray(mes)
        denom = float(np.sum((ua - ua.mean()) ** 2))
        if denom <= _EPS:
            continue
        slope = float(np.sum((ua - ua.mean()) * (ma - ma.mean())) / denom)
        if np.isfinite(slope):
            out[t] = slope
    return out


@register_operator(
    name="ts_mean_excess_slope",
    category="extreme_tail",
    business_category="extreme_tail",
    canonical="ts_mean_excess_slope",
    source="extreme_tail",
)
class TsMeanExcessSlope(SeriesOperator):
    """mean-excess 图斜率（尾部厚薄的尺度无关指标）。

    对尾部分位数阈值网格 ``p∈[0.55,0.95]`` 计算 ME(u)，对 u 做 OLS 斜率。
    GPD 尾部下等于 ``ξ/(1-ξ)``：负 = 有界轻尾，正 = 重尾，≈0 = 指数尾。比单独
    Hill 指数在阈值选择上更稳。PIT 安全、确定性。
    """

    metadata = _metadata(
        "ts_mean_excess_slope",
        "mean-excess 图斜率（≈ξ/(1-ξ)，负=轻尾正=重尾）。",
        ["x", "window", "side", "min_tail_count"],
        unit="ratio",
        cost=5,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 120,
        side: str = "upper",
        min_tail_count: int = 5,
        **_: Any,
    ) -> pd.DataFrame:
        side_k = str(side).lower()
        if side_k not in {"upper", "lower"}:
            raise ValueError("ts_mean_excess_slope requires side in {'upper','lower'}")
        return _frame_like_result(
            x,
            _column_map(
                x.to_numpy(dtype=float),
                lambda s: _mean_excess_slope_series(s, window, side_k, min_tail_count),
            ),
        )


# ---------------------------------------------------------------------------
# ts_gpd_shape_pwm (P2 deepening)
# ---------------------------------------------------------------------------

def _gpd_shape_pwm_series(
    series: np.ndarray, window: int, side: str, tail_fraction: float, min_tail_count: int
) -> np.ndarray:
    """GPD shape ξ via Probability-Weighted Moments (Hosking & Wallis 1987).

    With exceedances ``x_(1) ≤ ... ≤ x_(n)`` above threshold ``u``:
    ``b0 = mean(x)``, ``b1 = Σ ((i-1)/(n-1)) x_(i) / n`` and
    ``ξ̂ = 2 - b0 / (2 b1 - b0)`` (from τ2 = L2/L1 = 1/(2-ξ)).  Deterministic,
    no optimizer; fail-closed when the denominator degenerates.
    """
    n = series.shape[0]
    w = max(2, int(window))
    frac = float(tail_fraction)
    if not (0.0 < frac <= 0.5):
        raise ValueError("tail_fraction must satisfy 0 < tail_fraction <= 0.5")
    mtc = max(3, int(min_tail_count))
    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - w + 1)
        chunk = series[lo : t + 1]
        valid = chunk[np.isfinite(chunk)]
        if valid.size < mtc + 2:
            continue
        if side == "upper":
            u = float(np.quantile(valid, 1.0 - frac))
            exc = valid[valid > u] - u
        else:
            # R4-87: lower tail = excess ``u - x`` below the frac-quantile
            # threshold; defined for positive price/valuation series too.
            u = float(np.quantile(valid, frac))
            exc = u - valid[valid < u]
        if exc.size < mtc:
            continue
        exc = np.sort(exc)
        m = exc.size
        b0 = float(exc.mean())
        weights = np.arange(m, dtype=float) / max(m - 1, 1)
        b1 = float(np.sum(weights * exc) / m)
        denom = 2.0 * b1 - b0
        if abs(denom) <= _EPS:
            continue
        xi = 2.0 - b0 / denom
        if np.isfinite(xi):
            out[t] = xi
    return out


@register_operator(
    name="ts_gpd_shape_pwm",
    category="extreme_tail",
    business_category="extreme_tail",
    canonical="ts_gpd_shape_pwm",
    source="extreme_tail",
)
class TsGpdShapePwm(SeriesOperator):
    """GPD 形状参数 ξ（PWM/L-moment 估计，闭式无优化器）。

    ``ξ̂ = 2 - b0/(2 b1 - b0)``（Hosking-Wallis 1987）。ξ>0 = Fréchet 重尾，
    ξ=0 = Gumbel 指数尾，ξ<0 = 有界 Weibull 尾。与 ``ts_hill_tail_index`` 估计
    同一形状但用完全不同的估计量，作为交叉验证。PIT 安全、确定性。
    """

    metadata = _metadata(
        "ts_gpd_shape_pwm",
        "GPD 形状参数 ξ（PWM 闭式估计）。",
        ["x", "window", "side", "tail_fraction", "min_tail_count"],
        unit="ratio",
        cost=5,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 120,
        side: str = "upper",
        tail_fraction: float = 0.2,
        min_tail_count: int = 10,
        **_: Any,
    ) -> pd.DataFrame:
        side_k = str(side).lower()
        if side_k not in {"upper", "lower"}:
            raise ValueError("ts_gpd_shape_pwm requires side in {'upper','lower'}")
        return _frame_like_result(
            x,
            _column_map(
                x.to_numpy(dtype=float),
                lambda s: _gpd_shape_pwm_series(s, window, side_k, tail_fraction, min_tail_count),
            ),
        )


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({"ts_hill_tail_index", "ts_extremal_index", "ts_mean_excess_slope", "ts_gpd_shape_pwm"})
    _surface.extend_research_only({"ts_quantile_regression_beta"})


_register_surface()
