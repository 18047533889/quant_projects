# -*- coding: utf-8 -*-
"""Extreme-value tail shape and quantile relationship operators (2026-08 V3).

* ``ts_hill_tail_index``        — Hill estimator of the tail index ξ (shape, not
  size): how thick the extreme tail is (P1).  Upper/lower via ``side``.
* ``ts_quantile_regression_beta`` — exact quantile-regression slope β_q via a
  linear program (P2 research): the marginal relationship of ``x`` when the
  stock is in its own q-th return state.

Deterministic (fixed threshold fraction, LP with HiGHS), prefix-causal.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12


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

def _lower_tail_magnitude(valid: np.ndarray) -> np.ndarray:
    """R4-87 lower-tail magnitudes ``u - x > 0`` below a reference threshold.

    ``u`` is the median of the finite window values, so the lower-tail sample is
    the bottom half of the window — the sample-size mirror image of the upper
    tail's ``x > 0`` filter for a centred return series (median ≈ 0, so
    ``u - x ≈ -x``).  Using ``threshold - x`` instead of ``-x`` keeps the lower
    tail well-defined for *positive* price/valuation series, where ``-x`` is
    always negative and ``y > 0`` deleted the whole lower tail.
    """
    if valid.size == 0:
        return valid
    u = float(np.quantile(valid, 0.5))
    y = u - valid
    return y[y > 0.0]


def _hill_series(series: np.ndarray, window: int, side: str, tail_fraction: float, min_tail_count: int) -> np.ndarray:
    n = series.shape[0]
    w = max(2, int(window))
    # Invalid tail_fraction must fail loudly, never silently clip to a legal
    # value — silent clipping turns different ASTs into the same parameter and
    # corrupts the search space (P1-16).
    frac = float(tail_fraction)
    if not (0.0 < frac <= 0.5):
        raise ValueError("tail_fraction must satisfy 0 < tail_fraction <= 0.5")
    mtc = max(3, int(min_tail_count))
    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - w + 1)
        chunk = series[lo : t + 1]
        valid = chunk[np.isfinite(chunk)]
        if valid.size < mtc:
            continue
        if side == "upper":
            y = valid
        else:
            y = _lower_tail_magnitude(valid)
        y = y[y > 0.0]
        if y.size < mtc:
            continue
        k = int(np.floor(y.size * frac))
        if k < mtc:
            continue
        ys = np.sort(y)
        threshold = ys[y.size - k - 1]
        if threshold <= 0.0 or not np.isfinite(threshold):
            continue
        top = ys[y.size - k :]
        xi = float(np.mean(np.log(top / threshold)))
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
    """Hill 尾部指数 ξ（上/下尾厚度形状，非大小）。

    上尾对窗口内 X>0 排序;下尾对低阈值超量 ``u - x > 0``(u 为 frac 分位数,
    R4-87)——正价格/估值序列也有定义良好的下尾,而非被 ``-x`` 全删。取
    ``k=floor(n*frac)`` 个大值:``ξ = (1/k) Σ log(Y_{n-j+1}/Y_{n-k})``。ξ 越大
    尾部越厚。``min_tail_count`` 内样本不足则 NaN。与 ES/partial moment(尾部
    大小)互补。P1。
    """

    metadata = _metadata(
        "ts_hill_tail_index",
        "Hill 尾部指数 ξ（上/下尾厚度形状）。",
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
        "分位数回归斜率 β_q（精确 LP）。",
        ["y", "x", "window", "quantile"],
        unit="ratio",
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
    series: np.ndarray, window: int, side: str, q: float, min_exceed: int
) -> np.ndarray:
    """Leadbetter extremal index θ via the runs estimator.

    Over the trailing window count exceedances above the empirical q-quantile
    and the number of *runs* of consecutive exceedances (each run = one cluster).
    ``θ = clusters / exceedances``: ≈1 = extremes arrive as isolated single
    observations (Poisson-like); →0 = extremes cluster strongly (regime-like).
    """
    n = series.shape[0]
    w = max(2, int(window))
    quant = float(q)
    if not 0.0 < quant < 1.0:
        raise ValueError("q must be in (0, 1)")
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
        # R4-88: missing bars are censored, NOT counted as non-exceedances.
        # A missing bar between two extreme bars must not split one cluster into
        # two: ``prev`` is carried across NaN so an extreme after a gap continues
        # the current run.
        count = 0
        clusters = 0
        prev = False
        for v in chunk:
            if not np.isfinite(v):
                continue  # censor: leave ``prev`` unchanged
            e = bool(is_exceed(v))
            if e:
                count += 1
                if not prev:
                    clusters += 1
            prev = e
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
    （regime 型，波动聚集）。缺失 bar 按 R4-88 截删（censor）：缺失不当作
    non-extreme，故两个极端之间的一个缺失不会把同一簇拆成两簇；已确认的
    non-extreme 仍会断开运行。与 ``ts_extreme_cluster_ratio``（相邻极端对数/极
    端数）数学不同：连续 3 个极端前者 2/3、这里 1/3。PIT 安全、确定性。
    """

    metadata = _metadata(
        "ts_extremal_index",
        "极值指数 θ = 簇数/超阈次数（[0,1]，低=成簇）。",
        ["x", "window", "side", "q", "min_exceed"],
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
        **_: Any,
    ) -> pd.DataFrame:
        side_k = str(side).lower()
        if side_k not in {"upper", "lower"}:
            raise ValueError("ts_extremal_index requires side in {'upper','lower'}")
        return _frame_like_result(
            x,
            _column_map(
                x.to_numpy(dtype=float),
                lambda s: _extremal_index_series(s, window, side_k, q, min_exceed),
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
    positive = heavy tail, ~0 = exponential).
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
        us: list[float] = []
        mes: list[float] = []
        for p in np.linspace(0.55, 0.95, 9):
            if side == "upper":
                u = float(np.quantile(valid, p))
                exc = valid[valid > u] - u
            else:
                # R4-87: lower-tail thresholds are the (1-p) quantiles and the
                # exceedances are ``u - x``, keeping the lower tail defined for
                # positive price/valuation series.
                u = float(np.quantile(valid, 1.0 - p))
                exc = u - valid[valid < u]
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

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS)
        | {"ts_hill_tail_index", "ts_extremal_index", "ts_mean_excess_slope", "ts_gpd_shape_pwm"}
    )
    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS) | {"ts_quantile_regression_beta"}
    )


_register_surface()
