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

from factor_engine.cleaned_operators.base import OperatorMetadata, ParamRole, ParamSpec, SeriesOperator, register_operator
from factor_engine.cleaned_operators.parameter_validation import strict_finite_scalar, strict_integer
from factor_engine.cleaned_operators.rolling_pack import frame_like
from factor_engine.cleaned_operators.ts_model._rolling_core import pinball_quantile_fit

_EPS = 1e-12

# ISSUE 4 / R11: a numeric price-level heuristic is at most a data-quality
# warning, never a hard semantic gate.  This set deduplicates the warning so a
# given price-level-looking input warns once per process (module-level flag).
_WARNED_PRICE_LEVEL: set[Any] = set()


def _metadata(
    name: str, description: str, params: list[str], *, unit: str, cost: int,
    param_specs: dict[str, ParamSpec], panel_params: tuple[str, ...],
) -> OperatorMetadata:
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
        param_specs=param_specs,
        panel_params=panel_params,
        scalar_params=tuple(p for p in params if p not in panel_params),
    )


def _column_map(xv: np.ndarray, fn) -> np.ndarray:
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        out[:, c] = fn(xv[:, c])
    return out


# ---------------------------------------------------------------------------
# R62: shared trailing-window matrix helpers
# ---------------------------------------------------------------------------
# The tail kernels below were per-row Python loops (``for t in range(n)``:
# take the trailing window, drop NaN, run a scalar estimator).  They are now a
# single ``(T, w)`` trailing-aligned window matrix plus axis=1 numpy kernels:
# one batch covers every window, no Python loop over time survives.

def _winmat(x: np.ndarray, w: int) -> np.ndarray:
    """``(T, w)`` trailing-aligned window matrix (NaN before the series start).

    ``W[t, j]`` holds ``x[t - w + 1 + j]`` and the slots that would fall before
    row 0 are NaN, so ``W[t]`` is exactly the authority's trailing chunk
    ``x[max(0, t - w + 1) : t + 1]`` right-aligned.
    """
    n = int(x.shape[0])
    if n == 0:
        return np.empty((0, w), dtype=float)
    src = np.arange(n)[:, None] - (w - 1) + np.arange(w)[None, :]
    return np.where(src >= 0, x[np.clip(src, 0, n - 1)], np.nan)


def _linear_quantile(sorted_windows: np.ndarray, counts: np.ndarray, p: float) -> np.ndarray:
    """Row-wise ``np.quantile(row[:counts], p, method='linear')``.

    ``sorted_windows`` is the ascending-per-row window matrix (padding NaN in
    the last ``w - counts`` slots), which is never read.  The arithmetic
    (virtual index / floor / lerp direction) mirrors numpy's own ``_quantile``
    so the threshold is the same float the authority computes.
    """
    n = counts.astype(np.float64)
    # numpy 'linear': virtual_index = (n - 1) * q  (see numpy _QuantileMethods)
    virtual = (n - 1.0) * p
    top = np.maximum(counts - 1, 0)
    prev = np.floor(virtual)
    nxt = prev + 1.0
    above = virtual >= (counts - 1)
    below = virtual < 0
    gamma = virtual - prev
    i0 = np.where(above, top, np.where(below, 0, prev)).astype(np.intp)
    i1 = np.where(above, top, np.where(below, 0, nxt)).astype(np.intp)
    a = np.take_along_axis(sorted_windows, np.clip(i0, 0, None)[:, None], axis=1)[:, 0]
    b = np.take_along_axis(sorted_windows, np.clip(i1, 0, None)[:, None], axis=1)[:, 0]
    diff = b - a
    return np.where(gamma >= 0.5, b - diff * (1 - gamma), a + diff * gamma)



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

    R62: one ``(T, w)`` trailing-window batch replaces the per-row loop; the
    estimate is scale-invariant, so the 1e±300 panels need no rescaling.
    """
    n = series.shape[0]
    w = strict_integer(window, "window", minimum=2)
    # Invalid tail_fraction must fail loudly, never silently clip to a legal
    # value — silent clipping turns different ASTs into the same parameter and
    # corrupts the search space (P1-16).
    frac = strict_finite_scalar(tail_fraction, "tail_fraction")
    if not (0.0 < frac <= 0.5):
        raise ValueError("tail_fraction must satisfy 0 < tail_fraction <= 0.5")
    mtc = strict_integer(min_tail_count, "min_tail_count", minimum=3)
    _reject_price_level(series)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    # ONE kernel for both sides: upper operates on x, lower on the mirrored
    # series z = -x.  HillLower(x) == HillUpper(-x) exactly — there is no
    # separate lower-tail code path to drift.  After mirroring, the same
    # classic-Hill domain applies to both sides: a strictly-positive
    # threshold and strictly-positive observations above it.
    # Classic Hill is defined only for a strictly-positive threshold and
    # strictly-positive exceedance values.  A negative upper-tail threshold
    # (for example an all-negative window) is not rescued by a positive
    # ratio: its resulting negative estimate is outside the estimator's
    # domain.  The lower side uses the same rule after mirroring ``z=-x``.
    # R26-060..062: the LOWER tail is only a valid Hill problem on SIGNED
    # data (returns / residuals / downside losses), whose negative values
    # mirror onto positive, unbounded-above loss magnitudes.  A window with
    # NO negative values is a strictly-positive level / magnitude whose left
    # tail is BOUNDED below by 0 — classic Hill (``z = -x`` mirror) does not
    # apply to it and produces a misleading negative ξ.  Such a window fails
    # closed to NaN (explicitly unsupported lower tail for positive levels).
    W = _winmat(series, w)
    fin = np.isfinite(W)
    counts = fin.sum(axis=1)
    ok = counts >= mtc
    if side == "lower":
        min_valid = np.min(np.where(fin, W, np.inf), axis=1)
        ok = ok & ~(min_valid >= 0.0)
    Z = -W if side == "lower" else W
    # non-finite slots (padding, NaN, +/-inf) are excluded BEFORE the sort so the
    # ascending layout is guaranteed: finite values in [0, m), NaN after.
    u = _linear_quantile(np.sort(np.where(fin, Z, np.nan), axis=1), counts, 1.0 - frac)
    with np.errstate(invalid="ignore"):
        exc = fin & (Z > u[:, None])
    k = exc.sum(axis=1)
    # R16-140: classic Hill is defined on a STRICTLY-POSITIVE tail magnitude.
    # A ``u < 0`` threshold with a mixed-sign exceedance set would feed
    # ``log(negative)`` into the mean; that is undefined, not a valid
    # estimator.  Every ratio ``x_i / u`` must be strictly positive (same
    # sign, same as the threshold) — otherwise the window is NaN.
    ok = ok & np.isfinite(u) & (u > 0.0) & (k >= mtc)
    # Compute log(exc/u) without forming a potentially overflowing ratio.
    # log1p preserves nextafter-sized exceedances close to the threshold;
    # log differences cover the full finite dynamic range farther away.
    zz = np.where(exc, Z, 1.0)
    uu = np.where(exc, u[:, None], 1.0)
    delta = zz - uu
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_ratios = np.where(delta <= uu, np.log1p(delta / uu), np.log(zz) - np.log(uu))
    ok = ok & np.all(np.isfinite(log_ratios) | ~exc, axis=1)
    xi = np.where(exc, log_ratios, 0.0).sum(axis=1) / np.maximum(k, 1)
    return np.where(ok, xi, np.nan)


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
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=120, param_role=ParamRole.HORIZON),
            "side": ParamSpec(dtype=str, choices=("upper", "lower"), default="upper", searchable=False, param_role=ParamRole.POLICY),
            "tail_fraction": ParamSpec(dtype=float, min=0.0, max=0.5, default=0.2, param_role=ParamRole.THRESHOLD),
            "min_tail_count": ParamSpec(dtype=int, min=3, default=10, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
        panel_params=("x",),
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
    design = np.column_stack([np.ones(x.shape[0]), x])
    beta = pinball_quantile_fit(design, y, q)
    if beta is None:
        return np.nan
    return float(beta[1])

def _quantile_beta_series(y: np.ndarray, x: np.ndarray, window: int, q: float) -> np.ndarray:
    rows, cols = y.shape
    w = strict_integer(window, "window", minimum=2)
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
    backend="pandas_numpy",
    status="experimental",
)
class TsQuantileRegressionBeta(SeriesOperator):
    """精确分位数回归斜率 β_q（LP 求解）。

    最小化 ``Σ ρ_q(y - α - βx)``（``ρ_q(u)=u(q-I(u<0))``），exact 线性规划
    (HiGHS)，不用不稳定的 IRLS 近似。

    R26-063/064 语义澄清：β_q = ∂Q_y(q|x)/∂x 是 **conditional quantile slope**
    ——给定 x 时 y 的 q 分位点对 x 的边际变化率。它**不是**"当 y 自己处于
    q-tail 状态时对 y 做 OLS"的 tail-state regression（那是另一个需要独立
    实现的 canonical）。P2 / Research。
    """

    metadata = _metadata(
        "ts_quantile_regression_beta",
        "分位数回归斜率 β_q（精确 LP，单位 unit(y)/unit(x)）。",
        ["y", "x", "window", "quantile"],
        # A regression slope carries unit(y)/unit(x), NOT a uniform ratio
        # (review #40) — mirroring the ts_expectile_beta convention.
        unit="unit(y)/unit(x)",
        cost=8,
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=120, param_role=ParamRole.HORIZON),
            "quantile": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5, param_role=ParamRole.THRESHOLD),
        },
        panel_params=("y", "x"),
    )

    def _calculate_series(
        self, y: pd.DataFrame, x: pd.DataFrame, window: int = 120, quantile: float = 0.5, **_: Any
    ) -> pd.DataFrame:
        q = strict_finite_scalar(quantile, "quantile")
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

    R62: the sequential scan is closed-form.  Between two consecutive
    exceedances at window offsets ``p < j`` every intervening bar is by
    construction a finite non-exceedance, so the authority's ``gap`` is
    ``j - p - 1`` unless a NaN intervenes, in which case the NaN forces
    ``gap = run_length`` (a break).  Hence the cluster break is exactly
    ``(j - p - 1 >= run_length) | any NaN strictly between`` — a masked
    prefix-count expression over the ``(T, w)`` window batch.
    """
    n = series.shape[0]
    w = strict_integer(window, "window", minimum=2)
    quant = strict_finite_scalar(q, "q")
    if not 0.0 < quant < 1.0:
        raise ValueError("q must be in (0, 1)")
    rl = strict_integer(run_length, "run_length", minimum=1)
    mex = strict_integer(min_exceed, "min_exceed", minimum=2)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    W = _winmat(series, w)
    fin = np.isfinite(W)
    counts = fin.sum(axis=1)
    sorted_w = np.sort(np.where(fin, W, np.nan), axis=1)
    if side == "upper":
        thr = _linear_quantile(sorted_w, counts, quant)
        exc = fin & (W > thr[:, None])
    else:
        # R4-87: lower tail = strictly below the (1-quant) quantile, so a
        # positive price/valuation series keeps a well-defined lower tail.
        thr = _linear_quantile(sorted_w, counts, 1.0 - quant)
        exc = fin & (W < thr[:, None])
    # R11: gap-based clustering with an explicit run_length, and a NaN bar
    # BREAKS the current run (pre-R11 carried ``prev`` across NaN, so
    # ``Extreme, NaN, NaN, NaN, Extreme`` was miscounted as ONE cluster).
    offsets = np.arange(w, dtype=np.int64)
    seen_at = np.where(exc, offsets[None, :], -1)
    prev_incl = np.maximum.accumulate(seen_at, axis=1)
    prev = np.concatenate((np.full((n, 1), -1, dtype=np.int64), prev_incl[:, :-1]), axis=1)
    nan_incl = np.cumsum(~fin, axis=1)                       # NaNs in offsets 0..i
    nan_excl = np.concatenate(                              # NaNs in offsets 0..i-1
        (np.zeros((n, 1), dtype=np.int64), nan_incl[:, :-1]), axis=1
    )
    has_prev = exc & (prev >= 0)
    nan_between = has_prev & (
        (nan_excl - np.take_along_axis(nan_incl, np.clip(prev, 0, w - 1), axis=1)) > 0
    )
    gap_break = has_prev & ((offsets[None, :] - prev - 1) >= rl)
    breaks = (nan_between | gap_break).sum(axis=1)
    k = exc.sum(axis=1)
    ok = (counts >= 3) & (k >= mex)
    theta = np.where(k >= 1, 1.0 + breaks, 0.0) / np.maximum(k, 1)
    return np.where(ok, theta, np.nan)


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
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=120, param_role=ParamRole.HORIZON),
            "side": ParamSpec(dtype=str, choices=("upper", "lower"), default="upper", searchable=False, param_role=ParamRole.POLICY),
            "q": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.9, param_role=ParamRole.THRESHOLD),
            "min_exceed": ParamSpec(dtype=int, min=2, default=3, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
            "run_length": ParamSpec(dtype=int, min=1, default=1, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
        panel_params=("x",),
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

    R62: the nine threshold quantiles and their exceedance means are one
    ``(T, w)`` batch each; the OLS uses the authority's own
    ``Σ(u-ū)²`` / ``Σ(u-ū)(m-m̄)`` form (deliberately NOT rescaled): the
    authority overflows to NaN on the 1e+300 panel and underflows below ``_EPS``
    on 1e-300, and both fail-closed patterns are part of the contract.
    """
    n = series.shape[0]
    w = strict_integer(window, "window", minimum=2)
    mtc = strict_integer(min_tail_count, "min_tail_count", minimum=2)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    # R11: the lower tail is the exact mirror of the upper tail on ``y = -x``
    # (ME_lower(x) == ME_upper(-x)); reusing one code path keeps the slope
    # direction consistent with the documented interpretation on both sides.
    W = _winmat(series, w)
    fin = np.isfinite(W)
    counts = fin.sum(axis=1)
    work = -W if side == "lower" else W
    sorted_work = np.sort(np.where(fin, work, np.nan), axis=1)
    # prefix sums of the ascending window: the k exceedances are exactly the top
    # k values S[:, m-k : m], so their sum is one lookup into this table (the
    # exceedance count still comes from the exact ``> u`` mask, ties included).
    cw = np.cumsum(np.where(np.isfinite(sorted_work), sorted_work, 0.0), axis=1)
    csum = np.concatenate((np.zeros((n, 1)), cw[:, :-1]), axis=1)
    total = cw[:, -1]
    grid = np.linspace(0.55, 0.95, 9)
    ok = counts >= (mtc + 3)
    thr_found = np.zeros((n, 9), dtype=bool)
    u_mat = np.empty((n, 9), dtype=float)
    me_mat = np.empty((n, 9), dtype=float)
    for j, p in enumerate(grid):
        u = _linear_quantile(sorted_work, counts, float(p))
        k = (sorted_work > u[:, None]).sum(axis=1)
        keep = ok & (k >= mtc)
        start = np.clip(counts - k, 0, w - 1)
        top = np.take_along_axis(csum, start[:, None], axis=1)[:, 0]
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            me = (total - top) / np.maximum(k, 1) - u
        u_mat[:, j] = np.where(keep, u, np.nan)
        me_mat[:, j] = np.where(keep, me, np.nan)
        thr_found[:, j] = keep
    used = thr_found.sum(axis=1)
    ok = ok & (used >= 3)
    used_f = np.maximum(used, 1).astype(np.float64)
    u_z = np.where(thr_found, u_mat, 0.0)
    me_z = np.where(thr_found, me_mat, 0.0)
    u_bar = u_z.sum(axis=1) / used_f
    me_bar = me_z.sum(axis=1) / used_f
    du = np.where(thr_found, u_mat - u_bar[:, None], 0.0)
    dm = np.where(thr_found, me_mat - me_bar[:, None], 0.0)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        denom = (du * du).sum(axis=1)
        numer = (du * dm).sum(axis=1)
        slope = numer / denom
    ok = ok & (denom > _EPS) & np.isfinite(slope)
    return np.where(ok, slope, np.nan)


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
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=120, param_role=ParamRole.HORIZON),
            "side": ParamSpec(dtype=str, choices=("upper", "lower"), default="upper", searchable=False, param_role=ParamRole.POLICY),
            "min_tail_count": ParamSpec(dtype=int, min=2, default=5, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
        panel_params=("x",),
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

    R62: one ``(T, w)`` batch.  The exceedances are gathered straight out of
    the per-row sorted window (upper = top-k ascending, lower = bottom-k
    reversed) and the PWM weights are applied positionally, so no per-window
    Python loop survives.
    """
    n = series.shape[0]
    w = strict_integer(window, "window", minimum=2)
    frac = strict_finite_scalar(tail_fraction, "tail_fraction")
    if not (0.0 < frac <= 0.5):
        raise ValueError("tail_fraction must satisfy 0 < tail_fraction <= 0.5")
    mtc = strict_integer(min_tail_count, "min_tail_count", minimum=3)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    W = _winmat(series, w)
    fin = np.isfinite(W)
    counts = fin.sum(axis=1)
    ok = counts >= (mtc + 2)
    sorted_w = np.sort(np.where(fin, W, np.nan), axis=1)
    if side == "upper":
        u = _linear_quantile(sorted_w, counts, 1.0 - frac)
        exc = fin & (W > u[:, None])
    else:
        # R4-87: lower tail = excess ``u - x`` below the frac-quantile
        # threshold; defined for positive price/valuation series too.
        u = _linear_quantile(sorted_w, counts, frac)
        exc = fin & (W < u[:, None])
    k = exc.sum(axis=1)
    ok = ok & (k >= mtc)
    # exc ascending inside the run: upper = S[m-k .. m-1] - u,
    # lower = u - S[k-1 .. 0] (the mirrored bottom-k, still ascending).
    rank = np.arange(w, dtype=np.int64)[None, :]
    if side == "upper":
        gather = (counts - k)[:, None] + rank
    else:
        gather = (k - 1)[:, None] - rank
    in_run = rank < k[:, None]
    picked = np.take_along_axis(sorted_w, np.clip(gather, 0, w - 1), axis=1)
    with np.errstate(invalid="ignore"):
        ex_val = (picked - u[:, None]) if side == "upper" else (u[:, None] - picked)
    ex_val = np.where(in_run, ex_val, np.nan)
    weights = np.where(in_run, rank / np.maximum(k - 1, 1)[:, None], np.nan)
    kf = np.maximum(k, 1)
    b0 = np.where(in_run, ex_val, 0.0).sum(axis=1) / kf
    b1 = np.where(in_run, weights * ex_val, 0.0).sum(axis=1) / kf
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        denom = 2.0 * b1 - b0
        ok = ok & (np.abs(denom) > _EPS)
        xi = 2.0 - b0 / denom
    ok = ok & np.isfinite(xi)
    return np.where(ok, xi, np.nan)


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
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=120, param_role=ParamRole.HORIZON),
            "side": ParamSpec(dtype=str, choices=("upper", "lower"), default="upper", searchable=False, param_role=ParamRole.POLICY),
            "tail_fraction": ParamSpec(dtype=float, min=0.0, max=0.5, default=0.2, param_role=ParamRole.THRESHOLD),
            "min_tail_count": ParamSpec(dtype=int, min=3, default=10, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
        panel_params=("x",),
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
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({"ts_hill_tail_index", "ts_extremal_index", "ts_mean_excess_slope", "ts_gpd_shape_pwm"})
    _surface.extend_research_only({"ts_quantile_regression_beta"})


_register_surface()
