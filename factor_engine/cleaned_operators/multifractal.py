# -*- coding: utf-8 -*-
"""Multifractal spectrum operators (2026-08 geometry/math expansion).

The generalised Hurst exponent family characterises how the scaling of price
increments depends on the moment order ``q``.  For a trailing window ``X`` the
structure function at lag ``tau`` is the ``q``-th absolute moment of the
lag-``tau`` increments computed over *aligned finite pairs*,

    S_q(tau) = mean |X_{t+tau} - X_t|^q ,

and the scaling law ``log S_q(tau) = a + b * log tau`` is fitted by ordinary
least squares over the dyadic lags ``tau in {1, 2, 4, 8}``.  The generalised
Hurst exponent is ``H(q) = b / q``.

* ``ts_generalized_hurst_exponent``  — ``H(q)`` for a single moment order.
* ``ts_generalized_hurst_spread_q1_q4`` — ``H(1) - H(4)``, the spread of the
  generalised-Hurst curve over the two lowest/largest reviewed moment orders.
  Renamed P1-20 from the misleading ``ts_multifractal_width`` (which is kept
  as a working alias).
* ``ts_multifractal_spectrum_width`` — the true multifractal singularity-spectrum
  width ``alpha_max - alpha_min`` obtained from ``tau(q) = q*H(q) - 1`` via the
  Legendre transform (a smooth polynomial fit in ``q`` before differentiation).
* ``ts_multifractal_curvature``      — the quadratic coefficient ``c`` of the
  fit ``H(q) = a + b*q + c*q^2`` over the reviewed grid
  ``q in {0.5, 1, 2, 3, 4}``.

All operators are trailing-window, prefix-causal and deterministic.  NaN
inputs are dropped from the window via aligned finite pairs; a window that
lacks enough aligned pairs for the largest lag emits NaN (never Inf, never a
fabricated zero).  The CURRENT row is always required (P0-6): a NaN current
observation emits NaN instead of falling back to yesterday's contiguous run
and emitting a stale-history estimate.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, ParamSpec, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_bridge

_LAGS = (1, 2, 4, 8)
_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="multifractal",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "multifractal", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic", "current_row_required",
            f"signature:{','.join(params)}->series", "domain:scaling",
            f"unit:{unit}", f"cost:{cost}",
        ],
        # R4-95: the trailing ``window`` is a real horizon — it must be fully
        # accumulated before an estimate is emitted (R4-94).  Values inside the
        # window are used as aligned finite pairs, so at most ``window`` rows
        # are consumed (gaps reduce the pair count, never the row requirement).
        window_semantics="max_rows",
    )


def _check_window(window: int) -> int:
    w = int(window)
    if w < 2:
        raise ValueError("window must be >= 2")
    return w


_MIN_PAIRS_PER_LAG = 8
_MIN_LAGS_FOR_FIT = 3
_MIN_SCALING_R2 = 0.9


def _common_cohort(vals: np.ndarray, mode: str = "trailing_current_contiguous") -> np.ndarray:
    """Trailing contiguous finite run — the COMMON observation cohort (audit #91).

    Every dyadic lag must estimate its structure function on the SAME data
    slice.  A missing pattern inside the window must not let lag=1 use rows a
    lag=4 fit cannot see (different cohorts -> different scaling laws).  Using
    the trailing contiguous finite suffix gives every lag one shared cohort.

    P0-6: the default mode is ``trailing_current_contiguous`` — the CURRENT row
    is part of the cohort.  When the current observation is NaN the cohort is
    empty, so the row emits NaN instead of falling back to yesterday's run and
    emitting a stale-history H(q).  ``historical_last_contiguous`` is the named,
    NON-default mode that preserves the old walk-back fallback for callers that
    explicitly opt in.
    """
    v = np.asarray(vals, dtype=float)
    n = v.size
    if n == 0:
        return v
    if mode == "trailing_current_contiguous":
        if not np.isfinite(v[-1]):
            return v[:0]
    # historical_last_contiguous (or any unknown mode keeps the historical
    # behaviour): fall back to the last contiguous finite run before the gap.
    j = n
    while j > 0 and not np.isfinite(v[j - 1]):
        j -= 1
    i = j
    while i > 0 and np.isfinite(v[i - 1]):
        i -= 1
    return v[i:j]


def _structure_function(vals: np.ndarray, lag: int, q: float) -> float:
    """``mean |X_{t+tau} - X_t|^q`` over aligned finite pairs in the window."""
    n = int(vals.shape[0])
    if n < lag + 2:
        return np.nan
    x = vals[: n - lag]
    y = vals[lag:]
    m = np.isfinite(x) & np.isfinite(y)
    # Audit P1-D: every lag needs a minimum number of valid aligned pairs —
    # two points would let a single outlier dominate the moment.
    # Audit #89: raise the sample floor as |q| grows — higher moments are
    # dominated by extreme increments and need more observations to be stable.
    required = max(_MIN_PAIRS_PER_LAG, int(np.ceil(_MIN_PAIRS_PER_LAG * abs(float(q)))))
    if int(m.sum() < required:
        return np.nan
    d = np.abs(x[m] - y[m])
    s = float(np.mean(d ** q))
    if not np.isfinite(s) or s <= 0.0:
        return np.nan
    return s


def _hurst_generalized(vals: np.ndarray, q: float) -> float:
    """Generalised Hurst exponent ``H(q)`` from the dyadic-lag OLS fit.

    Audit P1-D: the log-log scaling fit needs at least three valid lags and a
    minimum R² — a two-point line would produce a spuriously precise Hurst.
    Audit #91: the trailing contiguous finite run is the COMMON cohort for all
    lags, so a missing pattern cannot make different lags estimate scaling on
    different data slices.
    """
    run = _common_cohort(vals)
    if run.shape[0] < _LAGS[-1] + 2:
        return np.nan
    log_t: list[float] = []
    log_s: list[float] = []
    for lag in _LAGS:
        s = _structure_function(run, lag, q)
        if not np.isfinite(s):
            continue
        log_t.append(np.log(float(lag)))
        log_s.append(np.log(s))
    if len(log_t) < _MIN_LAGS_FOR_FIT:
        return np.nan
    slope, intercept = np.polyfit(log_t, log_s, 1)
    if not np.isfinite(slope):
        return np.nan
    # R² of the log-log fit: fail closed when the scaling law is not clean.
    fitted = slope * np.asarray(log_t) + intercept
    ss_res = float(np.sum((np.asarray(log_s) - fitted) ** 2))
    ss_tot = float(np.sum((np.asarray(log_s) - np.mean(log_s)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    if not np.isfinite(r2) or r2 < _MIN_SCALING_R2:
        return np.nan
    return float(slope) / q


def _hurst_series(x2d: np.ndarray, window: int, q: float) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            # R4-94: ``window`` is a real horizon — the trailing window must be
            # fully accumulated before an H estimate is emitted.  Previously the
            # ``max(0, r - w + 1)`` truncation let a partial window start
            # estimating H as soon as the aligned-pair floor was met (e.g.
            # ~10 rows into a window=120 series), which is not the same scaling
            # law as the full window and produced unstable early values.
            if r + 1 < w:
                continue
            i0 = r - w + 1
            out[r, c] = _hurst_generalized(col[i0 : r + 1], q)
    return out


def _hurst_spread_series(x2d: np.ndarray, window: int) -> np.ndarray:
    """Generalised-Hurst spread ``H(1) - H(4)`` (P1-20 renamed canonical).

    This is the spread of the generalised-Hurst curve over the two lowest /
    two highest reviewed moment orders — NOT a multifractal singularity-spectrum
    width (which is ``ts_multifractal_spectrum_width``).  The current row is
    required (P0-6): ``_hurst_generalized`` returns NaN when the current
    observation is missing, so the spread emits NaN on a stale-history window.
    """
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            if r + 1 < w:
                continue
            i0 = r - w + 1
            chunk = col[i0 : r + 1]
            h1 = _hurst_generalized(chunk, 1.0)
            h4 = _hurst_generalized(chunk, 4.0)
            if np.isfinite(h1) and np.isfinite(h4):
                out[r, c] = h1 - h4
    return out


# Reviewed moment-order grid for the spectrum (audit #89): large positive q is
# dominated by single extreme increments and is not robust.
_Q_GRID = np.array([0.5, 1.0, 2.0, 3.0, 4.0])


def _spectrum_width_series(x2d: np.ndarray, window: int) -> np.ndarray:
    """True multifractal singularity-spectrum width ``alpha_max - alpha_min``.

    P1-20: ``tau(q) = q*H(q) - 1`` is built on the reviewed grid, a smooth
    polynomial ``tau(q)`` is fit over the finite ``H(q)`` points (a few ``q``
    values), and the Legendre transform gives ``alpha = d tau/dq`` and
    ``f(alpha) = q*alpha - tau(q)``; the width is the range of ``alpha`` over
    the fitted grid.  Fails closed (NaN) on insufficient history / too few
    finite ``q`` points.  The current row is required (P0-6).
    """
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    qs = _Q_GRID
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            if r + 1 < w:
                continue
            i0 = r - w + 1
            chunk = col[i0 : r + 1]
            hs = np.asarray([_hurst_generalized(chunk, q_) for q_ in qs], dtype=float)
            ok = np.isfinite(hs)
            # Audit #90-style floor: a quadratic tau(q) has 3 parameters; fitting
            # to only 3 q-points leaves zero residual DOF.  Require >= 4.
            if int(ok.sum() < 4:
                continue
            q_ok = qs[ok]
            tau = q_ok * hs[ok] - 1.0
            # Fit a smooth tau(q) before differentiating (Legendre transform).
            coeffs = np.polyfit(q_ok, tau, 2)
            if not np.all(np.isfinite(coeffs)):
                continue
            # alpha(q) = d(tau)/dq evaluated on the fitted q points.
            alpha_vals = np.polyval(np.polyder(coeffs), q_ok)
            if not np.all(np.isfinite(alpha_vals)):
                continue
            width = float(np.max(alpha_vals) - np.min(alpha_vals))
            if np.isfinite(width) and width >= 0.0:
                out[r, c] = width
    return out


def _curvature_series(x2d: np.ndarray, window: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    # Audit #89: reviewed q grid for the spectrum (large positive q is
    # dominated by single extreme increments and is not robust).
    qs = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            if r + 1 < w:
                continue
            i0 = r - w + 1
            chunk = col[i0 : r + 1]
            hs = np.asarray([_hurst_generalized(chunk, q_) for q_ in qs], dtype=float)
            ok = np.isfinite(hs)
            # Audit #90: a quadratic fit has 3 parameters; fitting to only 3
            # q-points leaves zero residual DOF (the fit is exactly
            # interpolated, not estimated).  Require >= 4 valid q points.
            if int(ok.sum() < 4:
                continue
            coeffs = np.polyfit(qs[ok], hs[ok], 2)
            if np.isfinite(coeffs[0]):
                out[r, c] = float(coeffs[0])
    return out


@register_operator(
    name="ts_generalized_hurst_exponent",
    category="multifractal",
    business_category="multifractal",
    canonical="ts_generalized_hurst_exponent",
    source="multifractal",
)
class TsGeneralizedHurstExponent(SeriesOperator):
    """广义 Hurst 指数 H(q)：log S_q(tau) 对 log tau 的 OLS 斜率 / q。

    q=2 接近经典 Hurst 指数；q 大时强调大增量（间歇性/尾部），q 小时强调
    典型尺度。常数窗口 / 最大 lag 有效配对不足 -> NaN。P1。

    q 取值受审计网格约束（audit #89）：``q ∈ {0.5, 1, 2, 3, 4}``——任意大的正
    q 只被少数极端增量主导，不稳健；且随 |q| 增长样本下限同步提高。所有 lag
    共用同一个尾部连续有限队列（audit #91），缺失模式不会让不同 lag 在不同数据
    切片上估标度指数。
    """

    metadata = _metadata(
        "ts_generalized_hurst_exponent",
        "广义 Hurst 指数 H(q)（dyadic-lag 结构函数 OLS 斜率 / q），无量纲。",
        ["x", "window", "q"],
        unit="ratio",
        cost=6,
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2),
        "q": ParamSpec(dtype=float, choices=(0.5, 1.0, 2.0, 3.0, 4.0)),
    }

    def _calculate_series(self, x: pd.DataFrame, window: int = 120, q: float = 2.0, **_: Any) -> pd.DataFrame:
        w = _check_window(window)
        qv = float(q)
        # Audit #89: only the reviewed grid is admissible; arbitrary large q is
        # dominated by a few extreme increments.
        if qv not in (0.5, 1.0, 2.0, 3.0, 4.0):
            raise ValueError("q must be one of {0.5, 1, 2, 3, 4} (reviewed grid)")
        return frame_like(x, _hurst_series(x.to_numpy(dtype=float), w, qv))


@register_operator(
    name="ts_generalized_hurst_spread_q1_q4",
    category="multifractal",
    business_category="multifractal",
    canonical="ts_generalized_hurst_spread_q1_q4",
    source="multifractal",
)
class TsGeneralizedHurstSpreadQ1Q4(SeriesOperator):
    """广义 Hurst 展幅 H(1)-H(4)（P1-20 诚实改名）。

    这是广义 Hurst 曲线在最低/最高两个已审时刻阶数上的展幅——不是多重分形
    奇异谱宽度（那是 ``ts_multifractal_spectrum_width``）。大 -> 不同 moment
    order 的标度行为差异大（多重分形 / 异质波动）；近 0 -> 单分形。旧名
    ``ts_multifractal_width`` 作为别名保留。P1。
    """

    metadata = _metadata(
        "ts_generalized_hurst_spread_q1_q4",
        "广义 Hurst 展幅 H(1)-H(4)。",
        ["x", "window"],
        unit="ratio",
        cost=7,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 120, **_: Any) -> pd.DataFrame:
        w = _check_window(window)
        return frame_like(x, _hurst_spread_series(x.to_numpy(dtype=float), w))


@register_operator(
    name="ts_multifractal_spectrum_width",
    category="multifractal",
    business_category="multifractal",
    canonical="ts_multifractal_spectrum_width",
    source="multifractal",
)
class TsMultifractalSpectrumWidth(SeriesOperator):
    """多重分形奇异谱宽 ``alpha_max - alpha_min``（P1-20 新增）。

    先估 H(q)（审计网格 q∈{0.5,1,2,3,4}），构造 tau(q) = q*H(q) - 1，
    对 q 拟合平滑多项式，再做 Legendre 变换 alpha = d tau/dq，
    f(alpha) = q*alpha - tau(q)，宽度 = alpha 在拟合网格上的跨度。
    历史不足 / 有效 q 点过少 -> NaN。P1。
    """

    metadata = _metadata(
        "ts_multifractal_spectrum_width",
        "多重分形奇异谱宽 alpha_max - alpha_min（Legendre 变换）。",
        ["x", "window"],
        unit="ratio",
        cost=8,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 120, **_: Any) -> pd.DataFrame:
        w = _check_window(window)
        return frame_like(x, _spectrum_width_series(x.to_numpy(dtype=float), w))


@register_operator(
    name="ts_multifractal_curvature",
    category="multifractal",
    business_category="multifractal",
    canonical="ts_multifractal_curvature",
    source="multifractal",
)
class TsMultifractalCurvature(SeriesOperator):
    """H(q) 对 q 的二次拟合曲率系数 c。

    H(q) = a + b*q + c*q^2 在审计网格 q∈{0.5, 1, 2, 3, 4} 拟合（P1-21：
    文档与实现统一为真实网格）；c<0 表示谱的弧形下弯（强间歇性），c≈0 表示
    近似线性谱。P2。
    """

    metadata = _metadata(
        "ts_multifractal_curvature",
        "H(q) 对 q 二次拟合的曲率系数 c。",
        ["x", "window"],
        unit="ratio",
        cost=8,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 120, **_: Any) -> pd.DataFrame:
        w = _check_window(window)
        return frame_like(x, _curvature_series(x.to_numpy(dtype=float), w))


_NEW_CANONICALS = (
    "ts_generalized_hurst_exponent",
    "ts_generalized_hurst_spread_q1_q4",
    "ts_multifractal_spectrum_width",
    "ts_multifractal_curvature",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)
    # P1-20 honest rename: ``ts_multifractal_width`` is a deprecated alias of
    # ``ts_generalized_hurst_spread_q1_q4`` (same H(1)-H(4) spread).  The old
    # spelling stays resolvable but is no longer a canonical.
    try:
        from cleaned_operators.registry import OperatorRegistry

        OperatorRegistry.register_alias(
            "ts_multifractal_width",
            "ts_generalized_hurst_spread_q1_q4",
        )
    except Exception:  # pragma: no cover - alias is best-effort, never fatal
        pass
    # P1-21: the curvature operator's documented q grid is now the real grid
    # (0.5,1,2,3,4).  Bump the semantic version so the identity changes visibly
    # to catalog consumers.
    try:
        from cleaned_operators.registry import OperatorRegistry

        _cat = OperatorRegistry._catalog.get("ts_multifractal_curvature")
        if _cat is not None:
            _cat["semantic_version"] = "2.0"
    except Exception:  # pragma: no cover - catalog is best-effort, never fatal
        pass


_register_surface()
