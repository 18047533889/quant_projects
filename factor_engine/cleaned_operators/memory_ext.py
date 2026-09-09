# -*- coding: utf-8 -*-
"""Serial-dependence memory operators (2026-08 geometry/math expansion).

Two complements:

* ``ts_autocorrelation_time`` — integrated autocorrelation time ``tau_int`` of
  the window, in bars (review R4-56), computed with the Geyer (1992)
  initial-MONOTONE-sequence estimator (audit #61): sample autocorrelations
  ``rho(k)`` use the STANDARD finite-N denominator ``gamma_k = (1/(N-k)) sum
  (x_t - xbar)(x_{t+k} - xbar), rho_k = gamma_k / gamma_0`` (audit #60) over the
  *trailing contiguous finite window* (review R4-57: the same cohort for the
  mean, gamma_0 and every lag-k numerator — never a re-connected axis); the
  cumulative-minimum sequence of positive pairs ``P_k=rho_(2k)+rho_(2k+1)``
  gives ``tau=-1+2 sum(P)``. The sum is NOT divided by ``max_lag``, so white noise
  gives ``~1`` bar regardless of ``max_lag``.  The precise spelling
  ``ts_integrated_autocorrelation_time`` is an alias of this canonical.
  ``ts_autocorrelation_time_initial_positive_sequence`` exposes the simpler
  Geyer initial-positive-sequence estimator for research use.
* ``ts_fractional_difference`` — the fractional differencing transform
  ``y_t = sum_{k=0}^{min(cutoff, t)} w_k * x_{t-k}`` with the binomial-weight
  recursion ``w_0 = 1, w_k = -w_{k-1} * (d - k + 1) / k`` for ``d in (-1, 1)``.
  ``d > 0`` removes long memory (stationarising), ``d < 0`` is fractional
  integration.  Audit #62: a fixed ``cutoff`` must follow ``d`` — the operator
  reports the discarded theoretical |weight| mass and FAILS CLOSED (NaN) when it
  exceeds ``max_discarded_weight_mass`` (for ``d < 0`` the weights decay like
  ``k^{-(1+d)}``, a divergent p-series, so any finite cutoff discards ~all mass
  and the gate always fails).  ``ts_fractional_difference_discarded_weight_mass``
  exposes the discarded mass as a diagnostic output.

All operators are trailing-window / causal, deterministic and NaN-safe: the
autocorrelation sums stop on missing lags, and a fractional-difference output
is NaN whenever any input in its causal support is NaN (values are never
re-connected across a gap).

NOTE on the parameter name: the fractional differencing order is exposed as
``fd`` (not ``d``) because the framework's base layer auto-normalises the
integer-typed parameter named ``d`` and would reject the required fractional
value ``0.4``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.rolling_pack import frame_like, register_polars_bridge


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="memory",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "memory", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:long_memory",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _check_window(window: int) -> int:
    w = int(window)
    if w < 2:
        raise ValueError("window must be >= 2")
    return w


def _trailing_contiguous_finite(chunk: np.ndarray) -> np.ndarray:
    """Trailing contiguous run of finite values (suffix after the last gap)."""
    finite = np.isfinite(chunk)
    if not np.any(finite):
        return np.array([], dtype=float)
    bad = np.flatnonzero(~finite)
    if bad.size == 0:
        return chunk
    return chunk[int(bad[-1]) + 1 :]


def _sample_acf(contig: np.ndarray, max_lag: int) -> np.ndarray:
    """Standard lag-k sample autocorrelations ``rho_0 .. rho_{min(max_lag, N-1)}``.

    Audit #60: the standard estimator uses a *separate* finite-N denominator per
    lag:

        gamma_k = (1/(N-k)) * sum_{t=1}^{N-k} (x_t - xbar)(x_{t+k} - xbar)
        rho_k   = gamma_k / gamma_0

    The old code divided every lag-k numerator by the full-sample variance
    (a denominator with N terms), which produced a finite-N bias that grew with
    ``k``.  Review R4-57 is preserved: the mean, gamma_0 and every lag-k
    numerator all come from the SAME trailing contiguous finite cohort — values
    are never re-connected across a gap.
    """
    n = int(contig.size)
    if n < 3:
        return np.empty(0, dtype=float)
    magnitude = float(np.max(np.abs(contig)))
    if not np.isfinite(magnitude) or magnitude == 0.0:
        return np.empty(0, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        centered = contig - contig[0]
    if not np.all(np.isfinite(centered)):
        centered = contig / magnitude
        centered -= centered[0]
    centered /= float(np.max(np.abs(centered))) or 1.0
    centered -= float(np.mean(centered))
    spread = float(np.max(np.abs(centered)))
    if spread == 0.0:
        return np.empty(0, dtype=float)
    centered /= spread
    gamma0 = float(np.dot(centered, centered)) / float(n)
    if not np.isfinite(gamma0) or gamma0 <= 0.0:
        return np.empty(0, dtype=float)
    m = min(int(max_lag), n - 1)
    rho = np.empty(m + 1, dtype=float)
    rho[0] = 1.0
    for k in range(1, m + 1):
        num = float(np.dot(centered[: n - k], centered[k:]))
        gamma_k = num / float(n - k)
        rho[k] = gamma_k / gamma0
    return rho


def _geyer_positive_pairs(rho: np.ndarray) -> np.ndarray | None:
    """Initial positive prefix of P_k = rho_(2k) + rho_(2k+1).

    Discard an unpaired final even lag. Nonfinite ACF is undefined. The
    N-k sample covariance policy is unchanged; no clipping forces a noisy
    finite-sample estimate to be a positive theoretical population value.
    """
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 1 or rho.size < 2 or not np.all(np.isfinite(rho)):
        return None
    count = rho.size // 2
    pairs = rho[:2*count].reshape(count, 2).sum(axis=1)
    stop = np.flatnonzero(pairs <= 0.0)
    return pairs[:int(stop[0])] if stop.size else pairs


def _geyer_ims_tau(rho: np.ndarray) -> float:
    """Geyer initial monotone sequence: cumulative minima of positive pairs.

    tau = -1 + 2 * sum(P_k); this is not truncation of individual rho_k.
    The theoretical shape properties assume a stationary reversible process;
    applying the estimator to financial data does not certify that assumption.
    """
    pairs = _geyer_positive_pairs(rho)
    if pairs is None:
        return np.nan
    return float(-1.0 + 2.0 * np.minimum.accumulate(pairs).sum())


def _geyer_ips_tau(rho: np.ndarray) -> float:
    """Geyer initial positive paired sequence, without monotone adjustment."""
    pairs = _geyer_positive_pairs(rho)
    return np.nan if pairs is None else float(-1.0 + 2.0 * pairs.sum())


def _autocorrelation_time(chunk: np.ndarray, max_lag: int) -> float:
    contig = _trailing_contiguous_finite(chunk)
    rho = _sample_acf(contig, max_lag)
    if rho.size == 0:
        return np.nan
    return _geyer_ims_tau(rho)


def _autocorrelation_time_series(x2d: np.ndarray, window: int, max_lag: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    ml = int(max_lag)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            out[r, c] = _autocorrelation_time(col[i0 : r + 1], ml)
    return out


def _autocorrelation_time_series_ips(x2d: np.ndarray, window: int, max_lag: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    ml = int(max_lag)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            rho = _sample_acf(_trailing_contiguous_finite(col[i0 : r + 1]), ml)
            if rho.size == 0:
                continue
            val = _geyer_ips_tau(rho)
            if np.isfinite(val):
                out[r, c] = val
    return out


def _fd_weights(fd: float, cutoff: int) -> np.ndarray:
    c = int(cutoff)
    w = np.empty(c + 1, dtype=float)
    w[0] = 1.0
    for k in range(1, c + 1):
        w[k] = -w[k - 1] * (fd - k + 1) / k
    return w


def _fd_discarded_weight_mass(fd: float, cutoff: int, _tail_terms: int = 50000) -> float:
    """Fraction of the theoretical |weight| mass discarded by a finite cutoff.

    Audit #62: the fractional-difference cutoff must FOLLOW ``d``.  For ``d < 0``
    (fractional integration) the weights are all positive and decay like
    ``k^{-(1+d)}`` — a DIVERGENT p-series (``1+d in (0,1)``) — so any finite
    cutoff discards essentially all of the theoretical mass (return 1.0).  For
    ``d > 0`` the absolute mass converges; the recursion is summed out to
    ``_tail_terms`` and an analytic power-law remainder ``|w_k| ~ |w_N| (k/N)^
    {-(d+1)}`` (``sum ~ |w_N| * N / d``) is added.  ``d == 0`` leaves only
    ``w_0 = 1``, so nothing is discarded.
    """
    c = int(cutoff)
    if fd < 0.0:
        return 1.0
    if fd == 0.0:
        return 0.0
    w = _fd_weights(fd, c)
    kept = float(np.sum(np.abs(w)))
    total = kept
    N = int(_tail_terms)
    w_prev = float(w[-1])
    for k in range(c + 1, N + 1):
        wk = -w_prev * (fd - k + 1) / k
        total += abs(wk)
        w_prev = wk
    total += abs(w_prev) * float(N) / fd
    if total <= 1e-15:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - kept / total)))


def _fractional_difference_series(x2d: np.ndarray, fd: float, cutoff: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    c = int(cutoff)
    w = _fd_weights(fd, c)
    for col in range(cols):
        colv = x2d[:, col]
        for r in range(rows):
            # R11 #41: the fractional filter has a causal support of ``cutoff+1``
            # terms (x_t .. x_{t-cutoff}).  The startup region ``r < cutoff``
            # cannot apply the declared filter — a shortened prefix is a DIFFERENT
            # transform under the same factor name, so it fails closed (NaN)
            # until the full filter history is available (min_periods =
            # cutoff + 1 rows).
            if r < c:
                continue
            seg = colv[r - c : r + 1]  # x_{t-c}..x_t
            if not np.isfinite(seg).all():
                out[r, col] = np.nan
                continue
            rev = seg[::-1]  # x_t, x_{t-1}, ..., x_{t-c}
            out[r, col] = float(np.dot(rev, w))
    return out


@register_operator(
    name="ts_autocorrelation_time",
    category="memory",
    business_category="memory",
    canonical="ts_autocorrelation_time",
    source="memory_ext",
)
class TsAutocorrelationTime(SeriesOperator):
    """积分自相关时间（Geyer 初始单调序列估计量），单位 bars。

    使用 Geyer initial-monotone-sequence：``P_k=rho_(2k)+rho_(2k+1)``，
    取首个非正配对之前的正前缀，再做累积最小值，``tau=-1+2*sum(P)``。
    丢弃未配对的最后滞后；不把单个负 ACF 当成终止点。每个滞后使用
    标准有限样本分母 ``rho_k = gamma_k/gamma_0``（``gamma_k`` 以 N-k 归一），
    消除随 k 增长的有限样本偏差。输出**不除以 max_lag**（R4-56）：白噪声 →
    τ≈1 bar，与 max_lag 无关。大 -> 强序列依赖；小 -> 近白噪声。统计量在
    尾部连续有限窗口上计算（R4-57），跨 gap 不重连。P1。
    """

    metadata = _metadata(
        "ts_autocorrelation_time",
        "积分自相关时间（bars，Geyer 正配对前缀的单调化，不除以 max_lag）。",
        ["x", "window", "max_lag"],
        unit="bars",
        cost=3,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 120, max_lag: int = 20, **_: Any) -> pd.DataFrame:
        w = _check_window(window)
        ml = int(max_lag)
        # R5-39: lag truncation is capped by the window — ``max_lag >= window``
        # is a nonsensical search node (lags beyond the window are never
        # computed, so it manufactures duplicate factors).
        if ml < 1:
            raise ValueError("max_lag must be >= 1")
        if ml >= w:
            raise ValueError("max_lag must be < window")
        return frame_like(x, _autocorrelation_time_series(x.to_numpy(dtype=float), w, ml))


@register_operator(
    name="ts_autocorrelation_time_initial_positive_sequence",
    category="memory",
    business_category="memory",
    canonical="ts_autocorrelation_time_initial_positive_sequence",
    source="memory_ext",
)
class TsAutocorrelationTimeInitialPositive(SeriesOperator):
    """积分自相关时间（Geyer 初始正序列估计量），单位 bars。

    Audit #61: 标准 Geyer initial-positive-sequence 估计量
    ``P_k=rho_(2k)+rho_(2k+1), tau=-1+2*sum(P)``，在首个非正配对之前截断。
    单调版本另见 ``ts_autocorrelation_time``；本拼写为完整性/研究保留。
    Audit #60: 自相关使用标准有限样本分母（``rho_k = gamma_k/gamma_0``）。P1。
    """

    metadata = _metadata(
        "ts_autocorrelation_time_initial_positive_sequence",
        "Geyer 初始正配对序列积分自相关时间（bars，P_k>0 前缀）。",
        ["x", "window", "max_lag"],
        unit="bars",
        cost=3,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 120, max_lag: int = 20, **_: Any) -> pd.DataFrame:
        w = _check_window(window)
        ml = int(max_lag)
        if ml < 1:
            raise ValueError("max_lag must be >= 1")
        if ml >= w:
            raise ValueError("max_lag must be < window")
        return frame_like(x, _autocorrelation_time_series_ips(x.to_numpy(dtype=float), w, ml))


@register_operator(
    name="ts_fractional_difference",
    category="memory",
    business_category="memory",
    canonical="ts_fractional_difference",
    source="memory_ext",
)
class TsFractionalDifference(SeriesOperator):
    """分数差分变换 y_t = sum_k w_k x_{t-k}（二项权重递归，d∈(-1,1)）。

    d>0 移除长记忆（平稳化）；d<0 为分数积分。输出长度与输入一致；
    因果支持内任意 NaN -> NaN。R11 #41：启动区 ``r < cutoff``（分数滤波
    历史未收敛）为 NaN，直到完整 ``cutoff+1`` 个因果项可用。P1。

    Audit #62: cutoff 必须跟随 d —— ``max_discarded_weight_mass`` 门控检查固定
    cutoff 丢弃的理论 |权重| 质量占比；当 ``discarded_mass > tolerance`` 时整列
    FAIL CLOSED（NaN）。d<0（分数积分）权重 ``k^{-(1+d)}`` 缓慢衰减、总质量
    发散，任何固定 cutoff 都会丢弃 ~全部质量 -> 门控拒绝。诊断输出见
    ``ts_fractional_difference_discarded_weight_mass``。
    """

    metadata = _metadata(
        "ts_fractional_difference",
        "分数差分变换（二项权重，d∈(-1,1) 内），启动区 NaN，丢弃质量门控。",
        ["x", "fd", "cutoff", "max_discarded_weight_mass"],
        unit="series",
        cost=3,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        fd: float = 0.4,
        cutoff: int = 20,
        max_discarded_weight_mass: float = 0.5,
        **_: Any,
    ) -> pd.DataFrame:
        fdv = float(fd)
        if not (np.isfinite(fdv) and -1.0 < fdv < 1.0):
            raise ValueError("fd must satisfy -1 < fd < 1")
        if isinstance(cutoff, (bool, np.bool_)):
            raise ValueError("cutoff must be an integer, not bool")
        cf = float(cutoff)
        if not np.isfinite(cf) or cf != float(int(cf)):
            raise ValueError("cutoff must be an integer")
        c = int(cf)
        if c < 1:
            raise ValueError("cutoff must be >= 1")
        tol = float(max_discarded_weight_mass)
        if not np.isfinite(tol) or not (0.0 <= tol <= 1.0):
            raise ValueError("max_discarded_weight_mass must be in [0, 1]")
        # Audit #62: the cutoff must follow d.  A fixed cutoff that discards more
        # than the declared tolerance of the theoretical |weight| mass is a
        # DIFFERENT transform under the same factor name — fail closed (NaN).
        discarded = _fd_discarded_weight_mass(fdv, c)
        if discarded > tol:
            return frame_like(x, np.full(x.shape, np.nan, dtype=float))
        return frame_like(x, _fractional_difference_series(x.to_numpy(dtype=float), fdv, c))


@register_operator(
    name="ts_fractional_difference_discarded_weight_mass",
    category="memory",
    business_category="memory",
    canonical="ts_fractional_difference_discarded_weight_mass",
    source="memory_ext",
)
class TsFractionalDifferenceDiscardedWeightMass(SeriesOperator):
    """分数差分固定 cutoff 丢弃的理论 |权重| 质量占比（0..1，仅依赖 fd/cutoff）。

    Audit #62 诊断输出：``discarded = 1 - kept_mass / total_mass``。d<0（分数
    积分）权重 k^{-(1+d)} 缓慢衰减、总质量发散 -> discarded=1；d>0 收敛，按
    解析尾部近似。配合 ``ts_fractional_difference`` 的
    ``max_discarded_weight_mass`` 门控使用。P1。
    """

    metadata = _metadata(
        "ts_fractional_difference_discarded_weight_mass",
        "分数差分固定 cutoff 丢弃的理论权重质量占比。",
        ["x", "fd", "cutoff"],
        unit="ratio",
        cost=1,
    )

    def _calculate_series(self, x: pd.DataFrame, fd: float = 0.4, cutoff: int = 20, **_: Any) -> pd.DataFrame:
        fdv = float(fd)
        if not (np.isfinite(fdv) and -1.0 < fdv < 1.0):
            raise ValueError("fd must satisfy -1 < fd < 1")
        if isinstance(cutoff, (bool, np.bool_)):
            raise ValueError("cutoff must be an integer, not bool")
        cf = float(cutoff)
        if not np.isfinite(cf) or cf != float(int(cf)):
            raise ValueError("cutoff must be an integer")
        c = int(cf)
        if c < 1:
            raise ValueError("cutoff must be >= 1")
        d = _fd_discarded_weight_mass(fdv, c)
        return frame_like(x, np.full(x.shape, d, dtype=float))


_NEW_CANONICALS = (
    "ts_autocorrelation_time",
    "ts_autocorrelation_time_initial_positive_sequence",
    "ts_fractional_difference",
    "ts_fractional_difference_discarded_weight_mass",
)


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


def _register_precise_alias() -> None:
    """R4-56: expose the precise name ``ts_integrated_autocorrelation_time``.

    The canonical stays ``ts_autocorrelation_time`` (polars_geometry_math
    hard-codes that name in its polars-backend list, so a canonical rename is
    not loadable without editing that module).  The precise spelling is
    registered as an alias so new DSL expressions can use it.
    """
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    OperatorRegistry.register_alias(
        "ts_integrated_autocorrelation_time",
        "ts_autocorrelation_time",
    )


_register_surface()
_register_precise_alias()
