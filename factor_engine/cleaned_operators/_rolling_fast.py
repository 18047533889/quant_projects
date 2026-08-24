"""快速滚动算子：优先 Numba 列向量化，避免 ``rolling.apply`` Python 回调。

供 ``time_series.py`` 与 polars 桥接复用的高性能滚动内核。
通过环境变量 ``FACTOR_ENGINE_DISABLE_NUMBA`` 可禁用 Numba 加速。
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from factor_engine.backend.operator_errors import OperatorParameterError

_NUMBA_DISABLED = os.environ.get("FACTOR_ENGINE_DISABLE_NUMBA", "").lower() in (
    "1",
    "true",
    "yes",
)


def _strict_window_int(value: Any, name: str) -> int:
    """Strict integer gate for shared rolling kernels (R16-067).

    A shared kernel must receive only binder-validated integers — never silent
    ``int(x)`` truncation or ``max(2, ...)`` clamping, which let ``20.9`` and
    ``20`` manufacture two ASTs with identical output.  A bool / string /
    non-integral value is a contract violation and raises.
    """
    from factor_engine.cleaned_operators.base import strict_int_param

    return strict_int_param(value, name)


def _finite_pair_mask(y: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """Paired finite mask for y/x (R16-068).

    ``notna()`` treats ±Inf as an observed value, which corrupts paired stats
    (a window "full" of Inf pairs passes the min_periods gate and yields a
    garbage beta/correlation).  All paired statistics must use the finite
    mask so Inf is never counted as a valid observation.
    """
    return pd.DataFrame(
        np.isfinite(y.to_numpy(dtype=np.float64, copy=False))
        & np.isfinite(x.to_numpy(dtype=np.float64, copy=False)),
        index=y.index,
        columns=y.columns,
    )


def _get_linear_weighted_1d():
    """惰性加载 Numba 线性加权滚动内核。

返回:
    Numba JIT 函数或 ``None``（Numba 不可用或已禁用时）。
"""
    if _NUMBA_DISABLED:
        return None
    try:
        from numba import njit  # type: ignore
    except ImportError:
        return None

    @njit(cache=True)
    def _linear_weighted_1d(arr: np.ndarray, weights: np.ndarray) -> np.ndarray:
        n = arr.shape[0]
        wlen = weights.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            start = i - wlen + 1
            if start < 0:
                start = 0
            seg_len = i - start + 1
            s = 0.0
            ws = 0.0
            for k in range(seg_len):
                v = arr[start + k]
                if np.isfinite(v):
                    w = weights[wlen - seg_len + k]
                    s += v * w
                    ws += w
            out[i] = s / ws if ws > 0.0 else np.nan
        return out

    return _linear_weighted_1d


_linear_weighted_1d_jit = _get_linear_weighted_1d()


# R16-070: the WMA partial-warmup identity is an explicit, versioned policy.
# A partial window of length L < W uses the L NEWEST age slots (weights
# ``W-L+1 .. W``) renormalized to unit mass — ``RenormalizedPartialWMA``.  The
# JIT and numpy fallback below implement exactly this, so the two backends
# cannot drift.  A ``FixedAgeWMA`` variant (full W-weights, unrenormalized)
# would be a DIFFERENT definition and must be its own canonical if ever wanted.
WMA_PARTIAL_POLICY = "RenormalizedPartialWMA_age_slot_anchored"

# R40 #195: the WMA partial policy enters the operator contract / factor
# identity / backend-parity evidence.  ``wma_partial_policy_digest()`` folds the
# policy string (and the sharing AGE_WEIGHTED policy it aligns with) into a
# stable digest that the WMA operators expose on their metadata.  Changing the
# policy automatically invalidates old cache/materialization identities because
# the digest is part of the semantic contract.
_WMA_POLICY_DIGEST_NS = "wma_partial_policy.v1"


def wma_partial_policy_digest() -> str:
    """Stable digest of the WMA partial-warmup policy (R40 #195).

    Includes the WMA policy and the age-weighted sibling policy it is anchored
    to, so a change to either policy surfaces in factor/cache identity.
    """
    import hashlib as _hashlib

    payload = f"{_WMA_POLICY_DIGEST_NS}|{WMA_PARTIAL_POLICY}|{AGE_WEIGHTED_PARTIAL_POLICY}"
    return _hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def check_wma_partial_policy(policy: str | None) -> str:
    """Validate a WMA partial policy against the single source of truth.

    ``None`` (caller did not declare) resolves to the canonical policy — but in
    production the caller must have declared it (the operator contract carries
    ``partial_policy_digest``).  An unknown policy string is rejected rather
    than silently using a different warmup definition.
    """
    if policy is None:
        return WMA_PARTIAL_POLICY
    if policy != WMA_PARTIAL_POLICY:
        raise ValueError(
            f"unknown WMA partial policy {policy!r}; the canonical policy is "
            f"{WMA_PARTIAL_POLICY!r} (R40 #195 — a different warmup definition "
            "must be its own canonical, never a silent variant)"
        )
    return policy


def _linear_weighted_1d_numpy(arr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """纯 numpy 实现的线性加权滚动均值（Numba 回退）。

参数:
    arr: 一维 numpy 数组。
    weights: 线性权重数组（长度 = 窗口）。

返回:
    与 ``arr`` 等长的一维加权均值数组。

策略 (R16-070): 与 Numba 内核完全一致 —— partial 窗口（长度 L < W）
使用最近 L 个 age slot 的权重（``weights[-L:]``，即 W-L+1 .. W）并归一化。
"""
    n = arr.shape[0]
    wlen = len(weights)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - wlen + 1)
        seg = arr[start : i + 1]
        ww = weights[-len(seg) :]
        mask = np.isfinite(seg)
        if not mask.any():
            out[i] = np.nan
        else:
            s = seg[mask]
            w = ww[mask]
            out[i] = np.dot(s, w) / w.sum()
    return out


def rolling_linear_weighted(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """线性衰减加权滚动均值（``ts_decay_linear`` / ``WMA`` 语义）。

参数:
    x: 输入宽表 panel。
    window: 滚动窗口长度，``min_periods=1``。

返回:
    线性加权均值 panel。

R16-070: partial-warmup 身份已显式化 —— ``WMA_PARTIAL_POLICY =
"RenormalizedPartialWMA_age_slot_anchored"``：partial 窗口用最近 L 个 age
slot 权重并归一化，Numba 与 numpy 回退完全一致（identity 不再模糊）。
"""
    weights = np.arange(1, window + 1, dtype=np.float64)
    fn = _linear_weighted_1d_jit or _linear_weighted_1d_numpy
    arr = x.to_numpy(dtype=np.float64, copy=False)
    out = np.empty_like(arr, dtype=np.float64)
    for j in range(arr.shape[1]):
        out[:, j] = fn(arr[:, j], weights)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def rolling_regression(
    y: pd.DataFrame,
    x: pd.DataFrame,
    *,
    window: int,
    min_periods: int = 3,
    lag: int = 0,
    retval: str = "slope",
) -> pd.DataFrame:
    """滚动 OLS 回归，全 panel 向量化。

参数:
    y: 因变量宽表 panel。
    x: 自变量宽表 panel。
    window: 滚动窗口长度。
    min_periods: 最少有效样本数，默认 3。
    lag: 自变量滞后 bar 数，负值返回全 NaN。
    retval: 返回值类型，``slope`` / ``intercept`` / ``r_squared`` / ``residual``。

返回:
    指定回归统计量的 panel。

NEW-021: 每个窗口内所有统计量（cov/var/mean）必须基于**同一 paired
cohort** —— 先取 ``mask = finite(x) & finite(y)``，再把 x/y 同时 mask 后滚动。
否则 x/y 缺失位置不同时，``Cov(y,x)`` 用 paired support，而 ``Var(x)``/均值
用各自完整 support，得到的 beta/intercept 不是同一批样本上的 OLS。

NEW-023: 未知 ``retval`` 直接 ``ValueError``，绝不静默回退到 ``slope``。
NEW-022: ``residual`` 保留为"当前窗口同 cohort 样本内残差的滚动均值"仅作
诊断用；生产 alpha 请用 ``ts_regression_forecast_error``（拟合严格截止 t-1）。
"""
    if lag < 0:
        from factor_engine.backend.operator_errors import FutureReferenceError

        raise FutureReferenceError(f"rolling_regression: negative lag {lag} references future data")
    if lag > 0:
        x = x.shift(lag)
    # NEW-021: single paired cohort for ALL statistics.
    # R16-068: finite mask — ±Inf is not a valid paired observation.
    valid = _finite_pair_mask(y, x)
    y_m = y.where(valid)
    x_m = x.where(valid)
    r_cov = y_m.rolling(window=window, min_periods=min_periods).cov(x_m)
    r_var_x = x_m.rolling(window=window, min_periods=min_periods).var()
    r_mean_y = y_m.rolling(window=window, min_periods=min_periods).mean()
    r_mean_x = x_m.rolling(window=window, min_periods=min_periods).mean()

    slope = r_cov / r_var_x.replace(0, np.nan)
    if retval == "slope":
        return slope

    intercept = r_mean_y - slope * r_mean_x
    if retval == "intercept":
        return intercept

    if retval == "r_squared":
        # Same-cohort correlation on the paired mask.
        corr = y_m.rolling(window=window, min_periods=min_periods).corr(x_m)
        return corr * corr

    if retval == "residual":
        # NEW-022: current in-sample residual on the SAME paired cohort (not a
        # second rolling mean of a mixed sequence).
        resid = y - (slope * x + intercept)
        return resid.where(valid)

    raise ValueError(
        f"rolling_regression: unknown retval {retval!r} (expected one of "
        "'slope', 'intercept', 'r_squared', 'residual')"
    )


def rolling_time_slope(x: pd.DataFrame, window: int, min_periods: int = 2) -> pd.DataFrame:
    """滚动对时间索引的 OLS 斜率（``Slope(close, n)`` 语义）。

NEW-020: 对每个窗口，用**有效样本的实际物理行偏移**做 OLS：
``t_valid = actual row offsets of finite values``，绝不用 ``0,1,2,...`` 的
假连续时间，也绝不删掉 NaN 后把前后两段重新当作相邻样本。

参数:
    x: 输入宽表 panel。
    window: 滚动窗口长度。
    min_periods: 最少有效样本数（OLS 至少 2）。

返回:
    时间趋势斜率 panel。
"""
    # R16-067: strict binder-validated ints — never ``max(2, int(...))`` silent
    # cast/clamp.  A sub-2 window/min_periods naturally fails closed (no window
    # reaches the OLS minimum), which is the correct contract.
    w = _strict_window_int(window, "window")
    mp = _strict_window_int(min_periods, "min_periods")
    arr = x.to_numpy(dtype=np.float64, copy=False)
    n, m = arr.shape
    out = np.full((n, m), np.nan, dtype=np.float64)
    for j in range(m):
        col = arr[:, j]
        for i in range(n):
            start = max(0, i - w + 1)
            seg = col[start : i + 1]
            valid = np.isfinite(seg)
            if int(valid.sum()) < mp:
                continue
            t = np.arange(start, i + 1, dtype=np.float64)[valid]
            v = seg[valid]
            tc = t - t.mean()
            denom = float(np.dot(tc, tc))
            if denom > 0.0:
                out[i, j] = float(np.dot(tc, v - v.mean()) / denom)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def _argextreme_1d(col: np.ndarray, window: int, *, maximum: bool, age: bool) -> np.ndarray:
    """1-D rolling extreme position/age with ``latest`` tie-break.

    ``index_from_oldest`` (``age=False``): 0-based offset from the window's
    OLDEST bar (0 = oldest).  ``age`` (``age=True``): bars since the most recent
    extreme (0 = current bar).  Ties always resolve to the NEWEST occurrence
    (``hits[-1]``) — R19-039..042/049 tie authority, identical across backends.
    A window with no finite value outputs NaN (Unknown).
    """
    out = np.full(col.shape[0], np.nan, dtype=np.float64)
    for i in range(col.shape[0]):
        start = max(0, i - window + 1)
        seg = col[start : i + 1]
        valid = np.isfinite(seg)
        if not valid.any():
            continue
        extreme = float(np.max(seg[valid])) if maximum else float(np.min(seg[valid]))
        hits = np.flatnonzero(valid & (seg == extreme))
        hit = int(hits[-1])
        out[i] = float((seg.size - 1 - hit) if age else hit)
    return out


def rolling_argmax(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """滚动窗口内最大值的位置（index_from_oldest，0=窗口最旧 bar，tie=latest）。

NEW-018: 全窗口无有限值时输出 ``NaN``（Unknown 不能伪装成合法极值位置）。
NEW-019: 目标极值只在 ``np.isfinite`` 的样本上求，``+Inf`` 既不作 target 也不
作 hits —— ``[finite, +Inf]`` 时不再出现 target/hits 不一致。

参数:
    x: 输入宽表 panel。
    window: 滚动窗口长度。

返回:
    极大值位置 panel。
"""
    return _apply_colwise_1d(
        x, lambda col: _argextreme_1d(col, window, maximum=True, age=False)
    )


def rolling_argmin(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """滚动窗口内最小值的位置（index_from_oldest，0=窗口最旧 bar，tie=latest）。

NEW-018/019: 同 :func:`rolling_argmax` —— 全 NaN 窗口输出 NaN；极值只在
有限样本上求。

参数:
    x: 输入宽表 panel。
    window: 滚动窗口长度。

返回:
    极小值位置 panel。
"""
    return _apply_colwise_1d(
        x, lambda col: _argextreme_1d(col, window, maximum=False, age=False)
    )


def rolling_days_since_extreme(x: pd.DataFrame, window: int, *, maximum: bool) -> pd.DataFrame:
    """滚动极值距当前 bar 的 bar 数（age，0=当前 bar，tie=latest）。

R19-039..042/049: ``ts_argmax`` / ``ts_argmin`` canonical 语义为 **age**
（0=当前/最新 bar），并列取最新 occurrence。index-from-oldest 语义由
:func:`rolling_argmax` / :func:`rolling_argmin`（及 canonical
``ts_argmax_index_from_oldest`` / ``ts_argmin_index_from_oldest``）提供。

参数:
    x: 输入宽表 panel。
    window: 滚动窗口长度。
    maximum: ``True`` 取最大极值（argmax），``False`` 取最小极值（argmin）。

返回:
    极值 age panel。
"""
    return _apply_colwise_1d(
        x, lambda col: _argextreme_1d(col, window, maximum=maximum, age=True)
    )


# R19-030: current-row observation policy for paired window statistics.
#
# A rolling correlation/covariance at row ``t`` is defined over the window's
# valid x/y pairs.  The chosen policy: the statistic MAY EXIST at a row where
# the current x/y pair is missing (NaN/Inf), as long as the window still holds
# >= min_periods finite pairs.  This matches the Numba fastpath
# (``backend.numba_kernels.rolling_corr_panel``) which skips non-finite pairs
# and only NaNs when fewer than ``min_count`` pairs are present.  The old pandas
# slow path forced ``current-row missing -> NaN`` via ``.where(valid)`` — a
# contradiction that manufactured cross-backend drift.  All backends now share
# the fastpath policy.
CURRENT_ROW_POLICY = "window_statistic_can_exist_without_current_observation"
CurrentObservationRole = CURRENT_ROW_POLICY


def rolling_signed_product(
    x: pd.DataFrame,
    window: int,
    min_periods: int = 1,
) -> pd.DataFrame:
    """Signed + zero-safe rolling product (R19-043/044).

    The old log-domain implementation replaced ``0`` with NaN (``[2,0,3]`` -> NaN
    instead of ``0``) and produced NaN for any negative input (``[-2,-3]`` ->
    NaN instead of ``6``).  The signed product maintains ``zero_count`` /
    ``sign_parity`` / ``sum_log_abs`` over the causal window:
      ``[2,0,3]``  -> ``0``
      ``[-2,-3]``  -> ``6``
      ``[-2,3]``   -> ``-6``
    ``±Inf`` and ``NaN`` are missing (skipped); an all-missing window is NaN.
    """
    w = _strict_window_int(window, "window")
    mp = _strict_window_int(min_periods, "min_periods")
    if mp > w:
        raise OperatorParameterError("min_periods must be <= window")
    finite = pd.DataFrame(
        np.isfinite(x.to_numpy(dtype=np.float64, copy=False)),
        index=x.index,
        columns=x.columns,
    )
    clean = x.where(finite)
    count = finite.rolling(w, min_periods=1).sum()
    zero_count = clean.eq(0).rolling(w, min_periods=1).sum()
    negative_count = clean.lt(0).rolling(w, min_periods=1).sum()
    log_abs = np.log(clean.abs().where(clean.ne(0)))
    with np.errstate(over="ignore", invalid="ignore"):
        magnitude = np.exp(log_abs.rolling(w, min_periods=1).sum())
    sign = pd.DataFrame(
        np.where((negative_count % 2).eq(0), 1.0, -1.0),
        index=x.index,
        columns=x.columns,
    )
    result = (magnitude * sign).mask(zero_count.gt(0), 0.0)
    # Fail closed on overflow: a product exceeding float64 (1e308*1e308) is not
    # a usable alpha -> NaN, never ±Inf.
    result = result.mask(~np.isfinite(result.to_numpy(dtype=np.float64, copy=False)))
    return result.where(count >= mp)


# R19-045/046: explicit partial-window / missing policy for ALL age-weighted
# operators (``ts_sum_decay``, ``ts_decay_exp_window``).  Weights are indexed
# oldest_to_newest (index 0 = oldest bar).  A partial window of length L < W
# uses the L NEWEST age slots (``weights[-L:]``) renormalized to unit mass —
# identical to the WMA ``RenormalizedPartialWMA_age_slot_anchored`` policy.
# Missing values (NaN/Inf) are skipped and the surviving weights renormalized,
# so one missing bar never collapses the whole window.
AGE_WEIGHTED_PARTIAL_POLICY = "RenormalizedPartialAgeWeighted_newest_slot_anchored"


def _age_weighted_1d_numpy(arr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n = arr.shape[0]
    wlen = weights.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - wlen + 1)
        seg = arr[start : i + 1]
        ww = weights[-len(seg) :]  # newest-L age slots (partial-window anchor)
        mask = np.isfinite(seg)
        if not mask.any():
            out[i] = np.nan
        else:
            s = seg[mask]
            w = ww[mask]
            out[i] = np.dot(s, w) / w.sum()
    return out


def rolling_age_weighted(
    x: pd.DataFrame, window: int, weights: np.ndarray
) -> pd.DataFrame:
    """Shared age-weighted rolling mean kernel (R19-045..048).

    ``weights`` is the FULL-window weight vector in oldest_to_newest order
    (index 0 = oldest bar); it is NOT required to be normalized — each window
    renormalizes internally.  Partial windows use ``weights[-L:]``
    (newest-L age slots) and missing values are skipped + reweighted
    (see :data:`AGE_WEIGHTED_PARTIAL_POLICY`).
    """
    w = _strict_window_int(window, "window")
    wts = np.asarray(weights, dtype=np.float64)
    if wts.ndim != 1 or wts.shape[0] != w:
        raise OperatorParameterError(
            f"rolling_age_weighted: weights must be a 1-D array of length {w}, "
            f"got shape {wts.shape}"
        )
    return _apply_colwise_1d(x, lambda col: _age_weighted_1d_numpy(col, wts))


def rolling_beta(
    y: pd.DataFrame,
    x: pd.DataFrame,
    *,
    window: int,
    min_periods: int = 5,
) -> pd.DataFrame:
    """滚动 Beta：``Cov(y,x) / Var(x)``。

NEW-024: 生产默认 ``min_periods=5`` —— 2 个点的斜率在统计上没有意义。
参数:
    y: 因变量宽表 panel。
    x: 自变量宽表 panel。
    window: 滚动窗口长度。
    min_periods: 最少有效配对样本数，默认 5。

返回:
    Beta 系数 panel。
"""
    # R16-068: finite mask — ±Inf is not a valid paired observation.
    valid = _finite_pair_mask(y, x)
    y_m = y.where(valid)
    x_m = x.where(valid)
    cov = y_m.rolling(window=window, min_periods=min_periods).cov(x_m)
    var = x_m.rolling(window=window, min_periods=min_periods).var()
    return (cov / var.replace(0, np.nan)).where(valid)


def _rolling_top_bottom_1d_numpy(
    arr: np.ndarray,
    window: int,
    k: int,
    *,
    top: bool,
    stat: str,
) -> np.ndarray:
    """一维窗口内 top/bottom-k 的 mean/sum/std（因果窗口）。

参数:
    arr: 一维 numpy 数组。
    window: 滚动窗口长度。
    k: 取 top/bottom 的个数。
    top: ``True`` 取最大 k 个，``False`` 取最小 k 个。
    stat: 聚合方式，``mean`` / ``sum`` / ``std``。

返回:
    与 ``arr`` 等长的一维结果数组。
"""
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - window + 1)
        seg = arr[start : i + 1]
        valid = seg[np.isfinite(seg)]
        # NEW-025: valid < k -> NaN (Unknown, not "the best few we happen to
        # have").  Same topk_5 must mean top-5 everywhere, never top-2/3.
        if valid.size < k or k < 1:
            out[i] = np.nan
            continue
        ordered = np.sort(valid)
        picked = ordered[-k:] if top else ordered[:k]
        if stat == "mean":
            out[i] = picked.mean()
        elif stat == "sum":
            out[i] = picked.sum()
        elif stat == "std":
            out[i] = picked.std(ddof=1) if picked.size >= 2 else np.nan
        else:
            out[i] = np.nan
    return out


def _cum_top_bottom_1d_numpy(
    arr: np.ndarray,
    k: int,
    *,
    top: bool,
    stat: str,
) -> np.ndarray:
    """一维扩展窗口内 top/bottom-k 的 mean/sum。

参数:
    arr: 一维 numpy 数组。
    k: 取 top/bottom 的个数。
    top: ``True`` 取最大 k 个，``False`` 取最小 k 个。
    stat: 聚合方式，``mean`` 或 ``sum``。

返回:
    与 ``arr`` 等长的一维结果数组。
"""
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        seg = arr[: i + 1]
        valid = seg[np.isfinite(seg)]
        # R16-069: fixed-k support, consistent with the rolling top/bottom-k
        # NEW-025 fail-closed policy.  ``take = min(k, valid.size)`` silently
        # downgraded ``top5`` to ``top2`` when only 2 samples were available —
        # top-5 must mean top-5 everywhere, never "the best few we happen to
        # have".  If available < k the output is NaN (Unknown).
        if valid.size < k or k < 1:
            out[i] = np.nan
            continue
        ordered = np.sort(valid)
        picked = ordered[-k:] if top else ordered[:k]
        out[i] = picked.mean() if stat == "mean" else picked.sum()
    return out


def _apply_colwise_1d(x: pd.DataFrame, fn) -> pd.DataFrame:
    """对 panel 每列应用一维 numpy 内核函数。

参数:
    x: 输入宽表 panel。
    fn: 接收一维数组返回一维结果的函数。

返回:
    逐列计算后的 panel。
"""
    arr = x.to_numpy(dtype=np.float64, copy=False)
    out = np.empty_like(arr, dtype=np.float64)
    for j in range(arr.shape[1]):
        out[:, j] = fn(arr[:, j])
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def rolling_top_n_mean(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """滚动窗口内前 N 大值的均值。

参数:
    x: 输入宽表 panel。
    n: 取最大的 N 个值。

返回:
    top-N 均值 panel。
"""
    return _apply_colwise_1d(
        x, lambda col: _rolling_top_bottom_1d_numpy(col, n, n, top=True, stat="mean")
    )


def rolling_top_n_sum(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """滚动窗口内前 N 大值的求和。

参数:
    x: 输入宽表 panel。
    n: 取最大的 N 个值。

返回:
    top-N 求和 panel。
"""
    return _apply_colwise_1d(
        x, lambda col: _rolling_top_bottom_1d_numpy(col, n, n, top=True, stat="sum")
    )


def rolling_top_n_std(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """滚动窗口内前 N 大值的标准差。

参数:
    x: 输入宽表 panel。
    n: 取最大的 N 个值。

返回:
    top-N 标准差 panel。
"""
    return _apply_colwise_1d(
        x, lambda col: _rolling_top_bottom_1d_numpy(col, n, n, top=True, stat="std")
    )


def rolling_bottom_n_mean(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """滚动窗口内后 N 小值的均值。

参数:
    x: 输入宽表 panel。
    n: 取最小的 N 个值。

返回:
    bottom-N 均值 panel。
"""
    return _apply_colwise_1d(
        x, lambda col: _rolling_top_bottom_1d_numpy(col, n, n, top=False, stat="mean")
    )


def rolling_bottom_n_sum(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """滚动窗口内后 N 小值的求和。

参数:
    x: 输入宽表 panel。
    n: 取最小的 N 个值。

返回:
    bottom-N 求和 panel。
"""
    return _apply_colwise_1d(
        x, lambda col: _rolling_top_bottom_1d_numpy(col, n, n, top=False, stat="sum")
    )


def cum_top_n_mean(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """扩展窗口内前 N 大值的均值。

参数:
    x: 输入宽表 panel。
    n: 取最大的 N 个值。

返回:
    累积 top-N 均值 panel。
"""
    return _apply_colwise_1d(
        x, lambda col: _cum_top_bottom_1d_numpy(col, n, top=True, stat="mean")
    )


def cum_top_n_sum(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """扩展窗口内前 N 大值的求和。

参数:
    x: 输入宽表 panel。
    n: 取最大的 N 个值。

返回:
    累积 top-N 求和 panel。
"""
    return _apply_colwise_1d(
        x, lambda col: _cum_top_bottom_1d_numpy(col, n, top=True, stat="sum")
    )


def rolling_top_n_mean_window(x: pd.DataFrame, window: int, k: int) -> pd.DataFrame:
    """指定窗口内前 k 大值的均值。

参数:
    x: 输入宽表 panel。
    window: 滚动窗口长度。
    k: 取最大的 k 个值。

返回:
    窗口 top-k 均值 panel。
"""
    return _apply_colwise_1d(
        x, lambda col: _rolling_top_bottom_1d_numpy(col, window, k, top=True, stat="mean")
    )


def rolling_top_n_sum_window(x: pd.DataFrame, window: int, k: int) -> pd.DataFrame:
    """指定窗口内前 k 大值的求和。

参数:
    x: 输入宽表 panel。
    window: 滚动窗口长度。
    k: 取最大的 k 个值。

返回:
    窗口 top-k 求和 panel。
"""
    return _apply_colwise_1d(
        x, lambda col: _rolling_top_bottom_1d_numpy(col, window, k, top=True, stat="sum")
    )
