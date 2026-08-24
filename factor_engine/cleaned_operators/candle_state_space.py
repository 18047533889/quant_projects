# -*- coding: utf-8 -*-
"""Candle state-space operators (2026-08 geometry/math expansion).

A candle is reduced to a 4-dimensional *state vector* per row per symbol (e.g.
body ratio, wick ratios, range/ATR), so a run of candles becomes a point
trajectory through feature space.  This module characterises how the *current
candle as an object* sits inside its own history:

* ``ts_vector_state_mahalanobis``          — anomaly of today's state vector vs
  the trailing covariance of its own history (shrinkage-regularised
  Mahalanobis distance).  High = today's candle is atypical for the symbol.
  Reference/query split (M-140): the reference is strictly ``<= t-1`` and the
  query row is excluded from its own reference.
* ``ts_vector_state_local_density``        — local density of today's state
  among its prior neighbours (inverse k-th nearest distance).  High = common
  state, low = rare / isolated state.  Reference/query split (M-140): the
  reference is strictly ``<= t-1`` and the query row is excluded from its own
  reference.
* ``ts_multivariate_matrix_profile_novelty`` — distance of the current trailing
  subsequence of the caller-supplied scalar ``x`` to its nearest *prior*
  subsequence under z-normalised Euclidean distance (novelty).
* ``ts_matrix_profile_motif_age``          — normalised age of the nearest
  historical match of the current subsequence (same matrix-profile kernel).

Shared kernels are private to this module.  All operators are trailing-window,
prefix-causal (row ``r`` uses rows ``<= r`` only) and deterministic; NaN inputs
are dropped from the window, and a window with no valid rows emits NaN.  Every
operator is per-column (each symbol processed independently), so one symbol's
state never leaks into another's.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    RelationalParamSpec,
    SeriesOperator,
    register_operator,
)
from factor_engine.cleaned_operators.rolling_pack import frame_like, register_polars_bridge

_EPS = 1e-12

# Model-audit Phase 4 (search-space hygiene): ``window`` and ``history`` are the
# trailing lookback band (the alpha horizon, HORIZON, searched);
# ``subsequence_length`` is the matrix-profile subsequence length — an
# estimator-resolution grid knob, never a full-resolution search dimension
# (M-115/M-162/M-170).
_MATRIX_PROFILE_PARAM_SPECS: dict[str, ParamSpec] = {
    "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON, searchable=True),
    "subsequence_length": ParamSpec(dtype=int, min=3, param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False),
    "history": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON, searchable=True),
}

# M-11xx: matrix-profile relational feasibility — a combination that is
# syntactically legal yet mathematically guaranteed to produce an empty search
# band (all-NaN) is rejected at binding:
#   * the history band must be able to hold at least one prior subsequence:
#     ``subsequence_length <= history``;
#   * ``window`` caps the band: ``history <= window``;
#   * the exclusion zone (``subsequence_length // 4``) plus the current
#     subsequence must still leave a non-empty historical band:
#     ``history >= subsequence_length + subsequence_length // 4``.
_MATRIX_PROFILE_RELATIONAL_SPECS: list[Any] = [
    RelationalParamSpec(
        "subsequence_length <= history",
        "ts_matrix_profile_* requires subsequence_length <= history "
        "(history must hold at least one prior subsequence; subsequence_length="
        "{subsequence_length}, history={history})",
    ),
    RelationalParamSpec(
        "history <= window",
        "ts_matrix_profile_* requires history <= window "
        "(window={window}, history={history})",
    ),
    RelationalParamSpec(
        "history >= subsequence_length + subsequence_length // 4",
        "ts_matrix_profile_* requires history >= subsequence_length + "
        "subsequence_length//4 (the exclusion zone must leave a non-empty "
        "historical band; history={history}, subsequence_length={subsequence_length})",
    ),
]

#: M-142: Mahalanobis fit-quality telemetry.  Populated on every (col, row) fit
#: attempt inside ``_mahalanobis_series``; the module-level
#: ``last_mahalanobis_telemetry()`` accessor exposes the most recent snapshot so
#: audit probes can see WHY a cell was NaN (mirroring
#: ``cross_section.panel_model.last_fit_telemetry`` /
#: ``ts_model.state_space.numba_dispatch_stats``).  Diagnostic only — no operator
#: surface is registered from it.
_LAST_MAHALANOBIS_TELEMETRY: dict[str, Any] = {
    "p_effective": None,
    "N_effective": None,
    "condition_number": None,
    "dropped_constant_dims": None,
    "failure_reason": "not_run",
}


def last_mahalanobis_telemetry() -> dict[str, Any]:
    """Fit-quality telemetry of the most recent Mahalanobis covariance fit.

    Keys: ``p_effective`` (kept feature dims after dropping constant features),
    ``N_effective`` (valid history rows used in the fit), ``condition_number``
    (of the shrunk covariance), ``dropped_constant_dims`` and ``failure_reason``
    (one of ``ok`` | ``no_finite_history`` | ``all_constant`` |
    ``insufficient_sample``; cells skipped because the current query row is
    incomplete do NOT overwrite the last genuine fit decision).  Returns a
    defensive copy.
    """
    return dict(_LAST_MAHALANOBIS_TELEMETRY)


def _set_mahalanobis_telemetry(
    *,
    p_effective: int | None,
    N_effective: int | None,
    condition_number: float | None,
    dropped_constant_dims: int | None,
    failure_reason: str,
) -> None:
    _LAST_MAHALANOBIS_TELEMETRY["p_effective"] = p_effective
    _LAST_MAHALANOBIS_TELEMETRY["N_effective"] = N_effective
    _LAST_MAHALANOBIS_TELEMETRY["condition_number"] = condition_number
    _LAST_MAHALANOBIS_TELEMETRY["dropped_constant_dims"] = dropped_constant_dims
    _LAST_MAHALANOBIS_TELEMETRY["failure_reason"] = failure_reason


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int,
              param_specs: dict[str, ParamSpec] | None = None,
              relational_specs: list[Any] | None = None) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="candle_state_space",
        description=description,
        param_names=params,
        return_type="series",
        param_specs={k: v for k, v in (param_specs or {}).items() if k in params},
        relational_specs=list(relational_specs) if relational_specs else [],
        tags=[
            "candle_state_space", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:vector_state",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


# ---------------------------------------------------------------------------
# kernels (per column; 2D panel = TradeDate x Symbol)
# ---------------------------------------------------------------------------
def _mahalanobis_series(
    f1: np.ndarray, f2: np.ndarray, f3: np.ndarray, f4: np.ndarray,
    window: int, shrinkage: float,
) -> np.ndarray:
    rows, cols = f1.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    lam = float(shrinkage)
    if not (0.0 <= lam <= 1.0):
        raise ValueError("shrinkage must be in [0, 1]")
    if w < 2:
        raise ValueError("window must be >= 2")
    feat = (f1, f2, f3, f4)
    p = len(feat)
    # M-142: reset per kernel call so a fresh call starts at "not_run" and the
    # accessor reflects THIS run's most recent (col, row) fit attempt.
    _set_mahalanobis_telemetry(
        p_effective=None, N_effective=None, condition_number=None,
        dropped_constant_dims=None, failure_reason="not_run",
    )
    for c in range(cols):
        for r in range(rows):
            i0 = max(0, r - w + 1)
            # R6-145: the mean/covariance are estimated on the HISTORY
            # [i0, r) only; the current row is the query, never part of its own
            # reference.  Including the current vector pulled the anomaly back
            # toward its own value.
            hist = np.stack([feat[j][i0:r, c] for j in range(p)], axis=1)  # (n, p)
            cur = np.stack([feat[j][r, c] for j in range(p)])  # (p,)
            if not np.all(np.isfinite(cur)):
                # current state vector must be complete; this is a data-level
                # skip, NOT a fit attempt — leave the telemetry on the last
                # genuine fit decision (M-142).
                continue
            finite = np.all(np.isfinite(hist), axis=1)
            valid = hist[finite].astype(float)
            if valid.shape[0] == 0:
                _set_mahalanobis_telemetry(
                    p_effective=None, N_effective=None, condition_number=None,
                    dropped_constant_dims=None, failure_reason="no_finite_history",
                )
                continue
            z = cur.astype(float)
            # R6-147: a feature constant over the history has zero variance and
            # a meaningless covariance column — standardising it with sd=1 makes
            # the distance depend on the feature's raw unit.  Drop constant
            # dimensions from both the history and the query before estimating.
            keep_cols = np.where(np.std(valid, axis=0) > _EPS)[0]
            dropped = p - int(keep_cols.size)
            N_eff = int(valid.shape[0])
            p_eff = int(keep_cols.size)
            if keep_cols.size == 0:
                _set_mahalanobis_telemetry(
                    p_effective=0, N_effective=N_eff, condition_number=None,
                    dropped_constant_dims=dropped, failure_reason="all_constant",
                )
                continue
            # R11 round-3 #66: high-dimensional sample floor.  Estimating a
            # p x p covariance (then pseudo-inverting it) from only a handful of
            # observations is meaningless; require N >= 5p effective observations
            # after dropping constant dimensions.
            if valid.shape[0] < 5 * keep_cols.size:
                _set_mahalanobis_telemetry(
                    p_effective=p_eff, N_effective=N_eff, condition_number=None,
                    dropped_constant_dims=dropped,
                    failure_reason="insufficient_sample",
                )
                continue
            valid = valid[:, keep_cols]
            z = z[keep_cols]
            # R11 round-3 #65: estimator consistency — a Mahalanobis distance
            # must pair ONE estimator family.  The old code centred with the
            # robust median but used the classical covariance (a half-robust /
            # half-classical mix that is inconsistent).  Use the classical
            # mean + ordinary covariance pair (a full robust centre + MCD
            # covariance is out of scope here).
            mu = valid.mean(axis=0)
            cov = np.atleast_2d(np.cov(valid, rowvar=False, ddof=1))
            cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
            shrunk = (1.0 - lam) * cov + lam * np.diag(np.diag(cov))
            try:
                cond = float(np.linalg.cond(shrunk))
            except (np.linalg.LinAlgError, ValueError):
                cond = float("inf")
            prec = np.linalg.pinv(shrunk + _EPS * np.eye(keep_cols.size))
            d = z - mu
            D = float(np.sqrt(max(0.0, float(d @ prec @ d))))
            out[r, c] = D
            _set_mahalanobis_telemetry(
                p_effective=p_eff, N_effective=N_eff, condition_number=cond,
                dropped_constant_dims=dropped, failure_reason="ok",
            )
    return out


def _local_density_series(
    f1: np.ndarray, f2: np.ndarray, f3: np.ndarray, f4: np.ndarray,
    window: int, k: int,
) -> np.ndarray:
    rows, cols = f1.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    kk = int(k)
    if w < 2:
        raise ValueError("window must be >= 2")
    if kk < 1:
        raise ValueError("k must be >= 1")
    feat = (f1, f2, f3, f4)
    p = len(feat)
    for c in range(cols):
        for r in range(rows):
            i0 = max(0, r - w + 1)
            # R6-145: scale (mean/std) is estimated on the HISTORY [i0, r) only;
            # the current row is the query and must not pull the density back
            # toward its own value.
            hist = np.stack([feat[j][i0:r, c] for j in range(p)], axis=1)  # (n, p)
            cur = np.stack([feat[j][r, c] for j in range(p)])  # (p,)
            if not np.all(np.isfinite(cur)):
                continue
            finite = np.all(np.isfinite(hist), axis=1)
            valid = hist[finite].astype(float)
            if valid.shape[0] < kk:
                continue  # need k prior valid rows
            # R6-147: a feature constant over the history would otherwise be
            # standardised with sd=1 and its distance would depend on the raw
            # unit.  Drop constant dimensions from history and query.
            keep_cols = np.where(np.std(valid, axis=0) > _EPS)[0]
            if keep_cols.size == 0:
                continue
            # R11 round-3 #66: high-dimensional sample floor for the KNN read.
            # A k-th-nearest distance in a p-dimensional feature space from too
            # few points is not a comparable density; require N >= 5p effective
            # observations after dropping constant dimensions.
            if valid.shape[0] < 5 * keep_cols.size:
                continue
            valid = valid[:, keep_cols]
            curv = cur[keep_cols]
            mu = valid.mean(axis=0)
            sd = valid.std(axis=0)
            sd = np.where(sd > _EPS, sd, 1.0)
            z = (valid - mu) / sd
            zc = (curv - mu) / sd
            dists = np.linalg.norm(z - zc, axis=1)
            rk = float(np.partition(dists, kk - 1)[kk - 1])  # k-th nearest distance
            # R6-146: an exact duplicate state (r_k == 0, common in A-share
            # 一字板 / repeated discrete states) would give density = 1/EPS —
            # an unbounded spike.  Output log-density (bounded for any r_k >= 0)
            # so a repeated state is a large-but-finite value, not an overflow
            # outlier.
            out[r, c] = -float(np.log(max(rk, 1e-3)))
    return out


def _z_normalize(sub: np.ndarray) -> np.ndarray:
    mu = sub.mean()
    sd = sub.std()
    if sd <= _EPS:
        # R11 round-3 #64: a constant subsequence z-normalises to all zeros
        # regardless of its level, so [10,10,10,10] and [100,100,100,100] would
        # collide into an identical "flat" pattern.  For shape-motif semantics a
        # constant pattern has NO defined shape — return NaN so the caller skips
        # it instead of matching flat segments of different levels to each other.
        return np.full_like(sub, np.nan, dtype=float)
    return (sub - mu) / sd


def _matrix_profile_series(
    x2d: np.ndarray, window: int, subsequence_length: int, history: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared matrix-profile kernel -> (novelty, motif_age, frequency, dispersion).

    For every row ``t`` the current trailing subsequence is ``x[t-L+1..t]`` and
    the historical search band for prior-subsequence start indices is
    ``[t-history .. t-L]`` (clamped to the data).  ``window`` is validated as an
    upper bound (``window >= history``) and also clamps the band, which is
    redundant once ``window >= history`` holds — it keeps the parameter an
    explicit part of the kernel contract.

    Audit P0 (matrix-profile rework):
    * an exclusion zone of ``L//4`` (>= m/4) keeps trivial self-matches out of
      the band;
    * the current pattern AND every candidate must be contiguous-finite — a gap
      never bridges two patterns (no NaN compression);
    * ``window``/``history`` are explicit parameters — there is no invisible
      hard-coded tail.
    """
    rows, cols = x2d.shape
    novelty = np.full((rows, cols), np.nan, dtype=float)
    age = np.full((rows, cols), np.nan, dtype=float)
    frequency = np.full((rows, cols), np.nan, dtype=float)
    dispersion = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    L = int(subsequence_length)
    h = int(history)
    if L < 2:
        raise ValueError("subsequence_length must be >= 2")
    if h < L:
        raise ValueError("history must be >= subsequence_length")
    if w < h:
        raise ValueError("window must be >= history")
    ez = max(0, int(L // 4))  # matrix-profile exclusion zone (>= m/4)
    for c in range(cols):
        x = x2d[:, c]
        for r in range(rows):
            if r + 1 < L:
                continue  # current subsequence not complete yet
            z = x[r - L + 1 : r + 1]
            if not np.all(np.isfinite(z)):
                continue
            s0 = max(0, r - h, r - w)
            s1 = r - L - ez  # strictly prior + exclusion zone
            if s0 > s1:
                continue
            zz = _z_normalize(z)
            if not np.all(np.isfinite(zz)):
                # R11 round-3 #64: the current subsequence is constant -> its
                # z-normalised shape is degenerate (all zeros) and no defined
                # shape distance exists -> NaN for this row.
                continue
            best_d = np.inf
            best_s = -1
            dists: list[float] = []
            # R6-149: use the RMS z-normalised Euclidean distance d/√L so
            # subsequence_length=20 and =80 patterns of the same strength are
            # comparable — the raw Euclidean distance grows ~√L with the window
            # length, making cross-parameter novelty values incomparable.
            inv_sqrt_L = 1.0 / float(np.sqrt(L))
            for s in range(s0, s1 + 1):
                cand = x[s : s + L]
                if not np.all(np.isfinite(cand)):
                    continue
                cz = _z_normalize(cand)
                if not np.all(np.isfinite(cz)):
                    # R11 round-3 #64: a constant historical candidate has no
                    # shape; it must not match the current pattern.
                    continue
                d = float(np.linalg.norm(zz - cz) * inv_sqrt_L)
                dists.append(d)
                if d < best_d:
                    best_d = d
                    best_s = s
            if not dists:
                continue
            novelty[r, c] = best_d
            # R11 round-3 #63: the age of the motif is the START-to-START age.
            # The current subsequence starts at ``r - L + 1`` (its end is ``r``),
            # and ``best_s`` is the historical best-match START.  The old
            # ``r - best_s`` counted end-to-start, systematically L-1 bars too
            # old.  True age = (r - L + 1) - best_s.
            age[r, c] = (r - L + 1 - best_s) / float(h)
            if len(dists) >= 3:
                thr = float(np.median(dists)) * 0.5
                frequency[r, c] = (
                    float(sum(d <= thr for d in dists)) / float(len(dists))
                )
                dispersion[r, c] = float(np.std(dists))
    return novelty, age, frequency, dispersion


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
@register_operator(
    name="ts_vector_state_mahalanobis",
    category="candle_state_space",
    business_category="candle_state_space",
    canonical="ts_vector_state_mahalanobis",
    source="candle_state_space",
)
class TsVectorStateMahalanobis(SeriesOperator):
    """状态向量马氏距离：今天的蜡烛对象相对自身历史协方差的异常度。

    ``z_t=(f1..f4)_t`` 对比窗口内逐特征均值 μ 与收缩协方差
    ``(1-λ)Σ+λ·diag(Σ)``（R11 round-3 #65：均值+普通协方差，经典估计量一致配对）；
    ``N >= 5p`` 样本地板（R11 round-3 #66）。距离越大 → 今天蜡烛相对自身历史越异常。P1。

    M-140 Reference/Query：历史参考严格 ≤ t-1（均值/协方差只由 ``[t-window, t)``
    估计），当前查询 = t，查询行排除在参考之外（R6-145）。
    """

    metadata = _metadata(
        "ts_vector_state_mahalanobis",
        "4D 状态向量相对窗口收缩协方差的马氏距离（蜡烛对象异常度）。历史参考严格 ≤t-1、当前查询=t，查询行排除在参考之外。",
        ["f1", "f2", "f3", "f4", "window", "shrinkage"],
        unit="ratio",
        cost=6,
    )

    def _calculate_series(
        self, f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame, f4: pd.DataFrame,
        window: int = 60, shrinkage: float = 0.5, **_: Any,
    ) -> pd.DataFrame:
        arr = _mahalanobis_series(
            f1.to_numpy(dtype=float), f2.to_numpy(dtype=float),
            f3.to_numpy(dtype=float), f4.to_numpy(dtype=float),
            window, shrinkage,
        )
        return frame_like(f1, arr)


@register_operator(
    name="ts_vector_state_local_density",
    category="candle_state_space",
    business_category="candle_state_space",
    canonical="ts_vector_state_local_density",
    source="candle_state_space",
)
class TsVectorStateLocalDensity(SeriesOperator):
    """状态向量局部密度：今天的状态在自身先验近邻中的疏密程度。

    各特征按窗口标准差标准化后，取当前点距窗口内前序点第 k 近的距离 r_k，
    输出 **log-density = -log(max(r_k, 1e-3))**（R6-146：原 1/(r_k+eps) 在
    一字板/重复状态 r_k=0 时爆到 1/EPS；log-density 对重复状态给出大而有限的值）。
    高 → 常见状态；低 → 稀有/孤立状态。P1。

    M-140 Reference/Query：历史参考严格 ≤ t-1（缩放参数/近邻都由 ``[t-window, t)``
    估计），当前查询 = t，查询行排除在参考之外（R6-145）。
    """

    metadata = _metadata(
        "ts_vector_state_local_density",
        "当前状态到前序窗口第 k 近邻距离的负对数（log-density，见 R6-146）。历史参考严格 ≤t-1、当前查询=t，查询行排除在参考之外。",
        ["f1", "f2", "f3", "f4", "window", "k"],
        unit="log",
        cost=6,
    )

    def _calculate_series(
        self, f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame, f4: pd.DataFrame,
        window: int = 60, k: int = 5, **_: Any,
    ) -> pd.DataFrame:
        arr = _local_density_series(
            f1.to_numpy(dtype=float), f2.to_numpy(dtype=float),
            f3.to_numpy(dtype=float), f4.to_numpy(dtype=float),
            window, k,
        )
        return frame_like(f1, arr)


@register_operator(
    # R6-148: the input is a SINGLE scalar series ``x``, so the old
    # ``ts_multivariate_matrix_profile_novelty`` name over-declared (it is not
    # a multichannel/multivariate matrix profile).  Renamed to the honest name;
    # the old name stays as an alias.
    name="ts_matrix_profile_novelty",
    category="candle_state_space",
    business_category="candle_state_space",
    canonical="ts_matrix_profile_novelty",
    source="candle_state_space",
)
class TsMultivariateMatrixProfileNovelty(SeriesOperator):
    """矩阵轮廓新颖度：当前子序列与其最近历史子序列的 z 归一欧氏距离。

    ``x`` 为调用方传入的标量序列；当前尾随子序列长度 L，搜索历史带
    ``[t-history .. t-L]`` 内最近的前序匹配。距离大 → 当前模式新颖。P1。
    """

    metadata = _metadata(
        "ts_matrix_profile_novelty",
        "当前子序列到最近历史子序列的 z 归一化 RMS 距离（新颖度）。",
        ["x", "window", "subsequence_length", "history"],
        unit="ratio",
        cost=8,
        param_specs=_MATRIX_PROFILE_PARAM_SPECS,
        relational_specs=_MATRIX_PROFILE_RELATIONAL_SPECS,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 120,
        subsequence_length: int = 10, history: int = 80, **_: Any,
    ) -> pd.DataFrame:
        novelty, _age, _freq, _disp = _matrix_profile_series(
            x.to_numpy(dtype=float), window, subsequence_length, history,
        )
        return frame_like(x, novelty)


@register_operator(
    name="ts_matrix_profile_motif_age",
    category="candle_state_space",
    business_category="candle_state_space",
    canonical="ts_matrix_profile_motif_age",
    source="candle_state_space",
)
class TsMatrixProfileMotifAge(SeriesOperator):
    """矩阵轮廓模式年龄：最近历史匹配起点距今的归一化年龄。

    与新颖度共享同一矩阵轮廓核；``s*`` 为最近匹配的前序子序列起点，
    当前子序列起点为 ``t - L + 1``，**起点到起点** 的年龄为
    ``age = (t - L + 1) - s*``（R11 round-3 #63：旧式 ``t - s*`` 系统性偏大
    L-1 根 bar），输出 ``age/history``。P1。
    """

    metadata = _metadata(
        "ts_matrix_profile_motif_age",
        "当前子序列最近历史匹配的归一化年龄 age/history。",
        ["x", "window", "subsequence_length", "history"],
        unit="ratio",
        cost=8,
        param_specs=_MATRIX_PROFILE_PARAM_SPECS,
        relational_specs=_MATRIX_PROFILE_RELATIONAL_SPECS,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 120,
        subsequence_length: int = 10, history: int = 80, **_: Any,
    ) -> pd.DataFrame:
        _novelty, age, _freq, _disp = _matrix_profile_series(
            x.to_numpy(dtype=float), window, subsequence_length, history,
        )
        return frame_like(x, age)


@register_operator(
    name="ts_matrix_profile_motif_frequency",
    category="candle_state_space",
    business_category="candle_state_space",
    canonical="ts_matrix_profile_motif_frequency",
    source="candle_state_space",
)
class TsMatrixProfileMotifFrequency(SeriesOperator):
    """矩阵轮廓模式频率：历史带内近邻匹配（距离 <= 中位距离/2）占比。

    与新颖度 / 模式年龄共享同一矩阵轮廓核，输出真正区分的统计量（audit
    P0：discord/motif 若都只是 min-distance，必须合并为新颖度；频率是
    “这个模式最近被重复看到多少次”的独立度量）。P1。
    """

    metadata = _metadata(
        "ts_matrix_profile_motif_frequency",
        "历史带内近邻匹配（dist <= median/2）占比。",
        ["x", "window", "subsequence_length", "history"],
        unit="ratio",
        cost=8,
        param_specs=_MATRIX_PROFILE_PARAM_SPECS,
        relational_specs=_MATRIX_PROFILE_RELATIONAL_SPECS,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 120,
        subsequence_length: int = 10, history: int = 80, **_: Any,
    ) -> pd.DataFrame:
        _novelty, _age, freq, _disp = _matrix_profile_series(
            x.to_numpy(dtype=float), window, subsequence_length, history,
        )
        return frame_like(x, freq)


@register_operator(
    name="ts_matrix_profile_neighbor_dispersion",
    category="candle_state_space",
    business_category="candle_state_space",
    canonical="ts_matrix_profile_neighbor_dispersion",
    source="candle_state_space",
)
class TsMatrixProfileNeighborDispersion(SeriesOperator):
    """矩阵轮廓近邻离散度：历史带内所有候选距离的标准差。

    与新颖度 / 模式年龄 / 频率共享同一矩阵轮廓核；高离散度 = 历史带内既有
    很近也有很远的模式（audit P0 的 neighbor_dispersion 输出）。P1。
    """

    metadata = _metadata(
        "ts_matrix_profile_neighbor_dispersion",
        "历史带内候选子序列距离的标准差（近邻离散度）。",
        ["x", "window", "subsequence_length", "history"],
        unit="ratio",
        cost=8,
        param_specs=_MATRIX_PROFILE_PARAM_SPECS,
        relational_specs=_MATRIX_PROFILE_RELATIONAL_SPECS,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 120,
        subsequence_length: int = 10, history: int = 80, **_: Any,
    ) -> pd.DataFrame:
        _novelty, _age, _freq, disp = _matrix_profile_series(
            x.to_numpy(dtype=float), window, subsequence_length, history,
        )
        return frame_like(x, disp)


_NEW_CANONICALS = (
    "ts_vector_state_mahalanobis",
    "ts_vector_state_local_density",
    "ts_matrix_profile_novelty",
    "ts_matrix_profile_motif_age",
    "ts_matrix_profile_motif_frequency",
    "ts_matrix_profile_neighbor_dispersion",
)

_OLD_MULTIVARIATE_NAME = "ts_multivariate_matrix_profile_novelty"


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    try:
        OperatorRegistry.register_alias(
            _OLD_MULTIVARIATE_NAME, "ts_matrix_profile_novelty"
        )
    except (KeyError, ValueError):
        pass  # already registered
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
