# -*- coding: utf-8 -*-
"""Advanced information-theoretic operators (2026-08 Gemini round, P1/P2).

Every kernel is a deterministic, prefix-causal, pandas-numpy reference that
returns a ``(TradeDate x Symbol)`` panel.  Randomness never appears in the
output: surrogates use deterministic circular shifts and binning is quantile
based, so repeated evaluation is bit-identical (the audit's determinism gate).

* ``ts_transfer_entropy``            — directional conditional information
  ``I(X_{s+lag}; Y_s | X_s)`` in nats (P1).
* ``ts_effective_transfer_entropy``  — TE minus the mean over deterministic
  structure-matched circular-shift surrogates; the negative side measures
  spurious coupling (P2).  Each surrogate circular-shifts the source on the
  ORIGINAL timeline, then rebuilds the transition triples and re-applies the
  valid mask, so the null shares the real estimate's missing-gap topology
  (review item #19).
* ``ts_score_rank_weighted_mean``    — exponential-rank (salience) weighted mean
  of a target, weighting the highest-``score`` observations most (P1).
* ``report_benford_js_divergence``   — first-digit distribution JS divergence
  against Benford's law over a rolling window of monetary amounts (P2 /
  research-only anomaly, never a fraud *probability*).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, ParamSpec, ParamRole, SeriesOperator, register_operator

_ALPHA = 0.5          # Jeffreys smoothing, fixed (not a search parameter).
# Deterministic circular-shift offsets for the effective-TE surrogate null.
# Each offset is the rotation amount (block length) of a circular block-shift
# of the SOURCE series on the ORIGINAL timeline; for the default window (60)
# these are on the order of ``ceil(0.1 * n) = 6`` and are fixed, so repeated
# evaluation is bit-identical (the audit's determinism gate).
_SURROGATE_OFFSETS = (7, 11, 17, 23, 31)
_EPS = 1e-12


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    domain: str,
    unit: str,
    cost: int,
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="information_theory",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "information_theory", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _rolling_apply_2d(values: np.ndarray, window: int, fn: Any, min_periods: int = 1) -> np.ndarray:
    rows, cols = values.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - window + 1)
            chunk = values[start : row + 1, col]
            out[row, col] = fn(chunk)
    return out


def _rolling_apply_2d_pair(a: np.ndarray, b: np.ndarray, window: int, fn: Any, min_periods: int = 1) -> np.ndarray:
    rows, cols = a.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - window + 1)
            out[row, col] = fn(a[start : row + 1, col], b[start : row + 1, col])
    return out


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Deterministic quantile bin edges over a window (never degenerate)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([-np.inf, np.inf], dtype=float)
    if n_bins < 2:
        n_bins = 2
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size == 1:
        # Fully degenerate input: every value identical.  Callers already
        # fail-closed on <2 distinct states, so this is only a well-defined
        # two-cell fallback so digitization below never sees a scalar.
        edges = np.array([edges[0] - 1.0, edges[0] + 1.0])
    # Duplicate quantiles (heavy ties) collapse cells.  Do NOT re-expand them to
    # equal-width bins: that switches quantile -> equal-width discretization
    # mid-algorithm and makes the factor jump as ties appear (review P1-49).
    # Keep the unique quantile edges; the *effective* cell count is
    # ``edges.size - 1`` and ``_te_from_transitions`` sizes its joint array to
    # it (see below).
    return edges


def _te_state_space_floor(nxb: int, nyb: int, min_cells_ratio: float) -> int:
    """Effective-state-space sample floor for the TE joint ``(x', x, y)``.

    The joint has ``nxb * nxb * nyb`` possible cells; a noisy estimate needs at
    least ``k * cells`` usable samples (audit #20) with a configurable ``k``
    (``min_cells_ratio``).  Returns ``>= 2`` so a degenerate 1-cell state space
    can never slip past the floor.
    """
    cells = int(nxb) * int(nxb) * int(nyb)
    k = float(min_cells_ratio)
    if not np.isfinite(k) or k <= 0.0:
        return max(2, int(cells))
    return max(2, int(np.ceil(k * cells)))


def _te_from_transitions(
    xs: np.ndarray,
    ys: np.ndarray,
    x_next: np.ndarray,
    bins: int,
    min_cells_ratio: float = 1.0,
    bins_x: np.ndarray | None = None,
    bins_y: np.ndarray | None = None,
) -> float:
    """Transfer entropy I(X';Y|X) from aligned transition triples, in nats.

    Causal contract (audit #24): ``x_next`` is ``xs`` shifted ``lag`` positions
    on the ORIGINAL time axis, so the "future" window only ever reaches the
    trailing window's last row — it never overlaps into a future beyond the
    current output row (no lookahead).

    Effective-state-space gate (audit #20): with ``nxb * nyb`` effective cells
    the estimate needs ``>= min_cells_ratio * nxb * nxb * nyb`` usable
    transitions; below that the plug-in histogram is dominated by Jeffreys
    smoothing mass and the output would be a noisy artefact — fail closed to
    NaN instead.

    ``bins_x`` / ``bins_y`` optionally supply PRECOMPUTED quantile edges so a
    caller can digitize several lagged samples against ONE shared reference
    binning (see ``_te_peak_window`` / ``_shared_bins`` — per-lag TE values are
    only comparable across lags when they share the same discretization).
    When omitted, edges are derived from the given ``xs`` / ``ys`` as before,
    so existing callers are unaffected.
    """
    n = xs.shape[0]
    if n < 2:
        return np.nan
    x_edges = _quantile_edges(xs, bins) if bins_x is None else bins_x
    y_edges = _quantile_edges(ys, bins) if bins_y is None else bins_y
    nxb = int(x_edges.size - 1)   # effective x cells = unique quantile edges - 1
    nyb = int(y_edges.size - 1)   # effective y cells
    if nxb < 2 or nyb < 2:
        # Every cell collapsed to one: conditional information on a single-bin
        # marginal is undefined.  Fail closed instead of returning a spurious 0
        # from a degenerate histogram (review P1-49).
        return np.nan
    if n < _te_state_space_floor(nxb, nyb, min_cells_ratio):
        return np.nan
    xb = np.clip(np.digitize(xs, x_edges) - 1, 0, nxb - 1).astype(np.int64)
    xnb = np.clip(np.digitize(x_next, x_edges) - 1, 0, nxb - 1).astype(np.int64)
    yb = np.clip(np.digitize(ys, y_edges) - 1, 0, nyb - 1).astype(np.int64)
    joint = np.zeros((nxb, nxb, nyb), dtype=np.float64)
    for k in range(n):
        joint[xnb[k], xb[k], yb[k]] += 1.0
    joint += _ALPHA  # Jeffreys smoothing: all cells > 0.
    joint /= joint.sum()
    # Marginals: p(x), p(x',x), p(x,y).  joint is indexed [x', x, y]; summing
    # axis 0 (x') leaves axes [x, y], so pxy is directly p(x,y) indexed [x, y].
    px = joint.sum(axis=(0, 2))          # over x' and y -> p(x)
    pxx = joint.sum(axis=2)              # p(x', x)
    pxy = joint.sum(axis=0)              # over x' -> p(x, y), indexed [x, y]
    # TE = sum p(x',x,y) * log( p(x',x,y) * p(x) / (p(x',x) * p(x,y)) ).
    te = 0.0
    for i in range(nxb):
        for j in range(nxb):
            for k in range(nyb):
                p = joint[j, i, k]
                if p <= _EPS:
                    continue
                te += p * (np.log(p) + np.log(px[i]) - np.log(pxx[j, i]) - np.log(pxy[i, k]))
    return float(max(te, 0.0))


def _transfer_entropy_window(
    tw: np.ndarray,
    sw: np.ndarray,
    bins: int,
    lag: int,
    min_transitions: int,
    min_cells_ratio: float = 1.0,
) -> float:
    """Transfer entropy for a *single* causal window (one computation, O(w log w)).

    The earlier implementation re-rolled every prefix inside each window (the
    outer row loop called a rolling sub-routine and kept only ``[-1]``), so one
    output row cost O(w^2).  This kernel computes the current window once.

    Causality (audit #24): ``lag`` is applied on the ORIGINAL time axis first
    (``x_next = tw[lag:]``), so the "future" window is always a subset of the
    trailing window ending at the current row — the estimate at row ``r`` never
    reads past row ``r``.
    """
    # Lag is applied on the ORIGINAL time axis first, then NaN rows are masked:
    # compacting NaN before lagging would let ``lag=1`` pair a day-1 value with
    # a day-3 value across a missing day-2 (a gap silently redefines the lag).
    n = tw.shape[0]
    if n < lag + 2:
        return np.nan
    x_t = tw[:-lag]
    y_t = sw[:-lag]
    x_next = tw[lag:]
    mask = np.isfinite(x_t) & np.isfinite(y_t) & np.isfinite(x_next)
    if int(mask.sum()) < max(lag + 2, min_transitions):
        return np.nan
    xs = x_t[mask]
    ys = y_t[mask]
    xn = x_next[mask]
    # Degenerate constant state: with < 2 distinct states the quantile bins are
    # arbitrary and Jeffreys smoothing would *invent* information; fail closed.
    if np.unique(xs).size < 2 or np.unique(ys).size < 2:
        return np.nan
    return _te_from_transitions(xs, ys, xn, bins, min_cells_ratio=min_cells_ratio)


@register_operator(
    name="ts_transfer_entropy",
    category="information_theory",
    business_category="information_theory",
    canonical="ts_transfer_entropy",
    source="advanced_information",
)
class TsTransferEntropy(SeriesOperator):
    """时序传递熵 ``I(X_{s+lag}; Y_s | X_s)``（nats，方向为 source→target）。

    窗口内对 ``(x', x, y)`` 三元组做分位数分箱（``bins``），加 Jeffreys 平滑
    (alpha=0.5)，输出 ``sum p log( p(x',x,y) p(x) / (p(x',x) p(x,y)) )``，负
    数值误差 clip 到 0。确定性强：分箱用分位数，无任何随机扰动。

    因果性（audit #24）：``lag`` 在原始时间轴上先应用，``x_next = tw[lag:]``
    永远只读到当前输出行为止（PIT-safe，无 lookahead）。有效状态空间门
    （audit #20）：可用样本数 ``N`` 必须 >= ``min_cells_ratio · nxb²·nyb``
    （``nxb/nyb`` 为有效分箱数），否则返回 NaN——绝不输出被 Jeffreys 平滑
    质量主导的噪声估计。
    """

    metadata = _metadata(
        "ts_transfer_entropy",
        "传递熵 I(X_{s+lag}; Y_s | X_s) nats（无量纲信息量），source→target 方向。",
        ["target", "source", "window", "bins", "lag", "min_transitions", "min_cells_ratio"],
        domain="price_volume",
        unit="nats",
        cost=5,
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2),
        "bins": ParamSpec(dtype=int, min=2, max=8, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        "lag": ParamSpec(dtype=int, min=1),
        "min_cells_ratio": ParamSpec(dtype=float, min=0.0, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(
        self,
        target: pd.DataFrame,
        source: pd.DataFrame,
        window: int = 60,
        bins: int = 3,
        lag: int = 1,
        min_transitions: Any = None,
        min_cells_ratio: float = 1.0,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        nb = int(bins)
        lg = int(lag)
        ratio = float(min_cells_ratio)
        if not (2 <= nb <= 8):
            raise ValueError("ts_transfer_entropy requires 2 <= bins <= 8")
        if lg < 1:
            raise ValueError("ts_transfer_entropy requires lag >= 1")
        if w < lg + 2:
            raise ValueError("ts_transfer_entropy requires window >= lag + 2")
        # With ``bins`` bins per variable the joint transition space has bins^3
        # cells; a handful of transitions would be dominated by Jeffreys smoothing
        # mass.  Default floor scales with the cell count.  Fail closed loudly when
        # the requested window cannot physically produce enough transitions (the
        # old default silently returned an all-NaN column for bins>=8).
        if min_transitions is None:
            mt = max(30, 3 * nb * nb)
        else:
            mt = max(lg + 2, int(min_transitions))
        # Audit #20: the effective-state-space gate (N >= k * cells) is a hard
        # feasibility floor — an infeasible (window, bins, lag, k) combination
        # is rejected at the call boundary rather than emitting a noisy estimate.
        mt = max(mt, int(np.ceil(ratio * nb * nb * nb)))
        if w - lg < mt:
            raise ValueError(
                f"ts_transfer_entropy window-lag ({w - lg}) < min_transitions ({mt}) "
                f"with bins={nb}, min_cells_ratio={ratio}; raise window or lower bins "
                "(default window=60 supports bins<=4)"
            )
        return _frame_like(
            target,
            _rolling_apply_2d_pair(
                target.to_numpy(dtype=float),
                source.to_numpy(dtype=float),
                w,
                lambda a, b: _transfer_entropy_window(a, b, nb, lg, mt, ratio),
            ),
        )


def _effective_transfer_entropy_window(
    tw: np.ndarray,
    sw: np.ndarray,
    bins: int,
    lag: int,
    min_transitions: int,
    min_cells_ratio: float = 1.0,
) -> float:
    """Effective TE for a *single* window (one computation, like the plain kernel).

    Shares the plain kernel's causality contract (audit #24) and effective-
    state-space gate (audit #20): the real TE and every surrogate go through
    ``_te_from_transitions`` with the same ``min_cells_ratio``.

    Surrogate null (review item #19): every surrogate circular-shifts the
    SOURCE series on the ORIGINAL timeline, rebuilds the ``(x_lead, y_lag)``
    transition triples on the shifted timeline, and THEN re-applies the valid
    mask (drops missing AFTER the shift).  The null therefore shares the exact
    same physical-time missing-gap topology as the real estimate — a NaN gap in
    the source stays a NaN gap in the surrogate — so ``E[TE_surrogate]`` is
    computed over structure-matched nulls.
    """
    n = tw.shape[0]
    if n < lag + 2:
        return np.nan
    x_t = tw[:-lag]
    y_t = sw[:-lag]
    x_next = tw[lag:]
    mask = np.isfinite(x_t) & np.isfinite(y_t) & np.isfinite(x_next)
    if int(mask.sum()) < max(lag + 2, min_transitions):
        return np.nan
    xs = x_t[mask]
    ys = y_t[mask]
    xn = x_next[mask]
    if np.unique(xs).size < 2 or np.unique(ys).size < 2:
        return np.nan
    real = _te_from_transitions(xs, ys, xn, bins, min_cells_ratio=min_cells_ratio)
    if not np.isfinite(real):
        return np.nan
    # Structure-matched surrogate null (review item #19).
    #
    # The plain TE path applies the raw time lag on the ORIGINAL timeline and
    # only THEN masks missing values, so the real estimate's transitions live on
    # the original physical-time grid (a NaN gap never redefines the lag).  The
    # old effective-TE surrogate instead COMPRESSED the source series (dropped
    # the NaN rows) and circularly shifted the compressed series — that erased
    # the physical-time missing-gap structure from the null, so a gap that
    # separates two dependent segments no longer blocked dependence in the
    # surrogate and the null expectation no longer matched the real TE's bias.
    #
    # Fix: circular/block-shift the SOURCE series on the ORIGINAL timeline
    # (``_SURROGATE_OFFSETS`` deterministic offsets, block length = shift
    # amount, on the order of ``ceil(0.1 * n)``), THEN rebuild the
    # ``(x_lead, y_lag)`` triples on the shifted timeline, THEN drop missing.
    surr: list[float] = []
    for off in _SURROGATE_OFFSETS:
        if off <= lag or off >= n:
            continue
        y_shifted = np.empty_like(sw)
        y_shifted[: n - off] = sw[off:]
        y_shifted[n - off :] = sw[:off]
        # Reconstruct (x_lead, y_lag) on the shifted timeline, then re-apply the
        # same valid mask (drop missing AFTER the shift).
        y_t_s = y_shifted[:-lag]
        mask_s = np.isfinite(x_t) & np.isfinite(y_t_s) & np.isfinite(x_next)
        if int(mask_s.sum()) < max(lag + 2, min_transitions):
            continue
        xs_s = x_t[mask_s]
        ys_s = y_t_s[mask_s]
        xn_s = x_next[mask_s]
        if np.unique(xs_s).size < 2 or np.unique(ys_s).size < 2:
            continue
        te_s = _te_from_transitions(xs_s, ys_s, xn_s, bins, min_cells_ratio=min_cells_ratio)
        if np.isfinite(te_s):
            surr.append(te_s)
    if not surr:
        return np.nan
    return real - float(np.mean(surr))


@register_operator(
    name="ts_effective_transfer_entropy",
    category="information_theory",
    business_category="information_theory",
    canonical="ts_effective_transfer_entropy",
    source="advanced_information",
    status="experimental",
)
class TsEffectiveTransferEntropy(SeriesOperator):
    """有效传递熵：TE 减去确定性 structure-matched surrogate 均值。

    surrogate 用固定偏移 ``[7,11,17,23,31]`` 在**原始时间轴**上对 source 做
    确定性循环平移（绝不随机 shuffle），再在平移后的时间轴上重建
    ``(x_lead, y_lag)`` 三元组并重新施加有效掩码（shift 之后才 drop missing），
    因此 null 与真实估计共享完全相同的缺失 gap 拓扑（review #19）。同一输入
    每次输出逐位一致。ETE 可为负（真实耦合弱于 surrogate），不 clip。
    P2 / Research。
    """

    metadata = _metadata(
        "ts_effective_transfer_entropy",
        "有效传递熵 TE - E[TE_surrogate]（确定性 circular shift），nats。",
        ["target", "source", "window", "bins", "lag", "min_transitions", "min_cells_ratio"],
        domain="price_volume",
        unit="nats",
        cost=6,
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2),
        "bins": ParamSpec(dtype=int, min=2, max=8, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        "lag": ParamSpec(dtype=int, min=1),
        "min_cells_ratio": ParamSpec(dtype=float, min=0.0, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(
        self,
        target: pd.DataFrame,
        source: pd.DataFrame,
        window: int = 60,
        bins: int = 3,
        lag: int = 1,
        min_transitions: Any = None,
        min_cells_ratio: float = 1.0,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        nb = int(bins)
        lg = int(lag)
        ratio = float(min_cells_ratio)
        if not (2 <= nb <= 8):
            raise ValueError("ts_effective_transfer_entropy requires 2 <= bins <= 8")
        if lg < 1:
            raise ValueError("ts_effective_transfer_entropy requires lag >= 1")
        if w < lg + 2:
            raise ValueError("ts_effective_transfer_entropy requires window >= lag + 2")
        if min_transitions is None:
            mt = max(30, 3 * nb * nb)
        else:
            mt = max(lg + 2, int(min_transitions))
        mt = max(mt, int(np.ceil(ratio * nb * nb * nb)))
        if w - lg < mt:
            raise ValueError(
                f"ts_effective_transfer_entropy window-lag ({w - lg}) < min_transitions "
                f"({mt}) with bins={nb}, min_cells_ratio={ratio}; raise window or lower bins"
            )
        return _frame_like(
            target,
            _rolling_apply_2d_pair(
                target.to_numpy(dtype=float),
                source.to_numpy(dtype=float),
                w,
                lambda a, b: _effective_transfer_entropy_window(a, b, nb, lg, mt, ratio),
            ),
        )


def _score_rank_weighted_mean(t: np.ndarray, sc: np.ndarray, decay: float) -> float:
    valid = np.isfinite(t) & np.isfinite(sc)
    if not valid.any():
        return np.nan
    tv = t[valid].astype(float)
    sv = sc[valid].astype(float)
    n = tv.size
    if n == 0:
        return np.nan
    # Average-tie ranks (highest score -> rank 0).  A stable argsort alone gives
    # tied scores distinct ranks in arrival order, silently weighting the earlier
    # observation more; equal scores must share the mean rank / equal weight.
    order = np.argsort(-sv, kind="mergesort")
    s_sorted = -sv[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j)
        i = j + 1
    w = np.power(decay, ranks)
    w /= w.sum()
    # ``w`` is indexed by the ORIGINAL observation order (ranks[order[i]] = ...),
    # so it must multiply ``tv`` in that same order.  ``tv[order]`` would re-sort
    # the target and multiply each weight against a *different* observation.
    return float(np.sum(w * tv))


@register_operator(
    name="ts_score_rank_weighted_mean",
    category="information_theory",
    business_category="information_theory",
    canonical="ts_score_rank_weighted_mean",
    source="advanced_information",
)
class TsScoreRankWeightedMean(SeriesOperator):
    """指数秩加权均值：按 ``score`` 从高到低排序后加权平均 ``target``。

    ``w_r = decay^r / sum decay^r``（r=0 是最高 score）。这是 salience / 显著性
    加权的基础原语；"显著日相对普通日贡献" 用 ``WeightedMean - ts_mean`` 表达。
    稳定排序 + 无随机，逐位确定。
    """

    metadata = _metadata(
        "ts_score_rank_weighted_mean",
        "指数秩加权均值（salience 加权）：按 score 降序加权平均 target。",
        ["target", "score", "window", "decay"],
        domain="price_volume",
        # The output is a weighted mean *of the target* — it inherits the target's
        # unit (price / amount / volatility / earnings), never a fixed ratio.
        unit="same_as:target",
        cost=3,
    )

    def _calculate_series(
        self,
        target: pd.DataFrame,
        score: pd.DataFrame,
        window: int = 60,
        decay: float = 0.85,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        d = float(decay)
        if not (0.0 < d <= 1.0):
            raise ValueError("ts_score_rank_weighted_mean requires 0 < decay <= 1")
        return _frame_like(
            target,
            _rolling_apply_2d_pair(
                target.to_numpy(dtype=float),
                score.to_numpy(dtype=float),
                w,
                lambda a, b: _score_rank_weighted_mean(a, b, d),
            ),
        )


def _benford_js(vals: np.ndarray) -> float:
    finite = vals[np.isfinite(vals)]
    finite = finite[finite != 0.0]
    if finite.size < 10:
        return np.nan
    digits = np.floor(np.abs(finite) / (10.0 ** np.floor(np.log10(np.abs(finite)))))
    digits = np.clip(digits, 1, 9).astype(np.int64)
    counts = np.bincount(digits, minlength=10)[1:].astype(np.float64)
    p = counts / counts.sum()
    benford = np.log10(1.0 + 1.0 / np.arange(1, 10, dtype=float))
    m = 0.5 * (p + benford)
    ok = m > 0.0
    pm = p[ok]
    bm = benford[ok]
    mm = m[ok]
    # KL with the 0 log 0 = 0 convention; log is computed only where the
    # numerator is positive so the eager log never sees a zero.
    rp = np.divide(pm, mm, out=np.ones_like(pm), where=pm > 0.0)
    rb = np.divide(bm, mm, out=np.ones_like(bm), where=bm > 0.0)
    kl_pm = float(np.sum(pm * np.log(rp)))
    kl_bm = float(np.sum(bm * np.log(rb)))
    js = 0.5 * (kl_pm + kl_bm)
    return float(np.sqrt(max(js, 0.0) / np.log(2.0)))


@register_operator(
    name="report_benford_js_divergence",
    category="information_theory",
    business_category="information_theory",
    canonical="report_benford_js_divergence",
    source="advanced_information",
    status="experimental",
)
class ReportBenfordJsDivergence(SeriesOperator):
    """滚动首位数字分布相对 Benford 定律的 JS 距离（归一化到 [0,1]）。

    输入必须是报表金额类量纲字段（严禁比率/百分比/代码/布尔）。本算子只描述
    *数字分布异常*，绝不解释为"造假概率"——firm-year divergence 的稳健性在
    文献中仍存争议，因此仅作为 Research 特征供下游检验。**用法警告**：对
    as-of forward-fill 的日频基本面 panel，同一份财报值会在多个交易日重复，
    首位数分布将主要反映*披露频率/forward-fill 持久性*而非财报数字本身的
    异常；应作用于同一份财报的多个金额项或 distinct 披露观测。P2 / Research。
    """

    metadata = _metadata(
        "report_benford_js_divergence",
        "首位数字分布相对 Benford 的 JS 距离 [0,1]（数字分布异常，非造假概率）。",
        ["amount", "window"],
        domain="fundamental",
        unit="ratio",
        cost=2,
    )

    def _calculate_series(self, amount: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        w = int(window)
        return _frame_like(amount, _rolling_apply_2d(amount.to_numpy(dtype=float), w, _benford_js))


# ---------------------------------------------------------------------------
# Transfer-entropy peak over a fixed lag grid (fused P1/P2)
# ---------------------------------------------------------------------------

_TE_LAGS = (1, 2, 3, 5, 10)


def _shared_bins(x: np.ndarray, y: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Reference quantile bin edges for a TE-peak window, shared by every lag.

    Built ONCE from the unshifted parent window (``_quantile_edges`` drops
    non-finite values).  Every lag then digitizes against these SAME edges, so
    the per-lag raw TE values only differ through genuine dependence/sample-size
    effects — NOT through each lag silently re-discretizing its own shifted
    sample.  Shifting still shrinks the per-lag sample; the shared edges keep the
    discretization constant (that is the point of the fix).
    """
    return _quantile_edges(x, n_bins), _quantile_edges(y, n_bins)


def _te_peak_window(
    tw: np.ndarray,
    sw: np.ndarray,
    bins: int,
    min_transitions: int,
    min_cells_ratio: float = 1.0,
) -> tuple[float, float]:
    """TE over the fixed lag grid (1,2,3,5,10) on one causal window.

    Returns ``(peak_strength, peak_lag_norm)`` — the max TE value and its lag
    normalized by the largest lag.  Binning is shared per window: one reference
    binning built once from the unshifted parent window (``_shared_bins``) is
    passed into every lag's ``_te_from_transitions``, so the per-lag TE values
    are comparable across lags.  This is the "fused primitive" that avoids
    re-discretizing five times in the AST.

    Every lag shares the plain kernel's causality (audit #24) and effective-
    state-space gate (audit #20).
    """
    n = tw.shape[0]
    x_edges, y_edges = _shared_bins(tw, sw, bins)
    if int(x_edges.size - 1) < 2 or int(y_edges.size - 1) < 2:
        # The shared reference binning collapsed to a single cell: conditional
        # information on a one-bin marginal is undefined (review P1-49).  Fail
        # closed instead of emitting a spurious value from a degenerate histogram.
        return np.nan, np.nan
    best = np.nan
    best_lag = np.nan
    for lag in _TE_LAGS:
        if n < lag + 2:
            continue
        # Lag on the original axis, then mask NaN (a gap must not redefine lag).
        x_t = tw[:-lag]
        y_t = sw[:-lag]
        x_next = tw[lag:]
        mask = np.isfinite(x_t) & np.isfinite(y_t) & np.isfinite(x_next)
        if int(mask.sum()) < max(lag + 2, min_transitions):
            continue
        xs = x_t[mask]
        ys = y_t[mask]
        xn = x_next[mask]
        if np.unique(xs).size < 2 or np.unique(ys).size < 2:
            continue
        # Only the *masked* transitions (xs, ys, xn) are aligned triples; passing
        # the unmasked ``x_next`` here would re-pair values across the NaN gaps
        # (P0-01 review) — a length mismatch / time misalignment.
        v = _te_from_transitions(
            xs, ys, xn, bins,
            min_cells_ratio=min_cells_ratio,
            bins_x=x_edges, bins_y=y_edges,
        )
        if not np.isfinite(v):
            continue
        if not np.isfinite(best) or v > best:
            best = v
            best_lag = float(lag)
    if np.isfinite(best):
        return best, best_lag / float(_TE_LAGS[-1])
    return np.nan, np.nan


def _te_peak_window_excess(
    tw: np.ndarray,
    sw: np.ndarray,
    bins: int,
    min_transitions: int,
    min_cells_ratio: float = 1.0,
    n_surrogates: int = 20,
    seed: int = 0,
) -> tuple[float, float]:
    """Surrogate-standardized TE peak: ``max_real - E[max_surrogate]``.

    ``peak_strength`` is a max over lags of raw TE, which is positively biased
    even when x,y are independent.  This variant destroys the x→y dependence
    with ``n_surrogates`` deterministic (seeded) permutations of the window's
    source series, recomputes the per-lag TE peak for each surrogate, and
    returns the excess ``max_real - mean(max_surrogate)`` plus the real peak's
    normalized lag.

    Every real and surrogate per-lag TE shares the SAME reference binning
    (``_shared_bins``), so the excess is a pure dependence-strength effect, not
    a discretization/sample-size artifact.  The seed default is fixed so the
    operator is reproducible for factor mining.

    Returns ``(excess, peak_lag_norm)``.
    """
    n = tw.shape[0]
    x_edges, y_edges = _shared_bins(tw, sw, bins)
    if int(x_edges.size - 1) < 2 or int(y_edges.size - 1) < 2:
        return np.nan, np.nan

    def _peak(y_series: np.ndarray) -> tuple[float, float]:
        best = np.nan
        best_lag = np.nan
        for lag in _TE_LAGS:
            if n < lag + 2:
                continue
            # Lag on the original axis, then mask NaN (a gap must not redefine lag).
            x_t = tw[:-lag]
            y_t = y_series[:-lag]
            x_next = tw[lag:]
            mask = np.isfinite(x_t) & np.isfinite(y_t) & np.isfinite(x_next)
            if int(mask.sum()) < max(lag + 2, min_transitions):
                continue
            xs = x_t[mask]
            ys = y_t[mask]
            xn = x_next[mask]
            if np.unique(xs).size < 2 or np.unique(ys).size < 2:
                continue
            v = _te_from_transitions(
                xs, ys, xn, bins,
                min_cells_ratio=min_cells_ratio,
                bins_x=x_edges, bins_y=y_edges,
            )
            if not np.isfinite(v):
                continue
            if not np.isfinite(best) or v > best:
                best = v
                best_lag = float(lag)
        return best, best_lag

    real, real_lag = _peak(sw)
    if not np.isfinite(real):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    finite_idx = np.nonzero(np.isfinite(sw))[0]
    surrogate_maxes: list[float] = []
    for _ in range(max(1, int(n_surrogates))):
        if finite_idx.size < 2:
            break
        sw_perm = sw.copy()
        # Permute the FINITE source values among the finite positions: the NaN
        # footprint and y's marginal distribution are preserved exactly, and the
        # x→y alignment is destroyed.
        sw_perm[finite_idx] = sw[finite_idx][rng.permutation(finite_idx.size)]
        s_best, _ = _peak(sw_perm)
        if np.isfinite(s_best):
            surrogate_maxes.append(s_best)
    if not surrogate_maxes:
        return np.nan, np.nan
    excess = real - float(np.mean(surrogate_maxes))
    return excess, real_lag / float(_TE_LAGS[-1])


@register_operator(
    name="ts_transfer_entropy_peak_strength",
    category="information_theory",
    business_category="information_theory",
    canonical="ts_transfer_entropy_peak_strength",
    source="advanced_information",
)
class TsTransferEntropyPeakStrength(SeriesOperator):
    """固定 lag 网格 (1,2,3,5,10) 上传递熵的峰值 ``max_l TE_l``（nats）。

    fused 原语：同一窗口只分箱一次，输出跨 lag 的最大 TE，避免在搜索树里为
    每个 lag 单独重算 TE。高 = 在某个典型时滞上存在强非线性条件信息传递。
    PIT 安全、确定性。
    """

    metadata = _metadata(
        "ts_transfer_entropy_peak_strength",
        "TE 在 lag∈{1,2,3,5,10} 上的峰值 max TE_l（nats）。",
        ["target", "source", "window", "bins", "min_transitions", "min_cells_ratio"],
        domain="price_volume",
        unit="nats",
        cost=7,
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2),
        "bins": ParamSpec(dtype=int, min=2, max=8, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        "min_cells_ratio": ParamSpec(dtype=float, min=0.0, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(
        self,
        target: pd.DataFrame,
        source: pd.DataFrame,
        window: int = 60,
        bins: int = 3,
        min_transitions: Any = None,
        min_cells_ratio: float = 1.0,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        nb = int(bins)
        ratio = float(min_cells_ratio)
        if not (2 <= nb <= 8):
            raise ValueError("ts_transfer_entropy_peak_strength requires 2 <= bins <= 8")
        if w < max(_TE_LAGS) + 2:
            raise ValueError("ts_transfer_entropy_peak_strength requires window >= 12")
        if min_transitions is None:
            mt = max(30, 3 * nb * nb)
        else:
            mt = max(max(_TE_LAGS) + 2, int(min_transitions))
        mt = max(mt, int(np.ceil(ratio * nb * nb * nb)))
        if w - max(_TE_LAGS) < mt:
            raise ValueError(
                f"ts_transfer_entropy_peak_strength window-lag ({w - max(_TE_LAGS)}) "
                f"< min_transitions ({mt}) with bins={nb}, min_cells_ratio={ratio}; "
                "raise window or lower bins"
            )
        return _frame_like(
            target,
            _rolling_apply_2d_pair(
                target.to_numpy(dtype=float),
                source.to_numpy(dtype=float),
                w,
                lambda a, b: _te_peak_window(a, b, nb, mt, ratio)[0],
            ),
        )


@register_operator(
    name="ts_transfer_entropy_peak_lag",
    category="information_theory",
    business_category="information_theory",
    canonical="ts_transfer_entropy_peak_lag",
    source="advanced_information",
)
class TsTransferEntropyPeakLag(SeriesOperator):
    """传递熵峰值所在时滞 ``argmax_l TE_l / 10``（归一化 [0,1]）。

    与 ``ts_transfer_entropy_peak_strength`` 共享同一 fused 计算，回答"信息传递
    的典型时滞有多长"。与滞后相关峰同类，但这里是**非线性条件信息**。PIT 安全。
    """

    metadata = _metadata(
        "ts_transfer_entropy_peak_lag",
        "TE 峰值 lag（归一化 l*/10，[0,1] 无量纲）。",
        ["target", "source", "window", "bins", "min_transitions", "min_cells_ratio"],
        domain="price_volume",
        unit="ratio",
        cost=7,
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2),
        "bins": ParamSpec(dtype=int, min=2, max=8, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        "min_cells_ratio": ParamSpec(dtype=float, min=0.0, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(
        self,
        target: pd.DataFrame,
        source: pd.DataFrame,
        window: int = 60,
        bins: int = 3,
        min_transitions: Any = None,
        min_cells_ratio: float = 1.0,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        nb = int(bins)
        ratio = float(min_cells_ratio)
        if not (2 <= nb <= 8):
            raise ValueError("ts_transfer_entropy_peak_lag requires 2 <= bins <= 8")
        if w < max(_TE_LAGS) + 2:
            raise ValueError("ts_transfer_entropy_peak_lag requires window >= 12")
        if min_transitions is None:
            mt = max(30, 3 * nb * nb)
        else:
            mt = max(max(_TE_LAGS) + 2, int(min_transitions))
        mt = max(mt, int(np.ceil(ratio * nb * nb * nb)))
        if w - max(_TE_LAGS) < mt:
            raise ValueError(
                f"ts_transfer_entropy_peak_lag window-lag ({w - max(_TE_LAGS)}) "
                f"< min_transitions ({mt}) with bins={nb}, min_cells_ratio={ratio}; "
                "raise window or lower bins"
            )
        return _frame_like(
            target,
            _rolling_apply_2d_pair(
                target.to_numpy(dtype=float),
                source.to_numpy(dtype=float),
                w,
                lambda a, b: _te_peak_window(a, b, nb, mt, ratio)[1],
            ),
        )


@register_operator(
    name="ts_transfer_entropy_peak_excess",
    category="information_theory",
    business_category="information_theory",
    canonical="ts_transfer_entropy_peak_excess",
    source="advanced_information",
)
class TsTransferEntropyPeakExcess(SeriesOperator):
    """固定 lag 网格 (1,2,3,5,10) 上的"超额"传递熵峰值（nats）。

    ``ts_transfer_entropy_peak_strength`` 是原始 TE 跨 lag 的 max——即使 x,y
    独立，该 max 也正偏。本算子用确定性（seeded）置换 surrogate 破坏 x→y 依赖，
    输出 ``max_real - mean(max_surrogate)``：独立时 ≈ 0，真实耦合时 > 0。
    所有 real / surrogate 的每个 lag 共享同一参考分箱（跨 lag 可比）。确定性、
    PIT 安全。``n_surrogates`` / ``seed`` 标记为 non-searchable（仅供调参，不进
    搜索语法）。
    """

    metadata = _metadata(
        "ts_transfer_entropy_peak_excess",
        "TE 峰值减 surrogate 均值（max_real - E[max_surrogate]，nats）。",
        ["target", "source", "window", "bins", "min_transitions", "min_cells_ratio",
         "n_surrogates", "seed"],
        domain="price_volume",
        unit="nats",
        cost=8,
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2),
        "bins": ParamSpec(dtype=int, min=2, max=8, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        "min_cells_ratio": ParamSpec(dtype=float, min=0.0, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        "n_surrogates": ParamSpec(dtype=int, min=1, default=20, searchable=False, param_role=ParamRole.NUMERICAL),
        "seed": ParamSpec(dtype=int, default=0, searchable=False, param_role=ParamRole.NUMERICAL),
    }

    def _calculate_series(
        self,
        target: pd.DataFrame,
        source: pd.DataFrame,
        window: int = 60,
        bins: int = 3,
        min_transitions: Any = None,
        min_cells_ratio: float = 1.0,
        n_surrogates: int = 20,
        seed: int = 0,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        nb = int(bins)
        ratio = float(min_cells_ratio)
        ns = max(1, int(n_surrogates))
        if not (2 <= nb <= 8):
            raise ValueError("ts_transfer_entropy_peak_excess requires 2 <= bins <= 8")
        if w < max(_TE_LAGS) + 2:
            raise ValueError("ts_transfer_entropy_peak_excess requires window >= 12")
        if min_transitions is None:
            mt = max(30, 3 * nb * nb)
        else:
            mt = max(max(_TE_LAGS) + 2, int(min_transitions))
        mt = max(mt, int(np.ceil(ratio * nb * nb * nb)))
        if w - max(_TE_LAGS) < mt:
            raise ValueError(
                f"ts_transfer_entropy_peak_excess window-lag ({w - max(_TE_LAGS)}) "
                f"< min_transitions ({mt}) with bins={nb}, min_cells_ratio={ratio}; "
                "raise window or lower bins"
            )
        return _frame_like(
            target,
            _rolling_apply_2d_pair(
                target.to_numpy(dtype=float),
                source.to_numpy(dtype=float),
                w,
                lambda a, b: _te_peak_window_excess(
                    a, b, nb, mt, ratio, n_surrogates=ns, seed=int(seed)
                )[0],
            ),
        )


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({
            "ts_transfer_entropy",
            "ts_score_rank_weighted_mean",
            "ts_transfer_entropy_peak_strength",
            "ts_transfer_entropy_peak_lag",
            "ts_transfer_entropy_peak_excess",
        })
    _surface.extend_research_only({"ts_effective_transfer_entropy", "report_benford_js_divergence"})


_register_surface()
