# -*- coding: utf-8 -*-
"""Conditional / band dependence operators (2026-08 market language, P2).

* ``ts_conditional_transfer_entropy`` — transfer entropy conditioned on a third
  series: ``I(T_{s+lag}; S_s | T_s, C_s)`` in nats.  Where plain TE says "S
  predicts T", CTE asks whether that predictive power survives *given* the
  conditioning state (regime, market, valuation...).
* ``ts_modwt_band_corr`` — correlation between the level-``band`` Haar MODWT
  detail components of two series over a window (band-limited co-movement,
  e.g. does turnover co-move with returns only on the weekly band?).

Both are deterministic and prefix-causal.  The MODWT uses a causal boundary
(``a[-1] = 0``), so no future sample leaks into the coefficients.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, ParamSpec, RelationalParamSpec, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, map_pair_rolling, register_polars_udf

_ALPHA = 0.5  # Jeffreys smoothing, matching the plain-TE kernel.
_EPS = 1e-12
# Audit #21: the magnitude of the deterministic tie-breaking perturbation,
# expressed as a fraction of the window's value range.
_TIE_JITTER = 1e-9


def _break_ties_deterministic(v: np.ndarray) -> np.ndarray:
    """Deterministically break exact ties for quantile binning (audit #21).

    The conditional TE quantile bins silently collapse when a series carries
    exact ties (0-return days / limit bars / discrete financials).  Each exact
    tie group is perturbed by an index-scaled epsilon keyed to its stable-sort
    position — never randomness — so digitization lands the tied states in
    distinct cells while genuinely distinct values are left untouched.  NaNs
    are preserved (they sort last and are never grouped into a tie run).
    """
    v = np.asarray(v, dtype=float)
    n = v.size
    if n < 2:
        return v
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return v
    # A fully-degenerate (constant) series carries no structure to recover:
    # leave it untouched so the kernel's ``< 2 distinct states`` check still
    # fails closed.  Perturbing a constant series would *invent* information.
    if np.unique(finite).size < 2:
        return v
    rng = float(np.nanmax(finite) - np.nanmin(finite))
    if not np.isfinite(rng) or rng <= _EPS:
        rng = 1.0
    out = v.copy()
    order = np.argsort(v, kind="mergesort")
    s = v[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        if j > i:
            for k in range(i, j + 1):
                frac = (k - i) / float(j - i)
                out[order[k]] += _TIE_JITTER * rng * frac
        i = j + 1
    return out


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    domain: str,
    unit: str,
    cost: int,
    extra_tags: tuple[str, ...] = (),
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="conditional_dependence",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "conditional_dependence", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic", *extra_tags,
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([-np.inf, np.inf], dtype=float)
    if n_bins < 2:
        n_bins = 2
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size == 1:
        # single distinct value: the caller's ``< 2 unique states`` check
        # normally guards this; keep a minimal 2-cell split defensively.
        return np.array([edges[0] - 1.0, edges[0] + 1.0], dtype=float)
    # Round-7 P0: keep the ACTUAL quantile cell structure.  With heavy ties the
    # unique quantile edges can be fewer than ``n_bins + 1``; the old fallback
    # silently re-binned with equal-width ``linspace`` edges, so the estimator
    # changed meaning with the data's tie rate.  Fewer unique edges simply mean
    # fewer effective cells — the kernel's digitize/clip handles that honestly.
    return edges


def _conditional_te_window(
    tw: np.ndarray,
    sw: np.ndarray,
    cw: np.ndarray,
    bins: int,
    lag: int,
    min_transitions: int,
    min_cells_ratio: float = 1.0,
) -> float:
    """Conditional transfer entropy I(T_{s+lag}; S_s | T_s, C_s) for one window."""
    # Audit #21: break exact ties in ALL three series BEFORE lagging so a tied
    # state cannot silently collapse a quantile bin and the perturbation is
    # consistent between the ``t_s`` and ``t_{s+lag}`` views of the same value.
    tw = _break_ties_deterministic(tw)
    sw = _break_ties_deterministic(sw)
    cw = _break_ties_deterministic(cw)
    # Lag is applied on the original time axis, then NaN rows are masked: a NaN
    # gap must not silently redefine the lag (see the plain-TE kernel).  The
    # "future" window ``tw[lag:]`` only ever reaches the trailing window's last
    # row (causal, no lookahead — audit #24).
    n = tw.shape[0]
    if n < lag + 2:
        return np.nan
    t_t = tw[:-lag]  # t_s
    s_t = sw[:-lag]  # s_s
    c_t = cw[:-lag]  # c_s
    t_next = tw[lag:]  # t_{s+lag}
    mask = np.isfinite(t_t) & np.isfinite(s_t) & np.isfinite(c_t) & np.isfinite(t_next)
    # R5 P1-43(a): a 4-D joint distribution over ``bins^4`` cells needs far more
    # transitions than a plain count floor — under-powered windows fail closed
    # to NaN instead of emitting a noisy CTE from a near-empty contingency table.
    required = 3 * (bins ** 4)
    if int(mask.sum()) < max(lag + 2, min_transitions, required):
        return np.nan
    xs = t_t[mask]
    ys = s_t[mask]
    cs = c_t[mask]
    xn = t_next[mask]
    if np.unique(xs).size < 2 or np.unique(ys).size < 2 or np.unique(cs).size < 2:
        return np.nan
    n = xs.shape[0]
    x_edges = _quantile_edges(xs, bins)
    y_edges = _quantile_edges(ys, bins)
    c_edges = _quantile_edges(cs, bins)
    xb = np.clip(np.digitize(xs, x_edges) - 1, 0, bins - 1).astype(np.int64)
    xnb = np.clip(np.digitize(xn, x_edges) - 1, 0, bins - 1).astype(np.int64)
    yb = np.clip(np.digitize(ys, y_edges) - 1, 0, bins - 1).astype(np.int64)
    cb = np.clip(np.digitize(cs, c_edges) - 1, 0, bins - 1).astype(np.int64)
    # Audit #20: effective-state-space sample gate.  The joint (t', t, s, c)
    # has ``nxb * nxb * nyb * ncb`` possible cells; below ``k * cells`` usable
    # transitions the plug-in histogram is dominated by smoothing mass.
    nxb = int(x_edges.size - 1)
    nyb = int(y_edges.size - 1)
    ncb = int(c_edges.size - 1)
    if nxb < 2 or nyb < 2 or ncb < 2:
        return np.nan
    ratio = float(min_cells_ratio)
    if np.isfinite(ratio) and ratio > 0.0:
        required = max(required, int(np.ceil(ratio * nxb * nxb * nyb * ncb)))
    if n < required:
        return np.nan
    joint = np.zeros((bins, bins, bins, bins), dtype=np.float64)  # [t', t, s, c]
    for k in range(n):
        joint[xnb[k], xb[k], yb[k], cb[k]] += 1.0
    # Audit #22: a conditioning bin that collapsed to zero empirical samples
    # makes the conditional term undefined — the Jeffreys smoothing mass alone
    # would fabricate a nonzero conditional TE.  Fail closed to NaN.
    c_emp = joint.sum(axis=(0, 1, 2))  # over t', t, s -> [c]
    if np.any(c_emp <= 0.0):
        return np.nan
    joint += _ALPHA
    joint /= joint.sum()
    # Marginals (axes: 0=t', 1=t, 2=s, 3=c).
    p4 = joint
    ptc = joint.sum(axis=(0, 2))  # [t, c]
    ptptc = joint.sum(axis=2)  # [t', t, c]
    ptsc = joint.sum(axis=0)  # [t, s, c]
    te = 0.0
    for i in range(bins):  # t'
        for j in range(bins):  # t
            for k in range(bins):  # s
                for l in range(bins):  # c
                    p = p4[i, j, k, l]
                    if p <= _EPS:
                        continue
                    p_denom = ptptc[i, j, l] * ptsc[j, k, l]
                    if p_denom <= _EPS:
                        continue
                    te += p * (np.log(p) + np.log(ptc[j, l]) - np.log(p_denom))
    return float(max(te, 0.0))


def _tri_rolling(a: np.ndarray, b: np.ndarray, c: np.ndarray, w: int, fn) -> np.ndarray:
    rows, cols = a.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for r in range(rows):
            lo = max(0, r - w + 1)
            out[r, col] = fn(a[lo : r + 1, col], b[lo : r + 1, col], c[lo : r + 1, col])
    return out


@register_operator(
    name="ts_conditional_transfer_entropy",
    category="conditional_dependence",
    business_category="conditional_dependence",
    canonical="ts_conditional_transfer_entropy",
    source="conditional_dependence",
    status="experimental",
)
class TsConditionalTransferEntropy(SeriesOperator):
    """条件传递熵 I(T_{s+lag}; S_s | T_s, C_s)（nats）。

    在给定条件序列 C（regime/market/valuation...）状态的前提下，S 对 T 的方向
    信息传递。用于检验预测关系是否在控制状态变量后依然存在（如 turnover→return
    的关系在估值高/低状态下是否显著不同）。分位数分箱 + Jeffreys 平滑，
    常量/退化状态 fail-closed。R5 P1-43(a)：四维联合 (t',t,s,c) 有 ``bins^4``
    个格子，bins 限 {2,3} 且需 N >> bins^4 样本，否则 fail-closed NaN。
    P2 / Research。
    """

    metadata = _metadata(
        "ts_conditional_transfer_entropy",
        "条件传递熵 I(T_{s+lag}; S_s | T_s, C_s) nats。",
        ["target", "source", "condition", "window", "bins", "lag", "min_transitions", "min_cells_ratio"],
        domain="price_volume",
        unit="nats",
        cost=7,
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2),
        "bins": ParamSpec(dtype=int, choices=(2, 3)),
        "lag": ParamSpec(dtype=int, min=1),
        "min_cells_ratio": ParamSpec(dtype=float, min=0.0),
    }

    def _calculate_series(
        self,
        target: pd.DataFrame,
        source: pd.DataFrame,
        condition: pd.DataFrame,
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
        # R5 P1-43(a): production restricts the search to bins in {2, 3}.  The
        # 4-D joint (t', t, s, c) has ``bins^4`` cells; 4/5 bins need more daily
        # transitions than a panel can supply and only manufacture noise.
        if nb not in (2, 3):
            raise ValueError(
                "ts_conditional_transfer_entropy requires bins in {2, 3} "
                "(higher bins need bins^4 samples a daily panel cannot provide)"
            )
        if lg < 1:
            raise ValueError("ts_conditional_transfer_entropy requires lag >= 1")
        if w < lg + 2:
            raise ValueError("ts_conditional_transfer_entropy requires window >= lag + 2")
        # Feasibility gate (R5 P1-43(a) / round-7 P0): the 4-D joint
        # ``(t', t, s, c)`` needs ``3*bins^4`` transitions to be estimable.  The
        # gate must use the SAME requirement as the kernel — the older looser
        # floor ``2*bins^3`` let the default ``window=60 / bins=3`` pass the gate
        # (59 >= 54) while the kernel silently returned all-NaN (3*3^4 = 243
        # required), i.e. a dead operator.  Reject infeasible (window, bins,
        # lag) combinations loudly at execution time instead of failing inside
        # the kernel.  Audit #20: the effective-state-space floor
        # ``k * bins^4`` is folded into the same feasibility gate.
        if min_transitions is None:
            mt = max(30, 2 * nb * nb * nb)
        else:
            mt = max(lg + 2, int(min_transitions))
        required = 3 * (nb ** 4)
        mt = max(mt, required)
        mt = max(mt, int(np.ceil(ratio * nb * nb * nb * nb)))
        if w - lg < mt:
            raise ValueError(
                "ts_conditional_transfer_entropy window-lag "
                f"({w - lg}) < required transitions ({mt}) with bins={nb}; "
                "raise window or lower bins (bins=3 needs window >= "
                f"{mt + lg}, bins=2 needs window >= {3 * 16 + lg})"
            )
        return frame_like(
            target,
            _tri_rolling(
                target.to_numpy(dtype=float),
                source.to_numpy(dtype=float),
                condition.to_numpy(dtype=float),
                w,
                lambda a, b, c: _conditional_te_window(a, b, c, nb, lg, mt, ratio),
            ),
        )


def _haar_detail(v: np.ndarray, level: int, band: int) -> np.ndarray:
    """Causal Haar MODWT detail coefficients at the requested band.

    Level ``j`` uses lag ``2^(j-1)`` (the Haar filter dilated by the level):
    ``W_j,t = (V_{j-1,t} - V_{j-1,t-2^(j-1)}) / sqrt(2)`` with a causal boundary
    (the first ``2^(j-1)`` rows see the zero-padded past).  The earlier code
    applied the lag-1 filter at *every* level, which is not a multi-resolution
    decomposition at all — band k was just the same lag-1 difference iterated k
    times.
    """
    a = np.asarray(v, dtype=float)
    for j in range(1, level + 1):
        dil = 2 ** (j - 1)
        if a.size < dil + 1:
            return np.full(a.shape, np.nan, dtype=float)
        prev = np.concatenate([np.zeros(dil), a[:-dil]])  # causal boundary (no lookahead)
        detail = (prev - a) / np.sqrt(2.0)
        smooth = (prev + a) / np.sqrt(2.0)
        a = smooth
        if j == band:
            return detail
    return a


def _modwt_band_corr_chunk(xc: np.ndarray, yc: np.ndarray, level: int, band: int) -> float:
    if xc.shape[0] < level + 4:
        return np.nan
    dx = _haar_detail(xc, level, band)
    dy = _haar_detail(yc, level, band)
    # R5 P1-43(b): the causal zero-padding contaminates the cone of influence —
    # the first ``2^band - 1`` samples (all boundary effects accumulated across
    # levels <= band).  Censor them so the correlation is computed on interior
    # coefficients only.
    n_coi = (1 << band) - 1
    dx = dx[n_coi:]
    dy = dy[n_coi:]
    ok = np.isfinite(dx) & np.isfinite(dy)
    if int(ok.sum()) < level + 3:
        return np.nan
    xx = dx[ok]
    yy = dy[ok]
    vx = float(np.var(xx))
    vy = float(np.var(yy))
    if vx <= _EPS or vy <= _EPS:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])


@register_operator(
    name="ts_modwt_band_corr",
    category="conditional_dependence",
    business_category="conditional_dependence",
    canonical="ts_modwt_band_corr",
    source="conditional_dependence",
    status="experimental",
)
class TsModwtBandCorr(SeriesOperator):
    """Haar MODWT 频带相关（level-band 细节系数窗口相关）。

    用因果边界 Haar MODWT 把 x、y 分解到多分辨率频带，取第 ``band`` 层的细节
    系数做窗口相关。回答"两序列是否只在某个频带（如周线/月线）协同"。PIT 安全
    （因果边界不引入未来）。R5 P1-43(b)：因果零填充污染 cone-of-influence
    （前 ``2^band - 1`` 个系数），相关计算前先 censor 掉；``level > band`` 不改变
    输出（死参数区间），故 level 上限按 band 约束。窗口下限（audit #25/#7）：
    Haar MODWT 第 ``band`` 层需要 ``2^band`` 个样本，加上被 censor 的锥内系数，
    ``window`` 必须 >= ``2^band + level + 6``，否则拒绝（RelationalParamSpec，
    binder/search 在消耗预算前即判定不可行）。输出为无量纲相关系数（audit #8）。
    P2 / Research。
    """

    metadata = _metadata(
        "ts_modwt_band_corr",
        "Haar MODWT 频带细节系数窗口相关 Corr(D_band x, D_band y)，无量纲相关系数。",
        ["x", "y", "window", "level", "band"],
        domain="price_volume",
        unit="corr",
        cost=6,
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=8),
        "level": ParamSpec(dtype=int, min=1),
        "band": ParamSpec(dtype=int, min=1),
    }
    # Audit #25/#7: declare the transform's minimum-window feasibility so the
    # binder/search grammar rejects guaranteed-NaN combinations before spending
    # budget.  ``2 ** band`` is the Haar dilation at the requested band; the +6
    # leaves interior samples after the cone-of-influence censor.
    metadata.relational_specs = [
        RelationalParamSpec(
            "window >= 2 ** band + level + 6",
            "ts_modwt_band_corr requires window >= 2**band + level + 6 "
            "(Haar MODWT minimum + cone-of-influence censor; "
            "window={window}, band={band}, level={level})",
        )
    ]

    def _calculate_series(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        window: int = 120,
        level: int = 3,
        band: int = 1,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        lv = int(level)
        bd = int(band)
        if lv < 1:
            raise ValueError("ts_modwt_band_corr requires level >= 1")
        if not (1 <= bd <= lv):
            raise ValueError("ts_modwt_band_corr requires 1 <= band <= level")
        # R5 P1-43(b): the Haar loop stops at ``band``, so ``level > band`` is a
        # dead parameter range that never changes the output — cap ``level`` at
        # ``band`` to keep the search surface meaningful.
        lv = min(lv, bd)
        # Cone-of-influence coefficients are censored, so the window must be long
        # enough to leave interior samples after dropping ``2^band - 1`` of them.
        if w < (1 << bd) + lv + 6:
            raise ValueError(
                "ts_modwt_band_corr requires window >= 2**band + level + 6 "
                "(boundary/cone-of-influence coefficients are censored)"
            )
        return frame_like(
            x,
            map_pair_rolling(
                x.to_numpy(dtype=float),
                y.to_numpy(dtype=float),
                w,
                lambda a, b: _modwt_band_corr_chunk(a, b, lv, bd),
            ),
        )


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_research_only({"ts_conditional_transfer_entropy", "ts_modwt_band_corr"})
    for _canon in ("ts_conditional_transfer_entropy", "ts_modwt_band_corr"):
        register_polars_udf(_canon)


_register_surface()
