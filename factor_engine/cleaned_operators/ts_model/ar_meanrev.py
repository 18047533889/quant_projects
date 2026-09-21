# -*- coding: utf-8 -*-
"""AR forecasting / innovation and mean-reversion operators (P0).

Daily panels in, daily panels out; causal rolling kernels per (instrument,
date).  ``ts_ar_coefficient`` and ``ts_variance_ratio`` already exist in
``regression_models``; this module adds the forecast / innovation forms and the
multi-horizon variance-ratio slope.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import ParamRole, ParamSpec, RelationalParamSpec, SeriesOperator, register_operator
from factor_engine.cleaned_operators.ts_model._rolling_core import (
    frame_like,
    metadata,
    ols_fit,
    trailing_contiguous_finite,
)

_CANONICALS: list[str] = []
_AR_CONFIGURED_HISTORY_CANONICALS: list[str] = []

# Model-audit Phase 4 (search-space hygiene): explicit ParamSpec declarations
# for the AR model scalars.  ``window`` is the alpha horizon (HORIZON, searched);
# ``order`` is the AR model order — an estimator-resolution knob, never a
# full-resolution search dimension (M-115/M-162/M-170).
#
# P1 (window-semantics governance): ``window`` is a MAX LOOKBACK, NOT a strict
# full-window requirement.  The kernel emits as soon as the minimum effective
# observation count (``order + 2`` valid rows) is present inside the trailing
# ``window`` — an EXPANDING warmup.  ``warmup_policy`` is a declared, catalog-
# visible parameter (part of the operator's semantic identity); ``"full"`` opts
# in to a strict full-history floor (the trailing window must be completely
# observed before any output).  ``min_history`` and ``min_effective_obs`` are
# both ``order + 2``.
_AR_WINDOW_SEMANTICS = "max_lookback"
_AR_WARMUP_POLICY = "expanding"            # default (legacy behaviour preserved)
_AR_FULL_WARMUP = "full"                   # strict full-history floor option
_AR_MIN_EFFECTIVE_OBS_EXPR = "order + 2"   # valid rows the OLS design must have
_AR_PARAM_SPECS: dict[str, ParamSpec] = {
    "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON, searchable=True),
    "order": ParamSpec(dtype=int, min=1, param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False),
    "warmup_policy": ParamSpec(dtype=str, choices=(_AR_WARMUP_POLICY, _AR_FULL_WARMUP),
                               param_role=ParamRole.POLICY, searchable=False),
}


def _declare_single(metadata_obj, specs: dict[str, ParamSpec], relations=()):
    metadata_obj.panel_params = ("x",)
    metadata_obj.panel_arity = 1
    metadata_obj.scalar_params = tuple(p for p in metadata_obj.param_names if p != "x")
    metadata_obj.param_specs = specs
    metadata_obj.relational_specs = list(relations)
    return metadata_obj


def _half_life_specs(*, prior: bool = False) -> dict[str, ParamSpec]:
    return {
        "window": ParamSpec(
            dtype=int, min=5, default=120, history_semantics="max_rows",
            history_formula="window + 1" if prior else None,
            param_role=ParamRole.HORIZON,
        ),
        "min_periods": ParamSpec(
            dtype=int, min=4, default=20, searchable=False,
            param_role=ParamRole.SUPPORT_POLICY,
        ),
    }


_HALF_LIFE_RELATIONS = (
    RelationalParamSpec("min_periods < window", message="window must be greater than min_periods"),
)


def validate_ar_configured_history(window: int, order: int, *, fit_lag: int = 0) -> int:
    """Return required input rows or reject a window that can never fit AR(p)."""
    required_window = 2 * int(order) + 2
    if int(window) < required_window:
        raise ValueError(
            "INSUFFICIENT_CONFIGURED_HISTORY: "
            f"window={window} requires at least {required_window} for order={order}"
        )
    return required_window + max(0, int(fit_lag))


def _ar_fit(seg: np.ndarray, order: int) -> tuple[np.ndarray | None, np.ndarray]:
    """Fit AR(order) on a 1-D segment; return (beta, lagged design rows).

    ``beta`` has length order+1 (intercept first).  ``design`` rows correspond
    to the same time indexes as the segment, with NaN for rows without enough
    lag history (used for the current-row forecast).
    """
    n = len(seg)
    design = np.full((n, order + 1), np.nan, dtype=float)
    for t in range(order, n):
        xv = seg[t - order : t][::-1]  # x_{t-1}, ..., x_{t-order}
        if np.all(np.isfinite(xv)):
            design[t] = np.concatenate([[1.0], xv])
    valid = np.all(np.isfinite(design), axis=1) & np.isfinite(seg)
    if valid.sum() < order + 2:
        return None, design
    X = design[valid]
    y = seg[valid]
    beta = ols_fit(X, y)
    return beta, design


def _ar_apply(vals: np.ndarray, window: int, order: int, stat: str, *, fit_lag: int = 0, stability_k: int = 0, warmup_policy: str = "expanding") -> np.ndarray:
    """Causal AR(order) rolling kernel.

    ``fit_lag=0`` (legacy) fits the AR model on the window *including* the
    current row and reports the in-sample fitted value.  ``fit_lag>=1`` fits on
    rows strictly before the current one and reports the genuine one-step-ahead
    forecast / innovation at the current row.  ``stability_k>0`` (with
    ``stat='coeff'``) reports the standard deviation of the AR slope over the
    last ``stability_k`` consecutive fits.

    Window semantics (P1): ``window`` is a MAX LOOKBACK, not a strict full
    window.  With ``warmup_policy="expanding"`` (default) the kernel emits as
    soon as the minimum effective observation count (``order + 2`` valid rows)
    is available inside the trailing window.  With ``warmup_policy="full"`` the
    trailing window must be completely observed (``fit_end >= window - 1``)
    before any output; a short series then degrades to all-NaN.
    """
    if warmup_policy not in ("expanding", "full"):
        raise ValueError(f"warmup_policy must be 'expanding' or 'full', got {warmup_policy!r}")
    n = len(vals)
    out = np.full(n, np.nan, dtype=float)
    o = max(1, int(order))
    w = int(window)
    lag = max(0, int(fit_lag))
    validate_ar_configured_history(w, o, fit_lag=lag)
    k = max(0, int(stability_k))
    for row in range(n):
        fit_end = row - lag
        if fit_end < 0:
            continue
        if warmup_policy == "full" and fit_end < w - 1:
            continue
        start = max(0, fit_end - w + 1)
        seg = vals[start : fit_end + 1]
        beta, design = _ar_fit(seg, o)
        if beta is None:
            continue
        if row < o:
            continue
        cur_xv = vals[row - o : row][::-1]  # x_{t-1}, ..., x_{t-order}
        if not np.all(np.isfinite(cur_xv)):
            continue
        cur = np.concatenate([[1.0], cur_xv])
        with np.errstate(over="ignore", invalid="ignore"):
            pred = float(np.dot(cur, beta))
        innov = float(vals[row] - pred) if np.isfinite(vals[row]) else np.nan
        if stat == "forecast":
            out[row] = pred
        elif stat == "innovation":
            out[row] = innov
        elif stat == "innovation_z":
            resid = np.full(len(seg), np.nan, dtype=float)
            valid = np.all(np.isfinite(design), axis=1) & np.isfinite(seg)
            resid[valid] = seg[valid] - design[valid] @ beta
            sd = float(np.nanstd(resid))
            if sd is not None and np.isfinite(sd) and sd > 0.0:
                out[row] = innov / sd
        elif stat == "coeff":
            out[row] = float(beta[1])  # first lag coefficient (index 0 is the intercept)
        elif stat == "coeff_stability":
            coeffs: list[float] = []
            for j in range(k):
                fe = row - lag - j
                if fe < 0:
                    break
                s2 = max(0, fe - w + 1)
                b2, _ = _ar_fit(vals[s2 : fe + 1], o)
                if b2 is None:
                    break
                coeffs.append(float(b2[1]))
            if len(coeffs) >= 2:
                out[row] = float(np.std(coeffs))
    return out


def _ar_op(name: str, description: str, unit: str, stat: str, *, fit_lag: int = 0, stability_k: int = 0, cost: int = 4, diagnostic_only: bool = False):
    _AR_CONFIGURED_HISTORY_CANONICALS.append(name)
    _desc = f"{description}（window=max lookback，非严格满窗；min_effective_obs=order+2；warmup_policy={_AR_WARMUP_POLICY} 渐进输出）"

    @register_operator(
        name=name,
        category="time_series_regression",
        business_category="time_series_regression",
        canonical=name,
        source="ts_model.ar_meanrev",
        backend="pandas_numpy",
        status="experimental",
        semantic_version="2.0",
    )
    class _ArOp(SeriesOperator):
        metadata = metadata(name, _desc, ["x", "window", "order", "warmup_policy"], unit=unit, cost=cost,
                            diagnostic_only=diagnostic_only, param_specs=_AR_PARAM_SPECS)
        # P1: window = max lookback (documented, machine-readable contract).
        metadata.window_semantics = _AR_WINDOW_SEMANTICS

        def _calculate_series(self, x, window=60, order=1, warmup_policy="expanding", **_):
            xv = x.to_numpy(dtype=float)
            rows, cols = xv.shape
            out = np.full((rows, cols), np.nan, dtype=float)
            for col in range(cols):
                out[:, col] = _ar_apply_vec(xv[:, col], int(window), int(order), stat,
                                            fit_lag=int(fit_lag), stability_k=int(stability_k),
                                            warmup_policy=str(warmup_policy))
            return frame_like(x, out)

        def validate_params(self, x, window=60, order=1, warmup_policy="expanding", **_):
            validate_ar_configured_history(window, order, fit_lag=fit_lag)
            return True

    return _ArOp


# In-sample (fitted-value) AR forms: fit on the window *including* the current
# row and report the fitted value / residual.  These are the honest names; the
# legacy ``ts_ar_forecast`` / ``ts_ar_innovation`` are aliases of the same
# kernels (fit_lag=0 is NOT a one-step-ahead forecast).
_ar_op("ts_ar_fitted_value", "AR(order) 窗口内拟合值(fit_lag=0, in-sample)。", "level", "forecast", diagnostic_only=True)
_ar_op("ts_ar_in_sample_resid", "当前实际值减窗口内 AR 拟合值(in-sample resid)。", "level", "innovation", diagnostic_only=True)
_ar_op("ts_ar_forecast", "AR(order) 当前值预测（legacy 名称——实为窗口内拟合值，见 ts_ar_fitted_value）。", "level", "forecast", diagnostic_only=True)
_ar_op("ts_ar_innovation", "当前实际值减 AR 预测（legacy 名称——实为样本内残差，见 ts_ar_in_sample_resid）。", "level", "innovation", diagnostic_only=True)
# M-043: ``ts_ar_innovation_z`` is an IN-SAMPLE kernel (fit_lag=0, fit includes
# the current row, identical to ``ts_ar_in_sample_resid`` but standardised by
# the window residual std).  Its name contains ``_innovation`` so the name-driven
# timing default auto-generates a PREDICTIVE contract (fit_cutoff_offset=1),
# which contradicts the in-sample kernel.  The reconciler must add an explicit
# MODEL_TIMING_CONTRACTS entry with ``fit_cutoff_offset=0`` (descriptive
# in-sample), like the other ``ts_ar_fitted_value`` / ``ts_ar_in_sample_resid``
# entries.
_ar_op("ts_ar_innovation_z", "AR 创新标准化（in-sample：拟合含当前样本，描述性标准化残差，非样本外创新）。", "level", "innovation_z", diagnostic_only=True)

# Prior-window (out-of-sample) AR forms: fit on t-window..t-1, forecast t.
_ar_op("ts_ar_prior_forecast", "AR(order) 截至 t-1 训练的一步预测。", "level", "forecast", fit_lag=1)
_ar_op("ts_ar_prior_innovation", "当前实际值减截至 t-1 训练的 AR 预测。", "level", "innovation", fit_lag=1)
_ar_op("ts_ar_prior_innovation_z", "AR 样本外创新 / 历史残差标准差。", "level", "innovation_z", fit_lag=1)
_ar_op("ts_ar_prior_coeff", "AR(order) 截至 t-1 训练的一阶滞后系数。", "level", "coeff", fit_lag=1)
_ar_op("ts_ar_coeff_stability", "AR 一阶滞后系数在最近 K 个严格截至 t-1 的滚动拟合中的标准差（因果 model-state alpha，衡量系数稳定性，非诊断）。", "level", "coeff_stability", fit_lag=1, stability_k=5, cost=6)




# ---------------------------------------------------------------------------
# R61 vectorized columns (no per-row Python loop)
# ---------------------------------------------------------------------------

def _ar_prefix_sums(vals: np.ndarray, o: int) -> dict:
    """Prefix sums of AR(order) design-row terms.  Design row at t is
    [1, x_{t-1}, ..., x_{t-order}] with response x_t; rows with any non-finite
    lag or non-finite response are zeroed (the authority drops them)."""
    n = vals.size
    valid = np.isfinite(vals)
    L = np.full((n, o), np.nan)
    for j in range(o):
        L[o:, j] = vals[o - 1 - j: n - 1 - j]
        valid &= np.isfinite(L[:, j])
    valid[:o] = False
    v = valid.astype(np.float64)
    terms = {"cnt": v, "y": vals * valid, "yy": vals * vals * valid}
    for j in range(o):
        terms[("L", j)] = L[:, j] * valid
        terms[("yL", j)] = vals * L[:, j] * valid
    for j in range(o):
        for k2 in range(j, o):
            terms[("LL", j, k2)] = L[:, j] * L[:, k2] * valid

    def _cs(x):
        return np.concatenate(([0.0], np.cumsum(np.where(np.isfinite(x), x, 0.0))))

    return {key: _cs(val) for key, val in terms.items()}, valid, L


def _ar_window_gram(cs: dict, o: int, t_hi: np.ndarray, t_lo: np.ndarray):
    """Batched Gram matrices G and rhs c for design rows t in [t_lo, t_hi]."""
    n = t_hi.size
    G = np.zeros((n, o + 1, o + 1))
    G[:, 0, 0] = cs["cnt"][t_hi + 1] - cs["cnt"][t_lo]
    for j in range(o):
        sL = cs[("L", j)][t_hi + 1] - cs[("L", j)][t_lo]
        G[:, 0, j + 1] = sL
        G[:, j + 1, 0] = sL
        G[:, j + 1, j + 1] = cs[("LL", j, j)][t_hi + 1] - cs[("LL", j, j)][t_lo]
        for k2 in range(j + 1, o):
            s = cs[("LL", j, k2)][t_hi + 1] - cs[("LL", j, k2)][t_lo]
            G[:, j + 1, k2 + 1] = s
            G[:, k2 + 1, j + 1] = s
    c = np.zeros((n, o + 1))
    c[:, 0] = cs["y"][t_hi + 1] - cs["y"][t_lo]
    for j in range(o):
        c[:, j + 1] = cs[("yL", j)][t_hi + 1] - cs[("yL", j)][t_lo]
    return G, c


def _pow2_column_scale(vals: np.ndarray) -> float:
    """Exact power-of-two multiplier bringing max|finite| to ~[0.5, 1).

    Power-of-two scaling is loss-free in binary floating point, so the scaled
    series is bit-exactly the original up to an exponent shift; Gram squares
    then stay in range for 1e-300/1e300-scale inputs (contract test:
    scale invariance without false NaN).
    """
    fin = vals[np.isfinite(vals)]
    if fin.size == 0:
        return 1.0
    m = float(np.max(np.abs(fin)))
    if not np.isfinite(m) or m == 0.0:
        return 1.0
    _, e = np.frexp(m)
    return 2.0 ** (-int(e))


def _unscale(out: np.ndarray, scale: float) -> np.ndarray:
    """Multiply by 1/scale in two safe half-steps (no overflow)."""
    if scale == 1.0:
        return out
    inv = 1.0 / scale
    e = np.frexp(inv)[1]
    h1 = 2.0 ** (e // 2)
    h2 = inv / h1
    return out * h1 * h2


def _ar_apply_vec(vals: np.ndarray, window: int, order: int, stat: str, *,
                  fit_lag: int = 0, stability_k: int = 0,
                  warmup_policy: str = "expanding") -> np.ndarray:
    """Vectorized twin of :func:`_ar_apply` — identical semantics, no per-row
    Python loop."""
    if warmup_policy not in ("expanding", "full"):
        raise ValueError(f"warmup_policy must be 'expanding' or 'full', got {warmup_policy!r}")
    n = vals.size
    o = max(1, int(order))
    w = int(window)
    lag = max(0, int(fit_lag))
    k = max(0, int(stability_k))
    validate_ar_configured_history(w, o, fit_lag=lag)
    # exact power-of-two rescale keeps Gram products in range on 1e+-300 data;
    # value-dimension outputs are scaled back at the exits below.
    scale = _pow2_column_scale(vals)
    if scale != 1.0:
        vals = vals * scale
    out = np.full(n, np.nan, dtype=float)
    rows = np.arange(n)
    fit_end = rows - lag
    ok = fit_end >= 0
    if warmup_policy == "full":
        ok &= fit_end >= w - 1
    ok &= rows >= o
    t_hi = np.clip(fit_end, 0, n - 1)
    start = np.clip(fit_end - w + 1, 0, n)
    t_lo = np.clip(start + o, 0, n)

    cs, valid, L = _ar_prefix_sums(vals, o)
    G, c = _ar_window_gram(cs, o, np.where(ok, t_hi, 0), np.where(ok, t_lo, 0))
    n_valid = G[:, 0, 0]
    # rank-deficient / ill-conditioned windows -> authority's
    # design_is_well_conditioned gate: cond(design) > 1e12 rejects.  For the
    # Gram G = X'X the singular values of X are sqrt(eigenvalues of G), so
    # cond(X) = sqrt(lam_max / lam_min) and the threshold maps to 1e24.
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.linalg.eigvalsh(G)
    lmin = lam[:, 0]
    lmax = lam[:, -1]
    # eigvalsh returns ~eps*lmax positive eigenvalues for EXACTLY singular
    # prefix-sum Grams (e.g. constant regressor windows), which would slip
    # past any finite ratio threshold; an exact det==0 test catches those.
    with np.errstate(divide="ignore", invalid="ignore"):
        detG = np.linalg.det(G)
    wellc = (lmin > 0.0) & (detG != 0.0) & np.isfinite(lmax) & (lmax <= 1e24 * lmin)
    ok &= n_valid >= (o + 2)
    beta = np.full((n, o + 1), np.nan)
    idx = np.where(ok & wellc)[0]
    if idx.size:
        beta[idx] = (np.linalg.pinv(G[idx]) @ c[idx][:, :, None])[:, :, 0]
    has_beta = np.zeros(n, dtype=bool)
    has_beta[idx] = np.all(np.isfinite(beta[idx]), axis=1)

    # current-row regressor: [1, x_{r-1}, ..., x_{r-o}] — the authority checks
    # its finiteness for EVERY stat (including coeff/coeff_stability).
    cur_ok = np.ones(n, dtype=bool)
    cur = np.zeros((n, o + 1))
    cur[:, 0] = 1.0
    for j in range(o):
        xv = np.full(n, np.nan)
        xv[o:] = vals[o - 1 - j: n - 1 - j]
        cur[:, j + 1] = np.where(np.isfinite(xv), xv, 0.0)
        cur_ok &= np.isfinite(xv)

    if stat == "coeff_stability":
        # beta1 series per fit row; chain of consecutive successful fits
        b1 = np.where(has_beta, beta[:, 1], np.nan)
        hb = has_beta
        runlen = np.zeros(n, dtype=np.int64)
        cnt = 0
        for i in range(n):        # O(n) scalar chain, cheap relative to old
            cnt = cnt + 1 if hb[i] else 0
            runlen[i] = cnt
        for i in np.where(ok)[0]:
            L2 = int(runlen[i])
            if L2 < 2:
                continue
            j2 = min(int(k) if k > 0 else L2, L2)
            seg = b1[i - j2 + 1: i + 1]
            seg = seg[np.isfinite(seg)]
            if seg.size >= 2 and cur_ok[i]:
                out[i] = float(np.std(seg))
        return out  # scale-invariant stat

    with np.errstate(over="ignore", invalid="ignore"):
        pred = np.einsum("ni,ni->n", cur, np.where(np.isfinite(beta), beta, 0.0))
    pred = np.where(has_beta & cur_ok, pred, np.nan)

    if stat == "forecast":
        out = _unscale(np.where(has_beta & cur_ok, pred, np.nan), scale)
    elif stat == "innovation":
        innov = np.where(np.isfinite(vals), vals - np.where(np.isfinite(pred), pred, 0.0), np.nan)
        out = _unscale(np.where(has_beta & cur_ok, innov, np.nan), scale)
    elif stat == "innovation_z":
        Syy = cs["yy"][np.where(ok, t_hi, 0) + 1] - cs["yy"][np.where(ok, t_lo, 0)]
        cLy = c  # [sum y, sum yL_j]
        bb = np.where(np.isfinite(beta), beta, 0.0)
        # ss_res = Syy - 2 b.c + b.G.b  (over valid design rows)
        Gb = np.einsum("nij,ni->nj", G, bb)
        ss = Syy - 2.0 * np.einsum("ni,ni->n", bb, cLy) + np.einsum("ni,ni->n", bb, Gb)
        cntv = np.maximum(n_valid, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            sd = np.sqrt(np.maximum(ss, 0.0) / cntv)
        innov = np.where(np.isfinite(vals), vals - np.where(np.isfinite(pred), pred, 0.0), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = innov / sd
        out = np.where(has_beta & cur_ok & np.isfinite(sd) & (sd > 0.0) & np.isfinite(z), z, np.nan)
    elif stat == "coeff":
        out = np.where(has_beta & cur_ok, beta[:, 1], np.nan)
    else:
        raise ValueError(f"unknown stat {stat!r}")
    return out


def _pair_ols_slope_prefix(vals: np.ndarray, w: int, min_periods: int, fit_lag: int = 0):
    """Rolling OLS slope b of d(seg) on seg[:-1] over trailing windows
    (vectorized).  Returns per-row (b, cnt) with NaN b for degenerate windows.
    fit_lag mirrors the authority: fit_end = row - fit_lag.

    The authority rescales the window by its max|finite| before the
    regression; the slope is scale-invariant, so the rescaling is skipped
    (the all-zero window degenerates through the x-variance gate instead).
    """
    n = vals.size
    rows = np.arange(n)
    fit_end = rows - max(0, int(fit_lag))
    # The intercept absorbs any constant shift of x or y, so centering both by
    # their column-level finite means leaves the slope untouched while removing
    # the large-mean cancellation that plagues cumsum Gram accumulation on
    # price-scale series.
    scale = _pow2_column_scale(vals)
    if scale != 1.0:
        vals = vals * scale
    fin = vals[np.isfinite(vals)]
    x_shift = float(np.mean(fin)) if fin.size else 0.0
    d_all = np.full(n, np.nan)
    d_all[1:] = vals[1:] - vals[:-1]
    dfin = d_all[np.isfinite(d_all)]
    d_shift = float(np.mean(dfin)) if dfin.size else 0.0
    vals_c = vals - x_shift
    d = d_all - d_shift
    xp = np.full(n, np.nan)
    xp[1:] = vals_c[:-1]
    pv = np.isfinite(xp) & np.isfinite(d)          # pair t valid: xp[t], d[t]
    v = pv.astype(np.float64)
    lo = np.clip(fit_end - w + 1, 0, n)            # window first row
    # pairs t in [max(lo,1), fit_end]  -> cumsum indices [t_lo, t_hi]
    # pair t uses vals[t-1] and vals[t]; the earliest in-window pair is
    # t = start + 1 (not start, whose x partner vals[start-1] lies outside
    # the window) and always t >= 1.
    t_lo = np.clip(lo + 1, 0, n)
    t_hi = np.clip(fit_end + 1, 0, n)

    def _cs(x):
        return np.concatenate(([0.0], np.cumsum(x)))

    C0 = _cs(v)
    CX = _cs(np.where(pv, xp, 0.0))
    CY = _cs(np.where(pv, d, 0.0))
    CXX = _cs(np.where(pv, xp * xp, 0.0))
    CXY = _cs(np.where(pv, xp * d, 0.0))
    cnt = C0[t_hi] - C0[t_lo]
    sx = CX[t_hi] - CX[t_lo]
    sy = CY[t_hi] - CY[t_lo]
    sxx = CXX[t_hi] - CXX[t_lo]
    sxy = CXY[t_hi] - CXY[t_lo]
    b = np.full(n, np.nan)
    ok = (cnt >= max(min_periods, 4)) & (fit_end >= 1)
    idx = np.where(ok)[0]
    if idx.size:
        G = np.zeros((idx.size, 2, 2))
        G[:, 0, 0] = cnt[idx]
        G[:, 0, 1] = sx[idx]
        G[:, 1, 0] = sx[idx]
        G[:, 1, 1] = sxx[idx]
        rhs = np.stack([sy[idx], sxy[idx]], axis=1)
        with np.errstate(all="ignore"):
            det = G[:, 0, 0] * G[:, 1, 1] - G[:, 0, 1] * G[:, 1, 0]
            tr = G[:, 0, 0] + G[:, 1, 1]
            disc = np.maximum(tr * tr - 4.0 * det, 0.0)
            sq = np.sqrt(disc)
            lam_max = 0.5 * (tr + sq)
            lam_min = 0.5 * (tr - sq)
            # authority: reject rank-deficient / cond(design) > 1e12 designs;
            # cond(X) = sqrt(lam_max/lam_min) -> threshold 1e24 (lam_min<=0
            # covers the authority's std(x)<=0 and magnitude==0 windows).
            good = (lam_min > 0.0) & (det != 0.0) & (lam_max <= 1e24 * lam_min)
        bb = np.full((idx.size, 2), np.nan)
        if good.any():
            detg = det[good]
            bb[good, 1] = (G[good, 0, 0] * rhs[good, 1] - G[good, 0, 1] * rhs[good, 0]) / detg
        b[idx] = bb[:, 1]
    return b, cnt


def _mean_reversion_half_life_vec(vals: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    n = vals.size
    b, _ = _pair_ols_slope_prefix(vals, window, min_periods, fit_lag=0)
    phi = 1.0 + b
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.where((phi > 0.0) & (phi < 1.0), np.log(0.5) / np.log(np.where((phi > 0) & (phi < 1), phi, 0.5)), np.nan)
    return hl


def _mean_reversion_ou_half_life_vec(vals: np.ndarray, window: int, min_periods: int,
                                     fit_lag: int = 0) -> np.ndarray:
    b, _ = _pair_ols_slope_prefix(vals, window, min_periods, fit_lag=fit_lag)
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.where(np.isfinite(b) & (b < 0.0), -np.log(2.0) / np.where(b < 0, b, -1.0), np.nan)
    return hl


def _variance_ratio_slope_vec(vals: np.ndarray, window: int, maxq: int, min_periods: int) -> np.ndarray:
    """Vectorized twin of _variance_ratio_slope (trailing contiguous finite
    run, magnitude-invariant, VR(q)-1 ~ log q slope)."""
    maxq = max(2, int(maxq))
    n = vals.size
    scale = _pow2_column_scale(vals)
    if scale != 1.0:
        vals = vals * scale
    s, k, lf = _mr_run_bounds(vals, window)
    cnt = k.astype(np.float64)

    def _cs(x):
        return np.concatenate(([0.0], np.cumsum(x)))

    # var1 over the run's first differences.  D1[i] = vals[i]-vals[i-1] is
    # backward-indexed; pairs inside the run [s, lf] are i in [s+1, lf], so the
    # cumsum window is [s+1, lf+1).
    D1 = np.full(n, np.nan)
    D1[1:] = vals[1:] - vals[:-1]
    d1v = np.isfinite(D1)
    CD1 = _cs(np.where(d1v, D1, 0.0))
    CD1sq = _cs(np.where(d1v, D1 * D1, 0.0))
    m1 = np.maximum(cnt - 1.0, 0.0)
    m1s = np.maximum(m1, 1.0)
    hi1 = np.clip(lf + 1, 0, n)
    lo1 = np.clip(s + 1, 0, n)
    sum1 = CD1[hi1] - CD1[lo1]
    ssq1 = CD1sq[hi1] - CD1sq[lo1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean1 = sum1 / m1s
        var1 = ssq1 / m1s - mean1 * mean1
    ok_win = (cnt >= max(max(min_periods, 6), max(min_periods, 3) + 1)) & (m1 >= 1) & (var1 > 0.0)

    out = np.full(n, np.nan)
    vr_mat = np.full((n, maxq + 1), np.nan)
    for q in range(2, maxq + 1):
        Dq = np.full(n, np.nan)
        Dq[q:] = vals[q:] - vals[:-q]
        dqv = np.isfinite(Dq)
        CDq = _cs(np.where(dqv, Dq, 0.0))
        CDqsq = _cs(np.where(dqv, Dq * Dq, 0.0))
        mq = cnt - q                    # len(qrets) = run_len - q
        hiq = np.clip(lf + 1, 0, n)     # backward-indexed Dq: pairs i in [s+q, lf]
        loq = np.clip(s + q, 0, n)
        ms = np.maximum(mq, 1.0)
        sumq = CDq[hiq] - CDq[loq]
        ssqq = CDqsq[hiq] - CDqsq[loq]
        with np.errstate(divide="ignore", invalid="ignore"):
            meanq = sumq / ms
            varq = ssqq / ms - meanq * meanq
            vr = np.where((mq >= 2) & (var1 > 0.0) & np.isfinite(varq),
                          varq / (q * np.where(var1 > 0.0, var1, 1.0)) - 1.0,
                          np.nan)
        vr_mat[:, q] = vr
    cntv = np.isfinite(vr_mat).sum(axis=1)
    ok = ok_win & (cntv >= 2)
    idx = np.where(ok)[0]
    if idx.size:
        V = np.where(np.isfinite(vr_mat[idx]), vr_mat[idx], 0.0)
        M = np.isfinite(vr_mat[idx]).astype(np.float64)
        cn = M.sum(axis=1)
        cn_safe = np.maximum(cn, 1.0)
        lx_full = np.full(maxq + 1, np.nan)
        lx_full[2:maxq + 1] = np.log(np.arange(2, maxq + 1, dtype=np.float64))
        lx0 = np.where(np.isfinite(lx_full), lx_full, 0.0)
        mx = (lx0 * M).sum(axis=1) / cn_safe
        my = V.sum(axis=1) / cn_safe
        dx = (lx0 - mx[:, None]) * M
        dy = (V - my[:, None]) * M
        denom = (dx * dx).sum(axis=1)
        num = (dx * dy).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(denom > 0, num / np.where(denom > 0, denom, 1.0), np.nan)
        out[idx] = slope
    return out


def _mr_run_bounds(vals: np.ndarray, w: int):
    """Trailing contiguous finite run bounds per row (window w)."""
    n = vals.size
    finite = np.isfinite(vals)
    idx = np.where(finite, np.arange(n), -1)
    last_fin = np.maximum.accumulate(idx)
    nan_idx = np.where(~finite, np.arange(n), -1)
    last_nan = np.maximum.accumulate(nan_idx)
    t = np.arange(n)
    lo = np.maximum(t - w + 1, 0)
    lf = np.maximum(last_fin, -1)
    s = np.maximum(last_nan[np.maximum(lf, 0)] + 1, lo)
    s = np.where(lf >= 0, s, lo)
    # the authority's trailing_contiguous_finite returns EMPTY when the last
    # row itself is NaN — the run must end at the current row, never bridge
    # trailing gaps.
    k = np.where((lf >= s) & (lf == t), lf - s + 1, 0)
    return s, k, lf


def _mean_reversion_half_life(vals: np.ndarray, window: int, min_periods: int) -> float:
    n = len(vals)
    start = max(0, n - window)
    seg = np.asarray(vals[start:], dtype=float)
    finite = seg[np.isfinite(seg)]
    magnitude = float(np.max(np.abs(finite))) if finite.size else 0.0
    if not np.isfinite(magnitude) or magnitude == 0.0:
        return np.nan
    seg = seg / magnitude
    xprev = seg[:-1]
    d = np.diff(seg)
    valid = np.isfinite(xprev) & np.isfinite(d)
    x, y = xprev[valid], d[valid]
    if len(x) < max(min_periods, 4) or np.std(x) <= 0.0:
        return np.nan
    design = np.column_stack([np.ones(len(x)), x])
    beta = ols_fit(design, y)
    if beta is None:
        return np.nan
    b = float(beta[1])
    # Exact discrete AR(1) half-life (audit P1-E): x_t = a + phi*x_{t-1} with
    # phi = 1 + b.  Only 0 < phi < 1 is mean-reverting; half_life =
    # ln(0.5)/ln(phi).  The old OU approximation -ln(2)/b is exposed separately
    # as ts_mean_reversion_ou_approx_half_life.
    phi = 1.0 + b
    if not np.isfinite(phi) or not (0.0 < phi < 1.0):
        return np.nan
    return float(np.log(0.5) / np.log(phi))


def _warn_if_trending_input(panel: np.ndarray, operator_name: str) -> None:
    """Warn (not raise) when a mean-reversion half-life input looks like a raw
    trending price rather than a stationary / spread / residual series.

    The half-life kernels regress ``d(seg)`` on ``seg[:-1]`` (audit M-044): they
    estimate the mean-reversion speed of a *stationary* deviation.  On a raw
    trending price the drift dominates and the implied ``phi`` / half-life is
    meaningless.  Detection is heuristic — a high linear-trend R² or a large
    ``|mean| / mean|Δ|`` drift ratio flags the series.  The operator still
    returns its (meaningless) value: this is a research-gate warning, not a
    fail-closed error.  Warnings are de-duplicated per call (one message total).
    """
    import warnings

    warned = False
    for col in range(panel.shape[1]):
        vals = panel[:, col]
        finite = vals[np.isfinite(vals)]
        if finite.size < 4:
            continue
        magnitude = float(np.max(np.abs(finite)))
        if not np.isfinite(magnitude) or magnitude == 0.0:
            continue
        finite = finite / magnitude
        x = np.arange(finite.size, dtype=float)
        xc = x - x.mean()
        denom = float(np.dot(xc, xc))
        if denom <= 0.0:
            continue
        slope = float(np.dot(xc, finite - finite.mean()) / denom)
        pred = finite.mean() + slope * xc
        ss_res = float(np.sum((finite - pred) ** 2))
        ss_tot = float(np.sum((finite - finite.mean()) ** 2))
        if ss_tot <= 0.0:
            continue
        r2 = 1.0 - ss_res / ss_tot
        diffs = np.diff(finite)
        mad = float(np.mean(np.abs(diffs))) if diffs.size else 0.0
        drift = abs(float(np.mean(finite))) / mad if mad > 0.0 else 0.0
        if r2 > 0.85 or drift > 5.0:
            if not warned:
                warnings.warn(
                    f"{operator_name}: input looks like a raw TRENDING price "
                    "(high linear-trend R² or large |mean|/mean|Δ| ratio); "
                    "mean-reversion half-life is only meaningful for a "
                    "spread/residual/stationary input — result is a research "
                    "gate, not a production signal.",
                    UserWarning,
                    stacklevel=3,
                )
                warned = True


@register_operator(
    name="ts_mean_reversion_half_life",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_mean_reversion_half_life",
    source="ts_model.ar_meanrev",
    backend="pandas_numpy",
    status="experimental",
)
class TsMeanReversionHalfLife(SeriesOperator):
    """均值回复半衰期 ln(0.5)/ln(1+beta)（离散 AR(1) 精确解）。

    INPUT SEMANTIC (audit M-044): the kernel regresses ``d(seg)`` on
    ``seg[:-1]`` and is only meaningful for a STATIONARY / SPREAD / RESIDUAL
    input.  Feeding a raw trending price produces a meaningless half-life; a
    runtime warning is raised when the input looks trending (research gate).
    """

    metadata = _declare_single(metadata(
        "ts_mean_reversion_half_life",
        "均值回复半衰期（AR(1) 精确离散）。输入必须为 spread/residual/stationary 序列；raw trending price 会给出无意义半衰期（研究 gate）。",
        ["x", "window", "min_periods"],
        unit="count", cost=3,
        input_units={"x": "spread_or_residual_or_stationary"},
    ), _half_life_specs(), _HALF_LIFE_RELATIONS)

    def _calculate_series(self, x, window=120, min_periods=20, **_):
        xv = x.to_numpy(dtype=float)
        _warn_if_trending_input(xv, "ts_mean_reversion_half_life")
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            out[:, col] = _mean_reversion_half_life_vec(xv[:, col], int(window), int(min_periods))
        return frame_like(x, out)


def _mean_reversion_ou_half_life(vals: np.ndarray, window: int, min_periods: int, fit_lag: int = 0) -> float:
    n = len(vals)
    # fit_lag=0: use all rows up to and including current (in-sample)
    # fit_lag=1: use rows strictly before current (prior/strict-prior)
    fit_end = n - 1 - fit_lag
    if fit_end < 0:
        return np.nan
    start = max(0, fit_end - window + 1)
    seg = np.asarray(vals[start : fit_end + 1], dtype=float)
    finite = seg[np.isfinite(seg)]
    magnitude = float(np.max(np.abs(finite))) if finite.size else 0.0
    if not np.isfinite(magnitude) or magnitude == 0.0:
        return np.nan
    seg = seg / magnitude
    xprev = seg[:-1]
    d = np.diff(seg)
    valid = np.isfinite(xprev) & np.isfinite(d)
    x, y = xprev[valid], d[valid]
    if len(x) < max(min_periods, 4) or np.std(x) <= 0.0:
        return np.nan
    design = np.column_stack([np.ones(len(x)), x])
    beta = ols_fit(design, y)
    if beta is None:
        return np.nan
    b = float(beta[1])
    if not np.isfinite(b) or b >= 0.0:
        return np.nan
    # OU continuous-time approximation -ln(2)/b (audit P1-E: kept under an
    # explicit _ou_approx name; the production canonical is the exact discrete
    # AR(1) half-life above).
    return float(-np.log(2.0) / b)


@register_operator(
    name="ts_mean_reversion_ou_approx_half_life",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_mean_reversion_ou_approx_half_life",
    source="ts_model.ar_meanrev",
    backend="pandas_numpy",
    status="experimental",
)
class TsMeanReversionOuApproxHalfLife(SeriesOperator):
    """均值回复半衰期 OU 近似 -log(2)/beta（beta<0 时定义）。

    INPUT SEMANTIC (audit M-044): like ``ts_mean_reversion_half_life``, the
    kernel regresses ``d(seg)`` on ``seg[:-1]`` and is only meaningful for a
    STATIONARY / SPREAD / RESIDUAL input.  A raw trending price yields a
    meaningless half-life; a runtime warning is raised when the input looks
    trending (research gate).
    """

    metadata = _declare_single(metadata(
        "ts_mean_reversion_ou_approx_half_life",
        "均值回复半衰期（OU 连续近似 -ln2/beta）。输入必须为 spread/residual/stationary 序列；raw trending price 会给出无意义半衰期（研究 gate）。",
        ["x", "window", "min_periods"],
        unit="count",
        cost=3,
        input_units={"x": "spread_or_residual_or_stationary"},
    ), _half_life_specs(), _HALF_LIFE_RELATIONS)

    def _calculate_series(self, x, window=120, min_periods=20, **_):
        xv = x.to_numpy(dtype=float)
        _warn_if_trending_input(xv, "ts_mean_reversion_ou_approx_half_life")
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            out[:, col] = _mean_reversion_ou_half_life_vec(
                xv[:, col], int(window), int(min_periods), fit_lag=0
            )
        return frame_like(x, out)


@register_operator(
    name="ts_mean_reversion_ou_approx_half_life_prior",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_mean_reversion_ou_approx_half_life_prior",
    source="ts_model.ar_meanrev",
    backend="pandas_numpy",
    status="experimental",
)
class TsMeanReversionOuApproxHalfLifePrior(SeriesOperator):
    """均值回复半衰期 OU 连续近似 -log(2)/beta（beta<0 时定义），严格截至 t-1 训练（fit_lag=1，因果槽位）。

    Each window trains strictly on rows before the current one (t-1) and
    reports the half-life at the current row, so the output is the causal
    out-of-sample statistic — the in-sample variant is
    ``ts_mean_reversion_ou_approx_half_life``.

    INPUT SEMANTIC (audit M-044): like the in-sample variant, the kernel
    regresses ``d(seg)`` on ``seg[:-1]`` and is only meaningful for a
    STATIONARY / SPREAD / RESIDUAL input.  A raw trending price yields a
    meaningless half-life; a runtime warning is raised when the input looks
    trending (research gate).
    """

    metadata = _declare_single(metadata(
        "ts_mean_reversion_ou_approx_half_life_prior",
        "均值回复半衰期（OU 连续近似 -ln2/beta，严格截至 t-1 训练）。输入必须为 spread/residual/stationary 序列；raw trending price 会给出无意义半衰期（研究 gate）。",
        ["x", "window", "min_periods"],
        unit="count",
        cost=3,
        input_units={"x": "spread_or_residual_or_stationary"},
    ), _half_life_specs(prior=True), _HALF_LIFE_RELATIONS)

    def _calculate_series(self, x, window=120, min_periods=20, **_):
        xv = x.to_numpy(dtype=float)
        _warn_if_trending_input(xv, "ts_mean_reversion_ou_approx_half_life_prior")
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            out[:, col] = _mean_reversion_ou_half_life_vec(
                xv[:, col], int(window), int(min_periods), fit_lag=1
            )
        return frame_like(x, out)


def _variance_ratio_slope(vals: np.ndarray, window: int, max_q: int, min_periods: int) -> float:
    maxq = max(2, int(max_q))
    # P1-90: when ``max_q`` exceeds what the window can ever support, every q in
    # the range 2..max_q cannot be evaluated and the operator would silently
    # fall back to a shorter q-range — manufacturing identical outputs for
    # different ``max_q`` parameters.  Fail loudly (independent of data
    # availability) instead of silently skipping.
    if maxq + 2 > int(window):
        raise ValueError(
            f"ts_variance_ratio_slope: max_q={max_q} needs at least "
            f"max_q+2={maxq + 2} rows in the window, but window={window}"
        )
    n = len(vals)
    start = max(0, n - window)
    seg = vals[start:]
    # P0-008: never bridge suspensions/gaps into adjacent observations — the
    # variance ratio runs over the most recent contiguous finite block only.
    finite = trailing_contiguous_finite(seg)
    if finite.size < max(min_periods, 6):
        return np.nan
    magnitude = float(np.max(np.abs(finite)))
    if not np.isfinite(magnitude) or magnitude == 0.0:
        return np.nan
    finite = finite / magnitude
    rets = np.diff(finite)
    if rets.size < max(min_periods, 3):
        return np.nan
    var1 = float(np.var(rets))
    if var1 <= 0.0:
        return np.nan
    logq = []
    vr = []
    for q in range(2, maxq + 1):
        if finite.size < q + 2:
            continue
        qrets = finite[q:] - finite[:-q]
        varq = float(np.var(qrets))
        vr.append(varq / (q * var1) - 1.0)
        logq.append(np.log(float(q)))
    if len(logq) < 2:
        return np.nan
    # P1-90: use a centered dot product (ddof=0 in both numerator and
    # denominator) instead of ``np.cov(logq, vr)[0, 1] / np.var(logq)``, whose
    # ddof=1 covariance / ddof=0 variance ratio biased the slope by n/(n-1).
    lx = np.asarray(logq, dtype=float) - float(np.mean(logq))
    ly = np.asarray(vr, dtype=float) - float(np.mean(vr))
    denom = float(np.dot(lx, lx))
    if denom <= 0.0:
        return np.nan
    return float(np.dot(lx, ly) / denom)


@register_operator(
    name="ts_variance_ratio_slope",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_variance_ratio_slope",
    source="ts_model.ar_meanrev",
    backend="pandas_numpy",
    status="experimental",
)
class TsVarianceRatioSlope(SeriesOperator):
    """多持有期方差比相对 log(q) 的斜率（趋势/随机游走/均值回复判别）。

    Slope of ``(VR(q) - 1)`` against ``log(q)`` across the horizons
    ``q = 2..max_q`` (audit M-045 sign convention): a POSITIVE slope means VR
    grows with q -> positive serial correlation -> TRENDING tendency; a NEGATIVE
    slope means VR falls with q -> negative serial correlation -> MEAN-REVERSION
    tendency; ~0 slope is consistent with a random walk.
    """

    metadata = _declare_single(metadata(
        "ts_variance_ratio_slope", "方差比相对 log(q) 的斜率（正=趋势，负=均值回复）。", ["x", "window", "max_q", "min_periods"], unit="dimensionless", output_unit="dimensionless", cost=4,
    ), {
        "window": ParamSpec(dtype=int, min=6, default=120, history_semantics="max_rows", param_role=ParamRole.HORIZON),
        "max_q": ParamSpec(dtype=int, min=3, default=10, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        "min_periods": ParamSpec(dtype=int, min=3, default=20, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
    }, (
        RelationalParamSpec("min_periods < window", message="window must be greater than min_periods"),
        RelationalParamSpec("max_q + 2 <= window", message="window must be at least max_q + 2"),
    ))

    def _calculate_series(self, x, window=120, max_q=10, min_periods=20, **_):
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            out[:, col] = _variance_ratio_slope_vec(xv[:, col], int(window), int(max_q), int(min_periods))
        return frame_like(x, out)


_CANONICALS.extend(
    [
        "ts_ar_fitted_value",
        "ts_ar_in_sample_resid",
        "ts_ar_forecast",
        "ts_ar_innovation",
        "ts_ar_innovation_z",
        "ts_mean_reversion_half_life",
        "ts_mean_reversion_ou_approx_half_life",
        "ts_mean_reversion_ou_approx_half_life_prior",
        "ts_variance_ratio_slope",
        "ts_ar_prior_forecast",
        "ts_ar_prior_innovation",
        "ts_ar_prior_innovation_z",
        "ts_ar_prior_coeff",
        "ts_ar_coeff_stability",
    ]
)

import factor_engine.cleaned_operators.operator_surface as _surface  # noqa: E402

_surface.extend_extended_only(set(_CANONICALS))
