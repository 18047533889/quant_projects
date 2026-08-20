# -*- coding: utf-8 -*-
"""R20-GROUPSTATE-BETA-DIRECTUSE: GLOBAL_STATE + GROUP_STATE stock-specific
causal dimensionless Direct Alpha canonicals.

Three canonicals, all time-series per column (columns are independent
cross-sections — no cross-column interaction anywhere):

* ``beta_residual_z``          — stock return minus the PRIOR-window beta times
                                 the market return, z-scored against the
                                 prior-window residual std.
* ``beta_divergence_pct``      — stock return minus prior-window beta times
                                 market return, normalized by |market return|
                                 (dimensionless divergence ratio).
* ``relative_strength_group_pct`` — stock trailing compound return minus the
                                 group ex-self mean trailing compound return
                                 (peer-relative momentum gap).

CAUSALITY CONTRACT (the family's defining property): the beta (and the
z / group statistics) are estimated on a STRICTLY PRIOR window that ENDS at
row t-1 — the signal row t NEVER enters its own fit (no contemporaneous beta
estimation at signal time).  This is the ``reg_forecast_error_pct`` /
``ts_ar_prior_*`` / ``fit_lag=1`` precedent, applied to the market model.

Duplicate audit (survey BEFORE implementing — documented skips + neighbors):

SURVEY of existing market-beta / group-relative families:
* ``rolling_beta_to_market`` (``_numpy_kernels.py`` L862 = ts_regression_slope_
  (index_ret, ret, window)) — the beta LEVEL, contemporaneous window INCLUDING
  t, not a stock-specific residual signal.  Different estimator + unit-bearing.
* ``downside_beta`` / ``tail_beta`` (``_numpy_kernels.py`` L875/L905) — beta
  levels on conditional subsamples.  Same differentiation.
* ``ts_regression_slope`` / ``ts_regression_intercept`` / ``ts_regression_r2``
  (rolling_pack.py / _numpy_kernels.py) — generic single-x regression stats;
  slope-only levels, contemporaneous.
* ``ts_multi_regression_resid_z`` / ``ts_huber_regression_resid_z`` /
  ``ts_ridge_regression_resid_z`` (ts_model/dynamic_regression.py, registered
  via ``_register_multi``) — the nearest neighbors to ``beta_residual_z``:
  generic y~x1..x4 rolling regression residual z with ``fit_lag=0``
  (in-sample diagnostic, window INCLUDES t) and ``fit_lag=1`` prior-fit
  variants, with an ``expanding`` warmup policy (window is a MAX LOOKBACK,
  not a strict full window).  Differentiation: this module pins the
  MARKET MODEL specifically (stock vs market, the GLOBAL_STATE mapping), a
  STRICT full prior window (min_periods == window — any NaN/±Inf inside the
  fit window -> NaN, fail-closed, never an expanding shortcut), and the
  divergence-ratio form ``beta_divergence_pct`` (residual over |market
  return|) that exists nowhere in the tree.
* ``reg_residual_zscore`` (technical/indicators_v2.py L1207) — time-trend
  (y~time) residual z, CONTEMPORANEOUS fit (window includes t).  Different
  regressor (time, not market), different causality class.
* ``reg_forecast_error_pct`` (technical/indicators_v2.py L1139) — time-trend
  one-step-ahead forecast error normalized by close.  The prior-fit
  precedent, but regressor = time and normalization = price level (not
  residual-std z, not |market| ratio).
* ``benchmark_excess_return`` (common/polars_returns.py L29, polars) —
  ret - benchmark_ret, level-unit excess, no beta, no window, no
  standardization, polars backend.
* ``ts_ar_prior_*`` (ts_model/ar_meanrev.py) — strictly-prior AR(1)
  family on a single series; no market regressor, no group dimension.
* group family neighbors (``relative_strength_group_pct``):
  ``group_ex_self_mean`` / ``group_ex_self_weighted_mean`` (group_ext.py
  L89/L119) — LOO group mean LEVEL of x (unit-bearing, cross-sectional
  only, no trailing return); ``group_peer_beta_deviation``
  (cross_section/peer_ops.py L108) — x minus LOO WEIGHTED peer mean for beta
  panels (weighted, beta-specific, cross-sectional, not a trailing-return
  relative strength); ``cs_demean`` / ``group_neutralize`` /
  ``group_zscore`` (common/polars_cs_basic.py L123, polars_native/
  group_batch1.py L418/L929) — whole-cross-section / FULL-GROUP self-
  INCLUSIVE transforms with no trailing-return semantics (and
  ``group_zscore`` zero-fills a degenerate std — the silent zero this
  family refuses).
* sibling R20 slice ``cleaned_operators/technical/exself_cs_v1.py``:
  ``ex_self_mean_gap`` / ``ex_self_zscore`` / ``ex_self_rank_pct`` /
  ``ex_self_mad_z`` — purely CROSS-SECTIONAL (per-row t only) LOO
  statistics of a raw panel value x.

DUPLICATE-AUDIT VERDICT — TWO ASSIGNED CANONICALS SKIPPED:
* ``peer_residual_z``      == ``ex_self_zscore``   ((x - LOO mean)/LOO std,
  per-row cross-sectional) — landed by the sibling R20-EXSELF-CS slice.
* ``within_group_rank_pct`` == ``ex_self_rank_pct`` (LOO average-rank
  percentile in [0,1], self excluded) — same slice.
  Re-landing them here would create exact duplicate canonicals under two
  registrations; per the audit-first rule they are SKIPPED and recorded in
  evidence/r2/R20-GROUPSTATE-BETA-DIRECTUSE.yaml.  This module therefore
  owns only the TIME-SERIES side of the brief (the prior-beta market-model
  family and the trailing group relative strength), which existed nowhere.

Family contract (all three landed canonicals):
* beta / mean / std estimated on the STRICTLY PRIOR window [t-window, t-1]
  (window rows, ENDING at t-1); the signal row t never enters its own fit.
* Fail-closed windows: any NaN or ±Inf inside the prior window -> NaN
  (min_periods == window, never expanding, never zero-filled); a NaN/±Inf
  signal-row input -> NaN.
* Degenerate denominators -> NaN, never 0: zero market variance
  (sxx <= eps), zero residual std, zero |market return| (for the ratio
  form), zero finite peers (for the group mean).
* Prefix-invariant: mutating row t cannot change any row < t+1's output
  (the fit at row t+1 reads rows [t+1-window, t] — the mutation at t is
  legitimately visible at t+1 and later only).
* All outputs dimensionless (z-score; residual/|market| ratio;
  return-minus-return gap).
* ``window`` ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON) with
  the runtime guard rejecting EXACTLY the same minimum (5 — a beta fit on
  fewer than 5 prior bars is noise; 5 also gives the OLS residual std
  (dof = window-2: intercept + slope consumed) at least 3 dof).
* Group semantics (``relative_strength_group_pct``): per-row LOO group mean
  of the trailing compound returns (peers = same label, self excluded,
  NaN/±Inf peers excluded from the mean, self NaN/±Inf -> NaN, zero finite
  peers -> NaN never 0).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import (
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    SeriesOperator,
    register_operator,
)

_EPS = 1e-12


def _pi(v: Any, name: str, minimum: int) -> int:
    """Runtime guard: strict integer >= minimum (bools rejected, 5.5-style
    silent truncation rejected).  The minimum MUST equal the ParamSpec min."""
    if isinstance(v, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        iv = int(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if iv != v:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if iv < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return iv


def _frame(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _as_panel(x: pd.DataFrame) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _prior_beta_stats(r: np.ndarray, m: np.ndarray) -> tuple[float, float, float] | None:
    """OLS fit r ~ 1 + m on a STRICTLY PRIOR window: returns
    (alpha, beta, resid_std) where beta = cov(r, m)/var(m),
    alpha = mean(r) - beta*mean(m) (the intercept), and resid_std is the
    residual std of the same fit (dof = n-2: intercept + slope consumed
    — the OLS residual SE, NOT ddof=1's n-1; review P1 wording fix).

    ``r`` / ``m`` are the prior-window slices (window rows ending at t-1).
    Returns (alpha, beta, resid_std) or None for a fail-closed/degenerate
    window: any NaN/±Inf in either series, or market variance <= eps.
    """
    n = r.size
    if n < 2 or not np.isfinite(r).all() or not np.isfinite(m).all():
        return None
    mc = m - m.mean()
    sxx = float(np.dot(mc, mc))
    if sxx <= _EPS:
        return None  # degenerate market variance -> NaN, never 0
    rc = r - r.mean()
    beta = float(np.dot(mc, rc)) / sxx
    alpha = float(r.mean()) - beta * float(m.mean())
    resid = r - (r.mean() + beta * mc)
    ss_res = float(np.dot(resid, resid))
    dof = n - 2  # intercept + slope consumed
    if dof < 1:
        return None
    resid_std = float(np.sqrt(ss_res / dof))
    return alpha, beta, resid_std


def _prior_map(
    ret: pd.DataFrame, mkt: pd.DataFrame, window: int, fn
) -> np.ndarray:
    """Per-column map over STRICTLY PRIOR windows [r-window, r-1].

    Row r's fit window is the ``window`` rows ENDING at r-1 (the signal row
    r never enters its own fit).  Rows with an incomplete prior window
    (warmup: r < window) stay NaN.  ``fn(prior_r, prior_m, r_ret, r_mkt)``
    applies the fail-closed signal-row policy itself.
    """
    rv = _as_panel(ret)
    mv = _as_panel(mkt)
    rows, cols = rv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            lo = r - window  # prior window is [r-window, r-1]
            if lo < 0:
                continue  # warmup: incomplete prior window
            out[r, c] = fn(rv[lo:r, c], mv[lo:r, c], rv[r, c], mv[r, c])
    return out


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def beta_residual_z(ret: pd.DataFrame, market_ret: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """(r_t - (alpha_prior + beta_prior*m_t)) / resid_std_prior —
    market-model residual z-score with a STRICTLY PRIOR fit.

    alpha, beta and resid_std are estimated by OLS of r on m (intercept
    included: r ~ 1 + m) over the prior window [t-window, t-1] (window rows
    ENDING at t-1; the signal row t NEVER enters its own fit); the residual
    at t is r_t - (alpha + beta*m_t).  Fail-closed: any NaN/±Inf in the
    prior window OR at the signal row -> NaN; degenerate prior market
    variance or zero prior residual std -> NaN, never 0.
    Dimensionless.  Window >= 5 (ParamSpec min == runtime guard)."""
    w = _pi(window, "window", 5)

    def _z(pr, pm, r_t, m_t):
        if not (np.isfinite(r_t) and np.isfinite(m_t)):
            return np.nan
        stats = _prior_beta_stats(pr, pm)
        if stats is None:
            return np.nan
        alpha, beta, resid_std = stats
        if not (np.isfinite(resid_std) and resid_std > _EPS):
            return np.nan
        resid = float(r_t) - (alpha + beta * float(m_t))
        return resid / resid_std

    return _frame(ret, _prior_map(ret, market_ret, w, _z))


def beta_divergence_pct(ret: pd.DataFrame, market_ret: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """(r_t - (alpha_prior + beta_prior*m_t)) / |m_t| — market-model
    divergence ratio.

    Same STRICTLY PRIOR fit as ``beta_residual_z`` (OLS of r on m with
    intercept, r ~ 1 + m, over [t-window, t-1], signal row excluded from
    its own fit); the residual r_t - (alpha + beta*m_t) is normalized by
    the ABSOLUTE market return at t — "how many market-moves worth of
    divergence" (dimensionless, sign preserved, |output| >= 1 means the
    stock moved against/away from its beta-implied move by at least the
    market's own magnitude).  Fail-closed: any NaN/±Inf in the prior window
    or at the signal row -> NaN; degenerate prior market variance -> NaN;
    |m_t| <= eps (flat market — the ratio is undefined) -> NaN, never 0.
    Window >= 5."""
    w = _pi(window, "window", 5)

    def _div(pr, pm, r_t, m_t):
        if not (np.isfinite(r_t) and np.isfinite(m_t)):
            return np.nan
        stats = _prior_beta_stats(pr, pm)
        if stats is None:
            return np.nan
        alpha, beta, _resid_std = stats
        denom = abs(float(m_t))
        if denom <= _EPS:
            return np.nan  # flat market bar: ratio undefined, never 0
        resid = float(r_t) - (alpha + beta * float(m_t))
        return resid / denom

    return _frame(ret, _prior_map(ret, market_ret, w, _div))


def relative_strength_group_pct(
    ret: pd.DataFrame, group: pd.DataFrame, window: int = 20
) -> pd.DataFrame:
    """Trailing compound return minus the group ex-self mean trailing
    compound return (peer-relative momentum gap).

    Per column c (stock): trail_c = prod(1 + r) - 1 over the trailing window
    [t-window+1, t] (window rows ENDING at and INCLUDING t — the trailing
    return is a factual observation of the stock's own realized past, not a
    fitted parameter; the GROUP mean is CROSS-SECTIONAL at row t over the
    same-label columns, self excluded, NaN/±Inf peers excluded).  Output =
    trail_c - mean(trail_peers).  Fail-closed: any NaN/±Inf inside the
    trailing return window -> NaN for that column; zero finite same-label
    peers (ex-self) -> NaN never 0; a degenerate |1+prod| never occurs
    (compound return is finite whenever the window is finite).  Dimensionless
    (return-minus-return).  Window >= 5 (ParamSpec min == runtime guard)."""
    w = _pi(window, "window", 5)
    rv = _as_panel(ret)
    gv = np.asarray(group)
    if rv.shape != gv.shape:
        raise ValueError(
            f"ret and group panels must share shape ({rv.shape} vs {gv.shape})"
        )
    rows, cols = rv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    finite = np.isfinite(rv)
    for r in range(rows):
        lo = r - w + 1
        if lo < 0:
            continue  # warmup
        # trailing compound return per column over [lo, r] (includes t)
        trail = np.full(cols, np.nan, dtype=float)
        for c in range(cols):
            chunk = rv[lo : r + 1, c]
            if np.isfinite(chunk).all():
                trail[c] = float(np.prod(1.0 + chunk) - 1.0)
        g_row = gv[r]
        # per-label sum of finite trailing returns (LOO mean via S - x)
        sums: dict[Any, float] = {}
        counts: dict[Any, int] = {}
        for c in range(cols):
            if np.isfinite(trail[c]):
                lab = g_row[c]
                sums[lab] = sums.get(lab, 0.0) + float(trail[c])
                counts[lab] = counts.get(lab, 0) + 1
        for c in range(cols):
            if not np.isfinite(trail[c]):
                continue  # self fail-closed
            lab = g_row[c]
            m = counts.get(lab, 0) - 1  # finite peers excluding self
            if m < 1:
                continue  # zero finite peers -> NaN never 0
            loo_mean = (sums[lab] - float(trail[c])) / float(m)
            out[r, c] = float(trail[c]) - loo_mean
    return _frame(ret, out)


# ---------------------------------------------------------------------------
# registration (event_state_v2 spec-driven table + single loop idiom)
# ---------------------------------------------------------------------------

_WIN_GE5 = ParamSpec(dtype=int, min=5, default=20, searchable=True, param_role=ParamRole.HORIZON)

_SPECS = [
    (
        "beta_residual_z",
        ["ret", "market_ret", "window"],
        beta_residual_z,
        "(r_t - beta_prior*m_t)/resid_std_prior — market-model residual z; beta and residual std fit on the STRICTLY PRIOR window [t-w, t-1] (signal row excluded from its own fit); NaN/Inf in window -> NaN; degenerate market variance or residual std -> NaN, never 0.",
    ),
    (
        "beta_divergence_pct",
        ["ret", "market_ret", "window"],
        beta_divergence_pct,
        "(r_t - beta_prior*m_t)/|m_t| — market-model divergence ratio; STRICTLY PRIOR beta window [t-w, t-1]; |m_t|<=eps (flat market) -> NaN, never 0; fail-closed NaN/Inf windows.",
    ),
    (
        "relative_strength_group_pct",
        ["ret", "group", "window"],
        relative_strength_group_pct,
        "Trailing compound return minus group ex-self mean trailing compound return (peer-relative momentum gap); trailing window ends at t; LOO group mean per row over same-label finite peers, self excluded; zero finite peers -> NaN, never 0.",
    ),
]

_PARAM_SPECS = {
    "beta_residual_z": {"window": _WIN_GE5},
    "beta_divergence_pct": {"window": _WIN_GE5},
    "relative_strength_group_pct": {"window": _WIN_GE5},
}


def _register(name, params, fn, desc, *, param_specs=None):
    meta = OperatorMetadata(
        name=name,
        category="technical_signal",
        description=desc,
        param_names=list(params),
        return_type="series",
        tags=["pit_safe", "causal", "production_extension"],
        param_specs=dict(param_specs or {}),
    )

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"GroupStateV1_{name}",
        (SeriesOperator,),
        {"metadata": meta, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="technical_signal",
        business_category="technical",
        canonical=name,
        source="technical_group_state_v1",
        backend="pandas_numpy",
        status="production",
    )(cls)
    # same extended-surface contract as technical/exself_cs_v1.py: the module
    # owns its EXTENDED_ONLY_CANONICALS entries at import time (layer
    # governance requires every registered canonical to be classified).
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({name})


for _name, _params, _fn, _desc in _SPECS:
    _register(_name, _params, _fn, _desc, param_specs=_PARAM_SPECS.get(_name))
