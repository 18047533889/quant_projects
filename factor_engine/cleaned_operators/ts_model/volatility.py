# -*- coding: utf-8 -*-
"""GARCH / GJR-GARCH and HAR-RV volatility operators (P2, experimental).

GARCH parameters are estimated per window by quasi-maximum likelihood
(variance targeting with a small numeric optimisation over alpha/beta).  These
are high-cost operators and must not enter default random factor search.

Typed input semantics (audit round-3, item 33): GARCH / GJR / HAR-from-return
operators require a RETURN / stationary series — fitting a non-stationary raw
price level is statistically wrong (variance-targeting on a price level is
meaningless and the conditional-variance recursion explodes).  Their metadata
declares ``input_units={"x": "return"}`` and the kernels reject a clearly
non-stationary price-level input (sd(level)/sd(diff) >> 1) at runtime.

HAR naming (audit round-3, item 34): the HAR kernel predicts the next-period
*realized variance* (RV).  The sqrt-canonical is a volatility forecast and is
named ``*_vol_forecast``; the raw-RV canonical is ``*_var_forecast``.  The two
are distinct, honestly named canonicals rather than one name claiming the
other's statistic.

Missing-gap policy (model-audit M-084): all GARCH / GJR kernels follow
``_GARCH_MISSING_POLICY = "fail_closed_on_gap"`` — any NaN inside the fit window
makes that row's output NaN (fail-closed).  Gaps are NEVER dropped-and-rescaled;
the conditional-variance recursion is not re-anchored across a hole.

HAR min-train split (model-audit M-086): the HAR kernel's ``window`` (the rolling
feature span) and ``_HAR_MIN_TRAIN_OBS`` (the minimum number of valid training
rows the OLS design must have) are two SEPARATE quantities — a policy choice, not
a derived value of ``window``.

Legacy HAR aliases (model-audit M-088): ``ts_har_rv_forecast`` and
``ts_har_rv_innovation_z`` are genuine registry compat aliases (not separate
canonicals) for ``ts_har_rv_next_vol_forecast`` and
``ts_har_rv_forecast_error_z`` respectively — mining must not double-search the
same kernel under two independent research candidates.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import ParamRole, ParamSpec, SeriesOperator, register_operator
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.ts_model._rolling_core import fit_linear_model_checked, frame_like, metadata

_CANONICALS: list[str] = []

# Model-audit Phase 4 (search-space hygiene): the GARCH / GJR / HAR scalars.
# Every canonical in this module exposes exactly one scalar tuning parameter —
# ``window`` (or the HAR ``window`` feature span) — which is the alpha horizon
# (HORIZON, searched).  The GARCH alpha/beta/gamma and HAR design are estimated
# inside the kernel, never user-searched (M-115/M-162/M-170).
#
# P1 (volatility parameter-domain governance): GARCH/GJR and HAR can never
# produce a finite output below a minimum trailing window — the strict-prior
# GARCH fit segment needs >= _GARCH_MIN_FIT_OBS rows and the HAR OLS design
# needs >= _HAR_MIN_WINDOW rows.  ``window`` is split into per-family ParamSpecs
# so a guaranteed-all-NaN window is rejected at the call boundary
# (binding-time raise via ``ParamSpec.min`` in ``validate_operator_call``)
# instead of silently returning all NaN.  ``window`` remains a MAX LOOKBACK
# (expanding warmup: a trailing segment shorter than ``window`` still fits as
# soon as the family's minimum sample is present).
_VOL_PARAM_SPECS: dict[str, ParamSpec] = {
    "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON, searchable=True),
}
_GARCH_MIN_WINDOW = 13   # fit_seg = window - 1 must be >= _GARCH_MIN_FIT_OBS
_GARCH_MIN_FIT_OBS = 12  # _fit_garch/_fit_gjr both refuse below this
_HAR_MIN_WINDOW = 30     # _har_rv needs n >= 30 for the OLS design
_GARCH_PARAM_SPECS: dict[str, ParamSpec] = {
    "window": ParamSpec(dtype=int, min=_GARCH_MIN_WINDOW, param_role=ParamRole.HORIZON, searchable=True),
}
_HAR_PARAM_SPECS: dict[str, ParamSpec] = {
    "window": ParamSpec(dtype=int, min=_HAR_MIN_WINDOW, param_role=ParamRole.HORIZON, searchable=True),
}
# P1: window semantics contract — a max lookback, not a strict full window.
_WINDOW_SEMANTICS = "max_lookback"

try:
    from scipy.optimize import minimize as _minimize
except Exception:  # pragma: no cover
    _minimize = None


# Audit round-3 (item 33): a raw nonstationary price level is statistically
# indistinguishable from a "constant + trend" path — sd(level)/sd(diff) >> 1.
# Returns / stationary series are centred and take both signs, so the ratio is
# O(1).  The threshold mirrors the repo-wide price-vs-return heuristic
# (cf. spectral_ext._looks_like_price_level / extreme_tail._reject_price_level).
_PRICE_LEVEL_RATIO_THRESHOLD = 5.0
_MIN_JUDGE_ROWS = 8
_EPS = 1e-12

# Model-audit M-084: explicit missing-gap policy for the GARCH / GJR kernels.
# Any NaN inside the fit window fails that row's output to NaN (fail-closed);
# gaps are never dropped-and-rescaled and the variance recursion is never
# re-anchored across a hole.  This makes the previously-incidental NaN
# propagation a documented, versioned contract.
_GARCH_MISSING_POLICY = "fail_closed_on_gap"

# Model-audit M-086: the minimum number of valid training rows the HAR OLS
# design must have.  This is a SEPARATE policy minimum from the rolling
# ``window`` (the feature span) — it is not derived from ``window`` and is
# versioned independently.  It is intentionally NOT a user-facing parameter.
_HAR_MIN_TRAIN_OBS = 25

# P1 (HAR sample-coverage governance): a HAR model must retain a minimum
# FRACTION of its rolling ``window`` in effective training rows, not just meet
# the absolute ``_HAR_MIN_TRAIN_OBS`` floor — otherwise a window=120 model could
# keep estimating on 25/120 = 1/5 of the history (the audit finding).  The
# double gate is ``N_effective >= _HAR_MIN_TRAIN_OBS AND
# N_effective >= _HAR_MIN_COVERAGE * window``.  Versioned; a change to this
# constant is a semantic change to every ts_har_* output.
_HAR_MIN_COVERAGE = 0.6

# Model-audit M-083: shared MLE fit cache.  ``_fit_garch`` / ``_fit_gjr``
# (variance-targeting Nelder-Mead) dominate the cost of every GARCH / GJR
# canonical.  Within a single operator call (``_calculate_series`` ->
# ``_apply``) identical fit segments (e.g. duplicate columns, or the same
# trailing window) are fit ONCE and reused.  The key is the EXACT input bytes so
# reuse is bit-identical.  The cache is cleared at the start and end of every
# ``_calculate_series`` (scoped to one call — no cross-row state leakage) and is
# additionally size-capped so direct kernel calls can never grow it unboundedly.
_GARCH_FIT_CACHE: dict[tuple[str, bytes], "tuple | None"] = {}
_GARCH_FIT_CACHE_MAX_ENTRIES = 4096
_GARCH_FIT_CACHE_KIND_GARCH = "garch"
_GARCH_FIT_CACHE_KIND_GJR = "gjr"
#: Sentinel distinguishing "not cached" from a cached ``None`` (failed fit).
_GARCH_CACHE_MISS = object()

# Model-audit M-081 telemetry: canonicals whose kernel intentionally fits
# THROUGH the current row (in-sample, descriptive, fit_cutoff_offset=0).  The
# reconciler consumes this to add explicit ``ModelTimingContract`` entries and to
# re-evaluate lanes.  ``ts_gjr_leverage`` is the ONLY GARCH/GJR canonical here —
# every other GARCH/GJR canonical fits strictly on ``seg[:-1]``.
_IN_SAMPLE_FIT_THROUGH_T: dict[str, dict[str, Any]] = {
    "ts_gjr_leverage": {
        "fit_through_t": True,
        "fit_cutoff_offset": 0,
        "descriptive": True,
        "flag_for_reconciler": (
            "add explicit ModelTimingContract(fit_cutoff_offset=0, descriptive); "
            "re-evaluate lane EXPENSIVE_CERTIFIED_ALPHA -> DIAGNOSTIC_RESEARCH "
            "unless a strict-prior GJR-leverage variant is added"
        ),
    },
}


def _looks_like_price_level(vals: np.ndarray) -> bool:
    finite = vals[np.isfinite(vals)]
    if finite.size < _MIN_JUDGE_ROWS:
        return False  # too short to judge; the kernel's min-window fails closed
    sd_level = float(np.std(finite))
    if not np.isfinite(sd_level) or sd_level <= _EPS:
        return False  # constant / degenerate -> not a level path
    sd_diff = float(np.std(np.diff(finite)))
    return sd_level / max(sd_diff, _EPS) > _PRICE_LEVEL_RATIO_THRESHOLD


def _register(name: str, description: str, params: list[str], unit: str, fn,
              *, input_units: dict[str, str] | None = None,
              output_unit: str | None = None,
              reject_price_level: bool = False,
              param_specs: "dict[str, ParamSpec] | None" = None):
    @register_operator(
        name=name,
        category="time_series_regression",
        business_category="time_series_regression",
        canonical=name,
        source="ts_model.volatility",
        backend="pandas_numpy",
        status="experimental",
    )
    class _VolOp(SeriesOperator):
        metadata = metadata(name, description, params, unit=unit, cost=8,
                            input_units=input_units, output_unit=output_unit,
                            param_specs=param_specs if param_specs is not None else _VOL_PARAM_SPECS)
        # P1: window semantics contract — max lookback (not strict full window).
        metadata.window_semantics = _WINDOW_SEMANTICS

        def _calculate_series(self, *args, **kwargs):
            # Statistical domain checks are evaluated by each rolling kernel on
            # its own trailing history.  A whole-frame precheck would let a
            # future suffix erase otherwise valid historical outputs (M04).
            # M-083: scope the shared-fit cache to this one operator call.  The
            # cache must never leak fits across rows / across calls.
            _GARCH_FIT_CACHE.clear()
            try:
                return fn(*args, **kwargs)
            finally:
                _GARCH_FIT_CACHE.clear()

    _CANONICALS.append(name)
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_research_only({name})
    return _VolOp


def _fit_garch(rets: np.ndarray) -> tuple[float, float, float] | None:
    """Return (omega, alpha, beta) for GARCH(1,1) via variance-targeting MLE.

    The optimisation result is only accepted when the solver reports success
    and the parameters lie in the stationary region (alpha, beta >= 0 and
    alpha + beta < 1).  A failed / non-converged fit returns ``None`` so the
    caller emits NaN rather than the solver's last (possibly invalid) iterate.
    """
    if _minimize is None or len(rets) < 10 or np.std(rets) <= 1e-12:
        return None
    long_var = float(np.var(rets))

    def _nll(params):
        a, b = params
        if a < 1e-6 or b < 1e-6 or a + b >= 0.999:
            return 1e12
        w = max(long_var * (1 - a - b), 1e-12)
        # Unify the likelihood's initial variance with the output recursion's
        # seed (_variance_path callers pass var(fit_seg)): both backcast from the
        # unconditional variance long_var = omega/(1-alpha-beta), not the
        # constant term omega alone.
        h = np.full(len(rets), long_var, dtype=float)
        for t in range(1, len(rets)):
            h[t] = _variance_step(h[t - 1], rets[t - 1], w, a, b)
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.sum(np.log(h) + rets ** 2 / h))
    try:
        res = _minimize(_nll, np.array([0.05, 0.9]), method="Nelder-Mead",
                        options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-6})
        if not getattr(res, "success", False):
            return None
        a, b = float(res.x[0]), float(res.x[1])
        if not (np.isfinite(a) and np.isfinite(b)):
            return None
        if a < 1e-6 or b < 1e-6 or a + b >= 0.999:
            return None
        w = max(long_var * (1 - a - b), 1e-12)
        return w, a, b
    except Exception:
        return None


def _fit_garch_cached(rets: np.ndarray) -> "tuple[float, float, float] | None":
    """Variance-targeting GARCH MLE with a per-call shared-fit cache (M-083).

    The key is the exact input bytes, so a cache hit is bit-identical to a fresh
    fit.  ``None`` (failed fit) is cached too, via the ``_GARCH_CACHE_MISS``
    sentinel.  The cache is cleared per ``_calculate_series`` and size-capped.
    """
    key = (_GARCH_FIT_CACHE_KIND_GARCH, rets.tobytes())
    hit = _GARCH_FIT_CACHE.get(key, _GARCH_CACHE_MISS)
    if hit is not _GARCH_CACHE_MISS:
        return hit  # type: ignore[return-value]
    if len(_GARCH_FIT_CACHE) >= _GARCH_FIT_CACHE_MAX_ENTRIES:
        _GARCH_FIT_CACHE.clear()
    result = _fit_garch(rets)
    _GARCH_FIT_CACHE[key] = result
    return result


def _variance_step(
    previous_variance: float,
    shock: float,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float = 0.0,
    *,
    asymmetric: bool = False,
) -> float:
    """One authoritative GARCH/GJR variance update.

    ``shock`` is the zero-mean innovation under this module's declared
    constant-zero mean model; callers with a fitted mean must pass the
    residual, never the raw return.  The return value is conditional variance,
    not volatility.  Non-finite state or shock fails closed.
    """
    if not np.isfinite(previous_variance) or not np.isfinite(shock):
        return np.nan
    leverage = gamma if asymmetric and shock < 0.0 else 0.0
    return float(omega + (alpha + leverage) * shock**2 + beta * previous_variance)


def _variance_path(
    seg: np.ndarray,
    w: float,
    a: float,
    b: float,
    h_init: float,
    gamma: float = 0.0,
    *,
    asymmetric: bool = False,
) -> float:
    """Recurse the GARCH/GJR conditional variance over ``seg``.

    ``h_init`` seeds the recursion and the result is the conditional variance
    governing the LAST observation of ``seg`` (information strictly before it).
    The caller chooses ``h_init`` so the current return can never seed its own
    standardisation denominator (audit P0: the init must exclude r_t).
    """
    h = h_init
    for i in range(1, len(seg)):
        h = _variance_step(h, seg[i - 1], w, a, b, gamma, asymmetric=asymmetric)
    return h


def _garch_path(rets: np.ndarray, window: int, stat: str, asymmetric: bool, r: float) -> float:
    """Rolling GARCH(1,1) / GJR statistic.

    Timing convention: the returned conditional variance ``h_last`` governs the
    *last* observed return (computed from information strictly before it), and
    ``h_next`` applies the same GARCH/GJR update as the history path, including
    ``gamma * I(r_t < 0)`` for GJR, and is the one-step-ahead forecast for the
    next period.  The standardised shock divides ``r_t`` by ``sqrt(h_last)`` —
    the variance that actually governed it.  Returns are innovations under the
    module's explicit constant-zero mean convention.

    P1 (parameter domain): ``window`` below ``_GARCH_MIN_WINDOW`` is a
    guaranteed-all-NaN combination (the strict-prior fit segment ``seg[:-1]``
    cannot reach ``_GARCH_MIN_FIT_OBS`` rows), so it raises at binding time
    instead of silently emitting NaN for every row.
    """
    w = int(window)
    if w < _GARCH_MIN_WINDOW:
        raise ValueError(
            f"GARCH/GJR window={w} is below the minimum {_GARCH_MIN_WINDOW}: "
            f"the strict-prior fit segment (window-1) needs at least "
            f"{_GARCH_MIN_FIT_OBS} rows, so a smaller window is all-NaN by "
            f"construction (excluded by the parameter-domain gate)."
        )
    if len(rets) < max(w, _GARCH_MIN_FIT_OBS + 1):
        return np.nan
    seg = rets[-w:]
    if _looks_like_price_level(seg):
        return np.nan
    if not np.all(np.isfinite(seg)):
        return np.nan
    # R35-P0-M01/M02/M03 (timing contract consistency): ALL GARCH/GJR statistics
    # fit their parameters strictly on <= t-1 (``fit_seg = seg[:-1]``), matching
    # ``ModelTimingContract.fit_cutoff_offset=1``.  The current return ``r_t`` is
    # never part of its own parameter fit — it only enters the *one-step state
    # update* (the shared ``_variance_step`` for the forecast) or the
    # standardised shock denominator.  This is the "Scheme B" definition: params
    # strictly prior, current shock used only for one-step update.  The variance
    # recursion is seeded from the variance of that same fit segment so no leak
    # reaches the output through the initial value either.
    fit_seg = seg[:-1]
    if len(fit_seg) < _GARCH_MIN_FIT_OBS:
        return np.nan
    # Missing-gap policy (M-084): any NaN inside the fit window fails this row
    # to NaN below — the MLE returns None on a non-finite input and we never
    # drop-and-rescale across the gap.  The shared-fit cache (M-083) is scoped
    # per operator call and bit-identical, so this is a pure reuse optimisation.
    if asymmetric:
        # GJR: h_t = w + (a + gamma*I(r<0))*r^2 + b*h
        params = _fit_gjr_cached(fit_seg)
    else:
        params = _fit_garch_cached(fit_seg)
    if params is None:
        return np.nan
    gamma = 0.0
    if asymmetric:
        w, a, gamma, b = params
    else:
        w, a, b = params
    if stat == "persistence":
        return float(a + b) if not asymmetric else float(a + 0.5 * gamma + b)
    # Seed the variance recursion from the fit segment only (excludes the
    # current return for the shock stat — no self-leak through the init).
    h_last = _variance_path(
        seg, w, a, b, float(np.var(fit_seg)), gamma, asymmetric=asymmetric
    )
    if stat == "forecast":
        # h_last governs the current return; the next-period forecast conditions
        # on it.
        h_next = _variance_step(
            h_last, seg[-1], w, a, b, gamma, asymmetric=asymmetric
        )
        if not np.isfinite(h_next):
            return np.nan
        return float(np.sqrt(max(h_next, 1e-12)))
    # standardized shock of the last return uses the variance that governed it
    if not np.isfinite(seg[-1]):
        return np.nan
    return float(seg[-1] / np.sqrt(max(h_last, 1e-12)))


def _fit_gjr(rets: np.ndarray) -> tuple[float, float, float, float] | None:
    if _minimize is None or len(rets) < 12 or np.std(rets) <= 1e-12:
        return None
    long_var = float(np.var(rets))

    def _nll(p):
        a, g, b = p
        if min(a, b, g) < 1e-6 or a + 0.5 * g + b >= 0.999:
            return 1e12
        w = max(long_var * (1 - a - 0.5 * g - b), 1e-12)
        # Unify with the output recursion seed: backcast from the unconditional
        # variance long_var (= omega/(1-a-0.5g-b)), matching _variance_path.
        h = np.full(len(rets), long_var, dtype=float)
        for t in range(1, len(rets)):
            h[t] = _variance_step(
                h[t - 1], rets[t - 1], w, a, b, g, asymmetric=True
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.sum(np.log(h) + rets ** 2 / h))
    try:
        res = _minimize(_nll, np.array([0.03, 0.05, 0.9]), method="Nelder-Mead",
                        options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-6})
        if not getattr(res, "success", False):
            return None
        a, g, b = float(res.x[0]), float(res.x[1]), float(res.x[2])
        if not (np.isfinite(a) and np.isfinite(g) and np.isfinite(b)):
            return None
        if min(a, b, g) < 1e-6 or a + 0.5 * g + b >= 0.999:
            return None
        w = max(long_var * (1 - a - 0.5 * g - b), 1e-12)
        return w, a, g, b
    except Exception:
        return None


def _fit_gjr_cached(rets: np.ndarray) -> "tuple[float, float, float, float] | None":
    """GJR-GARCH MLE with a per-call shared-fit cache (M-083).  Same contract as
    :func:`_fit_garch_cached` but for the asymmetric (leverage) fit."""
    key = (_GARCH_FIT_CACHE_KIND_GJR, rets.tobytes())
    hit = _GARCH_FIT_CACHE.get(key, _GARCH_CACHE_MISS)
    if hit is not _GARCH_CACHE_MISS:
        return hit  # type: ignore[return-value]
    if len(_GARCH_FIT_CACHE) >= _GARCH_FIT_CACHE_MAX_ENTRIES:
        _GARCH_FIT_CACHE.clear()
    result = _fit_gjr(rets)
    _GARCH_FIT_CACHE[key] = result
    return result


def _apply(x: pd.DataFrame, fn) -> pd.DataFrame:
    xv = x.to_numpy(dtype=float)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            out[row, col] = fn(xv[: row + 1, col])
    return frame_like(x, out)


_RETURN_INPUT = {"x": "return"}
_register("ts_garch_next_vol_forecast", "GARCH(1,1) 下一期条件波动率（观测最后收益之后）。window=max lookback（非严格满窗），最小 window=13。", ["x", "window"], "volatility",
           lambda x, window=120: _apply(x, lambda v: _garch_path(v, int(window), "forecast", False, 0.0)),
           input_units=_RETURN_INPUT, output_unit="volatility", reject_price_level=True,
           param_specs=_GARCH_PARAM_SPECS)
_register("ts_garch_vol_surprise", "GARCH 波动率意外：最近收益平方 / 条件方差 - 1。window=max lookback（非严格满窗），最小 window=13。", ["x", "window"], "level",
           lambda x, window=120: _apply(x, lambda v: _garch_vol_surprise(v, int(window))),
           input_units=_RETURN_INPUT, output_unit="level", reject_price_level=True,
           param_specs=_GARCH_PARAM_SPECS)
_register("ts_garch_persistence", "GARCH(1,1) 波动持续 alpha+beta。window=max lookback（非严格满窗），最小 window=13。", ["x", "window"], "level",
           lambda x, window=120: _apply(x, lambda v: _garch_path(v, int(window), "persistence", False, 0.0)),
           input_units=_RETURN_INPUT, output_unit="level", reject_price_level=True,
           param_specs=_GARCH_PARAM_SPECS)
_register("ts_garch_standardized_shock", "GARCH 标准化冲击 return/条件波动率（参数于 t-1 及以前拟合）。window=max lookback（非严格满窗），最小 window=13。", ["x", "window"], "level",
           lambda x, window=120: _apply(x, lambda v: _garch_path(v, int(window), "shock", False, 0.0)),
           input_units=_RETURN_INPUT, output_unit="level", reject_price_level=True,
           param_specs=_GARCH_PARAM_SPECS)
# Deprecated alias for the next-period forecast: a registry compatibility
# alias, NOT a separate canonical — mining must not double-search the same
# kernel under two independent research candidates.
OperatorRegistry.register_compat_alias(
    "ts_garch_vol_forecast",
    "ts_garch_next_vol_forecast",
    migration_reason="legacy name for the identical next-period GARCH forecast",
    deprecated_since="2026-08",
    removal_version="1.0",
)
_register("ts_gjr_garch_vol_forecast", "GJR-GARCH 波动预测（杠杆效应）。window=max lookback（非严格满窗），最小 window=13。", ["x", "window"], "volatility",
           lambda x, window=120: _apply(x, lambda v: _garch_path(v, int(window), "forecast", True, 0.0)),
           input_units=_RETURN_INPUT, output_unit="volatility", reject_price_level=True,
           param_specs=_GARCH_PARAM_SPECS)
_register("ts_gjr_leverage", "GJR 负收益冲击系数。window=max lookback（非严格满窗），最小 window=13。", ["x", "window"], "level",
           lambda x, window=120: _apply(x, lambda v: _gjr_leverage(v, int(window))),
           input_units=_RETURN_INPUT, output_unit="level", reject_price_level=True,
           param_specs=_GARCH_PARAM_SPECS)


def _garch_vol_surprise(vals: np.ndarray, window: int) -> float:
    """Realised variance of the last return relative to its conditional variance.

    ``rv_t / h_t - 1`` where ``h_t`` is the GARCH variance that governed the
    last observed return (information strictly before it).

    P1 (parameter domain): ``window`` below ``_GARCH_MIN_WINDOW`` is a
    guaranteed-all-NaN combination and raises at binding time.
    """
    w = int(window)
    if w < _GARCH_MIN_WINDOW:
        raise ValueError(
            f"GARCH window={w} is below the minimum {_GARCH_MIN_WINDOW}: "
            f"a smaller window is all-NaN by construction (parameter-domain gate)."
        )
    if len(vals) < max(w, _GARCH_MIN_FIT_OBS + 1):
        return np.nan
    seg = vals[-w:]
    if _looks_like_price_level(seg):
        return np.nan
    # P0-040 / P0 (this audit): params fit on <= t-1 (exclude the current
    # return) AND the variance recursion is seeded from that same fit segment,
    # so rv_t/h_t is a genuine out-of-sample surprise and the current return
    # cannot leak into its own denominator through the initial variance.
    if len(seg) < _GARCH_MIN_FIT_OBS + 1:
        return np.nan
    # Missing-gap policy (M-084): a NaN anywhere in the fit window yields NaN
    # here (fail-closed, never drop-and-rescale).  Shared-fit cache (M-083).
    params = _fit_garch_cached(seg[:-1])
    if params is None:
        return np.nan
    w, a, b = params
    h_last = _variance_path(seg, w, a, b, float(np.var(seg[:-1])))
    if not np.isfinite(seg[-1]) or h_last <= 1e-12:
        return np.nan
    return float(seg[-1] ** 2 / h_last - 1.0)


def _gjr_leverage(vals: np.ndarray, window: int) -> float:
    """GJR leverage coefficient gamma (asymmetric negative-return shock).

    M-081 timing (explicit): this is the ONLY GARCH/GJR canonical whose kernel
    fits its parameters THROUGH the current row — ``_fit_gjr(vals[-window:])``
    INCLUDES ``r_t``.  It is an in-sample, descriptive estimate
    (``fit_cutoff_offset=0``), NOT a predictive timing contract, and it is
    deliberately left unchanged: the in-sample estimate is honest (the current
    return genuinely informs its own leverage).  Reconciler must (a) add an
    explicit ``ModelTimingContract(fit_cutoff_offset=0, descriptive)`` and
    (b) re-evaluate the lane (currently EXPENSIVE_CERTIFIED_ALPHA ->
    DIAGNOSTIC_RESEARCH unless a strict-prior GJR-leverage variant is added).
    See the ``_IN_SAMPLE_FIT_THROUGH_T`` module telemetry.

    P1 (parameter domain): ``window`` below ``_GARCH_MIN_WINDOW`` is a
    guaranteed-all-NaN combination and raises at binding time.
    """
    w = int(window)
    if w < _GARCH_MIN_WINDOW:
        raise ValueError(
            f"GJR window={w} is below the minimum {_GARCH_MIN_WINDOW}: "
            f"a smaller window is all-NaN by construction (parameter-domain gate)."
        )
    if len(vals) < max(w, _GARCH_MIN_FIT_OBS + 1):
        return np.nan
    # Missing-gap policy (M-084): NaN anywhere in the fit window -> NaN below.
    # Shared-fit cache (M-083).
    seg = vals[-w:]
    if _looks_like_price_level(seg):
        return np.nan
    params = _fit_gjr_cached(seg)  # fit-through-t: r_t IS in the fit
    return np.nan if params is None else float(params[2])


def _har_rv(rv: np.ndarray, window: int, stat: str) -> float:
    """HAR model whose input is a *realized variance* series (not squared here).

    Features are RV_t, weekly mean and monthly mean of RV; the target is
    RV_{t+1}, so the model genuinely forecasts next-period RV from today's
    components.  This is the ``ts_har_rv_*`` / ``ts_har_from_return_*`` family
    contract.  ``ts_har_from_return_*`` squares its return input before calling
    this kernel.

    P1 (parameter domain / sample coverage): ``window`` below ``_HAR_MIN_WINDOW``
    is a guaranteed-all-NaN combination (the OLS design cannot form) and raises at
    binding time.  A finite output additionally requires the DOUBLE sample-
    coverage gate — ``N_effective >= _HAR_MIN_TRAIN_OBS`` AND
    ``N_effective >= _HAR_MIN_COVERAGE * window`` — so a window=120 model can no
    longer keep estimating on 25/120 = 1/5 of the history.
    """
    w = int(window)
    if w < _HAR_MIN_WINDOW:
        raise ValueError(
            f"HAR window={w} is below the minimum {_HAR_MIN_WINDOW}: the OLS "
            f"design cannot form below this (all-NaN by construction, "
            f"parameter-domain gate)."
        )
    seg = rv[-w:]
    n = len(seg)
    if n < _HAR_MIN_WINDOW:
        return np.nan
    daily = seg
    weekly = pd.Series(seg).rolling(5).mean().to_numpy()
    monthly = pd.Series(seg).rolling(22).mean().to_numpy()
    X = np.column_stack([np.ones(n), daily, weekly, monthly])
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(seg)

    def _coverage_ok(n_eff: int) -> bool:
        # P1 double gate: absolute floor AND fractional coverage of the window.
        return n_eff >= _HAR_MIN_TRAIN_OBS and n_eff >= _HAR_MIN_COVERAGE * w

    if stat in ("forecast", "var_forecast"):
        # Next-period forecast RV_{t+1}: target = next RV, features today.
        target = np.concatenate([seg[1:], [np.nan]])
        valid_t = np.isfinite(target) & valid
        # M-086: _HAR_MIN_TRAIN_OBS is a policy minimum separate from ``window``;
        # P1 adds the fractional-coverage floor alongside it.
        if not _coverage_ok(int(valid_t.sum())):
            return np.nan
        Xs = X[valid_t]
        y = target[valid_t]
        # ModelDesignGate: reject rank-deficient / ill-conditioned HAR design.
        beta = fit_linear_model_checked(Xs, y)
        if beta is None:
            return np.nan
        pred = float(np.dot(X[-1], beta))
        # Audit round-3 (item 34): the HAR model predicts the next-period
        # *realized variance* (the model is linear in RV components).  The
        # sqrt-canonical is honestly named a volatility forecast; the raw-RV
        # canonical keeps the variance scale.
        if stat == "forecast":
            return float(np.sqrt(max(pred, 0.0)))
        return float(max(pred, 0.0))
    # Current-period surprise RV_t - forecast(RV_t | t-1).  P0-039: the
    # training window is capped strictly before the last observation (feature
    # rows 0..n-3, targets rv[1..n-2]), and the forecast is made from the
    # t-1 feature row — never from X_t, and never mixing a t+1 forecast with a
    # current residual.
    fit_rows = np.arange(n - 2)  # 0 .. n-3
    # X_s and its one-step-ahead y_s keep the same original row identity.
    # Never filter/compress either side independently (M05).
    fit_valid = valid[fit_rows] & np.isfinite(seg[fit_rows + 1])
    Xs = X[fit_rows][fit_valid]
    y = seg[fit_rows + 1][fit_valid]
    # M-086: _HAR_MIN_TRAIN_OBS is a policy minimum separate from ``window``;
    # P1 adds the fractional-coverage floor alongside it.
    if not _coverage_ok(int(Xs.shape[0])) or Xs.shape[0] <= Xs.shape[1]:
        return np.nan
    # ModelDesignGate: an ill-conditioned HAR design fails closed (NaN).
    beta = fit_linear_model_checked(Xs, y)
    if beta is None:
        return np.nan
    if not np.all(np.isfinite(X[n - 2])) or not np.isfinite(seg[-1]):
        return np.nan
    pred_t = float(np.dot(X[n - 2], beta))  # forecast RV_{n-1} from t-1 info
    sd = float(np.std(y - Xs @ beta))
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.nan
    return float((seg[-1] - pred_t) / sd)


def _har_from_return(rets: np.ndarray, window: int, stat: str) -> float:
    """HAR over daily returns: squares them into an RV series internally."""
    if _looks_like_price_level(rets[-int(window):]):
        return np.nan
    return _har_rv(rets ** 2, window, stat)


# The historic operators accept a realized-variance panel (parameter named
# ``rv``); they no longer square it a second time.  Audit round-3 (item 34):
# the kernel predicts the next-period realized VARIANCE; the sqrt output is a
# volatility forecast and is named ``ts_har_rv_next_vol_forecast``, while the
# raw-RV forecast is the distinct ``ts_har_rv_next_var_forecast`` canonical.
_register("ts_har_rv_next_vol_forecast", "HAR-RV 下一期已实现波动率预测（对下一期 RV 预测取平方根，输入已实现方差）。window=max lookback（非严格满窗），最小 window=30；有效行需同时满足 N>=25 与 N/window>=0.6 双约束。", ["rv", "window"], "volatility",
           lambda x, window=120: _apply(x, lambda v: _har_rv(v, int(window), "forecast")),
           input_units={"rv": "realized_variance"}, output_unit="volatility",
           param_specs=_HAR_PARAM_SPECS)
OperatorRegistry.register_compat_alias(
    "ts_har_rv_next_forecast",
    "ts_har_rv_next_vol_forecast",
    migration_reason="legacy name claimed a realized-VARIANCE forecast but the kernel returns sqrt(RV) — a volatility forecast; renamed honestly",
    deprecated_since="2026-08",
    removal_version="1.0",
)
_register("ts_har_rv_next_var_forecast", "HAR-RV 下一期已实现方差预测（不取平方根，输入已实现方差）。window=max lookback（非严格满窗），最小 window=30；有效行双约束。", ["rv", "window"], "variance",
           lambda x, window=120: _apply(x, lambda v: _har_rv(v, int(window), "var_forecast")),
           input_units={"rv": "realized_variance"}, output_unit="variance",
           param_specs=_HAR_PARAM_SPECS)
_register("ts_har_rv_forecast_error_z", "RV 相对 HAR 预测的标准化偏差（输入已实现方差）。window=max lookback（非严格满窗），最小 window=30；有效行双约束。", ["rv", "window"], "level",
           lambda x, window=120: _apply(x, lambda v: _har_rv(v, int(window), "innovation_z")),
           param_specs=_HAR_PARAM_SPECS)
# The from-return variants square the daily return panel internally.
_register("ts_har_from_return_next_vol", "HAR 基于日收益的下一期波动预测（内部平方为 RV）。window=max lookback（非严格满窗），最小 window=30；有效行双约束。", ["ret", "window"], "volatility",
           lambda x, window=120: _apply(x, lambda v: _har_from_return(v, int(window), "forecast")),
           input_units=_RETURN_INPUT, output_unit="volatility", reject_price_level=True,
           param_specs=_HAR_PARAM_SPECS)
_register("ts_har_from_return_forecast_error_z", "日收益平方 RV 相对 HAR 预测的标准化偏差。window=max lookback（非严格满窗），最小 window=30；有效行双约束。", ["ret", "window"], "level",
           lambda x, window=120: _apply(x, lambda v: _har_from_return(v, int(window), "innovation_z")),
           input_units=_RETURN_INPUT, output_unit="level", reject_price_level=True,
           param_specs=_HAR_PARAM_SPECS)
# Model-audit M-088: ``ts_har_rv_forecast`` and ``ts_har_rv_innovation_z`` are
# genuine registry compat aliases (NOT separate canonicals) — each is byte-for-
# byte the same kernel/stat as its target canonical (``_har_rv`` ``forecast`` /
# ``innovation_z`` respectively), so mining must never double-search the same
# kernel under two independent research candidates.
OperatorRegistry.register_compat_alias(
    "ts_har_rv_forecast",
    "ts_har_rv_next_vol_forecast",
    migration_reason="legacy duplicate of the identical HAR next-period volatility forecast (same kernel _har_rv 'forecast', sqrt(RV))",
    deprecated_since="2026-08",
    removal_version="1.0",
)
OperatorRegistry.register_compat_alias(
    "ts_har_rv_innovation_z",
    "ts_har_rv_forecast_error_z",
    migration_reason="legacy duplicate of the identical HAR innovation-z statistic (same kernel _har_rv 'innovation_z')",
    deprecated_since="2026-08",
    removal_version="1.0",
)
