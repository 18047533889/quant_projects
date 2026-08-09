# -*- coding: utf-8 -*-
"""Extreme-value / event-count primitives (2026-08-08 Gemini V2 round).

* ``ts_pickands_tail_index`` — the Pickands (1975) tail-index estimator
  ``ξ̂ = (1/ln 2)·ln( (X_(n-k) - X_(n-2k)) / (X_(n-2k) - X_(n-4k)) )``.  This is
  the third estimator of the tail shape beside the existing Hill and GPD-PWM
  operators — estimator diversity, not a new tail concept, hence P1/extended.
  ``k`` is a fixed scan offset (not a search parameter surface).  For the
  lower tail the estimator is applied to the negated series.
* ``ts_evt_threshold_stability`` — a stability score ``1/(1+σ)`` where
  ``σ`` is the standard deviation of the Hill estimate ``ξ̂_Hill(k)`` as the
  threshold moves over ``k ∈ [k_min, k_max]``.  R5 P1-10/11: only the top-k
  tail used for each estimate must be positive (a signed return series with a
  few opposite-direction values no longer fails the whole window), and the
  output is monotone-increasing in stability (1 = perfectly stable tail,
  → 0 = fragile / regime-mixed).  Never a p-value.  R14-P1-22: the declared
  relational contract is ``window >= 2*k_max + 2`` — identical to the runtime
  guard — so a parameter region that is guaranteed all-NaN at runtime is never
  declared legal.
* ``event_allan_factor`` — the true Allan factor of a daily 0/1 event series
  at a single scale: ``AF(τ) = E[(N_{k+1} − N_k)²] / (2·E[N_k])`` over adjacent
  blocks of length ``τ`` (≈ 1 for a Poisson process, > 1 for clustered events,
  < 1 for regular/anti-clustered).  R5 P0-07: the previous kernel returned the
  Fano factor ``Var(N)/Mean(N)`` — a different statistic that also duplicated
  ``event_fano_factor`` — and it wrongly accepted non-event (continuous) input;
  the input contract is now {0, 1, NaN} and non-binary rows fail closed.
* ``event_allan_scaling_slope`` — regression slope of ``log2(AF(τ))`` against
  ``log2(τ)`` over dyadic scales (≈ 0 for Poisson, > 0 clustered, < 0
  regular).
* ``event_allan_log_mean`` — mean ``log2(AF(τ))`` over the dyadic scales.

All operators are strict-PIT, deterministic and NaN fail-closed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import ParamSpec, RelationalParamSpec, ParamRole
from cleaned_operators.closure.strict_scalar import strict_int, strict_float
from cleaned_operators.gemini_v2_common import (
    frame_like,
    register_dual,
    trailing_contiguous_finite,
    union_extended,
)

_EPS = 1e-12
_LN2 = np.log(2.0)

# R5 P1-01: authoritative contracts — ``k``/``k_min``/``k_max``/``scale`` are
# fixed ints (never silently truncated from a float), ``side`` is an enum.
_SIDE_SPEC = ParamSpec(dtype=str, choices=("upper", "lower"), searchable=True)
_PICKANDS_SPEC = {
    "window": ParamSpec(dtype=int, min=6),
    "k": ParamSpec(dtype=int, min=1, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    "side": _SIDE_SPEC,
}
# R6-167: Pickands requires window >= 4k + 1 (the kernel uses order stats
# X_(n-4k), X_(n-2k), X_(n-k)), otherwise it is guaranteed-NaN.  Declared as a
# relational constraint so search prunes the infeasible region before runtime.
_PICKANDS_RELATIONAL_SPECS = [
    RelationalParamSpec(
        "window >= 4 * k + 1",
        "Pickands requires window >= 4*k + 1 (window={window}, k={k})",
    )
]
_EVT_SPEC = {
    "window": ParamSpec(dtype=int, min=10),
    "k_min": ParamSpec(dtype=int, min=2),
    "k_max": ParamSpec(dtype=int, min=2, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    "side": _SIDE_SPEC,
}
# R6-168 + R14-P1-22 (feasibility unification): threshold stability needs a
# window of Hill estimates across k_min..k_max.  The RUNTIME guard requires
# ``v.size >= 2*k_max + 2`` (so the full k_min..k_max Hill ladder resolves
# instead of being truncated by the n//2 cap), so the DECLARED contract must
# express the same region — ``window >= 2*k_max + 2`` — or search would treat
# window=30 / k_max=20 as "legal" while every runtime call is guaranteed
# all-NaN.  ``k_min < k_max`` keeps more than one distinct threshold.
_EVT_RELATIONAL_SPECS = [
    RelationalParamSpec(
        "window >= 2 * k_max + 2",
        "EVT threshold stability requires window >= 2*k_max + 2 "
        "(window={window}, k_max={k_max})",
    ),
    RelationalParamSpec(
        "k_min < k_max",
        "EVT threshold stability requires k_min < k_max "
        "(k_min={k_min}, k_max={k_max})",
    ),
]
# R9-OP-007/008 (feasibility unification): the RUNTIME kernels of the Allan
# family fail closed on guarantees the declared ParamSpec did not express —
# ``event_allan_factor(window=8, scale=3)`` needs ``3*scale=9`` rows but the
# spec only said window>=8, and ``event_allan_scaling_slope(max_scale=2)`` can
# never produce the 3 dyadic scales the slope needs, so it was NaN for every
# input.  Each family has ONE feasibility function, used by both the declared
# contract and the runtime guard.
_ALLAN_FACTOR_SPEC = {
    "window": ParamSpec(dtype=int, min=8),
    "scale": ParamSpec(dtype=int, min=1, param_role=ParamRole.ESTIMATOR_RESOLUTION),
}
_ALLAN_SCALING_SPEC = {
    "window": ParamSpec(dtype=int, min=12),
    "max_scale": ParamSpec(dtype=int, min=4, param_role=ParamRole.ESTIMATOR_RESOLUTION),
}
_ALLAN_FACTOR_RELATIONAL_SPECS = [
    RelationalParamSpec(
        "window >= 3 * scale",
        "event_allan_factor requires window >= 3*scale for 3 adjacent blocks "
        "(window={window}, scale={scale})",
    )
]
# R6-157: the registry invariant requires keys(param_specs) ⊆ param_names.
# The scaling canonicals take ``event/window/max_scale`` only — per-canonical
# subsets are declared here.


def allan_factor_feasibility(*, window: int, scale: int) -> bool:
    """Single ``event_allan_factor`` feasibility contract (R9-OP-007)."""
    return int(window) >= 8 and int(scale) >= 1 and int(window) >= 3 * int(scale)


def allan_scaling_feasibility(*, window: int, max_scale: int) -> bool:
    """Single scaling-family feasibility contract (R9-OP-008).

    The slope / log-mean need >= 3 dyadic scales; the dyadic scales
    1,2,4 must all fit (``3*4 = 12 <= window``) AND ``max_scale`` must reach 4.
    """
    return int(window) >= 12 and int(max_scale) >= 4


# ---------------------------------------------------------------------------
# Pickands tail index
# ---------------------------------------------------------------------------
def _pickands_tail(v: np.ndarray, k: int, upper: bool) -> float:
    n = v.shape[0]
    kk = max(1, int(k))
    if n < 4 * kk + 1:
        return np.nan
    z = v if upper else -v
    s = np.sort(z)
    # order stats: X_(n), X_(n-1), ...
    a = s[n - 4 * kk]
    b = s[n - 2 * kk]
    d = s[n - kk]
    denom = b - a
    numer = d - b
    if denom <= _EPS or numer <= _EPS:
        return np.nan
    return float(np.log(numer / denom) / _LN2)


# ---------------------------------------------------------------------------
# EVT threshold stability (std of Hill over k)
# ---------------------------------------------------------------------------
def _hill_xi(s: np.ndarray, k: int) -> float:
    n = s.shape[0]
    if k < 1 or n - 1 - k < 0:
        return np.nan
    threshold = s[n - 1 - k]
    if threshold <= _EPS:
        return np.nan
    top = s[n - 1 - k + 1 : n]
    if top.size != k or np.any(top <= _EPS):
        return np.nan
    return float(np.mean(np.log(top / threshold)))


def _threshold_stability(v: np.ndarray, k_min: int, k_max: int, upper: bool) -> float:
    n = v.shape[0]
    z = v if upper else -v
    s = np.sort(z)
    kmax = min(int(k_max), n // 2)
    kmin = max(2, int(k_min))
    if kmax <= kmin:
        return np.nan
    # R5 P1-10: the Hill estimator does NOT require the whole window to be
    # positive — only the top-k tail used for estimation must be.  Signed
    # returns with a few opposite-direction values no longer fail the window;
    # each k-threshold is accepted iff its own top-k sample is positive.
    hills: list[float] = []
    for k in range(kmin, kmax + 1):
        h = _hill_xi(s, k)
        if np.isfinite(h):
            hills.append(h)
    if len(hills) < 3:
        return np.nan
    sigma = float(np.std(hills))
    # R5 P1-11: emit a genuine stability score in (0, 1] — 1/(1+σ_ξ) — so
    # HIGHER means MORE stable (the raw std had the opposite sign vs the name).
    return float(1.0 / (1.0 + sigma))


def _ts_pickands_tail_index(x: pd.DataFrame, window: int = 120, k: int = 10, side: str = "upper") -> pd.DataFrame:
    if int(window) < 6:
        raise ValueError("ts_pickands_tail_index requires window >= 6")
    side_s = str(side).lower()
    if side_s not in {"upper", "lower"}:
        raise ValueError("ts_pickands_tail_index requires side in {'upper','lower'}")
    rows, cols = x.shape
    w = int(window)
    arr = x.to_numpy(dtype=float)
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = arr[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo : r + 1])
            if v.size < 4 * int(k) + 1:
                continue
            val = _pickands_tail(v, int(k), side_s == "upper")
            if np.isfinite(val):
                out[r, c] = val
    return frame_like(x, out)


def _ts_evt_threshold_stability(x: pd.DataFrame, window: int = 120, k_min: int = 5, k_max: int = 20, side: str = "upper") -> pd.DataFrame:
    if int(window) < 10:
        raise ValueError("ts_evt_threshold_stability requires window >= 10")
    side_s = str(side).lower()
    if side_s not in {"upper", "lower"}:
        raise ValueError("ts_evt_threshold_stability requires side in {'upper','lower'}")
    rows, cols = x.shape
    w = int(window)
    arr = x.to_numpy(dtype=float)
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = arr[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo : r + 1])
            if v.size < int(k_max) * 2 + 2:
                continue
            val = _threshold_stability(v, int(k_min), int(k_max), side_s == "upper")
            if np.isfinite(val):
                out[r, c] = val
    return frame_like(x, out)


# ---------------------------------------------------------------------------
# Allan factor (R5 P0-07: the old kernel computed the Fano Factor
# Var(N)/Mean(N), which is a DIFFERENT statistic — the true Allan factor is
# AF(τ) = E[(N_{k+1} − N_k)²] / (2·E[N_k]) over adjacent blocks of length τ,
# and it is the scale-resolved statistic used in point-process literature.)
# ---------------------------------------------------------------------------
def _check_event_binary(v: np.ndarray) -> bool:
    """Input contract: every finite value must be 0/1 (event indicator)."""
    return bool(np.all((v == 0.0) | (v == 1.0)))


def _allan_factor_single(v: np.ndarray, scale: int) -> float:
    """True Allan factor at one scale: AF(τ) = E[(N_{k+1}−N_k)²]/(2·E[N_k])."""
    n = v.shape[0]
    m = strict_int(scale, "scale", lower=1)  # A-5: scale=0/-1 is a contract violation, never clamped
    if n < 3 * m:
        return np.nan
    n_blocks = n // m
    if n_blocks < 3:
        return np.nan
    blocks = v[n - n_blocks * m :].reshape(n_blocks, m)
    counts = blocks.sum(axis=1)
    mean_c = float(np.mean(counts))
    if mean_c <= _EPS:
        return np.nan
    diffs = np.diff(counts)  # N_{k+1} − N_k
    af = float(np.mean(diffs ** 2)) / (2.0 * mean_c)
    return float(af) if np.isfinite(af) else np.nan


def _allan_scale_logs(v: np.ndarray, max_scale: int) -> list[tuple[int, float]]:
    """Dyadic scales with their ``log2(AF(τ))`` (skipping degenerate scales)."""
    n = v.shape[0]
    ms = max(1, int(max_scale))
    logs: list[tuple[int, float]] = []
    m = 1
    while m <= ms:
        if n >= 3 * m:
            af = _allan_factor_single(v, m)
            if np.isfinite(af) and af > _EPS:
                logs.append((m, float(np.log2(af))))
        m *= 2
    return logs


def _allan_scaling(v: np.ndarray, max_scale: int) -> tuple[float | None, float | None]:
    """``(slope, log_mean)`` over the dyadic scales, or ``(None, None)`` when
    fewer than 3 scales resolve."""
    logs = _allan_scale_logs(v, max_scale)
    if len(logs) < 3:
        return None, None
    xs = np.log2(np.asarray([s for s, _ in logs], dtype=float))
    ys = np.asarray([y for _, y in logs], dtype=float)
    slope = float(np.polyfit(xs, ys, 1)[0])
    return slope, float(np.mean(ys))


def _ts_event_allan_factor(x: pd.DataFrame, window: int = 120, scale: int = 8) -> pd.DataFrame:
    if int(window) < 8:
        raise ValueError("event_allan_factor requires window >= 8")
    if int(scale) < 1:
        raise ValueError("event_allan_factor requires scale >= 1")
    rows, cols = x.shape
    w = int(window)
    arr = x.to_numpy(dtype=float)
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = arr[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo : r + 1])
            # R9-OP-007: runtime gate == declared feasibility contract; the run
            # length substitutes for ``window``.
            if not allan_factor_feasibility(window=v.size, scale=int(scale)):
                continue
            if not _check_event_binary(v):
                continue
            val = _allan_factor_single(v, int(scale))
            if np.isfinite(val):
                out[r, c] = val
    return frame_like(x, out)


def _ts_event_allan_scaling_slope(x: pd.DataFrame, window: int = 120, max_scale: int = 16) -> pd.DataFrame:
    if int(window) < 16:
        raise ValueError("event_allan_scaling_slope requires window >= 16")
    rows, cols = x.shape
    w = int(window)
    arr = x.to_numpy(dtype=float)
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = arr[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo : r + 1])
            # R9-OP-008: runtime gate == declared feasibility contract.
            if not allan_scaling_feasibility(window=v.size, max_scale=int(max_scale)):
                continue
            if not _check_event_binary(v):
                continue
            slope, _ = _allan_scaling(v, int(max_scale))
            if slope is not None and np.isfinite(slope):
                out[r, c] = slope
    return frame_like(x, out)


def _ts_event_allan_log_mean(x: pd.DataFrame, window: int = 120, max_scale: int = 16) -> pd.DataFrame:
    if int(window) < 16:
        raise ValueError("event_allan_log_mean requires window >= 16")
    rows, cols = x.shape
    w = int(window)
    arr = x.to_numpy(dtype=float)
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = arr[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo : r + 1])
            # R9-OP-008: runtime gate == declared feasibility contract.
            if not allan_scaling_feasibility(window=v.size, max_scale=int(max_scale)):
                continue
            if not _check_event_binary(v):
                continue
            _, log_mean = _allan_scaling(v, int(max_scale))
            if log_mean is not None and np.isfinite(log_mean):
                out[r, c] = log_mean
    return frame_like(x, out)


_SPECS: dict[str, dict[str, Any]] = {
    "ts_pickands_tail_index": {
        "fn": _ts_pickands_tail_index,
        "params": ["x", "window", "k", "side"],
        "category": "extreme_value",
        "domain": "tail",
        "unit": "level",
        "cost": 4,
        "tags_extra": ["state"],
        "output_unit": "level",
        "param_specs": _PICKANDS_SPEC,
        "relational_specs": _PICKANDS_RELATIONAL_SPECS,
    },
    "ts_evt_threshold_stability": {
        "fn": _ts_evt_threshold_stability,
        "params": ["x", "window", "k_min", "k_max", "side"],
        "category": "extreme_value",
        "domain": "tail",
        "unit": "level",
        "cost": 4,
        "tags_extra": ["state"],
        "output_unit": "level",
        "param_specs": _EVT_SPEC,
        "relational_specs": _EVT_RELATIONAL_SPECS,
    },
    "event_allan_factor": {
        "fn": _ts_event_allan_factor,
        "params": ["event", "window", "scale"],
        "category": "event_counting",
        "domain": "event",
        "unit": "level",
        "cost": 3,
        "tags_extra": ["state"],
        "output_unit": "level",
        "param_specs": _ALLAN_FACTOR_SPEC,
        "relational_specs": _ALLAN_FACTOR_RELATIONAL_SPECS,
    },
    "event_allan_scaling_slope": {
        "fn": _ts_event_allan_scaling_slope,
        "params": ["event", "window", "max_scale"],
        "category": "event_counting",
        "domain": "event",
        "unit": "level",
        "cost": 4,
        "tags_extra": ["state"],
        "output_unit": "level",
        "param_specs": _ALLAN_SCALING_SPEC,
    },
    "event_allan_log_mean": {
        "fn": _ts_event_allan_log_mean,
        "params": ["event", "window", "max_scale"],
        "category": "event_counting",
        "domain": "event",
        "unit": "level",
        "cost": 3,
        "tags_extra": ["state"],
        "output_unit": "level",
        "param_specs": _ALLAN_SCALING_SPEC,
    },
}


def _register() -> None:
    for canonical, spec in _SPECS.items():
        register_dual(
            canonical,
            spec["fn"],
            spec["params"],
            category=spec["category"],
            domain=spec["domain"],
            unit=spec["unit"],
            cost=spec["cost"],
            source="evt_allan",
            tags_extra=spec["tags_extra"],
            output_unit=spec.get("output_unit"),
            param_specs=spec.get("param_specs"),
            relational_specs=spec.get("relational_specs"),
        )
    union_extended(*_SPECS.keys())


_register()
