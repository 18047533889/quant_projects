# -*- coding: utf-8 -*-
"""Recurrence Quantification Analysis extension (2026-08-08 Gemini V2 round).

The existing ``recurrence_analysis`` module already covers RR / diagonal
entropy / trapping time / divergence.  This round adds the two classic RQA
line-structure statistics plus two line-length summaries, all derived from the
same recurrence matrix (one build per window):

* ``ts_recurrence_determinism``      — DET: fraction of recurrent points that
  belong to diagonal lines of length ≥ ``min_line`` (P0, daily / STATE).
* ``ts_recurrence_laminarity``       — LAM: fraction of recurrent points that
  belong to vertical lines of length ≥ ``min_line`` (P0, daily / STATE).
* ``ts_recurrence_mean_diagonal_length`` — mean diagonal line length (P1).
* ``ts_recurrence_longest_vertical_length`` — longest vertical line (P1).
* ``ts_rqa_determinism_fixed_rr``    — DET at a FIXED recurrence rate: epsilon is
  calibrated to a target RR first, then line topology is measured (R11 round-3
  #68, so DET is not conflated with recurrence density).
* ``ts_rqa_laminarity_fixed_rr``     — LAM at a FIXED recurrence rate (same
  epsilon-targeting routine).

Semantics mirror the sibling module exactly: phase-space embedding with
``ε = eps_fraction · 1.4826·MAD`` (scale-relative, deterministic), upper-
triangle diagonal scans and full-column vertical scans, and the trailing
contiguous finite run (a gap never compresses the time axis).

Production sample-size rule (R11 round-3 #67): DET/LAM (and the other line
statistics) are strongly sample-size dependent, so the effective length — the
longest trailing contiguous finite run — must cover at least ``0.8 * window``;
below that the reading is NaN (a gap must not be followed by a short-sample
recompute that is a different operator than a full-window reading).

DET = sum(diag_lengths) / (recurrences/2) — because R is symmetric, the upper
triangle holds exactly half the recurrent points and each off-diagonal point
lies in exactly one upper-triangular diagonal run.
LAM = sum(vert_lengths) / recurrences — vertical runs partition all recurrent
points (each point lies in exactly one maximal run of its column).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import ParamRole, ParamSpec, RelationalParamSpec
from cleaned_operators.closure.strict_scalar import strict_int, strict_float
from cleaned_operators.gemini_v2_common import (
    frame_like,
    register_dual,
    trailing_contiguous_finite,
    union_extended,
)

_EPS = 1e-12

# R5 P1-01: embedding ints are validated (dim=1.9 -> reject), ``theiler >= 0``.
# R6-116: ``eps_fraction`` is declared strictly inside (0,1) to MATCH the runtime
# check — the previous spec allowed 0/1 that search could generate but runtime
# rejected, a dead search region.
# R6-112: the Theiler window default is the embedding span (dim-1)*delay, since
# adjacent embedded points share (dim-1) coordinates and are trivially close.
_RQA_PARAM_SPECS = {
    "window": ParamSpec(dtype=int, min=2),
    "dim": ParamSpec(dtype=int, min=1),
    "delay": ParamSpec(dtype=int, min=1),
    "eps_fraction": ParamSpec(dtype=float, min=1e-4, max=1.0 - 1e-4),
    "min_line": ParamSpec(dtype=int, min=2),
    "min_periods": ParamSpec(dtype=int, min=4),
    "theiler": ParamSpec(dtype=int, min=0),
}
# R6-117: ``min_line`` must be strictly below the phase-space size
# M = window - (dim-1)*delay, or no diagonal/vertical line of that length can
# exist and the line statistic is unestimable (runtime returns NaN).  Declared
# as a relational constraint so search never emits a guaranteed-NaN combo.
_RQA_RELATIONAL_SPECS = [
    RelationalParamSpec(
        "min_line < window - (dim - 1) * delay",
        "min_line must be < effective embedding count window-(dim-1)*delay "
        "(min_line={min_line}, window={window}, dim={dim}, delay={delay})",
    )
]

# R11 round-3 #68: fixed-recurrence-rate DET/LAM replace the epsilon-multiplier
# knob with a target recurrence rate.  All the estimator-resolution knobs
# (embedding dim/delay, theiler window, and the target RR itself) are
# deliberately NOT search dimensions.
_RQA_FIXED_RR_PARAM_SPECS = {
    "window": ParamSpec(dtype=int, min=2),
    "dim": ParamSpec(
        dtype=int, min=1,
        param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False,
    ),
    "delay": ParamSpec(
        dtype=int, min=1,
        param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False,
    ),
    "target_rr": ParamSpec(
        dtype=float,
        min=1e-3,
        max=0.5,
        default=0.05,
        param_role=ParamRole.ESTIMATOR_RESOLUTION,
        searchable=False,
    ),
    "min_line": ParamSpec(dtype=int, min=2),
    "min_periods": ParamSpec(dtype=int, min=4),
    "theiler": ParamSpec(
        dtype=int, min=0,
        param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False,
    ),
}


def _rqa_line_statistics(R: np.ndarray, M: int, min_line: int) -> tuple[list[int], list[int]]:
    """Diagonal and vertical line-length runs of a recurrence matrix.

    Returns ``(diag_lengths, vert_lengths)`` — the lengths (in bars) of every
    maximal diagonal / vertical run of at least ``min_line`` recurrent points.
    """
    diag_lengths: list[int] = []
    for off in range(1, M):
        length = 0
        i, j = 0, off
        while j < M:
            if R[i, j]:
                length += 1
            else:
                if length >= min_line:
                    diag_lengths.append(length)
                length = 0
            i += 1
            j += 1
        if length >= min_line:
            diag_lengths.append(length)

    vert_lengths: list[int] = []
    for j in range(M):
        length = 0
        for i in range(M):
            if R[i, j]:
                length += 1
            else:
                if length >= min_line:
                    vert_lengths.append(length)
                length = 0
        if length >= min_line:
            vert_lengths.append(length)
    return diag_lengths, vert_lengths


def _rqa_line_summary(
    diag_lengths: list[int], vert_lengths: list[int], n_rec: float,
    min_line: int, M: int,
) -> dict[str, float]:
    """DET / LAM / mean-diag / longest-vert from line-length runs.

    DET = sum(diag_lengths)/(recurrences/2) — the recurrence matrix is symmetric,
    the upper triangle holds half the recurrent points and each off-diagonal
    point lies in exactly one upper-triangular diagonal run.
    LAM = sum(vert_lengths)/recurrences — vertical runs partition all recurrent
    points.
    P1-28: when ``min_line >= M`` no line of the requested length can exist —
    the line statistics are unestimable (returning 0 would read as "confirmed
    no deterministic line").  ``M`` is the phase-space size.
    """
    if int(min_line) >= M:
        determinism = np.nan
        laminarity = np.nan
        mean_diag = np.nan
        longest_vert = np.nan
    else:
        determinism = np.where(n_rec if diag_lengths else 0.0 != 0, (2.0 * sum(diag_lengths)) / n_rec if diag_lengths else 0.0, np.nan)
        laminarity = np.where(n_rec if vert_lengths else 0.0 != 0, (1.0 * sum(vert_lengths)) / n_rec if vert_lengths else 0.0, np.nan)
        mean_diag = float(np.mean(diag_lengths)) if diag_lengths else np.nan
        longest_vert = float(np.max(vert_lengths)) if vert_lengths else np.nan
    return {
        "determinism": determinism,
        "laminarity": laminarity,
        "mean_diagonal_length": mean_diag,
        "longest_vertical_length": longest_vert,
    }


def _rqa_epsilon_for_target_rr(
    D: np.ndarray, theiler: int, target_rr: float,
) -> tuple[float, int] | None:
    """Calibrate epsilon so the recurrence rate equals ``target_rr``.

    RR = #(eligible pairs with distance <= eps) / #(eligible pairs), where the
    eligible pairs are the off-diagonal pairs with ``|i-j| > theiler``.  By
    symmetry the threshold can be read off the upper-triangle eligible distances:
    ``eps`` is the ``ceil(target_rr * n)``-th smallest such distance.  Returns
    ``(eps, n_eligible_pairs)`` or ``None`` when fewer than four eligible pairs
    survive.
    """
    M = D.shape[0]
    iu, ju = np.triu_indices(M, k=1)
    dist = D[iu, ju]
    dist = dist[np.abs(iu - ju) > int(theiler)]
    n = dist.size
    if n < 4:
        return None
    k = max(1, min(n, int(np.ceil(float(target_rr) * n))))
    eps = float(np.partition(dist, k - 1)[k - 1])
    return eps, n


def _rqa_stats_window_fixed_rr(
    v: np.ndarray,
    dim: int,
    delay: int,
    target_rr: float,
    min_line: int,
    theiler: int | None = None,
) -> dict[str, float]:
    """DET / LAM at a FIXED recurrence rate (R11 round-3 #68).

    The classic ``eps_fraction * scale`` epsilon makes the recurrence RATE vary
    with the market distribution, so DET/LAM conflate recurrence DENSITY with
    LINE STRUCTURE.  Here epsilon is first calibrated so that RR ~ ``target_rr``
    (controlled density), and DET/LAM are then measured on that matrix — line
    topology at a comparable density across regimes.
    """
    M = v.shape[0] - (dim - 1) * delay
    if M < 4:
        return {}
    if dim == 1:
        P = v[:M].reshape(-1, 1)
    else:
        P = np.stack([v[i : i + M] for i in range(0, dim * delay, delay)], axis=1)
    if theiler is None:
        theiler = max(0, (dim - 1) * delay)
    D = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2))
    calibrated = _rqa_epsilon_for_target_rr(D, int(theiler), float(target_rr))
    if calibrated is None:
        return {}
    eps, n_eligible = calibrated
    time_mask = np.abs(np.arange(M)[:, None] - np.arange(M)[None, :]) > int(theiler)
    R = (D <= eps) & time_mask
    rate = np.where((2.0 * n_eligible) != 0, float(R.sum()) / (2.0 * n_eligible), np.nan)
    n_rec = float(R.sum())
    if n_rec <= 0:
        return {}
    diag_lengths, vert_lengths = _rqa_line_statistics(R, M, min_line)
    summary = _rqa_line_summary(diag_lengths, vert_lengths, n_rec, min_line, M)
    summary["rate"] = rate
    return summary


def _rqa_stats_window(
    v: np.ndarray,
    dim: int,
    delay: int,
    eps_fraction: float,
    min_line: int,
    theiler: int | None = None,
) -> dict[str, float]:
    """Full RQA statistics for one finite window (see module docstring).

    R6-113: phase-space distance grows ~√dim in a dim-dimensional embedding, so a
    fixed epsilon does not describe a fixed neighbourhood density across dim.
    Normalise epsilon by √dim: ``eps = eps_fraction * scale / sqrt(dim)``.
    R6-112: the Theiler window (temporal exclusion) defaults to the embedding
    span ``(dim-1)*delay`` — adjacent embedded points share (dim-1) coordinates
    and are naturally close, so excluding fewer than that inflates DET/LAM.
    """
    M = v.shape[0] - (dim - 1) * delay
    if M < 4:
        return {}
    if dim == 1:
        P = v[:M].reshape(-1, 1)
    else:
        P = np.stack([v[i : i + M] for i in range(0, dim * delay, delay)], axis=1)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= _EPS:
        scale = float(np.std(v))
    if not np.isfinite(scale) or scale <= _EPS:
        return {}
    # R6-113 / R11 P0-11: phase-space distance D = sqrt(Σ(x_j-y_j)²) grows as
    # scale·√dim; the threshold must scale WITH the distance (multiply by √dim)
    # to keep the recurrence rate comparable across dim.  The previous ``/√dim``
    # shrank eps while D grew, collapsing the rate as dim increased.
    eps = float(eps_fraction) * scale * float(np.sqrt(max(1, int(dim))))
    if theiler is None:
        theiler = max(0, (dim - 1) * delay)
    D = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2))
    # R5 P1-05 (Theiler window): pairs too close in TIME (|i − j| ≤ theiler) are
    # excluded from the recurrence matrix — a smooth price series makes adjacent
    # embedded points naturally close, which would otherwise inflate DET/LAM.
    # ``theiler=0`` keeps the historical behaviour.
    time_mask = np.abs(np.arange(M)[:, None] - np.arange(M)[None, :]) > int(theiler)
    R = (D <= eps) & time_mask

    # R11 P1-09: after the Theiler exclusion the recurrence RATE must be
    # normalised by the ADMISSIBLE pairs only (those that survived |i-j|>theiler
    # AND are off-diagonal), not by the full M(M-1) off-diagonal count.  Using
    # M(M-1) would mechanically deflate RR as theiler grows (fewer numerator
    # pairs but the same denominator) — a pure denominator effect, not signal.
    # Symmetric mask: #{(i,j): i≠j, |i-j|>theiler} = 2·#{i>j, |i-j|>theiler}.
    off_diag = ~np.eye(M, dtype=bool)
    admissible_pairs = int(np.count_nonzero(off_diag & time_mask))
    if admissible_pairs <= 0:
        return {}
    rate = float(R.sum()) / admissible_pairs if admissible_pairs != 0 else np.nan
    n_rec = float(R.sum())
    if n_rec <= 0:
        return {}

    diag_lengths, vert_lengths = _rqa_line_statistics(R, M, min_line)
    summary = _rqa_line_summary(diag_lengths, vert_lengths, n_rec, min_line, M)
    summary["rate"] = rate
    return summary


def _check_params(
    window: int, dim: int, delay: int, eps_fraction: float, min_line: int, theiler: int | None = None
) -> tuple[int, int, int, float, int, int]:
    # Master Spec A-4/5: window/dim/delay/min_line are user parameters — a
    # ``dim=0`` or ``window=1.5`` must RAISE (contract violation), never be
    # silently clamped to a legal value (false AST).
    w = strict_int(window, "window", lower=2)
    d = strict_int(dim, "dim", lower=1)
    dl = strict_int(delay, "delay", lower=1)
    ef = strict_float(eps_fraction, "eps_fraction")
    if not 0.0 < ef < 1.0:
        raise ValueError("eps_fraction must be in (0, 1)")
    if d * dl > 4:
        raise ValueError("dimension*delay must be <= 4 (embedding support)")
    ml = strict_int(min_line, "min_line", lower=2)
    # ``theiler=None`` -> auto-derived from the embedding (not a user clamp);
    # an explicit theiler is validated as a non-negative int.
    th = max(0, (d - 1) * dl) if theiler is None else strict_int(theiler, "theiler", lower=0)
    return w, d, dl, ef, ml, th


def _rqa_series(x2d: np.ndarray, window: int, dim: int, delay: int, eps_fraction: float, min_line: int, min_periods: int, key: str, theiler: int = 0) -> np.ndarray:
    rows, cols = x2d.shape
    w, d, dl, ef, ml, th = _check_params(window, dim, delay, eps_fraction, min_line, theiler)
    mp = max(4, int(min_periods))
    # R11 round-3 #67: DET/LAM are strongly sample-size dependent (recurrence
    # matrix size).  With ``min_periods`` too low a gap can be followed by a
    # short-sample recompute that is NOT the same operator as a full-window
    # reading.  Production rule: the effective length (longest trailing
    # contiguous finite run) must cover >= 0.8*window; below that -> NaN.
    min_eff = max(mp, int(np.ceil(0.8 * w)))
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo : r + 1])
            if v.size < min_eff:
                continue
            stats = _rqa_stats_window(v, d, dl, ef, ml, th)
            val = stats.get(key, np.nan)
            if np.isfinite(val):
                out[r, c] = float(val)
    return out


def _rqa_fixed_rr_series(
    x2d: np.ndarray, window: int, dim: int, delay: int, target_rr: float,
    min_line: int, min_periods: int, key: str, theiler: int | None = None,
) -> np.ndarray:
    """Trailing-window driver for the fixed-recurrence-rate DET/LAM operators.

    Applies the same production sample-size rule as the classic DET/LAM
    operators (R11 round-3 #67): the effective length must cover >= 0.8*window,
    otherwise NaN.
    """
    rows, cols = x2d.shape
    w = max(2, int(window))
    d = max(1, int(dim))
    dl = max(1, int(delay))
    if d * dl > 4:
        raise ValueError("dimension*delay must be <= 4 (embedding support)")
    ml = max(2, int(min_line))
    th = max(0, (d - 1) * dl if theiler is None else int(theiler))
    rr = float(target_rr)
    if not np.isfinite(rr) or not (0.0 < rr < 1.0):
        raise ValueError("target_rr must be in (0, 1)")
    mp = max(4, int(min_periods))
    min_eff = max(mp, int(np.ceil(0.8 * w)))
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo : r + 1])
            if v.size < min_eff:
                continue
            stats = _rqa_stats_window_fixed_rr(v, d, dl, rr, ml, th)
            val = stats.get(key, np.nan)
            if np.isfinite(val):
                out[r, c] = float(val)
    return out


def _ts_rqa_determinism_fixed_rr(
    x: pd.DataFrame,
    window: int = 60,
    dim: int = 1,
    delay: int = 1,
    target_rr: float = 0.05,
    min_line: int = 4,
    min_periods: int = 10,
    theiler: int | None = None,
) -> pd.DataFrame:
    out = _rqa_fixed_rr_series(
        x.to_numpy(dtype=float), window, dim, delay, target_rr,
        min_line, min_periods, "determinism", theiler,
    )
    return frame_like(x, out)


def _ts_rqa_laminarity_fixed_rr(
    x: pd.DataFrame,
    window: int = 60,
    dim: int = 1,
    delay: int = 1,
    target_rr: float = 0.05,
    min_line: int = 4,
    min_periods: int = 10,
    theiler: int | None = None,
) -> pd.DataFrame:
    out = _rqa_fixed_rr_series(
        x.to_numpy(dtype=float), window, dim, delay, target_rr,
        min_line, min_periods, "laminarity", theiler,
    )
    return frame_like(x, out)


def _ts_recurrence_determinism(
    x: pd.DataFrame,
    window: int = 60,
    dim: int = 1,
    delay: int = 1,
    eps_fraction: float = 0.1,
    min_line: int = 4,
    min_periods: int = 10,
    theiler: int | None = None,
) -> pd.DataFrame:
    out = _rqa_series(x.to_numpy(dtype=float), window, dim, delay, eps_fraction, min_line, min_periods, "determinism", theiler)
    return frame_like(x, out)


def _ts_recurrence_laminarity(
    x: pd.DataFrame,
    window: int = 60,
    dim: int = 1,
    delay: int = 1,
    eps_fraction: float = 0.1,
    min_line: int = 4,
    min_periods: int = 10,
    theiler: int | None = None,
) -> pd.DataFrame:
    out = _rqa_series(x.to_numpy(dtype=float), window, dim, delay, eps_fraction, min_line, min_periods, "laminarity", theiler)
    return frame_like(x, out)


def _ts_recurrence_mean_diagonal_length(
    x: pd.DataFrame,
    window: int = 60,
    dim: int = 1,
    delay: int = 1,
    eps_fraction: float = 0.1,
    min_line: int = 4,
    min_periods: int = 10,
    theiler: int | None = None,
) -> pd.DataFrame:
    out = _rqa_series(x.to_numpy(dtype=float), window, dim, delay, eps_fraction, min_line, min_periods, "mean_diagonal_length", theiler)
    return frame_like(x, out)


def _ts_recurrence_longest_vertical_length(
    x: pd.DataFrame,
    window: int = 60,
    dim: int = 1,
    delay: int = 1,
    eps_fraction: float = 0.1,
    min_line: int = 4,
    min_periods: int = 10,
    theiler: int | None = None,
) -> pd.DataFrame:
    out = _rqa_series(x.to_numpy(dtype=float), window, dim, delay, eps_fraction, min_line, min_periods, "longest_vertical_length", theiler)
    return frame_like(x, out)


_SPECS: dict[str, dict[str, Any]] = {
    "ts_recurrence_determinism": {
        "fn": _ts_recurrence_determinism,
        "params": ["x", "window", "dim", "delay", "eps_fraction", "min_line", "min_periods", "theiler"],
        "category": "time_series_recurrence",
        "domain": "path_geometry",
        "unit": "ratio",
        "cost": 6,
        "tags_extra": ["state"],
        "param_specs": _RQA_PARAM_SPECS,
    },
    "ts_recurrence_laminarity": {
        "fn": _ts_recurrence_laminarity,
        "params": ["x", "window", "dim", "delay", "eps_fraction", "min_line", "min_periods", "theiler"],
        "category": "time_series_recurrence",
        "domain": "path_geometry",
        "unit": "ratio",
        "cost": 6,
        "tags_extra": ["state"],
        "param_specs": _RQA_PARAM_SPECS,
    },
    "ts_recurrence_mean_diagonal_length": {
        "fn": _ts_recurrence_mean_diagonal_length,
        "params": ["x", "window", "dim", "delay", "eps_fraction", "min_line", "min_periods", "theiler"],
        "category": "time_series_recurrence",
        "domain": "path_geometry",
        "unit": "bars",
        "cost": 6,
        "tags_extra": [],
        "param_specs": _RQA_PARAM_SPECS,
    },
    "ts_recurrence_longest_vertical_length": {
        "fn": _ts_recurrence_longest_vertical_length,
        "params": ["x", "window", "dim", "delay", "eps_fraction", "min_line", "min_periods", "theiler"],
        "category": "time_series_recurrence",
        "domain": "path_geometry",
        "unit": "bars",
        "cost": 6,
        "tags_extra": [],
        "param_specs": _RQA_PARAM_SPECS,
    },
    # R11 round-3 #68: fixed-recurrence-rate DET/LAM — epsilon calibrated to a
    # target recurrence rate so the recurrence DENSITY is controlled and DET/LAM
    # measure LINE STRUCTURE alone.
    "ts_rqa_determinism_fixed_rr": {
        "fn": _ts_rqa_determinism_fixed_rr,
        "params": ["x", "window", "dim", "delay", "target_rr", "min_line", "min_periods", "theiler"],
        "category": "time_series_recurrence",
        "domain": "path_geometry",
        "unit": "ratio",
        "cost": 7,
        "tags_extra": ["state"],
        "param_specs": _RQA_FIXED_RR_PARAM_SPECS,
    },
    "ts_rqa_laminarity_fixed_rr": {
        "fn": _ts_rqa_laminarity_fixed_rr,
        "params": ["x", "window", "dim", "delay", "target_rr", "min_line", "min_periods", "theiler"],
        "category": "time_series_recurrence",
        "domain": "path_geometry",
        "unit": "ratio",
        "cost": 7,
        "tags_extra": ["state"],
        "param_specs": _RQA_FIXED_RR_PARAM_SPECS,
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
            source="rqa_ext",
            tags_extra=spec["tags_extra"],
            output_unit=spec["unit"],
            param_specs=spec.get("param_specs"),
            relational_specs=_RQA_RELATIONAL_SPECS,
        )
    union_extended(*_SPECS.keys())


_register()
