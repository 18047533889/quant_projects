# -*- coding: utf-8 -*-
"""Hankel / SSA structure operators (2026-08 geometry/math expansion).

The shared kernel prepares the trailing window as a *strict trailing contiguous
finite run* (production default — data on either side of a gap is never
re-connected), then forms the trajectory ``Hankel`` matrix ``H`` of shape
``(N - embedding_dim + 1, embedding_dim)`` from lagged rows, and performs an
SVD → singular values ``σ``.  The family then reads the singular spectrum:

* R4-66: the contiguous run must cover at least ``min_contiguous_fraction``
  (default 0.8) of the window, otherwise the row emits NaN.  Without this gate a
  sparse day could build an 18-point Hankel matrix and a full day a 60-point one
  under the same operator name — a different factor from one row to the next.
* R4-67: linear interpolation across a gap (``missing_mode="interpolate"``) is
  research-only and requires explicit opt-in; the production default is
  ``strict_contiguous``.
* P0-5: the CURRENT row is always required.  When the current observation is
  NaN the row emits NaN even if a finite trailing contiguous run would cover
  the window — the operator must never fall back to yesterday's run and emit a
  stale-history factor.

* ``ts_hankel_effective_rank``       — exponential of the entropy of the
  normalized squared-singular-value distribution, normalized by ``min(H.shape)``.
* ``ts_hankel_singular_gap``         — relative gap between the two largest
  singular values (a proxy for mode separation / structure strength).
* ``ts_ssa_reconstruction_residual`` — normalized mean-squared residual of the
  window reconstructed from the top ``n_components`` SVD modes (anti-diagonal
  averaging).  SELF_FIT_DESCRIPTIVE (M-094): the fit includes the current row,
  so this is an in-sample structural residual, NOT an out-of-sample anomaly
  measure; tagged ``self_fit_structural_residual`` / ``self_fit_descriptive``
  and flagged for the reconciler (DIAGNOSTIC_RESEARCH lane).
* ``ts_ssa_prior_reconstruction_error`` — NEW canonical (M-095): the SSA
  subspace is fitted on rows STRICTLY BEFORE the current row and the current
  observation is scored as a QUERY embedding against that prior subspace
  (``||q - V_k V_k^T q||^2 / ||q||^2``).  This is genuinely out-of-sample
  (current row never enters the fit) and is flagged for the reconciler
  (surface + layer_governance registration + model_timing contract + lane).

All operators are trailing-window, prefix-causal and deterministic.  Windows
too short to form a Hankel matrix emit NaN.  Invalid parameters raise
``ValueError``.  Production always uses ``missing_mode="strict_contiguous"``;
``interpolate`` is research-only and requires an explicit opt-in (M-096).
"""
from __future__ import annotations

from typing import Any, Sequence

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
_MIN_FINITE_FRAC = 0.5


# M-241: Hankel/SSA feasibility contracts, declared compile-time (ParamSpec +
# RelationalParamSpec) mirroring the runtime gates in ``_check_hankel_params`` —
# a guaranteed-infeasible combination (``window < embedding_dim``, or
# ``n_components >= min(window-embedding_dim+1, embedding_dim)``) is rejected by
# the central call validator BEFORE the rolling loop, exactly like the DMD family
# (``dmd._DMD_RELATIONAL_SPECS``).  ``_check_int`` (strict ints, R4-68) is
# retained as the runtime authority.
_HANKEL_BASE_PARAM_SPECS: dict[str, ParamSpec] = {
    "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    "embedding_dim": ParamSpec(
        dtype=int, min=2, param_role=ParamRole.ESTIMATOR_RESOLUTION
    ),
    "min_contiguous_fraction": ParamSpec(dtype=float, min=0.0, max=1.0),
}
_HANKEL_SSA_PARAM_SPECS: dict[str, ParamSpec] = dict(
    _HANKEL_BASE_PARAM_SPECS,
    **{"n_components": ParamSpec(dtype=int, min=1, param_role=ParamRole.ESTIMATOR_RESOLUTION)},
)
_HANKEL_BASE_RELATIONAL_SPECS: list[RelationalParamSpec] = [
    RelationalParamSpec(
        "window >= embedding_dim",
        "Hankel requires window >= embedding_dim "
        "(window={window}, embedding_dim={embedding_dim})",
    ),
]
# ``n_components < min(window-embedding_dim+1, embedding_dim)`` is NOT
# expressible in the restricted relational grammar (no function calls), so the
# one runtime condition is split into its two equivalent conjuncts — the same
# ``k < rows_h`` and ``k < e`` tests ``_ssa_*`` kernels enforce at runtime.
_HANKEL_SSA_RELATIONAL_SPECS: list[RelationalParamSpec] = list(_HANKEL_BASE_RELATIONAL_SPECS) + [
    RelationalParamSpec(
        "n_components < window - embedding_dim + 1",
        "SSA requires n_components < window-embedding_dim+1 (rows in the Hankel "
        "matrix) (n_components={n_components}, window={window}, "
        "embedding_dim={embedding_dim})",
    ),
    RelationalParamSpec(
        "n_components < embedding_dim",
        "SSA requires n_components < embedding_dim "
        "(n_components={n_components}, embedding_dim={embedding_dim})",
    ),
]

# M-241: Hankel/SSA feasibility telemetry.  Mirrors ``dmd.last_dmd_telemetry``:
# the most recent fail-closed row records WHY (window/embedding_dim infeasible,
# degenerate spectrum, ``n_components`` >= numerical rank).  Diagnostic
# accessor only — no operator surface is registered from it.
_LAST_HANKEL_TELEMETRY: dict[str, Any] = {
    "window": None,
    "embedding_dim": None,
    "n_components": None,
    "min_contiguous_fraction": None,
    "failure_reason": "not_run",
}


def last_hankel_telemetry() -> dict[str, Any]:
    """Telemetry for the most recent Hankel/SSA per-row fit.

    Returns ``{"window", "embedding_dim", "n_components",
    "min_contiguous_fraction", "failure_reason"}`` where ``failure_reason`` is
    one of ``"ok" | "insufficient_window" | "n_components_ge_rank" | "singular"
    | "invalid_params"``.  ``n_components`` is ``None`` for operators that do
    not take the parameter.
    """
    return dict(_LAST_HANKEL_TELEMETRY)


def _set_hankel_telemetry(
    *,
    window: int,
    embedding_dim: int,
    n_components: int | None,
    min_contiguous_fraction: float,
    failure_reason: str,
) -> None:
    _LAST_HANKEL_TELEMETRY["window"] = int(window)
    _LAST_HANKEL_TELEMETRY["embedding_dim"] = int(embedding_dim)
    _LAST_HANKEL_TELEMETRY["n_components"] = n_components
    _LAST_HANKEL_TELEMETRY["min_contiguous_fraction"] = float(min_contiguous_fraction)
    _LAST_HANKEL_TELEMETRY["failure_reason"] = failure_reason


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    cost: int,
    tags_extra: Sequence[str] = (),
    param_specs: dict[str, ParamSpec] | None = None,
    relational_specs: list[RelationalParamSpec] | None = None,
) -> OperatorMetadata:
    # R4-95: the trailing window is consumed as a strict trailing contiguous run;
    # exact row count is NOT required — rows with too short a run emit NaN.
    return OperatorMetadata(
        name=name,
        category="hankel",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "hankel", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic", "current_row_required",
            f"signature:{','.join(params)}->series", "domain:ts_structure",
            f"unit:{unit}", f"cost:{cost}",
            *tags_extra,
        ],
        window_semantics="trailing_contiguous",
        param_specs=dict(param_specs) if param_specs else {},
        relational_specs=list(relational_specs) if relational_specs else [],
    )


# M-094: ``ts_ssa_reconstruction_residual`` is a SELF-FIT structural residual.
# The fit window INCLUDES the current row (anti-diagonal reconstruction over
# ``col[i0:r+1]``), so the output measures how well the top-k modes describe
# the window that generated them — descriptive, NOT an out-of-sample anomaly.
# The reconciler consumes this to keep it in the DIAGNOSTIC_RESEARCH lane (it
# is already there at this HEAD via model_lane.py) and to document the timing
# contract as ``fit_cutoff_offset=0`` descriptive.
_SELF_FIT_STRUCTURAL_RESIDUAL: dict[str, dict[str, Any]] = {
    "ts_ssa_reconstruction_residual": {
        "self_fit_structural_residual": True,
        "fit_through_t": True,
        "fit_cutoff_offset": 0,
        "descriptive": True,
        "flag_for_reconciler": (
            "keep in DIAGNOSTIC_RESEARCH lane (self-fit structural residual, "
            "not an OOS anomaly); ModelTimingContract(fit_cutoff_offset=0, "
            "descriptive) is already explicit"
        ),
    },
}

# M-095: ``ts_ssa_prior_reconstruction_error`` is a NEW canonical — the SSA
# subspace is fitted on rows strictly before the current row and the current
# observation is scored as a QUERY.  Unlike ``ts_ssa_reconstruction_residual``
# this is genuinely OUT-OF-SAMPLE (the current row never enters the fit).  It
# is registered on the extended surface by this module's own idiom, but the
# reconciler must still add the reviewed surface / layer_governance /
# model_timing contract / lane.
_PRIOR_RECONSTRUCTION_FLAG: dict[str, dict[str, Any]] = {
    "ts_ssa_prior_reconstruction_error": {
        "fit_through_t": False,
        "current_row_is_query": True,
        "fit_cutoff_offset": 1,
        "out_of_sample": True,
        "flag_for_reconciler": (
            "new canonical ts_ssa_prior_reconstruction_error needs surface + "
            "layer_governance registration + explicit ModelTimingContract "
            "(fit_cutoff_offset=1, predictive — the current row is a query, "
            "never in the fit) + a reviewed model lane"
        ),
    },
}


# M-096: ``interpolate`` is a RESEARCH-ONLY missing-mode.  The production path is
# strict-contiguous — data on either side of a gap is never re-connected — and
# every public operator below passes ``missing_mode="strict_contiguous"``
# explicitly.  The ``interpolate`` branch is reachable ONLY when a caller
# explicitly opts in via ``missing_mode="interpolate"`` (never a default).  If a
# production interpolating variant is ever added it MUST be a separate,
# explicitly-named ``missing_mode`` opt-in gated by layer_governance /
# model_lane review — it must never silently activate in the production path.
_RESEARCH_ONLY_MISSING_MODES = frozenset({"interpolate"})


def _fill_window(
    chunk: np.ndarray,
    missing_mode: str = "strict_contiguous",
    min_contiguous_fraction: float = 0.8,
) -> np.ndarray | None:
    """Prepare a window for Hankel/SSA.

    Audit P1-A / M-096: the production default is ``strict_contiguous`` — the
    longest trailing contiguous finite run, with a 50%-finite coverage gate.
    Linear interpolation (which uses data *after* a gap to reconstruct
    observations before it) is RESEARCH-ONLY and requires an explicit
    ``missing_mode="interpolate"`` opt-in; it is never a default and never
    silently activates on the production path.

    R4-66: the contiguous run must cover at least ``min_contiguous_fraction`` of
    the window (default 0.8), otherwise ``None`` is returned so the row emits
    NaN.  Without this gate the same operator would alternate between an
    18-point and a 60-point Hankel matrix from day to day.

    P0-5: the current row is part of the contract.  In ``strict_contiguous``
    mode a NaN current observation returns ``None`` immediately (the row emits
    NaN) — the trailing run is never allowed to shift backwards past a missing
    current value and emit a stale-history factor.
    """
    # M-096 guard: an unknown ``missing_mode`` is a programming error and must
    # fail loudly, not silently fall through to the strict-contiguous branch
    # (which would make a typo'd ``missing_mode`` look like the production path).
    if missing_mode not in ("strict_contiguous", *_RESEARCH_ONLY_MISSING_MODES):
        raise ValueError(
            f"unknown missing_mode {missing_mode!r}; production must use "
            "'strict_contiguous', research-only interpolation must explicitly "
            "opt in with 'interpolate'"
        )
    if chunk.size == 0:
        return None
    # P0-5: the CURRENT row is required.  A NaN current observation must not
    # fall back to yesterday's contiguous finite run and emit a stale-history
    # Hankel/SSA factor (the old walk-back produced exactly that).  In strict
    # contiguous mode (the production default) a missing current row emits NaN;
    # only the research-only ``interpolate`` mode may reconstruct it.
    if missing_mode == "strict_contiguous" and not np.isfinite(chunk[-1]):
        return None
    finite = np.isfinite(chunk)
    n_fin = int(np.count_nonzero(finite))
    if n_fin / float(chunk.size) < _MIN_FINITE_FRAC:
        return None
    if n_fin == chunk.size:
        return chunk.astype(float)
    # M-096: this branch is REACHED ONLY via the explicit research-only opt-in
    # ``missing_mode="interpolate"``.  It re-connects data across a gap (using
    # values AFTER a gap to reconstruct observations before it), so it is
    # strictly non-causal and must never be the production path.  Every public
    # operator passes ``strict_contiguous``; if this branch is ever reached from
    # a public operator the wiring is broken.
    if missing_mode == "interpolate":
        n = chunk.size
        t = np.arange(n, dtype=float)
        filled = chunk.astype(float).copy()
        filled[~finite] = np.interp(t[~finite], t[finite], chunk[finite].astype(float))
        return filled
    # strict_contiguous (default): never re-connect data across a gap.
    end = chunk.size
    while end > 0 and not np.isfinite(chunk[end - 1]):
        end -= 1
    start = end
    while start > 0 and np.isfinite(chunk[start - 1]):
        start -= 1
    run_len = end - start
    if run_len / float(chunk.size) < float(min_contiguous_fraction):
        return None
    return chunk[start:end].astype(float)


def _hankel_singular_values(filled: np.ndarray, emb: int) -> np.ndarray | None:
    """SVD singular values (descending) of the trajectory Hankel matrix."""
    n = filled.size
    rows_h = n - emb + 1
    if rows_h < 1:
        return None
    h = np.empty((rows_h, emb), dtype=float)
    for j in range(emb):
        h[:, j] = filled[j : j + rows_h]
    s = np.linalg.svd(h, compute_uv=False)
    return s


def _hankel_effective_rank_series(
    x2d: np.ndarray,
    window: int,
    emb: int,
    missing_mode: str = "strict_contiguous",
    min_contiguous_fraction: float = 0.8,
) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w, e = int(window), int(emb)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            # R11 #92: the window W is the formal semantic.  A short warmup
            # prefix would give the SAME factor name a different effective
            # horizon, so rows before a full window (r+1 < w) fail closed (NaN).
            if r + 1 < w:
                # M-241: warmup — no full window yet (data-dependent, not a
                # parameter infeasibility).
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=None,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="insufficient_window",
                )
                continue
            i0 = max(0, r - w + 1)
            filled = _fill_window(col[i0 : r + 1], missing_mode, min_contiguous_fraction)
            if filled is None:
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=None,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="insufficient_window",
                )
                continue
            s = _hankel_singular_values(filled, e)
            if s is None:
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=None,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="insufficient_window",
                )
                continue
            energy = s ** 2
            total = float(energy.sum())
            if total <= 0 or not np.isfinite(total):
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=None,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="singular",
                )
                continue
            p = energy / total
            pos = p[p > 0]
            if pos.size == 0:
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=None,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="singular",
                )
                continue
            ent = -float(np.sum(pos * np.log(pos)))
            n = filled.size
            denom = min(n - e + 1, e)
            if denom > 0:
                out[r, c] = float(np.exp(ent) / denom)
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=None,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="ok",
                )
    return out


def _hankel_singular_gap_series(
    x2d: np.ndarray,
    window: int,
    emb: int,
    missing_mode: str = "strict_contiguous",
    min_contiguous_fraction: float = 0.8,
) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w, e = int(window), int(emb)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            # R11 #92: full window required before a value is emitted.
            if r + 1 < w:
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=None,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="insufficient_window",
                )
                continue
            i0 = max(0, r - w + 1)
            filled = _fill_window(col[i0 : r + 1], missing_mode, min_contiguous_fraction)
            if filled is None:
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=None,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="insufficient_window",
                )
                continue
            s = _hankel_singular_values(filled, e)
            if s is None:
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=None,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="insufficient_window",
                )
                continue
            # R11 #93: the spectrum must be well-defined on the ACTUAL
            # contiguous run — at least 2 singular values for a "gap" to exist.
            if s.size < 2 or s[0] <= _EPS:
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=None,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="singular",
                )
                continue
            s1 = float(s[0])
            s2 = float(s[1])
            total = float(s.sum())
            out[r, c] = (s1 - s2) / (total + _EPS)
            _set_hankel_telemetry(
                window=w, embedding_dim=e, n_components=None,
                min_contiguous_fraction=float(min_contiguous_fraction),
                failure_reason="ok",
            )
    return out


def _ssa_reconstruction_residual_series(
    x2d: np.ndarray,
    window: int,
    emb: int,
    n_components: int,
    missing_mode: str = "strict_contiguous",
    min_contiguous_fraction: float = 0.8,
) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w, e, k = int(window), int(emb), int(n_components)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            # R11 #92: full window required before a value is emitted.
            if r + 1 < w:
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=k,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="insufficient_window",
                )
                continue
            i0 = max(0, r - w + 1)
            filled = _fill_window(col[i0 : r + 1], missing_mode, min_contiguous_fraction)
            if filled is None:
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=k,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="insufficient_window",
                )
                continue
            n = filled.size
            rows_h = n - e + 1
            if rows_h < 1:
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=k,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="insufficient_window",
                )
                continue
            # R11 #93: the numerical-rank gate is on the ACTUAL contiguous run
            # (filled.size), never the nominal window — a gap-shrunken run that
            # still passes the coverage gate must not silently drop to fewer
            # reconstructed modes.
            if k >= min(rows_h, e):
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=k,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="n_components_ge_rank",
                )
                continue
            h = np.empty((rows_h, e), dtype=float)
            for j in range(e):
                h[:, j] = filled[j : j + rows_h]
            u, s, vt = np.linalg.svd(h, full_matrices=False)
            if s[0] <= _EPS or k > s.size:
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=k,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="singular",
                )
                continue
            # R11 #93: the numerical rank is read from the ACTUAL contiguous
            # run's singular spectrum — if the k-th singular value is
            # numerically zero relative to the largest, the requested
            # ``n_components`` exceeds the numerical rank of the run and the
            # reconstruction is degenerate (fail closed, never silently drop to
            # fewer modes).
            if s[k - 1] <= _EPS * s[0]:
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=k,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="singular",
                )
                continue
            hk = (u[:, :k] * s[:k]) @ vt[:k, :]
            recon = np.zeros(n, dtype=float)
            counts = np.zeros(n, dtype=float)
            for i in range(rows_h):
                for j in range(e):
                    recon[i + j] += hk[i, j]
                    counts[i + j] += 1.0
            recon /= counts
            resid = float(np.mean((filled - recon) ** 2))
            var_x = float(np.var(filled))
            # R11 #94: on a constant series Var(x) <= eps the normalised residual
            # ratio is undefined (0/0) — fail closed to NaN, never 0/EPS.
            if var_x <= _EPS:
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=k,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="singular",
                )
                continue
            out[r, c] = resid / var_x
            _set_hankel_telemetry(
                window=w, embedding_dim=e, n_components=k,
                min_contiguous_fraction=float(min_contiguous_fraction),
                failure_reason="ok",
            )
    return out


def _ssa_prior_reconstruction_error_series(
    x2d: np.ndarray,
    window: int,
    emb: int,
    n_components: int,
    missing_mode: str = "strict_contiguous",
    min_contiguous_fraction: float = 0.8,
) -> np.ndarray:
    """M-095: out-of-sample SSA novelty — fit on rows STRICTLY BEFORE the
    current row, score the current observation as a QUERY embedding.

    Prior fit: the strict trailing contiguous run ``col[i0:r]`` (current row
    EXCLUDED) is embedded into a trajectory Hankel matrix ``H``; the top-k
    right-singular subspace ``V_k`` (``(embedding_dim, k)``) is the prior
    low-rank structure.  Query: the current embedding ``q = [prior[-(e-1):],
    x_cur]`` (the length-``embedding_dim`` lagged vector ending at the current
    row) is projected onto ``V_k`` and the normalized squared reconstruction
    error ``||q - V_k V_k^T q||^2 / ||q||^2`` is emitted.  A small value means
    the current observation is well-explained by the past structure; a large
    value means a structural break / anomaly relative to the prior subspace.

    This is genuinely OUT-OF-SAMPLE — the current row never enters the fit —
    so it is NOT a self-fit descriptive residual (contrast
    ``ts_ssa_reconstruction_residual``, M-094).
    """
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w, e, k = int(window), int(emb), int(n_components)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            # R11 #92: full window required before a value is emitted.
            if r + 1 < w:
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=k,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="insufficient_window",
                )
                continue
            # P0-5: the CURRENT row is required.  A NaN current observation must
            # not fall back to yesterday's prior subspace and emit a stale
            # query — the operator fails closed.
            if not np.isfinite(col[r]):
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=k,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="insufficient_window",
                )
                continue
            i0 = max(0, r - w + 1)
            # M-095: the fit window is STRICTLY BEFORE the current row.
            prior = _fill_window(col[i0:r], missing_mode, min_contiguous_fraction)
            if prior is None or prior.size == 0:
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=k,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="insufficient_window",
                )
                continue
            n = prior.size
            rows_h = n - e + 1
            if rows_h < 1:
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=k,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="insufficient_window",
                )
                continue
            # R11 #93: the numerical-rank gate is on the ACTUAL contiguous run
            # (prior.size), never the nominal window.
            if k >= min(rows_h, e):
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=k,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="n_components_ge_rank",
                )
                continue
            h = np.empty((rows_h, e), dtype=float)
            for j in range(e):
                h[:, j] = prior[j : j + rows_h]
            _, s, vt = np.linalg.svd(h, full_matrices=False)
            if s[0] <= _EPS or k > s.size:
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=k,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="singular",
                )
                continue
            if s[k - 1] <= _EPS * s[0]:
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=k,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="singular",
                )
                continue
            # V_k: the top-k right-singular vectors span the prior low-rank
            # subspace in the embedding space.
            V_k = vt[:k, :].T  # (e, k)
            # Query embedding ending at the current row.
            q = np.concatenate([prior[-(e - 1):], [float(col[r])]])
            q_norm2 = float(np.dot(q, q))
            if q_norm2 <= _EPS:
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=k,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="singular",
                )
                continue
            proj = V_k @ (V_k.T @ q)
            resid_frac = float(np.dot(q - proj, q - proj)) / q_norm2
            if np.isfinite(resid_frac):
                out[r, c] = resid_frac
                _set_hankel_telemetry(
                    window=w, embedding_dim=e, n_components=k,
                    min_contiguous_fraction=float(min_contiguous_fraction),
                    failure_reason="ok",
                )
    return out


def _check_int(value: Any, name: str, minimum: int) -> int:
    """R4-68: strict integer contract — reject bools and non-integer floats so
    ``5.9`` never silently truncates to ``5`` and compiles to the same factor as
    ``5.0`` (a false search space)."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not bool")
    fv = float(value)
    if not np.isfinite(fv) or fv != float(int(fv)):
        raise ValueError(f"{name} must be an integer")
    iv = int(fv)
    if iv < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return iv


def _check_hankel_params(
    window: int, emb: int, n_components: int | None = None, min_contiguous_fraction: float = 0.8
) -> tuple[int, int, int | None, float]:
    """Runtime feasibility gate for the Hankel/SSA family (M-241).

    These conditions are the runtime authority and mirror the compile-time
    ``_HANKEL_BASE_RELATIONAL_SPECS`` / ``_HANKEL_SSA_RELATIONAL_SPECS``
    declarations: ``window >= embedding_dim`` and
    ``n_components < min(window - embedding_dim + 1, embedding_dim)``.  A
    combination that violates a relational spec is rejected here (loud
    ``ValueError``) before the rolling loop, and the rejection is recorded on
    ``last_hankel_telemetry()``.
    """
    w = _check_int(window, "window", 2)
    e = _check_int(emb, "embedding_dim", 2)
    if w - e + 1 < 1:
        _set_hankel_telemetry(
            window=w, embedding_dim=e, n_components=None,
            min_contiguous_fraction=float(min_contiguous_fraction),
            failure_reason="invalid_params",
        )
        raise ValueError("window must be >= embedding_dim")
    mcf = float(min_contiguous_fraction)
    if not np.isfinite(mcf) or not 0.0 < mcf <= 1.0:
        _set_hankel_telemetry(
            window=w, embedding_dim=e, n_components=None,
            min_contiguous_fraction=mcf,
            failure_reason="invalid_params",
        )
        raise ValueError("min_contiguous_fraction must be in (0, 1]")
    k = None
    if n_components is not None:
        k = _check_int(n_components, "n_components", 1)
        if k >= min(w - e + 1, e):
            _set_hankel_telemetry(
                window=w, embedding_dim=e, n_components=k,
                min_contiguous_fraction=mcf,
                failure_reason="invalid_params",
            )
            raise ValueError(
                "n_components must be < min(window-embedding_dim+1, embedding_dim)"
            )
    return w, e, k, mcf


@register_operator(
    name="ts_hankel_effective_rank",
    category="hankel",
    business_category="hankel",
    canonical="ts_hankel_effective_rank",
    source="hankel",
)
class TsHankelEffectiveRank(SeriesOperator):
    """Hankel 有效秩：``p_i = σ_i²/Σσ²``，``ER = exp(-Σ p log p)/min(H.shape)``。

    反映奇异谱的铺开程度——时间序列的有效自由度。纯正弦 → 接近 1/秩（低）；
    白噪声 → 接近 1（高）。P2。
    """

    metadata = _metadata(
        "ts_hankel_effective_rank",
        "奇异值能量分布的熵指数 / min(H.shape)（有效自由度）。",
        ["x", "window", "embedding_dim", "min_contiguous_fraction"],
        unit="ratio",
        cost=6,
        param_specs=_HANKEL_BASE_PARAM_SPECS,
        relational_specs=_HANKEL_BASE_RELATIONAL_SPECS,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, embedding_dim: int = 15, min_contiguous_fraction: float = 0.8, **_: Any
    ) -> pd.DataFrame:
        w, e, _, mcf = _check_hankel_params(window, embedding_dim, None, min_contiguous_fraction)
        return frame_like(x, _hankel_effective_rank_series(x.to_numpy(dtype=float), w, e, "strict_contiguous", mcf))


@register_operator(
    name="ts_hankel_singular_gap",
    category="hankel",
    business_category="hankel",
    canonical="ts_hankel_singular_gap",
    source="hankel",
)
class TsHankelSingularGap(SeriesOperator):
    """Hankel 奇异值间隙：``(σ_1 - σ_2)/(Σσ + eps)``。

    大 → 第一个模式远强于其余（强主导结构）；小 → 模式接近（噪声/多周期）。
    P2。
    """

    metadata = _metadata(
        "ts_hankel_singular_gap",
        "最大与次大奇异值的相对间隙（主导结构强度）。",
        ["x", "window", "embedding_dim", "min_contiguous_fraction"],
        unit="ratio",
        cost=6,
        param_specs=_HANKEL_BASE_PARAM_SPECS,
        relational_specs=_HANKEL_BASE_RELATIONAL_SPECS,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, embedding_dim: int = 15, min_contiguous_fraction: float = 0.8, **_: Any
    ) -> pd.DataFrame:
        w, e, _, mcf = _check_hankel_params(window, embedding_dim, None, min_contiguous_fraction)
        return frame_like(x, _hankel_singular_gap_series(x.to_numpy(dtype=float), w, e, "strict_contiguous", mcf))


@register_operator(
    name="ts_ssa_reconstruction_residual",
    category="hankel",
    business_category="hankel",
    canonical="ts_ssa_reconstruction_residual",
    source="hankel",
)
class TsSsaReconstructionResidual(SeriesOperator):
    """SSA 重构残差：前 ``n_components`` 个 SVD 模式经反对角平均重构窗口，
    ``residual = mean((x - x̂)²)/(var(x) + eps)``。

    残差低 → 序列几乎被少数结构模式解释（强可预测结构）；高 → 残差大（噪声/
    非线性）。P2。SELF_FIT_DESCRIPTIVE（M-094）：拟合窗口含当前样本，这是
    in-sample 结构残差，不是样本外异常测度——样本外版本用
    ``ts_ssa_prior_reconstruction_error``。
    """

    metadata = _metadata(
        "ts_ssa_reconstruction_residual",
        "前 n_components 个 SVD 模式重构窗口的归一化均方残差"
        "（SELF_FIT_DESCRIPTIVE：拟合含当前样本的描述性结构残差，非样本外异常测度；"
        "样本外版本见 ts_ssa_prior_reconstruction_error）。",
        ["x", "window", "embedding_dim", "n_components", "min_contiguous_fraction"],
        unit="ratio",
        cost=7,
        tags_extra=["self_fit_structural_residual", "self_fit_descriptive"],
        param_specs=_HANKEL_SSA_PARAM_SPECS,
        relational_specs=_HANKEL_SSA_RELATIONAL_SPECS,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 60,
        embedding_dim: int = 15,
        n_components: int = 3,
        min_contiguous_fraction: float = 0.8,
        **_: Any,
    ) -> pd.DataFrame:
        w, e, k, mcf = _check_hankel_params(window, embedding_dim, n_components, min_contiguous_fraction)
        return frame_like(x, _ssa_reconstruction_residual_series(x.to_numpy(dtype=float), w, e, k, "strict_contiguous", mcf))


@register_operator(
    name="ts_ssa_prior_reconstruction_error",
    category="hankel",
    business_category="hankel",
    canonical="ts_ssa_prior_reconstruction_error",
    source="hankel",
)
class TsSsaPriorReconstructionError(SeriesOperator):
    """SSA 样本外先验重构误差（M-095）：先验子空间只在当前行之前拟合
    （``prior = col[i0:r]``），当前观测作为 QUERY 嵌入投影到先验子空间
    ``V_k``：``err = ||q - V_k V_k^T q||²/||q||²``。

    小 → 当前观测被过去结构良好解释（平稳延续）；大 → 相对先验子空间的结构
    突变/异常。真正的样本外测度（当前行永不进入拟合），非自拟合描述性残差。
    """

    metadata = _metadata(
        "ts_ssa_prior_reconstruction_error",
        "当前观测相对先验低秩 SSA 子空间（仅 t-1 前拟合）的归一化查询重构误差"
        "（OUT-OF-SAMPLE：当前行是 query，永不进入拟合）。",
        ["x", "window", "embedding_dim", "n_components", "min_contiguous_fraction"],
        unit="ratio",
        cost=7,
        tags_extra=["prior_reconstruction_error", "current_row_is_query", "out_of_sample"],
        param_specs=_HANKEL_SSA_PARAM_SPECS,
        relational_specs=_HANKEL_SSA_RELATIONAL_SPECS,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 60,
        embedding_dim: int = 15,
        n_components: int = 3,
        min_contiguous_fraction: float = 0.8,
        **_: Any,
    ) -> pd.DataFrame:
        w, e, k, mcf = _check_hankel_params(window, embedding_dim, n_components, min_contiguous_fraction)
        return frame_like(x, _ssa_prior_reconstruction_error_series(x.to_numpy(dtype=float), w, e, k, "strict_contiguous", mcf))


_NEW_CANONICALS = (
    "ts_hankel_effective_rank",
    "ts_hankel_singular_gap",
    "ts_ssa_reconstruction_residual",
    "ts_ssa_prior_reconstruction_error",
)


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
