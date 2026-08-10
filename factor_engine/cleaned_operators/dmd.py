# -*- coding: utf-8 -*-
"""Dynamic Mode Decomposition primitives (2026-08-08 Gemini V2 round).

Hankel / delay-embedded DMD on a single series: the trailing window is
embedded into an ``L x K`` Hankel matrix (``L = dim`` rows, column stride
``delay``), and ``X = H[:, :-1]``, ``Y = H[:, 1:]`` approximate the linear
Koopman-style propagator ``Ã = U_r^T Y V_r Σ_r^{-1}`` from a rank-``r`` SVD of
``X``.  The eigenvalues of ``Ã`` give per-mode growth ``g = ln|λ|/Δt`` and
frequency ``f = arg(λ)/(2π·Δt)``; the *exact* DMD modes are
``Φ = Y V Σ^{-1} W`` (``W`` = eigenvectors of ``Ã``) and the amplitudes are the
pseudoinverse solution ``b = Φ† x_1``.  Mode energy is the finite-horizon sum
``E_j = Σ_{t=0}^{K-1} |b_j λ_j^t|²``.  P1-L: energy is carried in LOG space
throughout — ``log E_j = log|b_j|² + log Σ ρ_j^t`` (``ρ_j = |λ_j|²``) with the
geometric log-sum computed in a cancellation-free closed form, so a growing
mode (``|λ| > 1``) can never overflow or need an ``exp(700)`` clamp that would
distort relative mode energy.  Concentration is then ``exp(logsumexp(top_k) -
logsumexp(all))``, a stable log-ratio.

All three operators are Research-surface: SVD/eigen decomposition is
numerically delicate, so they are excluded from the default mining grammar
and fail closed whenever the embedding, the singular-value rank, or the
condition number is degenerate.

* ``ts_dmd_dominant_growth_rate``  (``log|λ|`` per bar — unit
  ``log_growth_per_bar``)
* ``ts_dmd_dominant_frequency``   (``|arg λ|/(2π)`` cycles per bar — unit
  ``cycles_per_bar``).  P1-L: the dominant OSCILLATORY frequency — the max
  energy mode among the imaginary modes (a max-energy mode that is real is
  level persistence, not oscillation); NaN only when NO mode is imaginary.
* ``ts_dmd_mode_concentration``   (top-``top_k`` mode energy share; ``top_k``
  exceeding the number of *physical* (conjugate-merged) modes fails closed)

P1-L (price-level vs return semantics): raw level series have ``λ ≈ 1`` as
their dominant mode (level persistence), which is a different dynamics from a
return series.  The variants make the input semantic explicit:
``ts_dmd_level_*`` declares ``x`` a price level (``price_level`` /
``log_price_level``) and ``ts_dmd_return_*`` declares ``x`` a return.  The
legacy unconstrained ``ts_dmd_*`` names remain for compatibility.

Deterministic (no randomized SVD / subsampling), strict-PIT, NaN fail-closed.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from cleaned_operators.base import ParamRole, ParamSpec, RelationalParamSpec
from cleaned_operators.gemini_v2_common import (
    frame_like,
    register_dual,
    trailing_contiguous_finite,
    union_research,
)

_EPS = 1e-12

# R16-087/088: VERSIONED numerical policy.  ``np.linalg.pinv`` default rcond is
# library-version dependent; a fixed rcond + relative imaginary-mode tolerance
# make the DMD decomposition reproducible and definition-stable across numpy
# versions (both values enter the definition/evidence hash).
_PINV_RCOND = 1e-12
_IMAG_MODE_RTOL = 1e-9

# R5 P1-01/P1-02: ``rank``/``dim``/``delay`` are validated ints (a fractional
# value is rejected, never silently truncated).
#
# Cross-cutting P1-L: DMD embedding knobs are ESTIMATOR_RESOLUTION (only small
# reviewed search grids, never full resolution — they change estimator bias, not
# the trading rule); ``window`` is the HORIZON dimension.
_DMD_PARAM_SPECS = {
    "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    "rank": ParamSpec(dtype=int, min=1, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    "dim": ParamSpec(dtype=int, min=2, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    "delay": ParamSpec(dtype=int, min=1, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    "top_k": ParamSpec(dtype=int, min=1, param_role=ParamRole.ESTIMATOR_RESOLUTION),
}
# R6-157: ``keys(param_specs) ⊆ param_names`` is a registry invariant.  The
# dominant-growth/frequency canonicals take ``window/rank/dim/delay`` only —
# ``top_k`` belongs solely to ``ts_dmd_mode_concentration``.  A shared spec
# dict would declare a contract for a parameter the canonical does not have.
_DMD_BASE_SPEC = {k: _DMD_PARAM_SPECS[k] for k in ("window", "rank", "dim", "delay")}
_DMD_CONCENTRATION_SPEC = dict(_DMD_PARAM_SPECS)
# R9-OP-005: ONE feasibility formula for the DMD family, shared by the
# ParamSpec/RelationalParamSpec (declared mirror) AND the runtime kernel guard.
# The declared relational specs below are the search-facing spelling of the SAME
# conditions ``dmd_feasibility`` enforces at runtime — never a fourth copy of the
# math.  ``window`` at runtime is the CONTIGUOUS finite run length, so a
# gap-shrunken window fails here exactly as an equally small declared window
# fails at compile time.
_DMD_RELATIONAL_SPECS = [
    RelationalParamSpec(
        "window - (dim - 1) * delay >= rank + 2",
        "DMD requires K = window-(dim-1)*delay >= rank+2 "
        "(window={window}, dim={dim}, delay={delay}, rank={rank})",
    ),
    RelationalParamSpec(
        "rank <= dim",
        "DMD requires rank <= dim (rank={rank}, dim={dim})",
    ),
]
_DMD_CONCENTRATION_RELATIONAL_SPECS = list(_DMD_RELATIONAL_SPECS) + [
    RelationalParamSpec(
        "top_k <= rank",
        "ts_dmd_mode_concentration requires top_k <= rank (top_k={top_k}, rank={rank})",
    )
]


def dmd_feasibility(*, window: int, rank: int, dim: int, delay: int, top_k: int | None = None) -> bool:
    """Single DMD feasibility contract (R9-OP-005).

    ``K = window - (dim-1)*delay`` is the embedding column count; a valid SVD
    propagator needs ``K >= rank+2`` and ``K >= 4``, ``rank <= dim`` (R6-200/201).
    ``top_k`` (concentration only) must satisfy ``1 <= top_k <= rank``.  Every
    layer — ParamSpec, RelationalParamSpec, history planning and the runtime
    kernel guard — expresses this one function so compile-valid params can never
    be runtime-guaranteed-NaN.
    """
    K = int(window) - (int(dim) - 1) * int(delay)
    if K < int(rank) + 2 or K < 4:
        return False
    if int(rank) < 1 or int(rank) > int(dim):
        return False
    if top_k is not None and (int(top_k) < 1 or int(top_k) > int(rank)):
        return False
    return True


def _logsumexp(xs: Sequence[float]) -> float:
    """Stable log-sum-exp: ``log( sum exp(x_i) )`` with the max pulled out."""
    m = float(max(xs))
    return m + float(np.log(np.sum(np.exp(np.asarray(xs, dtype=float) - m))))


def _log1mexp(x: float) -> float:
    """``log(1 - exp(x))`` computed stably for ``x < 0`` (R30 §15).

    For large negative ``x``, ``exp(x)`` underflows to 0 and the result is
    simply ``0``; near ``x == 0`` ``-expm1(x)`` avoids the catastrophic
    ``1 - exp(x)`` cancellation.
    """
    if x >= 0.0:  # pragma: no cover - guarded by callers
        return float("-inf")
    if x < -0.6931471805599453:  # log(2): exp(x) is tiny enough
        return float(np.log1p(-np.exp(x)))
    return float(np.log(-np.expm1(x)))


def _log_expm1(x: float) -> float:
    """``log(exp(x) - 1)`` computed stably for ``x > 0`` (R30 §15).

    ``exp(x)`` overflows to inf for ``x ~ 710+``; the identity
    ``log(exp(x)-1) = x + log1p(-exp(-x))`` keeps the result finite for any
    finite positive ``x``.
    """
    return float(x + np.log1p(-np.exp(-x)))


def _log_finite_horizon_sum(log_rho: float, K: int) -> float:
    """``log( sum_{t=0}^{K-1} rho^t )`` in a cancellation-free closed form,
    taking ``log_rho = log(rho)`` so ``rho = |λ|²`` is NEVER materialised.

    R26-123/124: ``rho = abs(lambda)**2`` overflows to inf for a finite but
    large lambda, and ``abs_b**2`` underflows to 0 for a tiny-but-nonzero mode
    amplitude.  Working in ``log_rho = 2·log|λ|`` keeps every branch in log
    space: near ``rho == 1`` the direct finite sum is used (``rho - 1`` never
    cancels catastrophically); ``rho > 1`` and ``rho < 1`` use the geometric
    closed form in log domain.  The result is finite for any finite ``log_rho``.

    R30 §15 (P0-011): the old code still executed ``r = exp(log_rho)`` to choose
    the branch, which overflowed to ``inf`` at ``log_rho = 1000`` and produced
    ``log(expm1(1000)) = log(inf) = inf``.  All branches are now decided and
    evaluated in log space via :func:`_log1mexp` / :func:`_log_expm1`; ``rho``
    is never materialised.
    """
    lr = float(log_rho)
    if lr <= float("-inf"):
        # rho == 0: only the t=0 term survives (0^0 == 1): sum == 1, log == 0.
        return 0.0
    if abs(lr) < 1e-6:
        # rho == exp(lr) ≈ 1: direct finite sum in LOG space.
        # sum_{t=0}^{K-1} rho^t ≈ K (all terms ≈ 1), so log ≈ log(K).
        return float(np.log(K))
    if lr > 0.0:
        K_log_r = K * lr
        # log((r^K - 1)/(r - 1)) = K log r + log(1 - r^{-K}) - log(r - 1)
        #   = K*lr + log1mexp(-K*lr) - log_expm1(lr)
        return (
            K_log_r
            + _log1mexp(-K_log_r)
            - _log_expm1(lr)
        )
    # r < 1: log((1 - r^K)/(1 - r)) = log1mexp(K*lr) - log1mexp(lr)
    # (K*lr < 0 and lr < 0, so both log1mexp arguments are negative).
    return _log1mexp(K * lr) - _log1mexp(lr)


def _hankel_dmd(v: np.ndarray, rank: int, dim: int, delay: int) -> dict[str, Any] | None:
    n = v.shape[0]
    L = max(2, int(dim))
    dl = max(1, int(delay))
    # R9-OP-005: runtime guard == declared feasibility contract (single source).
    if not dmd_feasibility(window=n, rank=int(rank), dim=L, delay=dl):
        return None
    K = n - (L - 1) * dl
    H = np.stack([v[i : i + K] for i in range(0, (L - 1) * dl + 1, dl)], axis=0)
    X = H[:, :-1]
    Y = H[:, 1:]
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # R5 P1-02: an infeasible rank is REJECTED (fail-closed), never silently
    # clipped down — ``rank=4`` and ``rank=3`` must not compile to the same
    # factor, which the old ``min(rank, S.shape[0]-1)`` did.
    r = int(rank)
    if r < 1 or r > S.shape[0]:
        return None
    if S[0] <= _EPS or S[r - 1] <= _EPS * S[0]:
        return None
    cond = S[0] / S[r - 1]
    if cond > 1e12:
        return None
    Ur = U[:, :r]
    Vr = Vt[:r, :].T
    Sr_inv = np.diag(1.0 / S[:r])
    A_tilde = Ur.T @ Y @ Vr @ Sr_inv
    # P0-06 rewrite: keep the (eigval, eigvector) PAIRS together so the filtered
    # subset stays aligned (eigenvalues alone cannot be re-paired to modes).
    eig_vals, W = np.linalg.eig(A_tilde)
    finite = np.isfinite(eig_vals)
    if not finite.all():
        eig_vals = eig_vals[finite]
        W = W[:, finite]
    if eig_vals.size == 0:
        return None
    # exact DMD modes Phi = Y V Sigma^{-1} W  and  amplitudes b = Phi^dagger x_1.
    # (The old code used the SVD-coordinate projection ``Ur.T @ H[:, 0]`` as if
    # it were the mode amplitude — that is not the standard DMD amplitude and
    # mis-orders the dominant mode / energies.)
    Phi = Y @ (Vr / S[:r]) @ W          # (L, r) exact modes
    x1 = X[:, 0]
    # R16-087: the pinv rcond is an explicit, VERSIONED numerical policy —
    # ``np.linalg.pinv``'s library-default rcond can change the effective rank
    # across numpy versions.  The value enters the definition/evidence hash.
    b = np.linalg.pinv(Phi, rcond=_PINV_RCOND) @ x1   # (r,) mode amplitudes
    # P1-L (#136): mode energy carried in LOG space throughout.  With
    # ``rho = |lam|^2``,  log E_j = log|b_j|^2 + log( sum_{t=0}^{K-1} rho_j^t ).
    # The geometric log-sum is a cancellation-free closed form (near ``rho == 1``
    # the direct finite sum is used so ``rho - 1`` never cancels catastrophically),
    # so a growing mode (``rho > 1``) can never overflow and no ``exp(700)``
    # clamp is needed — concentration is a stable log-ratio downstream.
    # R26-123/124: work in log domain throughout.  ``log_rho = 2·log|λ|`` (never
    # ``abs(λ)²``, which overflows); ``log_b2 = 2·log|b|`` (never ``abs_b**2``,
    # which underflows a tiny-but-nonzero mode to 0 and misreads it as
    # zero-energy).
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rho = np.where(eig_vals == 0.0, -np.inf, 2.0 * np.log(np.abs(eig_vals)))
    log_sum = np.array(
        [_log_finite_horizon_sum(float(log_rho[j]), int(K)) for j in range(eig_vals.size)],
        dtype=float,
    )
    # R16-085: a ZERO-amplitude mode has NO energy — ``log(|b|^2 + EPS)`` gave a
    # b=0 mode a fabricated non-zero energy via the machine epsilon.  ``-inf``
    # propagates the correct semantics (the mode ranks last / vanishes).
    abs_b = np.abs(b)
    # R28 §四十六: same errstate guard as ``log_rho`` so a zero-amplitude mode
    # resolves to ``-inf`` without emitting a spurious divide-by-zero warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_b2 = np.where(abs_b == 0.0, -np.inf, 2.0 * np.log(abs_b))
    log_energy = log_b2 + log_sum
    # R26-125: if EVERY mode has ``log_energy == -inf`` (all amplitudes exactly
    # zero), fail closed explicitly — never let ``-inf - -inf -> NaN`` emerge
    # from an accidental numeric path in the downstream log-ratio.
    if np.all(np.isneginf(log_energy)):
        return None
    order = np.argsort(-log_energy)
    return {
        "eig": eig_vals[order],
        "log_energy": log_energy[order],
    }


def _dmd_series(x2d: np.ndarray, window: int, rank: int, dim: int, delay: int, which: str, top_k: int = 2) -> np.ndarray:
    rows, cols = x2d.shape
    w = max(10, int(window))
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo : r + 1])
            # R9-OP-005: the runtime gate is the SAME feasibility contract as the
            # declared ParamSpec/RelationalParamSpec — the run length substitutes
            # for ``window``, so a gap-shrunken sample is rejected exactly as the
            # equivalent small declared window would be at compile time.
            if not dmd_feasibility(window=v.size, rank=int(rank), dim=int(dim), delay=int(delay)):
                continue
            res = _hankel_dmd(v, int(rank), int(dim), int(delay))
            if res is None:
                continue
            lam0 = res["eig"][0]
            if which == "growth":
                # R16-086: a ZERO eigenvalue has no defined growth rate —
                # ``log(max(|λ|, EPS))`` let a machine-constant decide an
                # arbitrary finite growth.  λ == 0 -> NaN (undefined).
                val = float(np.nan) if abs(lam0) == 0.0 else float(np.log(abs(lam0)))
            elif which == "frequency":
                # P1-L (#138): the useful metric is the dominant OSCILLATORY
                # frequency — the max-energy mode among the IMAGINARY modes.  A
                # real max-energy mode (λ ≈ 1) is level persistence, not an
                # oscillation, so it must NOT suppress the frequency to NaN.
                # NaN only when NO mode is imaginary.
                dom = None
                for lam in np.asarray(res["eig"]):  # already energy-descending
                    # R16-088: RELATIVE imaginary-mode tolerance (versioned), not
                    # an absolute hidden constant — an absolute ``1e-9`` had no
                    # scale semantics across differently-scaled series.
                    if abs(lam.imag) >= _IMAG_MODE_RTOL * max(1.0, abs(lam)):
                        dom = lam
                        break
                if dom is None:
                    continue
                # R6-202: for a real-valued time series the eigenvalues come in
                # conjugate pairs (λ, conj(λ)) with near-identical energies.  The
                # raw sign of arg(λ) can flip day-to-day from tie ordering even
                # though both signs describe the SAME oscillation.  Canonical
                # frequency uses |arg λ|/(2π) so the dominant frequency is stable
                # and unambiguous for a real series.
                val = float(abs(np.angle(dom)) / (2.0 * np.pi))
            else:
                # R6-200: top_k > rank is rejected, never silently clipped —
                # rank=3 with top_k=3,4,10 must not compile to the same factor.
                tk = int(top_k)
                if tk > int(rank):
                    continue
                # R6-203: mode concentration must count a conjugate pair ONCE —
                # a real oscillation splits its energy between the +f and -f
                # modes, so top_k=1 on the split spectrum understates the mode's
                # concentration.  Merge conjugate pairs (equal |λ| and equal
                # |arg λ|, distinct values).
                #
                # P1-L (#136): energies are LOG energies; a conjugate pair's
                # combined energy is logsumexp(pair_log, mate_log), and the final
                # concentration is exp(logsumexp(top_k) - logsumexp(all)) — a
                # stable log-ratio that never overflows even when |λ| ≫ 1.
                eig = np.asarray(res["eig"])
                log_energy = np.asarray(res["log_energy"], dtype=float)
                merged_log: list[float] = []
                used = np.zeros(eig.shape[0], dtype=bool)
                for i in range(eig.shape[0]):
                    if used[i]:
                        continue
                    pair_log = float(log_energy[i])
                    for j in range(i + 1, eig.shape[0]):
                        if used[j]:
                            continue
                        is_conj = (
                            eig[i] != eig[j]
                            and abs(abs(eig[i]) - abs(eig[j])) < 1e-6
                            and abs(abs(np.angle(eig[i])) - abs(np.angle(eig[j]))) < 1e-6
                        )
                        if is_conj:
                            m = max(pair_log, float(log_energy[j]))
                            pair_log = m + float(
                                np.log1p(np.exp(-abs(pair_log - float(log_energy[j]))))
                            )
                            used[j] = True
                            break
                    merged_log.append(pair_log)
                merged_log.sort(reverse=True)
                # R9-OP-004 (dead-parameter region): ``top_k > rank`` is already
                # rejected by the declared relational spec, but the *physical*
                # mode count can still be smaller than rank after conjugate-pair
                # merging — with rank=4, top_k=2,3,4 then all compile to the
                # SAME output when only 2 physical modes exist, a fake search
                # space.  Never ``min``-clip: exceeding the physical mode count
                # is a data-dependent infeasibility, so the canonical fails
                # closed (NaN) instead of silently collapsing to fewer modes.
                if tk > len(merged_log):
                    continue
                total_log = _logsumexp(merged_log)
                if not np.isfinite(total_log) or total_log <= float(np.log(_EPS)):
                    continue
                top_log = _logsumexp(merged_log[:tk])
                val = float(np.exp(top_log - total_log))
            if np.isfinite(val):
                out[r, c] = val
    return out


def _ts_dmd_dominant_growth_rate(x: pd.DataFrame, window: int = 120, rank: int = 4, dim: int = 4, delay: int = 1) -> pd.DataFrame:
    if int(window) < 10:
        raise ValueError("ts_dmd_dominant_growth_rate requires window >= 10")
    out = _dmd_series(x.to_numpy(dtype=float), int(window), int(rank), int(dim), int(delay), "growth")
    return frame_like(x, out)


def _ts_dmd_dominant_frequency(x: pd.DataFrame, window: int = 120, rank: int = 4, dim: int = 4, delay: int = 1) -> pd.DataFrame:
    if int(window) < 10:
        raise ValueError("ts_dmd_dominant_frequency requires window >= 10")
    out = _dmd_series(x.to_numpy(dtype=float), int(window), int(rank), int(dim), int(delay), "frequency")
    return frame_like(x, out)


def _ts_dmd_mode_concentration(x: pd.DataFrame, window: int = 120, rank: int = 4, dim: int = 4, delay: int = 1, top_k: int = 2) -> pd.DataFrame:
    if int(window) < 10:
        raise ValueError("ts_dmd_mode_concentration requires window >= 10")
    out = _dmd_series(x.to_numpy(dtype=float), int(window), int(rank), int(dim), int(delay), "concentration", int(top_k))
    return frame_like(x, out)


def _dmd_variant_spec(
    canonical: str,
    fn: Any,
    params: list[str],
    *,
    unit: str,
    input_units: dict[str, str],
    input_semantic: str,
) -> dict[str, Any]:
    """Build a P1-L level/return variant spec.

    Identical kernel, but the declared INPUT SEMANTIC differs: ``level``
    variants take a price level (``price_level`` / ``log_price_level``), return
    variants take a return.  This makes the level-vs-return dynamics explicit in
    the definition metadata so a search cannot silently mix a raw price level
    into an operator that means return dynamics (or vice versa).
    """
    is_conc = canonical.endswith("mode_concentration")
    return {
        "fn": fn,
        "params": params,
        "category": "dynamic_mode",
        "domain": "dynamical_systems",
        "unit": unit,
        "cost": 8,
        "tags_extra": [f"input_semantic:{input_semantic}"],
        "output_unit": unit,
        "param_specs": _DMD_CONCENTRATION_SPEC if is_conc else _DMD_BASE_SPEC,
        "relational_specs": _DMD_CONCENTRATION_RELATIONAL_SPECS if is_conc else _DMD_RELATIONAL_SPECS,
        "input_units": input_units,
    }


_SPECS: dict[str, dict[str, Any]] = {
    # R9-OP-006 (honest units): ``log|λ|`` is a per-sample (per-bar) log-growth
    # rate, NOT a plain ``level``; ``|arg λ|/(2π)`` is cycles PER BAR, not bare
    # ``cycles``.  Minute vs daily bars differ by 240x on both, so the unit must
    # carry the bar reference to keep the semantics comparable across markets.
    "ts_dmd_dominant_growth_rate": {
        "fn": _ts_dmd_dominant_growth_rate,
        "params": ["x", "window", "rank", "dim", "delay"],
        "category": "dynamic_mode",
        "domain": "dynamical_systems",
        "unit": "log_growth_per_bar",
        "cost": 8,
        "tags_extra": [],
        "output_unit": "log_growth_per_bar",
        "param_specs": _DMD_BASE_SPEC,
        "relational_specs": _DMD_RELATIONAL_SPECS,
    },
    "ts_dmd_dominant_frequency": {
        "fn": _ts_dmd_dominant_frequency,
        "params": ["x", "window", "rank", "dim", "delay"],
        "category": "dynamic_mode",
        "domain": "dynamical_systems",
        "unit": "cycles_per_bar",
        "cost": 8,
        "tags_extra": [],
        "output_unit": "cycles_per_bar",
        "param_specs": _DMD_BASE_SPEC,
        "relational_specs": _DMD_RELATIONAL_SPECS,
    },
    "ts_dmd_mode_concentration": {
        "fn": _ts_dmd_mode_concentration,
        "params": ["x", "window", "rank", "dim", "delay", "top_k"],
        "category": "dynamic_mode",
        "domain": "dynamical_systems",
        "unit": "ratio",
        "cost": 8,
        "tags_extra": [],
        "output_unit": "ratio",
        "param_specs": _DMD_CONCENTRATION_SPEC,
        "relational_specs": _DMD_CONCENTRATION_RELATIONAL_SPECS,
    },
    # P1-L (#137): explicit price-level vs return semantics.  The kernel is the
    # same DMD; the DIFFERENCE is the declared input contract — raw levels have
    # λ ≈ 1 as their dominant mode (level persistence), returns show oscillatory
    # dynamics, and a search must not mix the two.
    "ts_dmd_level_dominant_growth_rate": _dmd_variant_spec(
        "ts_dmd_level_dominant_growth_rate",
        _ts_dmd_dominant_growth_rate,
        ["x", "window", "rank", "dim", "delay"],
        unit="log_growth_per_bar",
        input_units={"x": "price_level"},
        input_semantic="level",
    ),
    "ts_dmd_level_dominant_frequency": _dmd_variant_spec(
        "ts_dmd_level_dominant_frequency",
        _ts_dmd_dominant_frequency,
        ["x", "window", "rank", "dim", "delay"],
        unit="cycles_per_bar",
        input_units={"x": "price_level"},
        input_semantic="level",
    ),
    "ts_dmd_level_mode_concentration": _dmd_variant_spec(
        "ts_dmd_level_mode_concentration",
        _ts_dmd_mode_concentration,
        ["x", "window", "rank", "dim", "delay", "top_k"],
        unit="ratio",
        input_units={"x": "price_level"},
        input_semantic="level",
    ),
    "ts_dmd_return_dominant_growth_rate": _dmd_variant_spec(
        "ts_dmd_return_dominant_growth_rate",
        _ts_dmd_dominant_growth_rate,
        ["x", "window", "rank", "dim", "delay"],
        unit="log_growth_per_bar",
        input_units={"x": "return"},
        input_semantic="return",
    ),
    "ts_dmd_return_dominant_frequency": _dmd_variant_spec(
        "ts_dmd_return_dominant_frequency",
        _ts_dmd_dominant_frequency,
        ["x", "window", "rank", "dim", "delay"],
        unit="cycles_per_bar",
        input_units={"x": "return"},
        input_semantic="return",
    ),
    "ts_dmd_return_mode_concentration": _dmd_variant_spec(
        "ts_dmd_return_mode_concentration",
        _ts_dmd_mode_concentration,
        ["x", "window", "rank", "dim", "delay", "top_k"],
        unit="ratio",
        input_units={"x": "return"},
        input_semantic="return",
    ),
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
            source="dmd",
            tags_extra=spec["tags_extra"],
            output_unit=spec.get("output_unit"),
            param_specs=spec.get("param_specs"),
            relational_specs=spec.get("relational_specs"),
            input_units=spec.get("input_units"),
        )
    union_research(*_SPECS.keys())


_register()
