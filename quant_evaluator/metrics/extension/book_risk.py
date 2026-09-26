"""M04 book-conditional tail-loss extension kernels (QE-EXT-SPEC-1.0 §7.4).

Kernels
-------
- ``book_conditional_loss``: given the current book's worst tail (loss
  ``-r0`` upper tail, §6.6 boundary weights ``w``), the candidate's expected
  return there is ``-sum(w * r1) / sum(w)``.  A negative value means the
  candidate makes money on the book's bad days.
- ``book_tail_loss_delta``: ``[-sum(w*r0)/sum(w)] - [-sum(w*r1)/sum(w)]`` —
  positive means improvement.  Both numbers use the **fixed book original
  tail dates** (one weight vector); they are never computed from two
  independently selected tails.  ``combined_book_against_book`` is the only
  comparison kind interpretable as portfolio conditional-loss improvement;
  ``standalone_against_book`` only describes complementarity.
- ``book_downside_beta``: ``sum(w*(r0-mu0w)*(r1-mu1w)) / sum(w*(r0-mu0w)^2)``
  with tail-weighted means.  Tail-mass and baseline weighted-variance gates
  are mandatory; no undefined "annualized beta" is produced.

Inputs
------
``baseline_returns`` is the explicit current-book net return series (not a
market label); ``candidate_returns`` may be standalone or the combined book,
declared via ``comparison_kind``.  Both series share one capital basis and
one clock: equal length is a structural contract (``ValueError``), interior
NaN/Inf is fail-closed INVALID_EVIDENCE (§1.3.2, §5.3 — no ninth state).
Book tail weights are computed once and shared across all candidate columns
(§7.4 backend note); ties split the boundary mass equally, so date
permutation never changes results (§6.6).

Backends
--------
``cpu_reference`` (default; readable FP64 oracle), ``cpu_fast`` (opt-in
vectorized), ``cuda`` (opt-in lazy CuPy).  No fastmath (§1.1).

Recommended backend — measured on qs-server-c (AMD EPYC 9K84, NVIDIA L20,
numpy 2.2.6 / cupy 14.2, OPENBLAS_NUM_THREADS=1; full data in
``qe_opt/ext_c_work/bench_m03_m04.json``).  All three kernels: CPU_FAST at
every tested shape (T=1250/5000 x F=1/10/50), e2e host-input.  CUDA e2e
never wins because the ~0.5 ms H2D floor dominates (e.g. (5000,50)
book_conditional_loss: CPU_FAST 2.05 ms vs CUDA 2.89 ms; (5000,50) beta:
3.54 ms vs 4.75 ms).  Device-resident CUDA is faster still (~0.3-0.7 ms)
and becomes the right choice only when the book + candidate panel already
live on the device.  No crossover in the tested shapes.
"""

from __future__ import annotations

import math

import numpy as np

from quant_evaluator.contracts.evidence_status import EvidenceStatus
from quant_evaluator.metrics.extension._tail import (
    as_real_f64,
    check_confidence,
    check_min_tail_mass,
    resolved_tail_mass,
    tail_weights_1d,
)
from quant_evaluator.metrics.extension.path_risk import (
    _check_backend,
    _check_min_periods,
    _evidence,
    _MIN_PERIODS_DEFAULT,
    _MIN_TAIL_MASS_DEFAULT,
)

_COMPARISON_KINDS = ("standalone_against_book", "combined_book_against_book")

_EPS = np.finfo(float).eps


# ---------------------------------------------------------------------------
# input preparation
# ---------------------------------------------------------------------------

def _prepare_book_pair(baseline_returns, candidate_returns):
    """Structural validation + family-unit strict window (§6.1 rule 4).

    Returns ``(baseline_w, candidate_w, squeezed, code)``.  Equal length is a
    structural contract; interior non-finite values, ``r < -1`` and
    ``r == -1`` fail closed through the shared path-risk codes.
    """
    base = as_real_f64(baseline_returns, "baseline_returns", ndim=1)
    if base.shape[0] == 0:
        raise ValueError("baseline_returns must contain at least one observation")
    cand = as_real_f64(candidate_returns, "candidate_returns")
    if cand.ndim == 1:
        cand = cand[:, None]
        squeezed = True
    elif cand.ndim == 2:
        squeezed = False
    else:
        raise ValueError("candidate_returns must be 1-D or 2-D")
    if cand.shape[0] != base.shape[0]:
        raise ValueError(
            "baseline_returns and candidate_returns must share the same clock "
            f"({base.shape[0]} != {cand.shape[0]} observations)"
        )
    joint = np.concatenate([base[:, None], cand], axis=1)
    finite = np.isfinite(joint)
    any_finite_row = finite.any(axis=1)
    if not bool(any_finite_row.any()):
        return base, cand, squeezed, "no_finite_observations"
    first = int(np.argmax(any_finite_row))
    end = int(joint.shape[0] - np.argmax(any_finite_row[::-1]))  # exclusive
    window = joint[first:end]
    if not bool(np.isfinite(window).all()):
        return base, cand, squeezed, "interior_missing_or_nonfinite_returns"
    if bool(np.any(window < -1.0)):
        return base, cand, squeezed, "return_basis_mismatch"
    if bool(np.any(window == -1.0)):
        return base, cand, squeezed, "terminal_total_loss"
    return (
        window[:, 0],
        window[:, 1:],
        squeezed,
        None,
    )


def _invalid_or_insufficient(code):
    if code == "no_finite_observations":
        return _evidence(
            EvidenceStatus.INSUFFICIENT_DATA,
            reason_detail="no_finite_observations",
        )
    diagnostics = {"reason_detail": code}
    if code == "terminal_total_loss":
        diagnostics["terminal_total_loss"] = True
        diagnostics["known_drawdown_lower_bound"] = 1.0
    return _evidence(EvidenceStatus.INVALID_EVIDENCE, **diagnostics)


def _check_comparison_kind(comparison_kind):
    if comparison_kind not in _COMPARISON_KINDS:
        raise ValueError(
            f"comparison_kind must be one of {_COMPARISON_KINDS}, "
            f"got {comparison_kind!r}"
        )
    return comparison_kind


def _book_tail_weights(baseline, confidence, backend):
    """One §6.6 boundary-weight vector for the book loss tail ``-r0``.

    The book tail is sorted/weighted exactly once and shared by every
    candidate column and every kernel in this module (§7.4 backend note).
    Returns ``(weights, tail_mass)``; weights are host float64 for CPU
    backends and device float64 for ``cuda``.
    """
    loss = -baseline
    if backend == "cuda":
        import cupy as cp

        loss_dev = cp.asarray(loss)
        n = loss_dev.shape[0]
        mass = resolved_tail_mass(n, confidence)
        index = min(n - 1, math.ceil(mass) - 1)
        cutoff = cp.sort(loss_dev)[::-1][index]
        above = loss_dev > cutoff
        tied = loss_dev == cutoff
        w = above.astype(cp.float64)
        n_above = int(cp.asnumpy(above.sum()))
        n_tied = int(cp.asnumpy(tied.sum()))
        if n_tied:
            w = cp.where(tied, (mass - n_above) / n_tied, w)
        return w, mass
    return tail_weights_1d(loss, confidence), resolved_tail_mass(loss.size, confidence)


def _is_device(w):
    """True for cupy arrays; numpy scalars also expose ``.device`` ('cpu'),
    so the ndarray type check comes first."""
    return not isinstance(w, np.ndarray) and hasattr(w, "device")


def _scalar_sum(w):
    return float(w.sum()) if not _is_device(w) else float(w.sum().get())


def _count_nonzero(w):
    if _is_device(w):
        import cupy as cp

        return int(cp.asnumpy(cp.count_nonzero(w)))
    return int(np.count_nonzero(w))


def _weighted_dot(w, columns):
    """``w^T columns`` for host or device weights (§7.4: w^T R matmul)."""
    if _is_device(w):
        import cupy as cp

        columns_dev = cp.asarray(columns)
        return cp.asnumpy(w @ columns_dev)
    return w @ columns


# ---------------------------------------------------------------------------
# shared evidence gates
# ---------------------------------------------------------------------------

def _book_common_checks(baseline, candidate, code, *, tail_confidence, min_tail_mass):
    if code is not None:
        return None, _invalid_or_insufficient(code)
    mass = resolved_tail_mass(baseline.shape[0], tail_confidence)
    if mass < min_tail_mass:
        return None, _evidence(
            EvidenceStatus.INSUFFICIENT_DATA,
            reason_detail="tail_mass_below_min",
            tail_mass=mass,
            min_tail_mass=min_tail_mass,
        )
    return mass, None


def _tail_result(value, squeezed, *, mass, comparison_kind, n_observations, extra=None):
    if squeezed:
        value = float(np.asarray(value).reshape(-1)[0])
    diagnostics = {
        "reason_detail": "ok",
        "tail_mass": mass,
        "comparison_kind": comparison_kind,
        "n_observations": int(n_observations),
    }
    if comparison_kind == "standalone_against_book":
        diagnostics["interpretation"] = (
            "standalone candidate vs book tail: complementarity only, "
            "not an add-on improvement claim"
        )
    if extra:
        diagnostics.update(extra)
    return _evidence(EvidenceStatus.COMPUTED, value, **diagnostics)


# ---------------------------------------------------------------------------
# public kernels
# ---------------------------------------------------------------------------

def book_conditional_loss(
    baseline_returns,
    candidate_returns,
    tail_confidence=0.95,
    min_tail_mass=_MIN_TAIL_MASS_DEFAULT,
    *,
    comparison_kind="standalone_against_book",
    min_periods=_MIN_PERIODS_DEFAULT,
    backend="cpu_reference",
):
    """Candidate expected return on the book's worst ``1 - q`` loss tail.

    ``-sum(w * r1) / sum(w)`` with §6.6 boundary weights from the book loss
    tail; negative values mean the candidate pays off when the book bleeds.
    """
    _check_backend(backend)
    _check_comparison_kind(comparison_kind)
    confidence = check_confidence(tail_confidence)
    mass_floor = check_min_tail_mass(min_tail_mass)
    min_periods = _check_min_periods(min_periods)
    baseline, candidate, squeezed, code = _prepare_book_pair(
        baseline_returns, candidate_returns
    )
    mass, failure = _book_common_checks(
        baseline, candidate, code,
        tail_confidence=confidence, min_tail_mass=mass_floor,
    )
    if failure is not None:
        return failure
    if baseline.shape[0] < min_periods:
        return _evidence(
            EvidenceStatus.INSUFFICIENT_DATA,
            reason_detail="observations_too_few",
            n_observations=int(baseline.shape[0]),
            min_periods=int(min_periods),
        )

    w, mass = _book_tail_weights(baseline, confidence, backend)
    weight_sum = _scalar_sum(w)
    values = -_weighted_dot(w, candidate) / weight_sum
    if backend != "cuda" and values.ndim == 0:
        values = float(values)
    return _tail_result(
        values,
        squeezed,
        mass=mass,
        comparison_kind=comparison_kind,
        n_observations=baseline.shape[0],
        extra={"n_book_tail_dates": _count_nonzero(w)},
    )


def book_tail_loss_delta(
    baseline_returns,
    candidate_returns,
    tail_confidence=0.95,
    min_tail_mass=_MIN_TAIL_MASS_DEFAULT,
    *,
    comparison_kind="standalone_against_book",
    min_periods=_MIN_PERIODS_DEFAULT,
    backend="cpu_reference",
):
    """Book tail loss minus candidate tail loss on the **fixed book tail**.

    ``[-sum(w*r0)/sum(w)] - [-sum(w*r1)/sum(w)]`` with one shared §6.6
    weight vector; positive means improvement.  Never two independently
    selected tails (§7.4).
    """
    _check_backend(backend)
    _check_comparison_kind(comparison_kind)
    confidence = check_confidence(tail_confidence)
    mass_floor = check_min_tail_mass(min_tail_mass)
    min_periods = _check_min_periods(min_periods)
    baseline, candidate, squeezed, code = _prepare_book_pair(
        baseline_returns, candidate_returns
    )
    mass, failure = _book_common_checks(
        baseline, candidate, code,
        tail_confidence=confidence, min_tail_mass=mass_floor,
    )
    if failure is not None:
        return failure
    if baseline.shape[0] < min_periods:
        return _evidence(
            EvidenceStatus.INSUFFICIENT_DATA,
            reason_detail="observations_too_few",
            n_observations=int(baseline.shape[0]),
            min_periods=int(min_periods),
        )

    w, mass = _book_tail_weights(baseline, confidence, backend)
    weight_sum = _scalar_sum(w)
    book_tail = -_weighted_dot(w, baseline) / weight_sum
    cand_tail = -_weighted_dot(w, candidate) / weight_sum
    values = book_tail - cand_tail
    if backend != "cuda" and values.ndim == 0:
        values = float(values)
    return _tail_result(
        values,
        squeezed,
        mass=mass,
        comparison_kind=comparison_kind,
        n_observations=baseline.shape[0],
        extra={
            "book_tail_loss": float(np.asarray(book_tail).reshape(-1)[0]),
            "n_book_tail_dates": _count_nonzero(w),
        },
    )


def book_downside_beta(
    baseline_returns,
    candidate_returns,
    tail_confidence=0.95,
    min_tail_mass=_MIN_TAIL_MASS_DEFAULT,
    *,
    comparison_kind="standalone_against_book",
    min_periods=_MIN_PERIODS_DEFAULT,
    backend="cpu_reference",
):
    """Tail-weighted downside beta of the candidate against the book.

    ``sum(w*(r0-mu0w)*(r1-mu1w)) / sum(w*(r0-mu0w)^2)`` with tail-weighted
    means (§7.4).  A degenerate baseline weighted variance fails closed
    instead of returning an infinite beta; no "annualized beta" is produced.
    """
    _check_backend(backend)
    _check_comparison_kind(comparison_kind)
    confidence = check_confidence(tail_confidence)
    mass_floor = check_min_tail_mass(min_tail_mass)
    min_periods = _check_min_periods(min_periods)
    baseline, candidate, squeezed, code = _prepare_book_pair(
        baseline_returns, candidate_returns
    )
    mass, failure = _book_common_checks(
        baseline, candidate, code,
        tail_confidence=confidence, min_tail_mass=mass_floor,
    )
    if failure is not None:
        return failure
    if baseline.shape[0] < min_periods:
        return _evidence(
            EvidenceStatus.INSUFFICIENT_DATA,
            reason_detail="observations_too_few",
            n_observations=int(baseline.shape[0]),
            min_periods=int(min_periods),
        )

    w, mass = _book_tail_weights(baseline, confidence, backend)
    is_device = _is_device(w)
    weight_sum = float(w.sum()) if not is_device else float(w.sum().get())
    baseline2 = baseline[:, None]
    mu0 = (_weighted_dot(w, baseline2) / weight_sum).reshape(-1)  # (F,)
    mu1 = (_weighted_dot(w, candidate) / weight_sum).reshape(-1)  # (F,)
    dev0 = baseline2 - mu0[None, :]   # (T, F)
    dev1 = candidate - mu1[None, :]   # (T, F)
    if is_device:
        import cupy as cp

        w_dev = w
        dev0_dev = cp.asarray(dev0)
        dev1_dev = cp.asarray(dev1)
        cov = (w_dev[:, None] * dev0_dev * dev1_dev).sum(axis=0)
        var0 = (w_dev[:, None] * dev0_dev * dev0_dev).sum(axis=0)
        cov = cp.asnumpy(cov)
        var0 = cp.asnumpy(var0)
    else:
        cov = (w[:, None] * dev0 * dev1).sum(axis=0)
        var0 = (w[:, None] * dev0 * dev0).sum(axis=0)
    scale = max(float(np.max(np.abs(var0))), _EPS)
    degenerate = np.broadcast_to(var0 <= 64.0 * _EPS * scale, cov.shape)
    values = np.full_like(cov, np.nan, dtype=np.float64)
    np.divide(cov, var0, out=values, where=~degenerate)
    if squeezed:
        value = float(values[0])
        if degenerate[0]:
            return _evidence(
                EvidenceStatus.INVALID_EVIDENCE,
                reason_detail="baseline_weighted_variance_degenerate",
                tail_mass=mass,
                comparison_kind=comparison_kind,
            )
    else:
        value = values
        if bool(degenerate.any()):
            return _evidence(
                EvidenceStatus.INVALID_EVIDENCE,
                reason_detail="baseline_weighted_variance_degenerate",
                tail_mass=mass,
                comparison_kind=comparison_kind,
                degenerate_columns=np.flatnonzero(degenerate).tolist(),
            )
    return _tail_result(
        value,
        squeezed,
        mass=mass,
        comparison_kind=comparison_kind,
        n_observations=baseline.shape[0],
        extra={"baseline_weighted_variance": float(np.asarray(var0).reshape(-1)[0]) if squeezed else None},
    )
