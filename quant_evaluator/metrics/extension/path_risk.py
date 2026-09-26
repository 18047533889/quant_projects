"""M03 path-risk extension kernels (QE-EXT-SPEC-1.0 §7.3).

Kernels
-------
- ``cdar``: conditional drawdown-at-risk — §6.6 upper-tail mean of the daily
  fractional drawdown path ``D_t = 1 - V_t / running_max(V, V0=1)``.  The
  drawdown path is reused verbatim from the repo's drawdown authority
  ``metrics/risk/drawdown_analysis.compute_drawdown_series``
  (``D_t = -drawdown_series``); the §7.3 log form
  ``1 - exp(logV_t - running_max(logV, 0))`` is the same mathematical
  quantity without the log/exp round trip, and the arithmetic form is the
  repo-wide drawdown series.  A path that never draws down yields CDaR = 0
  (COMPUTED, not missing) once the tail-mass gate is met.
- ``ced``: conditional expected drawdown — block-resampled scenario MDD
  distribution (§6.3 PCG64 plan, §6.5 ordered log-wealth path summaries)
  collapsed with the §6.6 upper-tail mean.  ``summary_mdd = -expm1(-d)``.
- ``drawdown_budget_exceedance``: strict scenario frequency
  ``mean(MDD_b > budget)`` from the same §6.5 distribution; 0 < budget < 1
  is required.  The output is an estimated scenario frequency, never a
  claimed true future default probability.
- ``joint_drawdown_occupancy``: paired observed-path co-occupancy
  ``mean((D0 > d0) & (D1 > d1))`` with explicit thresholds (default 0) plus
  each path's underwater fraction so low co-occupancy cannot be read as low
  risk when one path is simply never underwater.

Evidence policy (§1.3.2, §6.1, §5.3 eight states only)
------------------------------------------------------
New path metrics require a complete common window: leading/trailing
all-missing rows are trimmed jointly; any interior NaN/Inf, a non-strictly
increasing calendar, ``r == -1`` (terminal total loss: diagnostics
``terminal_total_loss=True, known_drawdown_lower_bound=1.0``) and ``r < -1``
(diagnostics ``reason_detail="return_basis_mismatch"``) are INVALID_EVIDENCE,
never silently compressed or zero-filled.  Short samples
(T < ``min_periods``) and tail mass below ``min_tail_mass`` are
INSUFFICIENT_DATA.

Backends
--------
``cpu_reference`` (default; readable FP64 oracle, bit-deterministic),
``cpu_fast`` (opt-in vectorized), ``cuda`` (opt-in, lazy CuPy import).
No ``fastmath`` anywhere (§1.1).  The GPU path validates on the host before
any H2D transfer (control-plane validation, CUDA_STRICT friendly).

Recommended backend — measured on qs-server-c (AMD EPYC 9K84, NVIDIA L20,
numpy 2.2.6 / cupy 14.2, OPENBLAS_NUM_THREADS=1; B=4999, block=10; full
data in ``qe_opt/ext_c_work/bench_m03_m04.json``).  Default remains
``cpu_reference`` (bit-deterministic oracle); fast paths are opt-in:
- cdar: CPU_FAST up to T*F ~ 3e4 (e.g. (1250,10) 0.59 ms CPU vs 0.99 ms
  CUDA; (5000,1) 0.22 ms vs 0.96 ms); CUDA beyond (e2e host-input:
  (5000,10) 1.13 ms vs 2.19 ms; (5000,50) 1.95 ms vs 15.5 ms).  Device-
  resident CUDA is flat ~0.75-1.0 ms at every shape.
- ced: CUDA for F >= 10 at every T (e2e: (1250,10) 4.6 ms vs 8.0 ms CPU_FAST;
  (5000,50) 19.0 ms vs 68.6 ms; H=252 probe 16.0 ms vs 18.3 ms); CPU_FAST
  for F == 1 (0.85 ms vs 3.1 ms).  Crossover between F=1 and F=10 tiles.
- drawdown_budget_exceedance: same pipeline/crossover as ced (e.g. (5000,50)
  CUDA 17.4 ms vs CPU_FAST 62.3 ms vs CPU_REFERENCE 8086 ms).
- joint_drawdown_occupancy: CPU_FAST at all shapes except the widest tile
  ((5000,50): CUDA 6.8 ms vs CPU 12.4 ms); two cumulative running-max passes
  transfer poorly below ~2e5 elements.
CPU_REFERENCE is 90x-8000x slower than CPU_FAST/CUDA for ced/budget and is
kept strictly as the correctness oracle.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from quant_evaluator.contracts.evidence_status import EvidenceStatus
from quant_evaluator.metrics.extension._tail import (
    as_real_f64,
    check_confidence,
    check_min_tail_mass,
    resolved_tail_mass,
    tail_weights_1d,
    upper_tail_mean_1d,
)

_BACKENDS = ("cpu_reference", "cpu_fast", "cuda")

_MIN_PERIODS_DEFAULT = 60      # §6.1 SCREEN default
_MIN_TAIL_MASS_DEFAULT = 10.0  # §6.1 SCREEN default (observation-equivalent)
_CED_DEFAULTS = dict(path_horizon=63, block_length=10, repetitions=4999, seed=0)


# ---------------------------------------------------------------------------
# evidence plumbing
# ---------------------------------------------------------------------------

def _evidence(status, value=None, **diagnostics):
    return {"status": status, "value": value, "diagnostics": diagnostics}


def _check_backend(backend):
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
    return backend


def _check_min_periods(min_periods):
    if isinstance(min_periods, (bool, np.bool_)) or not isinstance(
        min_periods, (int, np.integer)
    ):
        raise TypeError("min_periods must be an integer, not bool")
    if min_periods < 1:
        raise ValueError("min_periods must be a positive integer")
    return int(min_periods)


def _check_threshold(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# input preparation (§6.1 strict common-window policy)
# ---------------------------------------------------------------------------

def _check_calendar(calendar, n_rows):
    """Return an evidence diagnostics code, or None when usable.

    The calendar must be a 1-D strictly increasing sequence of the same
    length as the returns axis.  Duplicated or decreasing stamps mean the
    trading grid itself is broken -> INVALID_EVIDENCE (§1.3.2).  Gaps that
    keep the grid strictly increasing cannot be detected without a reference
    grid; callers must pass the true exchange calendar.
    """
    if calendar is None:
        return None
    cal = np.asarray(calendar)
    if cal.ndim != 1 or cal.shape[0] != n_rows:
        raise ValueError(
            f"calendar must be 1-D with {n_rows} stamps matching the returns axis, "
            f"got shape {cal.shape}"
        )
    if cal.shape[0] < 2:
        return None
    try:
        strictly_increasing = bool(np.all(cal[1:] > cal[:-1]))
    except TypeError as exc:
        raise TypeError(f"calendar stamps must be orderable: {exc}") from exc
    return None if strictly_increasing else "calendar_not_strictly_increasing"


def _prepare_return_panel(returns, *, calendar=None, name="returns"):
    """Validate dtype/shape and enforce the strict common window.

    Returns ``(panel, squeezed, code)`` where ``panel`` is the (T, F) float64
    window with leading/trailing all-missing rows trimmed jointly and
    ``code`` is one of:

    ``None``
        usable window;
    ``"calendar_not_strictly_increasing"``
        broken trading grid (checked before any trimming);
    ``"no_finite_observations"``
        every row missing;
    ``"interior_missing_or_nonfinite_returns"``
        NaN/Inf inside the common window (never compressed, §1.3.2);
    ``"return_basis_mismatch"``
        some finite ``r < -1`` (§6.1 rule 5);
    ``"terminal_total_loss"``
        some finite ``r == -1`` (§6.1 rule 5).
    """
    r = as_real_f64(returns, name)
    if r.ndim == 1:
        panel = r[:, None]
        squeezed = True
    elif r.ndim == 2:
        panel = r
        squeezed = False
    else:
        raise ValueError(f"{name} must be 1-D or 2-D, got {r.ndim}-D")
    if panel.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one observation")

    code = _check_calendar(calendar, panel.shape[0])
    if code is not None:
        return panel, squeezed, code

    finite = np.isfinite(panel)
    any_finite_row = finite.any(axis=1)
    if not bool(any_finite_row.any()):
        return panel, squeezed, "no_finite_observations"
    first = int(np.argmax(any_finite_row))
    end = int(panel.shape[0] - np.argmax(any_finite_row[::-1]))  # exclusive
    window = panel[first:end]
    if not bool(np.isfinite(window).all()):
        return panel, squeezed, "interior_missing_or_nonfinite_returns"
    if bool(np.any(window < -1.0)):
        return window, squeezed, "return_basis_mismatch"
    if bool(np.any(window == -1.0)):
        return window, squeezed, "terminal_total_loss"
    return window, squeezed, None


def _insufficient_or_invalid(code, extra=None):
    if code == "no_finite_observations":
        return _evidence(
            EvidenceStatus.INSUFFICIENT_DATA,
            reason_detail="no_finite_observations",
        )
    if code in ("return_basis_mismatch", "terminal_total_loss"):
        diagnostics = {"reason_detail": code}
        if code == "terminal_total_loss":
            diagnostics["terminal_total_loss"] = True
            diagnostics["known_drawdown_lower_bound"] = 1.0
        if extra:
            diagnostics.update(extra)
        return _evidence(EvidenceStatus.INVALID_EVIDENCE, **diagnostics)
    diagnostics = {"reason_detail": code}
    if extra:
        diagnostics.update(extra)
    return _evidence(EvidenceStatus.INVALID_EVIDENCE, **diagnostics)


def _positive_drawdown_path(panel, backend):
    """D = -drawdown_series from the repo drawdown authority (always >= 0).

    Zero wealth is an absorbing observed 100% loss (D == 1 from the wipeout
    onward); returns < -1 were already rejected upstream.  Returns a float64
    numpy array for CPU backends and a device-resident cupy array for
    ``cuda``.
    """
    if backend == "cuda":
        import cupy as cp

        from quant_evaluator.kernels.gpu.drawdown import _running_max

        ret_dev = cp.asarray(panel)
        wealth = cp.cumprod(1.0 + ret_dev, axis=0)
        running_max = cp.maximum(1.0, _running_max(wealth, axis=0))
        return (running_max - wealth) / running_max
    from quant_evaluator.metrics.risk.drawdown_analysis import compute_drawdown_series

    drawdown_series, _wealth, _running = compute_drawdown_series(panel)
    return -drawdown_series + 0.0  # normalize -0.0 at peaks to +0.0


# ---------------------------------------------------------------------------
# §6.5 / §12 ordered log-wealth path summaries
# ---------------------------------------------------------------------------

class PathSummary(NamedTuple):
    """Ordered log-wealth summary ``(total, high, low, log_drawdown)``.

    ``high``/``low`` are max/min prefix log-wealth including the V0=1 start
    (log-wealth 0); ``log_drawdown`` is the largest peak-to-trough log
    drop.  The empty-segment identity is ``(0, 0, 0, 0)``.  The merge is
    order-preserving, associative and non-commutative (§6.5).
    """

    total: float
    high: float
    low: float
    log_drawdown: float


def path_summary(returns) -> PathSummary:
    """§6.5/§12 summary of one return slice (reference oracle)."""
    r = as_real_f64(returns, "returns", ndim=1)
    if r.size == 0:
        return PathSummary(0.0, 0.0, 0.0, 0.0)
    if bool(np.any(r <= -1.0)):
        raise ValueError("Log-path scope requires r > -1; total loss is separately flagged")
    p = np.concatenate(([0.0], np.cumsum(np.log1p(r))))
    return PathSummary(
        float(p[-1]),
        float(p.max()),
        float(p.min()),
        float(np.max(np.maximum.accumulate(p) - p)),
    )


def merge_path(a: PathSummary, b: PathSummary) -> PathSummary:
    """§6.5 order-preserving merge: cross-block peak/trough handled by the
    ``high_A - total_A - low_B`` term."""
    return PathSummary(
        a.total + b.total,
        max(a.high, a.total + b.high),
        min(a.low, a.total + b.low),
        max(a.log_drawdown, b.log_drawdown, a.high - a.total - b.low),
    )


def summary_mdd(summary: PathSummary) -> float:
    """Percentage MDD of a summary: ``-expm1(-log_drawdown)``."""
    return float(-np.expm1(-summary.log_drawdown))


def block_plan(sample_length, output_length, block_length, repetitions, seed=0):
    """§6.3/§12 moving-block plan (verbatim semantics, PCG64, seed 0 default).

    Returns ``(starts (B, J) int64, lengths (J,) int64)``; the last block is
    the short remainder when ``output_length`` is not a multiple of
    ``block_length``.  The plan depends only on
    ``(sample_length, output_length, block_length, repetitions, seed)`` —
    never on factor count or tile width.
    """
    def _positive_int(value, name):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ) or value < 1:
            raise ValueError(f"{name} must be a positive integer, not bool")
        return int(value)

    n = _positive_int(sample_length, "sample_length")
    h = _positive_int(output_length, "output_length")
    block = _positive_int(block_length, "block_length")
    b = _positive_int(repetitions, "repetitions")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer, not bool")
    if b < 2 or block > n:
        raise ValueError("Invalid block plan")
    count = (h + block - 1) // block
    lengths = np.full(count, block, dtype=np.int64)
    lengths[-1] = h - (count - 1) * block
    rng = np.random.Generator(np.random.PCG64(seed))
    starts = rng.integers(0, n - block + 1, size=(b, count), dtype=np.int64)
    return starts, lengths


def _window_summaries(g, length):
    """Per-start fixed-length ordered summaries, vectorized (§6.5 plan 1).

    ``g`` is the (T, F) log1p panel.  Returns ``(total, high, low, d)``,
    each (T-length+1, F).  The global-prefix accumulation rounds slightly
    differently from per-slice ``np.cumsum``; agreement with the §12 oracle
    is pinned at 1e-12 in the tests (plan §7.3 tolerance).
    """
    t, f = g.shape
    s_count = t - length + 1
    prefix = np.zeros((t + 1, f))
    np.cumsum(g, axis=0, out=prefix[1:])
    base = prefix[:s_count]
    total = prefix[length:] - base
    run_max = base.copy()
    run_min = base.copy()
    d = np.zeros((s_count, f))
    for k in range(1, length + 1):
        v = prefix[k:k + s_count]
        np.maximum(run_max, v, out=run_max)
        np.minimum(run_min, v, out=run_min)
        np.maximum(d, run_max - v, out=d)
    high = run_max - base
    low = run_min - base
    return total, high, low, d


def _sampled_mdd_reference(panel, starts, lengths):
    """§12 oracle: per-column cached summaries merged in draw order."""
    b, j = starts.shape
    f = panel.shape[1]
    out = np.empty((b, f))
    for col in range(f):
        series = panel[:, col]
        cache = {}
        for scenario in range(b):
            row = starts[scenario]
            state = PathSummary(0.0, 0.0, 0.0, 0.0)
            for step in range(j):
                key = (int(row[step]), int(lengths[step]))
                summary = cache.get(key)
                if summary is None:
                    start, length = key
                    summary = path_summary(series[start:start + length])
                    cache[key] = summary
                state = merge_path(state, summary)
            out[scenario, col] = summary_mdd(state)
    return out


def _sampled_mdd_fast(panel, starts, lengths):
    """Vectorized prefix-window summaries merged in draw order (§6.5 plan 1)."""
    g = np.log1p(panel)
    unique_lengths = np.unique(lengths)
    cache = {
        int(length): _window_summaries(g, int(length))
        for length in unique_lengths
    }
    b, j = starts.shape
    f = panel.shape[1]
    total = np.zeros((b, f))
    high = np.zeros((b, f))
    low = np.zeros((b, f))
    d = np.zeros((b, f))
    for step in range(j):
        length = int(lengths[step])
        block_start = starts[:, step]
        t_b, h_b, l_b, d_b = (arr[block_start] for arr in cache[length])
        new_d = np.maximum(np.maximum(d, d_b), high - total - l_b)
        new_high = np.maximum(high, total + h_b)
        new_low = np.minimum(low, total + l_b)
        new_total = total + t_b
        total, high, low, d = new_total, new_high, new_low, new_d
    return -np.expm1(-d)


def _sampled_mdd_cuda(panel_dev, starts, lengths):
    """GPU direct-path scenario MDD (short-H baseline, §7.3).

    Chunked over scenarios so the (Bc, H+1, F) log-wealth tensor stays
    bounded.  Path includes the V0=1 start row (log-wealth 0), so the
    drawdown covers the initial-capital high-water mark.
    """
    import cupy as cp

    from quant_evaluator.kernels.gpu.drawdown import _running_max

    b, j = starts.shape
    f = panel_dev.shape[1]
    horizon = int(lengths.sum())
    lengths_host = np.asarray(lengths, dtype=np.int64)
    j_of_pos = np.repeat(np.arange(j), lengths_host)
    off_of_pos = np.concatenate(
        [np.arange(int(length)) for length in lengths_host]
    ) if j else np.zeros(0, dtype=np.int64)
    gather = np.asarray(starts)[:, j_of_pos] + off_of_pos[None, :]
    gather_dev = cp.asarray(gather)
    target_bytes = 256 * 1024 * 1024
    chunk = max(1, int(target_bytes // ((horizon + 1) * f * 8)))
    mdd = cp.empty((b, f))
    for lo in range(0, b, chunk):
        block = panel_dev[gather_dev[lo:lo + chunk]]
        scenario_b = block.shape[0]
        log_wealth = cp.empty((scenario_b, horizon + 1, f))
        log_wealth[:, 0] = 0.0
        cp.cumsum(cp.log1p(block), axis=1, out=log_wealth[:, 1:])
        running_max = cp.maximum(0.0, _running_max(log_wealth, axis=1))
        d = (running_max - log_wealth).max(axis=1)
        mdd[lo:lo + chunk] = -cp.expm1(-d)
    return mdd


def _upper_tail_mean_cuda(values_dev, confidence):
    """§6.6 upper-tail mean of a device (n, F) matrix -> (F,) cupy array."""
    import cupy as cp

    n = values_dev.shape[0]
    mass = resolved_tail_mass(n, confidence)
    index = min(n - 1, math.ceil(mass) - 1)
    cutoff = cp.sort(values_dev, axis=0)[::-1][index]
    above = values_dev > cutoff
    tied = values_dev == cutoff
    frac = (mass - above.sum(axis=0)) / tied.sum(axis=0)
    w = above.astype(cp.float64) + tied * frac
    return (w * values_dev).sum(axis=0) / w.sum(axis=0)


# ---------------------------------------------------------------------------
# scenario MDD distribution shared by ced / drawdown_budget_exceedance
# ---------------------------------------------------------------------------

def _scenario_mdd_distribution(
    panel,
    *,
    path_horizon,
    block_length,
    repetitions,
    seed,
    backend,
):
    starts, lengths = block_plan(
        panel.shape[0], path_horizon, block_length, repetitions, seed
    )
    if backend == "cuda":
        import cupy as cp

        panel_dev = cp.asarray(panel)
        mdd_dev = _sampled_mdd_cuda(panel_dev, starts, lengths)
        return starts, lengths, mdd_dev
    if backend == "cpu_fast":
        return starts, lengths, _sampled_mdd_fast(panel, starts, lengths)
    return starts, lengths, _sampled_mdd_reference(panel, starts, lengths)


def _ced_common_checks(
    panel,
    code,
    *,
    tail_confidence,
    min_tail_mass,
    min_periods,
    repetitions,
):
    """Shared evidence gates for the resampled-drawdown metrics."""
    if code is not None:
        return _insufficient_or_invalid(code)
    if panel.shape[0] < min_periods:
        return _evidence(
            EvidenceStatus.INSUFFICIENT_DATA,
            reason_detail="observations_too_few",
            n_observations=int(panel.shape[0]),
            min_periods=int(min_periods),
        )
    mass = resolved_tail_mass(repetitions, tail_confidence)
    if mass < min_tail_mass:
        return _evidence(
            EvidenceStatus.INSUFFICIENT_DATA,
            reason_detail="tail_mass_below_min",
            tail_mass=mass,
            min_tail_mass=min_tail_mass,
        )
    return None


# ---------------------------------------------------------------------------
# public kernels
# ---------------------------------------------------------------------------

def cdar(
    returns,
    tail_confidence=0.95,
    min_tail_mass=_MIN_TAIL_MASS_DEFAULT,
    *,
    calendar=None,
    min_periods=_MIN_PERIODS_DEFAULT,
    backend="cpu_reference",
):
    """Conditional drawdown-at-risk: ``UTM_q(D)`` over the observed path.

    ``D_t = 1 - V_t / running_max(V incl. V0=1)`` is the repo drawdown
    series negated (§7.3); the result is the §6.6 upper-tail mean of that
    path with the tail-mass gate ``m = (1-q)*T >= min_tail_mass``.
    A path that never draws down returns 0.0 once the gate is met.

    Returns ``{"status": EvidenceStatus, "value": float | (F,) ndarray,
    "diagnostics": dict}``.
    """
    _check_backend(backend)
    confidence = check_confidence(tail_confidence)
    mass_floor = check_min_tail_mass(min_tail_mass)
    min_periods = _check_min_periods(min_periods)
    panel, squeezed, code = _prepare_return_panel(returns, calendar=calendar)
    if code is not None:
        return _insufficient_or_invalid(code)
    if panel.shape[0] < min_periods:
        return _evidence(
            EvidenceStatus.INSUFFICIENT_DATA,
            reason_detail="observations_too_few",
            n_observations=int(panel.shape[0]),
            min_periods=int(min_periods),
        )
    mass = resolved_tail_mass(panel.shape[0], confidence)
    if mass < mass_floor:
        return _evidence(
            EvidenceStatus.INSUFFICIENT_DATA,
            reason_detail="tail_mass_below_min",
            tail_mass=mass,
            min_tail_mass=mass_floor,
        )

    d_path = _positive_drawdown_path(panel, backend)
    if backend == "cpu_reference":
        value = np.array(
            [upper_tail_mean_1d(d_path[:, col], confidence) for col in range(d_path.shape[1])]
        )
    elif backend == "cpu_fast":
        from quant_evaluator.metrics.extension._tail import upper_tail_mean_batch

        value = upper_tail_mean_batch(d_path, confidence)
    else:
        value_dev = _upper_tail_mean_cuda(d_path, confidence)
        import cupy as cp

        value = cp.asnumpy(value_dev)
        d_path = cp.asnumpy(d_path)
    if squeezed:
        value = float(value[0])
    return _evidence(
        EvidenceStatus.COMPUTED,
        value,
        reason_detail="ok",
        n_observations=int(panel.shape[0]),
        tail_mass=mass,
        max_drawdown_observed=float(d_path.max()),
        never_underwater=bool(d_path.max() == 0.0),
    )


def ced(
    returns,
    path_horizon=_CED_DEFAULTS["path_horizon"],
    block_length=_CED_DEFAULTS["block_length"],
    repetitions=_CED_DEFAULTS["repetitions"],
    tail_confidence=0.95,
    min_tail_mass=_MIN_TAIL_MASS_DEFAULT,
    seed=_CED_DEFAULTS["seed"],
    *,
    calendar=None,
    min_periods=_MIN_PERIODS_DEFAULT,
    backend="cpu_reference",
):
    """Conditional expected drawdown over block-resampled horizon paths.

    ``UTM_q(MDD_b)`` where each ``MDD_b`` is the §6.5 exact ordered-summary
    MDD of one moving-block resample of length ``path_horizon`` drawn with
    the §6.3 PCG64 plan (§7.3).  Source length must satisfy
    ``T >= max(min_periods, 2*block_length)``; H=63 and H=252 are two
    MetricInstances of this one kernel, not two ids.
    """
    _check_backend(backend)
    for name, value in (
        ("path_horizon", path_horizon),
        ("block_length", block_length),
        ("repetitions", repetitions),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ) or value < 1:
            raise ValueError(f"{name} must be a positive integer, not bool")
    if repetitions < 2:
        raise ValueError("repetitions must be >= 2")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer, not bool")
    confidence = check_confidence(tail_confidence)
    mass_floor = check_min_tail_mass(min_tail_mass)
    min_periods = _check_min_periods(min_periods)

    panel, squeezed, code = _prepare_return_panel(returns, calendar=calendar)
    if panel.shape[0] < 2 * int(block_length):
        insufficient = _evidence(
            EvidenceStatus.INSUFFICIENT_DATA,
            reason_detail="observations_too_few",
            n_observations=int(panel.shape[0]),
            required=max(min_periods, 2 * int(block_length)),
        )
        return insufficient if code is None else _insufficient_or_invalid(code)
    gate = _ced_common_checks(
        panel,
        code,
        tail_confidence=confidence,
        min_tail_mass=mass_floor,
        min_periods=min_periods,
        repetitions=int(repetitions),
    )
    if gate is not None:
        return gate

    starts, lengths, mdd = _scenario_mdd_distribution(
        panel,
        path_horizon=int(path_horizon),
        block_length=int(block_length),
        repetitions=int(repetitions),
        seed=int(seed),
        backend=backend,
    )
    if backend == "cuda":
        value_dev = _upper_tail_mean_cuda(mdd, confidence)
        import cupy as cp

        value = cp.asnumpy(value_dev)
        mdd_host = None  # summary-only: never download the (B, F) matrix
    elif backend == "cpu_fast":
        from quant_evaluator.metrics.extension._tail import upper_tail_mean_batch

        value = upper_tail_mean_batch(mdd, confidence)
    else:
        value = np.array(
            [upper_tail_mean_1d(mdd[:, col], confidence) for col in range(mdd.shape[1])]
        )
    if squeezed:
        value = float(value[0])
    return _evidence(
        EvidenceStatus.COMPUTED,
        value,
        reason_detail="ok",
        n_observations=int(panel.shape[0]),
        n_scenarios=int(repetitions),
        path_horizon=int(path_horizon),
        block_length=int(block_length),
        seed=int(seed),
        tail_mass=resolved_tail_mass(int(repetitions), confidence),
    )


def drawdown_budget_exceedance(
    returns,
    drawdown_budget,
    path_horizon=_CED_DEFAULTS["path_horizon"],
    block_length=_CED_DEFAULTS["block_length"],
    repetitions=_CED_DEFAULTS["repetitions"],
    seed=_CED_DEFAULTS["seed"],
    *,
    calendar=None,
    min_periods=_MIN_PERIODS_DEFAULT,
    backend="cpu_reference",
):
    """Strict scenario frequency ``mean(MDD_b > drawdown_budget)`` (§7.3).

    Shares the §6.5 scenario MDD distribution with :func:`ced`.  The budget
    is mandatory and must satisfy ``0 < budget < 1`` (input contract ->
    ``ValueError``).  The result is an estimated scenario frequency with the
    scenario count reported alongside; it is not a true future default
    probability and the budget must come from risk policy, not test-period
    tuning.
    """
    if isinstance(drawdown_budget, (bool, np.bool_)) or not isinstance(
        drawdown_budget, (int, float, np.integer, np.floating)
    ):
        raise TypeError("drawdown_budget must be a real scalar in (0, 1)")
    budget = float(drawdown_budget)
    if not math.isfinite(budget) or not 0.0 < budget < 1.0:
        raise ValueError(f"drawdown_budget must lie strictly between 0 and 1, got {budget!r}")
    _check_backend(backend)
    for name, value in (
        ("path_horizon", path_horizon),
        ("block_length", block_length),
        ("repetitions", repetitions),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ) or value < 1:
            raise ValueError(f"{name} must be a positive integer, not bool")
    if repetitions < 2:
        raise ValueError("repetitions must be >= 2")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer, not bool")
    min_periods = _check_min_periods(min_periods)

    panel, squeezed, code = _prepare_return_panel(returns, calendar=calendar)
    if panel.shape[0] < 2 * int(block_length):
        if code is not None:
            return _insufficient_or_invalid(code)
        return _evidence(
            EvidenceStatus.INSUFFICIENT_DATA,
            reason_detail="observations_too_few",
            n_observations=int(panel.shape[0]),
            required=max(min_periods, 2 * int(block_length)),
        )
    if code is not None:
        return _insufficient_or_invalid(code)
    if panel.shape[0] < min_periods:
        return _evidence(
            EvidenceStatus.INSUFFICIENT_DATA,
            reason_detail="observations_too_few",
            n_observations=int(panel.shape[0]),
            min_periods=int(min_periods),
        )

    starts, lengths, mdd = _scenario_mdd_distribution(
        panel,
        path_horizon=int(path_horizon),
        block_length=int(block_length),
        repetitions=int(repetitions),
        seed=int(seed),
        backend=backend,
    )
    if backend == "cuda":
        import cupy as cp

        counts = cp.asnumpy((mdd > budget).sum(axis=0))
        count = int(counts.sum())
        value = counts / float(repetitions)
    else:
        exceed = mdd > budget
        count = int(exceed.sum())
        value = exceed.mean(axis=0)
    if squeezed:
        value = float(value) if np.ndim(value) == 0 else float(value[0])
    return _evidence(
        EvidenceStatus.COMPUTED,
        value,
        reason_detail="ok",
        n_observations=int(panel.shape[0]),
        n_scenarios=int(repetitions),
        n_exceedances=count,
        drawdown_budget=budget,
        interpretation=(
            "estimated scenario frequency under the resampling plan, "
            "not a true future default probability"
        ),
    )


def joint_drawdown_occupancy(
    returns0,
    returns1,
    drawdown_threshold0=0.0,
    drawdown_threshold1=0.0,
    *,
    calendar=None,
    min_periods=_MIN_PERIODS_DEFAULT,
    backend="cpu_reference",
):
    """Paired observed-path co-occupancy ``mean((D0 > d0) & (D1 > d1))``.

    Both drawdown paths reuse the repo drawdown authority.  ``d0``/``d1``
    default to 0 (strictly underwater).  Diagnostics carry each path's
    underwater fraction: a low co-occupancy can simply mean one path is
    never underwater and is not by itself low risk (§7.3).
    """
    _check_backend(backend)
    threshold0 = _check_threshold(drawdown_threshold0, "drawdown_threshold0")
    threshold1 = _check_threshold(drawdown_threshold1, "drawdown_threshold1")
    min_periods = _check_min_periods(min_periods)

    base, base_squeezed, code0 = _prepare_return_panel(returns0, calendar=calendar, name="returns0")
    candidate, cand_squeezed, code1 = _prepare_return_panel(
        returns1, calendar=calendar, name="returns1"
    )
    # calendar check runs inside _prepare_return_panel against each length;
    # a shared bad calendar must invalidate the pair regardless of shape.
    if code0 is not None:
        return _insufficient_or_invalid(code0)
    if base.shape[0] != candidate.shape[0]:
        raise ValueError(
            f"returns0 and returns1 must share the same clock "
            f"({base.shape[0]} != {candidate.shape[0]} observations)"
        )
    # Family-unit strict window (§6.1 rule 4): joint trim + joint interior check.
    joint = np.concatenate([base, candidate], axis=1)
    joint_window, _, code = _prepare_return_panel(joint, calendar=calendar)
    if code is not None:
        return _insufficient_or_invalid(code)
    base_w = joint_window[:, :base.shape[1]]
    cand_w = joint_window[:, base.shape[1]:]
    if base_w.shape[0] < min_periods:
        return _evidence(
            EvidenceStatus.INSUFFICIENT_DATA,
            reason_detail="observations_too_few",
            n_observations=int(base_w.shape[0]),
            min_periods=int(min_periods),
        )

    d0 = _positive_drawdown_path(base_w, backend)
    d1 = _positive_drawdown_path(cand_w, backend)
    if backend == "cuda":
        import cupy as cp

        value = cp.asnumpy(cp.mean((d0 > threshold0) & (d1 > threshold1), axis=0))
        underwater0 = float(cp.asnumpy(cp.mean(d0 > 0.0)))
        underwater1 = float(cp.asnumpy(cp.mean(d1 > 0.0)))
    else:
        value = np.mean((d0 > threshold0) & (d1 > threshold1), axis=0)
        underwater0 = float(np.mean(d0 > 0.0))
        underwater1 = float(np.mean(d1 > 0.0))
    if base_squeezed and cand_squeezed:
        value = float(value[0])
    return _evidence(
        EvidenceStatus.COMPUTED,
        value,
        reason_detail="ok",
        n_observations=int(base_w.shape[0]),
        underwater_fraction_0=underwater0,
        underwater_fraction_1=underwater1,
        drawdown_threshold0=threshold0,
        drawdown_threshold1=threshold1,
        interpretation=(
            "co-occupancy of strictly-over-threshold drawdowns; low common "
            "occupancy may reflect a path that is never underwater rather "
            "than low risk"
        ),
    )
