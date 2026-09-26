"""CPU reference primitives for the QE extension metric families (plan §6).

FP64, deterministic, formula-for-formula implementations of the plan §6
shared primitives; the independent small-data oracles in plan §12 pin their
values.  These are the shared layer that the ``metrics/extension`` metric
producers (paired_performance, oos_delta, path_risk, book_risk, ...) compose;
they are NOT the CPU_FAST/Numba production kernels.

Reuse map (no second implementations of existing capabilities, plan §1.1):
- HAC long-run variance REUSES ``metrics.robustness.compute_hac_variance``
  (imported, not rewritten).  That kernel returns the variance of the mean,
  Ω/T with Ω = γ0 + 2 Σ (1-l/(L+1)) γl and every lag denominator T (Bartlett
  weights fixed), exactly the plan §6.7 formula; this primitive multiplies by
  T to expose Ω and adds the fail-closed valid mask the raw kernel lacks.
- ``ResamplingPlan`` generation follows plan §6.3 exactly: one NumPy PCG64
  draw shared by CPU/GPU; the sampler is an allowed CPU control-plane
  component and computes no financial statistics.

Fail-closed rules (plan §6.1): statistical reduction is FP64 (FP32/int inputs
are promoted exactly); time-series primitives require complete finite inputs
on a common grid (NaN/Inf are never compressed); log-path primitives require
r > -1 (total loss is flagged by the metric layer as INVALID_EVIDENCE, never
deleted); Ω <= 0 and unstable variances are reported via ``valid`` masks, no
epsilon is ever added to manufacture significance.

``workspace_bytes`` only decides production tiling; the reference
implementations validate it but never let it change a formula, a sample or
the RNG sequence.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.contracts.extension_policy import (
    HacLrvResult,
    MomentResult,
    PathMddResult,
    PathSummaryBlock,
    PrefixMoments,
    ResamplingPlan,
    ResamplingPolicy,
    SharedSortArtifact,
    TailMeanResult,
    _SORT_PRODUCTS,
    _integer,
)
from quant_evaluator.metrics.robustness import compute_hac_variance

__all__ = [
    "make_resampling_plan",
    "prefix_moments",
    "sample_moments",
    "hac_lrv",
    "path_block_summaries",
    "sample_path_mdd",
    "merge_path_summaries",
    "path_summary_1d",
    "upper_tail_mean",
    "upper_tail_weights",
    "shared_sort",
]


# ---------------------------------------------------------------------------
# shared validation helpers
# ---------------------------------------------------------------------------

def _stat_fp64(name: str, value: object, ndim: int, *,
               require_finite: bool = True) -> np.ndarray:
    """Promote a real numeric array to FP64 for reduction (plan §1.1 precision).

    bool/complex/object dtypes are rejected outright; FP32 and integer inputs
    are promoted to FP64 exactly (no precision loss, rank/tie preserving).
    """
    if not isinstance(value, np.ndarray):
        raise InvalidContractError(
            f"{name} must be a numpy ndarray, got {type(value).__name__}"
        )
    if value.dtype.kind not in "fiu":
        raise InvalidContractError(
            f"{name} must have a real numeric dtype (bool/complex/object rejected), "
            f"got {value.dtype}"
        )
    if value.ndim != ndim:
        raise InvalidContractError(f"{name} must be rank-{ndim}, got shape {value.shape}")
    array = value.astype(np.float64, copy=True)
    if require_finite and not np.isfinite(array).all():
        raise InvalidContractError(
            f"{name} must be complete and finite on the common grid; internal NaN/Inf "
            "must not be compressed into adjacent observations (plan §6.1.3)"
        )
    return array


def _check_workspace(workspace_bytes: int) -> int:
    return _integer(workspace_bytes, "workspace_bytes", 0)


def _int_matrix(name: str, value: object, ndim: int) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "iu" or value.ndim != ndim:
        raise InvalidContractError(
            f"{name} must be a rank-{ndim} integer ndarray, got "
            f"{type(value).__name__ if not isinstance(value, np.ndarray) else value.dtype!s}"
        )
    return np.asarray(value, dtype=np.int64)


# ---------------------------------------------------------------------------
# ResamplingPlan (plan §6.3)
# ---------------------------------------------------------------------------

def make_resampling_plan(policy: ResamplingPolicy, source_time_ref: str) -> ResamplingPlan:
    """Generate the frozen (starts (B,J), lengths (J,)) plan once (plan §6.3).

    NumPy PCG64 with the policy seed; one call serves the whole factor family
    on every device.  The trailing short block starts from the FULL-block start
    set — identical to drawing complete blocks and truncating.
    """
    if not isinstance(policy, ResamplingPolicy):
        raise InvalidContractError("policy must be a ResamplingPolicy")
    if not isinstance(source_time_ref, str) or not source_time_ref.strip():
        raise InvalidContractError("source_time_ref must be a non-empty string")
    block = int(policy.block_length)
    j = -(-int(policy.output_length) // block)
    lengths = np.full(j, block, dtype=np.int64)
    lengths[-1] = int(policy.output_length) - (j - 1) * block
    rng = np.random.Generator(np.random.PCG64(int(policy.seed)))
    # End-inclusive uniform over [0, sample_length - block_length]; one batched
    # draw consumes the PCG64 stream exactly as plan §12 block_plan does.
    starts = rng.integers(0, int(policy.sample_length) - block + 1,
                          size=(int(policy.repetitions), j), dtype=np.int64)
    return ResamplingPlan(starts=starts, lengths=lengths,
                          source_time_ref=source_time_ref, policy=policy)


# ---------------------------------------------------------------------------
# exact block moments (plan §6.4): the bootstrap never expands [B,T,F]
# ---------------------------------------------------------------------------

def prefix_moments(values_tf, *, workspace_bytes: int = 0) -> PrefixMoments:
    """Center and 0-padded prefix sums of z and z² (plan §6.4).

    Returns ``PrefixMoments(center (F,), p1 (T+1,F), p2 (T+1,F))`` with
    ``p1[j,f] = Σ_{t<j} z[t,f]`` and ``p2[j,f] = Σ_{t<j} z[t,f]²``.
    """
    _check_workspace(workspace_bytes)
    r = _stat_fp64("values_tf", values_tf, 2)
    center = r.mean(axis=0)
    z = r - center
    f = r.shape[1]
    p1 = np.vstack([np.zeros(f), np.cumsum(z, axis=0)])
    p2 = np.vstack([np.zeros(f), np.cumsum(z * z, axis=0)])
    return PrefixMoments(center=center, p1=p1, p2=p2)


def sample_moments(center, p1, p2, starts, lengths, *, workspace_bytes: int = 0) -> MomentResult:
    """Exact bootstrap block moments via prefix sums (plan §6.4).

    For each replicate the block sums gather ``P[hi] - P[lo]`` per block —
    O(B·J·F) work, O(B·F) temporaries; no [B,T,F] panel is ever materialized.
    The result is the mean and the ``ddof=1`` sample variance of the sampled
    path, equal to the day-by-day expansion within provable FP64 tolerance
    (plan §6.1.6: negative rounding beyond ``64·eps·max(scale, tiny)`` marks
    the column invalid instead of fabricating a variance).
    """
    _check_workspace(workspace_bytes)
    center = _stat_fp64("center", center, 1)
    p1 = _stat_fp64("p1", p1, 2)
    p2 = _stat_fp64("p2", p2, 2)
    if p1.shape != p2.shape or p1.shape[1] != center.shape[0]:
        raise InvalidContractError(
            f"p1 {p1.shape} / p2 {p2.shape} / center {center.shape} disagree"
        )
    t = p1.shape[0] - 1
    starts = _int_matrix("starts", starts, 2)
    lengths = _int_matrix("lengths", lengths, 1)
    b, j = starts.shape
    if lengths.shape != (j,) or np.any(lengths <= 0):
        raise InvalidContractError("lengths must be positive with shape (J,) matching starts")
    if np.any(starts < 0) or np.any(starts + lengths[None, :] > t):
        raise InvalidContractError("sample windows must lie inside the source interval [0, T)")
    n_star = int(lengths.sum())
    if n_star < 2 or t < 2:
        raise InvalidContractError("at least two effective observations are required")
    sum1 = np.zeros((b, center.shape[0]))
    sum2 = np.zeros_like(sum1)
    for jdx in range(j):
        lo = starts[:, jdx]
        hi = lo + int(lengths[jdx])
        sum1 += p1[hi] - p1[lo]
        sum2 += p2[hi] - p2[lo]
    centered_ss = sum2 - sum1 * sum1 / n_star
    tolerance = 64 * np.finfo(np.float64).eps * np.maximum(np.abs(sum2), np.finfo(np.float64).tiny)
    valid = centered_ss >= -tolerance
    var = np.maximum(centered_ss, 0.0) / (n_star - 1)
    var = np.where(valid, var, np.nan)
    mean = center[None, :] + sum1 / n_star
    return MomentResult(mean=mean, var=var, valid=valid)


# ---------------------------------------------------------------------------
# HAC long-run variance (plan §6.7) — REUSES the existing kernel
# ---------------------------------------------------------------------------

def hac_lrv(values_tf, *, max_lag: int, workspace_bytes: int = 0) -> HacLrvResult:
    """Long-run variance Ω per column via the existing Bartlett kernel (plan §6.7).

    Reuses :func:`metrics.robustness.compute_hac_variance` (imported, not a
    second implementation).  That kernel demeans internally, uses the fixed
    Bartlett weights ``1 - l/(L+1)`` and the lag denominator T, returning the
    variance of the mean Ω/T for a complete column; this primitive multiplies
    by T to report Ω itself (mean SE is then ``sqrt(Ω/T)``).  Fail-closed
    rules: a column with any non-finite entry is never compressed (the kernel
    would trim endpoints and change lag semantics) and is marked invalid; Ω ≤ 0
    or non-finite is invalid — no epsilon is added (plan §6.7).
    """
    _check_workspace(workspace_bytes)
    max_lag = _integer(max_lag, "max_lag", 0)
    values = values_tf if isinstance(values_tf, np.ndarray) else None
    if values is None:
        raise InvalidContractError("values_tf must be a numpy ndarray")
    if values.dtype.kind not in "fiu":
        raise InvalidContractError(
            f"values_tf must have a real numeric dtype, got {values.dtype}"
        )
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise InvalidContractError("values_tf must be rank-1 or rank-2 (T,F)")
    t, f = values.shape
    omega2 = np.full(f, np.nan)
    valid = np.zeros(f, dtype=bool)
    if t == 0:
        return HacLrvResult(omega2=omega2, valid=valid)
    matrix = values.astype(np.float64, copy=True)
    for col in range(f):
        column = matrix[:, col]
        if not np.isfinite(column).all():
            continue  # internal NaN/Inf: fail closed, never compress (plan §6.1.3)
        var_of_mean = compute_hac_variance(column.reshape(-1, 1), max_lag=max_lag,
                                           kernel="bartlett")[0]
        if not np.isfinite(var_of_mean):
            continue  # kernel requires len >= max_lag + 10
        omega = float(var_of_mean) * t
        omega2[col] = omega
        valid[col] = omega > 0.0
    return HacLrvResult(omega2=omega2, valid=valid)


# ---------------------------------------------------------------------------
# ordered path summaries: exact max drawdown as a combine operation (plan §6.5)
# ---------------------------------------------------------------------------

def _log_path_summary_block(block: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                        np.ndarray, np.ndarray]:
    """Summary (total, high, low, log_drawdown) of a leading-zero log prefix."""
    cs = np.cumsum(block, axis=0)
    prefix = np.vstack([np.zeros((1, block.shape[1])), cs])
    total = cs[-1]
    high = prefix.max(axis=0)
    low = prefix.min(axis=0)
    run_max = np.maximum.accumulate(prefix, axis=0)
    drawdown = (run_max - prefix).max(axis=0)
    return total, high, low, drawdown


def path_summary_1d(returns_1d) -> np.ndarray:
    """Ordered summary (total, high, low, log_drawdown) of one return path.

    Includes the initial wealth 0 in the prefix (plan §6.5); requires r > -1.
    """
    r = _stat_fp64("returns_1d", returns_1d, 1)
    if r.size and np.any(r <= -1.0):
        raise InvalidContractError(
            "log-path scope requires r > -1; r == -1 is terminal total loss "
            "(flagged separately by the metric layer), r < -1 is RETURN_BASIS_MISMATCH"
        )
    total, high, low, drawdown = _log_path_summary_block(np.log1p(r)[:, None])
    return np.array([total[0], high[0], low[0], drawdown[0]])


def merge_path_summaries(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Ordered, associative, NON-commutative merge of path summaries (plan §6.5).

    ``a`` precedes ``b`` in time; the cross-block peak-to-trough term
    ``h_A - s_A - l_B`` is handled here, so block order must never be
    permuted.  Operates on the trailing 4-axis; leading axes broadcast.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    total = a[..., 0] + b[..., 0]
    high = np.maximum(a[..., 1], a[..., 0] + b[..., 1])
    low = np.minimum(a[..., 2], a[..., 0] + b[..., 2])
    drawdown = np.maximum(np.maximum(a[..., 3], b[..., 3]), a[..., 1] - a[..., 0] - b[..., 2])
    return np.stack([total, high, low, drawdown], axis=-1)


def path_block_summaries(returns_tf, lengths, *, workspace_bytes: int = 0) -> PathSummaryBlock:
    """Precompute ordered summaries for every (start, needed length) (plan §6.5.1).

    O(T·l·F) simple scan per plan §6.5; the returned lookup has shape
    ``(S, L, F, 4)`` where ``S = T - min(lengths) + 1`` and ``L`` indexes the
    sorted unique block ``lengths``.  Cells outside a length's legal start
    range are NaN and are never consumed by legal plans.
    """
    _check_workspace(workspace_bytes)
    r = _stat_fp64("returns_tf", returns_tf, 2)
    lengths = _int_matrix("lengths", lengths, 1)
    t, f = r.shape
    if np.any(r <= -1.0):
        raise InvalidContractError(
            "log-path scope requires r > -1; r == -1 is terminal total loss "
            "(flagged separately by the metric layer), r < -1 is RETURN_BASIS_MISMATCH"
        )
    if t == 0 or lengths.size == 0 or np.any(lengths <= 0) or np.any(lengths > t):
        raise InvalidContractError("lengths must be positive and not exceed T")
    slots = np.unique(lengths)
    max_start = t - int(slots.min())
    lookup = np.full((max_start + 1, len(slots), f, 4), np.nan)
    g = np.log1p(r)
    for slot_idx, block_len in enumerate(slots):
        block_len = int(block_len)
        for start in range(0, t - block_len + 1):
            total, high, low, drawdown = _log_path_summary_block(g[start:start + block_len])
            lookup[start, slot_idx, :, 0] = total
            lookup[start, slot_idx, :, 1] = high
            lookup[start, slot_idx, :, 2] = low
            lookup[start, slot_idx, :, 3] = drawdown
    valid = np.ones(f, dtype=bool)  # complete finite input with r > -1: all columns usable
    return PathSummaryBlock(summary_lookup=lookup, length_slots=slots, valid=valid)


def sample_path_mdd(summary_lookup, starts, lengths, *, workspace_bytes: int = 0) -> PathMddResult:
    """Merge block summaries in order per replicate and convert to MDD (plan §6.5).

    ``mdd = -expm1(-log_drawdown)`` of the merged (total, high, low, drawdown)
    state.  Blocks are consumed strictly in plan order; the merge is
    associative but NOT commutative, so permuted block orders are invalid by
    construction.  ``lengths`` must be the same plan-lengths vector that
    produced ``summary_lookup`` (its sorted unique values are the L slots).
    """
    _check_workspace(workspace_bytes)
    lookup = np.asarray(summary_lookup, dtype=np.float64)
    if lookup.ndim != 4 or lookup.shape[3] != 4:
        raise InvalidContractError("summary_lookup must be (S, L, F, 4)")
    starts = _int_matrix("starts", starts, 2)
    lengths = _int_matrix("lengths", lengths, 1)
    b, j = starts.shape
    if j == 0 or lengths.shape != (j,) or np.any(lengths <= 0):
        raise InvalidContractError("lengths must be positive with shape (J,) matching starts")
    s_rows, l_slots, f, _ = lookup.shape
    slots = np.unique(lengths)
    # path_block_summaries stores sorted unique lengths as slots; require the
    # exact same set so slot indices align bit-for-bit with the producing call.
    if len(slots) != l_slots:
        raise InvalidContractError(
            "lengths must be the same plan-lengths vector that produced summary_lookup "
            f"({len(slots)} unique lengths vs {l_slots} slots)"
        )
    # S = T - min(lengths) + 1 recovers the source length T for bounds checks.
    t_total = s_rows + int(slots.min()) - 1
    if np.any(starts < 0) or np.any(starts + lengths[None, :] > t_total):
        raise InvalidContractError("sample windows must lie inside the source interval")
    mdd = np.empty((b, f))
    for bdx in range(b):
        state = np.zeros((f, 4))
        for jdx in range(j):
            start = int(starts[bdx, jdx])
            block_len = int(lengths[jdx])
            slot_index = int(np.searchsorted(slots, block_len))
            summary = lookup[start, slot_index]
            if not np.isfinite(summary).all():
                raise InvalidContractError(
                    f"lookup cell (start={start}, length={block_len}) is NaN — "
                    "the plan references a block the lookup does not cover"
                )
            state = merge_path_summaries(state, summary)
        mdd[bdx] = -np.expm1(-state[:, 3])
    valid = np.isfinite(mdd).all(axis=0)
    return PathMddResult(mdd=mdd, valid=valid)


# ---------------------------------------------------------------------------
# tail mean with a unified boundary-tie rule (plan §6.6)
# ---------------------------------------------------------------------------

def upper_tail_weights(values, confidence: float) -> np.ndarray:
    """Per-column boundary-tie weights summing exactly to the tail mass (plan §6.6).

    Dates above the cutoff weigh 1, dates below weigh 0, dates tied with the
    cutoff split the remaining mass equally — so re-ordering tied dates can
    never change the conditional result (plan §6.6, the book conditional
    loss rule).  Columns with non-finite entries yield NaN weights.
    """
    x = _stat_fp64("values", values, 2, require_finite=False)
    _validate_confidence(confidence)
    n = x.shape[0]
    weights = np.full(x.shape, np.nan)
    for col in range(x.shape[1]):
        column = x[:, col]
        if n == 0 or not np.isfinite(column).all():
            continue
        weights[:, col] = _tail_weights_1d(column, float(confidence))
    return weights


def _validate_confidence(confidence: float) -> float:
    if isinstance(confidence, (bool, np.bool_)) \
            or not isinstance(confidence, (int, float, np.integer, np.floating)) \
            or not np.isfinite(confidence) or not 0.0 < float(confidence) < 1.0:
        raise InvalidContractError("confidence must be finite and strictly between zero and one")
    return float(confidence)


def _tail_mass(n: int, confidence: float) -> float:
    mass = (1.0 - confidence) * n
    nearest = round(mass)
    if nearest >= 1 and abs(mass - nearest) <= 8 * np.finfo(np.float64).eps * max(1.0, mass):
        mass = float(nearest)  # snap near-integer masses to avoid cross-platform off-by-one
    return mass


def _tail_weights_1d(x: np.ndarray, confidence: float) -> np.ndarray:
    """Boundary-tie weights for one finite column (plan §6.6 / §12 tail_weights)."""
    n = x.shape[0]
    mass = _tail_mass(n, confidence)
    k = int(math.floor(mass))
    index = min(n - 1, int(math.ceil(mass)) - 1)
    cutoff = np.sort(x)[::-1][index]
    above = x > cutoff
    tied = x == cutoff
    weights = above.astype(np.float64)
    weights[tied] = (mass - int(above.sum())) / int(tied.sum())
    return weights


def upper_tail_mean(values_tf, *, confidence: float, min_mass: float = 0.0) -> TailMeanResult:
    """``UTM_q``: the mean of the worst ``1-q`` empirical mass (plan §6.6).

    NOT "filter x >= an interpolated quantile and average".  ``value = Σ w·x /
    Σ w`` with the boundary-tie weights of :func:`upper_tail_weights`;
    ``mass`` is the demanded tail mass m.  ``valid`` is False for empty or
    non-finite columns and when ``m < min_mass`` (plan §6.1 min_tail_mass).
    """
    x = _stat_fp64("values_tf", values_tf, 2, require_finite=False)
    confidence = _validate_confidence(confidence)
    min_mass = float(min_mass)
    if min_mass < 0 or not np.isfinite(min_mass):
        raise InvalidContractError("min_mass must be a finite non-negative number")
    n, f = x.shape
    value = np.full(f, np.nan)
    mass = np.full(f, np.nan)
    valid = np.zeros(f, dtype=bool)
    for col in range(f):
        column = x[:, col]
        if n == 0 or not np.isfinite(column).all():
            continue
        m = _tail_mass(n, confidence)
        weights = _tail_weights_1d(column, confidence)
        total_weight = weights.sum()
        if total_weight <= 0:
            continue
        value[col] = float(weights @ column / total_weight)
        mass[col] = m
        valid[col] = m >= min_mass
    return TailMeanResult(value=value, mass=mass, valid=valid)


# ---------------------------------------------------------------------------
# shared sort: one sort, many consumers (plan §6.2)
# ---------------------------------------------------------------------------

def shared_sort(values_tfn, mask_tfn, *, products: Tuple[str, ...] = ("rank",),
                workspace_bytes: int = 0, input_ref: str = "", axis_ref: str = "",
                mask_ref: str = "", orientation_ref: str = "",
                device_id: str = "cpu") -> SharedSortArtifact:
    """Stable cross-section sort artifact shared by rank/count/quantile consumers.

    NumPy reference (plan §6.2): per (t, f) row of width N, a stable argsort
    with masked/non-finite entries moved behind a +inf tail sentinel so they
    never share runs with legal finite values.  ``sorted_values`` keeps the
    input dtype; the sort key is FP64 (exact promotion, rank/tie preserving).
    Run boundaries are produced only when a consumer needs them ("rank" or
    "distinct_count"); quantile boundaries keep consuming the repo's own
    percentile rules (average-rank bucketing is NOT percentile bucketing).
    """
    _check_workspace(workspace_bytes)
    values = values_tfn
    if not isinstance(values, np.ndarray) or values.dtype.kind not in "fiu":
        raise InvalidContractError(
            "values_tfn must be a real numeric ndarray (bool/complex/object rejected)"
        )
    if values.ndim != 3:
        raise InvalidContractError(f"values_tfn must be rank-3 (T,F,N), got shape {values.shape}")
    mask = mask_tfn
    if not isinstance(mask, np.ndarray) or mask.dtype.kind != "b" or mask.shape != values.shape:
        raise InvalidContractError("mask_tfn must be a bool ndarray with the same shape as values_tfn")
    if not products:
        raise InvalidContractError("products must be a non-empty tuple")
    for product in products:
        if product not in _SORT_PRODUCTS:
            raise InvalidContractError(f"unknown requested product {product!r}; allowed: {_SORT_PRODUCTS}")
    t, f, n = values.shape
    r = t * f
    flat = values.reshape(r, n)
    valid = (mask & np.isfinite(values)).reshape(r, n)
    keys = np.where(valid, flat.astype(np.float64, copy=False), np.inf)
    order = np.argsort(keys, axis=1, kind="stable").astype(np.int64)
    sorted_values = np.take_along_axis(flat, order, axis=1)
    finite_count = valid.sum(axis=1).astype(np.int64)
    run_start = run_end = None
    if "rank" in products or "distinct_count" in products:
        sorted_keys = np.take_along_axis(keys, order, axis=1)
        sorted_valid = np.take_along_axis(valid, order, axis=1)
        idx = np.arange(n)[None, :]
        # A run starts at position 0, at a value change, and at each invalid
        # entry (invalid entries are their own empty runs behind the sentinel).
        new_run = np.ones((r, n), dtype=bool)
        new_run[:, 1:] = sorted_keys[:, 1:] != sorted_keys[:, :-1]
        new_run |= ~sorted_valid
        run_start = np.maximum.accumulate(np.where(new_run, idx, 0), axis=1).astype(np.int64)
        boundary_end = np.ones((r, n), dtype=bool)
        boundary_end[:, :-1] = sorted_keys[:, :-1] != sorted_keys[:, 1:]
        boundary_end |= ~sorted_valid
        # Next boundary at-or-after each position: a right-to-left MIN scan of
        # boundary positions (reversed order, carrying the smallest seen).
        # Rev index r holds original position n-1-r, paired with value n-1-r.
        rev_next = np.minimum.accumulate(
            np.where(boundary_end[:, ::-1], n - 1 - idx[0][None, :], n), axis=1
        )
        run_end = (rev_next[:, ::-1] + 1).astype(np.int64)
    return SharedSortArtifact(
        input_ref=str(input_ref),
        axis_ref=str(axis_ref),
        mask_ref=str(mask_ref),
        orientation_ref=str(orientation_ref),
        dtype=str(values.dtype),
        layout="tfn",
        order=order,
        sorted_values=sorted_values,
        finite_count=finite_count,
        requested_products=tuple(products),
        run_start=run_start,
        run_end_exclusive=run_end,
        workspace_bytes=_check_workspace(workspace_bytes),
        device_id=str(device_id),
    )
