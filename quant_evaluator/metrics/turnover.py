"""
Turnover metrics for factor portfolios.

Measures stability and trading cost implications.

Canonical turnover definition
-----------------------------
Turnover between two consecutive periods is defined as

    turnover_t = 0.5 * sum_n |w_t,n - w_{t-1},n|

on weights that each sum to 1 (per period, over finite assets). All three
entry points conform to this single authority:

- ``compute_turnover_series`` — the canonical implementation on explicit
  weight matrices (T, N).
- ``estimate_turnover_from_ranks`` — derives per-period proxy weights from
  cross-sectional average-tie ranks (``scipy.stats.rankdata`` with
  method="average"), normalized to sum 1 over finite assets, then applies
  the canonical ``compute_turnover_series``.
- ``registry_adapters.compute_turnover_value`` — the same rank-weight
  construction, exposed as a per-factor scalar for the metric registry.

Because weights are sum-1 normalized, turnover values are invariant to
universe size N (doubling the universe with duplicated assets leaves
turnover unchanged) and a complete rank inversion (Spearman rho = -1)
yields high turnover (~1.0), not zero.
"""

from typing import Optional
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch

# Try to import numba for JIT compilation
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # No-op decorator if numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])


@njit(cache=True)
def _compute_turnover_series_numba(weights: np.ndarray) -> np.ndarray:
    """
    Numba-optimized turnover series computation.

    Note: fastmath disabled to ensure correct NaN handling.

    Args:
        weights: Weight matrix (T, N)

    Returns:
        Turnover series (T,)
    """
    T, N = weights.shape
    turnover = np.empty(T, dtype=np.float64)
    turnover[0] = np.nan

    for t in range(1, T):
        sum_abs_delta = 0.0
        n_valid = 0

        for n in range(N):
            w0 = weights[t-1, n]
            w1 = weights[t, n]

            if np.isfinite(w0) and np.isfinite(w1):
                sum_abs_delta += abs(w1 - w0)
                n_valid += 1

        if n_valid == N and N > 0:
            turnover[t] = 0.5 * sum_abs_delta
        else:
            turnover[t] = np.nan

    return turnover


def compute_turnover(
    weights_t0: np.ndarray,
    weights_t1: np.ndarray,
    method: str = "half_sum_abs",
) -> float:
    """
    Compute turnover between two weight vectors.

    Canonical definition: 0.5 * sum(abs(delta_weights))

    Args:
        weights_t0: Weights at time t (N,)
        weights_t1: Weights at time t+1 (N,)
        method: Turnover method ("half_sum_abs")

    Returns:
        Turnover scalar, or NaN if invalid
    """
    if method != "half_sum_abs":
        raise ValueError(f"Unknown turnover method: {method}")

    # Unknown weights are not zero positions: a partial sum understates trades.
    weights_t0 = np.asarray(weights_t0, dtype=np.float64)
    weights_t1 = np.asarray(weights_t1, dtype=np.float64)
    if weights_t0.ndim != 1 or weights_t0.shape != weights_t1.shape:
        raise ValueError("weights must be matching one-dimensional vectors")
    mask = np.isfinite(weights_t0) & np.isfinite(weights_t1)

    if mask.size == 0 or not np.all(mask):
        return np.nan

    w0 = weights_t0[mask]
    w1 = weights_t1[mask]

    # Canonical turnover: 0.5 * sum(abs(delta))
    turnover = 0.5 * np.sum(np.abs(w1 - w0))

    return float(turnover)


def compute_turnover_series(
    weights: np.ndarray,
    method: str = "half_sum_abs",
) -> np.ndarray:
    """
    Compute turnover time series from weight matrix (vectorized).

    Optimized implementation using numba JIT compilation when available,
    falling back to numpy broadcasting for compatibility.

    Args:
        weights: Weight matrix (T, N)
        method: Turnover method

    Returns:
        Turnover series (T,), first observation is NaN
    """
    if method != "half_sum_abs":
        raise ValueError(f"Unknown turnover method: {method}")

    T, N = weights.shape

    if T < 2:
        return np.full(T, np.nan, dtype=np.float64)

    # Use numba-optimized version if available (10-20x faster)
    if HAS_NUMBA:
        return _compute_turnover_series_numba(weights)

    # Fallback: pure numpy vectorized implementation
    turnover_series = np.empty(T, dtype=np.float64)
    turnover_series[0] = np.nan

    # Process differences in-place using slicing
    w_diff = weights[1:] - weights[:-1]  # (T-1, N)

    # Compute finite mask efficiently
    finite_mask = np.isfinite(w_diff)  # (T-1, N)

    # Fast path: if no NaNs, use simple sum
    if np.all(finite_mask):
        turnover_series[1:] = 0.5 * np.sum(np.abs(w_diff), axis=1)
    else:
        # Need to handle NaNs: set them to zero for summation
        abs_diff = np.abs(w_diff)
        abs_diff[~finite_mask] = 0.0

        # Sum and check validity
        turnover_values = 0.5 * np.sum(abs_diff, axis=1)  # (T-1,)
        n_valid = np.sum(finite_mask, axis=1)  # (T-1,)

        # Set to NaN where no valid observations
        turnover_values[n_valid != N] = np.nan
        turnover_series[1:] = turnover_values

    return turnover_series


def _rank_weights_matrix_reference(values: np.ndarray, min_obs: int = 10) -> np.ndarray:
    """Bit-exact oracle of :func:`_rank_weights_matrix` (per-(t,f) loop)."""
    from scipy.stats import rankdata

    shape = values.shape
    T = shape[0]; N = shape[1]; F = shape[2]
    weights = np.full((T, N, F), np.nan, dtype=np.float64)

    for f in range(F):
        for t in range(T):
            row = values[t, :, f]
            finite = np.isfinite(row)
            n_finite = int(np.sum(finite))
            if n_finite < min_obs or n_finite < 2:
                continue
            ranks = rankdata(row[finite], method="average")
            weights[t, :, f] = 0.0
            # Sum-1 normalization: universe-size invariant proxy weights.
            weights[t, finite, f] = ranks / np.sum(ranks)

    return weights


def _rank_weights_matrix(values: np.ndarray, min_obs: int = 10) -> np.ndarray:
    """Compute per-period, per-factor rank-based proxy weights.

    For each (t, f) cross-section, assets are ranked with average-tie ranks
    (``scipy.stats.rankdata(method="average")``) and the ranks are normalized
    to sum 1 over the finite assets of that cross-section.
    Non-finite signals receive explicit zero proxy weight on an otherwise
    eligible date. Insufficient dates remain unknown, not cash.

    Args:
        values: Factor values (T, N, F)
        min_obs: Minimum finite assets required in a cross-section

    Returns:
        Rank-weight matrix (T, N, F), NaN where insufficient data

    Vectorized: the whole (T*F, N) panel is ranked in a single
    ``rankdata(axis=1)`` call.  Non-finite signals become ``+inf`` sentinels
    that always rank last, so the average rank of every finite asset is exactly
    what a per-row ``rankdata(row[finite])`` would yield (their rank numbers
    depend only on the relative order among finite values); only the finite
    ranks enter the sum-1 normalization.  This is bit-identical to the loop.
    """
    from scipy.stats import rankdata

    values = np.asarray(values, dtype=np.float64)
    shape = values.shape
    T = shape[0]; N = shape[1]; F = shape[2]
    # Row (t*F + f) of `flat` must equal ``values[t, :, f]`` (length N).  Since
    # ``values`` is (T, N, F), transpose to (T, F, N) before flattening.
    flat = values.transpose(0, 2, 1).reshape(T * F, N)
    finite_mask = np.isfinite(flat)
    n_finite = np.sum(finite_mask, axis=1)
    valid_row = (n_finite >= min_obs) & (n_finite >= 2)

    # Finite sentinel (row max + 1) ranks last without disturbing finite ranks;
    # a finite value avoids the scipy ``rankdata(axis=1)`` quirk with +inf.
    row_max = np.where(finite_mask, flat, -np.inf).max(axis=1, keepdims=True)
    sentinel = np.where(finite_mask, flat, row_max + 1.0)
    ranks_full = rankdata(sentinel, axis=1)  # (T*F, N)
    sum_finite = np.sum(np.where(finite_mask, ranks_full, 0.0), axis=1, keepdims=True)
    norm = ranks_full / sum_finite
    weights_flat = np.where(finite_mask, norm, 0.0)
    # Rows that do not meet min_obs are entirely unknown (NaN).
    weights_flat = np.where(valid_row[:, None], weights_flat, np.nan)
    return weights_flat.reshape(T, F, N).transpose(0, 2, 1)


def estimate_turnover_from_ranks(
    factor_batch: FactorBatch,
    window: int = 1,
) -> np.ndarray:
    """
    Estimate turnover from factor rank changes (proxy, not actual portfolio).

    This is a diagnostic approximation when actual weights are unavailable.
    Per period, cross-sectional average-tie ranks (normalized to sum 1 over
    finite assets) act as proxy weights; turnover is then the canonical
    0.5 * sum(|delta weights|) between t-window and t, computed via
    ``compute_turnover_series``.

    A full rank inversion (Spearman rho = -1) therefore yields turnover
    close to 1.0 (the maximum for sum-1 non-negative weights), NOT zero.

    Args:
        factor_batch: Factor values (T, N, F)
        window: Lag for comparison (default 1)

    Returns:
        Estimated turnover series (T, F)
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    values = np.asarray(factor_batch.values, dtype=np.float64)
    T, N, F = values.shape

    if T <= window:
        return np.full((T, F), np.nan, dtype=np.float64)

    # Rank-based proxy weights (average-tie ranks, sum-1 per cross-section).
    weights = _rank_weights_matrix(values, min_obs=10)  # (T, N, F)

    turnover_est = np.full((T, F), np.nan, dtype=np.float64)

    # Canonical turnover between t-window and t for each factor: 0.5 * sum
    # of absolute weight changes over jointly finite assets.
    w0 = weights[:-window, :, :]   # (T-window, N, F)
    w1 = weights[window:, :, :]    # (T-window, N, F)
    abs_diff = np.abs(w1 - w0)
    mask = np.isfinite(w0) & np.isfinite(w1)
    sum_abs = np.sum(np.where(mask, abs_diff, 0.0), axis=1)  # (T-window, F)
    n_valid = np.sum(mask, axis=1)                            # (T-window, F)
    # All assets must be observed (a partial sum understates trades); need >= 2.
    valid = (n_valid >= 2) & (n_valid == N)
    turnover_block = np.where(valid, 0.5 * sum_abs, np.nan)   # (T-window, F)
    turnover_est[window:, :] = turnover_block

    return turnover_est


def _estimate_turnover_from_ranks_reference(
    factor_batch: FactorBatch,
    window: int = 1,
) -> np.ndarray:
    """Bit-exact oracle of :func:`estimate_turnover_from_ranks` (per-(f,t) loop)."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    values = np.asarray(factor_batch.values, dtype=np.float64)
    T, N, F = values.shape

    if T <= window:
        return np.full((T, F), np.nan, dtype=np.float64)

    # Rank-based proxy weights (loop version, sum-1 per cross-section).
    weights = _rank_weights_matrix_reference(values, min_obs=10)  # (T, N, F)

    turnover_est = np.full((T, F), np.nan, dtype=np.float64)

    for f in range(F):
        for t in range(window, T):
            w0 = weights[t - window, :, f]
            w1 = weights[t, :, f]
            mask = np.isfinite(w0) & np.isfinite(w1)
            # All assets must be observed (a partial sum understates trades);
            # need at least 2 finite assets.
            if np.sum(mask) < 2 or not np.all(mask):
                continue
            turnover_est[t, f] = 0.5 * np.sum(np.abs(w1[mask] - w0[mask]))

    return turnover_est


def compute_turnover_matrix_batch(
    weights: np.ndarray,
    method: str = "half_sum_abs",
) -> np.ndarray:
    """
    Compute turnover for a batch of weight matrices (fully vectorized with einsum).

    Optimized for large batches of portfolios (e.g., multiple factors or strategies).
    Uses einsum for maximum efficiency on large arrays.

    Args:
        weights: Weight tensor (T, N, P) where P is number of portfolios
        method: Turnover method

    Returns:
        Turnover series (T, P), first observation is NaN for each portfolio
    """
    if method != "half_sum_abs":
        raise ValueError(f"Unknown turnover method: {method}")

    T, N, P = weights.shape

    if T < 2:
        return np.full((T, P), np.nan, dtype=np.float64)

    # For small batches, loop is faster due to overhead
    # Threshold determined empirically
    if P <= 10:
        # Use loop for small batches
        result = np.empty((T, P), dtype=np.float64)
        for p in range(P):
            result[:, p] = compute_turnover_series(weights[:, :, p])
        return result

    # Allocate output
    turnover_series = np.empty((T, P), dtype=np.float64)
    turnover_series[0, :] = np.nan

    # Compute differences (views, not copies)
    w_diff = weights[1:] - weights[:-1]  # (T-1, N, P)

    # Check for NaNs
    has_nans = not np.all(np.isfinite(w_diff))

    if not has_nans:
        # Fast path: no NaNs, direct einsum on absolute differences
        # einsum is faster than sum for large arrays
        turnover_series[1:, :] = 0.5 * np.einsum('tnp->tp', np.abs(w_diff))
    else:
        # Slow path: handle NaNs
        finite_mask = np.isfinite(w_diff)
        abs_diff = np.abs(w_diff)
        abs_diff[~finite_mask] = 0.0

        # Use einsum for summation
        turnover_values = 0.5 * np.einsum('tnp->tp', abs_diff)  # (T-1, P)

        # Check validity per portfolio per period
        n_valid = np.sum(finite_mask, axis=1)  # (T-1, P)
        turnover_values[n_valid != N] = np.nan

        turnover_series[1:, :] = turnover_values

    return turnover_series


def compute_cross_sectional_turnover(
    weights: np.ndarray,
    method: str = "half_sum_abs",
) -> float:
    """
    Compute average turnover across all time periods (single scalar).

    Useful for evaluating overall portfolio stability.

    Args:
        weights: Weight matrix (T, N)
        method: Turnover method

    Returns:
        Mean turnover (scalar), or NaN if insufficient data
    """
    turnover_series = compute_turnover_series(weights, method=method)
    return float(np.nanmean(turnover_series))


def compute_weighted_turnover(
    weights: np.ndarray,
    position_sizes: np.ndarray,
    method: str = "half_sum_abs",
) -> np.ndarray:
    """
    Compute position-size-weighted turnover.

    Standard turnover treats all weight changes equally. Weighted turnover
    accounts for the fact that changes in larger positions have greater impact.

    Args:
        weights: Weight matrix (T, N)
        position_sizes: Position sizes (T, N), e.g., abs(weights)
        method: Turnover method

    Returns:
        Weighted turnover series (T,)
    """
    if method != "half_sum_abs":
        raise ValueError(f"Unknown turnover method: {method}")

    T, N = weights.shape

    if T < 2:
        return np.full(T, np.nan, dtype=np.float64)

    # Get consecutive pairs
    w_t0 = weights[:-1, :]  # (T-1, N)
    w_t1 = weights[1:, :]   # (T-1, N)
    ps_t0 = position_sizes[:-1, :]  # (T-1, N)
    ps_t1 = position_sizes[1:, :]   # (T-1, N)

    # Average position size across periods
    avg_size = 0.5 * (ps_t0 + ps_t1)  # (T-1, N)

    # Finite mask
    finite_mask = (
        np.isfinite(w_t0) & np.isfinite(w_t1) &
        np.isfinite(ps_t0) & np.isfinite(ps_t1)
    )  # (T-1, N)

    # Weight changes multiplied by average position size
    delta = np.abs(w_t1 - w_t0)  # (T-1, N)
    weighted_delta = np.where(finite_mask, delta * avg_size, 0.0)

    # Weighted turnover: 0.5 * sum(|delta| * size) / sum(size)
    numerator = np.sum(weighted_delta, axis=1)  # (T-1,)
    denominator = np.sum(np.where(finite_mask, avg_size, 0.0), axis=1)  # (T-1,)

    with np.errstate(divide='ignore', invalid='ignore'):
        weighted_turnover = 0.5 * numerator / denominator

    # Mask periods with no valid observations
    n_valid = np.sum(finite_mask, axis=1)
    weighted_turnover = np.where((n_valid == N) & (N > 0), weighted_turnover, np.nan)

    # Prepend NaN
    return np.concatenate([[np.nan], weighted_turnover])


def compute_turnover_contribution(
    weights: np.ndarray,
    method: str = "half_sum_abs",
) -> np.ndarray:
    """
    Compute per-asset turnover contribution (vectorized with einsum).

    Decomposes total turnover into contributions from each asset.

    Args:
        weights: Weight matrix (T, N)
        method: Turnover method

    Returns:
        Turnover contribution matrix (T, N), where each row sums to that period's turnover
    """
    if method != "half_sum_abs":
        raise ValueError(f"Unknown turnover method: {method}")

    T, N = weights.shape

    # Initialize output
    contribution = np.full((T, N), np.nan, dtype=np.float64)

    if T < 2:
        return contribution

    # Get consecutive pairs
    w_t0 = weights[:-1, :]  # (T-1, N)
    w_t1 = weights[1:, :]   # (T-1, N)

    # Finite mask
    finite_mask = np.isfinite(w_t0) & np.isfinite(w_t1)  # (T-1, N)

    # Per-asset contribution: 0.5 * |delta|
    delta = w_t1 - w_t0  # (T-1, N)
    contribution_values = 0.5 * np.where(finite_mask, np.abs(delta), 0.0)  # (T-1, N)
    contribution_values[~np.all(finite_mask, axis=1), :] = np.nan

    # Store in output (first row remains NaN)
    contribution[1:, :] = contribution_values

    return contribution


# ---------------------------------------------------------------------------
# Missing-kernel round (2026-09-24): registry kernels for the turnover_stability
# and turnover_adjusted_ic catalog ids.  Both consume the canonical turnover
# series (e.g. ``estimate_turnover_from_ranks``) and implement only the
# documented cross-period reductions; no turnover numerics are duplicated here.
# ---------------------------------------------------------------------------


def _coerce_real_float_matrix(values, name: str) -> np.ndarray:
    """Reject bool / complex / object inputs and coerce to float64."""
    arr = np.asarray(values)
    if arr.dtype == object:
        raise TypeError(f"{name} must be a real numeric array, got object dtype")
    if arr.dtype == bool:
        raise TypeError(f"{name} must be a real numeric array, got bool dtype")
    if np.iscomplexobj(arr):
        raise TypeError(f"{name} must be a real numeric array, got complex dtype")
    return arr.astype(np.float64, copy=False)


def compute_turnover_stability_from_series(
    turnover_series: np.ndarray,
    min_periods: int = 2,
) -> np.ndarray:
    """Cross-period variance of the turnover rate, (F,).

    Doc metric-turnover_stability ("Variance of turnover rate across
    periods", bound 2026-09-24): per factor, the POPULATION variance
    (ddof=0) of the finite per-period turnovers — the same variance
    convention as the sibling ``coverage_stability`` binding — accumulated
    by ``np.var`` over the finite values in time order (bit-identical across
    calls).  Fewer than ``min_periods`` finite periods leave the factor NaN.

    Args:
        turnover_series: (T, F) or (T,) canonical turnover series
            (``0.5 * sum_i |w_t,i - w_{t-1},i|`` on sum-1 rank-proxy weights,
            e.g. from ``estimate_turnover_from_ranks``).
        min_periods: minimum finite periods required (>= 2).

    Returns:
        (F,) float64 variance array; NaN where the gate fails.
    """
    arr = _coerce_real_float_matrix(turnover_series, "turnover_series")
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("turnover_series must be (T,) or (T, F)")
    if isinstance(min_periods, bool) or not isinstance(min_periods, (int, np.integer)):
        raise TypeError("min_periods must be an integer")
    if min_periods < 2:
        raise ValueError("min_periods must be at least 2 for a variance")

    T, F = arr.shape
    out = np.full(F, np.nan, dtype=np.float64)
    for f in range(F):
        finite = arr[np.isfinite(arr[:, f]), f]
        if finite.size >= min_periods:
            out[f] = float(np.var(finite, ddof=0))
    return out


def compute_turnover_adjusted_ic_from_series(
    ic_series: np.ndarray,
    turnover_series: np.ndarray,
    min_periods: int = 20,
) -> np.ndarray:
    """Turnover-adjusted IC, (F,).

    Doc metric-turnover_adjusted_ic (bound 2026-09-24; the doc previously
    recorded ``M = UNAVAILABLE``):

        M_f = mean_{t: IC finite} IC_{t,f} / mean_{t: tau finite} tau_{t,f}

    where ``tau`` is the canonical rank-proxy turnover series.  The IC
    numerator requires at least ``min_periods`` finite days; the turnover
    denominator requires at least 2 finite periods and its mean must be
    finite and strictly positive, otherwise the factor stays NaN.  Means
    accumulate via ``np.mean`` over the finite values in time order
    (bit-identical across calls).

    Args:
        ic_series: (T, F) daily IC series.
        turnover_series: (T, F) canonical turnover series (same (T, F)
            shape as ``ic_series``), e.g. from ``estimate_turnover_from_ranks``.
        min_periods: minimum finite IC days required (>= 2).

    Returns:
        (F,) float64 array; NaN where a gate fails.
    """
    ic = _coerce_real_float_matrix(ic_series, "ic_series")
    tau = _coerce_real_float_matrix(turnover_series, "turnover_series")
    if ic.ndim != 2 or tau.ndim != 2:
        raise ValueError("ic_series and turnover_series must be (T, F)")
    if ic.shape != tau.shape:
        raise ValueError(
            f"ic_series {ic.shape} and turnover_series {tau.shape} must match"
        )
    if isinstance(min_periods, bool) or not isinstance(min_periods, (int, np.integer)):
        raise TypeError("min_periods must be an integer")
    if min_periods < 2:
        raise ValueError("min_periods must be at least 2")

    T, F = ic.shape
    out = np.full(F, np.nan, dtype=np.float64)
    for f in range(F):
        ic_finite = ic[np.isfinite(ic[:, f]), f]
        tau_finite = tau[np.isfinite(tau[:, f]), f]
        if ic_finite.size < min_periods or tau_finite.size < 2:
            continue
        mean_ic = float(np.mean(ic_finite))
        mean_tau = float(np.mean(tau_finite))
        if not np.isfinite(mean_tau) or mean_tau <= 0:
            continue
        out[f] = mean_ic / mean_tau
    return out
