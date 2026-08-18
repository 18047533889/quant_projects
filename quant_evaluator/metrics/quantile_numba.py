"""
High-performance quantile operations using Numba JIT compilation.

Achieves 5-10x speedup through:
1. JIT-compiled tight loops (eliminates Python overhead)
2. Vectorized numpy operations where possible
3. Optimized memory access patterns

QE-Q-P0-002: NumPy-Numba boundary parity
QE-Q-P0-004: Numba fastmath=False for NaN/Inf correctness
"""

from typing import Tuple
import numpy as np

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args or callable(args[0]) else decorator

    def prange(*args, **kwargs):
        return range(*args, **kwargs)

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.label_panel import normalize_label_panel


@jit(nopython=True, cache=True)  # QE-Q-P0-004: Remove fastmath for NaN/Inf correctness
def _assign_quantiles_jit(values: np.ndarray, n_quantiles: int, policy: str) -> np.ndarray:
    """
    JIT-compiled quantile assignment with tie policy enforcement.

    QE-Q-P0-001: Enforces tie-breaking policy.
    QE-Q-P0-002: Boundary logic matches NumPy searchsorted.
    QE-Q-P0-004: No fastmath to ensure NaN/Inf handling matches NumPy.

    Args:
        values: (T, N, F) array
        n_quantiles: Number of quantiles
        policy: Tie-breaking policy ('min' or 'max')

    Returns:
        Quantile assignments (T, N, F), -1 for NaN
    """
    T, N, F = values.shape
    quantiles = np.full((T, N, F), -1, dtype=np.int32)

    use_min_policy = (policy == "min")

    # Process each time-factor pair
    for t in range(T):
        for f in range(F):
            # Count valid values
            n_valid = 0
            for n in range(N):
                if not np.isnan(values[t, n, f]) and not np.isinf(values[t, n, f]):
                    n_valid += 1

            if n_valid < n_quantiles:
                continue

            # Extract valid values and track original indices
            v_valid = np.empty(n_valid, dtype=np.float64)
            indices_valid = np.empty(n_valid, dtype=np.int32)
            idx = 0
            for n in range(N):
                val = values[t, n, f]
                if not np.isnan(val) and not np.isinf(val):
                    v_valid[idx] = val
                    indices_valid[idx] = n
                    idx += 1

            # Sort to compute quantile boundaries
            v_sorted = np.sort(v_valid)

            # Compute quantile boundaries using np.percentile positions
            # Match NumPy's percentile linear interpolation
            boundaries = np.empty(n_quantiles - 1, dtype=np.float64)
            percentiles = np.linspace(0.0, 100.0, n_quantiles + 1)

            for q in range(1, n_quantiles):
                pct = percentiles[q]
                # NumPy percentile formula: index = (n-1) * pct/100
                pos = (n_valid - 1) * pct / 100.0
                idx_low = int(np.floor(pos))
                idx_high = int(np.ceil(pos))

                # Clamp to valid range
                idx_low = min(max(idx_low, 0), n_valid - 1)
                idx_high = min(max(idx_high, 0), n_valid - 1)

                # Linear interpolation
                # CRITICAL: When both values are equal, use the value directly
                # to avoid floating point errors from interpolation arithmetic
                if idx_low == idx_high or v_sorted[idx_low] == v_sorted[idx_high]:
                    boundaries[q - 1] = v_sorted[idx_low]
                else:
                    frac = pos - np.floor(pos)
                    boundaries[q - 1] = v_sorted[idx_low] * (1.0 - frac) + v_sorted[idx_high] * frac

            # Assign quantiles - mimic numpy searchsorted behavior
            # searchsorted returns the insertion position to maintain sorted order
            # side='left': insert BEFORE equal values (leftmost position where boundaries[i] >= val)
            # side='right': insert AFTER equal values (leftmost position where boundaries[i] > val)
            for i in range(n_valid):
                val = v_valid[i]
                orig_idx = indices_valid[i]

                # searchsorted: find insertion position in boundaries array
                q_bin = 0

                if use_min_policy:
                    # side='left': find leftmost i where boundaries[i] >= val
                    # This means: insert before any equal values
                    for q in range(n_quantiles - 1):
                        if boundaries[q] >= val:
                            break
                        q_bin += 1
                else:
                    # side='right': find leftmost i where boundaries[i] > val
                    # This means: insert after any equal values
                    for q in range(n_quantiles - 1):
                        if boundaries[q] > val:
                            break
                        q_bin += 1

                quantiles[t, orig_idx, f] = q_bin

    return quantiles


@jit(nopython=True, parallel=True, cache=True)  # QE-Q-P0-004: Remove fastmath
def _compute_quantile_returns_jit(
    values: np.ndarray,
    labels: np.ndarray,
    quantiles: np.ndarray,
    n_quantiles: int,
    min_assets: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled quantile return computation with parallel processing.

    QE-Q-P0-004: No fastmath to preserve NaN/Inf handling correctness.

    Args:
        values: Factor values (T, N, F)
        labels: Forward returns (T, N)
        quantiles: Quantile assignments (T, N, F)
        n_quantiles: Number of quantiles
        min_assets: Minimum assets per quantile

    Returns:
        (quantile_returns, quantile_counts) both shape (T, n_quantiles, F)
    """
    T, N, F = values.shape

    quantile_returns = np.full((T, n_quantiles, F), np.nan, dtype=np.float64)
    quantile_counts = np.zeros((T, n_quantiles, F), dtype=np.int32)

    # Parallel processing over time-factor pairs
    for t in prange(T):
        for f in range(F):
            # Temporary arrays for this (t, f) pair
            q_sums = np.zeros(n_quantiles, dtype=np.float64)
            q_counts = np.zeros(n_quantiles, dtype=np.int32)

            # Aggregate across assets
            for n in range(N):
                q_bin = quantiles[t, n, f]
                label_val = labels[t, n]

                # Check validity
                if q_bin >= 0 and q_bin < n_quantiles:
                    if not np.isnan(label_val) and not np.isinf(label_val):
                        q_sums[q_bin] += label_val
                        q_counts[q_bin] += 1

            # Compute means where we have sufficient assets
            for q in range(n_quantiles):
                quantile_counts[t, q, f] = q_counts[q]
                if q_counts[q] >= min_assets:
                    quantile_returns[t, q, f] = q_sums[q] / q_counts[q]

    return quantile_returns, quantile_counts


def assign_quantiles_numba(
    values: np.ndarray,
    n_quantiles: int = 5,
    method: str = "max",
) -> np.ndarray:
    """
    High-performance quantile assignment using Numba JIT.

    QE-Q-P0-001: Enforces tie-breaking policy.
    QE-Q-P0-002: Boundary logic matches NumPy implementation.

    Args:
        values: Input values shape (T, N) or (T, N, F)
        n_quantiles: Number of quantiles
        method: Tie-breaking policy ('min' or 'max'). Default 'max'.

    Returns:
        Quantile assignments (0 to n_quantiles-1), or -1 for NaN
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is not available. Install with: pip install numba")

    # Validate tie policy
    from quant_evaluator.contracts.quantile_policy import validate_tie_policy
    policy = validate_tie_policy(method)

    if values.ndim == 2:
        values = values[:, :, np.newaxis]

    quantiles = _assign_quantiles_jit(values, n_quantiles, policy.value)

    return quantiles[:, :, 0] if quantiles.shape[2] == 1 else quantiles


def compute_quantile_returns_numba(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    n_quantiles: int = 5,
    min_assets: int = 10,
    method: str = "max",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    High-performance quantile return computation using Numba JIT.

    QE-Q-P0-001: Enforces tie-breaking policy.
    Achieves 5-10x speedup through JIT compilation and parallel processing.

    Args:
        factor_batch: Factor values (T, N, F)
        label_bundle: Forward returns (T, N)
        n_quantiles: Number of quantiles
        min_assets: Minimum assets per quantile
        method: Tie-breaking policy ('min' or 'max'). Default 'max'.

    Returns:
        (quantile_returns, quantile_counts)
        quantile_returns: shape (T, n_quantiles, F)
        quantile_counts: shape (T, n_quantiles, F)
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is not available. Install with: pip install numba")

    from quant_evaluator.contracts.quantile_policy import validate_tie_policy
    policy = validate_tie_policy(method)

    values = factor_batch.values
    if factor_batch.validity is not None:
        values = np.where(factor_batch.validity, values, np.nan)
    labels, label_validity = normalize_label_panel(
        label_bundle, factor_batch.num_assets
    )
    if label_validity is not None:
        labels = np.where(label_validity, labels, np.nan)
    labels = np.ascontiguousarray(labels)

    # Assign quantiles with JIT
    quantiles = _assign_quantiles_jit(values, n_quantiles, policy.value)

    # Compute returns with JIT and parallelization
    quantile_returns, quantile_counts = _compute_quantile_returns_jit(
        values, labels, quantiles, n_quantiles, min_assets
    )

    return quantile_returns, quantile_counts


def is_numba_available() -> bool:
    """Check if Numba is available for JIT compilation."""
    return NUMBA_AVAILABLE
