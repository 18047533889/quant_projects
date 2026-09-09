"""
Pairwise factor correlation analysis.

Computes correlations between factor pairs over time to detect redundancy
and track evolving relationships.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Optional, Tuple
import numpy as np
from itertools import combinations

from quant_evaluator.contracts.factor_batch import FactorBatch


def iter_bounded_pairs(
    factor_count: int,
    candidate_pairs: Optional[Iterable[Tuple[int, int]]] = None,
    max_pairs: int = 10_000,
):
    """Yield validated sparse pair identities without materializing F squared."""
    if isinstance(max_pairs, bool) or not isinstance(max_pairs, (int, np.integer)) or max_pairs < 1:
        raise ValueError("candidate pair budget max_pairs must be a positive integer")
    if candidate_pairs is None:
        if factor_count * (factor_count - 1) // 2 > max_pairs:
            raise ValueError("candidate pair budget exceeded; provide a bounded candidate_pairs plan")
        pairs = combinations(range(factor_count), 2)
    else:
        pairs = iter(candidate_pairs)
    seen = set()
    for pair_number, pair in enumerate(pairs, start=1):
        if pair_number > max_pairs:
            raise ValueError("candidate pair budget exceeded")
        if (not isinstance(pair, tuple) or len(pair) != 2):
            raise ValueError("candidate pairs must be two-item tuples")
        i, j = pair
        if not (isinstance(i, (int, np.integer)) and isinstance(j, (int, np.integer))
                and 0 <= i < j < factor_count) or (i, j) in seen:
            raise ValueError("candidate pairs must be unique ordered in-range indices")
        seen.add((i, j))
        yield int(i), int(j)


class PairwiseMeasurementStatus(Enum):
    """Typed outcome of the authoritative pairwise-finite measurement."""

    COMPUTED = "computed"
    INSUFFICIENT_DATA = "insufficient_data"
    CONSTANT_INPUT = "constant_input"
    INVALID = "invalid"


@dataclass(frozen=True)
class PairwiseWindowEvidence:
    window_ref: str
    correlation: Optional[float]
    status: PairwiseMeasurementStatus
    pair_count: int
    n_days: int
    daily_pair_counts: Tuple[int, ...]
    confidence_interval: Tuple[Optional[float], Optional[float]]
    uncertainty_scale: str = "raw_correlation_hac_daily_corr"

    def __post_init__(self) -> None:
        if not self.window_ref:
            raise ValueError("window_ref is required")
        if self.uncertainty_scale != "raw_correlation_hac_daily_corr":
            raise ValueError("uncertainty_scale must declare raw-correlation HAC semantics")
        object.__setattr__(self, "daily_pair_counts", tuple(self.daily_pair_counts))
        if sum(self.daily_pair_counts) != self.pair_count:
            raise ValueError("daily_pair_counts must sum to pair_count")
        lo, hi = self.confidence_interval
        if self.status is PairwiseMeasurementStatus.COMPUTED:
            if self.correlation is None or not np.isfinite(self.correlation):
                raise ValueError("computed window requires finite correlation")
            if (lo is None) != (hi is None) or (lo is not None and lo > hi):
                raise ValueError("confidence interval must be ordered or wholly unavailable")
        elif self.correlation is not None or lo is not None or hi is not None:
            raise ValueError("non-computed window cannot carry value or uncertainty")


@dataclass(frozen=True)
class PairwiseCorrelationArtifact:
    """QE-owned signed pairwise result with overlap and comparison identity."""

    factor_id_a: str
    factor_id_b: str
    correlation: Optional[float]
    status: PairwiseMeasurementStatus
    pair_count: int
    n_days: int
    daily_pair_counts: Tuple[int, ...]
    method: str
    window_ref: str
    universe_ref: str
    sample_ref: str
    windows: Tuple[PairwiseWindowEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not self.factor_id_a or not self.factor_id_b or self.factor_id_a == self.factor_id_b:
            raise ValueError("pairwise artifact requires two distinct factor IDs")
        if self.method not in ("pearson", "spearman"):
            raise ValueError("method must be 'pearson' or 'spearman'")
        if not self.window_ref or not self.universe_ref or not self.sample_ref:
            raise ValueError("window_ref, universe_ref, and sample_ref are required")
        if self.pair_count < 0 or self.n_days < 0:
            raise ValueError("pair_count and n_days must be non-negative")
        object.__setattr__(self, "daily_pair_counts", tuple(self.daily_pair_counts))
        if sum(self.daily_pair_counts) != self.pair_count:
            raise ValueError("daily_pair_counts must sum to pair_count")
        if sum(count > 0 for count in self.daily_pair_counts) != self.n_days:
            raise ValueError("n_days must count days with at least one valid pair")
        if self.status is PairwiseMeasurementStatus.COMPUTED:
            if self.correlation is None or not np.isfinite(self.correlation):
                raise ValueError("COMPUTED pairwise evidence requires finite correlation")
            if not -1.0 <= float(self.correlation) <= 1.0:
                raise ValueError("correlation must be in [-1, 1]")
        elif self.correlation is not None:
            raise ValueError("non-COMPUTED pairwise evidence must not carry a numeric value")
        object.__setattr__(self, "windows", tuple(self.windows))
        if len({window.window_ref for window in self.windows}) != len(self.windows):
            raise ValueError("pairwise window refs must be unique")


def _measure_pair(x: np.ndarray, y: np.ndarray, method: str, min_obs: int):
    valid_mask = np.isfinite(x) & np.isfinite(y)
    pair_count = int(valid_mask.sum())
    if pair_count < min_obs:
        return None, PairwiseMeasurementStatus.INSUFFICIENT_DATA, valid_mask
    x_valid, y_valid = x[valid_mask], y[valid_mask]
    if np.std(x_valid) == 0 or np.std(y_valid) == 0:
        return None, PairwiseMeasurementStatus.CONSTANT_INPUT, valid_mask
    if method == "pearson":
        corr = float(np.corrcoef(x_valid, y_valid)[0, 1])
    else:
        from scipy import stats
        corr = float(stats.spearmanr(x_valid, y_valid)[0])
    if not np.isfinite(corr):
        return None, PairwiseMeasurementStatus.INVALID, valid_mask
    return corr, PairwiseMeasurementStatus.COMPUTED, valid_mask


def compute_pairwise_artifacts(
    factor_batch: FactorBatch,
    *,
    method: str = "pearson",
    min_obs: int = 30,
    window_ref: str,
    universe_ref: str,
    sample_ref: str,
    windows: Optional[Mapping[str, Tuple[int, int]]] = None,
    uncertainty_min_days: int = 20,
    hac_max_lag: int = 5,
    candidate_pairs: Optional[Iterable[Tuple[int, int]]] = None,
    max_pairs: int = 10_000,
) -> Tuple[PairwiseCorrelationArtifact, ...]:
    """Return typed sparse pair evidence; never translate unknown/NaN to zero."""
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown method: {method}. Must be 'pearson' or 'spearman'")
    if min_obs < 1:
        raise ValueError("min_obs must be >= 1")
    values = np.asarray(factor_batch.values, dtype=float)
    if factor_batch.validity is not None:
        values = np.where(factor_batch.validity, values, np.nan)
    t_count, asset_count, factor_count = values.shape
    window_slices = dict(windows or {window_ref: (0, t_count)})
    for ref, bounds in window_slices.items():
        if (not ref or len(bounds) != 2 or bounds[0] < 0 or
                bounds[0] >= bounds[1] or bounds[1] > t_count):
            raise ValueError("every pairwise window must be a non-empty ref and valid [start, end) slice")
    artifacts = []
    for i, j in iter_bounded_pairs(factor_count, candidate_pairs, max_pairs):
            x, y = values[:, :, i].reshape(-1), values[:, :, j].reshape(-1)
            corr, status, flat_mask = _measure_pair(x, y, method, min_obs)
            daily_counts = tuple(int(v) for v in flat_mask.reshape(t_count, asset_count).sum(axis=1))
            window_evidence = []
            for ref, (start, end) in window_slices.items():
                wx = values[start:end, :, i]
                wy = values[start:end, :, j]
                wcorr, wstatus, wmask = _measure_pair(
                    wx.reshape(-1), wy.reshape(-1), method, min_obs,
                )
                wcounts = tuple(int(v) for v in wmask.reshape(end-start, asset_count).sum(axis=1))
                ci = (None, None)
                if wstatus is PairwiseMeasurementStatus.COMPUTED:
                    daily_corr = []
                    for day in range(end-start):
                        dcorr, dstatus, _ = _measure_pair(wx[day], wy[day], method, 2)
                        if dstatus is PairwiseMeasurementStatus.COMPUTED:
                            daily_corr.append(dcorr)
                    if len(daily_corr) >= uncertainty_min_days:
                        from quant_evaluator.metrics.statistical_evidence import build_hac_evidence
                        lag = min(hac_max_lag, len(daily_corr) - 1)
                        hac = build_hac_evidence(
                            daily_corr, max_lag=lag,
                            min_periods=uncertainty_min_days,
                        )
                        if (hac.status == "VALID" and hac.standard_error is not None
                                and np.isfinite(hac.standard_error)):
                            radius = 1.959963984540054 * float(hac.standard_error)
                            ci = (max(-1.0, float(wcorr) - radius),
                                  min(1.0, float(wcorr) + radius))
                window_evidence.append(PairwiseWindowEvidence(
                    ref, wcorr, wstatus, int(wmask.sum()),
                    sum(v > 0 for v in wcounts), wcounts, ci,
                ))
            artifacts.append(PairwiseCorrelationArtifact(
                factor_batch.factor_ids[i], factor_batch.factor_ids[j], corr, status,
                int(flat_mask.sum()), sum(v > 0 for v in daily_counts), daily_counts,
                method, window_ref, universe_ref, sample_ref, tuple(window_evidence),
            ))
    return tuple(artifacts)


def compute_pairwise_correlation(
    factor_batch: FactorBatch,
    method: str = "pearson",
    min_obs: int = 30,
    max_output_elements: int = 10_000_000,
) -> np.ndarray:
    """
    Compute pairwise correlation matrix between factors across entire time period.

    Args:
        factor_batch: Input factor batch (T, N, F)
        method: "pearson" or "spearman"
        min_obs: Minimum observations required

    Returns:
        Correlation matrix of shape (F, F)
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown method: {method}. Must be 'pearson' or 'spearman'")

    values = factor_batch.values  # (T, N, F)
    T, N, F = values.shape
    if F * F > max_output_elements:
        raise ValueError("dense pairwise output budget exceeded; use compute_pairwise_artifacts")

    # Reshape to (T*N, F) for correlation computation
    values_flat = values.reshape(-1, F)

    # Apply validity mask if present
    if factor_batch.validity is not None:
        validity_flat = factor_batch.validity.reshape(-1, F)
        values_flat = np.where(validity_flat, values_flat, np.nan)

    corr_matrix = np.full((F, F), np.nan, dtype=np.float64)

    for i in range(F):
        for j in range(i, F):
            x = values_flat[:, i]
            y = values_flat[:, j]

            corr, status, _ = _measure_pair(x, y, method, min_obs)
            if status is PairwiseMeasurementStatus.INSUFFICIENT_DATA:
                continue
            if status is PairwiseMeasurementStatus.CONSTANT_INPUT:
                corr_matrix[i, j] = np.nan
                corr_matrix[j, i] = corr_matrix[i, j]
                continue
            if status is not PairwiseMeasurementStatus.COMPUTED:
                continue
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    return corr_matrix


def compute_rolling_pairwise_correlation(
    factor_batch: FactorBatch,
    window: int,
    method: str = "pearson",
    min_obs: int = 30,
    max_output_elements: int = 10_000_000,
) -> np.ndarray:
    """
    Compute rolling pairwise correlation over time.

    Args:
        factor_batch: Input factor batch (T, N, F)
        window: Rolling window size in time periods
        method: "pearson" or "spearman"
        min_obs: Minimum observations per window

    Returns:
        Rolling correlation array of shape (T, F, F)
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown method: {method}. Must be 'pearson' or 'spearman'")

    values = factor_batch.values  # (T, N, F)
    T, N, F = values.shape
    if T * F * F > max_output_elements:
        raise ValueError("dense rolling pairwise output budget exceeded; use sparse windows")

    rolling_corr = np.full((T, F, F), np.nan, dtype=np.float64)

    for t in range(window - 1, T):
        window_start = max(0, t - window + 1)
        window_values = values[window_start:t+1, :, :]  # (W, N, F)

        # Reshape for correlation
        window_flat = window_values.reshape(-1, F)

        # Apply validity mask if present
        if factor_batch.validity is not None:
            window_validity = factor_batch.validity[window_start:t+1, :, :]
            window_validity_flat = window_validity.reshape(-1, F)
            window_flat = np.where(window_validity_flat, window_flat, np.nan)

        # Compute correlation for this window
        for i in range(F):
            for j in range(i, F):
                x = window_flat[:, i]
                y = window_flat[:, j]

                valid_mask = np.isfinite(x) & np.isfinite(y)
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]

                if len(x_valid) < min_obs:
                    continue

                if np.std(x_valid) == 0 or np.std(y_valid) == 0:
                    rolling_corr[t, i, j] = np.nan
                    rolling_corr[t, j, i] = rolling_corr[t, i, j]
                    continue

                if method == "pearson":
                    corr = np.corrcoef(x_valid, y_valid)[0, 1]
                else:
                    from scipy import stats
                    corr, _ = stats.spearmanr(x_valid, y_valid)

                rolling_corr[t, i, j] = corr
                rolling_corr[t, j, i] = corr

    return rolling_corr


def compute_correlation_matrix(
    factor_batch: FactorBatch,
    cross_sectional: bool = False,
    method: str = "pearson",
    min_obs: int = 10,
    max_output_elements: int = 10_000_000,
) -> np.ndarray:
    """
    Compute correlation matrix with configurable aggregation.

    Args:
        factor_batch: Input factor batch (T, N, F)
        cross_sectional: If True, compute average cross-sectional correlation
                        If False, compute time-series correlation (default)
        method: "pearson" or "spearman"
        min_obs: Minimum observations required

    Returns:
        Correlation matrix of shape (F, F)
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown method: {method}. Must be 'pearson' or 'spearman'")

    values = factor_batch.values  # (T, N, F)
    T, N, F = values.shape
    if F * F > max_output_elements:
        raise ValueError("dense correlation output budget exceeded; use compute_pairwise_artifacts")

    if cross_sectional:
        # Incremental sum/count avoids materializing a T x F x F stack.
        corr_sum = np.zeros((F, F), dtype=np.float64)
        corr_count = np.zeros((F, F), dtype=np.int64)

        for t in range(T):
            values_t = values[t, :, :]  # (N, F)

            # Apply validity mask
            if factor_batch.validity is not None:
                validity_t = factor_batch.validity[t, :, :]
                values_t = np.where(validity_t, values_t, np.nan)

            # Compute correlation for this time period
            for i in range(F):
                for j in range(i, F):
                    x = values_t[:, i]
                    y = values_t[:, j]

                    valid_mask = np.isfinite(x) & np.isfinite(y)
                    x_valid = x[valid_mask]
                    y_valid = y[valid_mask]

                    if len(x_valid) < min_obs:
                        continue

                    if np.std(x_valid) == 0 or np.std(y_valid) == 0:
                        continue

                    if method == "pearson":
                        corr = np.corrcoef(x_valid, y_valid)[0, 1]
                    else:
                        from scipy import stats
                        corr, _ = stats.spearmanr(x_valid, y_valid)

                    if np.isfinite(corr):
                        corr_sum[i, j] += corr
                        corr_count[i, j] += 1
                        if i != j:
                            corr_sum[j, i] += corr
                            corr_count[j, i] += 1
        avg_corr = np.divide(
            corr_sum, corr_count,
            out=np.full((F, F), np.nan, dtype=np.float64), where=corr_count > 0,
        )

        return avg_corr

    else:
        # Time-series correlation (default behavior)
        return compute_pairwise_correlation(
            factor_batch, method=method, min_obs=min_obs,
            max_output_elements=max_output_elements,
        )
