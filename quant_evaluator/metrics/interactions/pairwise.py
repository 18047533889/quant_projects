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
import hashlib

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
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    universe_ref: str = ""
    sample_ref: str = ""

    def __post_init__(self) -> None:
        if not self.window_ref:
            raise ValueError("window_ref is required")
        if (isinstance(self.start_index, bool) or isinstance(self.end_index, bool) or
                not isinstance(self.start_index, int) or not isinstance(self.end_index, int) or
                self.start_index < 0 or self.start_index >= self.end_index):
            raise ValueError("window requires valid non-boolean [start_index, end_index)")
        if not self.universe_ref or not self.sample_ref:
            raise ValueError("window universe_ref and sample_ref are required")
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

    @property
    def window_identity(self) -> str:
        payload = "\x1f".join((self.universe_ref, self.sample_ref,
                               str(self.start_index), str(self.end_index)))
        return "qe-window:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        if (isinstance(self.pair_count, bool) or isinstance(self.n_days, bool) or
                not isinstance(self.pair_count, (int, np.integer)) or
                not isinstance(self.n_days, (int, np.integer)) or
                self.pair_count < 0 or self.n_days < 0):
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
        if len({window.window_identity for window in self.windows}) != len(self.windows):
            raise ValueError("pairwise windows must have unique time/sample identities")
        if any(window.universe_ref != self.universe_ref or window.sample_ref != self.sample_ref
               for window in self.windows):
            raise ValueError("pairwise windows must match parent universe/sample identity")


def _constant_input(values: np.ndarray) -> bool:
    # np.std may be nonzero for identical tiny floats due to rounding.
    return values.size == 0 or values.min() == values.max() or np.std(values) == 0


def _measure_pair(x: np.ndarray, y: np.ndarray, method: str, min_obs: int):
    valid_mask = np.isfinite(x) & np.isfinite(y)
    pair_count = int(valid_mask.sum())
    if pair_count < min_obs:
        return None, PairwiseMeasurementStatus.INSUFFICIENT_DATA, valid_mask
    x_valid, y_valid = x[valid_mask], y[valid_mask]
    if _constant_input(x_valid) or _constant_input(y_valid):
        return None, PairwiseMeasurementStatus.CONSTANT_INPUT, valid_mask
    if method == "pearson":
        corr = float(np.corrcoef(x_valid, y_valid)[0, 1])
    else:
        from scipy import stats
        corr = float(stats.spearmanr(x_valid, y_valid)[0])
    if not np.isfinite(corr):
        return None, PairwiseMeasurementStatus.INVALID, valid_mask
    return corr, PairwiseMeasurementStatus.COMPUTED, valid_mask


_EPS = float(np.finfo(float).eps)
_SUSPICION_FACTOR = 1e4


def _raw_corr(n, sx, sy, sxx, syy, sxy):
    """Pearson correlation from masked raw sums of pre-centered series."""
    with np.errstate(invalid="ignore", divide="ignore"):
        num = n * sxy - sx * sy
        den = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
        return np.where(den > 0, num / den, np.nan)


def _var_from_sums(n, sx, sxx):
    with np.errstate(invalid="ignore", divide="ignore"):
        return sxx / np.maximum(n, 1.0) - (sx / np.maximum(n, 1.0)) ** 2


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
    if method == "spearman":
        return _compute_pairwise_artifacts_reference(
            factor_batch, method=method, min_obs=min_obs, window_ref=window_ref,
            universe_ref=universe_ref, sample_ref=sample_ref, windows=windows,
            uncertainty_min_days=uncertainty_min_days, hac_max_lag=hac_max_lag,
            candidate_pairs=candidate_pairs, max_pairs=max_pairs,
        )
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
                    # Vectorized daily correlations (Pearson). Per-day pairwise
                    # masks are fixed across the window; series pre-centered by
                    # their window-wide mean. Days whose measurement is
                    # numerically delicate fall back to the verbatim per-day
                    # legacy call so the appended sequence is identical.
                    days = end - start
                    fxD = np.isfinite(wx)
                    fyD = np.isfinite(wy)
                    validD = fxD & fyD  # (days, N)
                    n_day = validD.sum(axis=1)
                    fx_flat = fxD.reshape(-1)
                    fy_flat = fyD.reshape(-1)
                    x_flat = wx.reshape(-1)
                    y_flat = wy.reshape(-1)
                    mu_x = float(np.mean(x_flat[fx_flat])) if fx_flat.any() else 0.0
                    mu_y = float(np.mean(y_flat[fy_flat])) if fy_flat.any() else 0.0
                    xc = np.where(fx_flat, x_flat - mu_x, 0.0)
                    yc = np.where(fy_flat, y_flat - mu_y, 0.0)
                    sprime_x = float(np.max(np.abs(xc))) if fx_flat.any() else 0.0
                    sprime_y = float(np.max(np.abs(yc))) if fy_flat.any() else 0.0
                    thr_x = _SUSPICION_FACTOR * _EPS * max(1.0, sprime_x) ** 2
                    thr_y = _SUSPICION_FACTOR * _EPS * max(1.0, sprime_y) ** 2
                    xz = np.where(validD.reshape(-1), xc, 0.0)
                    yz = np.where(validD.reshape(-1), yc, 0.0)
                    xzD = xz.reshape(days, asset_count)
                    yzD = yz.reshape(days, asset_count)
                    n_dayf = n_day.astype(np.float64)
                    sx = xzD.sum(axis=1)
                    sy = yzD.sum(axis=1)
                    sxx = (xzD * xzD).sum(axis=1)
                    syy = (yzD * yzD).sum(axis=1)
                    sxy = (xzD * yzD).sum(axis=1)
                    dcorr = _raw_corr(n_dayf, sx, sy, sxx, syy, sxy)
                    var_x = _var_from_sums(n_dayf, sx, sxx)
                    var_y = _var_from_sums(n_dayf, sy, syy)
                    with np.errstate(invalid="ignore"):
                        d_suspicious = (
                            (n_day < 2)
                            | ~(var_x > thr_x)
                            | ~(var_y > thr_y)
                            | ~np.isfinite(dcorr)
                        )
                    daily_corr = []
                    for day in range(end-start):
                        if d_suspicious[day]:
                            dcorr_ref, dstatus, _ = _measure_pair(wx[day], wy[day], method, 2)
                            if dstatus is PairwiseMeasurementStatus.COMPUTED:
                                daily_corr.append(dcorr_ref)
                        else:
                            daily_corr.append(float(dcorr[day]))
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
                    start_index=start, end_index=end,
                    universe_ref=universe_ref, sample_ref=sample_ref,
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

    # NOTE: measured at T=1250/N=300/F=5, the per-pair ``_measure_pair`` loop
    # (BLAS-backed corrcoef over the full flattened column) beats a masked
    # raw-moment vectorization (~75 ms vs ~88 ms), so this function keeps the
    # legacy loop verbatim.

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

    if method == "spearman" or T < window or window < 1:
        # Lean windows / rank-based path: verbatim legacy loop.
        for t in range(window - 1, T):
            window_start = max(0, t - window + 1)
            window_values = values[window_start:t+1, :, :]  # (W, N, F)

            window_flat = window_values.reshape(-1, F)

            if factor_batch.validity is not None:
                window_validity = factor_batch.validity[window_start:t+1, :, :]
                window_validity_flat = window_validity.reshape(-1, F)
                window_flat = np.where(window_validity_flat, window_flat, np.nan)

            for i in range(F):
                for j in range(i, F):
                    x = window_flat[:, i]
                    y = window_flat[:, j]

                    valid_mask = np.isfinite(x) & np.isfinite(y)
                    x_valid = x[valid_mask]
                    y_valid = y[valid_mask]

                    if len(x_valid) < min_obs:
                        continue

                    if _constant_input(x_valid) or _constant_input(y_valid):
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

    # Vectorized Pearson path. Windows are contiguous blocks in the flattened
    # (time-major) array, so per-window sums come from cumulative sums; the
    # per-window pairwise mask is fixed per factor pair (NaN pattern does not
    # move across windows). Series are pre-centered by their global mean so
    # the raw-moment formula stays accurate; suspicious (near-constant)
    # windows fall back to the verbatim legacy computation.
    if factor_batch.validity is not None:
        values = np.where(factor_batch.validity, values, np.nan)

    n_win = T - window + 1
    w_elems = window * N
    starts_flat = np.arange(n_win) * N
    ends_flat = starts_flat + w_elems

    def _window_sums(col):
        finite = np.isfinite(col)
        finite_any = bool(finite.any())
        mu = float(np.mean(col[finite])) if finite_any else 0.0
        centered = np.where(finite, col - mu, 0.0)
        c1 = np.concatenate([[0.0], np.cumsum(centered)])
        c2 = np.concatenate([[0.0], np.cumsum(centered * centered)])
        cf = np.concatenate([[0.0], np.cumsum(finite.astype(np.float64))])
        sx = c1[ends_flat] - c1[starts_flat]
        sxx = c2[ends_flat] - c2[starts_flat]
        sprime = float(np.max(np.abs(centered))) if finite_any else 0.0
        return centered, finite, sx, sxx, sprime

    for i in range(F):
        xc, fx, sx_all, sxx_all, sprime_x = _window_sums(values[:, :, i].reshape(-1))
        thr_x = _SUSPICION_FACTOR * _EPS * max(1.0, sprime_x) ** 2
        for j in range(i, F):
            yc, fy, sy_all, syy_all, sprime_y = _window_sums(values[:, :, j].reshape(-1))
            thr_y = _SUSPICION_FACTOR * _EPS * max(1.0, sprime_y) ** 2
            fpair = fx & fy
            xz = np.where(fpair, xc, 0.0)
            yz = np.where(fpair, yc, 0.0)
            c1 = np.concatenate([[0.0], np.cumsum(xz)])
            c2 = np.concatenate([[0.0], np.cumsum(xz * xz)])
            c3 = np.concatenate([[0.0], np.cumsum(yz)])
            c4 = np.concatenate([[0.0], np.cumsum(yz * yz)])
            c5 = np.concatenate([[0.0], np.cumsum(xz * yz)])
            cf = np.concatenate([[0.0], np.cumsum(fpair.astype(np.float64))])
            n = cf[ends_flat] - cf[starts_flat]
            sx = c1[ends_flat] - c1[starts_flat]
            sxx = c2[ends_flat] - c2[starts_flat]
            sy = c3[ends_flat] - c3[starts_flat]
            syy = c4[ends_flat] - c4[starts_flat]
            sxy = c5[ends_flat] - c5[starts_flat]
            corr = _raw_corr(n, sx, sy, sxx, syy, sxy)
            var_x = _var_from_sums(n, sx, sxx)
            var_y = _var_from_sums(n, sy, syy)
            with np.errstate(invalid="ignore"):
                suspicious = (
                    (n < min_obs)
                    | ~(var_x > thr_x)
                    | ~(var_y > thr_y)
                    | ~np.isfinite(corr)
                )
            # Eligible window ends: t = window-1 .. T-1 map to windows 0..n_win-1.
            ts = np.arange(window - 1, T)
            ok = ~suspicious
            rolling_corr[ts[ok], i, j] = corr[ok]
            rolling_corr[ts[ok], j, i] = corr[ok]
            for w_idx in np.flatnonzero(suspicious):
                t = window - 1 + w_idx
                window_start = max(0, t - window + 1)
                window_flat = values[window_start:t+1, :, :].reshape(-1, F)
                x = window_flat[:, i]
                y = window_flat[:, j]
                valid_mask = np.isfinite(x) & np.isfinite(y)
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]
                if len(x_valid) < min_obs:
                    continue
                if _constant_input(x_valid) or _constant_input(y_valid):
                    continue  # stays NaN
                corr_ref = np.corrcoef(x_valid, y_valid)[0, 1]
                rolling_corr[t, i, j] = corr_ref
                rolling_corr[t, j, i] = corr_ref

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
        # Vectorized per-day correlation across all factor pairs. Per-pair
        # (pairwise) deletion semantics are preserved: each (i, j) pair uses
        # only assets finite in both columns. Series are pre-centered by
        # per-(day, factor) means so the raw-moment formula stays accurate;
        # suspicious (near-constant) cells fall back to the verbatim legacy
        # computation.
        if factor_batch.validity is not None:
            values = np.where(factor_batch.validity, values, np.nan)
        finite = np.isfinite(values)  # (T, N, F)
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(values, axis=1)  # (T, F)
        xc = np.where(finite, values - mu[:, None, :], 0.0)
        sprime = np.max(np.abs(xc), axis=1)  # (T, F)
        thr = _SUSPICION_FACTOR * _EPS * np.maximum(1.0, sprime) ** 2  # (T, F)

        corr_sum = np.zeros((F, F), dtype=np.float64)
        corr_count = np.zeros((F, F), dtype=np.int64)

        for t in range(T):
            fin = finite[t]  # (N, F)
            pair = fin[:, :, None] & fin[:, None, :]  # (N, F, F)
            n_ij = pair.sum(axis=0).astype(np.float64)
            xz = xc[t]  # (N, F) centered, zero-filled
            xz3 = xz[:, :, None]
            yz3 = xz[:, None, :]
            pairz = pair.astype(np.float64)
            sx = np.sum(xz3 * pairz, axis=0)
            sy = np.sum(yz3 * pairz, axis=0)
            sxx = np.sum(xz3 * xz3 * pairz, axis=0)
            syy = np.sum(yz3 * yz3 * pairz, axis=0)
            sxy = np.sum(xz3 * yz3 * pairz, axis=0)
            corr = _raw_corr(n_ij, sx, sy, sxx, syy, sxy)
            var_x = _var_from_sums(n_ij, sx, sxx)
            var_y = _var_from_sums(n_ij, sy, syy)
            with np.errstate(invalid="ignore"):
                suspicious = (
                    (n_ij < min_obs)
                    | ~(var_x > thr[t][:, None])
                    | ~(var_y > thr[t][None, :])
                    | ~np.isfinite(corr)
                )
            ok = ~suspicious & np.isfinite(corr) & (np.arange(F)[:, None] <= np.arange(F)[None, :])
            corr_sum[ok] += corr[ok]
            corr_count[ok] += 1
            off_diag = ok & (np.arange(F)[:, None] < np.arange(F)[None, :])
            if off_diag.any():
                i_idx, j_idx = np.nonzero(off_diag)
                corr_sum[j_idx, i_idx] += corr[off_diag]
                corr_count[j_idx, i_idx] += 1
            if suspicious.any():
                values_t = values[t, :, :]
                for i, j in np.argwhere(suspicious & (np.arange(F)[:, None] <= np.arange(F)[None, :])):
                    x = values_t[:, i]
                    y = values_t[:, j]
                    valid_mask = np.isfinite(x) & np.isfinite(y)
                    x_valid = x[valid_mask]
                    y_valid = y[valid_mask]
                    if len(x_valid) < min_obs:
                        continue
                    if _constant_input(x_valid) or _constant_input(y_valid):
                        continue
                    c = np.corrcoef(x_valid, y_valid)[0, 1]
                    if np.isfinite(c):
                        corr_sum[i, j] += c
                        corr_count[i, j] += 1
                        if i != j:
                            corr_sum[j, i] += c
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


def _compute_pairwise_artifacts_reference(
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
    """Verbatim legacy oracle for equivalence testing."""
    values = np.asarray(factor_batch.values, dtype=float)
    if factor_batch.validity is not None:
        values = np.where(factor_batch.validity, values, np.nan)
    t_count, asset_count, factor_count = values.shape
    window_slices = dict(windows or {window_ref: (0, t_count)})
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
                    start_index=start, end_index=end,
                    universe_ref=universe_ref, sample_ref=sample_ref,
                ))
            artifacts.append(PairwiseCorrelationArtifact(
                factor_batch.factor_ids[i], factor_batch.factor_ids[j], corr, status,
                int(flat_mask.sum()), sum(v > 0 for v in daily_counts), daily_counts,
                method, window_ref, universe_ref, sample_ref, tuple(window_evidence),
            ))
    return tuple(artifacts)


def _compute_pairwise_correlation_reference(
    factor_batch: FactorBatch,
    method: str = "pearson",
    min_obs: int = 30,
    max_output_elements: int = 10_000_000,
) -> np.ndarray:
    """Verbatim legacy oracle for equivalence testing."""
    values = factor_batch.values  # (T, N, F)
    T, N, F = values.shape
    if F * F > max_output_elements:
        raise ValueError("dense pairwise output budget exceeded; use compute_pairwise_artifacts")

    values_flat = values.reshape(-1, F)

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


def _compute_rolling_pairwise_correlation_reference(
    factor_batch: FactorBatch,
    window: int,
    method: str = "pearson",
    min_obs: int = 30,
    max_output_elements: int = 10_000_000,
) -> np.ndarray:
    """Verbatim legacy oracle for equivalence testing."""
    values = factor_batch.values  # (T, N, F)
    T, N, F = values.shape
    if T * F * F > max_output_elements:
        raise ValueError("dense rolling pairwise output budget exceeded; use sparse windows")

    rolling_corr = np.full((T, F, F), np.nan, dtype=np.float64)

    for t in range(window - 1, T):
        window_start = max(0, t - window + 1)
        window_values = values[window_start:t+1, :, :]  # (W, N, F)

        window_flat = window_values.reshape(-1, F)

        if factor_batch.validity is not None:
            window_validity = factor_batch.validity[window_start:t+1, :, :]
            window_validity_flat = window_validity.reshape(-1, F)
            window_flat = np.where(window_validity_flat, window_flat, np.nan)

        for i in range(F):
            for j in range(i, F):
                x = window_flat[:, i]
                y = window_flat[:, j]

                valid_mask = np.isfinite(x) & np.isfinite(y)
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]

                if len(x_valid) < min_obs:
                    continue

                if _constant_input(x_valid) or _constant_input(y_valid):
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


def _compute_correlation_matrix_reference(
    factor_batch: FactorBatch,
    cross_sectional: bool = False,
    method: str = "pearson",
    min_obs: int = 10,
    max_output_elements: int = 10_000_000,
) -> np.ndarray:
    """Verbatim legacy oracle for equivalence testing (cross-sectional path)."""
    values = factor_batch.values  # (T, N, F)
    T, N, F = values.shape

    if cross_sectional:
        corr_sum = np.zeros((F, F), dtype=np.float64)
        corr_count = np.zeros((F, F), dtype=np.int64)

        for t in range(T):
            values_t = values[t, :, :]  # (N, F)

            if factor_batch.validity is not None:
                validity_t = factor_batch.validity[t, :, :]
                values_t = np.where(validity_t, values_t, np.nan)

            for i in range(F):
                for j in range(i, F):
                    x = values_t[:, i]
                    y = values_t[:, j]

                    valid_mask = np.isfinite(x) & np.isfinite(y)
                    x_valid = x[valid_mask]
                    y_valid = y[valid_mask]

                    if len(x_valid) < min_obs:
                        continue

                    if _constant_input(x_valid) or _constant_input(y_valid):
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

    return _compute_pairwise_correlation_reference(
        factor_batch, method=method, min_obs=min_obs,
        max_output_elements=max_output_elements,
    )
