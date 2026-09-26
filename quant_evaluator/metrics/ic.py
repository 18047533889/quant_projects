"""
Information Coefficient (IC) metrics.

Reference implementation: Pearson correlation and Spearman RankIC with
pairwise-finite filtering and average-tie handling.
"""

from typing import Optional, Tuple
import warnings

import numpy as np
from scipy import stats

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.errors import InsufficientObservations, InvalidContractError
from quant_evaluator.metrics.label_panel import normalize_label_panel


def _pairwise_finite_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute pairwise finite mask for two arrays.

    Args:
        x: First array
        y: Second array (must be broadcastable with x)

    Returns:
        Boolean mask where both x and y are finite
    """
    return np.isfinite(x) & np.isfinite(y)


def _corrcoef_pair_1d(x_valid: np.ndarray, y_valid: np.ndarray, n: int) -> float:
    """
    Bit-exact replication of ``np.corrcoef(x_valid, y_valid)[0, 1]`` for two
    1-D compressed arrays, minus the wrapper overhead of the public
    ``np.corrcoef``/``np.cov`` machinery (atleast_2d/copy checks, dtype
    resolution, diag try/except, complex branches).

    The operation sequence replicates numpy 2.2.6 ``cov`` + ``corrcoef``
    exactly (verified row-by-row by the fuzz checks in
    ``tests/metrics/test_ic_fast_backend.py`` and the bit-identical
    equivalence tests):

        X = float64 stack of (x, y)           # cov: concatenate((x, y), axis=0)
        avg = X.mean(axis=1)                  # average(weights=None) == mean
        X -= avg[:, None]
        c = np.dot(X, X.T.conj())             # BLAS gemm, same layouts
        c *= np.true_divide(1, n - 1)         # ddof=1 (cov default, bias=0)
        c /= sqrt(diag(c))[:, None]           # two sequential divisions
        c /= sqrt(diag(c))[None, :]
        np.clip(c.real, -1, 1, out=c.real)

    Must NOT be re-ordered or "simplified": any change to the op sequence
    can break the bit-identical contract enforced by
    ``tests/metrics/test_daily_ic_vectorized_equivalence.py``.
    """
    X = np.empty((2, n), dtype=np.float64)
    X[0] = x_valid
    X[1] = y_valid
    avg = X.mean(axis=1)
    X -= avg[:, None]
    c = np.dot(X, X.T.conj())
    c *= np.true_divide(1, n - 1)
    d = np.diag(c)
    stddev = np.sqrt(d.real)
    c /= stddev[:, None]
    c /= stddev[None, :]
    np.clip(c.real, -1, 1, out=c.real)
    return c[0, 1]


def _corrcoef_rank_last(rx: np.ndarray, ry: np.ndarray, n: int) -> float:
    """
    Bit-exact replication of ``np.corrcoef(np.column_stack((rx, ry)),
    rowvar=False)[1, 0]`` for average-tie rank vectors, with the wrapper
    overhead of ``np.corrcoef``/``np.cov`` removed.

    Bit-exactness is guaranteed mathematically, not just empirically:

    1. The sum of average-tie ranks of ``n`` items is exactly
       ``n(n+1)/2`` (each tie run averages to the exact mean of its
       consecutive integer positions), every rank is an exact
       half-integer, and every partial sum is a half-integer <=
       ``n(n+1)/2`` — exactly representable in float64.  Hence
       ``X.mean(axis=1)`` (any summation order) equals the exactly
       representable ``(n+1)/2`` bit-for-bit, and the centered values
       ``rx - (n+1)/2`` are exact half-integers.
    2. Every partial sum of ``sum(dx*dy)`` (and the squares) is a
       multiple of 0.25 bounded by ``n * (n-1)^2 / 4`` (< 2^53 for any
       realistic n), so the result is the exactly representable true
       value independent of accumulation order.  Therefore contiguous
       1-D ``np.dot`` calls produce bit-identical results to the
       strided BLAS gemm inside ``np.corrcoef`` (FMA changes nothing:
       the fused product-plus-add is exact here).
    3. The normalization replicates numpy 2.2.6 ``corrcoef`` exactly:
       ``c *= true_divide(1, n-1)``, then two sequential divisions by
       ``sqrt(diag(c))`` (row then column), then clip to [-1, 1].

    Must NOT be re-ordered: the normalization sequence (multiply by
    reciprocal, divide by s1, divide by s0) is part of the bit contract
    enforced by ``tests/metrics/test_daily_ic_vectorized_equivalence.py``.

    Args:
        rx: compressed average ranks of the factor cross-section (float64)
        ry: compressed average ranks of the label cross-section (float64)
        n: number of compressed entries (rx.size, must equal ry.size)

    Returns:
        The [1, 0] correlation coefficient (spearman IC for this cell)
    """
    avg = (n + 1) / 2
    dx = rx - avg
    dy = ry - avg
    c00 = np.dot(dx, dx)
    c01 = np.dot(dx, dy)
    c11 = np.dot(dy, dy)
    recip = np.true_divide(1, n - 1)
    stddev0 = np.sqrt(c00 * recip)
    stddev1 = np.sqrt(c11 * recip)
    r = ((c01 * recip) / stddev1) / stddev0
    if r > 1.0:
        return 1.0
    if r < -1.0:
        return -1.0
    return r


def _pearson_correlation(
    x: np.ndarray,
    y: np.ndarray,
    min_obs: int = 10,
    winsorize: float | None = None,
) -> float:
    """
    Pearson correlation with pairwise finite filtering.

    Args:
        x: Factor values (1D)
        y: Label values (1D)
        min_obs: Minimum observations required
        winsorize: Optional float in (0, 0.5); if set, winsorize both tails of
            x and y to the given fraction before correlating. None (default)
            keeps historical behavior (no winsorization).

    Returns:
        Correlation coefficient, or NaN if insufficient data or constant
    """
    mask = _pairwise_finite_mask(x, y)
    x_valid = x[mask]
    y_valid = y[mask]

    n = len(x_valid)
    if n < min_obs:
        return np.nan

    # Check for constants (robust to all-NaN / inf edges)
    if (np.nanmax(x_valid) - np.nanmin(x_valid) == 0) or (
        np.nanmax(y_valid) - np.nanmin(y_valid) == 0
    ):
        return np.nan

    if winsorize is not None:
        if not 0 < winsorize < 0.5:
            raise ValueError(
                f"winsorize must be in (0, 0.5) or None, got {winsorize!r}"
            )
        lower = winsorize
        upper = 1.0 - winsorize
        for series, index in ((x_valid, 0), (y_valid, 1)):
            q_lo, q_hi = np.nanquantile(series, [lower, upper])
            series = np.clip(series, q_lo, q_hi)
            if index == 0:
                x_win = series
            else:
                y_win = series
        x_valid, y_valid = x_win, y_win

    # Compute Pearson correlation
    corr = np.corrcoef(x_valid, y_valid)[0, 1]

    return corr


def _spearman_rank_correlation(x: np.ndarray, y: np.ndarray, min_obs: int = 10) -> float:
    """
    Spearman rank correlation with pairwise finite filtering and average ties.

    Args:
        x: Factor values (1D)
        y: Label values (1D)
        min_obs: Minimum observations required

    Returns:
        Rank correlation coefficient, or NaN if insufficient data,
        or constant. Statistical evidence strength is assessed separately;
        a binary nonconstant signal has a mathematically defined Spearman IC.
    """
    mask = _pairwise_finite_mask(x, y)
    x_valid = x[mask]
    y_valid = y[mask]

    n = len(x_valid)
    if n < min_obs:
        return np.nan

    # Only constancy is needed; do not sort each vector before ranking it.
    if n == 0 or np.all(x_valid == x_valid[0]) or np.all(y_valid == y_valid[0]):
        return np.nan

    # The coefficient path of scipy.spearmanr, without its unused p-value.
    # Keep its column layout and [1, 0] extraction (including rounding order).
    ranks = stats.rankdata(np.column_stack((x_valid, y_valid)), axis=0, method="average")
    return np.corrcoef(ranks, rowvar=False)[1, 0]


def compute_daily_ic(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    method: str = "pearson",
    min_assets: int = 20,
    backend: str = "exact",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute daily IC series for factor batch.

    Args:
        factor_batch: Input factor batch (T, N, F)
        label_bundle: Input label bundle (T, N) or (T,)
        method: "pearson" or "spearman"
        min_assets: Minimum valid assets per day (default raised to 20 so that
            small cross-sections with N<20 no longer produce spurious IC).
        backend: "exact" (default), "numba", "polars", or "gpu".

            "exact" is the default and produces the canonical bit-identical
            evidence payload (identity contract).  It runs the vectorized
            numpy path whose per-row correlation kernels replicate the
            ``np.corrcoef`` op sequence bit-for-bit.

            "numba" is an OPT-IN fast path backed by a JIT-compiled kernel
            (``quant_evaluator.kernels.numba_backend.numba_daily_ic_fast``).
            Its numerics differ from the exact path at the ulp level
            (different summation order: sequential JIT accumulation vs
            pairwise-summed means and BLAS gemm in the exact path; measured
            max |dIC| ~3e-16 on random panels, asserted below 1e-12 in
            ``tests/metrics/test_ic_fast_backend.py``).  NaN
            positions (insufficient data / constant cross-sections) are
            identical, but finite values are NOT bit-identical, therefore
            **evidence content hashes produced via backend="numba" differ
            from "exact"**.
            "polars" routes the same pairwise-complete per-day reduction
            through ``backends.polars_backend.polars_ic_batch`` (lazy
            groupby).  Its numerics also differ from the exact path at the
            ulp level (measured max |dIC| ~5.6e-17; asserted < 1e-12) and it
            shares the numba caveat: evidence hashes differ from "exact".
            One documented semantic divergence: days/factors whose
            pairwise-finite count is below ``min_assets`` report
            ``valid_counts == 0`` on the polars path (the group is filtered
            out of the plan), where the exact path records the true count.
            NaN positions are identical across all backends.  2026-09-25
            AB benchmark (T=1250, N=300): polars beats "exact" at every
            measured width (F=1: 29ms vs 48ms; F=10: 209ms vs 427ms;
            F=50: 1496ms vs 2210ms) and loses to "numba" everywhere
            (F=1: 12ms; F=10: 44ms; F=50: 143ms) - prefer "numba" for
            pure speed, "polars" for polars-native pipelines.  Evidence
            hashes from these opt-in paths differ from exact-path hashes**
            — do not mix them for the same evaluation identity.  The JIT
            kernel pays a one-time
            ~2-4s cold compile (including the numba import) per process
            on the first call; subsequent signatures within the process
            cost ~20ms, and repeat processes hit the on-disk numba cache
            (cache=True, written to the module's __pycache__).

            "gpu" is an OPT-IN CUDA path backed by
            ``kernels/gpu/correlation.py`` (``batched_pearson_ic`` /
            ``batched_spearman_ic`` over ``kernels/gpu/rank.py``'s batched
            average-tie rank; cupy, computed in float64 regardless of the
            input dtype).  2026-09-25 AB benchmark on L20 (best of 5 after
            warmup; "incl. transfer" counts host->device upload + device
            transpose + device->host result; panels ~15% NaN,
            min_assets=20): pearson incl. transfer 0.9/3.2/17.8/71.5ms vs
            numba 4.2/18.2/35.4/104.5ms at F=1/10/50/200 (T=1250, N=300)
            and 2.2ms vs 15.3ms at T=5000/N=50/F=10 — GPU wins at every
            measured shape (pure-compute 5-38x; incl. transfer >=1.5x
            except F=200 at 1.46x).  spearman incl. transfer: 4.9/23.5ms
            vs numba 5.5/38.1ms at F=1/10 (T=1250, N=300) and 12.8ms vs
            23.0ms at T=5000/N=50/F=10 (1.1-1.8x), but 151.9/540.0ms vs
            123.4/382.5ms at F=50/200 — spearman re-ranks the label per
            (t, f), so the (T, F, N) rank tensor dominates and GPU only
            pays off while F*N is small (measured crossover with N=300:
            wins at F*N <= 3000, loses at F*N >= 15000; with N=50, F=10
            still wins 1.8x).  Rule of thumb: prefer "gpu" for pearson at
            any shape, and for spearman on narrow-factor or long small-N
            panels; prefer "numba" for wide-factor spearman.  NaN
            positions and valid_counts are identical to the exact path:
            eligibility and the constant-rejection gate are pinned
            host-side on top of the kernel's own min_obs /
            zero-variance / distinct-level gates, so they match by
            construction; cells whose variance is within float64
            summation cancellation (~1e-16 relative) may additionally
            report NaN on this path.  Finite values differ at the ulp
            level (measured max |dIC| <= 4.5e-16 across the benchmark
            shapes; house rule rtol<=1e-8 / atol<=1e-10 holds), so
            evidence content hashes differ from "exact" — do not mix
            backends for the same evaluation identity.  The first call in
            a process pays CUDA context init (~1-2s); peak VRAM at
            T=1250/N=300/F=200 is ~0.7GB (rank scratch bounded by the
            1GiB chunked workspace in kernels/gpu/rank.py).

    Returns:
        (ic_series, valid_count_series)
        ic_series: shape (T, F) with IC per day per factor
        valid_count_series: shape (T, F) with count of valid obs

    Raises:
        InvalidContractError: If shapes incompatible
        InsufficientObservations: If no valid periods found
        ValueError: If method unknown, or backend unknown/unavailable
    """
    if factor_batch.num_times != len(label_bundle.values):
        raise InvalidContractError(
            f"Factor time axis ({factor_batch.num_times}) "
            f"does not match label length ({len(label_bundle.values)})"
        )

    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown method: {method}. Must be 'pearson' or 'spearman'")
    if backend not in ("exact", "numba", "polars", "gpu"):
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Must be 'exact', 'numba', 'polars' or 'gpu'"
        )

    values = factor_batch.values  # (T, N, F)
    labels, label_validity = normalize_label_panel(label_bundle, factor_batch.num_assets)

    T, N, F = values.shape
    ic_series = np.full((T, F), np.nan, dtype=np.float64)
    valid_counts = np.zeros((T, F), dtype=np.int32)

    # ---- Vectorized validity application + pairwise-finite mask. ----------
    # Semantics are identical to the historical per-day loop (kept verbatim
    # in ``_compute_daily_ic_reference``): invalid factor/label entries are
    # treated as NaN, and a pair is valid iff both entries are finite.
    v = values
    if factor_batch.validity is not None:
        v = np.where(factor_batch.validity, v, np.nan)
    lab = labels
    if label_validity is not None:
        lab = np.where(label_validity, lab, np.nan)

    mask = np.isfinite(v) & np.isfinite(lab)[:, :, None]  # (T, N, F)
    valid_counts[:, :] = np.sum(mask, axis=1).astype(np.int32)

    # A day/factor is eligible when it passes the original min_obs gate
    # (n >= min_assets).  n == 0 rows are excluded unconditionally: the
    # reference loop returns NaN for them via its empty-input constant check.
    eligible = (valid_counts >= min_assets) & (valid_counts > 0)

    if backend == "numba":
        # Opt-in JIT fast path: NaN positions match the exact path exactly
        # (same eligible/constant gates); finite values differ at the ulp
        # level (see docstring).  Lazy import keeps the default path free
        # of the numba import cost.
        from quant_evaluator.kernels.numba_backend import (
            NUMBA_AVAILABLE,
            numba_daily_ic_fast,
        )

        if not NUMBA_AVAILABLE:
            raise ValueError(
                "backend='numba' requires the optional numba package; "
                "install it or use backend='exact'"
            )
        ic_series = numba_daily_ic_fast(v, lab, mask, eligible, min_assets, method)
        return ic_series, valid_counts

    if backend == "polars":
        # Opt-in polars lazy-groupby path (see docstring for the AB numbers
        # and the valid_counts divergence).  ``v``/``lab`` already carry the
        # validity masks applied as NaN, so the adapter is called with
        # validity=None to avoid double application.
        from quant_evaluator.backends.polars_backend import (
            POLARS_AVAILABLE,
            polars_ic_batch,
        )

        if not POLARS_AVAILABLE:
            raise ValueError(
                "backend='polars' requires the optional polars package; "
                "install it or use backend='exact'"
            )
        factor_ids = tuple(factor_batch.factor_ids)
        ic_series, pl_counts = polars_ic_batch(
            v,
            lab,
            factor_ids,
            method=method,
            min_obs=min_assets,
            factor_validity=None,
            label_validity=None,
        )
        # NaN positions must coincide with the exact path's eligibility gate
        # even though polars reports count 0 for filtered-out groups.
        ic_series = np.asarray(ic_series, dtype=np.float64)
        ic_series[~eligible] = np.nan
        pl_counts = pl_counts.astype(np.int32, copy=False)
        pl_counts[~eligible] = valid_counts[~eligible]
        return ic_series, pl_counts

    if backend == "gpu":
        # Opt-in CUDA path (see docstring for the AB numbers / crossover).
        # ``v``/``lab`` already carry the validity masks applied as NaN, so
        # the kernels are called with validity=None to avoid double
        # application.
        try:
            import cupy as cp
        except ImportError:
            raise ValueError(
                "backend='gpu' requires the optional cupy package; "
                "install it or use backend='exact'"
            ) from None
        from quant_evaluator.kernels.gpu.correlation import (
            batched_pearson_ic,
            batched_spearman_ic,
        )

        # Host-side constant-rejection gate, vectorized: a cell is constant
        # iff its pairwise-finite factor (or label) values are all equal —
        # the same max-min == 0 decision as the exact path's compressed
        # check (and all-equal values <=> all-equal average ranks for
        # spearman).  Pinned on top of the kernel's own min_obs /
        # zero-variance / distinct-level gates so NaN positions match the
        # exact path by construction, not by numerical coincidence.
        x_min = np.where(mask, v, np.inf).min(axis=1)            # (T, F)
        x_max = np.where(mask, v, -np.inf).max(axis=1)           # (T, F)
        finite_lab = np.isfinite(lab)
        y_min = np.where(finite_lab, lab, np.inf).min(axis=1)    # (T,)
        y_max = np.where(finite_lab, lab, -np.inf).max(axis=1)   # (T,)
        constant = ((x_max - x_min) == 0.0) | (
            (y_max - y_min) == 0.0
        )[:, None]

        x_dev = cp.asarray(
            np.ascontiguousarray(v.transpose(0, 2, 1), dtype=np.float64)
        )  # (T, F, N) GPU-friendly layout, float64 compute
        y_dev = cp.asarray(np.ascontiguousarray(lab, dtype=np.float64))
        kern = (
            batched_spearman_ic if method == "spearman" else batched_pearson_ic
        )
        ic_dev, _ = kern(x_dev, y_dev, min_obs=min_assets)
        ic_series = cp.asnumpy(ic_dev).astype(np.float64, copy=False)
        # Match the exact path's corrcoef [-1, 1] clip (only ever moves a
        # value by the ~1e-15 ulp slack of the sums-based kernel).
        np.clip(ic_series, -1.0, 1.0, out=ic_series)
        ic_series[~eligible | constant] = np.nan
        return ic_series, valid_counts

    if method == "pearson":
        # Pearson keeps the exact historical per-row kernel (np.corrcoef on
        # the pairwise-compressed values); only the mask/count work moved to
        # the vectorized block above.  The per-row constant check and the
        # corrcoef op sequence are replicated bit-for-bit by
        # _corrcoef_pair_1d (see its docstring for the op-order contract).
        for t, f in np.argwhere(eligible):
            m = mask[t, :, f]
            x_valid = v[t, :, f][m]
            y_valid = lab[t][m]
            n = x_valid.size
            if n < min_assets or n == 0:
                continue
            # Constant check on the all-finite compressed values: max/min
            # == nanmax/nanmin here (no NaN can survive the mask), matching
            # the historical kernel bit-for-bit.
            if (x_valid.max() - x_valid.min() == 0) or (
                y_valid.max() - y_valid.min() == 0
            ):
                continue
            ic_series[t, f] = _corrcoef_pair_1d(x_valid, y_valid, n)
        return ic_series, valid_counts

    # ---- Spearman: vectorized average-tie ranks. ---------------------------
    # Bit-exactness contract (verified by equivalence tests): ranking a row
    # with invalid entries replaced by +inf sentinels produces exactly the
    # same average-tie ranks for the finite entries as ranking the
    # compressed valid entries (sentinels sort last and cannot join ties).
    # rankdata along axis=1 is row-independent, so ranking only the
    # ELIGIBLE rows yields bit-identical ranks for those rows while
    # skipping the wasted work on ineligible (t, f) cells.
    rows = np.argwhere(eligible)
    if rows.size:
        flat_rows = rows[:, 0] * F + rows[:, 1]
        sentinel_x = np.where(mask, v, np.inf).transpose(0, 2, 1).reshape(T * F, N)[flat_rows]
        sentinel_y = (
            np.where(mask, lab[:, :, None], np.inf)
            .transpose(0, 2, 1)
            .reshape(T * F, N)[flat_rows]
        )
        r_x = stats.rankdata(sentinel_x, axis=1, method="average")
        r_y = stats.rankdata(sentinel_y, axis=1, method="average")

        for k in range(rows.shape[0]):
            t, f = rows[k]
            m = mask[t, :, f]
            rx = r_x[k][m]
            ry = r_y[k][m]
            # The original constant check ran on the raw valid values; all-equal
            # values <=> all-equal average ranks, so checking ranks is exactly
            # equivalent (and avoids compressing the raw row again).  min==max
            # is the same decision as np.all(rx == rx[0]) without the bool
            # temporary.
            if rx.min() == rx.max() or ry.min() == ry.max():
                continue
            # Keep the column layout semantics and [1, 0] extraction
            # (including rounding order) of the historical coefficient path;
            # _corrcoef_rank_last replicates
            # np.corrcoef(..., rowvar=False)[1, 0] bit-for-bit.
            ic_series[t, f] = _corrcoef_rank_last(rx, ry, rx.size)

    return ic_series, valid_counts


def _compute_daily_ic_reference(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    method: str = "pearson",
    min_assets: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Historical per-day loop implementation of :func:`compute_daily_ic`.

    Kept verbatim as the exact-reference oracle for the vectorized public
    path.  Equivalence tests
    (``tests/metrics/test_daily_ic_vectorized_equivalence.py``) assert
    bit-identical ``(ic_series, valid_counts)`` between this reference and
    ``compute_daily_ic`` across seeds, dtypes, missingness patterns, ties
    and boundary sizes.  Do not "simplify" this function; it must remain the
    unoptimized semantics anchor.
    """
    if factor_batch.num_times != len(label_bundle.values):
        raise InvalidContractError(
            f"Factor time axis ({factor_batch.num_times}) "
            f"does not match label length ({len(label_bundle.values)})"
        )

    corr_fn = _pearson_correlation if method == "pearson" else _spearman_rank_correlation

    values = factor_batch.values  # (T, N, F)
    labels, label_validity = normalize_label_panel(label_bundle, factor_batch.num_assets)

    T, N, F = values.shape
    ic_series = np.full((T, F), np.nan, dtype=np.float64)
    valid_counts = np.zeros((T, F), dtype=np.int32)

    # Compute IC per day per factor
    for t in range(T):
        for f in range(F):
            factor_t = values[t, :, f]  # (N,)
            label_t = labels[t, :]       # (N,)

            # Apply validity masks if present
            if factor_batch.validity is not None:
                factor_valid = factor_batch.validity[t, :, f]
                factor_t = np.where(factor_valid, factor_t, np.nan)

            if label_validity is not None:
                label_t = np.where(label_validity[t], label_t, np.nan)

            # Compute correlation
            ic = corr_fn(factor_t, label_t, min_obs=min_assets)
            ic_series[t, f] = ic

            # Count valid observations
            mask = _pairwise_finite_mask(factor_t, label_t)
            valid_counts[t, f] = int(np.sum(mask))

    return ic_series, valid_counts


def _reject_boolean_ic_series(ic_series: np.ndarray) -> None:
    """
    Reject boolean IC series.

    True/False silently coerces to 1.0/0.0 (e.g. a validity mask), which
    would yield a plausible-looking mean IC that carries no information.

    Raises:
        ValueError: If ic_series has boolean dtype or contains Python bools.
    """
    if ic_series.dtype == bool:
        raise ValueError(
            "ic_series must be numeric, got boolean dtype "
            "(True/False would silently coerce to 1.0/0.0)"
        )
    if ic_series.dtype == object:
        if any(isinstance(v, (bool, np.bool_)) for v in ic_series.ravel()):
            raise ValueError(
                "ic_series must be numeric, got Python bools "
                "(True/False would silently coerce to 1.0/0.0)"
            )


def compute_mean_ic(
    ic_series: np.ndarray,
    valid_counts: Optional[np.ndarray] = None,
    min_periods: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean IC and its standard deviation across time.

    Args:
        ic_series: Daily IC series (T, F)
        valid_counts: Valid observation counts (T, F)
        min_periods: Minimum periods required for mean

    Returns:
        (mean_ic, ic_std) arrays of shape (F,)

    Raises:
        ValueError: If ic_series is boolean (or contains Python bools)
    """
    _reject_boolean_ic_series(np.asarray(ic_series))

    # Count non-NaN periods per factor
    valid_periods = np.sum(~np.isnan(ic_series), axis=0)  # (F,)

    # Compute mean and std with fully NaN-suppressed operations (all-NaN or
    # singleton-period columns are expected and produce NaN via the mask below,
    # not RuntimeWarnings).
    with np.errstate(invalid="ignore", divide="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_ic = np.nanmean(ic_series, axis=0)  # (F,)
            ic_std = np.nanstd(ic_series, axis=0, ddof=1)  # (F,)

    # Mask insufficient periods
    insufficient = valid_periods < min_periods
    mean_ic = np.where(insufficient, np.nan, mean_ic)
    ic_std = np.where(insufficient, np.nan, ic_std)

    # Mask non-finite results: inf in the series propagates through nanmean
    # into an infinite (invalid) mean/std — report NaN instead.
    mean_ic = np.where(np.isfinite(mean_ic), mean_ic, np.nan)
    ic_std = np.where(np.isfinite(ic_std), ic_std, np.nan)

    return mean_ic, ic_std


def compute_mean_ic_value(
    ic_series: np.ndarray,
    valid_counts: Optional[np.ndarray] = None,
    min_periods: int = 20,
) -> np.ndarray:
    """Compute only the mean IC component for registry execution."""
    mean_ic, _ = compute_mean_ic(
        ic_series,
        valid_counts=valid_counts,
        min_periods=min_periods,
    )
    return mean_ic


def compute_ic_std(
    ic_series: np.ndarray,
    valid_counts: Optional[np.ndarray] = None,
    min_periods: int = 20,
) -> np.ndarray:
    """Compute only the IC standard-deviation component."""
    _, ic_std = compute_mean_ic(
        ic_series,
        valid_counts=valid_counts,
        min_periods=min_periods,
    )
    return ic_std


def ic_significance(
    ic_series: np.ndarray,
    min_periods: int = 2,
) -> Tuple[float, float, float]:
    """
    Significance test for a daily IC series.

    Args:
        ic_series: 1D daily IC series
        min_periods: Minimum number of finite periods required

    Returns:
        (ir, t_stat, p_value)
        ir: information ratio = mean / std (ddof=1); NaN if std is 0
        t_stat: t = mean / (std / sqrt(n)); NaN if std is 0
        p_value: two-sided normal approximation probability
        All values NaN if fewer than min_periods finite observations.
    """
    series = np.asarray(ic_series, dtype=np.float64)

    # Reject booleans (True/False would silently coerce to 1.0/0.0)
    _reject_boolean_ic_series(series)

    finite = series[np.isfinite(series)]

    if finite.size < min_periods:
        return np.nan, np.nan, np.nan

    with np.errstate(invalid="ignore"):
        n = finite.size
        mean = np.mean(finite)
        if n < 2:
            return np.nan, np.nan, np.nan
        std = np.std(finite, ddof=1)
        if std == 0 or not np.isfinite(std):
            return np.nan, np.nan, np.nan
        ir = mean / std
        t_stat = mean / (std / np.sqrt(n))
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(t_stat)))

    return float(ir), float(t_stat), float(p_value)
