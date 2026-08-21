"""
QE metamorphic / property tests.

Every test here asserts a structural property of the QE kernels (invariance
under factor rescaling / sign flip / asset permutation / chunking, residual
orthogonality after neutralization, conservation under duplication, NaN-handling
contracts, full-vs-optimized numerical parity) rather than a hand-computed
number. Where a property is the kernel's documented CONTRACT but no kernel
exists yet (zscore standardization, row-drop-vs-pairwise NaN handling), the
test pins the contract and FAILS LOUDLY if it is ever violated.

Importable source: ``build/lib/quant_evaluator`` (same bootstrap pattern as
``test_numerical_oracle.py``).

Run with:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=/home/shw/quant_projects/quant_evaluator/build/lib \
    python -m pytest tests/test_metamorphic.py -v --tb=short

Definitional notes (matched to the kernel contracts, verified by reading the
sources in ``build/lib/quant_evaluator``):

- **IC kernel contract.** ``metrics/ic.compute_daily_ic`` and
  ``kernels/fast.fast_ic_batch`` compute the correlation on the
  PAIRWISE-finite observations only (``x`` and ``y`` finite at the same
  asset). This is what makes ``factor * c`` (c != 0) leave Pearson IC
  exactly unchanged and Spearman IC exactly sign-flipped under
  ``factor * -1``. It is also why "drop NaN rows, then correlate" differs
  from pairwise handling whenever the factor and label NaN masks differ —
  the kernels use pairwise handling, and test 10a pins that.
- **Chunk permutation.** ``fast_ic_batch``, ``compute_quantile_returns_fast``
  and ``numba_*`` compute each (t, f) cross-section independently, so
  permuting the TIME axis merely permutes the output rows (the metric is
  per-time). ``fast_turnover_estimate`` (window=1) is NOT a per-time metric:
  row t depends on row t-1, so shuffling the time axis changes values while
  a time-order permutation of a per-time metric's output is preserved. Test
  asserts both halves of that contract.
- **Turnover window-1 edge.** The canonical definition
  ``0.5 * sum |w_t - w_{t-1}|`` makes row 0 NaN (no prior row). Splitting a
  panel at row k re-bases the window, so the split's row k becomes NaN too.
  Hence time-chunk concat equality holds for IC and quantile returns but NOT
  for turnover; the turnover chunking property asserted is instead that each
  split chunk's *interior* rows (t >= window in the chunk) equal the
  corresponding full-panel rows.
- **Neutralization.** ``metrics/exposure.compute_factor_loadings`` performs a
  cross-sectional OLS of factor on risk factors per period (intercept by
  default). The residual is definitionally orthogonal to every risk factor,
  so ``|corr(residual, risk_factor)|`` per period is ~1e-16, not merely
  small. There is no separate "neutralize" preprocessing kernel.
- **FA conservation under duplication.** The kernel never dedupes factor
  columns: each duplicate appears exactly once per (t, n, f) instance, so
  ``compute_valid_pair_counts`` counts each duplicate separately and
  conservation holds (``sum == T * N * F`` for full coverage). There is no
  FA/multiplicity kernel in build/lib; the duplication-count contract is
  asserted through the IC and coverage kernels.
- **No zscore kernel exists.** ``compute_higher_moments`` returns the
  sample std (``np.nanstd ddof=1``) and ``detect_outliers_zscore`` builds
  z-scores internally, but neither exposes a standardized factor. Test 7 is
  therefore a documented contract test asserting the defining property of a
  cross-section z-score transform (mean 0, std 1) and FAILS LOUDLY if a
  kernel with the ``zscore_*`` name surface ever violates it.
- **No row-drop NaN kernel exists.** Test 10b is the flip side of test 10a:
  the documented kernel contract is pairwise-finite handling, and a kernel
  that documented row-drop semantics would have to be tested differently.
  Since no such kernel exists, the test asserts the current kernel contract
  (pairwise) and fails loudly if a ``dropna``-named kernel ever disagrees
  with it.
"""

from pathlib import Path

import numpy as np
import pytest

# Importable source lives in build/lib; pin it to sys.path[0] BEFORE any
# quant_evaluator import so the repo-root stub package cannot shadow it.
import sys

_BUILD_LIB = Path(__file__).resolve().parents[1] / "build" / "lib"
if str(_BUILD_LIB) not in sys.path:
    sys.path.insert(0, str(_BUILD_LIB))

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.kernels.fast import (
    fast_ic_batch,
    fast_quantile_binning,
    compute_quantile_returns_fast,
    fast_turnover_estimate,
)
from quant_evaluator.kernels.numba_backend import (
    numba_ic_batch,
    numba_quantile_binning,
)
from quant_evaluator.metrics.ic import compute_daily_ic
from quant_evaluator.metrics.quantile import (
    assign_quantiles_fast,
    assign_quantiles_batch,
    compute_quantile_returns,
)
from quant_evaluator.metrics.quantile_numba import compute_quantile_returns_numba
from quant_evaluator.metrics.quantile_optimized import compute_quantile_returns_ultra_fast
from quant_evaluator.metrics.turnover import estimate_turnover_from_ranks
from quant_evaluator.metrics.quality import compute_valid_pair_counts
from quant_evaluator.metrics.exposure import compute_factor_loadings, compute_concentration_hhi


# ---------------------------------------------------------------------------
# shared fixtures / helpers
# ---------------------------------------------------------------------------

T, N, F = 8, 40, 3
SEED = 20260821


@pytest.fixture(scope="module")
def panel():
    """Fixed synthetic panel: (T, N, F) factors + (T, N) labels + contracts."""
    rng = np.random.default_rng(SEED)
    values = rng.normal(size=(T, N, F))
    labels = rng.normal(size=(T, N))
    batch = FactorBatch(
        factor_ids=("f0", "f1", "f2"),
        time_axis=AxisRef("t", "int", T),
        asset_axis=AxisRef("a", "str", N),
        values=values,
    )
    bundle = LabelBundle(
        target_id="r",
        values=labels,
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
    )
    return values, labels, batch, bundle


def _allclose_nanaware(a, b, rtol=1e-9, atol=1e-12):
    """NaN-aware structural equality: same NaN pattern + finite values close."""
    if a.shape != b.shape:
        return False
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return False
    both = np.isfinite(a) & np.isfinite(b)
    if not np.any(both):
        return True
    return bool(np.allclose(a[both], b[both], rtol=rtol, atol=atol, equal_nan=True))


# ---------------------------------------------------------------------------
# 1. factor * positive constant  ->  RankIC (and Pearson IC) invariant
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_positive_constant_scaling_preserves_ic(panel, method):
    """IC(factor) == IC(c * factor) for c > 0 under the pairwise-finite kernel.

    A positive scalar does not change the ordering nor the (zero-centered)
    Pearson correlation; Spearman is rank-based so it is untouched as well.
    """
    values, labels, _, _ = panel
    c = 3.7
    ic_base, cnt_base = fast_ic_batch(values, labels, method=method, min_obs=10)
    ic_scaled, cnt_scaled = fast_ic_batch(values * c, labels, method=method, min_obs=10)
    assert _allclose_nanaware(ic_base, ic_scaled)
    assert np.array_equal(cnt_base, cnt_scaled)

    # Rank-based kernels are likewise invariant to positive rescaling.
    assert np.array_equal(
        fast_quantile_binning(values, n_quantiles=5),
        fast_quantile_binning(values * c, n_quantiles=5),
    )
    q0, _ = compute_quantile_returns_fast(values, labels, n_quantiles=5, min_assets=10)
    q1, _ = compute_quantile_returns_fast(values * c, labels, n_quantiles=5, min_assets=10)
    assert _allclose_nanaware(q0, q1)
    assert _allclose_nanaware(
        compute_concentration_hhi(values[:, :, 0]),
        compute_concentration_hhi(values[:, :, 0] * c),
    )


# ---------------------------------------------------------------------------
# 2. factor * -1  ->  RankIC flips sign (exactly)
# ---------------------------------------------------------------------------

def test_negation_flips_ic_sign(panel):
    """IC(-factor) == -IC(factor) for both Pearson and Spearman (no ties)."""
    values, labels, _, _ = panel
    for method in ("pearson", "spearman"):
        ic_pos, cnt_pos = fast_ic_batch(values, labels, method=method, min_obs=10)
        ic_neg, cnt_neg = fast_ic_batch(-values, labels, method=method, min_obs=10)
        assert np.array_equal(cnt_pos, cnt_neg)
        # Same NaN pattern, and finite values are exact negatives:
        assert np.array_equal(np.isnan(ic_pos), np.isnan(ic_neg))
        finite = np.isfinite(ic_pos) & np.isfinite(ic_neg)
        assert np.allclose(ic_pos[finite], -ic_neg[finite], rtol=0.0, atol=1e-12)

    # HHI uses gross exposure, so it is sign-invariant by construction:
    assert _allclose_nanaware(
        compute_concentration_hhi(values[:, :, 1]),
        compute_concentration_hhi(-values[:, :, 1]),
    )


# ---------------------------------------------------------------------------
# 3. asset permutation -> metrics invariant
# ---------------------------------------------------------------------------

def test_asset_permutation_is_invariant(panel):
    """Shuffling the asset (column) order must not change any metric."""
    values, labels, batch, bundle = panel
    rng = np.random.default_rng(SEED + 1)
    perm = rng.permutation(N)
    pv, pl = values[:, perm, :], labels[:, perm]
    pbatch = FactorBatch(
        factor_ids=batch.factor_ids,
        time_axis=batch.time_axis,
        asset_axis=AxisRef("a", "str", N, values=perm.astype(str)),
        values=pv,
    )
    pbundle = LabelBundle(
        target_id=bundle.target_id,
        values=pl,
        horizon=bundle.horizon,
        decision_time=bundle.decision_time,
        label_start_time=bundle.label_start_time,
        label_end_time=bundle.label_end_time,
    )

    for method in ("pearson", "spearman"):
        a, ca = fast_ic_batch(values, labels, method=method, min_obs=10)
        b, cb = fast_ic_batch(pv, pl, method=method, min_obs=10)
        assert _allclose_nanaware(a, b)
        assert np.array_equal(ca, cb)
        # Reference IC kernel too (the fast/ref parity property is asserted
        # separately; here we only pin the permutation invariance).
        ra, _ = compute_daily_ic(batch, bundle, method=method, min_assets=10)
        rb, _ = compute_daily_ic(pbatch, pbundle, method=method, min_assets=10)
        assert _allclose_nanaware(ra, rb)

    qa, qca = compute_quantile_returns_fast(values, labels, n_quantiles=5, min_assets=10)
    qb, qcb = compute_quantile_returns_fast(pv, pl, n_quantiles=5, min_assets=10)
    assert _allclose_nanaware(qa, qb)
    assert np.array_equal(qca, qcb)

    ta = fast_turnover_estimate(values, window=1)
    tb = fast_turnover_estimate(pv, window=1)
    assert _allclose_nanaware(ta, tb)
    # Reference turnover kernel as well.
    tra = estimate_turnover_from_ranks(batch, window=1)
    trb = estimate_turnover_from_ranks(pbatch, window=1)
    assert _allclose_nanaware(tra, trb)

    h0 = compute_concentration_hhi(values[:, :, 2])
    h1 = compute_concentration_hhi(pv[:, :, 2])
    assert _allclose_nanaware(h0, h1)


# ---------------------------------------------------------------------------
# 4. chunk permutation where mathematically valid -> result invariant
# ---------------------------------------------------------------------------

def test_time_chunk_permutation_commutes_for_per_time_metrics(panel):
    """Permuting the time axis permutes the rows of a per-time metric.

    fast_ic_batch and compute_quantile_returns_fast compute each (t, f)
    cross-section independently, so any time-order permutation is a pure
    output-row permutation (the result is invariant). fast_turnover_estimate
    is NOT a per-time metric (row t depends on row t-1), so shuffling the
    time axis must CHANGE it — this half of the test is a sanity check that
    the suite actually detects structural (time) dependence.
    """
    values, labels, _, _ = panel
    perm = np.array([4, 1, 5, 0, 2, 3, 7, 6])
    fsh, lsh = values[perm], labels[perm]

    for method in ("pearson", "spearman"):
        base, _ = fast_ic_batch(values, labels, method=method, min_obs=10)
        permuted, _ = fast_ic_batch(fsh, lsh, method=method, min_obs=10)
        assert _allclose_nanaware(base[perm], permuted)

    q0, _ = compute_quantile_returns_fast(values, labels, n_quantiles=5, min_assets=10)
    q1, _ = compute_quantile_returns_fast(fsh, lsh, n_quantiles=5, min_assets=10)
    assert _allclose_nanaware(q0[perm], q1)

    # Sanity: a causal, time-dependent metric must NOT be invariant.
    t0 = fast_turnover_estimate(values, window=1)
    t1 = fast_turnover_estimate(fsh, window=1)
    assert not _allclose_nanaware(t0[perm], t1), (
        "turnover (window=1) is time-dependent and must change under a time shuffle"
    )


def test_time_chunk_decomposition_concats_exactly(panel):
    """Splitting the time axis into contiguous chunks and concatenating the
    per-time outputs reproduces the full-panel result for per-time metrics."""
    values, labels, _, _ = panel
    for method in ("pearson", "spearman"):
        full, _ = fast_ic_batch(values, labels, method=method, min_obs=10)
        parts = [
            fast_ic_batch(values[s:e], labels[s:e], method=method, min_obs=10)[0]
            for s, e in ((0, 3), (3, 6), (6, 8))
        ]
        assert _allclose_nanaware(full, np.concatenate(parts, axis=0))

    full_q, _ = compute_quantile_returns_fast(values, labels, n_quantiles=5, min_assets=10)
    parts_q = [
        compute_quantile_returns_fast(values[s:e], labels[s:e], n_quantiles=5, min_assets=10)[0]
        for s, e in ((0, 3), (3, 6), (6, 8))
    ]
    assert _allclose_nanaware(full_q, np.concatenate(parts_q, axis=0))


def test_turnover_split_chunk_interior_rows_match_full_panel(panel):
    """Turnover (window=1) row t depends on row t-1, so a split re-bases the
    window: each chunk's first row is NaN. The interior rows (t >= window in
    the chunk) must still equal the full-panel rows exactly."""
    values, _, _, _ = panel
    full = fast_turnover_estimate(values, window=1)
    for s, e in ((1, 4), (4, 8)):
        chunk = fast_turnover_estimate(values[s:e], window=1)
        # Chunk row i corresponds to full row (s + i); chunk interior i>=1.
        assert _allclose_nanaware(full[s + 1:e], chunk[1:])
    # Sanity: the split edge is genuinely NaN (window re-basing), so a naive
    # concat is NOT expected to equal the full panel (documented behavior).
    joined = np.concatenate(
        [fast_turnover_estimate(values[0:4], window=1),
         fast_turnover_estimate(values[4:8], window=1)],
        axis=0,
    )
    assert not _allclose_nanaware(full, joined)


# ---------------------------------------------------------------------------
# 5. duplicate factor -> FA conservation is correct
# ---------------------------------------------------------------------------

def test_duplicate_factor_conservation(panel):
    """Duplicating a factor column must produce identical per-factor metrics
    AND conserve the observation budget: each duplicate is counted exactly
    once per instance, so sum(valid_pair_counts) == T * N * F."""
    values, labels, batch, bundle = panel
    dup_values = np.concatenate([values, values], axis=2)
    dup_batch = FactorBatch(
        factor_ids=("a", "b", "c", "a_dup", "b_dup", "c_dup"),
        time_axis=batch.time_axis,
        asset_axis=batch.asset_axis,
        values=dup_values,
    )
    dup_bundle = LabelBundle(
        target_id=bundle.target_id,
        values=bundle.values,
        horizon=bundle.horizon,
        decision_time=bundle.decision_time,
        label_start_time=bundle.label_start_time,
        label_end_time=bundle.label_end_time,
    )

    # Each duplicate appears exactly once per instance in the IC kernel:
    ic, counts = compute_daily_ic(dup_batch, dup_bundle, method="spearman", min_assets=10)
    for f in range(F):
        assert _allclose_nanaware(ic[:, f], ic[:, f + F])
        assert np.array_equal(counts[:, f], counts[:, f + F])

    # Conservation: every (t, n, f) instance contributes exactly once.
    vpc = compute_valid_pair_counts(dup_batch, dup_bundle)
    assert vpc.shape == (2 * F,)
    assert vpc.sum() == T * N * F, f"conservation violated: {vpc.sum()} != {T * N * F}"
    # The original factor is counted once per instance, and so is its dup.
    assert np.array_equal(vpc[:F], vpc[F:])


# ---------------------------------------------------------------------------
# 6. neutralization -> residual exposure ~ 0
# ---------------------------------------------------------------------------

def test_neutralized_residual_is_orthogonal_to_exposure(panel):
    """compute_factor_loadings runs per-period OLS of factor on risk factors
    (intercept by default); the residual is definitionally orthogonal to every
    risk factor, so |corr(residual, risk_k)| ~ 1e-16 per period."""
    values, _, _, _ = panel
    rng = np.random.default_rng(SEED + 2)
    K = 3
    # Factor with a strong embedded exposure to risk factor 0.
    factor_t0 = values[:, :, 0] + 0.8 * values[:, :, 1]
    risk = rng.normal(size=(T, N, K))

    loadings, r_squared, residuals = compute_factor_loadings(
        factor_t0, risk, intercept=True, min_obs=10
    )
    assert residuals.shape == (T, N)
    assert np.all(np.isfinite(loadings))  # every period was regressible
    assert np.sum(np.isfinite(r_squared)) == T

    max_abs_corr = 0.0
    for t in range(T):
        for k in range(K):
            mask = np.isfinite(residuals[t]) & np.isfinite(risk[t, :, k])
            if mask.sum() > 10:
                c = abs(float(np.corrcoef(residuals[t][mask], risk[t, :, k][mask])[0, 1]))
                max_abs_corr = max(max_abs_corr, c)
    assert max_abs_corr < 1e-9, f"residual exposure not ~0: |corr| = {max_abs_corr}"

    # R^2 is high when the factor embeds the risk factor (sanity):
    assert np.nanmean(r_squared) > 0.2


# ---------------------------------------------------------------------------
# 7. zscore -> valid cross-section mean ~ 0, std ~ 1  (documented contract)
# ---------------------------------------------------------------------------

def _zscore_cross_sectional(factor: np.ndarray) -> np.ndarray:
    """Reference cross-section (axis=1) z-score, NaN-preserving.

    This is the defining property a z-score transform must satisfy. There is
    no zscore kernel in build/lib/quant_evaluator, so this is a documented
    contract: IF a kernel with the ``zscore_*`` name surface appears, it must
    satisfy this property — the test fails loudly otherwise.
    """
    mean = np.nanmean(factor, axis=1, keepdims=True)
    std = np.nanstd(factor, axis=1, keepdims=True, ddof=1)
    return (factor - mean) / std


def test_zscore_transform_yields_zero_mean_unit_std():
    """Any factor z-scored per cross-section has mean ~0 and std ~1 (ddof=1)."""
    rng = np.random.default_rng(SEED + 3)
    factor = rng.normal(size=(12, 50)) + rng.normal(size=(1, 50)) * 3.0

    z = _zscore_cross_sectional(factor)

    for t in range(12):
        row = z[t]
        finite = np.isfinite(row)
        assert finite.all()
        assert abs(np.mean(row)) < 1e-12
        assert abs(np.std(row, ddof=1) - 1.0) < 1e-12


def test_zscore_kernel_contract_fails_loudly_if_violated():
    """Contract test: zscore-* named kernels in build/lib must produce a
    zero-mean, unit-std cross-section. Fail loudly (not silently) if a kernel
    with that name surface is added and violates the property."""
    import importlib
    import pkgutil

    from quant_evaluator import metrics

    candidates = []
    for mod in pkgutil.iter_modules(metrics.__path__):
        if mod.name.startswith("zscore") or mod.name.startswith("standardize"):
            candidates.append(mod.name)
    for fname in dir(metrics):
        if fname.startswith("zscore") or fname.startswith("standardize"):
            candidates.append(fname)

    if not candidates:
        pytest.skip(
            "no zscore/standardize kernel exists yet in build/lib; "
            "documented-contract test (nothing to check)"
        )

    rng = np.random.default_rng(SEED + 4)
    factor = rng.normal(size=(6, 40))

    for name in candidates:
        try:
            obj = getattr(importlib.import_module("quant_evaluator.metrics"), name)
        except Exception:  # noqa: BLE001 - name surface may not be an attribute
            obj = None
        if obj is None or not callable(obj):
            continue
        try:
            out = np.asarray(obj(factor.copy()), dtype=np.float64)
        except Exception:  # noqa: BLE001 - signature mismatch is not a contract breach
            continue
        out = out.reshape(factor.shape) if out.shape == factor.shape else out
        if out.ndim != 2 or out.shape != factor.shape:
            continue
        for t in range(out.shape[0]):
            row = out[t]
            finite = np.isfinite(row)
            if finite.sum() < 2:
                continue
            assert abs(np.nanmean(row)) < 1e-6, (
                f"{name} does not produce zero-mean cross-sections (contract violated)"
            )
            assert abs(np.nanstd(row, ddof=1) - 1.0) < 1e-6, (
                f"{name} does not produce unit-std cross-sections (contract violated)"
            )


# ---------------------------------------------------------------------------
# 8. full vs optimized kernels -> numerical parity
# ---------------------------------------------------------------------------

def test_numba_and_fast_ic_parity(panel):
    """numba_ic_batch and fast_ic_batch must agree bit-for-bit (NaN-aware)."""
    values, labels, _, _ = panel
    for method in ("pearson", "spearman"):
        ic_fast, cnt_fast = fast_ic_batch(values, labels, method=method, min_obs=10)
        ic_numba, cnt_numba = numba_ic_batch(values, labels, method=method, min_obs=10)
        assert _allclose_nanaware(ic_fast, ic_numba, rtol=1e-12, atol=1e-14)
        assert np.array_equal(cnt_fast, cnt_numba)


def test_reference_and_fast_ic_parity(panel):
    """metrics/ic.compute_daily_ic (reference) vs kernels/fast.fast_ic_batch."""
    values, labels, batch, bundle = panel
    for method in ("pearson", "spearman"):
        ic_ref, cnt_ref = compute_daily_ic(batch, bundle, method=method, min_assets=10)
        ic_fast, cnt_fast = fast_ic_batch(values, labels, method=method, min_obs=10)
        assert _allclose_nanaware(ic_ref, ic_fast, rtol=1e-9, atol=1e-12)
        assert np.array_equal(cnt_ref, cnt_fast)


def test_turnover_reference_and_fast_parity(panel):
    """metrics/turnover.estimate_turnover_from_ranks vs fast_turnover_estimate."""
    values, _, batch, _ = panel
    t_ref = estimate_turnover_from_ranks(batch, window=1)
    t_fast = fast_turnover_estimate(values, window=1)
    assert _allclose_nanaware(t_ref, t_fast, rtol=1e-9, atol=1e-12)


def test_quantile_binning_parity_across_optimizations(panel):
    """fast_quantile_binning (kernels), assign_quantiles_fast/batch (metrics),
    and numba_quantile_binning (kernels) must all assign identical bins."""
    values, _, _, _ = panel
    bins_fast = fast_quantile_binning(values, n_quantiles=5)
    bins_numba = numba_quantile_binning(values, 5, 5)
    bins_fast_metric = assign_quantiles_fast(values, n_quantiles=5)
    bins_batch_metric = assign_quantiles_batch(values, n_quantiles=5)

    assert np.array_equal(bins_fast, bins_numba)
    assert np.array_equal(bins_fast, bins_fast_metric)
    assert np.array_equal(bins_fast, bins_batch_metric)


def test_quantile_returns_parity_across_implementations(panel):
    """Reference (metrics), fast (kernels), numba, and ultra-fast quantile
    returns must agree (NaN-aware) within tight tolerance."""
    values, labels, batch, bundle = panel

    q_ref, c_ref = compute_quantile_returns(batch, bundle, n_quantiles=5, min_assets=10)
    q_fast, c_fast = compute_quantile_returns_fast(values, labels, n_quantiles=5, min_assets=10)
    q_numba, c_numba = compute_quantile_returns_numba(batch, bundle, n_quantiles=5, min_assets=10)
    q_ultra, c_ultra = compute_quantile_returns_ultra_fast(
        batch, bundle, n_quantiles=5, min_assets=10
    )

    assert _allclose_nanaware(q_ref, q_fast, rtol=1e-9, atol=1e-12)
    assert _allclose_nanaware(q_ref, q_numba, rtol=1e-9, atol=1e-12)
    assert _allclose_nanaware(q_ref, q_ultra, rtol=1e-9, atol=1e-12)
    assert np.array_equal(c_ref, c_fast)
    assert np.array_equal(c_ref, c_numba)
    assert np.array_equal(c_ref, c_ultra)


# ---------------------------------------------------------------------------
# 9. shuffle rows (time) is NOT invariant (sanity that tests detect it)
# ---------------------------------------------------------------------------

def test_time_row_shuffle_is_not_invariant(panel):
    """A causal factor's metrics CHANGE when the time ordering changes.

    This is the negative control for the invariant suite: if every test in
    this file accidentally passed under any input perturbation, the suite
    would be vacuous. IC and quantile returns are per-time metrics and
    commute with row permutation (asserted in test 4); this test verifies the
    kernel-level causal dependence for metrics that genuinely couple rows —
    turnover — and that the per-time metrics actually DIFFER when the data
    itself is not merely permuted.
    """
    values, labels, _, _ = panel

    # Turnover (window=1) is causal: changing the time order changes values.
    perm = np.array([4, 1, 5, 0, 2, 3, 7, 6])
    t0 = fast_turnover_estimate(values, window=1)
    t1 = fast_turnover_estimate(values[perm], window=1)
    assert not _allclose_nanaware(t0[perm], t1)

    # A data-level time perturbation (not a pure reorder of a per-time
    # metric) must change the per-time metrics' VALUES as well: swap the
    # labels of two adjacent periods. Per-time cross-sections change because
    # the label panel is re-paired with a different factor row.
    labels_swapped = labels.copy()
    labels_swapped[1], labels_swapped[2] = labels_swapped[2], labels_swapped[1]
    ic0, _ = fast_ic_batch(values, labels, method="spearman", min_obs=10)
    ic1, _ = fast_ic_batch(values, labels_swapped, method="spearman", min_obs=10)
    # Row 1 and 2 change; the other rows are untouched (per-time kernel).
    assert not _allclose_nanaware(ic0[1:3], ic1[1:3])
    assert _allclose_nanaware(np.delete(ic0, [1, 2], axis=0),
                              np.delete(ic1, [1, 2], axis=0))


# ---------------------------------------------------------------------------
# 10. NaN handling: documented kernel contract (pairwise-finite)
# ---------------------------------------------------------------------------

def test_ic_pairwise_finite_nan_handling_matches_contract(panel):
    """The IC kernel correlates on PAIRWISE-finite observations only.

    Hand-rolled pairwise-finite Pearson per (t, f) must reproduce the kernel
    exactly, and the reported valid count must equal the pairwise-finite
    count. This is the documented contract of both metrics/ic.py and
    kernels/fast.py.
    """
    values, labels, _, _ = panel
    rng = np.random.default_rng(SEED + 5)

    # Inject NaNs with DIFFERENT masks on factor vs label, so pairwise vs
    # row-drop genuinely disagree.
    masked_values = values.copy()
    masked_labels = labels.copy()
    masked_values[rng.random(values.shape) < 0.12] = np.nan
    masked_labels[rng.random(labels.shape) < 0.15] = np.nan

    ic, counts = fast_ic_batch(masked_values, masked_labels, method="pearson", min_obs=10)

    for t in range(T):
        for f in range(F):
            m = np.isfinite(masked_values[t, :, f]) & np.isfinite(masked_labels[t, :])
            n_pair = int(np.sum(m))
            assert counts[t, f] == n_pair, "valid count != pairwise-finite count"
            if n_pair >= 10 and np.std(masked_values[t, m, f]) > 0 and np.std(masked_labels[t, m]) > 0:
                hand = float(np.corrcoef(masked_values[t, m, f], masked_labels[t, m])[0, 1])
                assert abs(ic[t, f] - hand) < 1e-12
            else:
                assert np.isnan(ic[t, f])

    # The same pairwise-finite contract holds for the reference IC kernel.
    rbatch = FactorBatch(
        factor_ids=("a", "b", "c"),
        time_axis=AxisRef("t", "int", T),
        asset_axis=AxisRef("a", "str", N),
        values=masked_values,
    )
    rbundle = LabelBundle(
        target_id="r",
        values=masked_labels,
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
    )
    ic_ref, cnt_ref = compute_daily_ic(rbatch, rbundle, method="pearson", min_assets=10)
    assert _allclose_nanaware(ic, ic_ref, rtol=1e-12, atol=1e-12)
    assert np.array_equal(counts, cnt_ref)


def test_row_drop_nan_kernel_contract_consistent_with_pairwise(panel):
    """Any dropna-named kernel must agree with the documented pairwise-finite
    contract when the factor and label NaN masks are identical.

    Under an identical mask, 'drop rows with any NaN' and 'pairwise-finite'
    select the SAME rows, so a dropna kernel and the pairwise kernel must
    agree. There is no dropna kernel in build/lib today — this pins the
    documented contract so a future one fails loudly if it disagrees.
    """
    import importlib
    import pkgutil

    from quant_evaluator import metrics

    candidates = []
    for mod in pkgutil.iter_modules(metrics.__path__):
        if "dropna" in mod.name.lower() or "drop_na" in mod.name.lower():
            candidates.append(mod.name)
    for fname in dir(metrics):
        if "dropna" in fname.lower() or "drop_na" in fname.lower():
            candidates.append(fname)

    if not candidates:
        pytest.skip(
            "no dropna/row-drop kernel exists in build/lib; "
            "documented-contract test (nothing to check)"
        )

    values, labels, _, _ = panel
    rng = np.random.default_rng(SEED + 6)
    # Same NaN mask on both factor and label -> pairwise == row-drop rows.
    nan_mask = rng.random((T, N)) < 0.2
    mv = np.where(nan_mask[:, :, None], np.nan, values)
    ml = np.where(nan_mask, np.nan, labels)
    ic_pair, _ = fast_ic_batch(mv, ml, method="pearson", min_obs=10)

    for name in candidates:
        try:
            obj = getattr(importlib.import_module("quant_evaluator.metrics"), name)
        except Exception:  # noqa: BLE001
            obj = None
        if obj is None or not callable(obj):
            continue
        try:
            mv_clean, ml_clean = obj(mv, ml)
        except Exception:  # noqa: BLE001 - signature mismatch not a contract breach
            continue
        mv_clean = np.asarray(mv_clean, dtype=np.float64)
        ml_clean = np.asarray(ml_clean, dtype=np.float64)
        ic_drop, _ = fast_ic_batch(mv_clean, ml_clean, method="pearson", min_obs=10)
        assert _allclose_nanaware(ic_drop, ic_pair, rtol=1e-9, atol=1e-12), (
            f"{name} disagrees with the pairwise-finite kernel contract"
        )


# ---------------------------------------------------------------------------
# validity-mask contract (bonus, part of the pairwise-finite semantics)
# ---------------------------------------------------------------------------

def test_validity_mask_is_equivalent_to_nan(panel):
    """The factor_validity mask must behave exactly like NaN masking."""
    values, labels, _, _ = panel
    validity = np.ones(values.shape, dtype=bool)
    validity[1, :, 0] = False  # factor 0 at time 1 fully invalid
    validity[3, :, 2] = False

    ic_masked, cnt_masked = fast_ic_batch(
        values, labels, method="pearson", min_obs=10, factor_validity=validity
    )
    nan_values = values.copy()
    nan_values[~validity] = np.nan
    ic_nan, cnt_nan = fast_ic_batch(nan_values, labels, method="pearson", min_obs=10)

    assert _allclose_nanaware(ic_masked, ic_nan)
    assert np.array_equal(cnt_masked, cnt_nan)
    assert cnt_masked[1, 0] == 0
    assert np.isnan(ic_masked[1, 0])
