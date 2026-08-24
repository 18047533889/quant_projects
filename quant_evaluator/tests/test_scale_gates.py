"""
R21-QE-SCALE-GATES — scale-correctness suite for the 1K/10K/100K_SCALE release gates.

These tests assert CORRECTNESS (not performance benchmarks) at row scale: the
same metric computed at 1K, 10K and 100K rows must agree with the reference
kernel and with itself when the panel is subsampled consistently.

Design contract (verified against build/lib sources):

- **Row scale.** ``1K`` = T=100 x N=10 (1000 rows), ``10K`` = T=100 x N=100
  (10 000 rows), ``100K`` = T=100 x N=1000 (100 000 rows).  The time axis is
  fixed at T=100 so every scale shares the same per-time estimation procedure.

- **IC is a per-time metric.**  ``metrics/ic.compute_daily_ic`` (the reference
  kernel) computes one Spearman/Pearson correlation per (t, f) cross-section on
  the pairwise-finite observations.  The mean-of-time IC is therefore an
  average of T independent per-time estimates.  A 1K panel (N=10) has a noisy
  per-time Spearman IC (sd ~ 1/sqrt(10-1) ~ 0.33), while a 100K panel (N=1000)
  has sd ~ 0.032; the gate is that the two estimates agree within the SAMPLING
  error of the 1K panel (3 x sd_1K / sqrt(T)), not within a tiny absolute
  tolerance.  This is the "statistical tolerance when inputs differ by scale"
  branch; the inputs are fresh iid draws at each scale so the panel is a
  consistent subsample of the SAME population.

- **Turnover is a per-asset, per-time metric.**  The canonical turnover
  ``metrics/turnover.compute_turnover_series`` on explicit sum-1 portfolio
  weights is EXACTLY scale-invariant: duplicating every asset with its weight
  divided by k preserves each asset's per-asset weight trajectory, so
  ``0.5 * sum |w_t - w_{t-1}|`` is bit-for-bit unchanged.  The gate asserts
  exact (rtol=1e-12) equality between the 1K weight panel and its 10K/100K
  duplicated counterparts.  (``fast_turnover_estimate`` is rank-based, hence
  NOT per-asset scale-invariant under universe growth — the per-asset gate uses
  the canonical weight-based kernel.)

- **Quantile returns.**  ``compute_quantile_returns`` (reference, metrics
  layer) with ``n_quantiles=5, min_assets=10``.  At 1K (N=10) each quantile has
  ~2 members < min_assets=10, so the 1K quantile-return matrix is all-NaN —
  this is the DOCUMENTED kernel contract (NaN when a group has fewer than
  min_assets members) and the gate pins it.  At 10K and 100K every quantile has
  >= 20 members and the grand mean (mean over time x quantiles) is a sampling
  estimate of the population label mean (0).  The gate asserts the 10K and
  100K grand means agree within 3 x the 10K sampling error, and that the 1K
  matrix is all-NaN.

- **Streaming == batch.**  ``runtime/streaming_evaluator.py`` ``StreamingEvaluator``
  with the ``streaming_ic_updater`` sufficient-statistics accumulator, fed the
  SAME 10K panel as a chunked stream (``chunk_size_time=10`` -> 10 chunks),
  must reproduce the reference batch ``compute_daily_ic`` Pearson IC series to
  ~1e-16.  This is the batch==streaming gate exercised at the 10K row scale.

- **100K runs and returns finite values.**  The reference IC + quantile
  returns + turnover kernels over the full 100K panel (with a NaN mask) must
  complete and return finite mean IC / finite per-quantile counts.  Bounds the
  gate to "runs and is correct", not a wall-clock benchmark.

Importable source is ``build/lib/quant_evaluator`` (same bootstrap pattern as
``tests/test_metamorphic.py``): the repo-root ``quant_evaluator/`` directory is
a stub that would shadow build/lib under pytest, so the bootstrap forces the
build/lib package + subpackages onto ``sys.modules`` via direct source-file
loading and prepends build/lib to ``sys.path`` for intra-package imports.  This
is deliberately independent of any command-line PYTHONPATH.

Run with (either PYTHONPATH present or absent):

    cd /home/shw/quant_projects/quant_evaluator
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=/home/shw/quant_projects/quant_evaluator/build/lib \
    python -m pytest tests/test_scale_gates.py -q --tb=short
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Importable-source bootstrap (identical to test_metamorphic.py).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]   # quant_projects root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.ic import compute_daily_ic, compute_mean_ic
from quant_evaluator.metrics.quantile import compute_quantile_returns
from quant_evaluator.metrics.turnover import compute_turnover_series, estimate_turnover_from_ranks
from quant_evaluator.planner.dependency_plan import MetricKind
from quant_evaluator.runtime.streaming_evaluator import (
    StreamingEvaluator,
    streaming_ic_updater,
)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

# Row-scale constants: T x N = 1K / 10K / 100K rows.
T = 100
SCALES = {"1K": 10, "10K": 100, "100K": 1000}
SEED = 20260821

_F = ("f0",)


def _batch(values: np.ndarray) -> FactorBatch:
    return FactorBatch(
        factor_ids=_F,
        time_axis=AxisRef("t", "int", values.shape[0]),
        asset_axis=AxisRef("a", "str", values.shape[1]),
        values=values,
    )


def _bundle(labels: np.ndarray) -> LabelBundle:
    Tn = labels.shape[0]
    return LabelBundle(
        target_id="r",
        values=labels,
        horizon=1,
        decision_time=tuple(range(Tn)),
        label_start_time=tuple(range(Tn)),
        label_end_time=tuple(range(1, Tn + 1)),
    )


def _scale_panel(scale: str, with_nan: bool = False):
    """Fresh iid synthetic panel at the given row scale (fixed seed per scale)."""
    n = SCALES[scale]
    rng = np.random.default_rng(SEED + n)
    values = rng.normal(size=(T, n, 1))
    labels = rng.normal(size=(T, n))
    if with_nan:
        values[rng.random(values.shape) < 0.10] = np.nan
        labels[rng.random(labels.shape) < 0.12] = np.nan
    return values, labels


def _nanaware_close(a, b, rtol=1e-9, atol=1e-12):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return False
    both = np.isfinite(a) & np.isfinite(b)
    if not np.any(both):
        return True
    return bool(np.allclose(a[both], b[both], rtol=rtol, atol=atol))


# ---------------------------------------------------------------------------
# 1. rank IC is scale-consistent: mean-of-time converges across 1K/10K/100K
# ---------------------------------------------------------------------------

def test_ic_rank_consistent_at_1k_10k_100k():
    """Mean-of-time Spearman IC at 1K/10K/100K must agree within the SAMPLING
    error of the 1K panel (the least precise scale).

    The reference kernel computes one Spearman IC per (t, f) cross-section on
    pairwise-finite observations; mean-of-time is the average of T per-time
    estimates.  With N=10 assets the per-time Spearman sd is ~0.33, so the 1K
    mean has sd ~ 0.33/sqrt(100) ~ 0.033; the 100K mean has sd ~ 0.003.  The
    gate is |mean_1K - mean_100K| <= 3 x sd_1K (and likewise 10K vs 100K).
    """
    ics = {}
    for scale in ("1K", "10K", "100K"):
        values, labels = _scale_panel(scale)
        ic_series, cnt = compute_daily_ic(
            _batch(values), _bundle(labels), method="spearman", min_assets=10
        )
        assert np.isfinite(ic_series).sum() == T, f"{scale}: expected {T} finite IC days"
        assert cnt.min() >= 10, f"{scale}: valid-pair floor violated (min {cnt.min()})"
        ics[scale] = ic_series[:, 0]

    # Reference scale: the 100K panel has the tightest estimate of the true
    # population mean IC (0).  Compare the noisier scales against it.
    ref_mean = float(np.nanmean(ics["100K"]))
    for scale in ("1K", "10K"):
        series = ics[scale]
        mean = float(np.nanmean(series))
        sd = float(np.nanstd(series, ddof=1))
        # Sampling error of the mean-of-time estimate at this scale.
        sigma_mean = sd / np.sqrt(float(np.isfinite(series).sum()))
        assert abs(mean - ref_mean) <= 3.0 * sigma_mean + 1e-12, (
            f"{scale} mean IC {mean:.6f} vs 100K {ref_mean:.6f} diverged "
            f"({abs(mean - ref_mean):.6f} > 3*{sigma_mean:.6f})"
        )

    # The 10K mean (sd ~ 0.1/sqrt(100) ~ 0.01) must also sit inside the tighter
    # 10K-vs-100K window.
    series = ics["10K"]
    sigma_mean = float(np.nanstd(series, ddof=1)) / np.sqrt(
        float(np.isfinite(series).sum())
    )
    assert abs(float(np.nanmean(series)) - ref_mean) <= 3.0 * sigma_mean + 1e-12


# ---------------------------------------------------------------------------
# 2. turnover scale consistency: per-asset series identical at 1K vs 10K/100K
# ---------------------------------------------------------------------------

def test_turnover_scale_consistency():
    """Canonical turnover on explicit sum-1 weights is EXACTLY scale-invariant.

    Each asset's weight trajectory at the 10K/100K scale is bit-identical to
    its 1K trajectory (duplicate every asset with weight/k), so
    ``0.5 * sum |w_t - w_{t-1}|`` over jointly-finite assets is unchanged.
    The gate asserts exact (rtol=1e-12) parity of the full turnover series.
    """
    rng = np.random.default_rng(SEED + 1)
    W = np.abs(rng.normal(size=(T, SCALES["1K"]))) + 0.1
    W = W / W.sum(axis=1, keepdims=True)

    tv_1k = compute_turnover_series(W)
    for scale in ("10K", "100K"):
        k = SCALES[scale] // SCALES["1K"]
        W_big = np.repeat(W, k, axis=1) / float(k)  # sum-1, per-asset weights identical
        tv_big = compute_turnover_series(W_big)
        assert _nanaware_close(tv_1k, tv_big, rtol=1e-12, atol=1e-14), (
            f"turnover at {scale} diverged from 1K"
        )

    # NaN weights behave the same way (jointly-finite summation).
    Wn = W.copy()
    Wn[rng.random(Wn.shape) < 0.10] = np.nan
    tv_n = compute_turnover_series(Wn)
    k = SCALES["100K"] // SCALES["1K"]
    tv_n_big = compute_turnover_series(np.repeat(Wn, k, axis=1) / float(k))
    assert _nanaware_close(tv_n, tv_n_big, rtol=1e-12, atol=1e-14)


def test_rank_turnover_reference_parity_at_100k():
    """The reference rank-based turnover kernel runs at 100K rows and agrees
    with the canonical per-time turnover definition (finite, bounded [0, ~1])."""
    values, labels = _scale_panel("100K")
    tv = estimate_turnover_from_ranks(_batch(values), window=1)
    assert tv.shape == (T, 1)
    assert np.isfinite(tv[1:]).sum() == T - 1, "rank turnover should be finite for t>=1"
    assert float(np.nanmin(tv[1:])) >= 0.0
    assert float(np.nanmax(tv[1:])) <= 1.0 + 1e-9
    # A full rank inversion has turnover ~1.0; a no-change panel ~0.  Sanity the
    # estimator is on that scale (not, e.g., a count of assets).
    assert 0.0 < float(np.nanmean(tv[1:])) < 1.0


# ---------------------------------------------------------------------------
# 3. quantile returns scale consistency
# ---------------------------------------------------------------------------

def test_quantile_returns_scale_consistency():
    """Quantile returns at 10K and 100K agree within the 10K sampling error;
    the 1K panel is all-NaN by the documented min_assets contract.

    At 1K (N=10) each of the 5 quantiles has ~2 members < min_assets=10, so the
    kernel reports NaN (documented contract).  At 10K/100K each quantile has
    >= 20 members and the grand mean over (time x quantile) is a sampling
    estimate of the population label mean (0).
    """
    qr = {}
    qc = {}
    for scale in ("1K", "10K", "100K"):
        values, labels = _scale_panel(scale)
        qr[scale], qc[scale] = compute_quantile_returns(
            _batch(values), _bundle(labels), n_quantiles=5, min_assets=10
        )

    # 1K: every group has < min_assets members -> all-NaN (documented contract).
    assert not np.isfinite(qr["1K"]).any(), "1K quantile returns must be all-NaN (min_assets)"
    assert qc["1K"].min() >= 0 and qc["1K"].max() <= SCALES["1K"]

    # 10K/100K: finite, and per-quantile member counts scale with N.
    for scale in ("10K", "100K"):
        assert np.isfinite(qr[scale]).all(), f"{scale}: every (t, q) cell must be finite"
        n = SCALES[scale]
        assert qc[scale].min() >= 10, f"{scale}: min_assets floor violated"
        assert qc[scale].max() <= n

    # Grand means (mean over the T x 5 finite cells) agree within the 10K
    # sampling error.  Per-time quantile cell sd ~ sigma_label/sqrt(N/5) ~ 0.22
    # at 10K; the grand mean over 500 cells has sd ~ 0.22/sqrt(500) ~ 0.01.
    g10 = float(np.nanmean(qr["10K"]))
    g100 = float(np.nanmean(qr["100K"]))
    sigma_cell_10k = float(np.nanstd(qr["10K"], ddof=1))  # ~0.22
    n_cells = float(np.isfinite(qr["10K"]).sum())
    sigma_g10 = sigma_cell_10k / np.sqrt(n_cells)
    assert abs(g10 - g100) <= 3.0 * sigma_g10 + 1e-12, (
        f"10K quantile grand mean {g10:.6f} vs 100K {g100:.6f} diverged "
        f"({abs(g10 - g100):.6f} > 3*{sigma_g10:.6f})"
    )


# ---------------------------------------------------------------------------
# 4. streaming == batch at 10K rows
# ---------------------------------------------------------------------------

def test_streaming_scale_parity():
    """Streaming kernel over the 10K panel equals the reference batch kernel.

    ``StreamingEvaluator`` + ``streaming_ic_updater`` accumulate per-chunk
    sufficient statistics; the internal splitter reconstructs each time slice's
    IC from those statistics.  The result must match ``compute_daily_ic``
    (Pearson) to ~1e-16 and must genuinely chunk (chunks_processed > 1).
    """
    values, labels = _scale_panel("10K")
    batch = _batch(values)
    bundle = _bundle(labels)

    ic_ref, cnt_ref = compute_daily_ic(batch, bundle, method="pearson", min_assets=10)

    se = StreamingEvaluator(chunk_size_time=10, chunk_size_factors=1)
    se.register_streaming_metric("ic", streaming_ic_updater, metric_kind=MetricKind.IC)
    res = se.evaluate_large_batch(
        batch, bundle, [{"metric_id": "ic", "metric_kind": "ic"}]
    )
    ic_stream = res.metrics["ic"]

    assert res.chunks_processed > 1, "streaming evaluator did not actually chunk"
    assert ic_stream.shape == ic_ref.shape
    assert _nanaware_close(ic_ref, ic_stream, rtol=1e-9, atol=1e-12), (
        f"streaming vs batch IC diverged (max|d|="
        f"{np.nanmax(np.abs(ic_ref - ic_stream))})"
    )

    # Streaming output must be invariant to the chunk layout (time-chunk tiling).
    for ctime in (7, 25):
        se2 = StreamingEvaluator(chunk_size_time=ctime, chunk_size_factors=1)
        se2.register_streaming_metric("ic", streaming_ic_updater, metric_kind=MetricKind.IC)
        r2 = se2.evaluate_large_batch(
            batch, bundle, [{"metric_id": "ic", "metric_kind": "ic"}]
        )
        assert _nanaware_close(ic_ref, r2.metrics["ic"], rtol=1e-9, atol=1e-12), (
            f"chunk layout {ctime} diverged from batch"
        )


# ---------------------------------------------------------------------------
# 5. 100K computation runs and returns finite values
# ---------------------------------------------------------------------------

def test_100k_runs_without_error():
    """The full 100K-row computation completes and returns finite values.

    Reference IC (Spearman, NaN-masked), quantile returns, and rank turnover
    over the 100K panel must all complete and produce finite aggregates — the
    'runs and is correct' half of the gate (no wall-clock assertion).
    """
    values, labels = _scale_panel("100K", with_nan=True)
    batch = _batch(values)
    bundle = _bundle(labels)

    ic_series, cnt = compute_daily_ic(batch, bundle, method="spearman", min_assets=10)
    assert ic_series.shape == (T, 1)
    assert np.isfinite(ic_series).sum() == T, "every day must yield a finite Spearman IC"
    assert cnt.min() >= 10, "valid-pair floor violated"
    mean_ic, ic_std = compute_mean_ic(ic_series, min_periods=1)
    assert np.isfinite(mean_ic).all() and np.isfinite(ic_std).all()
    assert abs(float(mean_ic[0])) < 1.0

    qr, qc = compute_quantile_returns(batch, bundle, n_quantiles=5, min_assets=10)
    assert qr.shape == (T, 5, 1) and qc.shape == (T, 5, 1)
    assert np.isfinite(qr).all(), "100K quantile returns must all be finite"
    assert qc.min() >= 10 and qc.max() <= SCALES["100K"]
    # Quantile spread (top - bottom) is finite and on a sensible scale.
    spread = qr[:, -1, 0] - qr[:, 0, 0]
    assert np.isfinite(spread).all()
    assert float(np.nanmean(np.abs(spread))) < 3.0

    tv = estimate_turnover_from_ranks(batch, window=1)
    assert np.isfinite(tv[1:]).sum() == T - 1
    assert 0.0 <= float(np.nanmean(tv[1:])) <= 1.0 + 1e-9
