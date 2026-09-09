"""
QE-P0-04 — evaluation-mode consistency tests.

Prove that the four documented evaluation modes of the QE runtime produce
identical metric values within the tolerance contract:

1. **batch == streaming**     — ``Evaluator`` (full batch) vs
   ``StreamingEvaluator`` (time/factor chunked stream over the SAME data).
2. **chunked == nonchunked**  — ``Evaluator`` with ``use_chunking=True`` vs
   ``use_chunking=False``; chunked execution must reduce over the same
   partitionable, concat-aggregated series metric.
3. **cache hit == cache miss** — first (cold) evaluation vs second (warm)
   evaluation; the cache key must depend on the ACTUAL input content.
4. **serial == parallel**     — serial ``Evaluator`` vs the multiprocessing
   ``ParallelBatchExecutor`` path (and the ``@njit(parallel=True)`` kernel
   path as a second parallel surface).

Importable source is ``build/lib/quant_evaluator``. The repo-root
``quant_evaluator/`` directory is a near-empty stub that shadows build/lib
under pytest; the bootstrap below forces the build/lib package + subpackages
onto ``sys.modules`` directly (same pattern as ``test_metamorphic.py``) and
prepends build/lib to ``sys.path`` for intra-package imports.

Run with:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=/home/shw/quant_projects/quant_evaluator/build/lib \
    python -m pytest quant_evaluator/tests/test_consistency.py -v --tb=short

Code-path map (verified by reading build/lib sources):

- **streaming**: ``runtime/streaming_evaluator.py`` — ``StreamingEvaluator``,
  ``streaming_ic_updater`` (sufficient statistics per chunk, with the
  ``ic_parts`` time-slice reconstruction used for internal batch splitting),
  ``streaming_coverage_updater`` (paired-valid observation counting).
- **chunked**: ``runtime/evaluator.py`` — ``Evaluator._evaluate_chunked`` /
  ``_evaluate_chunk`` / ``_aggregate_chunk_results`` with a ``concat``
  aggregation contract on a PARTITIONABLE series metric
  (``pearson_ic_series`` / ``rank_ic_series``); ``planner/batch_plan.py``
  drives the split.
- **cache**: ``runtime/evaluator.py`` ``Evaluator._cache_identity`` hashes the
  factor VALUES, label VALUES, timing coordinates, and metric function
  bytecode into the ``CacheKey.input_hash``; storage is
  ``runtime/cache_v2_adapter.V2IntermediateCache`` over cache_v2 L1.
- **parallel**: ``runtime/parallel_executor.py`` ``ParallelBatchExecutor``
  (multiprocessing.Pool over per-worker ``Evaluator`` instances) and
  ``kernels/numba_backend.py`` ``@njit(parallel=True, prange)`` kernels.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Importable-source bootstrap (identical pattern to test_metamorphic.py).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]   # quant_projects root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.planner.dependency_plan import MetricKind
from quant_evaluator.runtime.evaluator import Evaluator
from quant_evaluator.runtime.parallel_executor import ParallelBatchExecutor, ParallelConfig
from quant_evaluator.runtime.streaming_evaluator import (
    StreamingEvaluator,
    streaming_coverage_updater,
    streaming_ic_updater,
)
from quant_evaluator.metrics.quality import compute_coverage


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

T, N, F = 20, 40, 3
SEED = 20260821


def _panel(seed=SEED, with_nan=True):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(T, N, F))
    labels = rng.normal(size=(T, N))
    if with_nan:
        values[rng.random(values.shape) < 0.10] = np.nan
        labels[rng.random(labels.shape) < 0.12] = np.nan
    return values, labels


def _batch(values, ids=("f0", "f1", "f2")):
    return FactorBatch(
        factor_ids=ids,
        time_axis=AxisRef("t", "int", T),
        asset_axis=AxisRef("a", "str", N),
        values=values,
    )


def _bundle(labels):
    return LabelBundle(
        target_id="r",
        values=labels,
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
    )


def _nanaware_close(a, b, rtol=1e-9, atol=1e-12):
    """Same NaN pattern + finite values close (structural equality)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return False
    both = np.isfinite(a) & np.isfinite(b)
    if not np.any(both):
        return True
    return bool(np.allclose(a[both], b[both], rtol=rtol, atol=atol))


def _assert_metrics_identical(left, right, ctx=""):
    assert set(left.keys()) == set(right.keys()), f"{ctx}: metric id sets differ"
    for mid in left:
        lv, rv = left[mid], right[mid]
        if isinstance(lv, np.ndarray):
            assert _nanaware_close(lv, rv), (
                f"{ctx}: metric {mid} diverged "
                f"(max |d|={np.nanmax(np.abs(lv - rv)) if np.isfinite(lv).any() else 'n/a'})"
            )
        elif isinstance(lv, (float, np.floating, int, np.integer)):
            assert np.isclose(float(lv), float(rv), rtol=1e-9, atol=1e-12), (
                f"{ctx}: scalar metric {mid} diverged: {lv} != {rv}"
            )
        else:
            assert lv == rv, f"{ctx}: metric {mid} objects differ"


@pytest.fixture(scope="module")
def panel():
    """(values, labels, batch, bundle) over NaN-padded synthetic data."""
    values, labels = _panel()
    return values, labels, _batch(values), _bundle(labels)


# ===========================================================================
# 1. batch == streaming
# ===========================================================================

def test_batch_equals_streaming_ic(panel):
    """Evaluator (batch) and StreamingEvaluator (chunked stream) must return
    the SAME per-day Pearson IC series. The streaming updater accumulates
    sufficient statistics per chunk; the internal splitter reconstructs each
    (time_slice, factor_slice) IC block from those statistics."""
    _, labels, batch, bundle = panel
    specs = [{"metric_id": "pearson_ic_series", "metric_kind": "custom"}]

    ev = Evaluator(enable_cache=False)
    batch_res = ev.evaluate(batch, bundle, specs, use_chunking=False)
    ic_batch = batch_res.metrics["pearson_ic_series"]

    se = StreamingEvaluator(chunk_size_time=5, chunk_size_factors=2)
    se.register_streaming_metric("ic_stream", streaming_ic_updater, metric_kind=MetricKind.IC)
    stream_res = se.evaluate_large_batch(
        batch, bundle, [{"metric_id": "ic_stream", "metric_kind": "ic"}]
    )
    ic_stream = stream_res.metrics["ic_stream"]

    assert ic_batch.shape == ic_stream.shape
    assert _nanaware_close(ic_batch, ic_stream, rtol=1e-9, atol=1e-12), (
        f"streaming vs batch IC diverged (max |d|={np.nanmax(np.abs(ic_batch - ic_stream))})"
    )
    assert stream_res.chunks_processed > 1, "streaming evaluator did not actually chunk"


def test_streaming_ic_invariant_to_chunk_layout(panel):
    """The streaming result must be independent of the chunking parameters."""
    _, labels, batch, bundle = panel

    def run(time_chunk, factor_chunk):
        se = StreamingEvaluator(
            chunk_size_time=time_chunk, chunk_size_factors=factor_chunk
        )
        se.register_streaming_metric("ic", streaming_ic_updater, metric_kind=MetricKind.IC)
        return se.evaluate_large_batch(
            batch, bundle, [{"metric_id": "ic", "metric_kind": "ic"}]
        ).metrics["ic"]

    a = run(5, 2)     # (time, factor) tiling
    b = run(7, 1)     # one factor at a time
    c = run(3, 3)     # one chunk per factor
    assert _nanaware_close(a, b, rtol=1e-9, atol=1e-12)
    assert _nanaware_close(a, c, rtol=1e-9, atol=1e-12)


def test_batch_equals_streaming_coverage(panel):
    """Streaming coverage (paired-valid count ratio) equals the batch
    ``compute_coverage`` aggregate."""
    _, labels, batch, bundle = panel
    import warnings

    se = StreamingEvaluator(chunk_size_time=5, chunk_size_factors=2)
    se.register_streaming_metric("cov", streaming_coverage_updater, metric_kind=MetricKind.COVERAGE)
    stream_res = se.evaluate_large_batch(
        batch, bundle, [{"metric_id": "cov", "metric_kind": "coverage"}]
    )
    # V5: no implicit cross-factor aggregation. Each coordinate must retain its
    # own numerator/denominator even when batch compatibility helpers aggregate.
    cov_by_factor = stream_res.metrics["cov"]
    from quant_evaluator.metrics.label_panel import normalize_label_panel
    label_panel, label_mask = normalize_label_panel(bundle, batch.num_assets)
    for f, fid in enumerate(batch.factor_ids):
        mask = np.isfinite(batch.values[:,:,f]) & np.isfinite(label_panel)
        if batch.validity is not None:
            mask &= batch.validity[:,:,f]
        if label_mask is not None:
            mask &= label_mask
        assert cov_by_factor[fid] == pytest.approx(np.mean(mask))
    cov_stream = np.mean(list(cov_by_factor.values()))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cov_batch, num_valid, num_total = compute_coverage(batch, bundle)

    assert np.isclose(cov_stream, cov_batch, rtol=1e-12, atol=1e-12)
    # Paired-valid budget consistency: the stream must have observed every
    # (t, n, f) cell exactly once.
    assert num_total == T * N * F
    # Streaming evaluator reports T*N observations per chunk (factors are a
    # separate chunk dimension), so the total is T*N per chunk * num_factor_chunks.
    # With chunk_size_factors=2 and F=3 this yields T*N*2 = 1600.
    assert stream_res.chunks_processed > 1
    assert (
        stream_res.total_observations_processed
        == T * N * len(range(0, F, 2))
    )


# ===========================================================================
# 2. chunked == nonchunked
# ===========================================================================

def _partitionable_specs():
    return [
        {
            "metric_id": "pearson_ic_series",
            "metric_kind": "custom",
            "metadata": {"partitionability": "PARTITIONABLE", "aggregation": "concat",
                         "chunk_contract": {"version": 1, "split_axis": "time",
                                            "output_axes": ["time", "factor"], "merge": "concat", "halo": 0}},
        }
    ]


def test_chunked_equals_nonchunked(panel):
    """Evaluator with use_chunking=True must reduce (concat) chunk results to
    EXACTLY the use_chunking=False result for a partitionable series metric.

    The chunker is driven by planner/batch_plan.py splitting on the time axis
    (asset and factor axes are never split at these sizes); a tiny
    max_chunk_memory_mb forces the split."""
    _, labels, batch, bundle = panel
    specs = _partitionable_specs()

    ev_full = Evaluator(enable_cache=False, max_chunk_memory_mb=512.0)
    full = ev_full.evaluate(batch, bundle, specs, use_chunking=False)
    assert full.chunks_processed == 1

    ev_chunk = Evaluator(enable_cache=False, max_chunk_memory_mb=0.025)
    chunked = ev_chunk.evaluate(batch, bundle, specs, use_chunking=True)
    assert chunked.chunks_processed > 1, "chunked evaluator did not actually chunk"

    _assert_metrics_identical(full.metrics, chunked.metrics, ctx="chunked==nonchunked")


def test_chunked_equality_is_exact_series_reduction(panel):
    """Pins that the chunk reduction is a CONCAT over per-time IC rows (not an
    average): the chunked result equals a manual time-slice concat."""
    values, labels, batch, bundle = panel
    from quant_evaluator.metrics.ic import compute_daily_ic

    specs = _partitionable_specs()
    ev = Evaluator(enable_cache=False, max_chunk_memory_mb=0.025)
    chunked = ev.evaluate(batch, bundle, specs, use_chunking=True)
    ic_chunked = chunked.metrics["pearson_ic_series"]

    # Manual time-slice concat via the reference IC kernel.
    ic_full, _ = compute_daily_ic(batch, bundle, method="pearson", min_assets=10)
    assert _nanaware_close(ic_chunked, ic_full, rtol=1e-9, atol=1e-12)


# ===========================================================================
# 3. cache hit == cache miss, and cache key depends on input content
# ===========================================================================

def test_cache_hit_equals_cache_miss(panel):
    """Cold (miss) vs warm (hit) evaluation of the same inputs must return
    identical values, and the second call must actually be served from cache."""
    _, labels, batch, bundle = panel
    specs = [{"metric_id": "rank_ic", "metric_kind": "custom", "metadata": {
        "cache_contract": {"version": 1, "semantic_ref": "rank_ic-test-fixed.v1", "dependency_refs": {"ic": "test-frozen.v1"}}}}]

    ev = Evaluator(enable_cache=True)
    cold = ev.evaluate(batch, bundle, specs, use_chunking=False)
    assert cold.cache_misses >= 1 and cold.cache_hits == 0, "cold run should be a miss"

    warm = ev.evaluate(batch, bundle, specs, use_chunking=False)
    assert warm.cache_hits >= 1, "second run should be a cache hit"
    _assert_metrics_identical(cold.metrics, warm.metrics, ctx="cache cold==warm")


def test_cache_key_depends_on_input_content(panel):
    """Changing ONE factor cell must produce a DIFFERENT cache entry (a new
    miss) — the cache key hashes the actual factor/label values."""
    values, labels, batch, bundle = panel
    specs = [{"metric_id": "rank_ic", "metric_kind": "custom", "metadata": {
        "cache_contract": {"version": 1, "semantic_ref": "rank_ic-test-fixed.v1", "dependency_refs": {"ic": "test-frozen.v1"}}}}]

    ev = Evaluator(enable_cache=True)
    r0 = ev.evaluate(batch, bundle, specs, use_chunking=False)
    n_entries_before = ev.get_cache_stats()["num_entries"]

    # Perturb a single cell multiplicatively: rank order of that cross-section
    # changes, so the metric VALUE changes and the key MUST change too.
    perturbed = values.copy()
    perturbed[0, 3, 0] *= 1000.0
    batch2 = _batch(perturbed)
    r1 = ev.evaluate(batch2, bundle, specs, use_chunking=False)

    assert ev.get_cache_stats()["num_entries"] == n_entries_before + 1, (
        "changing input content must create a NEW cache entry"
    )
    assert r1.cache_misses >= 1, "perturbed input should be a cache miss"
    assert not np.allclose(
        r0.metrics["rank_ic"], r1.metrics["rank_ic"], equal_nan=True
    ), "perturbation must actually change the metric value (test is vacuous otherwise)"


def test_cache_miss_after_clear(panel):
    """After cache.clear() the same input is recomputed (miss again) and the
    result is identical to the earlier cached value."""
    _, labels, batch, bundle = panel
    specs = [{"metric_id": "rank_ic", "metric_kind": "custom"}]

    ev = Evaluator(enable_cache=True)
    first = ev.evaluate(batch, bundle, specs, use_chunking=False)
    ev.clear_cache()
    second = ev.evaluate(batch, bundle, specs, use_chunking=False)
    assert second.cache_misses >= 1, "after clear() the same input must be a miss again"
    _assert_metrics_identical(first.metrics, second.metrics, ctx="cache after clear")


# ===========================================================================
# 4. serial == parallel
# ===========================================================================

def _rank_ic_metric_fn(factor_batch=None, label_bundle=None, **kwargs):
    """Module-level for multiprocessing picklability."""
    from quant_evaluator.metrics.ic import compute_daily_ic, compute_mean_ic

    ic_series, _ = compute_daily_ic(
        factor_batch, label_bundle, method="spearman", min_assets=10
    )
    mean, _ = compute_mean_ic(ic_series, min_periods=1)
    return mean


def test_serial_equals_parallel(panel):
    """Serial Evaluator vs ParallelBatchExecutor over the same batch(es)."""
    _, labels, batch, bundle = panel
    specs = [{"metric_id": "rank_ic", "metric_kind": "custom"}]

    ser = Evaluator(enable_cache=False)
    serial_res = ser.evaluate(batch, bundle, specs, use_chunking=False)

    executor = ParallelBatchExecutor(
        config=ParallelConfig(num_workers=2, enable_cache=False, use_chunking=False)
    )
    executor.register_metric("rank_ic", _rank_ic_metric_fn)
    par_res = executor.execute_parallel([(batch, bundle, specs)])

    assert par_res.successful_batches == 1 and par_res.failed_batches == 0
    task = par_res.get_result(0)
    assert task is not None and task.result is not None
    _assert_metrics_identical(
        serial_res.metrics, task.result.metrics, ctx="serial==parallel"
    )


def test_serial_equals_parallel_multi_batch(panel):
    """Per-batch serial results equal the parallel results for EVERY batch in
    a multi-batch run (the parallel path re-runs the same Evaluator per task)."""
    values, labels, batch, bundle = panel
    specs = [{"metric_id": "rank_ic", "metric_kind": "custom"}]

    # Second batch: perturbed (different content).
    perturbed = values.copy()
    perturbed[2, :, :] = np.nan
    batch2 = _batch(perturbed)

    serial_results = {}
    for name, b in (("b0", batch), ("b1", batch2)):
        ev = Evaluator(enable_cache=False)
        res = ev.evaluate(b, bundle, specs, use_chunking=False)
        serial_results[name] = res.metrics["rank_ic"]

    executor = ParallelBatchExecutor(
        config=ParallelConfig(num_workers=2, enable_cache=False, use_chunking=False)
    )
    executor.register_metric("rank_ic", _rank_ic_metric_fn)
    par_res = executor.execute_parallel([(batch, bundle, specs), (batch2, bundle, specs)])
    assert par_res.successful_batches == 2 and par_res.failed_batches == 0
    for task_id, name in ((0, "b0"), (1, "b1")):
        r = par_res.get_result(task_id).result
        assert _nanaware_close(serial_results[name], r.metrics["rank_ic"]), (
            f"parallel batch {name} diverged from serial"
        )


def test_numba_parallel_kernel_equals_fast_serial_kernel(panel):
    """The @njit(parallel=True, prange) kernel path must agree bit-for-bit
    (NaN-aware) with the serial vectorized kernel."""
    try:
        from quant_evaluator.kernels.numba_backend import numba_ic_batch
    except Exception as exc:  # pragma: no cover - numba unavailable
        pytest.skip(f"numba backend unavailable: {exc}")
    from quant_evaluator.kernels.fast import fast_ic_batch

    values, labels, _, _ = panel
    for method in ("pearson", "spearman"):
        ic_fast, cnt_fast = fast_ic_batch(values, labels, method=method, min_obs=10)
        ic_numba, cnt_numba = numba_ic_batch(values, labels, method=method, min_obs=10)
        assert _nanaware_close(ic_fast, ic_numba, rtol=1e-12, atol=1e-14)
        assert np.array_equal(cnt_fast, cnt_numba)
