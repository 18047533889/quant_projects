"""Producer regressions: request caps are not actual CUDA execution widths."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from quant_evaluator.scripts.source_execution_receipt import (
    validate_source_execution_receipt,
)
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.scripts import benchmark_real_cos_source_batch as harness


def _run(monkeypatch, *, width=4, cap=5, oom=0, reads=None, count=2):
    records = tuple((f"f{i}", f"cos://test/f{i}", "0" * 64, 1)
                    for i in range(5))
    rows = tuple((*row, "etag", "1" * 64) for row in records)
    source = SimpleNamespace(
        factor_ids=tuple(row[0] for row in records), reads=[],
        max_tile_size=cap, snapshot_id="test", prefetch_objects=False,
        tile_read_timings=[], closed=False,
    )
    source.close = lambda: setattr(source, "closed", True)
    monkeypatch.setattr(harness, "RealCosSource", lambda *a, **kw: source)

    def evaluate(*args, **kwargs):
        source.reads.extend([(0, 4), (4, 5)] if reads is None else reads)
        return SimpleNamespace(metadata={
            "backend_used": "cuda", "effective_max_tile_size": cap,
            "factor_tile_size": width, "oom_retries": oom,
            "factor_tiles_processed": count,
            "source_request_fingerprint": "2" * 64,
        }, observation_counts={})

    monkeypatch.setattr(harness, "evaluate_factor_source_batch", evaluate)
    result = harness.run_backend(
        "cuda_strict", records, rows, None, None, None, "3" * 64,
        cap, 16, GPUExecutionPolicy(), harness.PEARSON_SINGLE)
    assert source.closed
    return result[1]


def test_cuda_actual_width_may_be_below_request_cap(monkeypatch):
    receipt = _run(monkeypatch)
    assert receipt["effective_max_tile_size"] == 5
    assert receipt["actual_source_tile_size"] == 4
    assert receipt["actual_gpu_factor_tile_size"] == 4
    assert receipt["tile_ranges"] == [(0, 4), (4, 5)]
    assert len(receipt["execution_schedule_sha256"]) == 64


@pytest.mark.parametrize("kwargs", [
    {"width": True}, {"width": 4.0}, {"width": 6},
    {"oom": False}, {"oom": 0.0}, {"oom": 1},
    {"count": True}, {"count": 3},
    {"reads": [(0, 3), (3, 5)]},
    {"reads": [(0, 4), (3, 5)]},
    {"reads": [(0, 4)]},
])
def test_malformed_or_deviated_gpu_schedule_cannot_be_certified(monkeypatch, kwargs):
    with pytest.raises(ValueError):
        _run(monkeypatch, **kwargs)


def test_cpu_schedule_binds_request_width_not_gpu_metadata():
    receipt = validate_source_execution_receipt(
        reads=[(0, 5)], factor_count=5, admitted_cap=5,
        metadata={"backend_used": "cpu", "factor_tiles_processed": 1,
                  "factor_tile_size": 4})
    assert receipt["actual_source_tile_size"] == 5
    assert receipt["actual_gpu_factor_tile_size"] is None
    assert receipt["compute_tile_ranges"] == ((0, 5),)


@pytest.mark.parametrize("field,value", [
    ("factor_count", True), ("factor_count", 0), ("factor_count", 5.0),
    ("admitted_cap", True), ("admitted_cap", 0), ("admitted_cap", 5.0),
    ("reads", [[0, 5]]), ("reads", [(False, 5)]),
    ("reads", [(0, 5.0)]), ("reads", None),
    ("metadata", None),
])
def test_receipt_rejects_type_shortcuts_and_missing_fields(field, value):
    kwargs = {"factor_count": 5, "admitted_cap": 5, "reads": [(0, 5)],
              "metadata": {"backend_used": "cpu", "factor_tiles_processed": 1}}
    kwargs[field] = value
    with pytest.raises(ValueError):
        validate_source_execution_receipt(**kwargs)


@pytest.mark.parametrize("metadata", [
    {"backend_used": "cpu", "factor_tiles_processed": True},
    {"backend_used": "cuda", "factor_tiles_processed": 1},
    {"backend_used": "auto", "factor_tiles_processed": 1},
    {"factor_tiles_processed": 1},
])
def test_actual_backend_and_complete_execution_metadata_are_required(metadata):
    with pytest.raises(ValueError):
        validate_source_execution_receipt(
            reads=[(0, 5)], factor_count=5, admitted_cap=5, metadata=metadata)


@pytest.mark.skipif(os.environ.get("QE_SOURCE_EXECUTION_CUDA_TEST") != "1",
                    reason="opt in to actual CUDA source execution")
def test_real_cuda_cap129_actual128_schedule_and_scipy_pearson_oracle():
    from scipy.stats import pearsonr
    from quant_evaluator.api.factor_source import evaluate_factor_source_batch
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.factor_tile_source import FactorTile
    from quant_evaluator.contracts.label_bundle import LabelBundle

    rng = np.random.default_rng(61003)
    # The canonical device candidate list tops out at 128. No estimator or
    # memory-budget mocking is needed to expose cap != actual width.
    T, N, F = 32, 48, 129
    x = rng.normal(size=(T, N, F))
    y = rng.normal(size=(T, N))
    ta = AxisRef("time", "int64", T, np.arange(T, dtype=np.int64))
    aa = AxisRef("asset", "int64", N, np.arange(N, dtype=np.int64))
    ids = tuple(f"f{i}" for i in range(F))
    label = LabelBundle("return", y, 1, decision_time=tuple(range(T)),
                        label_start_time=tuple(range(T)),
                        label_end_time=tuple(range(1, T + 1)), asset_axis=aa)

    class Source:
        factor_ids = ids
        time_axis = ta
        asset_axis = aa
        dtype = "float64"
        snapshot_id = "small-synthetic-device-check"
        max_tile_size = 129

        def __init__(self):
            self.reads = []

        def read_tile(self, start, end):
            self.reads.append((start, end))
            return FactorTile(start, end, FactorBatch(
                ids[start:end], ta, aa, x[:, :, start:end]), self.snapshot_id)

        def close(self):
            pass

    expected = np.array([[pearsonr(x[t, :, f], y[t]).statistic
                          for f in range(F)] for t in range(T)])
    for backend, width in (("cpu", 129), ("cuda_strict", 128)):
        source = Source()
        result = evaluate_factor_source_batch(
            source, label, metrics=("pearson_ic", "pearson_ic_series"),
            backend=backend, max_tile_size=129)
        receipt = validate_source_execution_receipt(
            reads=source.reads, factor_count=F,
            admitted_cap=result.metadata["effective_max_tile_size"],
            metadata=result.metadata)
        assert result.metadata["effective_max_tile_size"] == 129
        assert receipt["actual_source_tile_size"] == width
        np.testing.assert_allclose(result.series_metrics["pearson_ic_series"],
                                   expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(result.scalar_metrics["pearson_ic"],
                                   expected.mean(axis=0), rtol=1e-12, atol=1e-14)
        for metric in ("pearson_ic", "pearson_ic_series"):
            np.testing.assert_array_equal(result.observation_counts[metric],
                                          np.full(F, T, dtype=np.int64))
