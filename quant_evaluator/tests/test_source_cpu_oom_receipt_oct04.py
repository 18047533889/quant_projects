"""CPU source-run receipts must state zero OOM retries without weakening CUDA."""

import numpy as np
import pandas as pd
import pytest

from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.scripts import benchmark_real_cos_source_batch as benchmark
from quant_evaluator.scripts.source_execution_receipt import validate_source_execution_receipt


class _MemorySource:
    """Small typed source double; factor evaluation and receipt building are real."""
    def __init__(self, records, source_rows, dates, assets, labels, manifest_sha,
                 tile_size, max_object_mib, prefetch_objects=False):
        self.factor_ids = tuple(row[0] for row in records)
        self.time_axis = AxisRef("time", "datetime64[ns]", len(dates),
                                 dates.to_numpy(dtype="datetime64[ns]"))
        self.asset_axis = AxisRef("asset", "str", len(assets),
                                  np.asarray(assets, dtype=str))
        self.dtype = "float64"
        self.snapshot_id = "in-memory-cpu-receipt"
        self.max_tile_size = tile_size
        self.prefetch_objects = prefetch_objects
        self.next_start = 0
        self.reads = []
        self.tile_read_timings = []
        rng = np.random.default_rng(511)
        self.values = rng.normal(size=(len(dates), len(assets), len(records)))

    def read_tile(self, start, end):
        assert start == self.next_start
        batch = FactorBatch(self.factor_ids[start:end], self.time_axis,
                            self.asset_axis, self.values[:, :, start:end])
        self.reads.append((start, end))
        self.next_start = end
        return FactorTile(start, end, batch, self.snapshot_id)

    def close(self):
        pass


def _labels(dates, assets):
    values = np.random.default_rng(512).normal(size=(len(dates), len(assets)))
    times = tuple(dates.to_numpy(dtype="datetime64[ns]"))
    return LabelBundle(
        "cpu-receipt-label", values, 1,
        decision_time=times,
        observation_time=times,
        signal_available_time=times,
        execution_time=times,
        label_start_time=tuple(dates.to_numpy(dtype="datetime64[ns]")),
        label_end_time=tuple(
            dates.to_numpy(dtype="datetime64[ns]") + np.timedelta64(1, "D")),
        asset_axis=AxisRef("asset", "str", len(assets),
                           np.asarray(assets, dtype=str)),
    )


def test_real_cpu_source_api_run_backend_receipt_states_integer_zero_oom(monkeypatch):
    monkeypatch.setattr(benchmark, "RealCosSource", _MemorySource)
    dates = pd.date_range("2024-01-01", periods=24, freq="B")
    assets = tuple(f"A{i:03d}" for i in range(32))
    records = tuple((f"f{i}", f"cos://object/{i}", f"{i:064x}", 100)
                    for i in range(3))
    source_rows = tuple((row[0], row[1], row[2], row[3], None, "b" * 64)
                        for row in records)

    bundle, receipt = benchmark.run_backend(
        "cpu", records, source_rows, dates, assets, _labels(dates, assets),
        "a" * 64, 2, 1, GPUExecutionPolicy(), ("pearson_ic",),
        source_adapter="legacy",
    )

    assert bundle.metadata["backend_used"] == "cpu"
    assert receipt["backend_used"] == "cpu"
    assert receipt["oom_retries"] == 0
    assert type(receipt["oom_retries"]) is int
    assert receipt["factor_tiles_processed"] == 2
    assert receipt["tile_ranges"] == [(0, 2), (2, 3)]


def test_cuda_schedule_validator_still_requires_explicit_integer_zero_oom():
    with pytest.raises(ValueError, match="CUDA qualification requires"):
        validate_source_execution_receipt(
            reads=[(0, 1)], factor_count=1, admitted_cap=1,
            metadata={"backend_used": "cuda", "factor_tile_size": 1,
                      "factor_tiles_processed": 1},
        )
