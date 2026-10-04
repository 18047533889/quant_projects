"""Source-reference orchestration gates, separate from independent metric math."""
import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.scripts.source_all24_oracle import (
    REFERENCE_METRICS, reference_source_all24,
)


class Source:
    def __init__(self, factors=3):
        self.time_axis = AxisRef("time", "int64", 4, np.arange(4, dtype=np.int64))
        self.asset_axis = AxisRef("asset", "int64", 20, np.arange(20, dtype=np.int64))
        self.factor_ids = tuple(f"f{i}" for i in range(factors))
        self.dtype = "float64"
        self.snapshot_id = "all24-fixture"
        self.max_tile_size = 3
        self.admitted_max_tile_size = 1
        self.reads = []
        self.closed = False

    def read_tile(self, start, end):
        self.reads.append((start, end))
        values = np.broadcast_to(np.arange(20)[None, :, None], (4, 20, end-start))
        batch = FactorBatch(self.factor_ids[start:end], self.time_axis,
                            self.asset_axis, values.astype(np.float64))
        return FactorTile(start, end, batch, self.snapshot_id)

    def close(self):
        self.closed = True


def labels_for(source, *, time_shift=0, asset_shift=0):
    times = source.time_axis.values + time_shift
    assets = AxisRef("asset", "int64", 20, source.asset_axis.values + asset_shift)
    return LabelBundle("forward", np.ones((4, 20)), 1,
                       decision_time=tuple(times), label_start_time=tuple(times),
                       label_end_time=tuple(times + 1), asset_axis=assets)


@pytest.mark.parametrize("option,value", [
    ("max_tile_size", 0), ("max_tile_size", True),
    ("max_result_bytes", 0), ("max_result_bytes", True),
    ("metrics", ("rank_ic",)), ("metrics", list(REFERENCE_METRICS)),
])
def test_malformed_request_is_rejected_before_source_reads(option, value):
    # Defect: an invalid request can begin expensive source reads.
    source = Source()
    with pytest.raises((TypeError, ValueError)):
        reference_source_all24(source, labels_for(source), **{option: value})
    assert source.reads == [] and not source.closed


def test_returned_array_budget_rejects_before_first_read():
    # Defect: output allocation happens before its budget is checked.
    source = Source()
    with pytest.raises(MemoryError):
        reference_source_all24(source, labels_for(source), max_result_bytes=1)
    assert source.reads == [] and not source.closed


@pytest.mark.parametrize("time_shift,asset_shift", [(1, 0), (0, 1)])
def test_misaligned_labels_rejected_before_first_read(time_shift, asset_shift):
    # Defect: source/label axis differences silently alter a real reference.
    source = Source()
    with pytest.raises(ValueError):
        reference_source_all24(source, labels_for(source, time_shift=time_shift,
                                                   asset_shift=asset_shift))
    assert source.reads == []


def _literal_parts(batch, labels):
    from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
    part = BatchEvaluationBundle(batch.factor_ids, labels.target_id)
    indices = np.array([int(name[1:]) for name in batch.factor_ids])
    for metric in REFERENCE_METRICS:
        if metric in ("rank_ic_series", "pearson_ic_series"):
            part.series_metrics[metric] = np.arange(4)[:, None] + indices[None, :]
        else:
            part.scalar_metrics[metric] = indices.astype(np.float64) + .5
        part.observation_counts[metric] = indices.astype(np.int64) + 1
    return (part,)


@pytest.mark.parametrize("metrics", [REFERENCE_METRICS, REFERENCE_METRICS[:15]])
def test_tiled_merge_preserves_factor_order_counts_and_requested_domain(monkeypatch, metrics):
    # Defect: merging tiles misorders factors, transposes series, or leaks extra metrics.
    import quant_evaluator.scripts.source_all24_oracle as module
    from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
    monkeypatch.setattr(module, "_reference_tile_parts", _literal_parts)
    source = Source()
    out = reference_source_all24(source, labels_for(source), metrics=metrics, max_tile_size=3)
    assert isinstance(out, BatchEvaluationBundle)
    assert source.reads == [(0, 1), (1, 2), (2, 3)] and not source.closed
    assert out.factor_ids == ("f0", "f1", "f2")
    assert set(out.scalar_metrics) | set(out.series_metrics) == set(metrics)
    for metric in metrics:
        np.testing.assert_array_equal(out.observation_counts[metric], [1, 2, 3])
        if metric in out.series_metrics:
            np.testing.assert_array_equal(out.series_metrics[metric],
                                          [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
        else:
            np.testing.assert_array_equal(out.scalar_metrics[metric], [.5, 1.5, 2.5])


def test_previous_batch_is_released_before_next_source_read(monkeypatch):
    # Defect: a loop reference retains the prior immutable tile during the next read.
    import weakref
    import quant_evaluator.scripts.source_all24_oracle as module
    monkeypatch.setattr(module, "_reference_tile_parts", _literal_parts)
    class EphemeralSource(Source):
        prior = None
        def read_tile(self, start, end):
            assert self.prior is None or self.prior() is None
            tile = super().read_tile(start, end)
            self.prior = weakref.ref(tile.batch)
            return tile
    source = EphemeralSource()
    out = reference_source_all24(source, labels_for(source), max_tile_size=1)
    assert out is not None and source.prior() is None


@pytest.mark.parametrize("defect", ["duplicate", "missing", "series_shape", "negative_counts"])
def test_malformed_independent_part_is_rejected_without_reading_next_tile(monkeypatch, defect):
    # Defect: a malformed component can silently contaminate the combined oracle.
    import quant_evaluator.scripts.source_all24_oracle as module
    def parts(batch, labels):
        part = _literal_parts(batch, labels)[0]
        if defect == "duplicate":
            return (part, part)
        if defect == "missing":
            del part.scalar_metrics["rank_ic"]
            del part.observation_counts["rank_ic"]
        if defect == "series_shape":
            part.series_metrics["rank_ic_series"] = np.ones((1, 4))
        if defect == "negative_counts":
            part.observation_counts["rank_ic"][:] = -1
        return (part,)
    monkeypatch.setattr(module, "_reference_tile_parts", parts)
    source = Source()
    with pytest.raises(ValueError):
        reference_source_all24(source, labels_for(source))
    assert source.reads == [(0, 1)]
