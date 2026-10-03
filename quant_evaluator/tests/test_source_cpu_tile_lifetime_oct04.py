"""Streaming CPU evaluation must release prior tiles before the next read."""
import weakref

import pytest
from quant_evaluator.api.factor_source import evaluate_factor_source_batch
from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.tests.test_public_factor_source_batch import (
    Source, _inputs, _assert_against_full_cpu, evaluate,
)


class FreshAllocatingSource(Source):
    def __init__(self, batch):
        super().__init__(batch)
        self.previous_refs = []
        self.live_before_reads = []

    def read_tile(self, start, end):
        alive = [name for name, ref in self.previous_refs if ref() is not None]
        self.live_before_reads.append(alive)
        assert not alive, f"previous tile retained before next allocation: {alive}"
        batch = FactorBatch(
            self.factor_ids[start:end], self.time_axis, self.asset_axis,
            self.batch.values[:, :, start:end].copy(),
            validity=self.batch.validity[:, :, start:end].copy(),
        )
        tile = FactorTile(start, end, batch, self.snapshot_id)
        self.previous_refs = [(name, weakref.ref(value)) for name, value in (
            ("tile", tile), ("batch", batch), ("values", batch.values),
            ("validity", batch.validity),
        )]
        self.reads.append((start, end))
        return tile


@pytest.mark.parametrize("metrics", [
    ("rank_ic",), ("rank_ic_series",), ("rank_ic", "rank_ic_series"),
    ("rank_ic", "rank_ic_series", "coverage", "quantile_spread"),
])
def test_cpu_releases_input_tile_before_next_allocation(metrics):
    batch, label = _inputs()
    source = FreshAllocatingSource(batch)
    result = evaluate_factor_source_batch(source, label, metrics=metrics, backend="cpu")
    reference = evaluate(batch, label, metrics=metrics, backend="cpu")
    _assert_against_full_cpu(result, reference, metrics)
    assert source.reads == [(0, 2), (2, 4), (4, 5)]
    assert source.live_before_reads == [[], [], []]
    assert all(ref() is None for _, ref in source.previous_refs)
