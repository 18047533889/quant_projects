"""Actual turnover observations require eligible rank-proxy weights."""
import numpy as np
import pytest

from quant_evaluator.api.factor_source import evaluate_factor_source_batch
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.contracts.label_bundle import LabelBundle


class Source:
    factor_ids = ("rank-factor",)
    dtype = "float64"
    snapshot_id = "turnover-count-fixture"
    max_tile_size = 1

    def __init__(self, observed, *, use_validity=False):
        self.time_axis = AxisRef("time", "int64", 4, np.arange(4, dtype=np.int64))
        self.asset_axis = AxisRef("asset", "int64", 12, np.arange(12, dtype=np.int64))
        self.values = np.tile(np.arange(12, dtype=np.float64)[None, :, None], (4, 1, 1))
        self.validity = np.ones(self.values.shape, dtype=bool) if use_validity else None
        for t, count in enumerate(observed):
            if use_validity:
                self.validity[t, count:, :] = False
            else:
                self.values[t, count:, :] = np.nan

    def read_tile(self, start, end):
        batch = FactorBatch(self.factor_ids, self.time_axis, self.asset_axis,
                            self.values, self.validity)
        return FactorTile(start, end, batch, self.snapshot_id)

    def close(self):
        pass


@pytest.mark.parametrize("observed,expected_count", [
    ([2, 2, 2, 2], 0),
    ([8, 8, 8, 8], 0), ([9, 9, 9, 9], 0),
    ([10, 10, 10, 10], 3), ([12, 12, 12, 12], 3),
    ([12, 8, 10, 10], 1),
])
@pytest.mark.parametrize("use_validity", [False, True])
@pytest.mark.parametrize("backend", ["cpu", "cuda_strict"])
def test_source_counts_only_transitions_with_two_eligible_rank_weight_rows(
    observed, expected_count, use_validity, backend
):
    if backend == "cuda_strict":
        pytest.importorskip("cupy")
    # Defect: counting >=2-asset dates when weights require >=10 assets.
    source = Source(observed, use_validity=use_validity)
    times = tuple(source.time_axis.values)
    labels = LabelBundle("forward", np.ones((4, 12)), 1,
                         decision_time=times, label_start_time=times,
                         label_end_time=tuple(np.arange(4)+1), asset_axis=source.asset_axis)
    out = evaluate_factor_source_batch(source, labels, metrics=("turnover",),
                                      backend=backend, max_tile_size=1)
    assert out.observation_counts["turnover"].tolist() == [expected_count]
    value = out.scalar_metrics["turnover"][0]
    assert np.isnan(value) if expected_count == 0 else value == 0.0
