"""Small full-request source warmup, separate from measured profiling."""
from __future__ import annotations

import numpy as np

from quant_evaluator.api.factor_source import evaluate_factor_source_batch
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.contracts.label_bundle import LabelBundle


def warm_source_profile(policy, backend, metrics):
    """Warm the exact requested metric plan without reading external data."""
    times = np.arange(65, dtype=np.int64)
    assets = AxisRef("asset", "int64", 100, np.arange(100, dtype=np.int64))
    time_axis = AxisRef("time", "int64", 65, times)
    rng = np.random.default_rng(20261004)
    values = rng.normal(size=(65, 100, 2))
    labels = LabelBundle("profile-warmup", rng.normal(size=(65, 100)), 1,
                         decision_time=tuple(times), label_start_time=tuple(times),
                         label_end_time=tuple(times + 1), asset_axis=assets)

    class WarmSource:
        factor_ids = ("profile-warm-a", "profile-warm-b")
        dtype, snapshot_id, max_tile_size = "float64", "profile-warmup-only", 2

        def __init__(self):
            self.time_axis, self.asset_axis = time_axis, assets

        def read_tile(self, start, end):
            batch = FactorBatch(self.factor_ids[start:end], time_axis, assets,
                                values[:, :, start:end])
            return FactorTile(start, end, batch, self.snapshot_id)

        def close(self):
            pass

    source = WarmSource()
    try:
        return evaluate_factor_source_batch(
            source, labels, metrics=metrics, backend=backend,
            max_tile_size=2, gpu_policy=policy,
        )
    finally:
        source.close()
