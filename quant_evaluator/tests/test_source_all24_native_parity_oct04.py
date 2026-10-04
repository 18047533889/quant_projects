"""Native CPU/CUDA source parity against the independent all-24 oracle."""
import numpy as np
import pytest

from quant_evaluator.api.factor_source import evaluate_factor_source_batch
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.scripts.source_all24_oracle import (
    REFERENCE_METRICS,
    reference_source_all24,
)


class _ArraySource:
    def __init__(self, batch):
        self.batch = batch
        self.factor_ids = batch.factor_ids
        self.time_axis = batch.time_axis
        self.asset_axis = batch.asset_axis
        self.dtype = batch.dtype
        self.snapshot_id = "oct04-native-parity-fixture"
        self.max_tile_size = 2

    def read_tile(self, start, end):
        tile_batch = FactorBatch(
            self.factor_ids[start:end], self.time_axis, self.asset_axis,
            self.batch.values[:, :, start:end],
            validity=self.batch.validity[:, :, start:end],
        )
        return FactorTile(start, end, tile_batch, self.snapshot_id)

    def close(self):
        pass


def _fixture(dtype):
    rng = np.random.default_rng(20261004)
    T, N, F = 65, 100, 5
    times = np.arange(T, dtype=np.int64)
    assets = np.arange(N, dtype=np.int64)
    time_axis = AxisRef("time", "int64", T, times)
    asset_axis = AxisRef("asset", "int64", N, assets)
    values = rng.normal(size=(T, N, F))
    values[:, :, 0] = np.tile(np.arange(N) % 8, (T, 1))
    values[:, :, 1] = 3.0
    values[:, :, 4] = np.nan
    validity = rng.random((T, N, F)) > 0.04
    validity[:, :, 4] = False
    values[~validity] = np.nan
    # Keep finite poison values behind false factor masks.
    validity[:, ::5, 3] = False
    values[:, ::5, 3] = 1e30 if dtype is np.float32 else 1e100
    # Sparse factor cross-section below the IC (20) and quantile-bucket (10)
    # minima on one date.
    validity[7, :, 2] = False
    values[7, :, 2] = np.nan
    validity[7, :8, 2] = True
    values[7, :8, 2] = rng.normal(size=8)
    batch = FactorBatch(tuple(f"f{i}" for i in range(F)), time_axis, asset_axis,
                        values.astype(dtype), validity=validity)

    labels = rng.normal(size=(T, N))
    label_validity = rng.random((T, N)) > 0.03
    labels[~label_validity] = np.nan
    labels[0, :] = 0.25
    label_validity[0, :] = True
    labels[1, :] = np.nan
    label_validity[1, :] = False
    labels[1, :8] = rng.normal(size=8)
    label_validity[1, :8] = True
    label = LabelBundle(
        "forward_return", labels.astype(dtype), 1,
        decision_time=tuple(times), label_start_time=tuple(times),
        label_end_time=tuple(times + 1), validity=label_validity,
        asset_axis=asset_axis,
    )
    return batch, label


@pytest.mark.parametrize("backend", ["cpu", "cuda_strict"])
@pytest.mark.parametrize("metrics", [REFERENCE_METRICS[:15], REFERENCE_METRICS],
                         ids=["historical15", "full24"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f32", "f64"])
def test_native_source_matches_independent_oracle(backend, metrics, dtype):
    if backend == "cuda_strict":
        pytest.importorskip("cupy")
    batch, labels = _fixture(dtype)
    source = _ArraySource(batch)
    native = evaluate_factor_source_batch(
        source, labels, metrics=metrics, backend=backend, max_tile_size=2,
    )
    reference = reference_source_all24(
        source, labels, metrics=metrics, max_tile_size=1,
        max_result_bytes=2 * 1024**2,
    )

    series_metrics = {"rank_ic_series", "pearson_ic_series"}
    for metric in metrics:
        native_group = native.series_metrics if metric in series_metrics else native.scalar_metrics
        reference_group = reference.series_metrics if metric in series_metrics else reference.scalar_metrics
        actual = native_group[metric]
        expected = reference_group[metric]
        assert np.array_equal(np.isfinite(actual), np.isfinite(expected)), (
            f"{backend}/{metric}: finite-value masks differ; native={actual!r}, reference={expected!r}"
        )
        try:
            np.testing.assert_allclose(
                actual, expected, rtol=0, atol=1e-10, equal_nan=True,
            )
        except AssertionError as exc:
            raise AssertionError(f"{backend}/{metric}: native/reference values mismatch: {exc}") from exc
        np.testing.assert_array_equal(
            native.observation_counts[metric], reference.observation_counts[metric],
            err_msg=f"{backend}/{metric}: observation counts differ",
        )
