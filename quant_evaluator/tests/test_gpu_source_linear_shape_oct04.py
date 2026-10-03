"""Public tiny-CUDA source parity for the six linear quantile-shape metrics."""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from quant_evaluator.api.factor_source import evaluate_factor_source_batch
from quant_evaluator.contracts.errors import UnsupportedMetricError
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate

SHAPE_METRICS = (
    "quantile_curvature", "quantile_tail_asymmetry", "quantile_adjacent_spread",
    "quantile_extreme_cliff", "top_quantile_cliff", "bottom_quantile_cliff",
)


class _TinySource:
    def __init__(self, values, validity, time_axis, asset_axis, factor_ids):
        self.values = values
        self.validity = validity
        self.time_axis = time_axis
        self.asset_axis = asset_axis
        self.factor_ids = tuple(factor_ids)
        self.dtype = "float64"
        self.snapshot_id = "tiny-public-cuda-shape-source-oct04"
        self.max_tile_size = 2
        self.reads = []
        self.closed = False

    def read_tile(self, start, end):
        self.reads.append((start, end))
        batch = FactorBatch(
            self.factor_ids[start:end], self.time_axis, self.asset_axis,
            self.values[:, :, start:end],
            validity=self.validity[:, :, start:end],
        )
        return FactorTile(start, end, batch, self.snapshot_id)

    def close(self):
        self.closed = True


def _inputs():
    rng = np.random.default_rng(20261004)
    t_count, n_assets, n_factors = 32, 72, 3
    times = np.arange(t_count, dtype=np.int64)
    time_axis = AxisRef("time", "int64", t_count, times)
    asset_axis = AxisRef(
        "asset", "str", n_assets,
        np.asarray([f"A{i:03d}" for i in range(n_assets)]),
    )
    values = rng.normal(size=(t_count, n_assets, n_factors))
    values[2, 0, 0] = np.nan
    values[7, 4, 1] = np.nan
    values[11, 25, 2] = np.nan
    validity = rng.random(values.shape) > 0.04
    batch = FactorBatch(tuple(f"f{i}" for i in range(n_factors)),
                        time_axis, asset_axis, values, validity=validity)
    label_values = rng.normal(size=(t_count, n_assets))
    label_values[4, 3] = np.nan
    label_validity = rng.random(label_values.shape) > 0.08
    labels = LabelBundle(
        "tiny-source-label", label_values, 1,
        decision_time=tuple(times), observation_time=tuple(times),
        signal_available_time=tuple(times), execution_time=tuple(times),
        label_start_time=tuple(times), label_end_time=tuple(times + 1),
        asset_axis=asset_axis, validity=label_validity,
    )
    return batch, labels, _TinySource(
        values, validity, time_axis, asset_axis, batch.factor_ids)


def _manual_profile(batch, label, n_quantiles=5, min_assets=10, min_periods=20):
    """Independent percentile membership, daily aggregation, and profile oracle."""
    t_count, _, n_factors = batch.values.shape
    daily = np.full((t_count, n_quantiles, n_factors), np.nan)
    for t in range(t_count):
        for f in range(n_factors):
            factor_ok = batch.validity[t, :, f] & np.isfinite(batch.values[t, :, f])
            ordered = np.sort(batch.values[t, factor_ok, f], kind="mergesort")
            if len(ordered) < n_quantiles:
                continue
            boundaries = []
            for q in range(1, n_quantiles):
                position = (q / n_quantiles) * (len(ordered) - 1)
                low = int(position)
                fraction = position - low
                boundaries.append(ordered[low] + fraction *
                                  (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]))
            bins = np.searchsorted(boundaries, batch.values[t, factor_ok, f], side="right")
            label_ok = label.validity[t] & np.isfinite(label.values[t])
            selected = label_ok[factor_ok]
            bins = bins[selected]
            selected_labels = label.values[t, factor_ok][selected]
            for q in range(n_quantiles):
                members = selected_labels[bins == q]
                if len(members) >= min_assets:
                    daily[t, q, f] = np.mean(members)
    profile = np.full((n_quantiles, n_factors), np.nan)
    for q in range(n_quantiles):
        for f in range(n_factors):
            finite = daily[:, q, f][np.isfinite(daily[:, q, f])]
            if len(finite) >= min_periods:
                profile[q, f] = np.mean(finite)
    return profile


def _manual_shape_values(profile):
    q_count, n_factors = profile.shape
    result = {name: np.full(n_factors, np.nan) for name in SHAPE_METRICS}
    for f in range(n_factors):
        column = profile[:, f]
        finite = np.isfinite(column)
        if q_count >= 3:
            interior = finite[:-2] & finite[1:-1] & finite[2:]
            if interior.any():
                result["quantile_curvature"][f] = np.mean(
                    column[2:][interior] - 2 * column[1:-1][interior] + column[:-2][interior])
            mid = q_count // 2
            if finite[0] and finite[mid] and finite[-1]:
                result["quantile_tail_asymmetry"][f] = column[-1] - 2 * column[mid] + column[0]
        pairs = finite[:-1] & finite[1:]
        if pairs.any():
            result["quantile_adjacent_spread"][f] = np.mean(
                np.abs(column[1:][pairs] - column[:-1][pairs]))
        if q_count >= 2 and finite[0] and finite[1] and finite[-1] and finite[-2]:
            result["quantile_extreme_cliff"][f] = (
                (column[-1] - column[-2]) + (column[1] - column[0])) / 2
        if q_count >= 2 and finite[-1] and finite[-2]:
            result["top_quantile_cliff"][f] = column[-1] - column[-2]
        if q_count >= 2 and finite[0] and finite[1]:
            result["bottom_quantile_cliff"][f] = column[1] - column[0]
    return result


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"CUDA runtime unavailable: {exc}")


def test_public_cuda_source_shape_matches_manual_and_full_cpu():
    batch, labels, source = _inputs()
    cpu = evaluate(batch, labels, metrics=SHAPE_METRICS, backend="cpu")
    try:
        result = evaluate_factor_source_batch(
            source, labels, metrics=SHAPE_METRICS, backend="cuda_strict",
            max_tile_size=2,
        )
    finally:
        source.close()

    assert source.closed
    assert source.reads == [(0, 2), (2, 3)]
    assert result.factor_ids == batch.factor_ids
    assert set(result.scalar_metrics) == set(SHAPE_METRICS)
    assert result.series_metrics == {}
    assert result.metadata["backend_used"] == "cuda"
    receipt = result.metadata["execution_receipt"]
    assert receipt["backend_requested"] == "cuda_strict"
    assert receipt["backend_used"] == "cuda"
    assert receipt["metric_backends"] == {name: "cuda" for name in SHAPE_METRICS}
    assert result.metadata["factor_tiles_processed"] == 2
    assert result.metadata["shape_kernel_backend"] == "cuda_strict"
    assert result.metadata["shape_kernel_no_fallback"] is True
    assert result.metadata["shape_kernel_dispatches"] >= len(SHAPE_METRICS)

    manual = _manual_shape_values(_manual_profile(batch, labels))
    for metric in SHAPE_METRICS:
        np.testing.assert_allclose(result.scalar_metrics[metric], manual[metric],
                                   rtol=1e-10, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(result.scalar_metrics[metric],
                                   cpu.artifacts[metric].values,
                                   rtol=1e-10, atol=1e-12, equal_nan=True)
        expected_counts = np.asarray([
            cpu.grouped_metrics[factor_id][metric].observation_count
            for factor_id in batch.factor_ids
        ], dtype=np.int64)
        np.testing.assert_array_equal(result.observation_counts[metric], expected_counts)
        np.testing.assert_array_equal(
            result.observation_counts[metric], np.isfinite(manual[metric]).astype(np.int64))


def test_unknown_source_metric_rejected_before_source_read():
    _, labels, source = _inputs()
    try:
        with pytest.raises(UnsupportedMetricError):
            evaluate_factor_source_batch(
                source, labels, metrics=("not_a_metric",), backend="cuda_strict",
                max_tile_size=2,
            )
    finally:
        source.close()
    assert source.reads == []
