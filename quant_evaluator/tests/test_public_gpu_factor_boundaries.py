"""Public CPU/CUDA parity at factor-axis batch boundaries."""

import numpy as np
import pytest

pytest.importorskip("cupy")

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.device_session import DeviceEvaluationSession
from quant_evaluator.runtime.evaluator import evaluate

from quant_evaluator.runtime.gpu_working_set import estimate_coverage_working_set_bytes

@pytest.mark.parametrize("num_factors", [33, 65])
def test_public_cuda_ic_and_daily_quantile_preserve_factor_boundary(num_factors, monkeypatch):
    rng = np.random.default_rng(33065 + num_factors)
    num_times, num_assets = 6, 48
    time_index = tuple(f"2026-01-{day:02d}" for day in range(1, num_times + 1))
    asset_ids = tuple(f"asset-{i:03d}" for i in range(num_assets))
    factor_ids = tuple(f"factor-{i:03d}" for i in range(num_factors))

    values = rng.normal(size=(num_times, num_assets, num_factors))
    labels = rng.normal(size=(num_times, num_assets))
    # Ties, factor-specific missingness, and label missingness exercise the
    # per-factor pair masks while the factor axis crosses 32- and 64-wide cuts.
    values[:, ::5, ::3] = np.round(values[:, ::5, ::3], 1)
    values[1::2, ::11, 1::4] = np.nan
    labels[::2, ::13] = np.nan
    # Leave fewer observations than min_assets for selected factors/dates.
    values[2, 12:, 0] = np.nan
    values[4, 12:, -1] = np.nan

    batch = FactorBatch(
        factor_ids=factor_ids,
        time_axis=AxisRef("time", "str", num_times, np.asarray(time_index)),
        asset_axis=AxisRef("asset", "str", num_assets, np.asarray(asset_ids)),
        values=values,
    )
    label_bundle = LabelBundle(
        target_id="forward_return",
        values=labels,
        horizon=1,
        decision_time=time_index,
        label_start_time=tuple(f"2026-02-{day:02d}" for day in range(1, num_times + 1)),
        label_end_time=tuple(f"2026-03-{day:02d}" for day in range(1, num_times + 1)),
    )
    metrics = ("rank_ic_series", "quantile_returns_daily")
    parameters = {
        "rank_ic_series": {"min_assets": 12},
        "quantile_returns_daily": {"n_quantiles": 4, "min_assets": 12},
    }

    uploaded = []
    original_stage_factors = DeviceEvaluationSession.stage_factors

    def track_stage_factors(session, tile_values, tile_factor_ids, layout="T,F,N"):
        uploaded.append(len(tile_factor_ids))
        return original_stage_factors(session, tile_values, tile_factor_ids, layout)

    monkeypatch.setattr(DeviceEvaluationSession, "estimate_tile", lambda *args, **kwargs: 32)
    monkeypatch.setattr(DeviceEvaluationSession, "stage_factors", track_stage_factors)

    cpu = evaluate(batch, label_bundle, metrics=metrics, metric_parameters=parameters)
    cuda = evaluate(
        batch, label_bundle, metrics=metrics, metric_parameters=parameters,
        backend="cuda_strict",
    )

    expected_tiles = [32, 1] if num_factors == 33 else [32, 32, 1]
    assert uploaded == expected_tiles

    cpu_ic, cuda_ic = cpu.artifacts["rank_ic_series"], cuda.artifacts["rank_ic_series"]
    assert cpu_ic.values.shape == cuda_ic.values.shape == (num_times, num_factors)
    np.testing.assert_allclose(cuda_ic.values, cpu_ic.values, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert cuda_ic.time_index == cpu_ic.time_index == time_index
    assert cuda_ic.time_axis == cpu_ic.time_axis
    assert cuda_ic.factor_axis == cpu_ic.factor_axis
    assert cuda_ic.factor_axis.factor_ids == factor_ids
    assert np.isnan(cuda_ic.values[2, 0])
    assert np.isnan(cuda_ic.values[4, -1])

    cpu_quantiles = cpu.artifacts["quantile_returns_daily"]
    cuda_quantiles = cuda.artifacts["quantile_returns_daily"]
    assert cpu_quantiles.values.shape == cuda_quantiles.values.shape == (num_times, 4, num_factors)
    np.testing.assert_allclose(
        cuda_quantiles.values, cpu_quantiles.values,
        rtol=1e-12, atol=1e-12, equal_nan=True,
    )
    np.testing.assert_array_equal(cuda_quantiles.counts, cpu_quantiles.counts)
    np.testing.assert_array_equal(cuda_quantiles.valid_mask, cpu_quantiles.valid_mask)
    assert cuda_quantiles.time_axis == cpu_quantiles.time_axis == time_index
    assert cuda_quantiles.quantile_axis == cpu_quantiles.quantile_axis == (0, 1, 2, 3)
    assert cuda_quantiles.factor_axis == cpu_quantiles.factor_axis == factor_ids
    assert np.any(cuda_quantiles.valid_mask)
    assert not cuda_quantiles.valid_mask[2, :, 0].any()
    assert not cuda_quantiles.valid_mask[4, :, -1].any()
    assert np.any(cuda_quantiles.counts > 0)


def test_public_cuda_coverage_budget_parity_with_validity_and_singleton_tail(monkeypatch):
    rng = np.random.default_rng(64032)
    num_times, num_assets, num_factors = 16, 32, 5
    time_index = tuple(f"2026-04-{day:02d}" for day in range(1, num_times + 1))
    asset_ids = tuple(f"asset-{i:03d}" for i in range(num_assets))
    factor_ids = tuple(f"factor-{i:03d}" for i in range(num_factors))

    values = rng.normal(size=(num_times, num_assets, num_factors))
    labels = rng.normal(size=(num_times, num_assets))
    factor_validity = rng.random(values.shape) > 0.12
    label_validity = rng.random(labels.shape) > 0.15
    values[0, 0, 0] = np.nan
    values[1, 1, 1] = np.inf
    values[2, 2, 2] = -np.inf
    labels[3, 3] = np.nan
    labels[4, 4] = np.inf
    labels[5, 5] = -np.inf

    batch = FactorBatch(
        factor_ids=factor_ids,
        time_axis=AxisRef("time", "str", num_times, np.asarray(time_index)),
        asset_axis=AxisRef("asset", "str", num_assets, np.asarray(asset_ids)),
        values=values,
        validity=factor_validity,
    )
    label_bundle = LabelBundle(
        target_id="forward_return",
        values=labels,
        horizon=1,
        decision_time=time_index,
        label_start_time=tuple(f"2026-05-{day:02d}" for day in range(1, num_times + 1)),
        label_end_time=tuple(f"2026-06-{day:02d}" for day in range(1, num_times + 1)),
        validity=label_validity,
    )

    estimate_two = estimate_coverage_working_set_bytes(
        num_times, num_assets, 2, values.dtype.itemsize, labels.dtype.itemsize,
    )
    estimate_four = estimate_coverage_working_set_bytes(
        num_times, num_assets, 4, values.dtype.itemsize, labels.dtype.itemsize,
    )
    budget = estimate_two + 2 * num_times * num_assets * 2
    assert budget < estimate_four + 2 * num_times * num_assets * 4
    sizing_session = DeviceEvaluationSession()
    sizing_session._vram_budget = budget
    assert sizing_session.estimate_tile(
        ("coverage",), num_times, num_assets, values.dtype.itemsize,
        validity_mask=True,
    ) == 2
    uploaded = []
    original_open = DeviceEvaluationSession._open
    original_stage_factors = DeviceEvaluationSession.stage_factors
    original_stage_masked_factors = DeviceEvaluationSession.stage_masked_factors

    def open_with_small_budget(session):
        original_open(session)
        session._vram_budget = budget
        session._pool.set_limit(size=budget)

    def track_stage_factors(session, tile_values, tile_factor_ids, layout="T,F,N"):
        uploaded.append(tuple(tile_factor_ids))
        return original_stage_factors(session, tile_values, tile_factor_ids, layout)

    def track_stage_masked_factors(
        session, tile_values, validity, tile_factor_ids, layout="T,F,N"
    ):
        uploaded.append(tuple(tile_factor_ids))
        return original_stage_masked_factors(
            session, tile_values, validity, tile_factor_ids, layout)

    monkeypatch.setattr(DeviceEvaluationSession, "_open", open_with_small_budget)
    monkeypatch.setattr(DeviceEvaluationSession, "stage_factors", track_stage_factors)
    monkeypatch.setattr(
        DeviceEvaluationSession, "stage_masked_factors", track_stage_masked_factors)

    cpu = evaluate(batch, label_bundle, metrics=("coverage",))
    cuda = evaluate(batch, label_bundle, metrics=("coverage",), backend="cuda_strict")

    assert [len(tile) for tile in uploaded] == [2, 2, 1]
    assert uploaded == [factor_ids[:2], factor_ids[2:4], factor_ids[4:]]
    cpu_coverage = cpu.artifacts["coverage"]
    cuda_coverage = cuda.artifacts["coverage"]
    valid_pairs = (factor_validity & label_validity[:, :, None]
                   & np.isfinite(values) & np.isfinite(labels)[:, :, None])
    expected_counts = valid_pairs.sum(axis=(0, 1))
    expected_coverage = expected_counts / (num_times * num_assets)
    np.testing.assert_allclose(cuda_coverage.values, cpu_coverage.values, rtol=0, atol=0)
    np.testing.assert_allclose(cuda_coverage.values, expected_coverage, rtol=0, atol=0)
    assert cuda_coverage.factor_axis == cpu_coverage.factor_axis
    assert cuda_coverage.factor_axis.factor_ids == factor_ids
    np.testing.assert_array_equal(
        cuda_coverage.provenance["observation_counts"],
        cpu_coverage.provenance["observation_counts"],
    )
    np.testing.assert_array_equal(cuda_coverage.provenance["observation_counts"], expected_counts)
    assert cuda_coverage.provenance["sample_unit"] == cpu_coverage.provenance["sample_unit"]
    for factor_id in factor_ids:
        actual = cuda.get_metric("coverage", factor_id)
        expected = cpu.get_metric("coverage", factor_id)
        assert actual.valid == expected.valid
