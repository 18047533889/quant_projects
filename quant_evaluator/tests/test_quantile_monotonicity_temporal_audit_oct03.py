"""Public CPU audit for overflow in the long-run quantile profile."""
import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate


def test_public_cpu_quantile_monotonicity_keeps_huge_flat_profile_finite():
    times, assets = 25, 50
    time_axis = AxisRef("time", "int64", times, np.arange(times, dtype=np.int64))
    asset_axis = AxisRef(
        "asset", "str", assets, np.array([f"a{i}" for i in range(assets)]),
    )
    values = np.tile(np.arange(assets, dtype=np.float64), (times, 1))[:, :, None]
    batch = FactorBatch(("f0",), time_axis, asset_axis, values)
    labels = LabelBundle(
        "huge-flat-profile", np.full((times, assets), 1e308), 1,
        decision_time=tuple(range(times)),
        label_start_time=tuple(range(1, times + 1)),
        label_end_time=tuple(range(2, times + 2)), asset_axis=asset_axis,
    )

    result = evaluate(
        batch, labels, metrics=("quantile_monotonicity",), backend="cpu",
        quantile_builder_parameters={"n_quantiles": 5, "min_assets": 10},
    )

    metric = result.get_metric("quantile_monotonicity", "f0")
    assert metric.valid
    assert metric.value == 0.0
    assert metric.observation_count == 4


def test_public_cpu_windowed_shape_stability_keeps_huge_profiles_finite():
    """Windowed profile summation must not overflow before taking its mean."""
    times, assets = 40, 50
    time_axis = AxisRef("time", "int64", times, np.arange(times, dtype=np.int64))
    asset_axis = AxisRef(
        "asset", "str", assets, np.array([f"a{i}" for i in range(assets)]),
    )
    values = np.tile(np.arange(assets, dtype=np.float64), (times, 1))[:, :, None]
    batch = FactorBatch(("f0",), time_axis, asset_axis, values)
    bucket_offsets = np.repeat(np.arange(5, dtype=np.float64), 10) * 1e293
    labels = LabelBundle(
        "huge-monotone-window-profiles",
        np.tile(1e308 + bucket_offsets[None, :], (times, 1)), 1,
        decision_time=tuple(range(times)),
        label_start_time=tuple(range(1, times + 1)),
        label_end_time=tuple(range(2, times + 2)), asset_axis=asset_axis,
    )

    result = evaluate(
        batch, labels, metrics=("shape_stability",), backend="cpu",
        quantile_builder_parameters={
            "n_quantiles": 5, "min_assets": 10, "window_size": 20,
        },
    )

    metric = result.get_metric("shape_stability", "f0")
    assert metric.valid
    assert metric.value > 0.99


@pytest.mark.parametrize("name", ["compute_shape_stability", "compute_shape_regime_stability"])
def test_identical_huge_three_window_profiles_have_unit_correlation(name):
    from quant_evaluator.metrics import shape_evidence

    # Identical nonconstant finite vectors have Pearson correlation exactly 1,
    # independently of their affine offset. Three windows also exercise the
    # leave-one-out temporal mean, where summing two 1e308 values overflows.
    profile = 1e308 + np.arange(5, dtype=np.float64) * 1e293
    windows = np.broadcast_to(profile[None, :, None], (3, 5, 1)).copy()
    got = getattr(shape_evidence, name)(windows)
    # Both APIs retain the documented Fisher-z clipping at 1 - 1e-9.
    assert got[0] == pytest.approx(1.0 - 1e-9, abs=2e-15, rel=0.0)
