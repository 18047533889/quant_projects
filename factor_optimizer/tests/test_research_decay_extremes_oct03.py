"""Small numerical guards for finite-but-extreme decay inputs."""
import warnings

import numpy as np

from factor_optimizer.research_numeric import guarded_finite_mean


def test_guarded_mean_preserves_ordinary_float64_means_bit_for_bit():
    values = np.array([[1.25, -2.5, 3.0], [4.0, 5.5, -6.25]], dtype=np.float64)
    for axis in (0, 1):
        actual = guarded_finite_mean(values, axis=axis)
        expected = np.mean(values, axis=axis)
        assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
    actual = guarded_finite_mean(values)
    expected = np.mean(values)
    assert actual.view(np.uint64) == expected.view(np.uint64)
    cube = values.reshape(1, 2, 3)
    for axis in ((0, 2), -1):
        actual = guarded_finite_mean(cube, axis=axis, keepdims=True)
        expected = np.mean(cube, axis=axis, keepdims=True)
        assert actual.shape == expected.shape
        assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
    actual = guarded_finite_mean(cube, keepdims=True)
    expected = np.mean(cube, keepdims=True)
    assert actual.shape == expected.shape
    assert actual.view(np.uint64) == expected.view(np.uint64)


def test_guarded_mean_repairs_only_finite_overflow_slices():
    values = np.array([[1e308, 1e308], [1e308, -1e308],
                       [np.inf, 1.0], [np.nan, 2.0]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = guarded_finite_mean(values, axis=1)
    assert np.isfinite(result[0]) and result[0] == 1e308
    assert result[1] == 0.0
    assert np.isposinf(result[2])
    assert np.isnan(result[3])
    assert not caught
    column = guarded_finite_mean(values[:2], axis=0)
    assert np.array_equal(column, np.array([1e308, 0.0]))


def test_guarded_mean_handles_subnormal_cancellation_and_scalar_shape():
    tiny = np.nextafter(0.0, 1.0)
    assert guarded_finite_mean(np.array([tiny, tiny])) == tiny
    assert guarded_finite_mean(np.array([1e308, -1e308])) == 0.0
    kept = guarded_finite_mean(np.array([[1e308, 1e308]]), axis=1, keepdims=True)
    assert kept.shape == (1, 1) and kept[0, 0] == 1e308
    assert guarded_finite_mean(np.array([1e308, 1e308]), axis=None) == 1e308
    kept_all = guarded_finite_mean(np.array([[1e308, 1e308], [1e308, 1e308]]),
                                   keepdims=True)
    assert kept_all.shape == (1, 1) and kept_all[0, 0] == 1e308


def test_decay_extreme_label_diagnostic_contains_only_finite_curves():
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from factor_optimizer.research_batch import BatchOptimizationConfig, automatic_time_split
    from factor_optimizer.research_decay import diagnose_layer_decay

    t, n = 84, 200
    rng = np.random.default_rng(8104)
    time = AxisRef("time", "int", t, np.arange(t))
    assets = AxisRef("asset", "str", n, np.array([f"a{i}" for i in range(n)]))
    values = rng.normal(size=(t, n)).astype(np.float32)
    values[3::19, ::11] = np.nan
    values[9, :] = 0.0
    validity = np.ones((t, n), dtype=bool)
    validity[5::13, ::8] = False
    batch = FactorBatch(("f",), time, assets, values[:, :, None],
                        validity=validity[:, :, None])
    labels_values = np.where(np.indices((t, n))[1] % 2 == 0, 1e308, 9e307)
    labels = LabelBundle("extreme-finite", labels_values, 1,
        decision_time=tuple(range(t)), label_start_time=tuple(range(1, t + 1)),
        label_end_time=tuple(range(2, t + 2)), asset_axis=assets)
    config = BatchOptimizationConfig(warmup_bars=10, minimum_train_days=20,
        minimum_validation_days=10, minimum_test_days=10, minimum_assets=3,
        minimum_coverage=.1)
    split = automatic_time_split(labels, config)
    assert np.isfinite(labels.values).all()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        record = diagnose_layer_decay(batch, labels, split, config, 0,
                                      minimum_assets_per_quantile=1)
    assert record["status"] == "available"
    curves = np.asarray([layer["mean_excess_returns"] for layer in record["layers"]])
    assert curves.shape == (20, len(record["lags"]))
    assert np.isfinite(curves).all()
    assert np.isfinite(record["proposed_half_lives"]).all()
    assert not caught


def test_decay_unrepresentable_centered_returns_are_unavailable(monkeypatch):
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from factor_optimizer.research_batch import BatchOptimizationConfig, automatic_time_split
    import factor_optimizer.research_decay as research_decay

    t, n = 84, 200
    time = AxisRef("time", "int", t, np.arange(t))
    assets = AxisRef("asset", "str", n, np.array([f"a{i}" for i in range(n)]))
    batch = FactorBatch(("f",), time, assets,
        np.ones((t, n, 1), dtype=np.float32))
    labels = LabelBundle("finite-labels", np.ones((t, n), dtype=np.float64), 1,
        decision_time=tuple(range(t)), label_start_time=tuple(range(1, t + 1)),
        label_end_time=tuple(range(2, t + 2)), asset_axis=assets)
    config = BatchOptimizationConfig(warmup_bars=10, minimum_train_days=20,
        minimum_validation_days=10, minimum_test_days=10, minimum_assets=3,
        minimum_coverage=.1)
    split = automatic_time_split(labels, config)
    cube = np.full((len(split.train_indices), 20, len(research_decay._DECAY_ASSIGNMENT_LAGS)),
                   -1e308, dtype=np.float64)
    cube[:, 0, :] = 1e308
    monkeypatch.setattr(research_decay, "_cached_lagged_quantile_panels",
                        lambda *args, **kwargs: cube)

    record = research_decay.diagnose_layer_decay(
        batch, labels, split, config, 0, minimum_assets_per_quantile=1)

    assert record["status"] == "unavailable"
    assert record["reason"] == "centered quantile returns are not representable as finite Float64"
