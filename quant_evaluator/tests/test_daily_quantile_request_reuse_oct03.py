"""CPU request-local reuse contracts for daily quantile panels."""
from __future__ import annotations

import numpy as np

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle


def _inputs(seed=82103, *, t=32, n=128, f=2):
    rng = np.random.default_rng(seed)
    time_axis = AxisRef("time", "int", t, np.arange(t))
    asset_axis = AxisRef("asset", "str", n, np.array([f"a{i}" for i in range(n)]))
    values = rng.integers(-4, 5, size=(t, n, f)).astype(np.float64)
    values[rng.random(values.shape) < .04] = np.nan
    labels = .01 * np.nan_to_num(values[:, :, 0]) + rng.normal(0, .02, size=(t, n))
    labels[rng.random(labels.shape) < .06] = np.nan
    valid = np.ones(values.shape, dtype=bool)
    valid[3::9, ::7, :] = False
    label_valid = np.ones(labels.shape, dtype=bool)
    label_valid[5::8, 1::9] = False
    batch = FactorBatch(tuple(f"f{i}" for i in range(f)), time_axis, asset_axis,
                        values, validity=valid)
    bundle = LabelBundle(
        "daily-quantile-reuse", labels, 1, validity=label_valid,
        decision_time=tuple(range(t)),
        label_start_time=tuple(range(1, t + 1)),
        label_end_time=tuple(range(2, t + 2)), asset_axis=asset_axis,
    )
    return batch, bundle


def _count_quantile_panel_builds(monkeypatch):
    import quant_evaluator.metrics.quantile as quantile
    import quant_evaluator.metrics.registry_adapters as adapters

    original = quantile.compute_quantile_returns_fast
    calls = []

    def counted(*args, **kwargs):
        kwargs["use_numba"] = False
        daily, counts = original(*args, **kwargs)
        calls.append((kwargs.get("n_quantiles", 5), kwargs.get("min_assets", 10),
                      daily.copy(), counts.copy()))
        return daily, counts

    monkeypatch.setattr(quantile, "compute_quantile_returns_fast", counted)
    monkeypatch.setattr(adapters, "compute_quantile_returns_fast", counted)
    return calls


def test_public_cpu_request_reuses_one_typed_daily_panel_across_quantile_metrics(monkeypatch):
    from quant_evaluator.runtime.evaluator import evaluate

    batch, labels = _inputs()
    calls = _count_quantile_panel_builds(monkeypatch)
    metrics = (
        "quantile_returns_daily", "quantile_returns_full", "quantile_spread",
        "quantile_monotonicity", "daily_quantile_monotonicity_series",
        "daily_quantile_monotonicity_rate",
    )
    result = evaluate(
        batch, labels, metrics=metrics, backend="cpu",
        quantile_builder_parameters={"n_quantiles": 5, "min_assets": 10},
        metric_parameters={
            "quantile_returns_full": {"min_periods": 2},
            "quantile_spread": {"min_periods": 3},
            "daily_quantile_monotonicity_rate": {"min_periods": 2},
        },
    )

    assert len(calls) == 1
    q, min_assets, expected_daily, expected_counts = calls[0]
    assert (q, min_assets) == (5, 10)
    daily = result.artifacts["quantile_returns_daily"]
    assert "input_binding" not in daily.provenance
    for name in ("quantile_returns_full", "quantile_spread"):
        assert "input_binding" not in result.artifacts[name].provenance
    np.testing.assert_array_equal(daily.values, expected_daily)
    np.testing.assert_array_equal(daily.counts, expected_counts)
    np.testing.assert_array_equal(
        daily.valid_mask,
        np.isfinite(expected_daily) & (expected_counts >= min_assets),
    )

    full = result.artifacts["quantile_returns_full"].values
    finite_counts = np.isfinite(expected_daily).sum(axis=0)
    full_expected = np.nansum(expected_daily, axis=0) / np.maximum(finite_counts, 1)
    full_expected[finite_counts < 2] = np.nan
    np.testing.assert_allclose(full, full_expected, rtol=0, atol=0, equal_nan=True)
    assert np.isfinite(full).any()

    spread = expected_daily[:, -1, :] - expected_daily[:, 0, :]
    joint_days = np.isfinite(spread).sum(axis=0)
    spread_expected = np.nansum(spread, axis=0) / np.maximum(joint_days, 1)
    spread_expected[joint_days < 3] = np.nan
    assert joint_days.max() > 0
    spread_artifact = result.artifacts["quantile_spread"]
    np.testing.assert_allclose(spread_artifact.values, spread_expected,
                               rtol=0, atol=0, equal_nan=True)
    assert tuple(spread_artifact.provenance["observation_counts"]) == tuple(joint_days)


def test_public_cpu_request_separates_daily_panel_cache_keys_by_q_and_min_assets(monkeypatch):
    from quant_evaluator.runtime.evaluator import evaluate

    batch, labels = _inputs(seed=82104)
    calls = _count_quantile_panel_builds(monkeypatch)
    # The profile metric uses Q=4/min_assets=3. Daily/full/spread retain their
    # public adapter defaults Q=5/min_assets=10 and should share that panel.
    evaluate(
        batch, labels, metrics=("quantile_returns_daily", "quantile_returns_full",
                                "quantile_spread", "quantile_monotonicity"),
        backend="cpu", quantile_builder_parameters={"n_quantiles": 4, "min_assets": 3},
        metric_parameters={"quantile_spread": {"n_quantiles": 5, "min_periods": 2}},
    )
    assert sorted((q, n) for q, n, _, _ in calls) == [(4, 3), (5, 10)]


def test_daily_metric_keeps_adapter_defaults_without_a_profile_metric(monkeypatch):
    from quant_evaluator.runtime.evaluator import evaluate

    batch, labels = _inputs(seed=82107)
    calls = _count_quantile_panel_builds(monkeypatch)
    result = evaluate(
        batch, labels, metrics=("quantile_returns_daily",), backend="cpu",
        quantile_builder_parameters={"n_quantiles": 4, "min_assets": 3},
    )

    assert [(q, n) for q, n, _, _ in calls] == [(5, 10)]
    daily = result.artifacts["quantile_returns_daily"]
    assert daily.values.shape[1] == 5
    np.testing.assert_array_equal(
        daily.valid_mask, np.isfinite(daily.values) & (daily.counts >= 10),
    )


def test_metric_order_does_not_change_panel_keys_or_results(monkeypatch):
    from quant_evaluator.runtime.evaluator import evaluate

    batch, labels = _inputs(seed=82108)
    calls = _count_quantile_panel_builds(monkeypatch)
    orders = (
        ("quantile_returns_daily", "quantile_spread", "quantile_monotonicity"),
        ("quantile_monotonicity", "quantile_spread", "quantile_returns_daily"),
    )
    snapshots = []
    for order in orders:
        before = len(calls)
        result = evaluate(
            batch, labels, metrics=order, backend="cpu",
            quantile_builder_parameters={"n_quantiles": 4, "min_assets": 3},
            metric_parameters={"quantile_spread": {"n_quantiles": 5, "min_periods": 2}},
        )
        request_calls = calls[before:]
        assert sorted((q, n) for q, n, _, _ in request_calls) == [(4, 3), (5, 10)]
        daily = result.artifacts["quantile_returns_daily"]
        assert daily.values.shape[1] == 5
        snapshots.append((
            daily.values.copy(), daily.counts.copy(),
            result.artifacts["quantile_spread"].values.copy(),
        ))

    assert len(calls) == 4
    np.testing.assert_array_equal(snapshots[0][0], snapshots[1][0])
    np.testing.assert_array_equal(snapshots[0][1], snapshots[1][1])
    np.testing.assert_allclose(snapshots[0][2], snapshots[1][2],
                               rtol=0, atol=0, equal_nan=True)


def test_quantile_panel_cache_is_request_local(monkeypatch):
    from quant_evaluator.runtime.evaluator import evaluate

    batch, labels = _inputs(seed=82105)
    calls = _count_quantile_panel_builds(monkeypatch)
    kwargs = dict(
        metrics=("quantile_returns_daily", "quantile_returns_full", "quantile_spread"),
        backend="cpu",
    )
    evaluate(batch, labels, **kwargs)
    evaluate(batch, labels, **kwargs)
    assert len(calls) == 2
