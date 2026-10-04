"""CPU-only mock audit for GPU temporal quantile-mean reuse candidates."""
from types import SimpleNamespace

import numpy as np
import pytest


def _mocked_executor(monkeypatch, metric_parameters=None):
    import quant_evaluator.runtime.gpu_executor as gpu_executor
    import quant_evaluator.kernels.gpu.quantile as quantile_kernel

    factors = np.zeros((3, 1, 4), dtype=np.float64)
    labels = np.ones((3, 4), dtype=np.float64)
    workspace_requests = []

    def workspace_budget(*, output_bytes=0):
        workspace_requests.append(output_bytes)
        return 1 << 20

    session = SimpleNamespace(
        _staged_factors={"__all__": factors},
        _staged_labels={"next_ret": labels},
        workspace_budget=workspace_budget,
        workspace_requests=workspace_requests,
        metadata=lambda: {},
    )
    executor = gpu_executor.GPUExecutor(session)
    executor.metric_parameters = metric_parameters or {}
    monkeypatch.setattr(gpu_executor, "_import_cp", lambda: np)
    monkeypatch.setattr(gpu_executor, "_to_cpu", lambda value: np.asarray(value).copy())

    quantile_calls = []

    def quantiles(_factors, staged_labels, *, n_quantiles, min_assets,
                  return_counts, workspace_bytes):
        quantile_calls.append((n_quantiles, min_assets, staged_labels.copy(), _factors.copy()))
        t, f, _ = _factors.shape
        values = np.ones((t, n_quantiles, f), dtype=np.float64)
        # Keep only the final two dates finite so separate min_periods gates
        # can be observed over the same raw cached reduction.
        values[0] = np.nan
        counts = np.full(values.shape, min_assets, dtype=np.int64)
        return values, counts

    monkeypatch.setattr(quantile_kernel, "batched_quantile_returns", quantiles)
    mean_calls = []

    def finite_mean(values, *, workspace_bytes):
        mean_calls.append(values)
        finite = np.isfinite(values)
        counts = np.sum(finite, axis=0)
        sums = np.sum(np.where(finite, values, 0.0), axis=0, dtype=np.float64)
        return sums / np.maximum(counts, 1), counts

    import quant_evaluator.kernels.gpu.finite_mean as finite_mean_module
    monkeypatch.setattr(finite_mean_module, "finite_mean_axis0", finite_mean)
    profiles = []

    import quant_evaluator.kernels.gpu.quantile_shape as shape_kernel
    monkeypatch.setattr(
        shape_kernel, "quantile_monotonicity",
        lambda profile, return_device=True: profiles.append(profile.copy()) or np.zeros(profile.shape[-1]),
    )
    return executor, session, quantile_calls, mean_calls, profiles


def test_full_and_monotonicity_reuse_raw_mean_but_keep_spread_separate(monkeypatch):
    executor, session, quantile_calls, mean_calls, _profiles = _mocked_executor(monkeypatch)
    executor.run(
        ("f0",),
        ("quantile_returns_full", "quantile_spread", "quantile_monotonicity"),
    )

    assert [(q, assets) for q, assets, _labels, _factors in quantile_calls] == [(5, 10)]
    assert len(mean_calls) == 2
    # Full and monotonicity share one raw daily-quantile reduction.
    assert mean_calls[0].ndim == 3
    # Spread reduces its distinct T-by-F series independently.
    assert mean_calls[1].ndim == 2
    assert session.workspace_requests.count(240) == 1
    assert session.workspace_requests.count(80) == 1
    assert session.workspace_requests.count(16) == 1


@pytest.mark.parametrize(
    "metrics",
    [("quantile_returns_full", "quantile_monotonicity"),
     ("quantile_monotonicity", "quantile_returns_full")],
)
def test_min_period_gates_are_independent_on_shared_raw_mean(monkeypatch, metrics):
    executor, _session, quantile_calls, mean_calls, profiles = _mocked_executor(
        monkeypatch,
        metric_parameters={
            "quantile_returns_full": {"min_periods": 2},
        },
    )
    result = executor.run(("f0",), metrics)

    assert [(q, assets) for q, assets, _labels, _factors in quantile_calls] == [(5, 10)]
    assert len(mean_calls) == 1
    assert mean_calls[0].ndim == 3
    assert np.isfinite(result.vector_metrics["quantile_returns_full"]).all()
    # The raw count is two; full admits it while monotonicity applies its
    # stricter threshold to the same un-gated cached result.
    assert np.isnan(profiles[0]).all()
    assert len(mean_calls) == 1


def test_quantile_key_isolates_n_quantiles_and_min_assets(monkeypatch):
    from types import SimpleNamespace

    executor, session, quantile_calls, mean_calls, _profiles = _mocked_executor(monkeypatch)
    import quant_evaluator.registry.metrics as metrics_registry

    def accepted_quantile_parameters(*, min_periods=20, n_quantiles=5, min_assets=10):
        return None

    original_get_metric = metrics_registry.get_metric

    def fake_metric(name):
        original = original_get_metric(name)
        return SimpleNamespace(
            compute_fn=accepted_quantile_parameters,
            requires=(),
            min_periods=original.min_periods,
        )

    monkeypatch.setattr(metrics_registry, "get_metric", fake_metric)
    executor.metric_parameters = {
        "quantile_returns_full": {"n_quantiles": 3, "min_assets": 7},
    }
    executor.quantile_builder_parameters = {"n_quantiles": 4, "min_assets": 8}
    executor.run(("f0",), ("quantile_returns_full", "quantile_monotonicity"))

    assert [(q, assets, ) for q, assets, _labels, _factors in quantile_calls] == [
        (3, 7), (4, 8),
    ]
    assert len(mean_calls) == 2
    assert session.workspace_requests.count(144) == 1
    assert session.workspace_requests.count(48) == 1
    assert session.workspace_requests.count(192) == 1
    assert session.workspace_requests.count(64) == 1


def test_quantile_cache_scope_does_not_cross_label_mask_runs(monkeypatch):
    executor, session, quantile_calls, mean_calls, _profiles = _mocked_executor(monkeypatch)
    metrics = ("quantile_returns_full", "quantile_monotonicity")
    executor.run(("f0",), metrics)
    session._staged_labels["next_ret"] = np.array(
        [[np.nan, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    executor.run(("f0",), metrics)

    assert len(quantile_calls) == 2
    assert len(mean_calls) == 2
    assert session.workspace_requests.count(80) == 2
    assert np.isfinite(quantile_calls[0][2]).all()
    assert not np.isfinite(quantile_calls[1][2][0, 0])


def test_quantile_cache_scope_does_not_cross_factor_mask_runs(monkeypatch):
    executor, session, quantile_calls, mean_calls, _profiles = _mocked_executor(monkeypatch)
    metrics = ("quantile_returns_full", "quantile_monotonicity")
    executor.run(("f0",), metrics)
    masked_factors = session._staged_factors["__all__"].copy()
    masked_factors[0, 0, 0] = np.nan
    session._staged_factors["__all__"] = masked_factors
    executor.run(("f0",), metrics)

    assert len(quantile_calls) == 2
    assert len(mean_calls) == 2
    assert session.workspace_requests.count(80) == 2
    assert np.isfinite(quantile_calls[0][3]).all()
    assert not np.isfinite(quantile_calls[1][3][0, 0, 0])
