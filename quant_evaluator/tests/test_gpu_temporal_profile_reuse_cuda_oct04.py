"""Small real-CUDA regression for temporal quantile profile cache reuse."""

import numpy as np
import pytest


@pytest.mark.parametrize(
    "metrics",
    [
        ("quantile_returns_full", "quantile_monotonicity"),
        ("quantile_monotonicity", "quantile_returns_full"),
    ],
)
def test_real_cuda_profile_reuse_keeps_metric_gates_and_run_scope(monkeypatch, metrics):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device is unavailable")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    from quant_evaluator.metrics.quantile_shape import compute_quantile_monotonicity
    from quant_evaluator.runtime.device_session import DeviceEvaluationSession
    from quant_evaluator.runtime.gpu_executor import GPUExecutor
    import quant_evaluator.kernels.gpu.finite_mean as finite_mean_module

    time_count, asset_count = 24, 50
    factor_values = np.broadcast_to(
        np.arange(asset_count, dtype=np.float64)[None, None, :],
        (time_count, 1, asset_count),
    ).copy()
    bucket_returns = np.array([0.0, 1.0, 2.0, 0.0, 4.0])
    bucket_by_asset = np.arange(asset_count) // 10
    labels = np.broadcast_to(
        bucket_returns[bucket_by_asset][None, :], (time_count, asset_count)
    ).copy()
    # Five dates have only nine finite label returns in bucket 2, below the
    # quantile kernel's min_assets=10 floor. Its raw mean has 19 valid dates.
    labels[:5, 20] = np.nan

    real_finite_mean = finite_mean_module.finite_mean_axis0
    finite_mean_calls = []

    def counted_finite_mean(values, *, workspace_bytes):
        finite_mean_calls.append(tuple(values.shape))
        return real_finite_mean(values, workspace_bytes=workspace_bytes)

    monkeypatch.setattr(finite_mean_module, "finite_mean_axis0", counted_finite_mean)

    with DeviceEvaluationSession() as session:
        session.stage_factors(factor_values, ("f0",))
        session.stage_labels(labels, "next_ret")
        executor = GPUExecutor(session)
        executor.metric_parameters = {"quantile_returns_full": {"min_periods": 2}}

        first = executor.run(("f0",), metrics)
        expected_profile = np.array([0.0, 1.0, 2.0, 0.0, 4.0])
        np.testing.assert_allclose(
            first.vector_metrics["quantile_returns_full"][:, 0],
            expected_profile,
            rtol=0.0,
            atol=1e-12,
        )
        gated_profile = expected_profile.copy()
        gated_profile[2] = np.nan  # 19 valid dates fails monotonicity's default 20.
        expected_monotonicity = compute_quantile_monotonicity(gated_profile)[0]
        np.testing.assert_allclose(
            first.scalar_metrics["quantile_monotonicity"][0],
            expected_monotonicity,
            rtol=0.0,
            atol=1e-12,
        )
        assert expected_monotonicity == 1.0
        assert finite_mean_calls == [(time_count, 5, 1)]

        # Mutating the staged labels between executor runs must miss the
        # run-local quantile/mean caches and produce the new flat profile.
        session.stage_labels(np.zeros_like(labels), "next_ret")
        second = executor.run(("f0",), metrics)
        np.testing.assert_allclose(
            second.vector_metrics["quantile_returns_full"][:, 0],
            np.zeros(5),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            second.scalar_metrics["quantile_monotonicity"],
            np.zeros(1),
            rtol=0.0,
            atol=1e-12,
        )
        assert finite_mean_calls == [
            (time_count, 5, 1),
            (time_count, 5, 1),
        ]
