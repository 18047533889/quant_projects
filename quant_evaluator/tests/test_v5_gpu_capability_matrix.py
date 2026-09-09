from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("cupy")

from quant_evaluator.adapters.execution_trajectory import trajectory_to_probe_artifact
from quant_evaluator.contracts.errors import UnsupportedMetricError
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.portfolio_inputs import HoldingReturnPanel, PortfolioSpec
from quant_evaluator.metrics.exposure_evidence import ExposurePanel
from quant_evaluator.runtime.device_session import DeviceEvaluationSession
from quant_evaluator.runtime.evaluator import evaluate
from quant_evaluator.runtime.gpu_executor import GPUExecutor
from vectorbt_qs.contracts.costs import CostScope
from vectorbt_qs.contracts.trajectories import TrajectoryRefs, build_research_trajectory


PORTFOLIO = ("sharpe_ratio", "sortino_ratio", "win_rate", "max_drawdown", "calmar_ratio")
EXPOSURE = (
    "industry_exposure", "size_exposure", "beta_exposure", "liquidity_exposure",
    "volatility_exposure", "momentum_exposure", "max_absolute_style_exposure",
    "exposure_drift", "purity_ratio",
)
BASE = tuple(sorted(GPUExecutor.SUPPORTED_METRICS - set(PORTFOLIO) - set(EXPOSURE)))


def contracts():
    rng = np.random.default_rng(20260908)
    t, n, f = 40, 100, 3
    dates = tuple(str(np.datetime64("2024-01-01") + i) for i in range(t))
    assets = np.asarray([f"A{i:03d}" for i in range(n)])
    x = rng.normal(size=(t, n, f)); y = .15 * x[:, :, 0] + rng.normal(size=(t, n))
    x[3, :8, 1] = np.nan; y[4, :7] = np.nan
    batch = FactorBatch(tuple(f"f{i}" for i in range(f)),
                        AxisRef("time", "date", t, np.asarray(dates)),
                        AxisRef("asset", "str", n, assets), x)
    labels = LabelBundle("h1", y, 1, decision_time=dates,
                         label_start_time=dates,
                         label_end_time=tuple(str(np.datetime64(i) + 1) for i in dates),
                         asset_axis=batch.asset_axis)
    return batch, labels


def assert_public_parity(cpu, gpu, metrics):
    assert gpu.metadata["backend_used"] == "cuda"
    assert gpu.metadata["gpu_device"] == 0
    assert gpu.metadata["peak_vram"] > 0
    for metric in metrics:
        np.testing.assert_allclose(gpu.artifacts[metric].values, cpu.artifacts[metric].values,
                                   rtol=1e-9, atol=1e-10, equal_nan=True)
        provenance = gpu.artifacts[metric].provenance
        if "execution_backend" in provenance:
            assert provenance["execution_backend"] == "cuda_strict"
            assert provenance["no_fallback"] is True
        for factor_id in cpu.factor_ids:
            left, right = cpu.get_metric(metric, factor_id), gpu.get_metric(metric, factor_id)
            assert (left is None) == (right is None)
            if left is not None:
                assert (left.valid, left.observation_count, left.sample_unit) == (
                    right.valid, right.observation_count, right.sample_unit)


def test_all_18_factor_label_metrics_execute_real_cpu_and_strict_cuda():
    batch, labels = contracts()
    cpu = evaluate(batch, labels, metrics=BASE)
    gpu = evaluate(batch, labels, metrics=BASE, backend="cuda_strict")
    assert len(BASE) == 18
    assert_public_parity(cpu, gpu, BASE)
    assert gpu.metadata["peak_vram"] > 0 and gpu.metadata["factor_tiles_processed"] >= 1


@pytest.mark.parametrize("metric", ("ic_ir", "pearson_ic_ir"))
@pytest.mark.parametrize(
    ("min_periods", "error", "message"),
    (
        (True, TypeError, "min_periods must be an integer"),
        (1.5, TypeError, "min_periods must be an integer"),
        (1, ValueError, "min_periods must be at least 2 for sample standard deviation"),
    ),
)
def test_icir_invalid_min_periods_has_same_public_cpu_cuda_contract(
    metric, min_periods, error, message
):
    batch, labels = contracts()
    kwargs = dict(
        metrics=(metric,),
        metric_parameters={metric: {"min_periods": min_periods}},
    )
    with pytest.raises(RuntimeError, match=message):
        evaluate(batch, labels, **kwargs)
    with pytest.raises(error, match=message):
        evaluate(batch, labels, backend="cuda_strict", **kwargs)


def test_all_5_portfolio_metrics_execute_real_cpu_and_strict_cuda():
    batch, labels = contracts()
    rng = np.random.default_rng(93)
    prices = 100 * np.cumprod(1 + rng.normal(0, .01, (40, 100)), axis=0)
    holdings = HoldingReturnPanel.from_prices(
        prices, time_axis=batch.time_axis, asset_axis=batch.asset_axis,
        source_ref="synthetic:gpu-capability-prices", price_basis="close")
    kwargs = dict(metrics=PORTFOLIO, holding_returns=holdings,
                  portfolio_spec=PortfolioSpec(holding=2, n_quantiles=5, per_side_cost=.001))
    cpu = evaluate(batch, labels, **kwargs)
    gpu = evaluate(batch, labels, backend="cuda_strict", **kwargs)
    assert_public_parity(cpu, gpu, PORTFOLIO)
    assert gpu.metadata["probe_trajectory_factor_tiles"] == gpu.metadata["factor_tiles_processed"]


def test_all_9_exposure_metrics_execute_real_cpu_and_strict_cuda():
    batch, labels = contracts(); rng = np.random.default_rng(94)
    risk = rng.normal(size=(40, 100, 6))
    panel = ExposurePanel(
        risk, style_names=("industry", "size", "beta", "liquidity", "volatility", "momentum"),
        source_ref="synthetic:gpu-capability-risk", provider="independent-fixture",
        date_index=tuple(batch.time_axis.values.tolist()), security_ids=tuple(batch.asset_axis.values.tolist()),
        factor_ids=batch.factor_ids, universe_snapshot_ref="universe:synthetic-100")
    kwargs = dict(metrics=EXPOSURE, exposure_panel=panel,
                  metric_parameters={"max_absolute_style_exposure": {"min_finite": 5},
                                     "purity_ratio": {"min_finite": 5}})
    cpu = evaluate(batch, labels, **kwargs)
    gpu = evaluate(batch, labels, backend="cuda_strict", **kwargs)
    assert_public_parity(cpu, gpu, EXPOSURE)
    assert gpu.metadata["exposure_kernel_dispatches"] >= 9


def test_v5_long_only_and_cost_metrics_are_strictly_unsupported_before_gpu_open(monkeypatch):
    batch, labels = contracts(); batch = replace(batch, factor_ids=("f",), values=batch.values[:, :, :1])
    dates = tuple(batch.time_axis.values.tolist())
    trajectory = build_research_trajectory(
        scenario_id="net", profile="LONG_ONLY_RESEARCH", dates=dates,
        gross_return=np.linspace(-.01, .01, 40), benchmark_return=np.zeros(40),
        cost_contributions={"fees": np.full(40, .001)},
        refs=TrajectoryRefs(("f",), "source:f", "portfolio:f", "cost:f", "benchmark:f"),
        scope=CostScope.NET_ASSUMED)
    trajectory = replace(trajectory, contributions={**trajectory.contributions,
                         "investment_fraction": tuple(np.full(40, .8))})
    legs = {"turnover_cost": "cost_drag", "tracking_error": "active",
            "information_ratio": "active", "relative_max_drawdown": "relative_return",
            "mean_investment_fraction": "investment_fraction"}
    opened = []
    monkeypatch.setattr(DeviceEvaluationSession, "_open", lambda self: opened.append(True))
    for metric, leg in legs.items():
        artifact = trajectory_to_probe_artifact(
            trajectory, expected_portfolio_profile="LONG_ONLY_RESEARCH",
            expected_cost_profile="net-base", expected_leg=leg)
        with pytest.raises(UnsupportedMetricError, match=metric):
            evaluate(batch, labels, metrics=(metric,), portfolio_returns=artifact,
                     backend="cuda_strict")
    assert opened == []


def test_supported_matrix_is_exactly_32_without_metadata_only_entries():
    assert len(BASE) + len(PORTFOLIO) + len(EXPOSURE) == 32
    assert set(BASE) | set(PORTFOLIO) | set(EXPOSURE) == GPUExecutor.SUPPORTED_METRICS
