"""Full metric warmup must initialize real public source computations."""
import numpy as np
import pytest
from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.runtime.source_profile_report_schema import F61_ALL15_METRICS, F61_ALL24_METRICS
from quant_evaluator.scripts.source_profile_warmup import warm_source_profile


@pytest.mark.parametrize("metrics", [F61_ALL15_METRICS, F61_ALL24_METRICS], ids=["all15", "all24"])
@pytest.mark.parametrize("backend,expected_backend", [("cpu", "cpu"), ("cuda_strict", "cuda")])
def test_warmup_computes_complete_metric_domain(metrics, backend, expected_backend):
    result = warm_source_profile(GPUExecutionPolicy(), backend, metrics)
    assert type(result) is BatchEvaluationBundle
    assert result.factor_ids == ("profile-warm-a", "profile-warm-b")
    assert set(result.scalar_metrics) | set(result.series_metrics) == set(metrics)
    assert result.metadata["backend_used"] == expected_backend
    assert result.metadata["metric_backends"] == {metric: expected_backend for metric in metrics}
    if backend == "cuda_strict":
        assert result.metadata["execution_receipt"]["backend_used"] == "cuda"
        assert result.metadata["execution_receipt"]["backend_requested"] == "cuda_strict"
    for metric in metrics:
        values = (result.series_metrics if metric.endswith("_series") else result.scalar_metrics)[metric]
        if metric.endswith("_series"):
            assert values.shape == (65, 2)
        assert np.isfinite(values).all(), metric
    assert result.observation_counts["factor_turnover_rate"].tolist() == [64, 64]
