"""Internal request artifacts are not user-selectable metric variants."""
import pytest
from quant_evaluator.contracts.metric_instance import MetricInstance


@pytest.mark.parametrize("metric_id", ["quantile_returns_full", "quantile_returns_daily"])
@pytest.mark.parametrize("name", ["daily_quantile_artifact", "_bind_request_inputs"])
def test_metric_instance_rejects_caller_injected_runtime_inputs(metric_id, name):
    with pytest.raises(ValueError, match="Invalid metric instance parameters"):
        MetricInstance(metric_id, parameters={name: None})
