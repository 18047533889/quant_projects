"""Public facade backend names must not silently select an unintended CPU route."""

import pytest

from quant_evaluator.api.requests import EvaluationRequest
from quant_evaluator.backends.selector import BackendType
from quant_evaluator.contracts.backend_policy import BackendPolicy
from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.contracts.metric_instance import MetricInstance
from quant_evaluator.runtime.evaluator import evaluate
from quant_evaluator.tests.test_v3_public_artifacts import inputs


@pytest.mark.parametrize("backend", [
    "cuad", "numba", "polars", "cpu_fast", "cpu_fp64", "",
    BackendType.NUMBA, BackendType.POLARS, BackendPolicy.CPU_FAST, 7,
])
def test_unknown_public_backend_fails_closed(backend):
    batch, label = inputs()
    with pytest.raises(InvalidContractError, match="Unsupported public evaluate backend"):
        evaluate(batch, label, metrics=("coverage",), backend=backend)


@pytest.mark.parametrize("backend", ["numba", BackendType.POLARS, BackendPolicy.CPU_FAST])
def test_unknown_backend_rejected_before_metric_instance_delegation(backend):
    batch, label = inputs()
    request = EvaluationRequest(
        batch, label, metric_instances=(MetricInstance("coverage"),)
    )
    with pytest.raises(InvalidContractError, match="Unsupported public evaluate backend"):
        evaluate(request, backend=backend)


def test_supported_cpu_and_auto_enum_keep_working():
    batch, label = inputs()
    cpu = evaluate(batch, label, metrics=("coverage",), backend="CPU")
    auto = evaluate(batch, label, metrics=("coverage",), backend=BackendPolicy.AUTO)
    assert cpu.metric_values == auto.metric_values
    assert auto.metadata["backend_used"] == "cpu"
