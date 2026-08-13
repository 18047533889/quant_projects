"""
Test suite for mock adapter fixtures.
"""

import pytest
import numpy as np

from integration_tests.fixtures import (
    MockDataAccessAdapter,
    MockFactorEngineAdapter,
    MockEvaluationBackend,
    AdapterStatus,
    AdapterResponse,
    create_stable_adapters,
    create_flaky_adapters,
    create_slow_adapters,
)


def test_mock_da_adapter_read_factor():
    """Test mock DA adapter factor reading."""
    adapter = MockDataAccessAdapter(fail_rate=0.0, seed=42)

    response = adapter.read_factor(
        factor_id="test_factor",
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="top_500",
    )

    assert response.status == AdapterStatus.SUCCESS
    assert response.data is not None
    assert response.data.ndim == 2
    assert adapter.request_count == 1


def test_mock_da_adapter_write_factor():
    """Test mock DA adapter factor writing."""
    adapter = MockDataAccessAdapter(fail_rate=0.0, seed=42)

    data = np.random.randn(100, 50)
    metadata = {"timing": "daily", "pit_safe": True}

    response = adapter.write_factor(
        factor_id="new_factor",
        data=data,
        metadata=metadata,
    )

    assert response.status == AdapterStatus.SUCCESS
    assert adapter.request_count == 1

    # Read it back
    read_response = adapter.read_factor(
        factor_id="new_factor",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    assert read_response.status == AdapterStatus.SUCCESS
    np.testing.assert_array_equal(read_response.data, data)


def test_mock_da_adapter_read_universe():
    """Test mock DA adapter universe reading."""
    adapter = MockDataAccessAdapter(fail_rate=0.0, seed=42)

    response = adapter.read_universe(
        universe_id="top_500",
        as_of_date="2024-01-01",
    )

    assert response.status == AdapterStatus.SUCCESS
    assert response.data is not None
    assert len(response.data) > 0
    assert adapter.request_count == 1


def test_mock_da_adapter_get_schema():
    """Test mock DA adapter schema retrieval."""
    adapter = MockDataAccessAdapter(fail_rate=0.0, seed=42)

    # Write factor first
    data = np.random.randn(50, 30)
    metadata = {"timing": "daily", "frequency": "D"}
    adapter.write_factor("test_factor", data, metadata)

    # Get schema
    response = adapter.get_schema("test_factor")

    assert response.status == AdapterStatus.SUCCESS
    assert response.data["timing"] == "daily"


def test_mock_da_adapter_get_schema_not_found():
    """Test schema retrieval for non-existent factor."""
    adapter = MockDataAccessAdapter(fail_rate=0.0, seed=42)

    response = adapter.get_schema("non_existent")

    assert response.status == AdapterStatus.NOT_AVAILABLE
    assert response.error_message is not None


def test_mock_da_adapter_failure_mode():
    """Test mock DA adapter simulated failures."""
    adapter = MockDataAccessAdapter(fail_rate=1.0, seed=42)  # Always fail

    response = adapter.read_factor(
        factor_id="test",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    assert response.status == AdapterStatus.FAILED
    assert response.error_message is not None


def test_mock_da_adapter_request_history():
    """Test request tracking."""
    adapter = MockDataAccessAdapter(fail_rate=0.0, seed=42)

    adapter.read_factor("factor1", "2024-01-01", "2024-12-31")
    adapter.read_universe("universe1", "2024-01-01")
    adapter.write_factor("factor2", np.random.randn(10, 5), {})

    assert adapter.request_count == 3
    assert len(adapter.request_history) == 3

    # Check history details
    assert adapter.request_history[0]["method"] == "read_factor"
    assert adapter.request_history[0]["factor_id"] == "factor1"
    assert adapter.request_history[1]["method"] == "read_universe"
    assert adapter.request_history[2]["method"] == "write_factor"


def test_mock_da_adapter_reset():
    """Test adapter reset functionality."""
    adapter = MockDataAccessAdapter(fail_rate=0.0, seed=42)

    adapter.write_factor("test", np.random.randn(10, 5), {})
    adapter.read_factor("test", "2024-01-01", "2024-12-31")

    assert adapter.request_count == 2

    adapter.reset()

    assert adapter.request_count == 0
    assert len(adapter.request_history) == 0

    # Schema should be cleared
    response = adapter.get_schema("test")
    assert response.status == AdapterStatus.NOT_AVAILABLE


def test_mock_fe_adapter_execute_operator():
    """Test mock FE adapter operator execution."""
    adapter = MockFactorEngineAdapter(fail_rate=0.0, seed=42)

    inputs = {"data": np.random.randn(100, 50)}
    params = {"window": 20}

    response = adapter.execute_operator(
        operator_name="rolling_mean",
        inputs=inputs,
        params=params,
    )

    assert response.status == AdapterStatus.SUCCESS
    assert response.data is not None
    assert response.data.shape == (100, 50)
    assert adapter.execution_count == 1


def test_mock_fe_adapter_register_operator():
    """Test registering custom operators."""
    adapter = MockFactorEngineAdapter(fail_rate=0.0, seed=42)

    def custom_op(inputs, params):
        return inputs["x"] * params["multiplier"]

    adapter.register_operator(
        "custom_multiply",
        custom_op,
        metadata={"category": "arithmetic"},
    )

    # Execute registered operator
    inputs = {"x": np.array([1.0, 2.0, 3.0])}
    params = {"multiplier": 2.0}

    response = adapter.execute_operator("custom_multiply", inputs, params)

    assert response.status == AdapterStatus.SUCCESS
    np.testing.assert_array_equal(response.data, np.array([2.0, 4.0, 6.0]))


def test_mock_fe_adapter_operator_exception():
    """Test operator execution with exception."""
    adapter = MockFactorEngineAdapter(fail_rate=0.0, seed=42)

    def buggy_op(inputs, params):
        raise ValueError("Intentional error")

    adapter.register_operator("buggy", buggy_op)

    response = adapter.execute_operator(
        "buggy",
        inputs={"data": np.array([1.0])},
        params={},
    )

    assert response.status == AdapterStatus.FAILED
    assert "Intentional error" in response.error_message


def test_mock_fe_adapter_materialize_factor():
    """Test factor materialization."""
    adapter = MockFactorEngineAdapter(fail_rate=0.0, seed=42)

    response = adapter.materialize_factor(
        factor_expr="momentum(close, 20)",
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="top_500",
    )

    assert response.status == AdapterStatus.SUCCESS
    assert response.data is not None
    assert response.data.ndim == 2


def test_mock_fe_adapter_get_operator_metadata():
    """Test operator metadata retrieval."""
    adapter = MockFactorEngineAdapter(fail_rate=0.0, seed=42)

    metadata = {"category": "time_series", "parameters": ["window"]}
    adapter.register_operator("rolling_mean", lambda i, p: i["data"], metadata)

    response = adapter.get_operator_metadata("rolling_mean")

    assert response.status == AdapterStatus.SUCCESS
    assert response.data["category"] == "time_series"


def test_mock_fe_adapter_execution_history():
    """Test execution tracking."""
    adapter = MockFactorEngineAdapter(fail_rate=0.0, seed=42)

    adapter.execute_operator("op1", {"data": np.array([1.0])}, {"p": 1})
    adapter.execute_operator("op2", {"data": np.array([2.0])}, {"p": 2})

    assert adapter.execution_count == 2
    assert len(adapter.execution_history) == 2
    assert adapter.execution_history[0]["operator"] == "op1"
    assert adapter.execution_history[1]["operator"] == "op2"


def test_mock_evaluation_backend_evaluate():
    """Test mock evaluation backend."""
    backend = MockEvaluationBackend(fail_rate=0.0, seed=42)

    factor_values = np.random.randn(100, 50, 3)
    label_values = np.random.randn(100, 50)

    response = backend.evaluate(
        factor_values=factor_values,
        label_values=label_values,
        metric_ids=("pearson_ic", "rank_ic"),
    )

    assert response.status == AdapterStatus.SUCCESS
    assert "pearson_ic" in response.data
    assert "rank_ic" in response.data


def test_mock_evaluation_backend_register_metric():
    """Test registering custom metrics."""
    backend = MockEvaluationBackend(fail_rate=0.0, seed=42)

    def custom_metric(factor_values, label_values):
        return {"value": 0.123, "std_error": 0.01}

    backend.register_metric("custom_ic", custom_metric)

    response = backend.evaluate(
        factor_values=np.random.randn(50, 30, 1),
        label_values=np.random.randn(50, 30),
        metric_ids=("custom_ic",),
    )

    assert response.status == AdapterStatus.SUCCESS
    assert response.data["custom_ic"]["value"] == 0.123


def test_mock_evaluation_backend_metric_exception():
    """Test metric computation with exception."""
    backend = MockEvaluationBackend(fail_rate=0.0, seed=42)

    def buggy_metric(factor_values, label_values):
        raise RuntimeError("Computation failed")

    backend.register_metric("buggy_ic", buggy_metric)

    response = backend.evaluate(
        factor_values=np.random.randn(50, 30, 1),
        label_values=np.random.randn(50, 30),
        metric_ids=("buggy_ic",),
    )

    assert response.status == AdapterStatus.SUCCESS
    assert "error" in response.data["buggy_ic"]


def test_create_stable_adapters():
    """Test stable adapter factory."""
    da, fe = create_stable_adapters(seed=42)

    assert isinstance(da, MockDataAccessAdapter)
    assert isinstance(fe, MockFactorEngineAdapter)
    assert da.fail_rate == 0.0
    assert fe.fail_rate == 0.0


def test_create_flaky_adapters():
    """Test flaky adapter factory."""
    da, fe = create_flaky_adapters(seed=42)

    assert da.fail_rate > 0.0
    assert fe.fail_rate > 0.0


def test_create_slow_adapters():
    """Test slow adapter factory."""
    da, fe = create_slow_adapters(seed=42)

    assert da.latency_ms >= 500.0
    assert fe.latency_ms >= 1000.0


def test_adapter_response_dataclass():
    """Test AdapterResponse structure."""
    response = AdapterResponse(
        status=AdapterStatus.SUCCESS,
        data=np.array([1, 2, 3]),
        metadata={"key": "value"},
        latency_ms=15.0,
    )

    assert response.status == AdapterStatus.SUCCESS
    assert response.data is not None
    assert response.metadata["key"] == "value"
    assert response.latency_ms == 15.0
    assert response.error_message is None


def test_mock_da_adapter_reproducibility():
    """Test that same seed produces same behavior."""
    adapter1 = MockDataAccessAdapter(fail_rate=0.0, seed=123)
    adapter2 = MockDataAccessAdapter(fail_rate=0.0, seed=123)

    response1 = adapter1.read_factor("test", "2024-01-01", "2024-12-31")
    response2 = adapter2.read_factor("test", "2024-01-01", "2024-12-31")

    np.testing.assert_array_equal(response1.data, response2.data)


def test_mock_fe_adapter_reproducibility():
    """Test FE adapter reproducibility."""
    adapter1 = MockFactorEngineAdapter(fail_rate=0.0, seed=456)
    adapter2 = MockFactorEngineAdapter(fail_rate=0.0, seed=456)

    inputs = {"data": np.array([1.0, 2.0, 3.0])}
    params = {"window": 5}

    response1 = adapter1.execute_operator("test_op", inputs, params)
    response2 = adapter2.execute_operator("test_op", inputs, params)

    np.testing.assert_array_equal(response1.data, response2.data)
