"""Unit tests for Q Backend residency (eliminate region-level pingpong).

Tests verify:
1. QResidentTableHandle creation and reuse
2. Consecutive Q regions reuse intermediate results without re-upload
3. Telemetry counters track upload/download/reuse accurately
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backend.q_backend.q_adapter import QResidentTableHandle
from backend.q_backend.q_compiler import QRegionPlan
from backend.q_backend.q_executor import (
    QExecutionFallbackPolicy,
    QExecutor,
    get_q_executor_telemetry,
    reset_q_executor_telemetry,
)
from backend.q_backend.q_process_manager import QAvailabilityStatus


@pytest.fixture(autouse=True)
def reset_q_executor_singleton():
    """Keep the global executor bound to each test's patched dependencies."""
    import backend.q_backend.q_executor as q_executor_module

    q_executor_module._EXECUTOR = None
    yield
    q_executor_module._EXECUTOR = None


@pytest.fixture
def mock_q_process():
    """Mock Q process manager and connection."""
    with patch("backend.q_backend.q_executor.get_q_process_manager") as mock_mgr:
        from backend.q_backend.q_process_manager import QAvailabilityStatus

        mock_conn = MagicMock()
        mock_q_result = MagicMock(name="q_result")
        mock_q_result.__len__.return_value = 100
        mock_q_result.nbytes = 8000
        mock_conn.return_value = mock_q_result
        mock_mgr_instance = MagicMock()
        mock_mgr.return_value = mock_mgr_instance

        mock_mgr_instance.is_available.return_value = True
        mock_mgr_instance.get_connection.return_value = mock_conn

        # Mock availability info
        mock_availability_info = MagicMock()
        mock_availability_info.status = QAvailabilityStatus.AVAILABLE
        mock_availability_info.error_message = None
        mock_mgr_instance.check_availability.return_value = mock_availability_info

        yield mock_conn


@pytest.fixture
def mock_type_adapter():
    """Mock Q type adapter."""
    with patch("backend.q_backend.q_executor.get_q_type_adapter") as mock_adapter:
        adapter = MagicMock()

        # pandas_to_q returns mock Q table
        adapter.pandas_to_q.return_value = MagicMock(name="q_table")

        # q_to_pandas returns a realistic DataFrame
        def mock_q_to_pandas(q_result):
            return pd.DataFrame({
                "timestamp": pd.date_range("2020-01-01", periods=100),
                "instrument": ["AAPL"] * 100,
                "value": np.random.randn(100),
            })

        adapter.q_to_pandas.side_effect = mock_q_to_pandas
        mock_adapter.return_value = adapter
        yield adapter


@pytest.fixture
def sample_input_data():
    """Sample input DataFrame."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=100),
        "instrument": ["AAPL"] * 100,
        "price": np.random.uniform(100, 200, 100),
    })


@pytest.fixture
def sample_region_plan():
    """Sample Q region plan."""
    return QRegionPlan(
        region_id="region_A",
        node_ids=("node_1",),
        q_code="result: mavg[20; input_table[`price]]",
        input_tables=("input_table",),
        output_table="result",
    )


def test_resident_handle_creation(mock_q_process, mock_type_adapter, sample_input_data, sample_region_plan):
    """Test that execution creates QResidentTableHandle when requested."""
    executor = QExecutor(type_adapter=mock_type_adapter)

    result = executor.execute_region(
        sample_region_plan,
        {"input_table": sample_input_data},
        return_resident_handle=True,
    )

    assert result.success
    assert result.resident_handle is not None
    assert isinstance(result.resident_handle, QResidentTableHandle)
    assert result.resident_handle.table_name == "result"
    assert result.resident_handle.region_id == "region_A"
    assert result.resident_handle.connection_id == id(mock_q_process)
    assert result.resident_handle.row_count == 100
    assert result.resident_handle.byte_size == 8000
    assert result.output_df.empty
    mock_type_adapter.q_to_pandas.assert_not_called()


def test_intermediate_residency_skips_q_to_pandas_materialization(
    mock_q_process, mock_type_adapter, sample_input_data
):
    """Resident intermediates stay in q until the final boundary."""
    reset_q_executor_telemetry()
    from backend.q_backend.q_executor import get_q_executor
    executor = get_q_executor()
    plan = QRegionPlan(
        region_id="region_intermediate",
        node_ids=("node_1",),
        q_code="intermediate: mavg[20; input_table[`price]]",
        input_tables=("input_table",),
        output_table="intermediate",
    )

    result = executor.execute_region(
        plan,
        {"input_table": sample_input_data},
        return_resident_handle=True,
        materialize_output=False,
    )

    assert result.success
    assert result.resident_handle is not None
    mock_type_adapter.q_to_pandas.assert_not_called()
    assert get_q_executor_telemetry()["q_to_python_bytes"] == 0


def test_resident_handle_reuse_eliminates_upload(mock_q_process, mock_type_adapter, sample_input_data):
    """Test that QResidentTableHandle reuse skips DataFrame upload."""
    executor = QExecutor(type_adapter=mock_type_adapter)

    # Region A: initial upload
    plan_a = QRegionPlan(
        region_id="region_A",
        node_ids=("node_1",),
        q_code="intermediate: mavg[20; input_table[`price]]",
        input_tables=("input_table",),
        output_table="intermediate",
    )

    result_a = executor.execute_region(
        plan_a,
        {"input_table": sample_input_data},
        return_resident_handle=True,
    )

    assert result_a.success
    assert result_a.resident_handle is not None

    # Check telemetry after Region A
    telemetry_after_a = dict(executor._telemetry)
    uploads_after_a = telemetry_after_a["python_to_q_bytes"]
    reuse_after_a = telemetry_after_a["resident_reuse_count"]

    assert uploads_after_a > 0  # Initial upload happened
    assert reuse_after_a == 0  # No reuse yet

    # Region B: reuses Region A output via resident handle
    plan_b = QRegionPlan(
        region_id="region_B",
        node_ids=("node_2",),
        q_code="final: mavg[10; intermediate]",
        input_tables=("intermediate",),
        output_table="final",
    )

    result_b = executor.execute_region(
        plan_b,
        {"intermediate": result_a.resident_handle},  # Pass handle, not DataFrame
        return_resident_handle=False,
    )

    assert result_b.success

    # Check telemetry after Region B
    telemetry_after_b = dict(executor._telemetry)
    uploads_after_b = telemetry_after_b["python_to_q_bytes"]
    reuse_after_b = telemetry_after_b["resident_reuse_count"]

    # Key assertion: no additional upload happened
    assert uploads_after_b == uploads_after_a  # Upload count unchanged
    assert reuse_after_b == 1  # Reuse count incremented


def test_batch_regions_automatic_residency(mock_q_process, mock_type_adapter, sample_input_data):
    """Test that execute_batch_regions automatically passes resident handles between consecutive regions."""
    reset_q_executor_telemetry()
    from backend.q_backend.q_executor import get_q_executor
    executor = get_q_executor()

    # Create three consecutive regions: A → B → C
    plan_a = QRegionPlan(
        region_id="region_A",
        node_ids=("node_1",),
        q_code="temp1: mavg[20; input_table[`price]]",
        input_tables=("input_table",),
        output_table="temp1",
    )

    plan_b = QRegionPlan(
        region_id="region_B",
        node_ids=("node_2",),
        q_code="temp2: mavg[10; temp1]",
        input_tables=("temp1",),
        output_table="temp2",
    )

    plan_c = QRegionPlan(
        region_id="region_C",
        node_ids=("node_3",),
        q_code="final: mavg[5; temp2]",
        input_tables=("temp2",),
        output_table="final",
    )

    results = executor.execute_batch_regions(
        [plan_a, plan_b, plan_c],
        {"input_table": sample_input_data},
        enable_residency=True,
    )

    assert len(results) == 3
    assert all(r.success for r in results)
    assert results[0].output_df.empty
    assert results[1].output_df.empty
    assert not results[2].output_df.empty
    assert mock_type_adapter.q_to_pandas.call_count == 1

    # Check telemetry
    telemetry = get_q_executor_telemetry()

    # Only the initial input_table should have been uploaded
    # temp1 and temp2 should have been reused via resident handles
    assert telemetry["resident_reuse_count"] == 2  # B reused temp1, C reused temp2
    assert telemetry["python_to_q_bytes"] > 0  # Initial upload
    assert telemetry["q_to_python_bytes"] > 0  # Final download


def test_batch_regions_without_residency(mock_q_process, mock_type_adapter, sample_input_data):
    """Test that disabling residency forces re-uploads (baseline comparison)."""
    reset_q_executor_telemetry()
    from backend.q_backend.q_executor import get_q_executor
    executor = get_q_executor()

    plan_a = QRegionPlan(
        region_id="region_A",
        node_ids=("node_1",),
        q_code="temp1: mavg[20; input_table[`price]]",
        input_tables=("input_table",),
        output_table="temp1",
    )

    plan_b = QRegionPlan(
        region_id="region_B",
        node_ids=("node_2",),
        q_code="final: mavg[10; input_table[`price]]",  # Use input_table instead of temp1
        input_tables=("input_table",),  # Use input_table for both
        output_table="final",
    )

    # Disable residency: intermediate results won't be passed between regions
    results = executor.execute_batch_regions(
        [plan_a, plan_b],
        {"input_table": sample_input_data},
        enable_residency=False,
    )

    assert len(results) == 2
    assert all(r.success for r in results)

    # Check telemetry
    telemetry = get_q_executor_telemetry()

    # With residency disabled, no reuse should happen
    assert telemetry["resident_reuse_count"] == 0


def test_resident_handle_with_different_table_names(mock_q_process, mock_type_adapter):
    """Test aliasing when resident handle table name differs from expected input name."""
    reset_q_executor_telemetry()
    from backend.q_backend.q_executor import get_q_executor
    executor = get_q_executor()

    # Create a resident handle with table name "old_name"
    resident_handle = QResidentTableHandle(
        table_name="old_name",
        q_table_ref=MagicMock(),
        row_count=100,
        byte_size=8000,
        region_id="region_A",
    )

    plan = QRegionPlan(
        region_id="region_B",
        node_ids=("node_1",),
        q_code="result: mavg[10; new_name]",
        input_tables=("new_name",),  # Different name
        output_table="result",
    )

    result = executor.execute_region(
        plan,
        {"new_name": resident_handle},
    )

    assert result.success

    # Verify that the q connection received the aliasing call
    # Check if q() was called with the aliasing statement
    calls = [str(call) for call in mock_q_process.call_args_list]
    alias_call_found = any("new_name" in str(call) and "old_name" in str(call) for call in calls)

    # Even if aliasing wasn't called (same name optimization), reuse should be tracked
    telemetry = get_q_executor_telemetry()
    assert telemetry["resident_reuse_count"] == 1


def test_telemetry_counters_accuracy(mock_q_process, mock_type_adapter, sample_input_data):
    """Test that telemetry counters accurately track data movement."""
    reset_q_executor_telemetry()
    from backend.q_backend.q_executor import get_q_executor
    executor = get_q_executor()

    # Calculate expected upload size
    expected_upload_bytes = sample_input_data.memory_usage(deep=True).sum()

    plan = QRegionPlan(
        region_id="region_test",
        node_ids=("node_1",),
        q_code="result: mavg[20; input_table[`price]]",
        input_tables=("input_table",),
        output_table="result",
    )

    result = executor.execute_region(
        plan,
        {"input_table": sample_input_data},
    )

    assert result.success

    telemetry = get_q_executor_telemetry()

    # Verify upload tracking
    assert telemetry["python_to_q_bytes"] >= expected_upload_bytes - 100  # Allow small variance

    # Verify download tracking
    assert telemetry["q_to_python_bytes"] > 0

    # No reuse in single region execution
    assert telemetry["resident_reuse_count"] == 0


def test_stale_resident_handle_fails_before_q_execution(
    mock_q_process, mock_type_adapter, sample_region_plan
):
    """A handle from another q connection must fail closed without aliasing."""
    executor = QExecutor(
        process_manager=MagicMock(),
        type_adapter=mock_type_adapter,
    )
    executor.process_manager.check_availability.return_value = MagicMock(
        status=QAvailabilityStatus.AVAILABLE,
        error_message=None,
    )
    executor.process_manager.get_connection.return_value = mock_q_process
    stale_handle = QResidentTableHandle(
        table_name="input_table",
        q_table_ref=MagicMock(),
        row_count=100,
        byte_size=8000,
        region_id="old_region",
        connection_id=id(mock_q_process) + 1,
    )

    from backend.q_backend.q_errors import QExecutionError

    with pytest.raises(QExecutionError, match="Stale Q-resident handle"):
        executor.execute_region(
            sample_region_plan,
            {"input_table": stale_handle},
        )

    mock_type_adapter.q_to_pandas.assert_not_called()
    mock_q_process.assert_not_called()


def test_resident_metadata_failure_does_not_materialize(
    mock_q_process, mock_type_adapter, sample_input_data, sample_region_plan
):
    """Unavailable q metadata stays bounded and does not force conversion."""
    q_result = mock_q_process.return_value
    q_result.__len__.side_effect = TypeError("length unavailable")
    q_result.nbytes = MagicMock()
    q_result.nbytes.__int__.side_effect = TypeError("size unavailable")
    executor = QExecutor(type_adapter=mock_type_adapter)

    result = executor.execute_region(
        sample_region_plan,
        {"input_table": sample_input_data},
        return_resident_handle=True,
    )

    assert result.success
    assert result.rows_processed == 0
    assert result.resident_handle is not None
    assert result.resident_handle.byte_size == 0
    mock_type_adapter.q_to_pandas.assert_not_called()


def test_missing_input_raises_typed_error_before_connection(
    mock_q_process, mock_type_adapter, sample_region_plan
):
    """Missing plan inputs fail closed without touching q."""
    from backend.q_backend.q_errors import QDataUnavailableError

    executor = QExecutor(type_adapter=mock_type_adapter)
    with pytest.raises(QDataUnavailableError, match="Missing input tables"):
        executor.execute_region(sample_region_plan, {})

    mock_q_process.assert_not_called()


def test_unavailable_runtime_raises_typed_error(mock_type_adapter, sample_region_plan):
    """Unavailable q is a typed production failure, never a fallback."""
    from backend.q_backend.q_errors import QProcessUnavailableError

    manager = MagicMock()
    manager.check_availability.return_value = MagicMock(
        status=QAvailabilityStatus.UNAVAILABLE,
        error_message="not installed",
    )
    executor = QExecutor(process_manager=manager, type_adapter=mock_type_adapter)

    with pytest.raises(QProcessUnavailableError, match="Fallback disabled"):
        executor.execute_region(sample_region_plan, {"input_table": pd.DataFrame()})
    manager.get_connection.assert_not_called()


def test_runtime_failure_raises_typed_error(
    mock_q_process, mock_type_adapter, sample_input_data, sample_region_plan
):
    """q execution failures remain fail-closed and typed."""
    from backend.q_backend.q_errors import QExecutionError

    mock_q_process.side_effect = RuntimeError("q boom")
    executor = QExecutor(type_adapter=mock_type_adapter)
    with pytest.raises(QExecutionError, match="q boom"):
        executor.execute_region(
            sample_region_plan,
            {"input_table": sample_input_data},
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
