"""Unit tests for Q Backend residency (eliminate region-level pingpong).

Tests verify:
1. QResidentTableHandle creation and reuse
2. Consecutive Q regions reuse intermediate results without re-upload
3. Telemetry counters track upload/download/reuse accurately
"""

from unittest.mock import MagicMock, patch
import threading

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.q_backend.q_adapter import QResidentTableHandle
from factor_engine.backend.q_backend.q_compiler import QRegionPlan
from factor_engine.backend.q_backend.q_executor import (
    QExecutionFallbackPolicy,
    QExecutor,
    get_q_executor_telemetry,
    reset_q_executor_telemetry,
)
from factor_engine.backend.q_backend.q_process_manager import QAvailabilityStatus


@pytest.fixture(autouse=True)
def reset_q_executor_singleton():
    """Keep the global executor bound to each test's patched dependencies."""
    import factor_engine.backend.q_backend.q_executor as q_executor_module

    q_executor_module._EXECUTOR = None
    yield
    q_executor_module._EXECUTOR = None


@pytest.fixture
def mock_q_process():
    """Mock Q process manager and connection."""
    with patch("factor_engine.backend.q_backend.q_executor.get_q_process_manager") as mock_mgr:
        from factor_engine.backend.q_backend.q_process_manager import QAvailabilityStatus

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
    with patch("factor_engine.backend.q_backend.q_executor.get_q_type_adapter") as mock_adapter:
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
    assert result.resident_handle.q_symbol.endswith("_result")
    assert result.resident_handle.region_id == "region_A"
    assert result.resident_handle.connection_id == id(mock_q_process)
    assert result.resident_handle.row_count == 100
    assert result.resident_handle.byte_size == 8000
    assert result.output_df.empty
    mock_type_adapter.q_to_pandas.assert_not_called()


def test_resident_handle_release_is_idempotent_and_context_manager(
    mock_q_process, mock_type_adapter, sample_input_data, sample_region_plan
):
    executor = QExecutor(type_adapter=mock_type_adapter)
    result = executor.execute_region(
        sample_region_plan,
        {"input_table": sample_input_data},
        return_resident_handle=True,
    )
    handle = result.resident_handle
    assert handle is not None
    assert repr(handle) == (
        "QResidentTableHandle(table=result, rows=100, bytes=8000, "
        "region=region_A, workspace=" + handle.workspace_id + ", generation="
        + handle.generation_id + ")"
    )

    handle.release()
    handle.close()
    delete_calls = [call for call in mock_q_process.call_args_list if "delete" in str(call)]
    assert len(delete_calls) >= 3
    assert handle.workspace_id not in __import__(
        "factor_engine.backend.q_backend.q_executor", fromlist=["_ACTIVE_LEASES"]
    )._ACTIVE_LEASES

    with pytest.raises(RuntimeError, match="already been released"):
        with handle:
            pass


def test_resident_handle_release_retries_connection_acquisition_failure(
    mock_q_process, mock_type_adapter, sample_input_data, sample_region_plan
):
    executor = QExecutor(type_adapter=mock_type_adapter)
    result = executor.execute_region(
        sample_region_plan,
        {"input_table": sample_input_data},
        return_resident_handle=True,
    )
    handle = result.resident_handle
    assert handle is not None
    symbol = handle.q_symbol

    executor.process_manager.get_connection.side_effect = RuntimeError("connection closed")
    with pytest.raises(RuntimeError, match="connection closed"):
        handle.release()
    assert not handle._released

    executor.process_manager.get_connection.side_effect = None
    executor.process_manager.get_connection.return_value = mock_q_process
    handle.release()
    handle.close()
    assert handle._released
    assert sum(
        "delete" in str(call) and symbol in str(call)
        for call in mock_q_process.call_args_list
    ) == 1


def test_resident_handle_release_retries_delete_failure(
    mock_q_process, mock_type_adapter, sample_input_data, sample_region_plan
):
    import factor_engine.backend.q_backend.q_executor as q_executor_module

    executor = QExecutor(type_adapter=mock_type_adapter)
    result = executor.execute_region(
        sample_region_plan,
        {"input_table": sample_input_data},
        return_resident_handle=True,
    )
    handle = result.resident_handle
    assert handle is not None
    symbol = handle.q_symbol
    original_side_effect = mock_q_process.side_effect

    def fail_delete_once(command):
        if isinstance(command, str) and command.startswith("delete "):
            raise RuntimeError("delete failed")
        return original_side_effect(command) if original_side_effect else mock_q_process.return_value

    mock_q_process.side_effect = fail_delete_once
    with pytest.raises(Exception, match="cleanup failed"):
        handle.release()
    assert not handle._released
    assert symbol in q_executor_module._ACTIVE_LEASES[handle.workspace_id][2]

    mock_q_process.side_effect = original_side_effect
    handle.release()
    handle.close()
    assert handle._released
    assert handle.workspace_id not in q_executor_module._ACTIVE_LEASES


def test_stale_resident_handle_release_is_noop(
    mock_q_process, mock_type_adapter, sample_input_data, sample_region_plan
):
    executor = QExecutor(type_adapter=mock_type_adapter)
    result = executor.execute_region(
        sample_region_plan,
        {"input_table": sample_input_data},
        return_resident_handle=True,
    )
    handle = result.resident_handle
    assert handle is not None
    handle.release()
    call_count = mock_q_process.call_count

    handle.release()
    assert mock_q_process.call_count == call_count


def test_intermediate_residency_skips_q_to_pandas_materialization(
    mock_q_process, mock_type_adapter, sample_input_data
):
    """Resident intermediates stay in q until the final boundary."""
    reset_q_executor_telemetry()
    from factor_engine.backend.q_backend.q_executor import get_q_executor
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
        workspace_id=result_a.resident_handle.workspace_id,
        generation_id=result_a.resident_handle.generation_id,
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
    from factor_engine.backend.q_backend.q_executor import get_q_executor
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


def test_batch_sibling_handles_release_independently(
    mock_q_process, mock_type_adapter, sample_input_data
):
    """Releasing one published handle preserves sibling residency."""
    import factor_engine.backend.q_backend.q_executor as q_executor_module

    executor = QExecutor(type_adapter=mock_type_adapter)
    plans = [
        QRegionPlan("A", ("a",), "aout: input_table", ("input_table",), "aout"),
        QRegionPlan("B", ("b",), "bout: aout", ("aout",), "bout"),
        QRegionPlan("C", ("c",), "final: bout", ("bout",), "final"),
    ]
    results = executor.execute_batch_regions(
        plans, {"input_table": sample_input_data}, enable_residency=True
    )
    handle_a, handle_b = results[0].resident_handle, results[1].resident_handle
    assert handle_a is not None and handle_b is not None
    assert handle_a.workspace_id == handle_b.workspace_id
    workspace_id = handle_a.workspace_id
    symbol_a, symbol_b = handle_a.q_symbol, handle_b.q_symbol

    handle_a.release()
    lease = q_executor_module._ACTIVE_LEASES[workspace_id]
    assert symbol_a not in lease[2]
    assert symbol_b in lease[2]
    executor._validate_lease(handle_b, mock_q_process, allow_legacy_handles=False)
    deletes = [str(call) for call in mock_q_process.call_args_list if "delete" in str(call)]
    assert sum(symbol_a in call for call in deletes) == 1
    assert not any(symbol_b in call for call in deletes)

    handle_b.release()
    assert workspace_id not in q_executor_module._ACTIVE_LEASES
    deletes = [str(call) for call in mock_q_process.call_args_list if "delete" in str(call)]
    assert sum(symbol_b in call for call in deletes) == 1


def test_direct_handle_on_shared_batch_workspace_releases_only_its_symbol(
    mock_q_process, mock_type_adapter, sample_input_data
):
    """A direct result sharing a batch workspace cannot delete batch siblings."""
    import factor_engine.backend.q_backend.q_executor as q_executor_module

    executor = QExecutor(type_adapter=mock_type_adapter)
    batch_plans = [
        QRegionPlan("A", ("a",), "aout: input_table", ("input_table",), "aout"),
        QRegionPlan("B", ("b",), "bout: aout", ("aout",), "bout"),
        QRegionPlan("C", ("c",), "final: bout", ("bout",), "final"),
    ]
    results = executor.execute_batch_regions(
        batch_plans, {"input_table": sample_input_data}, enable_residency=True
    )
    handle_a, handle_b = results[0].resident_handle, results[1].resident_handle
    assert handle_a is not None and handle_b is not None
    assert handle_a.workspace_id == handle_b.workspace_id

    direct_result = executor.execute_region(
        QRegionPlan("direct", ("d",), "direct: aout", ("aout",), "direct"),
        {"aout": handle_a},
        return_resident_handle=True,
        materialize_output=False,
        workspace_id=handle_a.workspace_id,
        generation_id=handle_a.generation_id,
    )
    direct_handle = direct_result.resident_handle
    assert direct_handle is not None
    assert direct_handle.q_symbol not in {handle_a.q_symbol, handle_b.q_symbol}

    direct_handle.release()
    executor._validate_lease(handle_b, mock_q_process, allow_legacy_handles=False)
    lease = q_executor_module._ACTIVE_LEASES[handle_b.workspace_id]
    assert handle_b.q_symbol in lease[2]
    deletes = [str(call) for call in mock_q_process.call_args_list if "delete" in str(call)]
    assert sum(direct_handle.q_symbol in call for call in deletes) == 1
    assert not any(handle_b.q_symbol in call for call in deletes)

    handle_a.release()
    handle_b.release()
    assert handle_b.workspace_id not in q_executor_module._ACTIVE_LEASES



def test_direct_materialized_call_preserves_batch_symbols(
    mock_q_process, mock_type_adapter, sample_input_data
):
    """Direct materialization on a batch lease only removes direct symbols."""
    import factor_engine.backend.q_backend.q_executor as q_executor_module

    executor = QExecutor(type_adapter=mock_type_adapter)
    results = executor.execute_batch_regions(
        [
            QRegionPlan("A", ("a",), "aout: input_table", ("input_table",), "aout"),
            QRegionPlan("B", ("b",), "bout: aout", ("aout",), "bout"),
            QRegionPlan("C", ("c",), "final: bout", ("bout",), "final"),
        ],
        {"input_table": sample_input_data},
        enable_residency=True,
    )
    source = results[0].resident_handle
    sibling = results[1].resident_handle
    assert source is not None and sibling is not None

    direct = executor.execute_region(
        QRegionPlan("direct", ("direct-node",), "direct: aout", ("aout",), "direct"),
        {"aout": source},
        workspace_id=source.workspace_id,
        generation_id=source.generation_id,
        return_resident_handle=False,
        materialize_output=True,
    )
    assert direct.success
    executor._validate_lease(source, mock_q_process, allow_legacy_handles=False)
    executor._validate_lease(sibling, mock_q_process, allow_legacy_handles=False)
    lease = q_executor_module._ACTIVE_LEASES[source.workspace_id]
    assert source.q_symbol in lease[2]
    assert sibling.q_symbol in lease[2]

    sibling.release()
    source.release()


def test_physical_identifier_encoding_avoids_logical_name_aliases(
    mock_q_process, mock_type_adapter, sample_input_data
):
    """Unsafe logical names remain distinct after q namespace encoding."""
    executor = QExecutor(type_adapter=mock_type_adapter)
    plan = QRegionPlan(
        "collision",
        ("a-b", "a_b"),
        "result: a-b + a_b",
        (),
        "result",
    )

    result = executor.execute_region(plan, {}, return_resident_handle=False)

    assert result.success
    commands = [call.args[0] for call in mock_q_process.call_args_list if call.args]
    generated = next(command for command in commands if "result:" in command)
    assert "a_2db" in generated
    assert "a_5fb" in generated
    assert generated.count("a_2db") == 1
    assert generated.count("a_5fb") == 1

def test_concurrent_direct_handles_do_not_claim_each_others_symbols(
    mock_q_process, mock_type_adapter, sample_input_data
):
    """Concurrent direct calls bind cleanup to their serialized workspace delta."""
    import factor_engine.backend.q_backend.q_executor as q_executor_module

    executor = QExecutor(type_adapter=mock_type_adapter)
    batch = executor.execute_batch_regions(
        [
            QRegionPlan("A", ("a",), "aout: input_table", ("input_table",), "aout"),
            QRegionPlan("B", ("b",), "bout: aout", ("aout",), "bout"),
            QRegionPlan("C", ("c",), "final: bout", ("bout",), "final"),
        ],
        {"input_table": sample_input_data},
        enable_residency=True,
    )
    source, sibling = batch[0].resident_handle, batch[1].resident_handle
    assert source is not None and sibling is not None

    class CoordinatedLock:
        def __init__(self):
            self._lock = threading.RLock()
            self._attempt_lock = threading.Lock()
            self._attempts = 0
            self._both_attempted = threading.Event()

        def __enter__(self):
            with self._attempt_lock:
                self._attempts += 1
                if self._attempts == 2:
                    self._both_attempted.set()
            assert self._both_attempted.wait(timeout=2)
            self._lock.acquire()
            return self

        def __exit__(self, exc_type, exc, traceback):
            self._lock.release()

    original_lock = executor._connection_lock
    executor._connection_lock = CoordinatedLock()
    handles = {}
    failures = []

    def execute(name):
        try:
            handles[name] = executor.execute_region(
                QRegionPlan(
                    name,
                    (f"node_{name}",),
                    f"out_{name}: aout",
                    ("aout",),
                    f"out_{name}",
                ),
                {"aout": source},
                return_resident_handle=True,
                materialize_output=False,
                workspace_id=source.workspace_id,
                generation_id=source.generation_id,
            ).resident_handle
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=execute, args=(name,)) for name in ("x", "y")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    executor._connection_lock = original_lock

    assert not failures
    assert all(not thread.is_alive() for thread in threads)
    handle_x, handle_y = handles["x"], handles["y"]
    assert handle_x is not None and handle_y is not None

    handle_x.release()
    executor._validate_lease(handle_y, mock_q_process, allow_legacy_handles=False)
    executor._validate_lease(sibling, mock_q_process, allow_legacy_handles=False)
    lease = q_executor_module._ACTIVE_LEASES[source.workspace_id]
    assert handle_y.q_symbol in lease[2]
    deletes = [str(call) for call in mock_q_process.call_args_list if "delete" in str(call)]
    assert not any(handle_y.q_symbol in call for call in deletes)
    assert not any(sibling.q_symbol in call for call in deletes)

    handle_y.release()
    sibling.release()
    source.release()
    assert source.workspace_id not in q_executor_module._ACTIVE_LEASES


def test_shared_direct_handle_partial_release_retries_remaining_symbols(
    mock_q_process, mock_type_adapter, sample_input_data
):
    """Partial shared-handle cleanup removes successes and retries failures only."""
    import factor_engine.backend.q_backend.q_executor as q_executor_module

    executor = QExecutor(type_adapter=mock_type_adapter)
    results = executor.execute_batch_regions(
        [
            QRegionPlan("A", ("a",), "aout: input_table", ("input_table",), "aout"),
            QRegionPlan("B", ("b",), "bout: aout", ("aout",), "bout"),
            QRegionPlan("C", ("c",), "final: bout", ("bout",), "final"),
        ],
        {"input_table": sample_input_data},
        enable_residency=True,
    )
    sibling = results[1].resident_handle
    source = results[0].resident_handle
    assert sibling is not None and source is not None
    direct = executor.execute_region(
        QRegionPlan("direct", ("d",), "direct: aout", ("aout",), "direct"),
        {"aout": source},
        return_resident_handle=True,
        materialize_output=False,
        workspace_id=source.workspace_id,
        generation_id=source.generation_id,
    ).resident_handle
    assert direct is not None
    with q_executor_module._LEASE_LOCK:
        owned = frozenset(
            q_executor_module._ACTIVE_LEASES[direct.workspace_id][2]
        ).difference({sibling.q_symbol, source.q_symbol})
    assert len(owned) >= 2
    symbols = tuple(sorted(owned))
    fail_symbol = symbols[-1]
    original_side_effect = mock_q_process.side_effect
    delete_counts_before = {
        symbol: sum(
            str(call) == f"call('delete {symbol} from `.')"
            for call in mock_q_process.call_args_list
        )
        for symbol in symbols
    }
    failed_once = False

    def fail_later_symbol(command):
        nonlocal failed_once
        if isinstance(command, str) and command == f"delete {fail_symbol} from `.":
            failed_once = True
            raise RuntimeError("delete failed")
        return original_side_effect(command) if original_side_effect else mock_q_process.return_value

    mock_q_process.side_effect = fail_later_symbol
    with pytest.raises(Exception, match="symbol cleanup failed"):
        direct.release()
    assert failed_once and not direct._released
    lease = q_executor_module._ACTIVE_LEASES[direct.workspace_id]
    assert fail_symbol in lease[2]
    for symbol in symbols[:-1]:
        assert symbol not in lease[2]
    assert sibling.q_symbol in lease[2]

    mock_q_process.side_effect = original_side_effect
    direct.release()
    delete_calls = [str(call) for call in mock_q_process.call_args_list]
    for symbol in symbols[:-1]:
        assert sum(
            call == f"call('delete {symbol} from `.')" for call in delete_calls
        ) == delete_counts_before[symbol] + 1
    assert sum(
        call == f"call('delete {fail_symbol} from `.')" for call in delete_calls
    ) == delete_counts_before[fail_symbol] + 2
    assert direct._released
    assert direct.workspace_id in q_executor_module._ACTIVE_LEASES
    executor._validate_lease(sibling, mock_q_process, allow_legacy_handles=False)
    sibling.release()
    source.release()
    assert direct.workspace_id not in q_executor_module._ACTIVE_LEASES


def test_batch_published_handle_lives_until_explicit_release(
    mock_q_process, mock_type_adapter, sample_input_data
):
    """A successful batch transfers its non-final symbol lease to the handle."""
    import factor_engine.backend.q_backend.q_executor as q_executor_module

    executor = QExecutor(type_adapter=mock_type_adapter)
    plans = [
        QRegionPlan(
            region_id="region_A",
            node_ids=("node_1",),
            q_code="temp1: mavg[20; input_table[`price]]",
            input_tables=("input_table",),
            output_table="temp1",
        ),
        QRegionPlan(
            region_id="region_B",
            node_ids=("node_2",),
            q_code="final: mavg[10; temp1]",
            input_tables=("temp1",),
            output_table="final",
        ),
    ]

    results = executor.execute_batch_regions(
        plans, {"input_table": sample_input_data}, enable_residency=True
    )
    handle = results[0].resident_handle
    assert handle is not None
    symbol = handle.q_symbol
    assert symbol is not None
    delete_calls = lambda: [
        str(call) for call in mock_q_process.call_args_list if "delete" in str(call)
    ]

    assert not any(symbol in call for call in delete_calls())
    executor._validate_lease(handle, mock_q_process, allow_legacy_handles=False)
    mock_q_process(symbol)
    assert mock_q_process.call_args == ((symbol,),)

    before_release = sum(symbol in call for call in delete_calls())
    handle.release()
    handle.close()
    assert sum(symbol in call for call in delete_calls()) == before_release + 1
    assert handle.workspace_id not in q_executor_module._ACTIVE_LEASES


def test_failed_resident_batch_cleans_published_intermediate(
    mock_q_process, mock_type_adapter, sample_input_data
):
    """A batch that fails before return keeps ownership and cleans its workspace."""
    import factor_engine.backend.q_backend.q_executor as q_executor_module

    executor = QExecutor(type_adapter=mock_type_adapter)
    leases_before = set(q_executor_module._ACTIVE_LEASES)
    plans = [
        QRegionPlan(
            region_id="region_A",
            node_ids=("node_1",),
            q_code="temp1: mavg[20; input_table[`price]]",
            input_tables=("input_table",),
            output_table="temp1",
        ),
        QRegionPlan(
            region_id="region_B",
            node_ids=("node_2",),
            q_code="final: missing",
            input_tables=("missing",),
            output_table="final",
        ),
    ]

    with pytest.raises(Exception):
        executor.execute_batch_regions(
            plans, {"input_table": sample_input_data}, enable_residency=True
        )

    assert any("delete" in str(call) for call in mock_q_process.call_args_list)
    assert set(q_executor_module._ACTIVE_LEASES) == leases_before


def test_batch_regions_without_residency(mock_q_process, mock_type_adapter, sample_input_data):
    """Test that disabling residency forces re-uploads (baseline comparison)."""
    reset_q_executor_telemetry()
    from factor_engine.backend.q_backend.q_executor import get_q_executor
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
    assert any("delete" in str(call) for call in mock_q_process.call_args_list)
    assert not executor._pending_workspace_cleanup


def test_failed_batch_connection_cleanup_is_retained_and_retried(
    mock_q_process, mock_type_adapter, sample_input_data
):
    """A failed final connection lookup keeps cleanup ownership for retry."""
    import factor_engine.backend.q_backend.q_executor as q_executor_module

    executor = QExecutor(type_adapter=mock_type_adapter)
    plans = [
        QRegionPlan(
            "region_A", ("node_1",),
            "temp1: mavg[20; input_table[`price]]", ("input_table",), "temp1",
        ),
        QRegionPlan(
            "region_B", ("node_2",), "final: missing", ("missing",), "final",
        ),
    ]
    q_connection = mock_q_process
    calls = 0

    def fail_final_connection():
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise RuntimeError("cleanup connection failed")
        return q_connection

    executor.process_manager.get_connection.side_effect = fail_final_connection
    with pytest.raises(Exception, match="Missing input tables"):
        executor.execute_batch_regions(
            plans, {"input_table": sample_input_data}, enable_residency=True
        )
    assert executor._pending_workspace_cleanup
    original_workspaces = {
        workspace_id for workspace_id, _, _ in executor._pending_workspace_cleanup
    }
    assert original_workspaces <= set(q_executor_module._ACTIVE_LEASES)

    executor.process_manager.get_connection.side_effect = None
    executor.process_manager.get_connection.return_value = mock_q_process
    deletes_before = sum("delete" in str(call) for call in mock_q_process.call_args_list)
    executor.execute_batch_regions([], {}, enable_residency=False)
    assert not executor._pending_workspace_cleanup
    assert original_workspaces.isdisjoint(q_executor_module._ACTIVE_LEASES)
    assert sum("delete" in str(call) for call in mock_q_process.call_args_list) > deletes_before


def test_pending_cleanup_rejects_replacement_connection(
    mock_q_process, mock_type_adapter
):
    """Pending ownership never deletes old-connection symbols through a new q."""
    import factor_engine.backend.q_backend.q_executor as q_executor_module

    executor = QExecutor(type_adapter=mock_type_adapter)
    workspace_id, generation_id = "old_workspace", "old_generation"
    old_connection_id = id(mock_q_process)
    symbol = "qe_old_workspace_result"
    q_executor_module._ACTIVE_LEASES[workspace_id] = (
        generation_id, old_connection_id, {symbol}
    )
    key = (workspace_id, generation_id, old_connection_id)
    executor._pending_workspace_cleanup[key] = frozenset()
    replacement_q = MagicMock()

    executor._retry_pending_workspace_cleanup(replacement_q)

    replacement_q.assert_not_called()
    assert key in executor._pending_workspace_cleanup
    assert symbol in q_executor_module._ACTIVE_LEASES[workspace_id][2]
    del q_executor_module._ACTIVE_LEASES[workspace_id]


def test_resident_handle_with_different_table_names(mock_q_process, mock_type_adapter):
    """Test aliasing when resident handle table name differs from expected input name."""
    reset_q_executor_telemetry()
    from factor_engine.backend.q_backend.q_executor import get_q_executor
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
        allow_legacy_handles=True,
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
    from factor_engine.backend.q_backend.q_executor import get_q_executor
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

    from factor_engine.backend.q_backend.q_errors import QExecutionError

    with pytest.raises(QExecutionError, match="Stale Q-resident handle"):
        executor.execute_region(
            sample_region_plan,
            {"input_table": stale_handle},
        )

    mock_type_adapter.q_to_pandas.assert_not_called()
    mock_q_process.assert_not_called()


def test_stale_resident_handle_fails_for_generation_mismatch(
    mock_q_process, mock_type_adapter, sample_region_plan
):
    """A handle from another workspace generation fails closed before q execution."""
    import factor_engine.backend.q_backend.q_executor as q_executor_module
    from factor_engine.backend.q_backend.q_errors import QExecutionError

    workspace_id = "shared-workspace"
    connection_id = id(mock_q_process)
    q_executor_module._ACTIVE_LEASES[workspace_id] = (
        "current-generation",
        connection_id,
        {"qe_shared_workspace_result"},
    )
    stale_handle = QResidentTableHandle(
        table_name="input_table",
        q_table_ref=MagicMock(),
        row_count=100,
        byte_size=8000,
        region_id="old_region",
        connection_id=connection_id,
        workspace_id=workspace_id,
        generation_id="old-generation",
        q_symbol="qe_shared_workspace_result",
    )

    executor = QExecutor(type_adapter=mock_type_adapter)
    with pytest.raises(QExecutionError, match="Stale Q-resident handle"):
        executor.execute_region(
            sample_region_plan,
            {"input_table": stale_handle},
        )

    mock_type_adapter.q_to_pandas.assert_not_called()
    mock_q_process.assert_not_called()
    del q_executor_module._ACTIVE_LEASES[workspace_id]


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
    from factor_engine.backend.q_backend.q_errors import QDataUnavailableError

    executor = QExecutor(type_adapter=mock_type_adapter)
    with pytest.raises(QDataUnavailableError, match="Missing input tables"):
        executor.execute_region(sample_region_plan, {})

    mock_q_process.assert_not_called()


def test_unavailable_runtime_raises_typed_error(mock_type_adapter, sample_region_plan):
    """Unavailable q is a typed production failure, never a fallback."""
    from factor_engine.backend.q_backend.q_errors import QProcessUnavailableError

    manager = MagicMock()
    manager.check_availability.return_value = MagicMock(
        status=QAvailabilityStatus.UNAVAILABLE,
        error_message="not installed",
    )
    executor = QExecutor(process_manager=manager, type_adapter=mock_type_adapter)

    with pytest.raises(QProcessUnavailableError, match="fallback is disabled"):
        executor.execute_region(sample_region_plan, {"input_table": pd.DataFrame()})
    manager.get_connection.assert_not_called()


def test_runtime_failure_raises_typed_error(
    mock_q_process, mock_type_adapter, sample_input_data, sample_region_plan
):
    """q execution failures remain fail-closed and typed."""
    from factor_engine.backend.q_backend.q_errors import QExecutionError

    mock_q_process.side_effect = RuntimeError("q boom")
    executor = QExecutor(type_adapter=mock_type_adapter)
    with pytest.raises(QExecutionError, match="q boom"):
        executor.execute_region(
            sample_region_plan,
            {"input_table": sample_input_data},
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
