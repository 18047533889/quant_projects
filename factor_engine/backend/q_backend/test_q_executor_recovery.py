"""Focused recovery tests for the q executor public ABI and runtime policy."""

from __future__ import annotations

from dataclasses import fields
import threading
import time
from unittest.mock import MagicMock

import pandas as pd
import pytest

from backend.q_backend.q_compiler import QRegionPlan
from backend.q_backend.q_adapter import QResidentTableHandle
from backend.q_backend.q_errors import (
    QDataUnavailableError,
    QExecutionError,
    QProcessUnavailableError,
)
from backend.q_backend.q_executor import (
    QExecutionFallbackPolicy,
    QExecutionResult,
    QExecutor,
    get_q_executor,
    get_q_executor_telemetry,
    reset_q_executor_telemetry,
)
from backend.q_backend.q_process_manager import QAvailabilityStatus


def _plan(
    region_id: str,
    inputs: tuple[str, ...],
    output: str,
) -> QRegionPlan:
    return QRegionPlan(
        region_id=region_id,
        node_ids=(region_id,),
        q_code=f"{output}: 0!{inputs[0]}",
        input_tables=inputs,
        output_table=output,
    )


def test_public_executor_abi_is_importable_and_constructible():
    policy = QExecutionFallbackPolicy()
    executor = QExecutor(process_manager=MagicMock(), type_adapter=MagicMock())

    assert policy.allow_fallback is False
    assert policy.fail_on_unavailable is True
    assert isinstance(executor, QExecutor)
    assert isinstance(get_q_executor(), QExecutor)
    assert set(get_q_executor_telemetry()) == {
        "python_to_q_bytes",
        "q_to_python_bytes",
        "resident_reuse_count",
    }
    assert {field.name for field in fields(QExecutionResult)} >= {
        "region_id",
        "output_df",
        "success",
        "resident_handle",
    }
    reset_q_executor_telemetry()


def test_q_backend_module_import_resolves_executor_symbols():
    from backend.q_backend.q_backend import QBackend

    assert QBackend.__name__ == "QBackend"


def test_unavailable_runtime_raises_typed_error_even_with_fallback_named():
    manager = MagicMock()
    manager.check_availability.return_value = MagicMock(
        status=QAvailabilityStatus.UNAVAILABLE,
        error_message="runtime absent",
    )
    executor = QExecutor(process_manager=manager, type_adapter=MagicMock())
    policy = QExecutionFallbackPolicy(
        allow_fallback=True,
        fallback_backend="pandas",
        fail_on_unavailable=True,
    )

    with pytest.raises(QProcessUnavailableError, match="Runtime fallback is disabled"):
        executor.execute_region(
            _plan("r1", ("base",), "out"),
            {"base": pd.DataFrame({"x": [1.0]})},
            fallback_policy=policy,
        )


class _QResult:
    nbytes = 8

    def __len__(self):
        return 1


class _RecordingQ:
    def __init__(self, *, fail_on_execute=None, fail_on_delete=False):
        self.calls = []
        self.fail_on_execute = fail_on_execute
        self.fail_on_delete = fail_on_delete
        self.result = _QResult()
        self._execution_failed = False

    def __setitem__(self, key, value):
        self.calls.append(("bind", key))

    def __call__(self, query):
        self.calls.append(("call", query))
        if isinstance(query, str) and query.startswith("delete "):
            if self.fail_on_delete:
                raise RuntimeError("cleanup failed")
            return None
        if self.fail_on_execute is not None and not self._execution_failed:
            self._execution_failed = True
            raise self.fail_on_execute
        return self.result


def _available_executor(connection, adapter=None):
    manager = MagicMock()
    manager.check_availability.return_value = MagicMock(
        status=QAvailabilityStatus.AVAILABLE,
        error_message=None,
    )
    manager.get_connection.return_value = connection
    adapter = adapter or MagicMock()
    adapter.pandas_to_q.return_value = object()
    adapter.q_to_pandas.return_value = pd.DataFrame({"value": [1.0]})
    return QExecutor(process_manager=manager, type_adapter=adapter)


def test_unbound_resident_handle_is_rejected_in_production():
    connection = _RecordingQ()
    executor = _available_executor(connection)

    with pytest.raises(QExecutionError, match="Unbound Q-resident handle rejected"):
        executor.execute_region(
            _plan("r1", ("base",), "out"),
            {
                "base": QResidentTableHandle(
                    table_name="base",
                    q_table_ref=object(),
                    row_count=1,
                    byte_size=1,
                    region_id="legacy",
                )
            },
        )


def test_baseexception_still_cleans_execution_workspace():
    connection = _RecordingQ(fail_on_execute=KeyboardInterrupt())
    executor = _available_executor(connection)

    with pytest.raises(KeyboardInterrupt):
        executor.execute_region(
            _plan("cancel", ("base",), "out"),
            {"base": pd.DataFrame({"x": [1.0]})},
            workspace_id="cancel-workspace",
            generation_id="cancel-generation",
        )

    deletes = [query for kind, query in connection.calls if kind == "call" and query.startswith("delete ")]
    assert any("qe_cancel_workspace_base" in query for query in deletes)
    assert any("qe_cancel_workspace_out" in query for query in deletes)


def test_cleanup_failure_does_not_replace_original_execution_error():
    connection = _RecordingQ(fail_on_execute=RuntimeError("execution failed"), fail_on_delete=True)
    executor = _available_executor(connection)

    with pytest.raises(QExecutionError, match="execution failed"):
        executor.execute_region(
            _plan("failure", ("base",), "out"),
            {"base": pd.DataFrame({"x": [1.0]})},
            workspace_id="failure-workspace",
            generation_id="failure-generation",
        )


def test_sequential_execution_rejects_stale_handle_after_cleanup():
    connection = _RecordingQ()
    executor = _available_executor(connection)
    handle_result = executor.execute_region(
        _plan("source", ("base",), "intermediate"),
        {"base": pd.DataFrame({"x": [1.0]})},
        return_resident_handle=True,
        workspace_id="stale-workspace",
        generation_id="stale-generation",
    )
    handle = handle_result.resident_handle
    assert handle is not None

    executor.execute_region(
        _plan("consumer", ("intermediate",), "final"),
        {"intermediate": handle},
        workspace_id="stale-workspace",
        generation_id="stale-generation",
    )

    with pytest.raises(QExecutionError, match="Stale Q-resident handle"):
        executor.execute_region(
            _plan("reuse", ("intermediate",), "after_cleanup"),
            {"intermediate": handle},
            workspace_id="stale-workspace",
            generation_id="stale-generation",
        )


def test_direct_success_cleans_physical_q_symbols():
    connection = _RecordingQ()
    executor = _available_executor(connection)

    result = executor.execute_region(
        _plan("direct", ("base",), "logical-output"),
        {"base": pd.DataFrame({"x": [1.0]})},
        workspace_id="physical-workspace",
        generation_id="physical-generation",
    )

    assert result.success
    deletes = [query for kind, query in connection.calls if kind == "call" and query.startswith("delete ")]
    assert {
        "delete qe_physical_workspace_base from `.",
        "delete qe_physical_workspace_logical_output from `.",
    } <= set(deletes)


def test_batch_final_success_cleans_all_physical_q_symbols():
    connection = _RecordingQ()
    executor = _available_executor(connection)
    plans = [
        _plan("left", ("base",), "left-out"),
        _plan("right", ("base",), "right-out"),
        _plan("diamond", ("base", "left-out", "right-out"), "final-out"),
    ]

    results = executor.execute_batch_regions(plans, {"base": pd.DataFrame({"x": [1.0]})})

    assert len(results) == 3
    deletes = [query for kind, query in connection.calls if kind == "call" and query.startswith("delete ")]
    for symbol in ("base", "left_out", "right_out", "final_out"):
        assert any(f"delete qe_" in query and f"_{symbol} from `." in query for query in deletes)


def test_batch_diamond_fan_in_executes_with_cleanup():
    connection = _RecordingQ()
    executor = _available_executor(connection)
    plans = [
        _plan("left", ("base",), "left-out"),
        _plan("right", ("base",), "right-out"),
        _plan("diamond", ("base", "left-out", "right-out"), "final-out"),
    ]

    results = executor.execute_batch_regions(plans, {"base": pd.DataFrame({"x": [1.0]})})

    assert all(result.success for result in results)
    execution_queries = [query for kind, query in connection.calls if kind == "call" and not query.startswith("delete ")]
    assert any("left_out" in query for query in execution_queries)
    assert any("right_out" in query for query in execution_queries)
    assert any("final_out" in query for query in execution_queries)
    manager = MagicMock()
    manager.check_availability.return_value = MagicMock(
        status=QAvailabilityStatus.AVAILABLE,
        error_message=None,
    )
    connection = MagicMock()
    connection.side_effect = RuntimeError("q exploded")
    manager.get_connection.return_value = connection
    adapter = MagicMock()
    adapter.pandas_to_q.return_value = MagicMock()
    executor = QExecutor(process_manager=manager, type_adapter=adapter)

    with pytest.raises(QExecutionError, match="q exploded"):
        executor.execute_region(
            _plan("r1", ("base",), "out"),
            {"base": pd.DataFrame({"x": [1.0]})},
        )

    adapter.q_to_pandas.assert_not_called()


def test_missing_declared_input_fails_before_runtime_call():
    manager = MagicMock()
    manager.check_availability.return_value = MagicMock(
        status=QAvailabilityStatus.AVAILABLE,
        error_message=None,
    )
    executor = QExecutor(process_manager=manager, type_adapter=MagicMock())

    with pytest.raises(QDataUnavailableError, match="right"):
        executor.execute_region(
            _plan("fanin", ("left", "right"), "out"),
            {"left": pd.DataFrame({"x": [1.0]})},
        )

    manager.get_connection.assert_not_called()


def test_batch_fan_in_preserves_base_and_all_resident_predecessors(monkeypatch):
    executor = QExecutor(process_manager=MagicMock(), type_adapter=MagicMock())
    base = pd.DataFrame({"x": [1.0]})
    seen_inputs: dict[str, set[str]] = {}

    def fake_execute(plan, input_data, **kwargs):
        seen_inputs[plan.region_id] = set(input_data)
        is_resident = kwargs["return_resident_handle"]
        handle = None
        if is_resident:
            from backend.q_backend.q_adapter import QResidentTableHandle

            handle = QResidentTableHandle(
                table_name=plan.output_table,
                q_table_ref=object(),
                row_count=1,
                byte_size=8,
                region_id=plan.region_id,
                connection_id=None,
            )
        return QExecutionResult(
            region_id=plan.region_id,
            output_df=pd.DataFrame() if is_resident else pd.DataFrame({"value": [1.0]}),
            execution_time_ms=1.0,
            rows_processed=1,
            success=True,
            resident_handle=handle,
        )

    monkeypatch.setattr(executor, "execute_region", fake_execute)
    plans = [
        _plan("a", ("base",), "a_out"),
        _plan("b", ("base",), "b_out"),
        _plan("c", ("base", "a_out", "b_out"), "final"),
    ]

    results = executor.execute_batch_regions(plans, {"base": base})

    assert len(results) == 3
    assert seen_inputs == {
        "a": {"base"},
        "b": {"base"},
        "c": {"base", "a_out", "b_out"},
    }


def test_invalid_output_mode_is_rejected_before_readiness_check():
    manager = MagicMock()
    executor = QExecutor(process_manager=manager, type_adapter=MagicMock())

    with pytest.raises(ValueError, match="materialize output or return"):
        executor.execute_region(
            _plan("r1", ("base",), "out"),
            {"base": pd.DataFrame({"x": [1.0]})},
            materialize_output=False,
            return_resident_handle=False,
        )
    manager.check_availability.assert_not_called()


def test_readiness_exception_is_typed():
    manager = MagicMock()
    manager.check_availability.side_effect = RuntimeError("probe failed")
    executor = QExecutor(process_manager=manager, type_adapter=MagicMock())

    with pytest.raises(QProcessUnavailableError, match="readiness check failed"):
        executor.execute_region(
            _plan("r1", ("base",), "out"),
            {"base": pd.DataFrame({"x": [1.0]})},
        )


def test_runtime_failure_never_soft_fails_from_policy_flags():
    manager = MagicMock()
    manager.check_availability.return_value = MagicMock(
        status=QAvailabilityStatus.AVAILABLE,
        error_message=None,
    )
    manager.get_connection.return_value.side_effect = RuntimeError("q exploded")
    executor = QExecutor(process_manager=manager, type_adapter=MagicMock())
    policy = QExecutionFallbackPolicy(
        allow_fallback=False,
        fallback_backend=None,
        fail_on_unavailable=False,
    )

    with pytest.raises(QExecutionError, match="q exploded"):
        executor.execute_region(
            _plan("r1", ("base",), "out"),
            {"base": pd.DataFrame({"x": [1.0]})},
            fallback_policy=policy,
        )


def test_two_executor_instances_serialize_shared_connection():
    manager = MagicMock()
    manager.check_availability.return_value = MagicMock(
        status=QAvailabilityStatus.AVAILABLE,
        error_message=None,
    )
    active = 0
    peak_active = 0
    state_lock = threading.Lock()

    class Connection:
        def __setitem__(self, key, value):
            pass

        def __call__(self, query):
            nonlocal active, peak_active
            if ":" not in query:
                return [1.0]
            with state_lock:
                active += 1
                peak_active = max(peak_active, active)
            time.sleep(0.02)
            with state_lock:
                active -= 1
            return None

    manager.get_connection.return_value = Connection()
    adapter = MagicMock()
    adapter.pandas_to_q.return_value = object()
    adapter.q_to_pandas.return_value = pd.DataFrame({"value": [1.0]})
    executors = [
        QExecutor(process_manager=manager, type_adapter=adapter),
        QExecutor(process_manager=manager, type_adapter=adapter),
    ]
    threads = [
        threading.Thread(
            target=executor.execute_region,
            args=(_plan(f"r{i}", ("base",), f"out{i}"), {"base": pd.DataFrame({"x": [1.0]})}),
        )
        for i, executor in enumerate(executors)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert peak_active == 1
