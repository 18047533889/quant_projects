"""Focused recovery tests for the q executor public ABI and runtime policy."""

from __future__ import annotations

from dataclasses import fields
import threading
import time
from unittest.mock import MagicMock

import pandas as pd
import pytest

from factor_engine.backend.q_backend.q_compiler import QRegionPlan
from factor_engine.backend.q_backend.q_adapter import QResidentTableHandle
from factor_engine.backend.q_backend.q_errors import (
    QDataUnavailableError,
    QExecutionError,
    QProcessUnavailableError,
    QUnavailableError,
)
from factor_engine.backend.q_backend.q_executor import (
    QExecutionFallbackPolicy,
    QExecutionResult,
    QExecutor,
    get_q_executor,
    get_q_executor_telemetry,
    reset_q_executor_telemetry,
)
from factor_engine.backend.q_backend.q_process_manager import QAvailabilityStatus


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
    from factor_engine.backend.q_backend.q_backend import QBackend

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

    # Backlog #60: an absent q runtime fails closed with the typed
    # QUnavailableError (a QProcessUnavailableError subclass), even when a
    # fallback backend is NAMED in the policy — a named alternative is not a
    # silent runtime substitution.
    with pytest.raises(QUnavailableError, match="fail-closed, no silent fallback"):
        executor.execute_region(
            _plan("r1", ("base",), "out"),
            {"base": pd.DataFrame({"x": [1.0]})},
            fallback_policy=policy,
        )



def test_unbound_resident_handle_is_rejected_in_production():
    manager = MagicMock()
    manager.check_availability.return_value = MagicMock(
        status=QAvailabilityStatus.AVAILABLE,
        error_message=None,
    )
    connection = MagicMock()
    manager.get_connection.return_value = connection
    executor = QExecutor(process_manager=manager, type_adapter=MagicMock())

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


def test_runtime_exception_propagates_as_q_execution_error():
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
            from factor_engine.backend.q_backend.q_adapter import QResidentTableHandle

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




class _DiamondQResult:
    nbytes = 8

    def __len__(self):
        return 1


class _DiamondQRecorder:
    def __init__(self):
        self.calls: list[str] = []
        self.result = _DiamondQResult()

    def __setitem__(self, key, value):
        self.calls.append(f"bind {key}")

    def __call__(self, query):
        self.calls.append(query)
        if isinstance(query, str) and query.startswith("delete "):
            return None
        return self.result


def test_batch_diamond_fan_in_routes_both_physical_predecessors_into_final_q_code():
    """The final q plan must consume both routed resident predecessor symbols."""
    connection = _DiamondQRecorder()
    manager = MagicMock()
    manager.check_availability.return_value = MagicMock(
        status=QAvailabilityStatus.AVAILABLE,
        error_message=None,
    )
    manager.get_connection.return_value = connection
    adapter = MagicMock()
    adapter.pandas_to_q.return_value = object()
    adapter.q_to_pandas.return_value = pd.DataFrame({"value": [3.0]})
    executor = QExecutor(process_manager=manager, type_adapter=adapter)
    plans = [
        _plan("left", ("base",), "left-out"),
        _plan("right", ("base",), "right-out"),
        QRegionPlan(
            region_id="diamond",
            node_ids=("diamond",),
            q_code="final-out: left-out + right-out + base",
            input_tables=("base", "left-out", "right-out"),
            output_table="final-out",
        ),
    ]

    results = executor.execute_batch_regions(
        plans,
        {"base": pd.DataFrame({"x": [1.0]})},
    )

    assert all(result.success for result in results)
    final_q_code = next(
        query
        for query in connection.calls
        if query.startswith("qe_") and "+" in query
    )
    assert "left_2dout" in final_q_code
    assert "right_2dout" in final_q_code
    assert "_base" in final_q_code


def test_compiler_intermediates_are_namespaced_and_cleaned_with_the_workspace():
    connection = _DiamondQRecorder()
    manager = MagicMock()
    manager.check_availability.return_value = MagicMock(
        status=QAvailabilityStatus.AVAILABLE,
        error_message=None,
    )
    manager.get_connection.return_value = connection
    adapter = MagicMock()
    adapter.pandas_to_q.return_value = object()
    adapter.q_to_pandas.return_value = pd.DataFrame({"value": [1.0]})
    executor = QExecutor(process_manager=manager, type_adapter=adapter)
    plan = QRegionPlan(
        region_id="nodes",
        node_ids=("first-node", "middle.node", "final-node"),
        q_code="first-node: 0!base; middle.node: first-node; final-node: middle.node",
        input_tables=("base",),
        output_table="final-node",
    )

    executor.execute_region(plan, {"base": pd.DataFrame({"x": [1.0]})})

    emitted = next(query for query in connection.calls if "middle_2enode:" in query)
    assert "first-node" not in emitted
    assert "middle.node" not in emitted
    assert "final-node" not in emitted
    deleted = {
        query.removeprefix("delete ").removesuffix(" from `.")
        for query in connection.calls
        if query.startswith("delete ")
    }
    compiler_symbols = {
        symbol
        for symbol in deleted
        if symbol.endswith(("first_2dnode", "middle_2enode", "final_2dnode"))
    }
    assert len(compiler_symbols) == 3


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
