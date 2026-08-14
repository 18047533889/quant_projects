"""Focused recovery tests for the q executor public ABI and runtime policy."""

from __future__ import annotations

from dataclasses import fields
from unittest.mock import MagicMock

import pandas as pd
import pytest

from backend.q_backend.q_compiler import QRegionPlan
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
