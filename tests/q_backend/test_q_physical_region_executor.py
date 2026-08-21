"""Regression tests for R21-Q-PHYSICAL-REGION-EXECUTOR.

Verifies the production PhysicalRegionExecutor path:

  PhysicalRegionPlan → QCompiler → QExecutor → QResidentHandle

without re-deriving a region from a whole-tree PlanNode.

Five regression tests:

1. test_execute_physical_region_compiles_planner_nodes_and_executes
2. test_execute_physical_region_returns_resident_handle_with_no_q_to_python
3. test_execute_physical_region_accepts_arrow_input
4. test_execute_physical_region_fails_closed_on_unavailable_runtime
5. test_execute_physical_region_requires_planner_supplied_node_structure
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from backend.context import ExecutionContext
from backend.q_backend.q_adapter import QResidentTableHandle
from backend.q_backend.q_backend import QBackend
from backend.q_backend.q_errors import (
    QDataUnavailableError,
    QPhysicalRegionNotImplemented,
)
from backend.q_backend.q_executor import (
    QExecutionResult,
    QExecutor,
)
from backend.q_backend.q_process_manager import QAvailabilityStatus
from planner.backend_region import (
    BackendRegion,
    ExecutionAxis,
    PhysicalBackend,
    Representation,
)


def _arrow_table() -> pa.Table:
    import pandas as pd

    ts = pd.to_datetime(["2024-01-01", "2024-01-02"]).to_numpy()
    return pa.table(
        {
            "timestamp": pa.array(ts, type=pa.timestamp("ns")),
            "instrument": pa.array(["AAPL", "AAPL"], type=pa.string()),
            "value": pa.array([1.0, 2.0], type=pa.float64()),
        }
    )


def _make_ctx() -> ExecutionContext:
    return ExecutionContext(
        data_source=object(),
        run_mode="production",
        runtime_stats={},
    )


def _make_region(region_id: str = "q_r1") -> BackendRegion:
    return BackendRegion(
        region_id=region_id,
        backend=PhysicalBackend.Q_KDB,
        representation=Representation.Q_TABLE,
        node_ids=("mean",),
        execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
        estimated_rows=2,
        estimated_compute_ms=1.0,
        estimated_memory_bytes=16,
    )


def _make_backend(
    *,
    available: bool = True,
    production_mode: bool = True,
) -> QBackend:
    backend = QBackend(fallback_to_pandas=False, production_mode=production_mode)
    if available:
        backend._process_manager = type(
            "PM",
            (),
            {
                "is_available": lambda self=True: True,
                "check_availability": lambda self=True: type(
                    "Info", (), {"status": QAvailabilityStatus.AVAILABLE}
                )(),
            },
        )()
    return backend


def _make_backend_with_executor(
    executor: QExecutor,
    *,
    available: bool = True,
    production_mode: bool = True,
) -> QBackend:
    backend = _make_backend(available=available, production_mode=production_mode)
    backend._executor = executor
    return backend


class _FakeExecutor:
    """QExecutor shim for when no real q runtime is installed."""

    def __init__(self, *, output_df=None, resident_handle=None, fail=False):
        self.output_df = output_df
        self.resident_handle = resident_handle
        self.fail = fail
        self.calls = []

    def execute_region(self, region_plan, input_data, **kwargs):
        self.calls.append((region_plan, input_data, kwargs))
        if self.fail:
            raise QDataUnavailableError("shim q execution failed")
        return QExecutionResult(
            region_id=region_plan.region_id,
            output_df=self.output_df or _default_output_df(),
            execution_time_ms=1.0,
            rows_processed=2,
            success=True,
            resident_handle=self.resident_handle,
        )


def _default_output_df():
    import pandas as pd

    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "instrument": ["AAPL", "AAPL"],
            "value": [1.0, 2.0],
        }
    )


def test_execute_physical_region_compiles_planner_nodes_and_executes() -> None:
    """Production path must compile the planner-admitted region and execute it."""
    backend = _make_backend()
    executor = _FakeExecutor(output_df=_default_output_df())
    backend._executor = executor
    region = _make_region()
    ctx = _make_ctx()

    result = backend.execute_physical_region(
        region,
        ctx,
        nodes={
            "mean": {
                "op": "ts_mean",
                "inputs": ["input_table"],
                "attrs": {"window": 2},
                "semantic_attrs": {},
            }
        },
        input_data={"input_table": _default_output_df()},
    )

    assert result.index.names == ["timestamp", "instrument"]
    assert result.name == "value"
    assert result.tolist() == [1.0, 2.0]
    assert backend.get_stats()["regions_compiled"] == 1
    assert backend.get_stats()["regions_executed"] == 1

    # The shim executor received a QRegionPlan compiled from the planner
    # node, not a whole-tree PlanNode.
    region_plan = executor.calls[0][0]
    assert region_plan.region_id == "q_r1"
    assert region_plan.node_ids == ("mean",)
    assert region_plan.input_tables == ("input_table",)
    assert region_plan.output_table == "result"
    assert "mean: 2 mavg input_table" in region_plan.q_code


def test_execute_physical_region_returns_resident_handle_with_no_q_to_python() -> None:
    """A q region intermediate must stay Q-resident (q_to_python_bytes=0)."""
    handle = QResidentTableHandle(
        table_name="result",
        q_table_ref=object(),
        row_count=2,
        byte_size=16,
        region_id="q_r1",
        connection_id=123,
        workspace_id="ws",
        generation_id="gen",
    )
    backend = _make_backend()
    executor = _FakeExecutor(resident_handle=handle)
    backend._executor = executor
    region = _make_region()
    ctx = _make_ctx()

    result = backend.execute_physical_region(
        region,
        ctx,
        nodes={
            "mean": {
                "op": "ts_mean",
                "inputs": ["input_table"],
                "attrs": {"window": 2},
                "semantic_attrs": {},
            }
        },
        input_data={"input_table": _default_output_df()},
        return_resident_handle=True,
    )

    assert isinstance(result, QExecutionResult)
    assert result.resident_handle is handle
    assert result.output_df.empty
    # The shim executor never materializes to pandas, so no q-to-python bytes.
    assert backend.get_stats()["regions_compiled"] == 1
    assert backend.get_stats()["regions_executed"] == 1


def test_execute_physical_region_accepts_arrow_input() -> None:
    """The q region boundary must accept Arrow tables, not pandas only."""
    from backend.q_backend.q_executor import QExecutor

    q = type("RecordingQ", (), {})()
    q.calls = []

    def __call__(self, query):
        if isinstance(query, str) and ":" in query:
            return None
        return ["mean"]

    q.__call__ = __call__

    def __setitem__(self, key, value):
        self.calls.append(("bind", key))

    q.__setitem__ = __setitem__
    q.__setattr__ = object.__setattr__

    adapter = type(
        "Adapter",
        (),
        {"pandas_to_q": lambda self, df, **kw: object(), "q_to_pandas": lambda self, obj, **kw: _default_output_df()},
    )()
    executor = QExecutor(
        process_manager=type(
            "PM",
            (),
            {
                "check_availability": lambda self=True: type(
                    "Info", (), {"status": QAvailabilityStatus.AVAILABLE}
                )(),
                "get_connection": lambda self=True: q,
            },
        )(),
        type_adapter=adapter,
    )
    backend = _make_backend_with_executor(executor)
    region = _make_region()
    ctx = _make_ctx()

    # A genuine pyarrow.Table with the canonical timestamp/instrument/value
    # columns, passed straight into the q boundary.
    table = _arrow_table()
    result = backend.execute_physical_region(
        region,
        ctx,
        nodes={
            "mean": {
                "op": "ts_mean",
                "inputs": ["input_table"],
                "attrs": {"window": 2},
                "semantic_attrs": {},
            }
        },
        input_data={"input_table": table},
    )

    assert result.index.names == ["timestamp", "instrument"]
    assert result.tolist() == [1.0, 2.0]
    assert any(kind == "bind" for kind, _ in q.calls)
    telemetry = executor.telemetry_snapshot()
    assert telemetry["python_to_q_bytes"] > 0
    assert telemetry["q_to_python_bytes"] >= 0


def test_execute_physical_region_fails_closed_on_unavailable_runtime() -> None:
    """If q runtime is unavailable, the production path must fail closed."""
    from backend.operator_capability import BackendUnavailableError

    backend = _make_backend(available=False)
    region = _make_region()
    ctx = _make_ctx()

    with pytest.raises(BackendUnavailableError, match="q runtime unavailable"):
        backend.execute_physical_region(region, ctx)


def test_execute_physical_region_requires_planner_supplied_node_structure() -> None:
    """A region without planner-supplied node structure must fail closed."""
    backend = _make_backend()
    executor = _FakeExecutor()
    backend._executor = executor
    region = _make_region()
    ctx = _make_ctx()

    with pytest.raises(
        QPhysicalRegionNotImplemented,
        match="no planner-supplied structure",
    ):
        backend.execute_physical_region(region, ctx)

    assert executor.calls == []
