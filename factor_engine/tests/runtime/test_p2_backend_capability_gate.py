# -*- coding: utf-8 -*-
"""P2 split-brain closure: runtime capability gate for PhysicalBackend.

The ``PhysicalBackend`` enum declares PANDAS_NUMPY / POLARS_PANEL / POLARS_LONG /
DUCKDB_SQL / CLICKHOUSE_SQL / Q_KDB, but the runtime resolver
(``runtime.engine._physical_backend_for_region``) only wires an executor for the
first four.  A planner that routed a production region to CLICKHOUSE_SQL or Q_KDB
would produce a plan the runtime cannot execute (split-brain).

This test locks in the fail-closed contract:
  (a) ``physical_backend_runtime_capable`` is False for Q_KDB and CLICKHOUSE_SQL today,
  (b) it is True for the four wired backends,
  (c) routing a region to Q_KDB / CLICKHOUSE_SQL raises the honest
      ``PhysicalBackendNotRuntimeCapableError`` (not the generic error), and
  (d) the batch-global planner guard rejects any plan carrying a non-capable backend.
"""
from __future__ import annotations

import pytest

from factor_engine.planner.backend_region import (
    BackendRegion,
    ExecutionAxis,
    PhysicalBackend,
    PhysicalRegionPlan,
    Representation,
)
from factor_engine.runtime.engine import (
    PhysicalBackendNotRuntimeCapableError,
    PhysicalPlanRequiredError,
    _physical_backend_for_region,
    assert_all_backends_runtime_capable,
    physical_backend_runtime_capable,
)


def test_capability_gate_false_for_q_and_clickhouse_today():
    # Real executor classes exist (QBackend / ClickHousePushdownBackend) but are
    # NOT wired into the runtime resolver, so the honest answer is False.
    assert physical_backend_runtime_capable(PhysicalBackend.Q_KDB) is False
    assert physical_backend_runtime_capable(PhysicalBackend.CLICKHOUSE_SQL) is False


def test_capability_gate_true_for_the_four_wired_backends():
    assert physical_backend_runtime_capable(PhysicalBackend.PANDAS_NUMPY) is True
    assert physical_backend_runtime_capable(PhysicalBackend.POLARS_PANEL) is True
    assert physical_backend_runtime_capable(PhysicalBackend.POLARS_LONG) is True
    assert physical_backend_runtime_capable(PhysicalBackend.DUCKDB_SQL) is True


def test_capability_gate_accepts_plain_backend_strings():
    assert physical_backend_runtime_capable("duckdb_sql") is True
    assert physical_backend_runtime_capable("q_kdb") is False
    assert physical_backend_runtime_capable("clickhouse_sql") is False


def test_resolver_raises_honest_error_for_q_and_clickhouse():
    # Use a minimal backend whose class name matches a wired executor name for the
    # four capable backends, and must hit the not-capable branch for Q_KDB / CLICKHOUSE_SQL.
    class SqlBackend:
        runtime_backend_label = "duckdb_sql"

    # DuckDB still resolves fine (wired).
    assert _physical_backend_for_region(SqlBackend(), PhysicalBackend.DUCKDB_SQL) is not None

    for not_capable in (PhysicalBackend.Q_KDB, PhysicalBackend.CLICKHOUSE_SQL):
        with pytest.raises(PhysicalBackendNotRuntimeCapableError) as exc:
            _physical_backend_for_region(SqlBackend(), not_capable)
        assert "not runtime-capable" in str(exc.value)
        # It is a distinct type, not the plain generic error.
        assert type(exc.value) is not PhysicalPlanRequiredError


def test_hybrid_resolver_raises_honest_error_for_q_and_clickhouse():
    class _LongBackend:
        pass

    class _Hybrid:
        def __init__(self):
            self._pandas = object()
            self._polars = object()
            self._sql = object()

        def _long_backend(self):
            return _LongBackend()

    for not_capable in (PhysicalBackend.Q_KDB, PhysicalBackend.CLICKHOUSE_SQL):
        with pytest.raises(PhysicalBackendNotRuntimeCapableError) as exc:
            _physical_backend_for_region(_Hybrid(), not_capable)
        assert "not runtime-capable" in str(exc.value)


def _region(backend: PhysicalBackend) -> BackendRegion:
    return BackendRegion(
        region_id=f"r-{backend.value}",
        backend=backend,
        representation=infer_representation_for(backend),
        node_ids=("n1",),
        execution_axis=ExecutionAxis.GLOBAL_PANEL,
        estimated_rows=100,
        estimated_compute_ms=1.0,
        estimated_memory_bytes=8,
    )


def infer_representation_for(backend: PhysicalBackend) -> Representation:
    return {
        PhysicalBackend.PANDAS_NUMPY: Representation.PANDAS_LONG,
        PhysicalBackend.POLARS_PANEL: Representation.POLARS_LONG,
        PhysicalBackend.POLARS_LONG: Representation.POLARS_LAZY_LONG,
        PhysicalBackend.DUCKDB_SQL: Representation.DUCKDB_RELATION,
        PhysicalBackend.CLICKHOUSE_SQL: Representation.DUCKDB_RELATION,
        PhysicalBackend.Q_KDB: Representation.Q_TABLE,
    }[backend]


def _plan(*regions: BackendRegion) -> PhysicalRegionPlan:
    return PhysicalRegionPlan(
        plan_id="p2-gate",
        regions=regions,
        edges=(),
        topological_order=tuple(r.region_id for r in regions),
        root_region_ids=(regions[-1].region_id,),
        total_compute_ms=1.0,
        total_transfer_ms=0.0,
        total_ttdc_ms=1.0,
        peak_memory_bytes=8,
        plan_hash="p2-gate",
        logical_node_count=sum(len(r.node_ids) for r in regions),
        backend_switch_count=0,
        native_fraction=1.0,
    )


def test_planner_guard_rejects_non_capable_backend():
    plan = _plan(
        _region(PhysicalBackend.PANDAS_NUMPY),
        _region(PhysicalBackend.Q_KDB),
    )
    with pytest.raises(PhysicalBackendNotRuntimeCapableError) as exc:
        assert_all_backends_runtime_capable(plan)
    assert "Q_KDB" in str(exc.value) or "q_kdb" in str(exc.value)


def test_planner_guard_accepts_capable_backends():
    plan = _plan(
        _region(PhysicalBackend.PANDAS_NUMPY),
        _region(PhysicalBackend.DUCKDB_SQL),
    )
    assert_all_backends_runtime_capable(plan)  # no raise
