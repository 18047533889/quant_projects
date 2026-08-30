# -*- coding: utf-8 -*-
"""P2 split-brain: registry-free forced Q_KDB / CLICKHOUSE selection.

The runtime must have ONE authority for what physical backend an operator/plan
actually executes on.  When the planner selects ``PhysicalBackend.Q_KDB`` or
``PhysicalBackend.CLICKHOUSE_SQL`` in a registry-free context (no live q
runtime, no wired clickhouse executor), the executor must either execute on
that backend truthfully or raise a typed, fail-closed error — it must NEVER
silently run pandas/polars and return results as if the planned backend ran.

This file lives in ``tests/multibackend/`` (NOT ``tests/q_backend/``) and
covers:

  (a) ``_physical_backend_for_region`` raises ``PhysicalBackendNotRuntimeCapableError``
      for Q_KDB / CLICKHOUSE_SQL today (typed, not silent),
  (b) ``_assert_executor_identity`` rejects a plain duckdb-labeled executor when
      the plan routed the region to Q_KDB / CLICKHOUSE_SQL (identity mismatch),
  (c) the plan-vs-executor identity assert never fires for the four wired
      backends when the executor label matches the planned backend,
  (d) the batch-global planner guard rejects any plan carrying a non-capable
      backend even when the optimizer's candidate filter is bypassed.
"""
from __future__ import annotations

import pytest

from factor_engine.backend.operator_capability import BackendUnavailableError
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
    _assert_executor_identity,
    _physical_backend_for_region,
    assert_all_backends_runtime_capable,
    physical_backend_runtime_capable,
)


class _PandasBackend:
    """Minimal executor with a runtime_backend_label (like real executor classes)."""

    runtime_backend_label = "pandas_numpy"

    def execute(self, plan: object, ctx: object) -> str:  # pragma: no cover
        return "pandas-result"


class _PolarsBackend(_PandasBackend):
    runtime_backend_label = "polars_panel"


class _DuckSqlBackend:
    """Plain duckdb-labeled SQL executor (like DuckDBPushdownBackend/SqlBackend)."""

    runtime_backend_label = "duckdb_sql"


class _ClickHousePushdownBackend(_DuckSqlBackend):
    """A genuine ClickHousePushdownBackend-class executor (label clickhouse_sql).

    Mirror of the real ``ClickHousePushdownBackend`` which declares the concrete
    dialect identity.  The identity assert special-cases this class for
    CLICKHOUSE_SQL regions only.
    """

    runtime_backend_label = "clickhouse_sql"


class _HybridBackend:
    runtime_backend_label = "hybrid"

    def __init__(self) -> None:
        self._pandas = _PandasBackend()
        self._polars = _PolarsBackend()
        self._sql = _DuckSqlBackend()

    @classmethod
    def _as_test_hybrid(cls) -> type:
        """Real HybridBackend lives at factor_engine.backend.hybrid_backend;
        the resolver keys on the exact class name, so a test double must present
        the same name to exercise the hybrid branch."""

    def _long_backend(self) -> object:
        return _PandasBackend()


def _hybrid() -> _HybridBackend:
    """Build a HybridBackend test double whose class name matches the resolver key."""
    return type("HybridBackend", (_HybridBackend,), {"_class_name_hook": lambda self: None})()


def _region(backend: PhysicalBackend, *, region_id: str = "r") -> BackendRegion:
    return BackendRegion(
        region_id=region_id,
        backend=backend,
        representation=Representation.PANDAS_LONG,
        node_ids=("n1",),
        execution_axis=ExecutionAxis.GLOBAL_PANEL,
        estimated_rows=100,
        estimated_compute_ms=1.0,
        estimated_memory_bytes=8,
    )


def _plan(backend: PhysicalBackend, region_id: str = "r") -> PhysicalRegionPlan:
    return PhysicalRegionPlan(
        plan_id="p2-split-brain",
        regions=(_region(backend, region_id=region_id),),
        edges=(),
        topological_order=(region_id,),
        root_region_ids=(region_id,),
        total_compute_ms=1.0,
        total_transfer_ms=0.0,
        total_ttdc_ms=1.0,
        peak_memory_bytes=8,
        plan_hash="p2-split-brain",
        logical_node_count=1,
        backend_switch_count=0,
        native_fraction=1.0,
    )


def test_forced_q_kdb_resolution_raises_typed_error_never_pandas() -> None:
    """Force Q_KDB in a registry-free context: typed fail-closed, not silent pandas."""
    for configured in (_PandasBackend(), _DuckSqlBackend()):
        with pytest.raises(PhysicalBackendNotRuntimeCapableError) as exc:
            _physical_backend_for_region(configured, PhysicalBackend.Q_KDB)
        assert "not runtime-capable" in str(exc.value)
        assert type(exc.value) is not PhysicalPlanRequiredError


def test_forced_clickhouse_resolution_raises_typed_error_never_pandas() -> None:
    """Force CLICKHOUSE_SQL in a registry-free context: typed fail-closed, never pandas."""
    for configured in (_PandasBackend(), _DuckSqlBackend()):
        with pytest.raises(PhysicalBackendNotRuntimeCapableError) as exc:
            _physical_backend_for_region(configured, PhysicalBackend.CLICKHOUSE_SQL)
        assert "not runtime-capable" in str(exc.value)


def test_forced_q_kdb_hybrid_resolution_raises_typed_error() -> None:
    with pytest.raises(PhysicalBackendNotRuntimeCapableError):
        _physical_backend_for_region(_hybrid(), PhysicalBackend.Q_KDB)
    # A hybrid with a plain duckdb delegate must refuse a CLICKHOUSE_SQL region
    # (executing it on the duckdb SQL delegate WOULD be silent mis-routing).
    with pytest.raises(PhysicalBackendNotRuntimeCapableError):
        _physical_backend_for_region(_hybrid(), PhysicalBackend.CLICKHOUSE_SQL)


def test_forced_q_kdb_hybrid_resolution_routes_to_real_q_when_available(monkeypatch) -> None:
    """When a real q runtime IS provisioned, a Q_KDB hybrid region must route to
    the production QBackend executor (Backlog #60 real integration path) rather
    than fail or fall back to a non-q delegate."""
    hybrid = _hybrid()
    q_backend = object()

    import factor_engine.backend.q_backend.q_backend as q_backend_module
    import factor_engine.backend.q_backend.q_process_manager as q_pm_module

    monkeypatch.setattr(q_pm_module, "is_q_available", lambda: True)
    monkeypatch.setattr(
        q_backend_module, "QBackend",
        lambda **kwargs: q_backend,
    )
    result = _physical_backend_for_region(hybrid, PhysicalBackend.Q_KDB)
    assert result is q_backend


def test_forced_clickhouse_hybrid_with_clickhouse_delegate_is_truthful() -> None:
    """A hybrid whose sql delegate is a genuine ClickHousePushdownBackend may
    serve a CLICKHOUSE_SQL region — that is truthful execution, not fallback."""
    class _WithClickHouse(_HybridBackend):
        def __init__(self) -> None:
            super().__init__()
            self._sql = _ClickHousePushdownBackend()

    with_ch = type("HybridBackend", (_WithClickHouse,), {})()
    assert (
        _physical_backend_for_region(with_ch, PhysicalBackend.CLICKHOUSE_SQL)
        is with_ch._sql
    )


def test_identity_assert_rejects_duckdb_label_for_q_and_clickhouse_plan() -> None:
    """A duckdb-labeled executor must never execute a Q_KDB / CLICKHOUSE region."""
    for planned in (PhysicalBackend.Q_KDB, PhysicalBackend.CLICKHOUSE_SQL):
        with pytest.raises(PhysicalPlanRequiredError, match="identity mismatch"):
            _assert_executor_identity(_DuckSqlBackend(), planned, HybridBackend=False)


def test_identity_assert_rejects_mismatch_for_wired_backends() -> None:
    """The identity assert is not Q/CH-specific: any plan-vs-executor mismatch fails."""
    with pytest.raises(PhysicalPlanRequiredError, match="identity mismatch"):
        _assert_executor_identity(_DuckSqlBackend(), PhysicalBackend.POLARS_LONG, HybridBackend=False)
    with pytest.raises(PhysicalPlanRequiredError, match="identity mismatch"):
        _assert_executor_identity(_PandasBackend(), PhysicalBackend.DUCKDB_SQL, HybridBackend=False)


def test_identity_assert_passes_when_label_matches_plan() -> None:
    pandas_backend = _PandasBackend()
    assert (
        _assert_executor_identity(pandas_backend, PhysicalBackend.PANDAS_NUMPY, HybridBackend=False)
        is pandas_backend
    )
    duck = _DuckSqlBackend()
    assert (
        _assert_executor_identity(duck, PhysicalBackend.DUCKDB_SQL, HybridBackend=False)
        is duck
    )
    # An executor with NO label (research-only double) is left untouched.
    class _Bare:
        pass

    bare = _Bare()
    assert (
        _assert_executor_identity(bare, PhysicalBackend.PANDAS_NUMPY, HybridBackend=False)
        is bare
    )


def test_hybrid_resolution_applies_identity_assert_to_concrete_delegate() -> None:
    """HybridBackend decomposition must also pass the identity assert per delegate."""
    hybrid = _hybrid()
    assert (
        _physical_backend_for_region(hybrid, PhysicalBackend.PANDAS_NUMPY)
        is hybrid._pandas
    )
    assert (
        _physical_backend_for_region(hybrid, PhysicalBackend.POLARS_PANEL)
        is hybrid._polars
    )
    assert (
        _physical_backend_for_region(hybrid, PhysicalBackend.DUCKDB_SQL)
        is hybrid._sql
    )
    # A hybrid whose duckdb delegate has a mismatched label must fail the assert.
    class _MismatchedHybrid(_HybridBackend):
        def __init__(self) -> None:
            super().__init__()
            self._sql = _PandasBackend()

    with pytest.raises(PhysicalPlanRequiredError, match="identity mismatch"):
        _physical_backend_for_region(
            type("HybridBackend", (_MismatchedHybrid,), {})(),
            PhysicalBackend.DUCKDB_SQL,
        )

    # A hybrid whose _long_backend resolves to a non-PolarsLongBackend still
    # fails with the fixed-executor error before the identity assert.
    class _NoLongHybrid(_HybridBackend):
        def _long_backend(self) -> object:
            return _PandasBackend()

    with pytest.raises(PhysicalPlanRequiredError, match="no fixed PolarsLongBackend"):
        _physical_backend_for_region(
            type("HybridBackend", (_NoLongHybrid,), {})(),
            PhysicalBackend.POLARS_LONG,
        )

    # POLARS_LONG resolves to a real PolarsLongBackend delegate when present.
    class _PolarsLongBackend:
        runtime_backend_label = "polars_long"

    class _WithLong(_HybridBackend):
        def __init__(self) -> None:
            super().__init__()
            self._long = _PolarsLongBackend()

        def _long_backend(self) -> object:
            return self._long

    with_long = type("HybridBackend", (_WithLong,), {})()
    assert (
        _physical_backend_for_region(with_long, PhysicalBackend.POLARS_LONG)
        is with_long._long
    )

    # The alternative long shape (delegate exposing ``_polars_long``) is also
    # honored.
    class _LongHolder:
        def __init__(self) -> None:
            self._polars_long = _PolarsLongBackend()

    holder_obj = _LongHolder()

    class _WithLongHolder(_HybridBackend):
        def _long_backend(self) -> object:
            return holder_obj

    holder = type("HybridBackend", (_WithLongHolder,), {})()
    assert (
        _physical_backend_for_region(holder, PhysicalBackend.POLARS_LONG)
        is holder_obj._polars_long
    )


def test_clickhouse_pushdown_class_passes_clickhouse_identity_only() -> None:
    """A genuine ClickHousePushdownBackend may serve a CLICKHOUSE_SQL region;
    it must still be rejected for every OTHER region backend."""
    ch = _ClickHousePushdownBackend()
    assert (
        _assert_executor_identity(ch, PhysicalBackend.CLICKHOUSE_SQL, HybridBackend=False)
        is ch
    )
    for other in (
        PhysicalBackend.Q_KDB,
        PhysicalBackend.PANDAS_NUMPY,
        PhysicalBackend.POLARS_LONG,
        PhysicalBackend.DUCKDB_SQL,
    ):
        with pytest.raises(PhysicalPlanRequiredError, match="identity mismatch"):
            _assert_executor_identity(ch, other, HybridBackend=False)
    # The special case is label-scoped: a plain duckdb-labeled SqlBackend must
    # still be rejected for a CLICKHOUSE_SQL region (it is not a clickhouse
    # executor, so honoring the region on it WOULD be silent mis-routing).
    with pytest.raises(PhysicalPlanRequiredError, match="identity mismatch"):
        _assert_executor_identity(
            _DuckSqlBackend(), PhysicalBackend.CLICKHOUSE_SQL, HybridBackend=False
        )


def test_backend_unavailable_error_type_exists_for_fail_closed_gate() -> None:
    """BackendUnavailableError is the typed signal production callers catch."""
    assert issubclass(BackendUnavailableError, RuntimeError)


def test_planner_guard_rejects_clickhouse_plan() -> None:
    with pytest.raises(PhysicalBackendNotRuntimeCapableError, match="CLICKHOUSE_SQL"):
        assert_all_backends_runtime_capable(_plan(PhysicalBackend.CLICKHOUSE_SQL))
    with pytest.raises(PhysicalBackendNotRuntimeCapableError, match="Q_KDB"):
        assert_all_backends_runtime_capable(_plan(PhysicalBackend.Q_KDB))


def test_planner_guard_accepts_capable_plan() -> None:
    assert_all_backends_runtime_capable(_plan(PhysicalBackend.PANDAS_NUMPY))
    assert_all_backends_runtime_capable(_plan(PhysicalBackend.DUCKDB_SQL))


def test_capability_gate_stays_false_for_q_and_clickhouse() -> None:
    assert physical_backend_runtime_capable(PhysicalBackend.Q_KDB) is False
    assert physical_backend_runtime_capable(PhysicalBackend.CLICKHOUSE_SQL) is False
