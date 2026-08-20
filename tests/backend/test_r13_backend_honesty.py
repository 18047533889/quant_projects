# -*- coding: utf-8
"""R13 backend-honesty audit round.

Covers:

* P0-63  backend-kind taxonomy — a pandas-delegating polars UDF is never
         reported as a native polars backend;
* P1-64  gap-coverage registration is a labeled delegation layer (topology
         honesty: cleanup removes native bridges; the UDF layer is explicit);
* P1-65  plan-cost router applies a conversion penalty to delegate polars slots
         so a delegate is never preferred over native pandas;
* P1-66  lazy execution must not mutate the data source;
* P1-67  ``execute_lazy`` must not mutate backend shared state;
* P2-68  explicit ``operator_backend="pandas_numpy"`` is respected;
* P0-69  SQL pushdown distinguishes ALL / EMPTY / LIST instrument filters;
* P2-83  gap-coverage count equals real registrations;
* P1-84  capability-quality labels (delegate reported as delegate);
* P1-62  ``with_scope`` memory accounting always equals resident bytes.
"""
from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("polars")
import numpy as np

from planner.logical_plan import PlanNode
from runtime.perf_config import PerfConfig


@pytest.fixture(scope="module")
def loaded():
    from cleaned_operators import load_all

    load_all()
    yield


# ---------------------------------------------------------------------------
# P0-63 / P1-84 backend-kind taxonomy
# ---------------------------------------------------------------------------


def test_polars_backend_kind_classifies_gap_coverage_as_delegate(loaded):
    from backend.polars_backend_kind import (
        PolarsImplementationKind,
        canonical_polars_is_delegate,
        canonical_polars_kind,
    )
    from cleaned_operators.registry import OperatorRegistry

    # Every polars slot registered by the gap-coverage UDF layer (source
    # ``polars_udf``) must classify as a pandas delegate, never native.
    delegate = []
    for canonical in OperatorRegistry.list_canonical():
        if "polars" not in OperatorRegistry.backends_for(canonical):
            continue
        meta = (
            (OperatorRegistry._catalog.get(canonical, {}) or {}).get("backend_meta") or {}
        ).get("polars", {}) or {}
        if str(meta.get("source") or "") == "polars_udf":
            assert canonical_polars_kind(canonical) == PolarsImplementationKind.POLARS_UDF_PANDAS_DELEGATE
            assert canonical_polars_is_delegate(canonical)
            delegate.append(canonical)
    assert len(delegate) > 0


def test_polars_backend_kind_native_not_delegate(loaded):
    from backend.polars_backend_kind import canonical_polars_is_delegate
    from cleaned_operators.registry import OperatorRegistry

    # A genuinely native canonical (ts_mean has a real polars kernel) must NOT be
    # classified as a delegate.
    assert not canonical_polars_is_delegate("ts_mean")


def test_capability_quality_labels_delegate(loaded):
    from backend.polars_backend_kind import capability_quality

    assert capability_quality("ts_mean", "pandas_numpy") == "pandas_native"
    assert capability_quality("ts_mean", "duckdb_sql") == "duckdb_native_sql"
    assert capability_quality("ts_mean", "clickhouse_sql") == "clickhouse_native_sql"
    # A delegate-only canonical reports delegate, never "native".
    from backend.polars_backend_kind import canonical_polars_is_delegate
    from cleaned_operators.registry import OperatorRegistry

    delegates = [
        c
        for c in OperatorRegistry.list_canonical()
        if "polars" in OperatorRegistry.backends_for(c) and canonical_polars_is_delegate(c)
    ]
    assert delegates
    assert capability_quality(delegates[0], "polars") == "polars_pandas_delegate"


# ---------------------------------------------------------------------------
# P1-64 gap-coverage topology honesty
# ---------------------------------------------------------------------------


def test_gap_coverage_registers_labeled_delegation_layer(loaded):
    from cleaned_operators.polars_gap_coverage import register_polars_gap_coverage
    from cleaned_operators.registry import OperatorRegistry

    n = register_polars_gap_coverage()
    # Idempotent: re-running adds nothing new (all polars slots already exist).
    assert n == 0
    # Every polars slot that came from the gap-coverage layer carries the
    # explicit ``polars_udf`` source marker (a labeled delegation layer).
    gap = 0
    for canonical in OperatorRegistry.list_canonical():
        if "polars" not in OperatorRegistry.backends_for(canonical):
            continue
        meta = (
            (OperatorRegistry._catalog.get(canonical, {}) or {}).get("backend_meta") or {}
        ).get("polars", {}) or {}
        if str(meta.get("source") or "") == "polars_udf":
            gap += 1
    assert gap > 0


# ---------------------------------------------------------------------------
# P1-65 router cost must know delegate conversions
# ---------------------------------------------------------------------------


def test_router_cost_delegate_never_beats_pandas(loaded):
    from backend.polars_backend_kind import canonical_polars_is_delegate
    from backend.plan_cost_router import _cost
    from cleaned_operators.registry import OperatorRegistry

    delegates = [
        c
        for c in OperatorRegistry.list_canonical()
        if "pandas_numpy" in OperatorRegistry.backends_for(c)
        and "polars" in OperatorRegistry.backends_for(c)
        and canonical_polars_is_delegate(c)
    ]
    assert delegates
    rows = 500_000
    for canonical in delegates:
        pandas_cost = _cost(canonical, "pandas_numpy", rows)
        polars_cost = _cost(
            canonical,
            "polars_panel",
            rows,
            delegate_polars=canonical_polars_is_delegate(canonical),
        )
        assert polars_cost > pandas_cost, (
            f"{canonical}: delegate polars ({polars_cost:.3f}) must not beat "
            f"native pandas ({pandas_cost:.3f})"
        )


def test_router_ranks_pandas_over_delegate_plan(loaded):
    from backend.plan_cost_router import choose_plan_route
    from backend.polars_backend_kind import canonical_polars_is_delegate
    from backend.context import ExecutionContext
    from cleaned_operators.registry import OperatorRegistry
    from tests.helpers import InMemorySeriesSource

    delegates = [
        c
        for c in OperatorRegistry.list_canonical()
        if "pandas_numpy" in OperatorRegistry.backends_for(c)
        and "polars" in OperatorRegistry.backends_for(c)
        and canonical_polars_is_delegate(c)
    ]
    assert delegates
    canonical = delegates[0]

    col = PlanNode(op="column", inputs=[], attrs={"name": "close"})
    plan = PlanNode(op=canonical, inputs=[col], attrs={})
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=5), ["A"]],
        names=["timestamp", "instrument"],
    )
    source = InMemorySeriesSource(data={"close": pd.Series([1.0] * len(idx), index=idx)})
    ctx = ExecutionContext(
        data_source=source,
        run_mode="research",
        perf=PerfConfig(operator_backend="auto"),
    )
    route = choose_plan_route(plan, ctx)
    # A plan whose ONLY polars slot is a delegate must route to native pandas.
    assert route.backend == "pandas_numpy", route


# ---------------------------------------------------------------------------
# P1-66 / P1-67 lazy execution state
# ---------------------------------------------------------------------------


def test_execute_lazy_does_not_mutate_backend(loaded):
    from unittest.mock import MagicMock, patch

    from backend.polars_backend import PolarsBackend
    from backend.pandas_backend import PandasBackend
    from backend.context import ExecutionContext

    backend = PolarsBackend(use_lazy=False)
    src = MagicMock()
    src.enable_lazy_scan = MagicMock()
    ctx = ExecutionContext(data_source=src, panel_cache={})
    with patch.object(PandasBackend, "_eval", return_value=pd.Series([1.0])):
        backend.execute_lazy(MagicMock(), ctx)
    assert backend.use_lazy is False, "execute_lazy must not mutate _use_lazy"
    src.enable_lazy_scan.assert_called_once_with(True)


class _FakeLazySource:
    """A data source exposing the lazy-scan mutation surface."""

    def __init__(self):
        self._lazy_scan = False
        self.read_auto = False

    def enable_lazy_scan(self, enabled: bool = True):
        self._lazy_scan = bool(enabled)
        if enabled:
            self.read_auto = True


def test_lazy_apply_restores_source(loaded):
    from backend.polars_backend import PolarsBackend

    src = _FakeLazySource()
    restore: list = []
    PolarsBackend._apply_lazy_scan(src, restore=restore)
    # During the run lazy read is active...
    assert src._lazy_scan is True and src.read_auto is True
    for fn in restore:
        fn()
    # ...and afterwards the pre-run eager semantics are restored.
    assert src._lazy_scan is False and src.read_auto is False


def test_lazy_then_normal_same_backend_identical_read_semantics(loaded):
    from unittest.mock import MagicMock, patch

    from backend.polars_backend import PolarsBackend
    from backend.pandas_backend import PandasBackend
    from backend.context import ExecutionContext

    backend = PolarsBackend(use_lazy=False)
    src = _FakeLazySource()
    ctx = ExecutionContext(data_source=src, panel_cache={})
    ds = ctx.data_source  # the ExecutionContext-wrapped source the engine reads

    with patch.object(PandasBackend, "_eval", return_value=pd.Series([1.0])):
        backend.execute_lazy(MagicMock(), ctx)
    # After the lazy run the OBSERVABLE read semantics of the source the engine
    # actually reads are restored to eager.
    assert ds.read_auto is False and ds._lazy_scan is False

    # A subsequent NORMAL run on the SAME source/backend reads eagerly, exactly
    # like a fresh backend.
    with patch.object(PandasBackend, "_eval", return_value=pd.Series([1.0])):
        backend.execute(MagicMock(), ctx)
    assert ds.read_auto is False and ds._lazy_scan is False


# ---------------------------------------------------------------------------
# P2-68 explicit operator_backend respected
# ---------------------------------------------------------------------------


def test_explicit_pandas_numpy_operator_backend_respected(loaded):
    from backend.polars_backend import PolarsBackend
    from backend.context import ExecutionContext
    from backend.cleaned_bridge import _operator_backend_preference

    backend = PolarsBackend()
    ctx = ExecutionContext(
        data_source=None,  # type: ignore[arg-type]
        perf=PerfConfig(operator_backend="pandas_numpy"),
        panel_cache={},
    )
    ctx = backend._with_polars_perf(ctx)
    # The explicit pandas_numpy preference must NOT be downgraded to auto.
    assert ctx.perf.operator_backend == "pandas_numpy"
    # And the preference mechanism used by operator resolution reads it verbatim,
    # so a debug/parity run can actually force pandas.
    assert _operator_backend_preference(ctx) == "pandas_numpy"


def test_auto_operator_backend_stays_auto(loaded):
    from backend.polars_backend import PolarsBackend
    from backend.context import ExecutionContext

    backend = PolarsBackend()
    ctx = ExecutionContext(
        data_source=None,  # type: ignore[arg-type]
        perf=PerfConfig(operator_backend="auto"),
        panel_cache={},
    )
    ctx = backend._with_polars_perf(ctx)
    assert ctx.perf.operator_backend == "auto"


# ---------------------------------------------------------------------------
# P0-69 SQL pushdown EMPTY vs ALL universe
# ---------------------------------------------------------------------------


def test_sql_pushdown_filter_empty_vs_all(loaded):
    from backend.sql_pushdown.emitter import (
        InstrumentFilterKind,
        SqlDialect,
        SqlPushdownFilter,
        _build_filter_clause,
    )

    filt_none = SqlPushdownFilter(
        time_column="ts",
        instrument_column="inst",
        instruments=(),
        instrument_filter_kind=InstrumentFilterKind.ALL,
    )
    assert _build_filter_clause(filt_none, dialect=SqlDialect.DUCKDB) == ""

    filt_empty = SqlPushdownFilter(
        time_column="ts",
        instrument_column="inst",
        instruments=(),
        instrument_filter_kind=InstrumentFilterKind.EMPTY,
    )
    assert "WHERE FALSE" in _build_filter_clause(filt_empty, dialect=SqlDialect.DUCKDB)

    filt_list = SqlPushdownFilter(
        time_column="ts",
        instrument_column="inst",
        instruments=("000001.SZ",),
        instrument_filter_kind=InstrumentFilterKind.LIST,
    )
    clause = _build_filter_clause(filt_list, dialect=SqlDialect.DUCKDB)
    assert "'000001.SZ'" in clause and "IN" in clause


def test_instrument_filter_kind_mapping(loaded):
    from backend.sql_pushdown.executor import _instrument_filter_kind
    from backend.sql_pushdown.emitter import InstrumentFilterKind

    assert _instrument_filter_kind(None) == (InstrumentFilterKind.ALL, ())
    assert _instrument_filter_kind([]) == (InstrumentFilterKind.EMPTY, ())
    assert _instrument_filter_kind(()) == (InstrumentFilterKind.EMPTY, ())
    kind, insts = _instrument_filter_kind(["000001.SZ"])
    assert kind == InstrumentFilterKind.LIST and insts == ("000001.SZ",)


# ---------------------------------------------------------------------------
# P2-83 register_polars_udf success counting
# ---------------------------------------------------------------------------


def test_gap_coverage_count_equals_real_registrations(loaded, monkeypatch):
    from unittest.mock import patch

    from cleaned_operators.polars_gap_coverage import register_polars_gap_coverage
    from cleaned_operators.registry import OperatorRegistry

    # The registry is frozen after load_all(), so verify the counting logic with
    # mocks: ``register_polars_udf`` only counts when a polars slot actually
    # appears in the registry afterwards.
    canonicals = ["alpha_op", "beta_op", "gamma_op", "delta_op"]
    state = {
        "alpha_op": ["pandas_numpy"],                # eligible
        "beta_op": ["pandas_numpy"],                 # eligible
        "gamma_op": ["pandas_numpy"],                # eligible
        "delta_op": ["pandas_numpy", "polars"],      # already has polars
    }
    registered: set[str] = set()

    monkeypatch.setattr(OperatorRegistry, "list_canonical", staticmethod(lambda: list(canonicals)))
    monkeypatch.setattr(
        OperatorRegistry, "backends_for", staticmethod(lambda c: list(state.get(c, [])))
    )

    # Simulate a real registration: the UDF adds a polars slot.
    def _real_udf(canonical):
        state[canonical] = list(state.get(canonical, [])) + ["polars"]
        registered.add(canonical)

    with patch("cleaned_operators.polars_gap_coverage.register_polars_udf", side_effect=_real_udf):
        n = register_polars_gap_coverage()
    assert n == 3, n  # alpha/beta/gamma registered; delta already had polars
    assert registered == {"alpha_op", "beta_op", "gamma_op"}

    # Idempotent: after real registration, re-running registers nothing new.
    with patch("cleaned_operators.polars_gap_coverage.register_polars_udf", side_effect=_real_udf):
        assert register_polars_gap_coverage() == 0

    # Simulate a polars-unavailable environment: ``register_polars_udf`` no-ops
    # (no polars slot appears).  The count must be 0 — no blind increment.
    state = {c: ["pandas_numpy"] for c in canonicals}
    registered.clear()
    monkeypatch.setattr(
        OperatorRegistry, "backends_for", staticmethod(lambda c: list(state.get(c, [])))
    )

    def _noop_udf(canonical):
        pass  # polars runtime missing -> nothing registered

    with patch("cleaned_operators.polars_gap_coverage.register_polars_udf", side_effect=_noop_udf):
        assert register_polars_gap_coverage() == 0


# ---------------------------------------------------------------------------
# P1-62 with_scope memory accounting
# ---------------------------------------------------------------------------


def test_with_scope_memory_accounting_matches_resident_bytes(loaded):
    from runtime.resource_governor import (
        reset_global_governor,
        global_memory_governor,
        estimate_object_bytes,
    )
    from storage.cache import CacheManager

    reset_global_governor()
    gov = global_memory_governor()
    layer = "test_r13_p162"
    cache = CacheManager(layer_name=layer)
    big = pd.DataFrame({"a": np.arange(200, dtype=float)})
    size = estimate_object_bytes(big)
    assert size > 0

    def accounting():
        return gov._usage.get(layer, 0)

    # set through the parent reserves + accounts symmetrically.
    cache.set("k0", big)
    assert accounting() == cache._bytes

    # scopes share the same backing store: every scope's byte counter is in
    # lockstep with the governor accounting.
    scope_a = cache.with_scope("a")
    scope_b = cache.with_scope("b")
    assert accounting() == scope_a._bytes == scope_b._bytes == cache._bytes

    scope_a.set("k1", big)
    scope_b.set("k2", big)
    assert accounting() == cache._bytes == scope_a._bytes == scope_b._bytes
    assert cache._bytes == size * 3

    # replace an existing key: releases old accounting, reserves the new.
    scope_a.set("k1", big)
    assert accounting() == cache._bytes == scope_a._bytes == scope_b._bytes

    # repeated set/clear through ANY scope never drifts the governor below the
    # true resident byte counter.
    scope_b.set("k3", big)
    cache.set("k4", big)
    assert accounting() == cache._bytes == scope_a._bytes == scope_b._bytes

    # budget-driven eviction inside set() also keeps accounting symmetric.
    tiny_layer = "test_r13_p162_tiny"
    tiny = CacheManager(layer_name=tiny_layer, budget_bytes=size * 2)
    tiny.set("t1", big)
    tiny.set("t2", big)
    tiny.set("t3", big)  # evicts t1 within set()
    assert gov._usage.get(tiny_layer, 0) == tiny._bytes
    assert tiny._bytes <= size * 2

    scope_a.clear_memory()
    assert accounting() == 0 == cache._bytes == scope_a._bytes == scope_b._bytes
    reset_global_governor()
