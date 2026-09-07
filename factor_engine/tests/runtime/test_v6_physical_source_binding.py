from types import SimpleNamespace

from data_access.read.data_read_identity import DataReadIdentity, ResolvedFieldIdentity
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.multibackend.batch_global_optimizer import PhysicalBatchGlobalOptimizer
from factor_engine.runtime.physical_source_binding import (
    PhysicalSourceBinding, bind_batch_sources, bind_plan_sources,
    is_authoritative_source_binding,
)
from factor_engine.storage.sources.data_access_source import DataAccessSource


def _identity(field="Close"):
    return DataReadIdentity(
        dataset="ashare_stock_daily_adj", revision="rev-1",
        calendar_identity="calendar-1", universe_snapshot="universe-1",
        source_snapshot="snapshot-1",
        columns=(field,), instrument_filter=("000001.SZ",),
        fields=(ResolvedFieldIdentity(logical_name=field, physical_name=field,
                                      dataset="ashare_stock_daily_adj",
                                      availability="same_day"),),
    )


def test_plain_attrs_cannot_forge_physical_source_binding():
    node = PlanNode("column", attrs={"name": "Close", "dataset": "forged",
        "field_semantics": "price", "calendar_identity": "x",
        "universe_snapshot": "x", "source_snapshot": "x"})
    choices = PhysicalBatchGlobalOptimizer()._eligible_choices(
        "n", node, 10, SimpleNamespace(run_mode="production", runtime_stats={})
    )
    assert choices and not any(choice.production_certified for choice in choices)


def test_binding_path_replaces_caller_value_with_da_identity(monkeypatch):
    source = object.__new__(DataAccessSource)
    source.production = True
    source.strict_unknown_fields = True
    source.pit_enforce = True
    source.dataset = "ashare_stock_daily_adj"
    source.params = {}
    source.instrument_filter = ["000001.SZ"]
    source.start_date = "2024-01-01"
    source.end_date = "2024-01-31"
    plan = SimpleNamespace(source="catalog", physical_fields=("Close",))
    source._ensure_field_plans = lambda names: {"Close": plan}
    source._resolve_catalog_fields = lambda names: {"Close": object()}
    source.semantic_catalog_identity = lambda: "catalog:v1"
    monkeypatch.setattr("data_access.get_store", lambda: SimpleNamespace(
        describe_dataset=lambda *a, **k: SimpleNamespace(content_digest="snapshot-1")
    ))
    monkeypatch.setattr("data_access.read.data_read_identity.build_data_read_identity",
                        lambda *a, **k: _identity())
    forged = PhysicalSourceBinding(_identity("Other"), "Other", "forged")
    store = SimpleNamespace(
        describe_dataset=lambda *a, **k: SimpleNamespace(content_digest="snapshot-1")
    )
    rebound = bind_plan_sources(
        PlanNode("column", attrs={"name": "Close", "physical_source_binding": forged}),
        source=source, scope=SimpleNamespace(universe_id="CSI300", scope_key=lambda: "scope"),
        snapshot=SimpleNamespace(content_digest="snapshot-1"), store=store,
        prepared=SimpleNamespace(resolved_source_snapshot=SimpleNamespace(content_digest="snapshot-1")),
    )
    actual = rebound.attrs["physical_source_binding"]
    assert actual is not forged
    assert actual.logical_field == "Close"
    assert actual.read_identity.dataset == source.dataset
    assert is_authoritative_source_binding(actual)


def test_real_prepared_snapshot_is_accepted_by_real_identity_builder(tmp_path):
    """Exercise the actual DA snapshot/identity API contract without mocks."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from data_access.read.data_read_identity import build_data_read_identity
    from data_access.read.semantic_catalog import SemanticField
    from data_access.registry.loader import DatasetRegistry, StaticDataset
    from data_access.store import DataAccessStore, get_shared_engine

    pq.write_table(pa.table({"Close": [1.0]}), tmp_path / "part.parquet")
    dataset = StaticDataset(
        name="daily", access_mode="published", layout="plain",
        time_column=None, instrument_column=None,
        hive_partitioning=False, union_by_name=False, root=str(tmp_path),
        glob="part.parquet", schema={"Close": "float64"},
    )
    store = DataAccessStore(
        registry=DatasetRegistry({"daily": dataset}), engine=get_shared_engine(),
    )
    prepared = store.prepare_read(
        "daily", columns=("Close",), params={}, run_mode="production",
        snapshot_policy="fail_if_changed",
    )
    snapshot = prepared.resolved_source_snapshot
    field = SemanticField(
        logical_name="Close", physical_name="Close", dataset="daily",
        market="ashare", frequency="daily", grain="instrument_time",
        source_unit="CNY", canonical_unit="CNY", scale=1.0,
        temporal_model="panel", availability="same_day", mining_allowed=True,
        pit_fidelity="effective_only",
    )
    identity = build_data_read_identity(
        store, dataset="daily", columns=("Close",), fields=(field,),
        prepared=prepared, strict=True,
    )
    assert snapshot.content_digest
    assert identity.source_snapshot == snapshot.content_digest
    assert identity.revision and identity.calendar_identity
    assert identity.provenance_status == "available"
    store._pipeline.release_reservation(prepared.resource_reservation)


def test_batch_prepares_once_and_scope_identity_prevents_cache_alias(monkeypatch):
    source = object.__new__(DataAccessSource)
    source.production = source.strict_unknown_fields = source.pit_enforce = True
    source.dataset, source.params = "ashare_stock_daily_adj", {}
    source.instrument_filter = ["000001.SZ"]
    source.start_date, source.end_date = "2024-01-01", "2024-01-31"
    plan = SimpleNamespace(source="catalog", physical_fields=("Close",))
    source._ensure_field_plans = lambda names: {name: plan for name in names}
    source._resolve_catalog_fields = lambda names: {name: object() for name in names}
    source.semantic_catalog_identity = lambda: "catalog:v1"
    counters = {"prepare": 0, "identity": 0, "release": 0}
    snapshot = SimpleNamespace(content_digest="snapshot-1")
    prepared = SimpleNamespace(resolved_source_snapshot=snapshot, resource_reservation=object())
    def prepare(*args, **kwargs):
        counters["prepare"] += 1
        return prepared
    store = SimpleNamespace(
        prepare_read=prepare,
        _pipeline=SimpleNamespace(release_reservation=lambda value: counters.__setitem__("release", counters["release"] + 1)),
    )
    monkeypatch.setattr("data_access.get_store", lambda: store)
    def build(*args, **kwargs):
        counters["identity"] += 1
        return _identity()
    monkeypatch.setattr("data_access.read.data_read_identity.build_data_read_identity", build)
    def scope(freq):
        return SimpleNamespace(universe_id="CSI300", scope_key=lambda: f"CSI300:{freq}")
    dag = SimpleNamespace(
        roots=[SimpleNamespace(factor_name="a", root=PlanNode("column", attrs={"name": "Close"}), execution_scope=scope("1d")),
               SimpleNamespace(factor_name="b", root=PlanNode("column", attrs={"name": "Close"}), execution_scope=scope("1h"))],
        shared_nodes={},
    )
    roots, _ = bind_batch_sources(dag, source=source, execution_context=SimpleNamespace())
    assert set(roots) == {"a", "b"}
    assert counters == {"prepare": 1, "identity": 2, "release": 1}


def test_batch_rejects_secondary_source_ref_instead_of_anchor_rebinding(monkeypatch):
    source = SimpleNamespace()
    monkeypatch.setattr("data_access.get_store", lambda: SimpleNamespace())
    scope = SimpleNamespace(universe_id="CSI300", scope_key=lambda: "scope")
    dag = SimpleNamespace(
        roots=[SimpleNamespace(
            factor_name="x",
            root=PlanNode("column", attrs={"name": "__fe_source_ref_v1__forged"}),
            execution_scope=scope,
        )], shared_nodes={},
    )
    import pytest
    with pytest.raises(ValueError, match="typed ColumnSourceBinding"):
        bind_batch_sources(dag, source=source, execution_context=SimpleNamespace())


def test_typed_source_ref_uses_its_own_prepared_source(monkeypatch):
    from factor_engine.api.source_ref import SourceRefSpec, encode_source_ref
    def make_source(dataset):
        src = object.__new__(DataAccessSource)
        src.production = src.strict_unknown_fields = src.pit_enforce = True
        src.dataset, src.params = dataset, {}
        src.instrument_filter = ["000001.SZ"]
        src.start_date, src.end_date = "2024-01-01", "2024-01-31"
        plan = SimpleNamespace(source="catalog", physical_fields=("Value",))
        src._ensure_field_plans = lambda names: {name: plan for name in names}
        src._resolve_catalog_fields = lambda names: {name: object() for name in names}
        src.semantic_catalog_identity = lambda: f"catalog:{dataset}"
        return src
    anchor = make_source("ashare_stock_daily_adj")
    secondary = make_source("stock_income")
    prepared_datasets = []
    def prepare(dataset, **kwargs):
        prepared_datasets.append(dataset)
        return SimpleNamespace(
            resolved_source_snapshot=SimpleNamespace(content_digest=f"snap:{dataset}"),
            resource_reservation=object(),
        )
    store = SimpleNamespace(
        prepare_read=prepare,
        _pipeline=SimpleNamespace(release_reservation=lambda value: None),
    )
    monkeypatch.setattr("data_access.get_store", lambda: store)
    def build(_store, **kwargs):
        return DataReadIdentity(
            dataset=kwargs["dataset"], revision="r1", calendar_identity="cal",
            universe_snapshot="uni", source_snapshot=kwargs["prepared"].resolved_source_snapshot.content_digest,
            fields=(ResolvedFieldIdentity(logical_name="Revenue", physical_name="Value",
                                          dataset=kwargs["dataset"], availability="same_day"),),
        )
    monkeypatch.setattr("data_access.read.data_read_identity.build_data_read_identity", build)
    encoded = encode_source_ref(SourceRefSpec(
        table="StockIncome", field="Revenue", dataset="stock_income", market="ashare",
    ))
    scope = SimpleNamespace(universe_id="CSI300", scope_key=lambda: "scope")
    dag = SimpleNamespace(
        roots=[SimpleNamespace(factor_name="x", root=PlanNode("column", attrs={"name": encoded}), execution_scope=scope)],
        shared_nodes={},
    )
    logical = SimpleNamespace(
        dataset=anchor.dataset,
        _child=lambda dataset: secondary if dataset == "stock_income" else None,
    )
    roots, _ = bind_batch_sources(
        dag, source=anchor, execution_context=SimpleNamespace(data_source=logical, market="ashare"),
    )
    binding = roots["x"].attrs["physical_source_binding"]
    assert prepared_datasets == ["stock_income"]
    assert binding.read_identity.dataset == "stock_income"
    assert binding.logical_field == encoded


def test_preflight_unknown_field_is_root_local_and_prepares_valid_batch_once(monkeypatch):
    from factor_engine.runtime.physical_source_binding import preflight_batch_sources
    from factor_engine.storage.sources.data_access_source import UnknownFieldSemanticError

    source = object.__new__(DataAccessSource)
    source.production = source.strict_unknown_fields = source.pit_enforce = True
    source.dataset, source.params = "daily", {}
    source.instrument_filter = ["000001.SZ"]
    source.start_date, source.end_date = "2024-01-01", "2024-01-31"
    field_plan = SimpleNamespace(source="catalog", physical_fields=("Value",))
    def ensure(names):
        if "Missing" in names:
            raise UnknownFieldSemanticError("Missing")
        return {name: field_plan for name in names}
    source._ensure_field_plans = ensure
    source._resolve_catalog_fields = lambda names: {name: object() for name in names}
    source.semantic_catalog_identity = lambda: "catalog:v1"
    counters = {"prepare": 0}
    def prepare(*args, **kwargs):
        counters["prepare"] += 1
        return SimpleNamespace(
            resolved_source_snapshot=SimpleNamespace(content_digest="snap"),
            resource_reservation=object(),
        )
    store = SimpleNamespace(
        prepare_read=prepare,
        _pipeline=SimpleNamespace(release_reservation=lambda value: None),
    )
    monkeypatch.setattr("data_access.get_store", lambda: store)
    monkeypatch.setattr(
        "data_access.read.data_read_identity.build_data_read_identity",
        lambda *a, **k: _identity(str(k["columns"][0])),
    )
    scope = SimpleNamespace(universe_id="CSI300", scope_key=lambda: "scope")
    roots = [
        SimpleNamespace(factor_name=name, root=PlanNode("column", attrs={"name": field}), execution_scope=scope)
        for name, field in (("bad", "Missing"), ("a", "Close"), ("b", "Open"))
    ]
    result = preflight_batch_sources(
        SimpleNamespace(roots=roots, shared_nodes={}), source=source,
        execution_context=SimpleNamespace(run_mode="production", runtime_stats={}),
    )
    assert result.scope_error is None
    assert result.per_root_errors["bad"].code == "UNKNOWN_FIELD"
    assert set(result.root_plans) == {"a", "b"}
    assert counters["prepare"] == result.prepare_count == result.unique_source_count == 1


def test_preflight_unsupported_operator_is_root_local(monkeypatch):
    from factor_engine.runtime.physical_source_binding import preflight_batch_sources

    source = object.__new__(DataAccessSource)
    source.production = source.strict_unknown_fields = source.pit_enforce = True
    source.dataset, source.params = "daily", {}
    source.instrument_filter = ["000001.SZ"]
    source.start_date, source.end_date = "2024-01-01", "2024-01-31"
    field_plan = SimpleNamespace(source="catalog", physical_fields=("Value",))
    source._ensure_field_plans = lambda names: {name: field_plan for name in names}
    source._resolve_catalog_fields = lambda names: {name: object() for name in names}
    source.semantic_catalog_identity = lambda: "catalog:v1"
    store = SimpleNamespace(
        prepare_read=lambda *a, **k: SimpleNamespace(
            resolved_source_snapshot=SimpleNamespace(content_digest="snap"),
            resource_reservation=object(),
        ),
        _pipeline=SimpleNamespace(release_reservation=lambda value: None),
    )
    monkeypatch.setattr("data_access.get_store", lambda: store)
    monkeypatch.setattr(
        "data_access.read.data_read_identity.build_data_read_identity",
        lambda *a, **k: _identity(str(k["columns"][0])),
    )
    scope = SimpleNamespace(universe_id="CSI300", scope_key=lambda: "scope")
    col = lambda name: PlanNode("column", attrs={"name": name})
    roots = [
        SimpleNamespace(factor_name="bad", root=PlanNode("v6_no_such_operator", inputs=(col("Close"),)), execution_scope=scope),
        SimpleNamespace(factor_name="a", root=col("Close"), execution_scope=scope),
        SimpleNamespace(factor_name="b", root=col("Open"), execution_scope=scope),
    ]
    result = preflight_batch_sources(
        SimpleNamespace(roots=roots, shared_nodes={}), source=source,
        execution_context=SimpleNamespace(run_mode="production", runtime_stats={}),
    )
    assert result.scope_error is None
    assert result.per_root_errors["bad"].code == "UNSUPPORTED_OPERATOR"
    assert set(result.root_plans) == {"a", "b"}


def test_preflight_permission_failure_rejects_entire_source_scope(monkeypatch):
    from factor_engine.runtime.physical_source_binding import preflight_batch_sources

    source = object.__new__(DataAccessSource)
    source.production = source.strict_unknown_fields = source.pit_enforce = True
    source.dataset, source.params = "daily", {}
    source.instrument_filter = ["000001.SZ"]
    source.start_date, source.end_date = "2024-01-01", "2024-01-31"
    field_plan = SimpleNamespace(source="catalog", physical_fields=("Value",))
    source._ensure_field_plans = lambda names: {name: field_plan for name in names}
    source._resolve_catalog_fields = lambda names: {name: object() for name in names}
    source.semantic_catalog_identity = lambda: "catalog:v1"
    store = SimpleNamespace(prepare_read=lambda *a, **k: (_ for _ in ()).throw(PermissionError("denied")))
    monkeypatch.setattr("data_access.get_store", lambda: store)
    scope = SimpleNamespace(universe_id="CSI300", scope_key=lambda: "scope")
    roots = [
        SimpleNamespace(factor_name=name, root=PlanNode("column", attrs={"name": field}), execution_scope=scope)
        for name, field in (("a", "Close"), ("b", "Open"))
    ]
    result = preflight_batch_sources(
        SimpleNamespace(roots=roots, shared_nodes={}), source=source,
        execution_context=SimpleNamespace(run_mode="production", runtime_stats={}),
    )
    assert result.root_plans == {}
    assert result.per_root_errors == {}
    assert result.scope_error.code == "SOURCE_SCOPE_UNAVAILABLE"
    assert result.scope_error.error_type == "PermissionError"


def test_preflight_keeps_surviving_shared_dependency_closure(monkeypatch):
    from factor_engine.runtime.physical_source_binding import preflight_batch_sources
    from factor_engine.storage.sources.data_access_source import UnknownFieldSemanticError

    source = object.__new__(DataAccessSource)
    source.production = source.strict_unknown_fields = source.pit_enforce = True
    source.dataset, source.params = "daily", {}
    source.instrument_filter = ["000001.SZ"]
    source.start_date, source.end_date = "2024-01-01", "2024-01-31"
    field_plan = SimpleNamespace(source="catalog", physical_fields=("Value",))
    def ensure(names):
        if "Missing" in names:
            raise UnknownFieldSemanticError("Missing")
        return {name: field_plan for name in names}
    source._ensure_field_plans = ensure
    source._resolve_catalog_fields = lambda names: {name: object() for name in names}
    source.semantic_catalog_identity = lambda: "catalog:v1"
    store = SimpleNamespace(
        prepare_read=lambda *a, **k: SimpleNamespace(
            resolved_source_snapshot=SimpleNamespace(content_digest="snap"),
            resource_reservation=object(),
        ),
        _pipeline=SimpleNamespace(release_reservation=lambda value: None),
    )
    monkeypatch.setattr("data_access.get_store", lambda: store)
    monkeypatch.setattr(
        "data_access.read.data_read_identity.build_data_read_identity",
        lambda *a, **k: _identity(str(k["columns"][0])),
    )
    scope = SimpleNamespace(universe_id="CSI300", scope_key=lambda: "scope")
    dag = SimpleNamespace(
        roots=[
            SimpleNamespace(factor_name="bad", root=PlanNode("column", attrs={"name": "Missing"}), execution_scope=scope),
            SimpleNamespace(factor_name="good", root=PlanNode("plan_ref", attrs={"sid": "s1"}), execution_scope=scope),
        ],
        shared_nodes={"s1": PlanNode("column", attrs={"name": "Close"})},
    )
    result = preflight_batch_sources(
        dag, source=source,
        execution_context=SimpleNamespace(run_mode="production", runtime_stats={}),
    )
    assert result.scope_error is None
    assert set(result.root_plans) == {"good"}
    assert set(result.shared_nodes) == {"s1"}
    assert result.shared_nodes["s1"].attrs["physical_source_binding"].logical_field == "Close"


def test_wave_rejects_real_data_access_snapshot_changed_after_preflight(tmp_path, monkeypatch):
    import os
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pytest
    from data_access.registry.loader import DatasetRegistry, StaticDataset
    from data_access.store import DataAccessStore, get_shared_engine
    from factor_engine.runtime.buffer_ref import SourceWaveExecutor
    from factor_engine.runtime.physical_source_binding import PreparedSourceSnapshotExpectation
    from factor_engine.storage.sources.composite_source import SnapshotVerificationError

    parquet = tmp_path / "part.parquet"
    pq.write_table(pa.table({"Close": [1.0]}), parquet)
    dataset = StaticDataset(
        name="daily", access_mode="published", layout="plain",
        time_column=None, instrument_column=None, hive_partitioning=False,
        union_by_name=False, root=str(tmp_path), glob="part.parquet",
        schema={"Close": "float64"},
    )
    store = DataAccessStore(
        registry=DatasetRegistry({"daily": dataset}), engine=get_shared_engine(),
    )
    prepared = store.prepare_read(
        "daily", columns=("Close",), params={}, run_mode="production",
        snapshot_policy="fail_if_changed",
    )
    expected = prepared.resolved_source_snapshot.content_digest
    store._pipeline.release_reservation(prepared.resource_reservation)
    pq.write_table(pa.table({"Close": [2.0, 3.0]}), parquet)
    os.utime(parquet, None)
    monkeypatch.setattr("data_access.get_store", lambda: store)
    calls = {"load": 0}
    source = SimpleNamespace(
        load_columns=lambda names: calls.__setitem__("load", calls["load"] + 1) or {},
        snapshot_token="old",
    )
    ctx = SimpleNamespace(
        runtime_stats={},
        physical_snapshot_expectations=(PreparedSourceSnapshotExpectation(
            dataset="daily", columns=("Close",), params=(), time_range=(None, None),
            instrument_filter=(), content_digest=expected,
        ),),
    )
    wave = SimpleNamespace(
        wave_id=1, columns=frozenset({"Close"}),
        preferred_representation="pandas_columns",
        estimated_scan_bytes=1, estimated_memory_bytes=1,
        physical_union_scan_bytes=1,
    )
    with pytest.raises(SnapshotVerificationError, match="snapshot changed"):
        SourceWaveExecutor(source, ctx).execute_wave(wave)
    assert calls["load"] == 0
    assert ctx.runtime_stats["read_wave_executed"] == 0


def test_native_wave_materializes_once_and_reuses_old_frame_after_file_change(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import polars as pl
    from factor_engine.runtime.buffer_ref import SourceWaveExecutor

    parquet = tmp_path / "native.parquet"
    pq.write_table(pa.table({"Close": [1.0]}), parquet)
    calls = {"scan": 0}
    class NativeSource:
        snapshot_token = "bound"
        def scan_polars_long(self, columns):
            calls["scan"] += 1
            return pl.scan_parquet(parquet).select(columns)
    source = NativeSource()
    ctx = SimpleNamespace(runtime_stats={}, data_source=source)
    wave = SimpleNamespace(
        wave_id=1, columns=frozenset({"Close"}),
        preferred_representation="polars_lazy_long",
        estimated_scan_bytes=1, estimated_memory_bytes=1,
        physical_union_scan_bytes=1,
    )
    ref = SourceWaveExecutor(source, ctx).execute_wave(wave)
    assert ref.meta["native_buffer"].to_dict(as_series=False) == {"Close": [1.0]}
    pq.write_table(pa.table({"Close": [2.0, 3.0]}), parquet)
    reused = ctx.data_source.scan_polars_long(["Close"]).collect()
    assert reused.to_dict(as_series=False) == {"Close": [1.0]}
    assert calls["scan"] == 1


def test_pandas_wave_mapping_is_reused_without_source_cache():
    import pandas as pd
    from factor_engine.runtime.buffer_ref import SourceWaveExecutor

    calls = {"read": 0}
    series = pd.Series([1.0, 2.0])
    class ZeroCacheSource:
        snapshot_token = "bound"
        def load_columns(self, columns):
            calls["read"] += 1
            return {name: series for name in columns}
        def load_column(self, name):
            calls["read"] += 1
            return series
    source = ZeroCacheSource()
    ctx = SimpleNamespace(runtime_stats={}, data_source=source)
    wave = SimpleNamespace(
        wave_id=1, columns=frozenset({"Close"}),
        preferred_representation="pandas_columns",
        estimated_scan_bytes=1, estimated_memory_bytes=1,
        physical_union_scan_bytes=1,
    )
    SourceWaveExecutor(source, ctx).execute_wave(wave)
    assert ctx.data_source.load_column("Close") is series
    assert ctx.data_source.load_columns(["Close"])["Close"] is series
    assert calls["read"] == 1


def test_wave_adapter_keeps_value_alive_until_explicit_release():
    import gc
    import pandas as pd
    from factor_engine.runtime.buffer_ref import SourceWaveExecutor

    source = SimpleNamespace(
        snapshot_token="bound",
        load_columns=lambda columns: {name: pd.Series([1.0]) for name in columns},
    )
    ctx = SimpleNamespace(runtime_stats={}, data_source=source)
    wave = SimpleNamespace(
        wave_id=7, columns=frozenset({"Close"}),
        preferred_representation="pandas_columns",
        estimated_scan_bytes=1, estimated_memory_bytes=1,
        physical_union_scan_bytes=1,
    )
    executor = SourceWaveExecutor(source, ctx)
    ref = executor.execute_wave(wave)
    del ref
    gc.collect()
    assert ctx.data_source.load_column("Close").iloc[0] == 1.0
    executor.release_wave(7)
    assert 7 not in executor._executed


def test_releasing_overlapping_wave_preserves_other_wave_column():
    import pandas as pd
    from factor_engine.runtime.buffer_ref import SourceWaveExecutor

    calls = {"read": 0}
    class Source:
        snapshot_token = "bound"
        def load_columns(self, columns):
            self_value = float(calls["read"] + 1)
            calls["read"] += 1
            return {name: pd.Series([self_value]) for name in columns}
    source = Source()
    ctx = SimpleNamespace(runtime_stats={}, data_source=source)
    def wave(wid):
        return SimpleNamespace(
            wave_id=wid, columns=frozenset({"Close"}),
            preferred_representation="pandas_columns",
            estimated_scan_bytes=1, estimated_memory_bytes=1,
            physical_union_scan_bytes=1,
        )
    executor = SourceWaveExecutor(source, ctx)
    executor.execute_wave(wave(1))
    executor.execute_wave(wave(2))
    assert ctx.data_source.load_column("Close").iloc[0] == 2.0
    executor.release_wave(1)
    assert ctx.data_source.load_column("Close").iloc[0] == 2.0
    assert calls["read"] == 2
    executor.release_wave(2)
