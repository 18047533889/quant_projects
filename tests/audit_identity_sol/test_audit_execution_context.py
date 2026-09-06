import pytest

from factor_engine.runtime.production_policy import ProductionPolicyViolation, resolve_run_mode
from factor_engine.storage.data_scope import compute_data_scope


class AnonymousSource:
    instrument_filter=None


class ParquetLike:
    root="/data/panel"
    instrument_filter=None
    def __init__(self,max_files=None,recursive=False):
        self.max_files=max_files; self.recursive=recursive


def test_unknown_explicit_and_environment_modes_fail_closed(monkeypatch):
    with pytest.raises(ProductionPolicyViolation): resolve_run_mode("prodution")
    with pytest.raises(ProductionPolicyViolation): resolve_run_mode("   ")
    assert resolve_run_mode("paper")=="paper"
    monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE","prodution")
    with pytest.raises(ProductionPolicyViolation): resolve_run_mode()


def test_anonymous_sources_are_object_isolated_not_generic_all_market():
    a,b=AnonymousSource(),AnonymousSource()
    assert compute_data_scope(a).startswith("ephemeral:")
    assert compute_data_scope(a)!=compute_data_scope(b)
    class SchemaOnly:
        timestamp_column="timestamp"; recursive=False; pit_enforce=True
        fields={"close":"float64"}; instrument_filter=None
    first, second = SchemaOnly(), SchemaOnly()
    assert compute_data_scope(first) != compute_data_scope(second)


def test_anonymous_checkpoint_scope_is_unique_even_for_repeated_probe():
    from factor_engine.runtime.stateful_incremental import _source_snapshot_scope
    source = AnonymousSource()
    scopes = {_source_snapshot_scope(source, mode="research") for _ in range(100)}
    assert len(scopes) == 100
    assert all(scope.startswith("ephemeral:") for scope in scopes)


def test_read_options_bind_stable_source_identity():
    assert compute_data_scope(ParquetLike(1,False))!=compute_data_scope(ParquetLike(None,False))
    assert compute_data_scope(ParquetLike(None,True))!=compute_data_scope(ParquetLike(None,False))
    with pytest.raises(TypeError,match="keys must be strings"):
        compute_data_scope(type("Bad",(),{"root":"/x","params":{1:"x"},"instrument_filter":None})())


def test_make_context_can_explicitly_disable_base_cache():
    from factor_engine.runtime.engine import FactorEngine
    engine=object.__new__(FactorEngine)
    engine.cache=object(); engine.data_source=AnonymousSource(); engine.run_mode="research"; engine.production_fallback_policy=None
    ctx=engine._make_context(cache_allowed=False)
    assert ctx.cache is None


def test_make_context_all_entrypoints_disable_cache_for_anonymous_source():
    from factor_engine.runtime.engine import FactorEngine
    engine=object.__new__(FactorEngine)
    engine.cache=object(); engine.data_source=AnonymousSource(); engine.run_mode="research"; engine.production_fallback_policy=None
    assert engine._make_context().cache is None


def test_precompiled_binding_rejects_mixed_plan_before_io(monkeypatch):
    from types import SimpleNamespace
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.runtime.engine import FactorEngine
    engine=object.__new__(FactorEngine)
    expected=PlanNode(op="literal",attrs={"value":1})
    analysis=SimpleNamespace(ir=PlanNode(op="literal",attrs={"value":1}),lookback=0,referenced_columns=())
    monkeypatch.setattr(engine,"compile",lambda *a,**k:(expected,analysis))
    with pytest.raises(ProductionPolicyViolation,match="binding mismatch"):
        engine._assert_precompiled_binding(SimpleNamespace(),PlanNode(op="literal",attrs={"value":2}),analysis,pit_enforce=True,pit_forbid_forward_fill=False)


def test_production_filter_does_not_impersonate_universe_snapshot():
    from types import SimpleNamespace
    from factor_engine.planner.dag import FactorExecutionScope
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.runtime.engine import assert_execution_scope_contract
    source=SimpleNamespace(instrument_filter=["A","B"])
    with pytest.raises(ProductionPolicyViolation,match="not historical membership"):
        assert_execution_scope_contract(FactorExecutionScope(universe_id="CSI300",market="A"),PlanNode(op="rank"),factor_name="f",data_source=source,mode="production")


@pytest.mark.parametrize("case",["tampered_digest","wrong_universe","no_historical_interval","member_filter_mismatch"])
def test_universe_snapshot_failclosed_cases(case):
    from dataclasses import replace
    from types import SimpleNamespace
    from data_access.r30.universe_snapshot import UniverseSnapshot
    from factor_engine.planner.dag import FactorExecutionScope
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.runtime.engine import assert_execution_scope_contract
    snapshot=UniverseSnapshot.build("CSI300","A",["A","B"],"m1","t1","source-v1")
    universe="CSI300"; members=["A","B"]
    if case=="tampered_digest":snapshot=replace(snapshot,snapshot_id="x")
    if case=="wrong_universe":universe="CSI500"
    if case=="member_filter_mismatch":members=["A"]
    source=SimpleNamespace(universe_snapshot=snapshot,instrument_filter=members,start_date="2020-01-01",end_date="2020-12-31")
    with pytest.raises(ProductionPolicyViolation):
        assert_execution_scope_contract(FactorExecutionScope(universe_id=universe,market="A"),PlanNode(op="rank"),factor_name="f",data_source=source,mode="production")


def test_config_batch_rejects_duplicate_names_before_scope_grouping(monkeypatch):
    from types import SimpleNamespace
    from factor_engine.runtime.engine import FactorEngine
    items=iter([
        (SimpleNamespace(),SimpleNamespace(name="same"),SimpleNamespace()),
        (SimpleNamespace(),SimpleNamespace(name="same"),SimpleNamespace()),
    ])
    monkeypatch.setattr(FactorEngine,"from_config",classmethod(lambda cls,path,profile=None:next(items)))
    with pytest.raises(ValueError,match="same"):
        FactorEngine.run_many_from_config(["a.yml","b.yml"])


def test_boundary_probe_reopens_full_contract_and_observes_intermediate_bar(monkeypatch):
    import pandas as pd
    from factor_engine.runtime.stateful_incremental import BoundaryProbeResult,_boundary_timeline
    idx=pd.MultiIndex.from_tuples([
        (pd.Timestamp("2026-02-27"),"A"),(pd.Timestamp("2026-03-02"),"A"),(pd.Timestamp("2026-03-03"),"A")
    ],names=["timestamp","instrument"])
    full=pd.Series([1.,2.,3.],index=idx)
    class Windowed:
        def execution_spec(self):return {"type":"probe","start_date":"2026-03-03","end_date":"2026-03-03","pit_enforce":True,"market":"US"}
    captured={}
    class Fresh:
        def load_column(self,name):return full
    def rebuild(spec):captured.update(spec);return Fresh()
    monkeypatch.setattr("factor_engine.storage.factory.build_data_source",rebuild)
    status,bars=_boundary_timeline(Windowed(),"close",since="2026-02-27",start="2026-03-03",mode="production")
    assert status is BoundaryProbeResult.PROVEN_CONTIGUOUS
    assert pd.Timestamp("2026-03-02",tz="UTC") in bars
    assert captured["pit_enforce"] is True and captured["market"]=="US"


def test_boundary_probe_without_fresh_session_contract_is_unknown():
    from factor_engine.runtime.stateful_incremental import BoundaryProbeResult,_boundary_timeline
    status,bars=_boundary_timeline(object(),"close",since="2026-02-27",start="2026-03-03",mode="production")
    assert status is BoundaryProbeResult.UNKNOWN and bars==[]
