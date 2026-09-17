"""SourceRef semantic identity must reach DataAccess before panel validation."""
from types import SimpleNamespace
import pandas as pd
import pytest
from factor_engine.api.source_ref import SourceRefSpec,encode_source_ref
from factor_engine.storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource

def fixture(monkeypatch):
    import factor_engine.storage.sources.lqtp_logical_source as base
    anchor=pd.MultiIndex.from_tuples([(pd.Timestamp("2025-01-02"),"A")],names=["timestamp","instrument"])
    calls=[]
    class Child:
        data_snapshot_id="fixture"
        def __init__(self,**kwargs):
            calls.append(kwargs);self.kwargs=kwargs
            if kwargs["dataset"]=="ashare_index_constituent":
                assert kwargs["semantic_filters"].get("IndexSymbol")
        def load_columns(self,fields):
            identity=self.kwargs["semantic_filters"].get("IndexSymbol")
            data={"IndexSymbol":identity,"Weight":1.0 if identity=="000300.SH" else 2.0}
            if kwargs_dataset := self.kwargs.get("dataset"):
                if "topten" in kwargs_dataset:
                    rank=self.kwargs["semantic_filters"]["ShareholderRank"]
                    data.update(ShareNumber=100.0*rank,ShareholderId=f"entity-{rank}",ShareholderRank=rank)
            return {name:pd.Series([data[name]],index=anchor,name=name) for name in fields}
        def load_column(self,name):return self.load_columns([name])[name]
        def close(self):pass
    monkeypatch.setattr(base,"DataAccessSource",Child)
    inner=SimpleNamespace(start_date="2025-01-02",end_date="2025-01-02",
                          instrument_filter=["A"],semantic_filters={})
    source=LQTPLogicalDataSource(inner);source._cache["__anchor_index__"]=anchor
    return source,calls,inner

@pytest.mark.parametrize("batch",[False,True])
def test_exact_index_filter_reaches_child_and_is_not_an_instrument(batch,monkeypatch):
    source,calls,inner=fixture(monkeypatch)
    specs=[SourceRefSpec("IndexConstituent","Weight",params=(("IndexSymbol",symbol),))
           for symbol in ("000300.SH","000905.SH")]
    if batch:
        names=[encode_source_ref(s) for s in specs]
        result=source.load_source_refs_batch(names)
        values=[result[n].iloc[0] for n in names]
    else: values=[source._load_source_ref(s).iloc[0] for s in specs]
    assert values==[1.0,2.0]
    assert [c["semantic_filters"]["IndexSymbol"] for c in calls]==["000300.SH","000905.SH"]
    assert all(c["instrument_filter"]==["A"] for c in calls)
    assert inner.semantic_filters=={}
    deps=source.collect_source_dependencies()
    assert {d.get("IndexSymbol") for d in deps}=={"000300.SH","000905.SH"}
    assert len(deps)==2

@pytest.mark.parametrize("batch",[False,True])
def test_missing_index_stays_fail_closed(batch,monkeypatch):
    from factor_engine.storage.sources.data_access_source import MissingDataDependencyError
    source,calls,_=fixture(monkeypatch);s=SourceRefSpec("IndexConstituent","Weight")
    with pytest.raises(MissingDataDependencyError,match="IndexSymbol"):
        if batch: source.load_source_refs_batch([encode_source_ref(s)])
        else: source._load_source_ref(s)
    assert calls==[]

def test_explicit_identity_overrides_ambient_without_mutating_parent(monkeypatch):
    source,calls,inner=fixture(monkeypatch)
    inner.semantic_filters={"IndexSymbol":"000905.SH"}
    source._load_source_ref(SourceRefSpec("IndexConstituent","Weight",params=(("IndexSymbol","000300.SH"),)))
    assert calls[0]["semantic_filters"]=={"IndexSymbol":"000300.SH"}
    assert inner.semantic_filters=={"IndexSymbol":"000905.SH"}


def test_lineage_retains_fields_is_idempotent_and_order_independent(monkeypatch):
    source,_,_=fixture(monkeypatch)
    rows=[{"field":"Close","IndexSymbol":"000300.SH"},
          {"field":"PreClose","IndexSymbol":"000300.SH"},
          {"field":"Close","IndexSymbol":"000905.SH"}]
    for row in rows+rows:
        source._record_dependency("index",kind="benchmark",snapshot_id="v1",**row)
    expected=source.source_dependency_hash()
    assert len(source.collect_source_dependencies())==3
    source._dependency_store().clear()
    for row in reversed(rows):source._record_dependency("index",kind="benchmark",snapshot_id="v1",**row)
    assert source.source_dependency_hash()==expected


@pytest.mark.parametrize("batch",[False,True])
def test_holder_rank_reaches_physical_reader_and_keeps_identity(batch,monkeypatch):
    source,calls,inner=fixture(monkeypatch)
    specs=[SourceRefSpec("StockTopTenShareholder","ShareNumber",params=(("ShareholderRank",rank),)) for rank in (1,2)]
    if batch:
        names=[encode_source_ref(s) for s in specs]
        result=source.load_source_refs_batch(names)
        values=[result[n].iloc[0] for n in names]
    else:values=[source._load_source_ref(s).iloc[0] for s in specs]
    assert values==[100.,200.]
    assert [c["semantic_filters"] for c in calls]==[{"ShareholderRank":1},{"ShareholderRank":2}]
    assert {d["ShareholderRank"] for d in source.collect_source_dependencies()}=={1,2}
    assert inner.semantic_filters=={}


@pytest.mark.parametrize("params",[{},{"aggregate":"top1"},{"rank":0},{"rank":True},{"rank":1,"ShareholderRank":2}])
def test_holder_unsupported_reduction_fails_before_read(params,monkeypatch):
    from factor_engine.storage.sources.data_access_source import MissingDataDependencyError
    source,calls,_=fixture(monkeypatch)
    spec=SourceRefSpec("StockTopTenShareholder","ShareNumber",params=tuple(params.items()))
    with pytest.raises(MissingDataDependencyError):source._load_source_ref(spec)
    assert calls==[]
