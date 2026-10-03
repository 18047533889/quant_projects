"""Mock-only contracts for the real automatic TRAIN/VALIDATION audit."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("real_automatic_audit", SCRIPTS / "audit_real_automatic_oct03.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
def sample():
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from factor_optimizer.research_batch import automatic_time_split
    n,a=500,256; times=np.arange(n); assets=np.asarray([f"a{i}.SZ" for i in range(a)])
    ta,aa=AxisRef("time","int",n,times),AxisRef("asset","str",a,assets)
    batch=FactorBatch(("mock",),ta,aa,np.arange(n*a,dtype=np.float32).reshape(n,a,1))
    labels=LabelBundle("mock-return",np.full((n,a),.01,dtype=np.float32),1,
        decision_time=tuple(times),label_start_time=tuple(times+1),label_end_time=tuple(times+2),asset_axis=aa)
    split=automatic_time_split(labels); old=AUDIT.load_real_source.__globals__; sha="b"*64
    provenance={"manifest_uri":old["MANIFEST_PREFIX"]+"a"*64+"/landing_manifest.json",
        "sources":[{"factor":"mock","uri":old["FACTOR_POOL"]+f"/{sha}/mock.parquet","etag":"mock-etag",
        "sha256":sha,"downloaded_bytes":123,"manifest_sha256":"c"*64,
        "source_status":"evaluated_optimization_pending","expression":"mock"}],
        "retained_factor_ids":["mock"],"asset_selection_split":split.identity,
        "asset_selection_train_days":len(split.train_indices),"max_factor_bytes":old["MAX_FACTOR_BYTES"],
        "max_batch_factor_bytes":old["MAX_BATCH_BYTES"]}
    return batch,labels,provenance,{"mock":{"treatment":"raw"}}
def result_for(batch,labels,*,test_evaluated=False):
    from factor_optimizer.research_batch import automatic_time_split
    from quant_evaluator.contracts.factor_batch import FactorBatch
    split=automatic_time_split(labels); values=np.array(batch.values,copy=True)
    output=FactorBatch(batch.factor_ids,batch.time_axis,batch.asset_axis,values,
        validity=np.isfinite(values),context_refs={"optimization_mode":"research_only","split":split.identity})
    plan=SimpleNamespace(identity="plan-hash")
    item=SimpleNamespace(factor_id="mock",status="selected",selected_family="RAW",reason="mock",
        plan_identity=plan.identity,plan=plan,train_gain=.2,validation_lower_bound=.1,
        validation_candidate_identity="candidate",validation_coverage=.9,materialization_error=None,
        training_diagnostics={"candidate_budget":{"proposed":2}},baseline_diagnostics={},joint_diagnostics={},
        candidates=({"family":"RAW","score":.3},))
    return SimpleNamespace(optimized=output,factors={"mock":item},split=split,
        execution_mode="research_only",test_evaluated=test_evaluated)
def prepare(monkeypatch):
    data=sample(); monkeypatch.setattr(AUDIT,"check_environment",lambda:None)
    monkeypatch.setattr(AUDIT,"check_resource_headroom",lambda:None); return data
def test_run_audit_forwards_lineages_and_valid_summary(monkeypatch):
    batch,labels,provenance,lineages=prepare(monkeypatch); observed={}
    def runner(b,y,**kwargs): observed.update(kwargs); return result_for(b,y)
    report=AUDIT.run_audit(source_loader=lambda:(batch,labels,provenance,lineages),auto_runner=runner,resource_check=lambda:None)
    assert observed["lineages"] is lineages and observed["allow_research"] is True
    assert report["test_evaluated"] is False
    assert report["source_fingerprints"]["before"]==report["source_fingerprints"]["after"]
    assert report["automatic"]["mock"]["train_gain"] == .2
    assert report["automatic"]["mock"]["candidates"][0]["score"] == .3
@pytest.mark.parametrize("damage,match",[("ids","factor IDs/order"),("time","time axis"),
    ("validity","validity"),("test","TEST unscored")])
def test_validate_result_rejects_malformed_output(monkeypatch,damage,match):
    batch,labels,_,_=prepare(monkeypatch); result=result_for(batch,labels,test_evaluated=(damage=="test"))
    if damage=="ids": object.__setattr__(result.optimized,"factor_ids",("wrong",))
    if damage=="time":
        from quant_evaluator.contracts.factor_batch import AxisRef
        vals=np.asarray(batch.time_axis.values)+1
        object.__setattr__(result.optimized,"time_axis",AxisRef("time","int",len(vals),vals))
    if damage=="validity": object.__setattr__(result.optimized,"validity",np.zeros(batch.values.shape,bool))
    from factor_optimizer.research_batch import automatic_time_split
    with pytest.raises(ValueError,match=match): AUDIT.validate_result(result,batch,automatic_time_split(labels))
def test_run_audit_rejects_input_mutation(monkeypatch):
    batch,labels,provenance,lineages=prepare(monkeypatch)
    def mutate(b,y,**kwargs):
        out=result_for(b,y)
        changed=np.array(b.values,copy=True)
        changed[0,0,0]+=1
        object.__setattr__(b,"values",changed)
        return out
    with pytest.raises(RuntimeError,match="mutated its bound source"):
        AUDIT.run_audit(source_loader=lambda:(batch,labels,provenance,lineages),auto_runner=mutate,resource_check=lambda:None)
def test_report_is_exclusive_and_capped(monkeypatch,tmp_path):
    batch,labels,provenance,lineages=prepare(monkeypatch)
    report=AUDIT.run_audit(source_loader=lambda:(batch,labels,provenance,lineages),
        auto_runner=lambda b,y,**kwargs:result_for(b,y),resource_check=lambda:None)
    path=tmp_path/"audit.json"; AUDIT.write_report_exclusive(path,report)
    with pytest.raises(FileExistsError): AUDIT.write_report_exclusive(path,report)
    old=sys.modules["audit_real_training_methods_oct03"]; monkeypatch.setattr(old,"MAX_REPORT_BYTES",1)
    with pytest.raises(ValueError,match="1 MiB"):
        AUDIT.run_audit(source_loader=lambda:(batch,labels,provenance,lineages),
            auto_runner=lambda b,y,**kwargs:result_for(b,y),resource_check=lambda:None)

def test_nonfinite_metric_is_rejected_not_silently_filled(monkeypatch):
    batch,labels,provenance,lineages=prepare(monkeypatch)
    def runner(b,y,**kwargs):
        result=result_for(b,y); result.factors["mock"].train_gain=float("nan"); return result
    with pytest.raises(ValueError,match="Out of range float"):
        AUDIT.run_audit(source_loader=lambda:(batch,labels,provenance,lineages),
                        auto_runner=runner,resource_check=lambda:None)
