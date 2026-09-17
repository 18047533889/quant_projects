import json
import pytest
from reconcile_r20_full import apply_runtime_abort
def row():
    return dict(source_row="2",id="x",current_formula="close",compile_status="COMPILED",
                execution_status="EXECUTED",execution_reason="",value_count="40",finite_count="40",
                migration_changes="[]")
def event():
    return dict(source_row=2,id="x",executed_formula="close",status="BATCH_ABORTED",
                error_type="PhysicalPlanRequiredError",error="region row estimate unavailable",
                execution_window=["2026-04-20","2026-04-24"],execution_symbol_count=8)
def test_aborted_batch_never_keeps_old_success_counts():
    r=row();apply_runtime_abort(r,event(),"test.jsonl.gz")
    assert r["compile_status"]=="COMPILED"
    assert r["execution_status"]=="BATCH_ABORTED"
    assert r["value_count"]==r["finite_count"]==""
    assert "not an individual factor diagnosis" in r["execution_reason"]
    assert "EXECUTED" in json.loads(r["migration_changes"])[0]
@pytest.mark.parametrize("key,value",[("executed_formula","open"),("id","other"),("source_row",3),("status","EXECUTED"),("error","")])
def test_runtime_lineage_rejects_wrong_scope(key,value):
    e=event();e[key]=value
    with pytest.raises(ValueError):apply_runtime_abort(row(),e,"test")
