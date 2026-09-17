import copy
import pytest
from evidence.factor_catalog_20260916.reconcile_r19 import update_row

def pair():
    row={"current_formula":"old(x)","original_formula":"original(x)",
         "compile_status":"COMPILE_FAILED","migration_changes":"[]",
         "execution_status":"EXECUTED","value_count":"50","finite_count":"40",
         "result_hash":"a"*64,"execution_evidence_file":"old-evidence"}
    ev={"before_formula":"old(x)","current_formula":"new(x)","changes":["documented"],
        "status":"COMPILED","error":"","bindings":[{"table":"T","column":"X"}],"binding_failures":[]}
    return row,ev

def test_change_clears_execution_and_preserves_original():
    row,ev=pair()
    assert update_row(row,ev,"new-evidence")
    assert row["original_formula"]=="original(x)"
    assert row["execution_status"]=="NOT_RUN"
    assert row["result_hash"]==row["finite_count"]==row["execution_evidence_file"]==""
    assert "old-evidence" in row["migration_changes"]
    assert row["current_fields"]=="T.X"

@pytest.mark.parametrize("mutation",[
    {"before_formula":"wrong"},
    {"status":"COMPILED","error":"still wrong"},
    {"status":"COMPILED","binding_failures":[{"input":"missing"}]},
    {"status":"COMPILE_FAILED","error":""},
    {"status":"NOT_RUN"},
    {"changes":[]},
])
def test_inconsistent_evidence_rejected(mutation):
    row,ev=pair();ev.update(mutation)
    with pytest.raises(ValueError):update_row(row,ev,"new")

def test_failure_keeps_actual_failure_reason():
    row,ev=pair();ev.update(status="COMPILE_FAILED",error="missing field")
    update_row(row,ev,"new")
    assert row["execution_status"]=="COMPILE_FAILED"
    assert row["execution_reason"]=="missing field"
