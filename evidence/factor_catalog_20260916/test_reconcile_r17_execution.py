import csv, gzip, hashlib, json
from pathlib import Path
from types import SimpleNamespace

import pytest

from reconcile_r17_execution import reconcile


FIELDS = [
    "source_row","id","original_formula","current_formula","migration_changes",
    "execution_status","execution_reason","execution_validation_scope",
    "execution_evidence_file","value_count","finite_count","result_hash",
    "retry_after_batch_abort","batch_abort_error_type","batch_abort_error",
    "backend_path","execution_window","execution_symbol_count",
]


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _gzip_jsonl(path, rows):
    with gzip.open(path,"wt",encoding="utf-8") as stream:
        for row in rows: stream.write(json.dumps(row)+"\n")


def _fixture(tmp_path):
    formula1="add(close, 1)"; formula2="subtract(close, 1)"
    checkpoint=tmp_path/"source.csv.gz"
    with gzip.open(checkpoint,"wt",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS); writer.writeheader()
        for n,formula in ((1,formula1),(2,formula2)):
            row={key:"" for key in FIELDS}; row.update(source_row=n,id=f"f{n}",original_formula=formula,current_formula=formula,migration_changes="[]",execution_status="NOT_RUN")
            writer.writerow(row)
    request=tmp_path/"request.jsonl.gz"; _gzip_jsonl(request,[{"source_row":1,"id":"f1","formula":formula1},{"source_row":2,"id":"f2","formula":formula2}])
    execution=tmp_path/"run.output.jsonl.gz"; _gzip_jsonl(execution,[
        {"source_row":1,"id":"f1","status":"EXECUTED","executed_formula":formula1,"value_count":8,"finite_count":7,"result_hash":"a"*64},
        {"source_row":2,"id":"f2","status":"EXECUTION_FAILED","executed_formula":formula2,"error_type":"RuntimeError","error":"boom"},
    ])
    summary=tmp_path/"run.output.summary.json"; summary.write_text(json.dumps({"complete":True,"processed":2,"requested_limit":2,"counts":{"EXECUTED":1,"EXECUTION_FAILED":1},"input":str(request),"output":str(execution),"window":["2025-01-01","2025-01-31"],"symbols":["A"]}))
    args=SimpleNamespace(checkpoint=checkpoint,checkpoint_sha256=_sha(checkpoint),input=request,input_sha256=_sha(request),execution=execution,execution_sha256=_sha(execution),summary=summary,summary_sha256=_sha(summary),expected_records=2,expected_batch_counts="EXECUTED=1,EXECUTION_FAILED=1",expected_catalog_counts="EXECUTED=1,EXECUTION_FAILED=1",output_prefix=tmp_path/"out")
    return args


def _read_output(path):
    with gzip.open(path,"rt",encoding="utf-8-sig",newline="") as stream: return list(csv.DictReader(stream))


def _replace_first_result(args, item):
    with gzip.open(args.execution, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    rows[0] = {"source_row": 1, "id": "f1", "executed_formula": "add(close, 1)", **item}
    _gzip_jsonl(args.execution, rows)
    args.execution_sha256 = _sha(args.execution)
    summary = json.loads(args.summary.read_text())
    summary["counts"] = {item["status"]: 1, "EXECUTION_FAILED": 1}
    args.summary.write_text(json.dumps(summary))
    args.summary_sha256 = _sha(args.summary)
    args.expected_batch_counts = f"{item['status']}=1,EXECUTION_FAILED=1"
    args.expected_catalog_counts = args.expected_batch_counts
    return args


def test_reconciles_success_and_real_failure_without_not_run(tmp_path):
    args=_fixture(tmp_path); payload=reconcile(args); rows=_read_output(tmp_path/"out.csv.gz")
    assert payload["batch_execution_status_counts"]=={"EXECUTED":1,"EXECUTION_FAILED":1}
    assert rows[0]["finite_count"]=="7" and rows[0]["result_hash"]=="a"*64
    assert rows[1]["execution_reason"]=="RuntimeError: boom"
    assert rows[1]["value_count"]==rows[1]["finite_count"]==rows[1]["result_hash"]==""
    assert all(row["execution_status"]!="NOT_RUN" for row in rows)
    assert all("R17_EXECUTION_RECONCILE" in row["migration_changes"] for row in rows)


def test_reconciles_all_nonfinite_as_actual_execution_and_preserves_result(tmp_path):
    item = {"status": "EXECUTED_ALL_NONFINITE", "value_count": 2560,
            "finite_count": 0, "result_hash": "b" * 64,
            "backend_path": ["polars", "native"]}
    args = _replace_first_result(_fixture(tmp_path), item)
    payload = reconcile(args)
    row = _read_output(tmp_path / "out.csv.gz")[0]
    assert payload["batch_execution_status_counts"]["EXECUTED_ALL_NONFINITE"] == 1
    assert row["execution_status"] == "EXECUTED_ALL_NONFINITE"
    assert row["execution_reason"] == "execution completed with zero finite values"
    assert row["value_count"] == "2560" and row["finite_count"] == "0"
    assert row["result_hash"] == "b" * 64
    assert json.loads(row["backend_path"]) == ["polars", "native"]


@pytest.mark.parametrize("overrides", [
    {"finite_count": 1},
    {"result_hash": ""},
    {"value_count": True},
    {"finite_count": False},
])
def test_rejects_contradictory_all_nonfinite_evidence(tmp_path, overrides):
    item = {"status": "EXECUTED_ALL_NONFINITE", "value_count": 2560,
            "finite_count": 0, "result_hash": "b" * 64, **overrides}
    args = _replace_first_result(_fixture(tmp_path), item)
    with pytest.raises(ValueError, match="all-nonfinite"):
        reconcile(args)


def _split_into_two_groups(args, tmp_path):
    with gzip.open(args.input,"rt") as stream: requests=[json.loads(x) for x in stream]
    with gzip.open(args.execution,"rt") as stream: results=[json.loads(x) for x in stream]
    inputs=[]; executions=[]; summaries=[]
    for index in range(2):
        inp=tmp_path/f"request{index}.jsonl.gz"; out=tmp_path/f"run{index}.output.jsonl.gz"; summary=tmp_path/f"run{index}.output.summary.json"
        _gzip_jsonl(inp,[requests[index]]); _gzip_jsonl(out,[results[index]])
        counts={results[index]["status"]:1}; summary.write_text(json.dumps({"complete":True,"processed":1,"requested_limit":1,"counts":counts,"input":str(inp),"output":str(out),"window":["2025-01-01","2025-01-31"],"symbols":["A"]}))
        inputs.append(inp); executions.append(out); summaries.append(summary)
    args.input=inputs; args.input_sha256=[_sha(x) for x in inputs]; args.execution=executions; args.execution_sha256=[_sha(x) for x in executions]
    args.summary=summaries; args.summary_sha256=[_sha(x) for x in summaries]; args.expected_records=[1,1]; args.expected_batch_counts=["EXECUTED=1","EXECUTION_FAILED=1"]
    return args


def test_reconciles_multiple_independently_pinned_groups_in_one_pass(tmp_path):
    args=_split_into_two_groups(_fixture(tmp_path),tmp_path); payload=reconcile(args)
    assert len(payload["execution_groups"])==2
    assert payload["execution_records"]==2
    assert payload["batch_execution_status_counts"]=={"EXECUTED":1,"EXECUTION_FAILED":1}
    assert {row["execution_status"] for row in _read_output(tmp_path/"out.csv.gz")}=={"EXECUTED","EXECUTION_FAILED"}


def test_rejects_duplicate_identity_across_groups(tmp_path):
    args=_split_into_two_groups(_fixture(tmp_path),tmp_path)
    with gzip.open(args.input[0],"rt") as stream: request=json.loads(next(stream))
    with gzip.open(args.execution[0],"rt") as stream: result=json.loads(next(stream))
    _gzip_jsonl(args.input[1],[request]); _gzip_jsonl(args.execution[1],[result])
    summary=json.loads(args.summary[1].read_text()); summary.update(input=str(args.input[1]),output=str(args.execution[1]),counts={"EXECUTED":1}); args.summary[1].write_text(json.dumps(summary))
    args.input_sha256[1]=_sha(args.input[1]); args.execution_sha256[1]=_sha(args.execution[1]); args.summary_sha256[1]=_sha(args.summary[1]); args.expected_batch_counts[1]="EXECUTED=1"
    with pytest.raises(ValueError,match="across groups"): reconcile(args)


def test_rejects_misaligned_group_arguments(tmp_path):
    args=_split_into_two_groups(_fixture(tmp_path),tmp_path); args.summary_sha256=args.summary_sha256[:1]
    with pytest.raises(ValueError,match="do not align"): reconcile(args)


def test_rejects_wrong_pin(tmp_path):
    args=_fixture(tmp_path); args.execution_sha256="0"*64
    with pytest.raises(ValueError,match="pinned execution hash"): reconcile(args)


def test_rejects_wrong_formula(tmp_path):
    args=_fixture(tmp_path); rows=[]
    with gzip.open(args.execution,"rt") as stream: rows=[json.loads(x) for x in stream]
    rows[0]["executed_formula"]="close"; _gzip_jsonl(args.execution,rows); args.execution_sha256=_sha(args.execution)
    with pytest.raises(ValueError,match="formula mismatch"): reconcile(args)


def test_rejects_duplicate_identity(tmp_path):
    args=_fixture(tmp_path); row={"source_row":1,"id":"f1","formula":"add(close, 1)"}; _gzip_jsonl(args.input,[row,row]); args.input_sha256=_sha(args.input)
    with pytest.raises(ValueError,match="duplicate"): reconcile(args)


def test_rejects_incomplete_or_conflicting_summary(tmp_path):
    args=_fixture(tmp_path); summary=json.loads(args.summary.read_text()); summary["complete"]=False; args.summary.write_text(json.dumps(summary)); args.summary_sha256=_sha(args.summary)
    with pytest.raises(ValueError,match="summary mismatch"): reconcile(args)


def test_rejects_source_that_does_not_cover_request(tmp_path):
    args=_fixture(tmp_path); req={"source_row":3,"id":"f3","formula":"close"}; result={"source_row":3,"id":"f3","status":"EXECUTION_FAILED","executed_formula":"close","error_type":"E","error":"bad"}
    _gzip_jsonl(args.input,[req]); _gzip_jsonl(args.execution,[result]); args.input_sha256=_sha(args.input); args.execution_sha256=_sha(args.execution); args.expected_records=1; args.expected_batch_counts="EXECUTION_FAILED=1"
    summary={"complete":True,"processed":1,"requested_limit":1,"counts":{"EXECUTION_FAILED":1},"input":str(args.input),"output":str(args.execution)}; args.summary.write_text(json.dumps(summary)); args.summary_sha256=_sha(args.summary)
    with pytest.raises(ValueError,match="does not cover"): reconcile(args)


@pytest.mark.parametrize("item",[
    {"status":"EXECUTED","value_count":8,"finite_count":0,"result_hash":"a"*64},
    {"status":"EXECUTED","value_count":8,"finite_count":7,"result_hash":"z"*64},
    {"status":"BATCH_ABORTED","error_type":"","error":"boom"},
])
def test_rejects_malformed_terminal_evidence(tmp_path,item):
    args=_fixture(tmp_path); item={"source_row":1,"id":"f1","executed_formula":"add(close, 1)",**item}; second={"source_row":2,"id":"f2","status":"EXECUTION_FAILED","executed_formula":"subtract(close, 1)","error_type":"E","error":"bad"}; _gzip_jsonl(args.execution,[item,second]); args.execution_sha256=_sha(args.execution)
    with pytest.raises(ValueError): reconcile(args)
