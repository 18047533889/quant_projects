import gzip, hashlib, json

import pytest

from reconcile_r15_combined import _load_compile, _load_execution


def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def _compile(tmp_path, records, source_sha="source"):
    path=tmp_path/"compile.jsonl.gz"
    with gzip.open(path,"wt") as f:
        for item in records: f.write(json.dumps(item)+"\n")
    path.with_suffix(".summary.json").write_text(json.dumps({"complete":True,"processed":len(records),"requested":len(records),"input_sha256":source_sha,"output_sha256":_sha(path)}))
    return path


def _c(i=1, source_sha="source"):
    formula="add(close, 1)"
    return {"source_row":i,"id":f"f{i}","executed_formula":formula,"formula_sha256":hashlib.sha256(formula.encode()).hexdigest(),"source_csv_sha256":source_sha,"compile_status":"COMPILED"}


def test_compile_loader_accepts_closed_exact_shard(tmp_path):
    path=_compile(tmp_path,[_c()])
    records,_=_load_compile([path],"source")
    assert set(records)=={(1,"f1")}


def test_compile_loader_rejects_duplicate_and_wrong_source(tmp_path):
    path=_compile(tmp_path,[_c(),_c()])
    with pytest.raises(ValueError,match="duplicate compile"):
        _load_compile([path],"source")


def test_compile_loader_rejects_partial_name(tmp_path):
    path=tmp_path/"compile.jsonl.gz.partial"; path.write_bytes(b"x")
    with pytest.raises(ValueError,match="not closed"):
        _load_compile([path],"source")


def test_execution_loader_rejects_failed_record(tmp_path):
    path=tmp_path/"execution.jsonl.gz"
    with gzip.open(path,"wt") as f:
        f.write(json.dumps({"source_row":1,"id":"f1","status":"BATCH_ABORTED","executed_formula":"x"})+"\n")
    path.with_suffix(".summary.json").write_text(json.dumps({"complete":True,"processed":1,"requested_limit":1}))
    with pytest.raises(ValueError,match="non-success"):
        _load_execution([path])
