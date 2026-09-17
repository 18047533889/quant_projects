"""Real no-read pipeline integration: rewrites invalidate old evidence."""
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest


def test_r13_pipeline_compiles_new_formula_without_reusing_execution(tmp_path, monkeypatch):
    from review_catalog_r12 import main
    columns = [
        "source_row", "id", "original_formula", "current_formula", "migration_changes",
        "original_fields", "original_tables", "domain", "pit", "logic", "batch",
        "bindings", "current_fields", "current_tables", "static_status", "static_reason",
        "compile_status", "compile_reason", "compile_evidence_file", "compile_validation_scope",
        "execution_status", "execution_reason", "value_count", "finite_count", "result_hash",
        "retry_after_batch_abort", "batch_abort_error_type", "batch_abort_error",
        "backend_path", "execution_evidence_file", "execution_validation_scope",
        "execution_window", "execution_symbol_count",
    ]
    row = dict.fromkeys(columns, "")
    old = "cs_rank_gaussian(close, 3.0)"
    row.update(source_row="2", id="r12_fixture", original_formula=old,
               current_formula=old, migration_changes="[]",
               logic="高斯化 cs_rank_gaussian", original_fields="high,low,close",
               original_tables="StockDailyBarAdj", compile_status="COMPILE_FAILED",
               execution_status="EXECUTED", result_hash="old-value-hash",
               execution_evidence_file="old-smoke.gz")
    source = tmp_path / "source.csv.gz"
    with gzip.open(source, "wt", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)
    prefix = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "review", "--recipe-set", "r13", "--checkpoint", str(source),
        "--expected-source-sha256", hashlib.sha256(source.read_bytes()).hexdigest(),
        "--output-prefix", str(prefix),
    ])
    main()
    with gzip.open(str(prefix) + ".csv.gz", "rt", encoding="utf-8-sig", newline="") as stream:
        actual = next(csv.DictReader(stream))
    assert actual["original_formula"] == old
    assert actual["current_formula"] == "cs_rank_gaussian(close, 'blom')"
    assert actual["compile_status"] == "COMPILED"
    assert actual["execution_status"] == "NOT_RUN"
    assert actual["result_hash"] == actual["execution_evidence_file"] == ""
    assert actual["source_row"] == "2" and actual["id"] == "r12_fixture"
    manifest = json.loads(Path(str(prefix) + ".manifest.json").read_text())
    assert manifest["counts"]["changed"] == 1
    for key, suffix in [("checkpoint", ".csv.gz"), ("input", ".input.jsonl.gz"),
                        ("decisions", ".decisions.jsonl.gz")]:
        assert hashlib.sha256(Path(str(prefix) + suffix).read_bytes()).hexdigest() == manifest["output_sha256"][key]
    assert not list(tmp_path.glob("*.partial"))


def test_truncated_input_never_publishes_checkpoint(tmp_path, monkeypatch):
    from review_catalog_r12 import main
    source = tmp_path / "source.csv.gz"
    with gzip.open(source, "wt") as stream:
        stream.write("source_row,id,original_formula,current_formula,execution_status,execution_validation_scope\n")
    source.write_bytes(source.read_bytes()[:-8])
    prefix = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "review", "--recipe-set", "r13", "--checkpoint", str(source),
        "--expected-source-sha256", hashlib.sha256(source.read_bytes()).hexdigest(),
        "--output-prefix", str(prefix),
    ])
    with pytest.raises(EOFError):
        main()
    assert not list(tmp_path.glob("out*"))
