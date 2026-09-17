import csv
import gzip
import hashlib
import json
from pathlib import Path

from reconcile_r12_smoke import main


FIELDS = [
    "source_row", "id", "original_formula", "current_formula",
    "execution_status", "execution_reason", "value_count", "finite_count",
    "result_hash", "retry_after_batch_abort", "batch_abort_error_type",
    "batch_abort_error", "backend_path", "execution_evidence_file",
    "execution_validation_scope", "execution_window", "execution_symbol_count",
]


def test_all_nonfinite_evidence_is_not_promoted_to_finite_execution(tmp_path):
    checkpoint = tmp_path / "source.csv.gz"
    rows = [
        {"source_row": "2", "id": "finite", "original_formula": "a(x)",
         "current_formula": "a(x)", "execution_status": "NOT_RUN"},
        {"source_row": "3", "id": "all_nan", "original_formula": "b(x)",
         "current_formula": "b(x)", "execution_status": "NOT_RUN"},
    ]
    with gzip.open(checkpoint, "wt", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**dict.fromkeys(FIELDS, ""), **row})
    shard = tmp_path / "real.jsonl.gz"
    records = [
        {"source_row": 2, "id": "finite", "executed_formula": "a(x)",
         "status": "EXECUTED", "value_count": 10, "finite_count": 8,
         "result_hash": "finite-hash"},
        {"source_row": 3, "id": "all_nan", "executed_formula": "b(x)",
         "status": "EXECUTED_ALL_NONFINITE", "value_count": 10,
         "finite_count": 0, "result_hash": "all-nan-hash"},
    ]
    with gzip.open(shard, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    ledger = tmp_path / "closed.ledger.jsonl"
    ledger.write_text("\n".join(json.dumps(event) for event in (
        {"event": "start"},
        {"event": "batch_complete", "output": shard.name, "processed": 2},
        {"event": "sweep_closed", "processed": 2, "requested": 2,
         "requested_count_completed": True},
    )) + "\n", encoding="utf-8")
    prefix = tmp_path / "out"
    main([
        "--checkpoint", str(checkpoint), "--expected-sha",
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "--ledger", str(ledger), "--output-prefix", str(prefix),
    ])
    with gzip.open(str(prefix) + ".csv.gz", "rt", encoding="utf-8-sig", newline="") as stream:
        actual = list(csv.DictReader(stream))
    assert actual[0]["execution_status"] == "EXECUTED"
    assert actual[0]["finite_count"] == "8"
    assert actual[1]["execution_status"] == "EXECUTED_ALL_NONFINITE"
    assert actual[1]["finite_count"] == "0"
    assert actual[1]["execution_validation_scope"] == (
        "formula_matched_research_smoke_not_code_certification"
    )
