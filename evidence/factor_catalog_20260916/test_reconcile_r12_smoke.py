import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from reconcile_r12_smoke import main


FIELDS = [
    "source_row", "id", "original_formula", "current_formula",
    "execution_status", "execution_reason", "value_count", "finite_count",
    "result_hash", "retry_after_batch_abort", "batch_abort_error_type",
    "batch_abort_error", "backend_path", "execution_evidence_file",
    "execution_validation_scope", "execution_window", "execution_symbol_count",
]


def _checkpoint(path: Path) -> str:
    rows = [
        {
            "source_row": "2", "id": "changed", "original_formula": "old(x)",
            "current_formula": "new(x)", "execution_status": "NOT_RUN",
        },
        {
            "source_row": "3", "id": "crash", "original_formula": "donchian(x)",
            "current_formula": "donchian(x)", "execution_status": "NATIVE_CRASH",
            "execution_reason": "existing bounded crash",
        },
        {
            "source_row": "4", "id": "unrun", "original_formula": "keep(x)",
            "current_formula": "keep(x)", "execution_status": "NOT_RUN",
        },
    ]
    with gzip.open(path, "wt", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for partial in rows:
            writer.writerow({**dict.fromkeys(FIELDS, ""), **partial})
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(**overrides):
    return {
        "source_row": 2,
        "id": "changed",
        "executed_formula": "new(x)",
        "status": "EXECUTED",
        "value_count": 20,
        "finite_count": 18,
        "result_hash": "fresh-hash",
        "backend_path": ["pandas_numpy"],
        "execution_window": {"start": "2026-01-01", "end": "2026-01-31"},
        "execution_symbol_count": 2,
        **overrides,
    }


def _closed_ledger(tmp_path: Path, records, *, processed=None):
    shard = tmp_path / "batch-000000.jsonl.gz"
    with gzip.open(shard, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    ledger = tmp_path / "run.ledger.jsonl"
    complete = {
        "event": "batch_complete", "output": str(shard),
        "processed": len(records) if processed is None else processed,
        "source_offset": 0, "next_offset": len(records),
    }
    ledger.write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                {"event": "start"}, complete,
                {"event": "sweep_closed", "processed": complete["processed"],
                 "requested": complete["processed"],
                 "requested_count_completed": True},
            )
        ) + "\n",
        encoding="utf-8",
    )
    return ledger, shard


def _run(checkpoint, sha, ledger, prefix):
    return main([
        "--checkpoint", str(checkpoint), "--expected-sha", sha,
        "--ledger", str(ledger), "--output-prefix", str(prefix),
    ])


def test_closed_formula_matched_batch_merges_without_touching_other_rows(tmp_path):
    checkpoint = tmp_path / "source.csv.gz"
    sha = _checkpoint(checkpoint)
    ledger, shard = _closed_ledger(tmp_path, [_record()])
    prefix = tmp_path / "reconciled"

    manifest = _run(checkpoint, sha, ledger, prefix)

    output = Path(str(prefix) + ".csv.gz")
    with gzip.open(output, "rt", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["source_row"], row["id"]) for row in rows] == [
        ("2", "changed"), ("3", "crash"), ("4", "unrun")
    ]
    assert rows[0]["original_formula"] == "old(x)"
    assert rows[0]["current_formula"] == "new(x)"
    assert rows[0]["execution_status"] == "EXECUTED"
    assert rows[0]["result_hash"] == "fresh-hash"
    assert rows[0]["execution_evidence_file"] == shard.name
    assert rows[1]["execution_status"] == "NATIVE_CRASH"
    assert rows[1]["execution_reason"] == "existing bounded crash"
    assert rows[2]["execution_status"] == "NOT_RUN"
    assert manifest["merged_records"] == 1
    assert manifest["input_checkpoint_sha256"] == sha
    assert manifest["closed_evidence"][0]["sha256"] == hashlib.sha256(
        shard.read_bytes()
    ).hexdigest()
    assert manifest["status_counts"]["NATIVE_CRASH"] == 1
    assert manifest["scope"] == "formula_matched_research_smoke_not_code_certification"


@pytest.mark.parametrize("defect", ["wrong_formula", "duplicate", "count_mismatch"])
def test_invalid_complete_batch_publishes_nothing(tmp_path, defect):
    checkpoint = tmp_path / "source.csv.gz"
    sha = _checkpoint(checkpoint)
    records = [_record()]
    processed = None
    if defect == "wrong_formula":
        records[0]["executed_formula"] = "old(x)"
    elif defect == "duplicate":
        records.append(_record())
    else:
        processed = 2
    ledger, _ = _closed_ledger(tmp_path, records, processed=processed)
    prefix = tmp_path / "out"

    with pytest.raises(ValueError):
        _run(checkpoint, sha, ledger, prefix)

    assert not list(tmp_path.glob("out*"))


def test_partial_ledger_publishes_nothing(tmp_path):
    checkpoint = tmp_path / "source.csv.gz"
    sha = _checkpoint(checkpoint)
    ledger, _ = _closed_ledger(tmp_path, [_record()])
    lines = ledger.read_text(encoding="utf-8").splitlines()
    ledger.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    prefix = tmp_path / "out"

    with pytest.raises(ValueError, match="closed"):
        _run(checkpoint, sha, ledger, prefix)

    assert not list(tmp_path.glob("out*"))


def test_truncated_gzip_publishes_nothing(tmp_path):
    checkpoint = tmp_path / "source.csv.gz"
    sha = _checkpoint(checkpoint)
    ledger, shard = _closed_ledger(tmp_path, [_record()])
    shard.write_bytes(shard.read_bytes()[:-8])
    prefix = tmp_path / "out"

    with pytest.raises((EOFError, gzip.BadGzipFile)):
        _run(checkpoint, sha, ledger, prefix)

    assert not list(tmp_path.glob("out*"))
