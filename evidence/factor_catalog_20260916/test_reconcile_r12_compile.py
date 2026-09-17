import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from reconcile_r12_compile import main


FIELDS = [
    "source_row", "id", "original_formula", "current_formula",
    "static_status", "static_reason", "compile_status", "compile_reason",
    "compile_evidence_file", "compile_validation_scope", "execution_status",
    "execution_reason", "result_hash", "execution_evidence_file",
    "execution_validation_scope",
]


def _checkpoint(path: Path) -> str:
    rows = [
        {
            "source_row": "2", "id": "factor_a", "original_formula": "legacy(x)",
            "current_formula": "fin_net_borrowing_cashflow(x)",
            "static_status": "FIELDS_UNRESOLVED", "static_reason": "keep-static",
            "compile_status": "COMPILE_FAILED", "compile_reason": "old-error",
            "execution_status": "NOT_RUN", "execution_reason": "keep-execution",
            "result_hash": "keep-result", "execution_evidence_file": "keep.jsonl.gz",
            "execution_validation_scope": "keep-scope",
        },
        {
            "source_row": "3", "id": "factor_b", "original_formula": "other(x)",
            "current_formula": "other(x)", "static_status": "PARSED_FIELDS_BOUND",
            "compile_status": "COMPILED", "execution_status": "NATIVE_CRASH",
            "execution_reason": "keep-crash",
        },
    ]
    with gzip.open(path, "wt", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**dict.fromkeys(FIELDS, ""), **row})
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(path: Path, records=None) -> str:
    if records is None:
        records = [{
            "source_row": "2", "id": "factor_a",
            "current_formula": "fin_net_borrowing_cashflow(x)",
            "status": "COMPILED", "error": "", "code_hash": "code-abc",
        }]
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(checkpoint, checkpoint_sha, evidence, evidence_sha, prefix):
    return main([
        "--checkpoint", str(checkpoint),
        "--expected-checkpoint-sha", checkpoint_sha,
        "--evidence", str(evidence),
        "--expected-evidence-sha", evidence_sha,
        "--output-prefix", str(prefix),
    ])


def test_compile_only_merge_changes_only_compile_evidence_fields(tmp_path):
    checkpoint = tmp_path / "source.csv.gz"
    checkpoint_sha = _checkpoint(checkpoint)
    evidence = tmp_path / "compile.jsonl.gz"
    evidence_sha = _evidence(evidence)
    prefix = tmp_path / "out"

    manifest = _run(checkpoint, checkpoint_sha, evidence, evidence_sha, prefix)

    with gzip.open(str(prefix) + ".csv.gz", "rt", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    updated, untouched = rows
    assert updated["compile_status"] == "COMPILED"
    assert updated["compile_reason"] == ""
    assert updated["compile_evidence_file"] == evidence.name
    assert "code-abc" in updated["compile_validation_scope"]
    assert updated["original_formula"] == "legacy(x)"
    assert updated["current_formula"] == "fin_net_borrowing_cashflow(x)"
    assert updated["static_status"] == "FIELDS_UNRESOLVED"
    assert updated["static_reason"] == "keep-static"
    assert updated["execution_status"] == "NOT_RUN"
    assert updated["execution_reason"] == "keep-execution"
    assert updated["result_hash"] == "keep-result"
    assert updated["execution_evidence_file"] == "keep.jsonl.gz"
    assert untouched["execution_status"] == "NATIVE_CRASH"
    assert untouched["execution_reason"] == "keep-crash"
    assert manifest["merged_records"] == 1
    assert manifest["evidence_sha256"] == evidence_sha
    assert manifest["factor_counts"] == {"fin_net_borrowing_cashflow": 1}
    assert manifest["scope"] == "compile_only_formula_matched_not_execution_or_code_certification"


def test_duplicate_evidence_identity_publishes_nothing(tmp_path):
    checkpoint = tmp_path / "source.csv.gz"
    checkpoint_sha = _checkpoint(checkpoint)
    evidence = tmp_path / "compile.jsonl.gz"
    record = {
        "source_row": "2", "id": "factor_a",
        "current_formula": "fin_net_borrowing_cashflow(x)",
        "status": "COMPILED", "error": "", "code_hash": "code-abc",
    }
    evidence_sha = _evidence(evidence, [record, record])
    prefix = tmp_path / "out"

    with pytest.raises(ValueError, match="duplicate"):
        _run(checkpoint, checkpoint_sha, evidence, evidence_sha, prefix)

    assert not list(tmp_path.glob("out*"))


def test_formula_mismatch_publishes_nothing(tmp_path):
    checkpoint = tmp_path / "source.csv.gz"
    checkpoint_sha = _checkpoint(checkpoint)
    evidence = tmp_path / "compile.jsonl.gz"
    evidence_sha = _evidence(evidence, [{
        "source_row": "2", "id": "factor_a", "current_formula": "old(x)",
        "status": "COMPILED", "error": "", "code_hash": "code-abc",
    }])
    prefix = tmp_path / "out"

    with pytest.raises(ValueError, match="formula"):
        _run(checkpoint, checkpoint_sha, evidence, evidence_sha, prefix)

    assert not list(tmp_path.glob("out*"))


def test_wrong_evidence_hash_publishes_nothing(tmp_path):
    checkpoint = tmp_path / "source.csv.gz"
    checkpoint_sha = _checkpoint(checkpoint)
    evidence = tmp_path / "compile.jsonl.gz"
    _evidence(evidence)
    prefix = tmp_path / "out"

    with pytest.raises(ValueError, match="SHA256"):
        _run(checkpoint, checkpoint_sha, evidence, "0" * 64, prefix)

    assert not list(tmp_path.glob("out*"))


def test_truncated_evidence_gzip_publishes_nothing(tmp_path):
    checkpoint = tmp_path / "source.csv.gz"
    checkpoint_sha = _checkpoint(checkpoint)
    evidence = tmp_path / "compile.jsonl.gz"
    _evidence(evidence)
    evidence.write_bytes(evidence.read_bytes()[:-8])
    evidence_sha = hashlib.sha256(evidence.read_bytes()).hexdigest()
    prefix = tmp_path / "out"

    with pytest.raises((EOFError, gzip.BadGzipFile)):
        _run(checkpoint, checkpoint_sha, evidence, evidence_sha, prefix)

    assert not list(tmp_path.glob("out*"))


def test_evidence_changed_after_read_publishes_nothing(tmp_path, monkeypatch):
    import reconcile_r12_compile as module

    checkpoint = tmp_path / "source.csv.gz"
    checkpoint_sha = _checkpoint(checkpoint)
    evidence = tmp_path / "compile.jsonl.gz"
    evidence_sha = _evidence(evidence)
    prefix = tmp_path / "out"
    original_load = module._load_evidence

    def load_then_mutate(path, expected_sha):
        result = original_load(path, expected_sha)
        path.write_bytes(path.read_bytes() + b"changed-after-read")
        return result

    monkeypatch.setattr(module, "_load_evidence", load_then_mutate)
    with pytest.raises(ValueError, match="changed after validation"):
        _run(checkpoint, checkpoint_sha, evidence, evidence_sha, prefix)

    assert not list(tmp_path.glob("out*"))
