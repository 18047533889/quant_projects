import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from reconcile_r13_compile_chunks import main


FIELDS = [
    "source_row", "id", "original_formula", "current_formula",
    "static_status", "static_reason", "compile_status", "compile_reason",
    "compile_evidence_file", "compile_validation_scope", "execution_status",
    "execution_reason", "result_hash", "execution_evidence_file",
    "execution_validation_scope",
]


def _checkpoint(path: Path, changed: bool = False) -> str:
    rows = []
    for source_row, name in ((2, "OBV"), (3, "AROON"), (4, "CCI")):
        formula = f"{name}(x)"
        rows.append({
            "source_row": str(source_row), "id": f"f{source_row}",
            "original_formula": formula,
            "current_formula": "obv(x)" if changed and source_row == 2 else formula,
            "static_status": "PARSED_FIELDS_BOUND", "static_reason": "keep-static",
            "compile_status": "COMPILE_FAILED",
            "compile_reason": f"Unsupported function: {name}",
            "execution_status": "NOT_RUN", "execution_reason": "keep-execution",
            "result_hash": "keep-result",
        })
    with gzip.open(path, "wt", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**dict.fromkeys(FIELDS, ""), **row})
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chunk(tmp_path: Path, source_sha: str, duplicate: bool = False):
    evidence = tmp_path / "chunk.jsonl.gz"
    records = []
    for source_row, name in ((2, "OBV"), (3, "AROON"), (4, "CCI")):
        records.append({
            "source_row": source_row, "id": f"f{source_row}",
            "current_formula": f"{name}(x)", "status": "COMPILED",
            "error": "", "code_hash": "code-r13",
        })
    if duplicate:
        records.append(dict(records[-1]))
    with gzip.open(evidence, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    bindings = tmp_path / "chunk.bindings.jsonl.gz"
    with gzip.open(bindings, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps({"binding": "fixture"}) + "\n")
    manifest = tmp_path / "chunk.manifest.json"
    manifest.write_text(json.dumps({
        "bindings": bindings.name,
        "bindings_sha256": hashlib.sha256(bindings.read_bytes()).hexdigest(),
        "checkpoint_sha256": source_sha,
        "code_hash": "code-r13",
        "evidence": evidence.name,
        "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        "records": len(records),
        "scope": "current_code_no_read_compile_only_not_execution",
    }), encoding="utf-8")
    return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()


def _run(tmp_path: Path, source: Path, source_sha: str, checkpoint: Path,
         checkpoint_sha: str, manifest: Path, manifest_sha: str):
    return main([
        "--source-checkpoint", str(source), "--expected-source-sha", source_sha,
        "--checkpoint", str(checkpoint), "--expected-checkpoint-sha", checkpoint_sha,
        "--chunk-manifest", str(manifest),
        "--expected-chunk-manifest-sha", manifest_sha,
        "--expected-code-hash", "code-r13",
        "--expected-targets", "3", "--expected-excluded", "1",
        "--aggregate-evidence", str(tmp_path / "aggregate.jsonl.gz"),
        "--output-prefix", str(tmp_path / "r13b"),
    ])


def test_exact_coverage_excludes_changed_formula_and_reuses_compile_reconciler(tmp_path):
    source = tmp_path / "r12e.csv.gz"
    source_sha = _checkpoint(source)
    checkpoint = tmp_path / "r13a.csv.gz"
    checkpoint_sha = _checkpoint(checkpoint, changed=True)
    manifest, manifest_sha = _chunk(tmp_path, source_sha)
    result = _run(tmp_path, source, source_sha, checkpoint, checkpoint_sha, manifest, manifest_sha)

    assert result["aggregation"]["target_records"] == 3
    assert result["aggregation"]["included_records"] == 2
    assert result["aggregation"]["excluded_records"] == 1
    assert result["merged_records"] == 2
    with gzip.open(tmp_path / "r13b.csv.gz", "rt", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    excluded, *merged = rows
    assert excluded["current_formula"] == "obv(x)"
    assert excluded["compile_status"] == "COMPILE_FAILED"
    assert excluded["execution_reason"] == "keep-execution"
    assert all(row["compile_status"] == "COMPILED" for row in merged)
    assert all(row["static_reason"] == "keep-static" for row in rows)
    assert all(row["result_hash"] == "keep-result" for row in rows)


def test_duplicate_chunk_identity_publishes_nothing(tmp_path):
    source = tmp_path / "r12e.csv.gz"
    source_sha = _checkpoint(source)
    checkpoint = tmp_path / "r13a.csv.gz"
    checkpoint_sha = _checkpoint(checkpoint, changed=True)
    manifest, manifest_sha = _chunk(tmp_path, source_sha, duplicate=True)
    with pytest.raises(ValueError, match="duplicate"):
        _run(tmp_path, source, source_sha, checkpoint, checkpoint_sha, manifest, manifest_sha)
    assert not (tmp_path / "aggregate.jsonl.gz").exists()
    assert not list(tmp_path.glob("r13b*"))


def test_unapproved_manifest_hash_publishes_nothing(tmp_path):
    source = tmp_path / "r12e.csv.gz"
    source_sha = _checkpoint(source)
    checkpoint = tmp_path / "r13a.csv.gz"
    checkpoint_sha = _checkpoint(checkpoint, changed=True)
    manifest, _ = _chunk(tmp_path, source_sha)
    with pytest.raises(ValueError, match="manifest SHA256"):
        _run(tmp_path, source, source_sha, checkpoint, checkpoint_sha, manifest, "0" * 64)
    assert not (tmp_path / "aggregate.jsonl.gz").exists()
    assert not list(tmp_path.glob("r13b*"))


def test_wrong_expected_excluded_count_publishes_nothing(tmp_path):
    source = tmp_path / "r12e.csv.gz"
    source_sha = _checkpoint(source)
    checkpoint = tmp_path / "r13a.csv.gz"
    checkpoint_sha = _checkpoint(checkpoint, changed=True)
    manifest, manifest_sha = _chunk(tmp_path, source_sha)
    argv = [
        "--source-checkpoint", str(source), "--expected-source-sha", source_sha,
        "--checkpoint", str(checkpoint), "--expected-checkpoint-sha", checkpoint_sha,
        "--chunk-manifest", str(manifest),
        "--expected-chunk-manifest-sha", manifest_sha,
        "--expected-code-hash", "code-r13",
        "--expected-targets", "3", "--expected-excluded", "0",
        "--aggregate-evidence", str(tmp_path / "aggregate.jsonl.gz"),
        "--output-prefix", str(tmp_path / "r13b"),
    ]
    with pytest.raises(ValueError, match="excluded count"):
        main(argv)
    assert not (tmp_path / "aggregate.jsonl.gz").exists()
    assert not list(tmp_path.glob("r13b*"))
