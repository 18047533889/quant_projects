import csv
import gzip
import json
import pytest
from export_review import export_review

def packed(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return path

@pytest.mark.parametrize("filename", ["review.csv", "review.csv.gz"])
def test_static_pass_is_not_execution_and_preserves_formula(tmp_path, filename):
    row = {"source_row": 2, "id": "A", "formula": "Close", "name": "中文"}
    source = packed(tmp_path / "input.gz", [row])
    audit = packed(tmp_path / "audit.gz", [dict(row, current_formula="close", status="PARSED_FIELDS_BOUND")])
    output = tmp_path / filename
    counts = export_review(source, audit, output)
    assert counts["execution:NOT_RUN"] == 1
    opener = gzip.open if filename.endswith(".gz") else open
    with opener(output, "rt", encoding="utf-8-sig", newline="") as stream:
        out = next(csv.DictReader(stream))
    assert (out["original_formula"], out["current_formula"], out["name"]) == ("Close", "close", "中文")
    with pytest.raises(FileExistsError):
        export_review(source, audit, output)

def test_old_formula_evidence_requires_rerun(tmp_path):
    row = {"source_row": 2, "id": "A", "formula": "Close"}
    source = packed(tmp_path / "input.gz", [row])
    audit = packed(tmp_path / "audit.gz", [dict(row, current_formula="close", status="PARSED_FIELDS_BOUND")])
    smoke = packed(tmp_path / "smoke.gz", [dict(row, executed_formula="Close", status="EXECUTED")])
    counts = export_review(source, audit, tmp_path / "review.csv", [smoke])
    assert counts["execution:NOT_RUN"] == 1

def test_identity_mismatch_rejected(tmp_path):
    source = packed(tmp_path / "input.gz", [{"source_row": 2, "id": "A", "formula": "close"}])
    audit = packed(tmp_path / "audit.gz", [{"source_row": 3, "id": "A", "formula": "close"}])
    with pytest.raises(ValueError, match="identity"):
        export_review(source, audit, tmp_path / "review.csv")

def test_matching_real_evidence_is_separate(tmp_path):
    row = {"source_row": 2, "id": "A", "formula": "close"}
    source = packed(tmp_path / "input.gz", [row])
    audit = packed(tmp_path / "audit.gz", [dict(row, current_formula="close", status="PARSED_FIELDS_BOUND")])
    smoke = packed(tmp_path / "smoke.gz", [dict(row, executed_formula="close", status="EXECUTED", value_count=4, finite_count=3)])
    counts = export_review(source, audit, tmp_path / "review.csv", [smoke])
    assert counts["execution:EXECUTED"] == 1
