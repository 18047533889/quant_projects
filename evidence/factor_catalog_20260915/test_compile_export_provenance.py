import csv
from test_export_review import packed
from export_review import export_review


def test_compile_evidence_remains_explicitly_historical(tmp_path):
    row = {"source_row": 2, "id": "A", "formula": "close"}
    source = packed(tmp_path / "input.gz", [row])
    audit = packed(tmp_path / "audit.gz", [dict(row, current_formula="close", status="PARSED_FIELDS_BOUND")])
    compiled = packed(tmp_path / "compile.gz", [dict(row, current_formula="close", status="COMPILED")])
    output = tmp_path / "review.csv"
    export_review(source, audit, output, compile_paths=[compiled])
    with output.open(encoding="utf-8-sig") as stream:
        result = next(csv.DictReader(stream))
    assert result["compile_status"] == "COMPILED"
    assert result["compile_evidence_file"] == "compile.gz"
    assert result["compile_validation_scope"] == "historical_formula_matched_not_current_code_certification"
