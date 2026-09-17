import csv
import json
from test_export_review import packed
from export_review import export_review


def test_backend_provenance_is_retained_but_not_current_code_certification(tmp_path):
    row = {"source_row": 2, "id": "A", "formula": "close"}
    source = packed(tmp_path / "input.gz", [row])
    audit = packed(tmp_path / "audit.gz", [dict(row, current_formula="close", status="PARSED_FIELDS_BOUND")])
    route = {"primary_route": "pandas", "used_sql_pushdown": False}
    smoke = packed(tmp_path / "smoke.gz", [
        dict(row, executed_formula="close", status="EXECUTED", backend_path=route)
    ])
    output = tmp_path / "review.csv"
    export_review(source, audit, output, [smoke])
    with output.open(encoding="utf-8-sig") as stream:
        exported = next(csv.DictReader(stream))
    assert json.loads(exported["backend_path"]) == route
    assert exported["execution_evidence_file"] == "smoke.gz"
    assert exported["execution_validation_scope"] == "historical_formula_matched_not_current_code_certification"
