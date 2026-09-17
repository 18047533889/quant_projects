import csv
import json
import pytest
from test_export_review import packed
from export_review import export_review


@pytest.mark.parametrize("scope", [
    {"execution_window": ["2025-01-01", "2026-04-30"], "execution_symbol_count": 8},
    {},
])
def test_export_keeps_sample_scope_without_inventing_legacy_scope(tmp_path, scope):
    row = {"source_row": 2, "id": "A", "formula": "close"}
    source = packed(tmp_path / "input.gz", [row])
    audit = packed(tmp_path / "audit.gz", [
        dict(row, current_formula="close", status="PARSED_FIELDS_BOUND")
    ])
    smoke = packed(tmp_path / "smoke.gz", [
        dict(row, executed_formula="close", status="EXECUTED", **scope)
    ])
    output = tmp_path / "review.csv"
    export_review(source, audit, output, [smoke])
    with output.open(encoding="utf-8-sig") as stream:
        exported = next(csv.DictReader(stream))
    if scope:
        assert json.loads(exported["execution_window"]) == ["2025-01-01", "2026-04-30"]
        assert exported["execution_symbol_count"] == "8"
    else:
        assert exported["execution_window"] == ""
        assert exported["execution_symbol_count"] == ""
