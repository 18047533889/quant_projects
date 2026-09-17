import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from factor_engine.tools.catalog_r17_transport_corrections import (
    APPROVED_TRANSPORT_CORRECTIONS,
    correct_transport_row,
)


CATALOG = Path("evidence/factor_catalog_20260916/factor_catalog_review_r16a.csv.gz")


def _approved_rows():
    rows = {}
    with gzip.open(CATALOG, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            source_row = int(row.get("source_row") or row["\ufeffsource_row"])
            if source_row in APPROVED_TRANSPORT_CORRECTIONS:
                rows[source_row] = row
    assert set(rows) == set(APPROVED_TRANSPORT_CORRECTIONS)
    return rows


def test_all_30_pinned_rows_correct_exactly_32_calls_and_invalidate_evidence():
    corrected = [correct_transport_row(row) for row in _approved_rows().values()]
    assert len(corrected) == 30
    assert sum(row["current_formula"].count("window=60") for row in corrected) >= 32
    assert all("recent_window=20, old_window=40" not in row["current_formula"] for row in corrected)
    assert all(row["compile_status"] == "NOT_RUN" for row in corrected)
    assert all(row["execution_status"] == "NOT_RUN" for row in corrected)
    assert all(not row["compile_evidence_file"] for row in corrected)
    assert all(not row["execution_evidence_file"] for row in corrected)
    assert all("CORRECTION ts_quantile_transport" in row["migration_changes"] for row in corrected)


def test_other_migrations_are_preserved_and_old_change_is_retained_as_history():
    row = _approved_rows()[55174]
    before = json.loads(row["migration_changes"])
    corrected = correct_transport_row(row)
    after = json.loads(corrected["migration_changes"])
    assert after[:-1] == before
    assert "ts_cross_quantilogram.target_q" in corrected["migration_changes"]
    assert "prior_compile_status" in after[-1]
    assert "prior_execution_status" in after[-1]


def test_two_sibling_calls_are_both_corrected():
    corrected = correct_transport_row(_approved_rows()[66532])
    assert corrected["current_formula"].count("window=60") == 2
    assert "recent_window=20" not in corrected["current_formula"]


def test_unapproved_row_is_noop_and_pinned_hash_mismatch_fails_closed():
    row = _approved_rows()[54777]
    other = dict(row)
    other["\ufeffsource_row"] = "1"
    assert correct_transport_row(other) == other
    tampered = dict(row)
    tampered["current_formula"] += " "
    with pytest.raises(ValueError, match="identity/hash mismatch"):
        correct_transport_row(tampered)


def test_unicode_prefix_and_multiline_formula_use_ast_byte_accurate_spans(monkeypatch):
    formula = (
        "add(field('中文字段', table='Demo'),\n"
        "    ts_quantile_transport_slope(\n"
        "        ret, recent_window=20, old_window=40))"
    )
    digest = hashlib.sha256(formula.encode("utf-8")).hexdigest()
    monkeypatch.setitem(
        APPROVED_TRANSPORT_CORRECTIONS, 999999, ("unicode_multiline", digest, 1),
    )
    row = {
        "source_row": "999999", "id": "unicode_multiline",
        "current_formula": formula, "migration_changes": "[]",
        "compile_status": "COMPILED", "execution_status": "EXECUTED",
    }
    corrected = correct_transport_row(row)
    assert corrected["current_formula"] == (
        "add(field('中文字段', table='Demo'),\n"
        "    ts_quantile_transport_slope(ret, window=60))"
    )
