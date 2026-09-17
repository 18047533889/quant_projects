import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from factor_engine.tools.catalog_r18_gaussian_corrections import (
    APPROVED_GAUSSIAN_CORRECTIONS,
    correct_gaussian_row,
)


CATALOG = Path("evidence/factor_catalog_20260916/factor_catalog_review_r17c.csv.gz")


def _approved_rows():
    rows = {}
    with gzip.open(CATALOG, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            source_row = int(row.get("source_row") or row["\ufeffsource_row"])
            if source_row in APPROVED_GAUSSIAN_CORRECTIONS:
                rows[source_row] = row
    assert set(rows) == set(APPROVED_GAUSSIAN_CORRECTIONS)
    return rows


def test_three_pinned_rows_withdraw_only_gaussian_assumption_and_invalidate_evidence():
    rows = _approved_rows()
    corrected = [correct_gaussian_row(row) for row in rows.values()]
    assert len(corrected) == 3
    assert all("cs_rank_gaussian" in row["current_formula"] for row in corrected)
    assert all("'blom'" not in row["current_formula"] for row in corrected)
    assert all(row["current_formula"].count(", 3.0)") == 1 for row in corrected)
    assert all(row["compile_status"] == "NOT_RUN" for row in corrected)
    assert all(row["execution_status"] == "NOT_RUN" for row in corrected)
    assert all(not row["compile_evidence_file"] for row in corrected)
    assert all(not row["execution_evidence_file"] for row in corrected)


def test_other_edits_and_existing_history_are_preserved():
    row = _approved_rows()[26552]
    before = json.loads(row["migration_changes"])
    corrected = correct_gaussian_row(row)
    after = json.loads(corrected["migration_changes"])
    assert after[:-1] == before
    assert "price_impact(ret, amount, 20)" in corrected["current_formula"]
    assert "amihud_illiquidity" not in corrected["current_formula"]
    assert "prior_compile_status" in after[-1]
    assert "prior_execution_status" in after[-1]


def test_unapproved_row_is_noop_and_pin_mismatch_fails_closed():
    row = _approved_rows()[26549]
    other = dict(row)
    other["source_row"] = "1"
    assert correct_gaussian_row(other) == other
    tampered = dict(row)
    tampered["current_formula"] += " "
    with pytest.raises(ValueError, match="identity/hash mismatch"):
        correct_gaussian_row(tampered)


def test_multiline_unicode_span_and_call_count_are_guarded(monkeypatch):
    formula = "add(field('中文'),\n cs_rank_gaussian(ret, 'blom'))"
    digest = hashlib.sha256(formula.encode()).hexdigest()
    monkeypatch.setitem(APPROVED_GAUSSIAN_CORRECTIONS, 999999, ("unicode", digest, 1))
    row = {"source_row": "999999", "id": "unicode", "current_formula": formula,
           "migration_changes": "[]", "compile_status": "COMPILED", "execution_status": "EXECUTED"}
    assert correct_gaussian_row(row)["current_formula"] == "add(field('中文'),\n cs_rank_gaussian(ret, 3.0))"
