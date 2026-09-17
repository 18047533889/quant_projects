"""Rewritten formulas must never inherit successful execution evidence."""
import pytest


def _row():
    return dict(source_row="12", id="factor", original_formula="OLD(close)",
                current_formula="OLD(close)", migration_changes="[]",
                compile_status="COMPILED", compile_reason="",
                execution_status="EXECUTED", execution_reason="",
                value_count="5", finite_count="5", result_hash="abc",
                execution_evidence_file="old.gz",
                execution_validation_scope="old",
                compile_evidence_file="old_compile.gz",
                compile_validation_scope="old", backend_path="pandas",
                execution_window='["2025-01-01","2025-02-01"]',
                execution_symbol_count="8")


def test_rewrite_invalidates_old_compile_and_execution_without_mutating_input():
    from factor_engine.tools.catalog_review_evidence import apply_reviewed_formula
    row = _row()
    actual = apply_reviewed_formula(row, "NEW(close)", ("explicit redesign",))
    assert row["execution_status"] == "EXECUTED"
    assert actual["original_formula"] == "OLD(close)"
    assert actual["current_formula"] == "NEW(close)"
    assert actual["execution_status"] == actual["compile_status"] == "NOT_RUN"
    for key in ("result_hash", "value_count", "finite_count", "execution_evidence_file",
                "compile_evidence_file", "backend_path", "execution_window",
                "execution_symbol_count"):
        assert actual[key] == ""
    assert "explicit redesign" in actual["migration_changes"]


def test_no_formula_change_preserves_evidence_and_deduplicates_notes():
    from factor_engine.tools.catalog_review_evidence import apply_reviewed_formula
    row = _row()
    first = apply_reviewed_formula(row, row["current_formula"], ("checked",))
    second = apply_reviewed_formula(first, row["current_formula"], ("checked",))
    assert second["execution_status"] == "EXECUTED"
    assert second["result_hash"] == "abc"
    assert second["migration_changes"] == '["checked"]'


def test_malformed_change_ledger_fails_instead_of_discarding_history():
    from factor_engine.tools.catalog_review_evidence import apply_reviewed_formula
    row = _row()
    row["migration_changes"] = "not json"
    with pytest.raises(ValueError):
        apply_reviewed_formula(row, "NEW(close)", ("change",))


def test_empty_formula_is_not_a_successful_repair():
    from factor_engine.tools.catalog_review_evidence import apply_reviewed_formula
    with pytest.raises(ValueError):
        apply_reviewed_formula(_row(), " ", ("change",))


def test_unchanged_native_crash_is_not_resurrected():
    from factor_engine.tools.catalog_review_evidence import apply_reviewed_formula
    row = _row()
    row.update(execution_status="NATIVE_CRASH", result_hash="",
               execution_validation_scope="post_donchian_macro_fix_native_crash_no_result")
    actual = apply_reviewed_formula(row, row["current_formula"], ())
    assert actual == row
    assert (actual["source_row"], actual["id"], actual["original_formula"]) == (
        "12", "factor", "OLD(close)"
    )


def _source(tmp_path, status="NATIVE_CRASH", scope="post_donchian_macro_fix_native_crash_no_result"):
    import csv, gzip, hashlib
    path = tmp_path / "source.csv.gz"
    row = _row()
    row.update(original_formula="donchian_position(close,high,low,20)",
               execution_status=status, execution_validation_scope=scope)
    with gzip.open(path, "wt", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_anchor_rejects_mismatched_digest(tmp_path):
    from factor_engine.tools.catalog_review_evidence import validate_review_source
    path, digest = _source(tmp_path)
    assert validate_review_source(path, digest)["rows"] == 1
    with pytest.raises(ValueError, match="SHA"):
        validate_review_source(path, "0" * 64)


def test_source_anchor_rejects_stale_donchian_success(tmp_path):
    from factor_engine.tools.catalog_review_evidence import validate_review_source
    path, digest = _source(tmp_path, status="EXECUTED",
                          scope="historical_formula_matched_not_current_code_certification")
    with pytest.raises(ValueError, match="Donchian"):
        validate_review_source(path, digest)


def test_truncated_source_gzip_is_rejected_even_with_matching_digest(tmp_path):
    import hashlib
    from factor_engine.tools.catalog_review_evidence import validate_review_source
    path, _ = _source(tmp_path)
    path.write_bytes(path.read_bytes()[:-8])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(EOFError):
        validate_review_source(path, digest)
