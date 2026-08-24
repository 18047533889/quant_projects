"""
Unit tests for the metric coverage compiler.

Verifies that the compiler actually scans real code (it never hand-writes a
catalog claim), produces a deterministic, sorted coverage matrix CSV with the
expected header, and that the prod_status mapping is internally consistent
(kernel YES + registry YES + artifact + test + report => READY; kernel NO =>
NOT_IMPL; otherwise GAP).
"""

import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_evaluator.metrics.coverage_compiler import (
    compile_coverage_rows,
    compile_metric_coverage,
    _pkg_root,
    _resolve_implementation,
)


def test_resolve_implementation_real_kernel():
    """The compiler must resolve a real catalog kernel to True."""
    ok, detail = _resolve_implementation(
        "quant_evaluator.metrics.ic.compute_daily_ic"
    )
    assert ok is True, detail
    ok, _ = _resolve_implementation("quant_evaluator.metrics.ic.compute_daily_ic")
    assert ok


def test_resolve_implementation_fails_closed_on_missing():
    ok, detail = _resolve_implementation("quant_evaluator.metrics.ic.does_not_exist_fn")
    assert ok is False
    ok, _ = _resolve_implementation("not_a_real_module.compute_x")
    assert ok is False


def test_compile_rows_cover_all_catalog_metrics():
    """One row per catalog metric (the compiler scans the real catalog)."""
    from quant_evaluator.metrics.catalog import list_all_metric_ids

    rows = compile_coverage_rows()
    row_ids = {r["metric"] for r in rows}
    catalog_ids = set(list_all_metric_ids())
    assert row_ids == catalog_ids
    assert len(rows) == len(catalog_ids)


def test_rows_have_expected_columns():
    rows = compile_coverage_rows()
    expected = {"metric", "kernel", "registry", "artifact", "test", "report", "prod_status"}
    for row in rows:
        assert set(row.keys()) == expected


def test_prod_status_mapping_consistent():
    """kernel NO => NOT_IMPL; all five YES => READY; anything else => GAP."""
    rows = compile_coverage_rows()
    for row in rows:
        if row["kernel"] == "NO":
            assert row["prod_status"] == "NOT_IMPL", row
        elif all(row[c] == "YES" for c in ("kernel", "registry", "artifact", "test", "report")):
            assert row["prod_status"] == "READY", row
        else:
            assert row["prod_status"] == "GAP", row


def test_compiler_writes_deterministic_csv():
    """The CSV is sorted by metric and has the expected header + row count."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "METRIC_COVERAGE_COMPILER.csv")
        compile_metric_coverage(out)
        with open(out, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames == [
                "metric", "kernel", "registry", "artifact", "test", "report", "prod_status"
            ]
            rows = list(reader)
        ids = [r["metric"] for r in rows]
        assert ids == sorted(ids)
        # header + one row per metric
        from quant_evaluator.metrics.catalog import list_all_metric_ids

        assert len(rows) == len(list_all_metric_ids())


def test_pkg_root_points_at_quant_evaluator():
    root = _pkg_root()
    assert os.path.basename(root) == "quant_evaluator"
