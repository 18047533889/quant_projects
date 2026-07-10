# -*- coding: utf-8
"""Fast path coverage report CI 门禁。"""
from __future__ import annotations

import pytest

from backend.fastpath_coverage import build_fastpath_coverage_matrix, summarize_fastpath_coverage
from backend.operator_capability import resolve_canonical
from backend.sql_tiers import SQL_PRODUCTION_SAFE_CANONICALS
from cleaned_operators.operator_spec import PRODUCTION_CORE_CANONICALS


@pytest.fixture(scope="module")
def coverage_rows():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()
    return build_fastpath_coverage_matrix()


def test_production_core_in_coverage_report(coverage_rows):
    by_canon = {r.canonical: r for r in coverage_rows}
    skip = {"column", "literal", "materialized_series", "plan_ref"}
    missing = [
        c for c in PRODUCTION_CORE_CANONICALS if c not in skip and c not in by_canon
    ]
    assert not missing, f"PRODUCTION_CORE 未出现在 coverage matrix: {missing}"


def test_sql_production_safe_subset_emitter_ok(coverage_rows):
    by_canon = {r.canonical: r for r in coverage_rows}
    bad = [
        c
        for c in SQL_PRODUCTION_SAFE_CANONICALS
        if c not in {"column", "literal"} and not by_canon.get(c, by_canon.get(resolve_canonical(c))).sql_emitter_ok
    ]
    assert not bad, f"SQL_PRODUCTION_SAFE emitter 失败: {bad[:15]}"


def test_production_allowed_without_fastpath_has_reason(coverage_rows):
    bad = [
        r.canonical
        for r in coverage_rows
        if r.allow_in_production and not r.production_fast_path and not r.fastpath_block_reason
    ]
    assert not bad, f"production 算子缺少 fastpath_block_reason: {bad[:20]}"


def test_fastpath_summary_non_empty(coverage_rows):
    summary = summarize_fastpath_coverage(coverage_rows)
    assert summary["production_allowed_count"] > 0
    assert summary["production_fast_path_count"] > 0
