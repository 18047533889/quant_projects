# -*- coding: utf-8
"""Production fast path 证据链 contract：production-safe 须有真实 parity case。"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def test_polars_production_safe_has_executed_parity(_loaded):
    from backend.fastpath_evidence import missing_polars_parity_evidence

    missing = missing_polars_parity_evidence()
    assert not missing, f"Polars production-safe 缺 parity evidence: {missing[:20]}"


def test_duckdb_production_safe_has_executed_parity(_loaded):
    from backend.fastpath_evidence import missing_duckdb_parity_evidence

    missing = missing_duckdb_parity_evidence()
    assert not missing, f"DuckDB production-safe 缺 SQL execute parity: {missing[:20]}"


def test_evidence_summary_counts(_loaded):
    from backend.fastpath_evidence import evidence_summary

    s = evidence_summary()
    assert s["polars_evidence"] == s["polars_required"]
    assert s["duckdb_evidence"] == s["duckdb_required"]
