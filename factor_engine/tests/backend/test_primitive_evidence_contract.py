# -*- coding: utf-8
"""Primitive evidence JSON 与 parity 测试 case 一致。"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def _merged_cases():
    from tests.backend_parity.evidence_case_registry import merge_case_lists
    from tests.backend_parity.test_p0_edge_cases_triple_parity import DUCKDB_EDGE_CASES, EDGE_CASES
    from tests.backend_parity.test_polars_long_no_pandas_path import NO_PANDAS_CASES
    from tests.backend_parity.test_production_core_triple_parity import DUCKDB_CASES, MEMORY_CASES
    from tests.backend_parity.test_production_safe_bulk_parity import DUCKDB_BULK_CASES, POLARS_BULK_CASES

    return {
        "polars_reference": merge_case_lists(MEMORY_CASES, POLARS_BULK_CASES),
        "duckdb_reference": merge_case_lists(DUCKDB_CASES, DUCKDB_BULK_CASES),
        "polars_edge": merge_case_lists(EDGE_CASES),
        "duckdb_edge": merge_case_lists(DUCKDB_EDGE_CASES),
        "no_fallback": merge_case_lists(NO_PANDAS_CASES),
    }


def test_primitive_polars_reference_matches_cases(_loaded):
    from backend.primitive_evidence import POLARS_REFERENCE_PARITY_VERIFIED
    from tests.backend_parity.evidence_case_registry import case_names

    cases = _merged_cases()
    assert POLARS_REFERENCE_PARITY_VERIFIED == case_names(cases["polars_reference"])


def test_primitive_duckdb_reference_matches_cases(_loaded):
    from backend.primitive_evidence import DUCKDB_REFERENCE_PARITY_VERIFIED
    from tests.backend_parity.evidence_case_registry import case_names

    cases = _merged_cases()
    assert DUCKDB_REFERENCE_PARITY_VERIFIED == case_names(cases["duckdb_reference"])


def test_primitive_duckdb_real_sql_matches_cases(_loaded):
    from backend.primitive_evidence import DUCKDB_REAL_SQL_VERIFIED
    from tests.backend_parity.evidence_case_registry import case_names

    cases = _merged_cases()
    assert DUCKDB_REAL_SQL_VERIFIED == case_names(cases["duckdb_reference"])


def test_primitive_dual_backend_is_six_way_intersection(_loaded):
    from backend.primitive_evidence import (
        DUCKDB_EDGE_VERIFIED,
        DUCKDB_REAL_SQL_VERIFIED,
        DUCKDB_REFERENCE_PARITY_VERIFIED,
        NO_FALLBACK_VERIFIED,
        POLARS_EDGE_VERIFIED,
        POLARS_REFERENCE_PARITY_VERIFIED,
        PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE,
    )

    expected = (
        POLARS_REFERENCE_PARITY_VERIFIED
        & POLARS_EDGE_VERIFIED
        & DUCKDB_REFERENCE_PARITY_VERIFIED
        & DUCKDB_REAL_SQL_VERIFIED
        & DUCKDB_EDGE_VERIFIED
        & NO_FALLBACK_VERIFIED
    )
    assert PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE == expected


def test_primitive_no_fallback_matches_cases(_loaded):
    from backend.primitive_evidence import NO_FALLBACK_VERIFIED
    from tests.backend_parity.evidence_case_registry import case_names

    cases = _merged_cases()
    assert NO_FALLBACK_VERIFIED == case_names(cases["no_fallback"])


def test_primitive_polars_edge_matches_cases(_loaded):
    from backend.primitive_evidence import POLARS_EDGE_VERIFIED
    from tests.backend_parity.evidence_case_registry import case_names

    cases = _merged_cases()
    assert POLARS_EDGE_VERIFIED == case_names(cases["polars_edge"])


def test_primitive_duckdb_edge_matches_cases(_loaded):
    from backend.primitive_evidence import DUCKDB_EDGE_VERIFIED
    from tests.backend_parity.evidence_case_registry import case_names

    cases = _merged_cases()
    assert DUCKDB_EDGE_VERIFIED == case_names(cases["duckdb_edge"])


def test_sql_production_safe_subset_of_primitive_evidence(_loaded):
    from backend.primitive_evidence import PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
    from backend.sql_tiers import SQL_PRODUCTION_SAFE_CANONICALS

    meta = {"column", "literal"}
    assert SQL_PRODUCTION_SAFE_CANONICALS - meta <= PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
