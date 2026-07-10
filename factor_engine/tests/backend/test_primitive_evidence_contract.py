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


def test_primitive_polars_reference_matches_memory_cases(_loaded):
    from backend.primitive_evidence import POLARS_REFERENCE_PARITY_VERIFIED
    from tests.backend_parity.test_production_core_triple_parity import MEMORY_CASES

    tested = frozenset(name for name, _ in MEMORY_CASES)
    assert POLARS_REFERENCE_PARITY_VERIFIED == tested


def test_primitive_duckdb_reference_matches_duckdb_cases(_loaded):
    from backend.primitive_evidence import DUCKDB_REFERENCE_PARITY_VERIFIED
    from tests.backend_parity.test_production_core_triple_parity import DUCKDB_CASES

    tested = frozenset(name for name, _ in DUCKDB_CASES)
    assert DUCKDB_REFERENCE_PARITY_VERIFIED == tested


def test_primitive_duckdb_real_sql_matches_duckdb_cases(_loaded):
    from backend.primitive_evidence import DUCKDB_REAL_SQL_VERIFIED
    from tests.backend_parity.test_production_core_triple_parity import DUCKDB_CASES

    tested = frozenset(name for name, _ in DUCKDB_CASES)
    assert DUCKDB_REAL_SQL_VERIFIED == tested


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


def test_primitive_no_fallback_matches_no_pandas_cases(_loaded):
    from backend.primitive_evidence import NO_FALLBACK_VERIFIED
    from tests.backend_parity.test_polars_long_no_pandas_path import NO_PANDAS_CASES

    tested = frozenset(name for name, _ in NO_PANDAS_CASES)
    assert NO_FALLBACK_VERIFIED == tested


def test_primitive_polars_edge_matches_edge_cases(_loaded):
    from backend.primitive_evidence import POLARS_EDGE_VERIFIED
    from tests.backend_parity.test_p0_edge_cases_triple_parity import EDGE_CASES

    tested = frozenset(name for name, _ in EDGE_CASES)
    assert POLARS_EDGE_VERIFIED == tested


def test_primitive_duckdb_edge_matches_duckdb_edge_cases(_loaded):
    from backend.primitive_evidence import DUCKDB_EDGE_VERIFIED
    from tests.backend_parity.test_p0_edge_cases_triple_parity import DUCKDB_EDGE_CASES

    tested = frozenset(name for name, _ in DUCKDB_EDGE_CASES)
    assert DUCKDB_EDGE_VERIFIED == tested


def test_sql_production_safe_subset_of_primitive_evidence(_loaded):
    from backend.primitive_evidence import PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
    from backend.sql_tiers import SQL_PRODUCTION_SAFE_CANONICALS

    meta = {"column", "literal"}
    assert SQL_PRODUCTION_SAFE_CANONICALS - meta <= PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
