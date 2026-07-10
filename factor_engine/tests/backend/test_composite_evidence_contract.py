# -*- coding: utf-8
"""Composite evidence manifest 与测试覆盖一致（禁止白名单漂移）。"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def test_composite_reference_evidence_matches_reference_cases(_loaded):
    from backend.composite_evidence import COMPOSITE_REFERENCE_TO_LOWERED_PANDAS_VERIFIED
    from tests.backend_parity.composite_reference_helpers import COMPOSITE_REFERENCE_CASES

    tested = frozenset(c.canon for c in COMPOSITE_REFERENCE_CASES)
    assert COMPOSITE_REFERENCE_TO_LOWERED_PANDAS_VERIFIED == tested


def test_composite_polars_evidence_matches_triple_parity_cases(_loaded):
    from backend.composite_evidence import COMPOSITE_LOWERED_POLARS_VERIFIED
    from tests.backend_parity.test_composite_lowered_triple_parity import COMPOSITE_TRIPLE_PARITY_CASES

    tested = frozenset(c["canon"] for c in COMPOSITE_TRIPLE_PARITY_CASES)
    assert COMPOSITE_LOWERED_POLARS_VERIFIED == tested


def test_composite_duckdb_evidence_matches_triple_parity_cases(_loaded):
    from backend.composite_evidence import COMPOSITE_LOWERED_DUCKDB_VERIFIED
    from tests.backend_parity.test_composite_lowered_triple_parity import COMPOSITE_TRIPLE_PARITY_CASES

    tested = frozenset(c["canon"] for c in COMPOSITE_TRIPLE_PARITY_CASES)
    assert COMPOSITE_LOWERED_DUCKDB_VERIFIED == tested


def test_composite_edge_evidence_matches_edge_cases(_loaded):
    from backend.composite_evidence import COMPOSITE_EDGE_VERIFIED
    from tests.backend_parity.composite_edge_helpers import COMPOSITE_EDGE_CASES

    tested = frozenset(c.canon for c in COMPOSITE_EDGE_CASES)
    assert COMPOSITE_EDGE_VERIFIED == tested


def test_composite_edge_evidence_is_subset_of_reference(_loaded):
    from backend.composite_evidence import (
        COMPOSITE_EDGE_VERIFIED,
        COMPOSITE_REFERENCE_TO_LOWERED_PANDAS_VERIFIED,
    )

    assert COMPOSITE_EDGE_VERIFIED <= COMPOSITE_REFERENCE_TO_LOWERED_PANDAS_VERIFIED


def test_composite_full_parity_is_intersection(_loaded):
    from backend.composite_evidence import (
        COMPOSITE_FULL_PARITY_VERIFIED,
        COMPOSITE_LOWERED_DUCKDB_VERIFIED,
        COMPOSITE_LOWERED_POLARS_VERIFIED,
        COMPOSITE_REFERENCE_TO_LOWERED_PANDAS_VERIFIED,
    )

    expected = (
        COMPOSITE_REFERENCE_TO_LOWERED_PANDAS_VERIFIED
        & COMPOSITE_LOWERED_POLARS_VERIFIED
        & COMPOSITE_LOWERED_DUCKDB_VERIFIED
    )
    assert COMPOSITE_FULL_PARITY_VERIFIED == expected
