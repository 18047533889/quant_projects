# -*- coding: utf-8
"""Primitive evidence：case registry 与 verified artifact 契约。"""
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


def _case_registry_sets():
    from backend.primitive_evidence import (
        CASE_REGISTRY_DUCKDB_EDGE,
        CASE_REGISTRY_DUCKDB_NAN_EDGE,
        CASE_REGISTRY_DUCKDB_REFERENCE,
        CASE_REGISTRY_NO_FALLBACK,
        CASE_REGISTRY_POLARS_EDGE,
        CASE_REGISTRY_POLARS_REFERENCE,
        CASE_REGISTRY_SIX_WAY,
    )

    return {
        "polars_reference": CASE_REGISTRY_POLARS_REFERENCE,
        "duckdb_reference": CASE_REGISTRY_DUCKDB_REFERENCE,
        "polars_edge": CASE_REGISTRY_POLARS_EDGE,
        "duckdb_edge": CASE_REGISTRY_DUCKDB_EDGE,
        "duckdb_nan_edge": CASE_REGISTRY_DUCKDB_NAN_EDGE,
        "no_fallback": CASE_REGISTRY_NO_FALLBACK,
        "six_way": CASE_REGISTRY_SIX_WAY,
    }


def test_primitive_case_registry_matches_cases(_loaded):
    from scripts.sync_primitive_evidence import _collect_cases

    expected = _collect_cases()
    expected.pop("_six_way_count", None)
    reg = _case_registry_sets()
    assert reg["polars_reference"] == frozenset(expected["polars_reference_parity"])
    assert reg["duckdb_reference"] == frozenset(expected["duckdb_reference_parity"])
    assert reg["polars_edge"] == frozenset(expected["polars_edge_verified"])
    assert reg["duckdb_edge"] == frozenset(expected["duckdb_edge_verified"])
    assert reg["no_fallback"] == frozenset(expected["no_fallback_verified"])


def test_primitive_case_registry_six_way_intersection(_loaded):
    reg = _case_registry_sets()
    expected = (
        reg["polars_reference"]
        & reg["polars_edge"]
        & reg["duckdb_reference"]
        & (reg["duckdb_edge"] | reg["duckdb_nan_edge"])
        & reg["no_fallback"]
    )
    assert reg["six_way"] == expected


def test_primitive_verified_subset_of_case_registry(_loaded):
    from backend.primitive_evidence import (
        DUCKDB_EDGE_VERIFIED,
        DUCKDB_REAL_SQL_VERIFIED,
        DUCKDB_REFERENCE_PARITY_VERIFIED,
        NO_FALLBACK_VERIFIED,
        POLARS_EDGE_VERIFIED,
        POLARS_REFERENCE_PARITY_VERIFIED,
        PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE,
    )

    reg = _case_registry_sets()
    assert POLARS_REFERENCE_PARITY_VERIFIED <= reg["polars_reference"]
    assert DUCKDB_REFERENCE_PARITY_VERIFIED <= reg["duckdb_reference"]
    assert DUCKDB_REAL_SQL_VERIFIED <= reg["duckdb_reference"]
    assert POLARS_EDGE_VERIFIED <= reg["polars_edge"]
    assert DUCKDB_EDGE_VERIFIED <= reg["duckdb_edge"]
    assert NO_FALLBACK_VERIFIED <= reg["no_fallback"]
    assert PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE <= reg["six_way"]


def test_verified_artifact_has_provenance_when_present(_loaded):
    from pathlib import Path

    from backend.evidence_provenance import evidence_artifact_valid, load_verified_artifact

    path = Path(__file__).resolve().parents[2] / "evidence" / "primitive_verified.json"
    if not path.is_file():
        pytest.skip("primitive_verified.json 尚未生成")
    data = load_verified_artifact()
    prov = data.get("provenance") or {}
    if evidence_artifact_valid():
        assert prov.get("artifact_kind") == "test_passed"
        assert prov.get("commit_sha")
        assert prov.get("passed_at")
        assert prov.get("case_registry_hash")


def test_sql_production_safe_subset_of_primitive_evidence(_loaded):
    from backend.primitive_evidence import PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
    from backend.sql_tiers import SQL_PRODUCTION_SAFE_CANONICALS

    meta = {"column", "literal"}
    assert SQL_PRODUCTION_SAFE_CANONICALS - meta <= PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
