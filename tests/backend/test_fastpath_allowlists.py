# -*- coding: utf-8
"""Fastpath allowlist + operator manifest。"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def test_production_fastpath_allowlist_subset_of_production(_loaded):
    from backend.fastpath_allowlists import production_allowlist, production_fastpath_allowlist

    fast = production_fastpath_allowlist()
    prod = production_allowlist()
    assert fast <= prod


def test_operator_manifest_has_fastpath_fields(_loaded):
    from backend.operator_manifest import build_operator_manifest_entry

    entry = build_operator_manifest_entry("ts_mean")
    assert entry["canonical"] == "ts_mean"
    assert "production_fast_path" in entry
    assert "duckdb_sql" in entry
    assert "polars_long" in entry
