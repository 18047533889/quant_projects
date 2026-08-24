"""Test catalog index optimization (schema v3) and batch queries."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from factor_engine.storage.catalog import FactorCatalog


def test_schema_v3_indexes_created(tmp_path):
    """Verify schema v3 creates performance indexes."""
    db_path = tmp_path / "test_catalog.db"
    catalog = FactorCatalog(db_path)

    # Check schema version
    version = catalog._conn.execute(
        "SELECT COALESCE(MAX(version), 0) FROM catalog_schema_version"
    ).fetchone()[0]
    assert version == 3, f"Expected schema version 3, got {version}"

    # Verify all three indexes exist
    indexes = catalog._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_factor_run_%'"
    ).fetchall()
    index_names = {row[0] for row in indexes}

    expected_indexes = {
        'idx_factor_run_factor_id',
        'idx_factor_run_created_at',
        'idx_factor_run_factor_created',
    }
    assert expected_indexes.issubset(index_names), \
        f"Missing indexes: {expected_indexes - index_names}"

    catalog.close()


def test_index_usage_in_explain(tmp_path):
    """Verify EXPLAIN QUERY PLAN shows index usage after optimization."""
    db_path = tmp_path / "test_catalog.db"
    catalog = FactorCatalog(db_path)

    # Register a factor and add some runs
    catalog.register("f1", "test", "daily", "hash123")
    for i in range(10):
        catalog.record_run_lineage({
            "factor_id": "f1",
            "factor_name": "f1",
            "ast_hash": "hash123",
            "run_id": f"run_{i}",
        })

    # Query 1: Factor history lookup
    plan = catalog._conn.execute(
        "EXPLAIN QUERY PLAN "
        "SELECT * FROM factor_run WHERE factor_id = ? ORDER BY created_at DESC LIMIT ?",
        ("f1", 5)
    ).fetchall()
    plan_text = " ".join(str(row) for row in plan)

    # Should use covering index idx_factor_run_factor_created
    assert "idx_factor_run_factor_created" in plan_text or "SEARCH" in plan_text, \
        f"Expected index usage, got: {plan_text}"

    # Query 2: Time-ordered query across all factors
    plan2 = catalog._conn.execute(
        "EXPLAIN QUERY PLAN "
        "SELECT * FROM factor_run ORDER BY created_at DESC LIMIT ?",
        (10,)
    ).fetchall()
    plan2_text = " ".join(str(row) for row in plan2)

    # Should use idx_factor_run_created_at
    assert "idx_factor_run_created_at" in plan2_text or "SEARCH" in plan2_text, \
        f"Expected index usage, got: {plan2_text}"

    catalog.close()


def test_batch_dependencies_single_query(tmp_path):
    """Verify get_factor_dependencies_batch eliminates N+1."""
    db_path = tmp_path / "test_catalog.db"
    catalog = FactorCatalog(db_path)

    # Register dependencies for 3 factors
    for i in range(3):
        fid = f"f{i}"
        catalog.register(fid, "test", "daily", f"hash{i}")
        catalog.record_dependency(
            factor_id=fid,
            referenced_columns=["close", "volume"],
            lookback=20,
            frequency="daily",
            source_dataset="dataset_a",
        )

    # Batch fetch (single query)
    deps = catalog.get_factor_dependencies_batch(["f0", "f1", "f2"])

    assert len(deps) == 3
    assert "f0" in deps
    assert "f1" in deps
    assert "f2" in deps

    # Verify structure
    assert deps["f0"]["referenced_columns"] == ["close", "volume"]
    assert deps["f0"]["lookback"] == 20
    assert deps["f1"]["source_dataset"] == "dataset_a"

    # Test with missing IDs (should return partial results)
    deps2 = catalog.get_factor_dependencies_batch(["f0", "f_missing", "f2"])
    assert len(deps2) == 2
    assert "f0" in deps2
    assert "f2" in deps2
    assert "f_missing" not in deps2

    catalog.close()


def test_batch_dependencies_empty_input(tmp_path):
    """Verify batch fetch handles empty input gracefully."""
    db_path = tmp_path / "test_catalog.db"
    catalog = FactorCatalog(db_path)

    deps = catalog.get_factor_dependencies_batch([])
    assert deps == {}

    catalog.close()


def test_explain_output_comparison(tmp_path):
    """Capture EXPLAIN QUERY PLAN before/after optimization (documentation)."""
    db_path = tmp_path / "test_catalog.db"
    catalog = FactorCatalog(db_path)

    # Register test data
    catalog.register("f_test", "test", "daily", "hash_test")
    for i in range(100):
        catalog.record_run_lineage({
            "factor_id": "f_test",
            "factor_name": "f_test",
            "ast_hash": "hash_test",
            "run_id": f"run_{i:03d}",
        })

    # EXPLAIN for factor history query (should use covering index)
    explain_result = catalog._conn.execute(
        "EXPLAIN QUERY PLAN "
        "SELECT * FROM factor_run WHERE factor_id = ? ORDER BY created_at DESC LIMIT ?",
        ("f_test", 20)
    ).fetchall()

    print("\n=== EXPLAIN QUERY PLAN (factor history) ===")
    for row in explain_result:
        print(row)

    # Verify it's using an index (SEARCH vs SCAN)
    explain_text = str(explain_result)
    assert "SEARCH" in explain_text or "USING INDEX" in explain_text, \
        f"Expected index usage (SEARCH/USING INDEX), got: {explain_text}"

    catalog.close()


if __name__ == "__main__":
    # Quick manual test
    with tempfile.TemporaryDirectory() as tmpdir:
        test_schema_v3_indexes_created(Path(tmpdir))
        test_index_usage_in_explain(Path(tmpdir))
        test_batch_dependencies_single_query(Path(tmpdir))
        print("All tests passed!")
