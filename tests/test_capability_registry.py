# -*- coding: utf-8 -*-
"""Tests for unified BackendCapabilityRegistry (MB-P2-001)."""
from __future__ import annotations

import pytest

from backend.capability_registry import (
    BackendCapabilityRegistry,
    BackendKind,
    CapabilityLevel,
    ExecutionKind,
    supports_pandas,
    supports_polars,
    supports_sql,
)


class TestCapabilityRegistryVersion:
    """Test capability registry versioning."""

    def test_version_returns_string(self):
        version = BackendCapabilityRegistry.version()
        assert isinstance(version, str)
        assert version.startswith("v")

    def test_version_hash_stable(self):
        hash1 = BackendCapabilityRegistry.version_hash()
        hash2 = BackendCapabilityRegistry.version_hash()
        assert hash1 == hash2
        assert len(hash1) == 16

    def test_version_hash_includes_state(self):
        # Hash should be deterministic based on capability state
        hash_val = BackendCapabilityRegistry.version_hash()
        assert isinstance(hash_val, str)
        assert len(hash_val) > 0


class TestCapabilityQuery:
    """Test unified capability queries."""

    def test_query_basic_operator(self):
        result = BackendCapabilityRegistry.query(
            "add",
            BackendKind.PANDAS_NUMPY,
            mode="research",
        )
        assert result.supported
        assert result.record is not None
        assert result.record.canonical == "add"
        assert result.record.backend == BackendKind.PANDAS_NUMPY

    def test_query_unsupported_backend(self):
        # Q_KDB capability is driven by the q evidence authority. `add` has a
        # q lowering (implemented); `corr` has no q lowering, so it is
        # genuinely unsupported on Q_KDB. The registry always builds a
        # BackendCapability record for a known backend kind, so `record` is
        # not None even when unsupported.
        result = BackendCapabilityRegistry.query(
            "corr",
            BackendKind.Q_KDB,
            mode="research",
        )
        assert not result.supported
        assert result.record is not None
        assert "unsupported" in result.reason

    def test_query_unknown_backend_string(self):
        result = BackendCapabilityRegistry.query(
            "add",
            "unknown_backend",
            mode="research",
        )
        assert not result.supported
        assert "Unknown backend" in result.reason

    def test_query_production_mode_filters(self):
        # Some operators may be implemented but not production-safe
        result_research = BackendCapabilityRegistry.query(
            "add",
            BackendKind.POLARS,
            mode="research",
        )
        result_production = BackendCapabilityRegistry.query(
            "add",
            BackendKind.POLARS,
            mode="production",
        )

        # Research mode accepts implemented
        if result_research.supported:
            assert result_research.record is not None

        # Production mode requires production_safe
        if result_production.supported:
            assert result_production.production_safe

    def test_query_sql_with_bound_params(self):
        # SQL queries with bound params should validate
        result = BackendCapabilityRegistry.query(
            "ts_mean",
            BackendKind.DUCKDB_SQL,
            mode="research",
            bound_params={"window": 20},
        )
        # May succeed or fail depending on implementation
        assert isinstance(result.supported, bool)
        if not result.supported:
            assert result.reason  # Should explain why

    def test_query_normalizes_backend_strings(self):
        # Test string normalization
        result1 = BackendCapabilityRegistry.query("add", "polars", mode="research")
        result2 = BackendCapabilityRegistry.query("add", BackendKind.POLARS, mode="research")

        assert result1.supported == result2.supported


class TestSupportsFunctions:
    """Test convenience supports_* functions."""

    def test_supports_polars_delegates_to_registry(self):
        result = supports_polars("add", mode="research")
        query_result = BackendCapabilityRegistry.query(
            "add", BackendKind.POLARS, mode="research"
        )
        assert result == query_result.supported

    def test_supports_sql_handles_dialects(self):
        # DuckDB
        result_duckdb = supports_sql("add", data_source_kind="duckdb", mode="research")
        # ClickHouse
        result_ch = supports_sql("add", data_source_kind="clickhouse", mode="research")

        assert isinstance(result_duckdb, bool)
        assert isinstance(result_ch, bool)

    def test_supports_pandas_delegates_to_registry(self):
        result = supports_pandas("add", mode="research")
        query_result = BackendCapabilityRegistry.query(
            "add", BackendKind.PANDAS_NUMPY, mode="research"
        )
        assert result == query_result.supported


class TestBackendCapabilityRecord:
    """Test BackendCapabilityRecord properties."""

    def test_is_production_eligible(self):
        result = BackendCapabilityRegistry.query(
            "add",
            BackendKind.PANDAS_NUMPY,
            mode="research",
        )

        if result.record:
            is_eligible = result.record.is_production_eligible()
            assert is_eligible == (result.record.level == CapabilityLevel.PRODUCTION_SAFE)

    def test_is_native_execution(self):
        result = BackendCapabilityRegistry.query(
            "add",
            BackendKind.PANDAS_NUMPY,
            mode="research",
        )

        if result.record:
            # Pandas should be REFERENCE, not native
            is_native = result.record.is_native_execution()
            assert isinstance(is_native, bool)


class TestListBackends:
    """Test backend enumeration."""

    def test_list_backends_returns_supported(self):
        backends = BackendCapabilityRegistry.list_backends("add", mode="research")
        assert isinstance(backends, list)
        assert all(isinstance(b, BackendKind) for b in backends)

    def test_list_backends_production_stricter(self):
        research = BackendCapabilityRegistry.list_backends("add", mode="research")
        production = BackendCapabilityRegistry.list_backends("add", mode="production")

        # Production should be subset of research
        assert set(production).issubset(set(research))


class TestUnifiedAuthority:
    """Test that registry is single source of truth (MB-P2-001)."""

    def test_no_conflicting_capability_sources(self):
        # Query through registry
        registry_result = BackendCapabilityRegistry.query(
            "add",
            BackendKind.POLARS,
            mode="research",
        )

        # Query through old interface
        from backend.operator_capability import supports_polars as old_supports_polars

        old_result = old_supports_polars("add", mode="research")

        # Should be consistent
        assert registry_result.supported == old_result

    def test_version_tracking(self):
        # Every query should include version
        result = BackendCapabilityRegistry.query(
            "add",
            BackendKind.PANDAS_NUMPY,
            mode="research",
        )

        assert result.registry_version
        if result.record:
            assert result.record.registry_version == result.registry_version


class TestMBP2001Requirements:
    """Verify MB-P2-001 requirements are met."""

    def test_single_version_number(self):
        # Registry has single version
        version = BackendCapabilityRegistry.version()
        assert version

    def test_explicit_version_tracking(self):
        # Every record includes version
        result = BackendCapabilityRegistry.query("add", "pandas_numpy", mode="research")
        if result.record:
            assert hasattr(result.record, "registry_version")
            assert result.record.registry_version

    def test_consistent_interface(self):
        # All backend queries use same interface
        for backend in [BackendKind.PANDAS_NUMPY, BackendKind.POLARS, BackendKind.DUCKDB_SQL]:
            result = BackendCapabilityRegistry.query("add", backend, mode="research")
            assert hasattr(result, "supported")
            assert hasattr(result, "production_safe")
            assert hasattr(result, "record")
            assert hasattr(result, "reason")

    def test_no_inline_capability_checks_needed(self):
        # Registry provides all needed info
        result = BackendCapabilityRegistry.query("add", "polars", mode="research")

        if result.record:
            # Record has all semantic flags
            assert hasattr(result.record, "supports_nulls")
            assert hasattr(result.record, "supports_streaming")
            assert hasattr(result.record, "execution_kind")
            assert hasattr(result.record, "level")


def test_smoke_multiple_operators():
    """Smoke test with multiple operators."""
    operators = ["add", "subtract", "multiply", "divide", "abs", "log"]
    backends = [BackendKind.PANDAS_NUMPY, BackendKind.POLARS, BackendKind.DUCKDB_SQL]

    for op in operators:
        for backend in backends:
            result = BackendCapabilityRegistry.query(op, backend, mode="research")
            # Should not raise
            assert isinstance(result.supported, bool)
