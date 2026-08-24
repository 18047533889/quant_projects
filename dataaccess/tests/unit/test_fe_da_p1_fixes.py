"""Tests for FE-DA P1 remediation fixes."""

from pathlib import Path

import pytest


def test_p1_3_bare_exception_import_error_narrow(monkeypatch):
    """P1-3: _strict_mutation_lock_required narrows to ImportError only."""
    from data_access.read import query_budget
    from data_access.store import _strict_mutation_lock_required

    monkeypatch.setattr(query_budget, "is_strict_semantics", lambda: False)
    assert _strict_mutation_lock_required() is False

    monkeypatch.setattr(
        query_budget,
        "is_strict_semantics",
        lambda: (_ for _ in ()).throw(ImportError("circular")),
    )
    assert _strict_mutation_lock_required() is True

    monkeypatch.setattr(
        query_budget,
        "is_strict_semantics",
        lambda: (_ for _ in ()).throw(RuntimeError("runtime failure")),
    )
    with pytest.raises(RuntimeError, match="runtime failure"):
        _strict_mutation_lock_required()


def test_p1_6_env_var_validation_cos_read_mode(monkeypatch):
    """P1-6: DATA_ACCESS_COS_READ_MODE validates allowed values."""
    from data_access.store import _cos_mode_is_auto

    for val in ["mirror", "auto", "remote", "local"]:
        monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", val)
        assert _cos_mode_is_auto() is (val == "auto")

    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "invalid_mode")
    with pytest.raises(ValueError, match="DATA_ACCESS_COS_READ_MODE.*invalid"):
        _cos_mode_is_auto()


def test_p1_6_env_var_validation_allow_full_sync(monkeypatch):
    """P1-6: DATA_ACCESS_ALLOW_FULL_COS_SYNC validates boolean values."""
    from data_access.cos_storage_runtime import _allow_full_sync

    for val in ["1", "true", "yes", "on", "TRUE", "Yes", "ON"]:
        monkeypatch.setenv("DATA_ACCESS_ALLOW_FULL_COS_SYNC", val)
        assert _allow_full_sync() is True

    for val in ["0", "false", "no", "off", "", "FALSE"]:
        monkeypatch.setenv("DATA_ACCESS_ALLOW_FULL_COS_SYNC", val)
        assert _allow_full_sync() is False

    monkeypatch.setenv("DATA_ACCESS_ALLOW_FULL_COS_SYNC", "maybe")
    with pytest.raises(ValueError, match="DATA_ACCESS_ALLOW_FULL_COS_SYNC.*invalid"):
        _allow_full_sync()


def test_p1_6_env_var_validation_snapshot_ttl(monkeypatch):
    """P1-6: FACTOR_ENGINE_DATA_SNAPSHOT_TTL_SECONDS validates positive float."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "factor_engine"))
    from factor_engine.storage.sources.data_access_source import DataAccessSource

    for val in ["60", "120.5", "0.1"]:
        monkeypatch.setenv("FACTOR_ENGINE_DATA_SNAPSHOT_TTL_SECONDS", val)
        source = DataAccessSource(dataset="test_dataset")
        assert source._snapshot_ttl_seconds == float(val)

    for val in ["-10", "0", "not_a_number"]:
        monkeypatch.setenv("FACTOR_ENGINE_DATA_SNAPSHOT_TTL_SECONDS", val)
        with pytest.raises(
            ValueError,
            match="FACTOR_ENGINE_DATA_SNAPSHOT_TTL_SECONDS.*invalid",
        ):
            DataAccessSource(dataset="test_dataset")


def test_p1_5_production_research_parity_authority(monkeypatch):
    """P1-5: Verify run_mode parity - single authority via RuntimeModeIdentity."""
    from data_access.read.query_budget import is_strict_semantics
    from data_access.runtime.mode_identity import (
        current_runtime_mode,
        is_strict_semantics_authority,
        reset_runtime_mode_identity,
        set_runtime_mode_identity,
    )
    from data_access.security.run_mode import RunMode

    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)

    token = set_runtime_mode_identity(RunMode("production"), source="test")
    try:
        assert current_runtime_mode().value == "production"
        assert is_strict_semantics_authority() is True
        assert is_strict_semantics() is True
    finally:
        reset_runtime_mode_identity(token)

    token = set_runtime_mode_identity(RunMode("interactive_research"), source="test")
    try:
        assert current_runtime_mode().value == "interactive_research"
        assert is_strict_semantics_authority() is False
        assert is_strict_semantics() is False
    finally:
        reset_runtime_mode_identity(token)
