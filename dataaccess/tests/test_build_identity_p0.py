from __future__ import annotations

import importlib.metadata
import subprocess

import pytest

from data_access.core.build_metadata import (
    BuildInfo,
    ScmVersionResolver,
    generate_build_info,
    load_build_info,
    validate_revision,
    validate_version_match,
)


def test_malformed_external_revision_rejected() -> None:
    for value in ("abc123", "unknown", "g" * 40, "0" * 39):
        with pytest.raises(Exception):
            validate_revision(value)


def test_runtime_metadata_never_invokes_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("runtime build metadata invoked git")

    monkeypatch.setattr(subprocess, "run", fail)
    info = load_build_info(production=False)
    assert info.version == "0.10.2"
    assert info.build_sha is None


def test_production_missing_metadata_fails_closed() -> None:
    with pytest.raises(Exception):
        load_build_info(production=True)


def test_build_fields_validate_full_identity() -> None:
    info = BuildInfo(
        version="0.10.2",
        build_sha="a" * 40,
        build_id="a" * 12,
        build_time="2026-08-15T00:00:00Z",
        dirty=False,
    )
    info.validate_production_ready()


def test_generate_build_info_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "get_build_sha": "a" * 40,
        "get_version_from_git": "0.10.2",
        "is_dirty": False,
        "get_current_tag": "v0.10.2",
        "get_current_branch": "main",
        "_run": "2026-08-15T08:00:00+08:00",
    }
    for method, value in values.items():
        monkeypatch.setattr(ScmVersionResolver, method, lambda self, *args, value=value: value)

    first = generate_build_info()
    monkeypatch.setattr(ScmVersionResolver, "get_current_branch", lambda self: "detached")
    second = generate_build_info()

    assert first.build_time == "2026-08-15T00:00:00Z"
    assert first.build_id == second.build_id
    assert first.branch != second.branch


def test_source_date_epoch_is_deterministic_and_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ScmVersionResolver, "get_build_sha", lambda self: "b" * 40)
    monkeypatch.setattr(ScmVersionResolver, "get_version_from_git", lambda self: "0.10.2")
    monkeypatch.setattr(ScmVersionResolver, "is_dirty", lambda self: False)
    monkeypatch.setattr(ScmVersionResolver, "get_current_tag", lambda self: None)
    monkeypatch.setattr(ScmVersionResolver, "get_current_branch", lambda self: "main")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")

    assert generate_build_info().build_time == "1970-01-01T00:00:00Z"

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "invalid")
    with pytest.raises(Exception, match="SOURCE_DATE_EPOCH"):
        generate_build_info()


def test_runtime_uses_generated_version_without_distribution_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("runtime build identity consulted distribution metadata")

    monkeypatch.setattr(importlib.metadata, "version", fail)
    assert load_build_info(production=False).version == "0.10.2"


def test_runtime_missing_metadata_rejected_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "data_access.core.build_metadata._generated_build_info",
        lambda: BuildInfo(None, None, None, None, None),
    )
    with pytest.raises(Exception, match="unavailable"):
        load_build_info(production=True)


    with pytest.raises(Exception, match="version mismatch"):
        validate_version_match("0.10.1", "0.10.2")


def test_version_match_returns_canonical_declared_value() -> None:
    assert validate_version_match("0.10.2", "0.10.2") == "0.10.2"


def test_missing_version_metadata_rejected() -> None:
    with pytest.raises(Exception, match="unavailable"):
        validate_version_match(None, "0.10.2")


def test_installed_style_import_has_single_version_authority() -> None:
    import data_access
    from data_access import _build_info

    assert data_access.__version__ == _build_info.__version__ == "0.10.2"
    assert data_access.__build_sha__ == _build_info.__build_sha__
