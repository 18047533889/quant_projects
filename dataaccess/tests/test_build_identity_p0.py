from __future__ import annotations

import subprocess

import pytest

from data_access.core.build_metadata import BuildInfo, load_build_info, validate_revision


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


def test_installed_style_import_has_single_version_authority() -> None:
    import data_access
    from data_access import _build_info

    assert data_access.__version__ == _build_info.__version__ == "0.10.2"
    assert data_access.__build_sha__ == _build_info.__build_sha__
