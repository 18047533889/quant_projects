from __future__ import annotations

import subprocess
from datetime import datetime, timezone

import pytest

from data_access.read.execution_identity import ExecutionIdentity, get_current_execution_identity
from data_access.runtime import startup_subject


def _identity(**changes) -> ExecutionIdentity:
    values = {
        "build_sha": "a" * 40,
        "package_version": "1.2.3",
        "build_id": "build-a",
        "build_time": datetime(2026, 8, 15, 12, 30, tzinfo=timezone.utc),
        "runtime_mode": "production",
        "semantic_execution_version": "v1",
    }
    values.update(changes)
    return ExecutionIdentity(**values)


def test_execution_identity_full_sha256_covers_every_declared_component() -> None:
    base = _identity()

    assert len(base.digest()) == 64
    assert all(character in "0123456789abcdef" for character in base.digest())

    variants = (
        _identity(build_sha="b" * 40),
        _identity(package_version="1.2.4"),
        _identity(build_id="build-b"),
        _identity(build_time=datetime(2026, 8, 15, 12, 31, tzinfo=timezone.utc)),
        _identity(runtime_mode="research"),
        _identity(semantic_execution_version="v2"),
    )
    assert all(variant.digest() != base.digest() for variant in variants)


def test_execution_identity_normalizes_aware_build_time_to_utc() -> None:
    utc = _identity()
    same_instant = _identity(
        build_time=datetime.fromisoformat("2026-08-15T20:30:00+08:00")
    )

    assert utc.digest() == same_instant.digest()


def test_incomplete_execution_identity_cannot_form_digest() -> None:
    identity = ExecutionIdentity(
        package_version="1.2.3",
        runtime_mode="production",
        semantic_execution_version="v1",
    )

    assert identity.available is False
    with pytest.raises(Exception, match="build_sha"):
        identity.digest()


def test_current_execution_identity_fails_closed_without_build_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        "data_access.core.build_metadata.load_build_info",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("metadata unavailable")),
    )

    with pytest.raises(RuntimeError, match="metadata unavailable"):
        get_current_execution_identity()


def test_current_execution_identity_uses_frozen_build_and_runtime_authorities(monkeypatch) -> None:
    frozen = type(
        "FrozenBuildInfo",
        (),
        {
            "build_sha": "c" * 40,
            "version": "1.2.3",
            "build_id": "build-c",
            "build_time": "2026-08-15T00:00:00Z",
        },
    )()
    monkeypatch.setattr(
        "data_access.core.build_metadata.load_build_info",
        lambda **kwargs: frozen,
    )
    monkeypatch.setattr(
        "data_access.runtime.mode_identity.current_runtime_mode",
        lambda: type("Mode", (), {"value": "production"})(),
    )

    identity = get_current_execution_identity()

    assert identity.available is True
    assert identity.build_sha == "c" * 40
    assert identity.runtime_mode == "production"
    assert len(identity.digest()) == 64


    full_sha = "b" * 40
    monkeypatch.setattr(
        "data_access.core.build_metadata.load_build_info",
        lambda: type("FrozenBuildInfo", (), {"build_sha": full_sha})(),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("runtime build identity invoked subprocess"),
    )

    assert startup_subject._get_build_sha() == full_sha


def test_startup_build_sha_fails_closed_when_frozen_metadata_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "data_access.core.build_metadata.load_build_info",
        lambda: type("FrozenBuildInfo", (), {"build_sha": None})(),
    )

    assert startup_subject._get_build_sha() == "unknown"
