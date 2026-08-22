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
        lambda: (_ for _ in ()).throw(RuntimeError("metadata unavailable")),
    )

    with pytest.raises(RuntimeError, match="metadata unavailable"):
        startup_subject._get_build_sha()


def test_startup_subject_uses_full_width_correctness_inputs(monkeypatch) -> None:
    registry_names = {"dataset-" + "r" * 80: object()}
    store = type("Store", (), {"_registry": type("Registry", (), {"_datasets": registry_names})()})()
    contract_digest = "c" * 64
    policy_digest = "p" * 64
    generation = "generation-" + "g" * 80

    store.contract_ir = lambda: type("Contract", (), {"digest": lambda self: contract_digest})()
    policy = type("Policy", (), {"digest": lambda self: policy_digest})()
    context = type("Context", (), {"access_policy": policy})()
    monkeypatch.setattr("data_access.security.runtime.resolve_runtime_context", lambda: context)
    monkeypatch.setattr(
        "data_access.security.credentials._global_credential_provider",
        lambda: type("Provider", (), {"generation": generation})(),
    )

    assert len(startup_subject._get_registry_digest(store)) == 64
    assert startup_subject._get_contract_digest(store) == contract_digest
    assert startup_subject._get_policy_digest() == policy_digest
    assert startup_subject._get_credential_generation() == generation


def test_startup_subject_invalidates_on_changes_after_old_truncation_boundary() -> None:
    base = startup_subject.StartupSubjectDigest(
        build_sha="a" * 40,
        package_version="1.2.3",
        registry_digest="r" * 64,
        contract_digest="c" * 64,
        policy_digest="p" * 64,
        credential_generation="generation-" + "g" * 64,
    )
    variants = (
        startup_subject.StartupSubjectDigest(**{**base.__dict__, "registry_digest": "r" * 63 + "x"}),
        startup_subject.StartupSubjectDigest(**{**base.__dict__, "contract_digest": "c" * 63 + "x"}),
        startup_subject.StartupSubjectDigest(**{**base.__dict__, "policy_digest": "p" * 63 + "x"}),
        startup_subject.StartupSubjectDigest(**{**base.__dict__, "credential_generation": "generation-" + "g" * 63 + "x"}),
    )

    assert all(variant.to_digest() != base.to_digest() for variant in variants)
