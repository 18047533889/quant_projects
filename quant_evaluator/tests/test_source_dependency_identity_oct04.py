"""Strict, bounded identities for the source-route dependency closure."""

import sys
from types import ModuleType

import pytest

from quant_evaluator.runtime import source_dependency_identity as identity_module
from quant_evaluator.runtime.source_dependency_identity import (
    COMPONENT_MODULES, SourceDependencyIdentityError,
    capture_source_dependency_identity,
)


def _install_component_packages(monkeypatch, base):
    roots = {}
    for component in COMPONENT_MODULES:
        prefix = component + "."
        for name in tuple(sys.modules):
            if name == component or name.startswith(prefix):
                monkeypatch.delitem(sys.modules, name, raising=False)
        package = base / component / "__init__.py"
        package.parent.mkdir(parents=True, exist_ok=True)
        package.write_text(f'"""{component} test package."""\n', encoding="utf-8")
        leaf = package.parent / "implementation.py"
        leaf.write_text("VALUE = 1\n", encoding="utf-8")
        module = ModuleType(component)
        module.__file__ = str(package)
        module.__path__ = [str(package.parent)]
        monkeypatch.setitem(sys.modules, component, module)
        roots[component] = leaf
    return roots


def test_dependency_digest_changes_for_every_component_source_tree(monkeypatch, tmp_path):
    files = _install_component_packages(monkeypatch, tmp_path)
    first = capture_source_dependency_identity(max_records=256, max_bytes=1024**2)
    assert first.schema == "source_dependency_identity.v1"
    assert tuple(name for name, _ in first.component_digests) == COMPONENT_MODULES
    assert first.total_bytes_hashed > 0
    assert first.total_files_hashed >= len(COMPONENT_MODULES) * 2
    assert all(receipt.full_content_checked
               for _, receipt in first.component_receipts)

    for component in COMPONENT_MODULES:
        previous = first.digest
        files[component].write_text("VALUE = 2\n", encoding="utf-8")
        changed = capture_source_dependency_identity(max_records=256, max_bytes=1024**2)
        assert changed.digest != previous
        assert dict(changed.component_digests)[component] != dict(first.component_digests)[component]
        first = changed


def test_missing_or_namespace_dependency_root_fails_closed(monkeypatch, tmp_path):
    _install_component_packages(monkeypatch, tmp_path)
    monkeypatch.delitem(sys.modules, COMPONENT_MODULES[1])
    with pytest.raises(SourceDependencyIdentityError, match="unavailable"):
        capture_source_dependency_identity()

    _install_component_packages(monkeypatch, tmp_path)
    namespace = ModuleType(COMPONENT_MODULES[2])
    namespace.__file__ = None
    namespace.__path__ = [str(tmp_path / "one"), str(tmp_path / "two")]
    monkeypatch.setitem(sys.modules, COMPONENT_MODULES[2], namespace)
    with pytest.raises(SourceDependencyIdentityError, match="root"):
        capture_source_dependency_identity()


def test_total_record_and_byte_budgets_are_enforced_across_components(monkeypatch, tmp_path):
    _install_component_packages(monkeypatch, tmp_path)
    with pytest.raises(SourceDependencyIdentityError, match="scan failed|budget"):
        capture_source_dependency_identity(max_records=7, max_bytes=1024**2)
    with pytest.raises(SourceDependencyIdentityError, match="scan failed|budget"):
        capture_source_dependency_identity(max_records=256, max_bytes=1)


def test_source_scanner_exception_fails_closed(monkeypatch, tmp_path):
    _install_component_packages(monkeypatch, tmp_path)

    def fail(*args, **kwargs):
        raise OSError("synthetic source scan failure")

    monkeypatch.setattr(identity_module.ProcessSourceIdentity, "identify", fail)
    with pytest.raises(SourceDependencyIdentityError, match="identity"):
        capture_source_dependency_identity()
