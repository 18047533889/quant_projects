"""Keep declared runtime payload exclusions effective for Python modules."""
import runpy
from pathlib import Path

import pytest
from setuptools import Distribution
from setuptools.command.build_py import build_py


@pytest.mark.parametrize("wildcard", ["", "*"])
def test_payload_hook_filters_modules_without_running_setup(monkeypatch, wildcard):
    def forbidden_setup(*args, **kwargs):
        raise AssertionError("importing the build hook must not execute setup")

    monkeypatch.setattr("setuptools.setup", forbidden_setup)
    package_root = Path(__file__).resolve().parents[2]
    hook = runpy.run_path(str(package_root / "setup.py"))["DeclaredPayloadBuildPy"]
    entries = [("pkg", name[:-3], "pkg/" + name) for name in
               ["engine.py", "test_engine.py", "setup.py", "secret_impl.py"]]
    monkeypatch.setattr(build_py, "find_package_modules", lambda *args: entries)
    distribution = Distribution()
    distribution.exclude_package_data = {
        wildcard: ["test_*.py", "setup.py"], "pkg": ["secret_*.py"]}
    command = hook(distribution)
    assert command.find_package_modules("pkg", "pkg") == [entries[0]]


def test_payload_hook_keeps_unexcluded_runtime_modules(monkeypatch):
    package_root = Path(__file__).resolve().parents[2]
    hook = runpy.run_path(str(package_root / "setup.py"))["DeclaredPayloadBuildPy"]
    entries = [("pkg", "engine", "pkg/engine.py")]
    monkeypatch.setattr(build_py, "find_package_modules", lambda *args: entries)
    command = hook(Distribution())
    assert command.find_package_modules("pkg", "pkg") == entries
