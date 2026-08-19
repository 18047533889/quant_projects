"""Regression tests for the standalone package boundary."""

from pathlib import Path

from setuptools import find_packages
from setuptools.config.pyprojecttoml import read_configuration


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_setuptools_discovery_includes_only_modeling_adapters():
    """The wheel package list must not absorb the legacy modeling namespace."""
    config = read_configuration(str(_PROJECT_ROOT / "pyproject.toml"), expand=False)
    discovery = config["tool"]["setuptools"]["packages"]["find"]
    packages = find_packages(
        where=str(_PROJECT_ROOT),
        include=discovery["include"],
    )

    assert packages
    assert all(
        package == "modeling_adapters" or package.startswith("modeling_adapters.")
        for package in packages
    )
    assert "modeling" not in packages
    assert not any(package.startswith("modeling.") for package in packages)
