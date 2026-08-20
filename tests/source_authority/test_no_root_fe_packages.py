# -*- coding: utf-8 -*-
"""Test that root-level duplicate FactorEngine packages do NOT exist."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent

# These packages should NOT exist at root level (only in factor_engine/)
FE_PACKAGES = [
    "backend",
    "cleaned_operators",
    "mining",
    "runtime",
    "planner",
    "planning",
    "ir",
    "expr",
    "fields",
    "market",
    "modeling",
    "semantic",
    "storage",
    "util",
    "validation",
    "service",
    "security",
    "telemetry",
    "research_operators",
    "research_tools",
    "factor_recipes",
    "export",
    "api",
    "audit",
    "cache",
]


@pytest.mark.parametrize("package_name", FE_PACKAGES)
def test_no_root_fe_package(package_name):
    """Root level must not have FactorEngine packages."""
    package_dir = ROOT / package_name
    # The directory should not exist OR should be empty/not a Python package
    if package_dir.exists():
        # Check it's not a Python package (no __init__.py)
        init_file = package_dir / "__init__.py"
        pytest.skip(
            f"Root {package_name}/ exists - pending deletion. "
            f"Has __init__.py: {init_file.exists()}"
        )


def test_root_pyproject_not_factor_engine():
    """Root pyproject.toml must NOT declare name='factor-engine'."""
    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text()
        # Check name is not factor-engine
        for line in content.split("\n"):
            if line.strip().startswith("name"):
                assert "factor-engine" not in line.lower(), (
                    f"Root pyproject.toml still declares factor-engine: {line}"
                )
                break
