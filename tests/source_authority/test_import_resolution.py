# -*- coding: utf-8 -*-
"""Test that FactorEngine imports resolve to factor_engine/ submodule."""
import sys
from pathlib import Path

import pytest

# factor_engine/ submodule must be first in sys.path for correct imports
FE_ROOT = Path(__file__).resolve().parent.parent.parent / "factor_engine"


@pytest.fixture(autouse=True)
def _ensure_fe_path():
    """Ensure factor_engine/ is first in sys.path for this test module."""
    fe_str = str(FE_ROOT)
    original_path = sys.path.copy()
    # Put factor_engine first
    sys.path = [fe_str] + [p for p in sys.path if p != fe_str]
    yield
    sys.path = original_path


@pytest.mark.parametrize(
    "module_name",
    [
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
    ],
)
def test_import_resolves_to_factor_engine(module_name):
    """Each FE module must resolve to factor_engine/, not root."""
    import importlib

    # Clear any cached module
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Import
    mod = importlib.import_module(module_name)
    mod_file = getattr(mod, "__file__", None)
    if mod_file:
        assert str(FE_ROOT) in mod_file, (
            f"{module_name} resolved to {mod_file}, "
            f"expected it to be under {FE_ROOT}"
        )
