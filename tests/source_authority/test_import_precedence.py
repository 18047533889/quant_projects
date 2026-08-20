# -*- coding: utf-8 -*-
"""Test import precedence behavior for FactorEngine modules."""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
FE = ROOT / "factor_engine"


def test_import_precedence_factor_engine_first():
    """When factor_engine is first in sys.path, imports resolve there."""
    fe_str = str(FE)
    root_str = str(ROOT)

    # Save original
    original_path = sys.path.copy()

    try:
        # Clear cached modules
        for mod_name in ["backend", "cleaned_operators", "mining"]:
            if mod_name in sys.modules:
                del sys.modules[mod_name]

        # Set factor_engine first
        sys.path = [fe_str] + [p for p in sys.path if p not in (fe_str, root_str)]

        # Import
        mod = importlib.import_module("backend")
        mod_file = getattr(mod, "__file__", None)
        assert mod_file is not None
        assert fe_str in mod_file, f"Expected factor_engine, got {mod_file}"

    finally:
        sys.path = original_path


def test_import_precedence_root_first():
    """When root is first in sys.path, imports resolve to root (undesired)."""
    fe_str = str(FE)
    root_str = str(ROOT)

    # Save original
    original_path = sys.path.copy()

    try:
        # Clear cached modules
        for mod_name in ["backend", "cleaned_operators", "mining"]:
            if mod_name in sys.modules:
                del sys.modules[mod_name]

        # Set root first
        sys.path = [root_str] + [p for p in sys.path if p not in (fe_str, root_str)]

        # Import
        mod = importlib.import_module("backend")
        mod_file = getattr(mod, "__file__", None)
        assert mod_file is not None
        assert root_str in mod_file, f"Expected root, got {mod_file}"

    finally:
        sys.path = original_path


def test_current_default_resolves_to_root():
    """Current default sys.path resolves backend to root (the bug we're fixing)."""
    # This documents the current broken state
    mod_file = None
    if "backend" in sys.modules:
        mod_file = getattr(sys.modules["backend"], "__file__", None)
    else:
        try:
            mod = importlib.import_module("backend")
            mod_file = getattr(mod, "__file__", None)
        except ImportError:
            pytest.skip("Cannot import backend (missing deps)")

    if mod_file:
        # Currently resolves to root - this documents the problem
        assert str(ROOT) in mod_file, f"Expected root (current bug), got {mod_file}"
