# -*- coding: utf-8 -*-
"""Test that FactorEngine imports resolve to factor_engine/ submodule."""
import sys
from pathlib import Path

import pytest

# factor_engine/ submodule must be first in sys.path for correct imports
FE_ROOT = Path(__file__).resolve().parent.parent.parent / "factor_engine"


@pytest.fixture(autouse=True)
def _ensure_fe_path():
    """Ensure factor_engine/ is first in sys.path for this test module.

    CRITICAL: this must be an "add to front, dedupe" operation, NOT drop the
    primary path.  ``sys.path[0]`` on Linux is always the *directory of the
    module being run* (or the pytest rootdir for a rootdir-less run).  The
    PREVIOUS implementation re-created ``sys.path`` from scratch, which dropped
    it; when the new list had no element that satisfied
    ``sys.path[0] == new_list[0]`` == "is this exact list at position 0?",
    CPython re-inserted the script dir AT THE FRONT.  That put the repo ROOT
    (the pytest rootdir) in front of FE_ROOT, so every FE module resolved to
    the root duplicate package, shadowing factor_engine authority -- exactly
    the failure these tests were written to catch.
    """
    fe_str = str(FE_ROOT)
    original_path = sys.path.copy()
    if fe_str not in sys.path:
        sys.path.insert(0, fe_str)
    else:
        # Move FE_ROOT to the front, leaving the rest of sys.path intact.
        sys.path.remove(fe_str)
        sys.path.insert(0, fe_str)
    # Post-condition: FE_ROOT sorted before any duplicate entries and
    # sys.path[0] is unchanged unless it WAS a duplicate of FE_ROOT.
    assert sys.path[0] == fe_str, (
        f"FE_ROOT should be first in sys.path, got {sys.path[0]!r}"
    )
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

    # Clear any cached module AND all of its already-imported submodules
    # (e.g. "factor_engine.planner.logical_plan").  Otherwise a stale submodule cached from
    # the root copy shadows the factor_engine one via relative imports.
    prefix = module_name + "."
    for cached_name in [
        m for m in sys.modules if m == module_name or m.startswith(prefix)
    ]:
        del sys.modules[cached_name]

    # Import
    mod = importlib.import_module(module_name)
    mod_file = getattr(mod, "__file__", None)
    # Fail-closed: no __file__ or wrong location means a root duplicate
    # package is shadowing factor_engine authority.
    assert mod_file is not None, (
        f"{module_name} has no __file__ - "
        f"root duplicate package 遮蔽了 FE authority"
    )
    assert str(FE_ROOT) in str(Path(mod_file).resolve()), (
        f"{module_name} resolved to {mod_file}, NOT under {FE_ROOT} - "
        f"root duplicate package 遮蔽了 FE authority"
    )


# Core FE packages that must NEVER resolve to a root-level duplicate.
FE_CORE_PACKAGES = [
    "backend",
    "runtime",
    "planner",
    "planning",
    "mining",
    "cleaned_operators",
]


@pytest.mark.parametrize("module_name", FE_CORE_PACKAGES)
def test_fe_core_package_file_under_factor_engine(module_name):
    """Identity: core FE package __file__ must physically live under factor_engine/."""
    import importlib

    # Clear any cached module AND all of its already-imported submodules,
    # same as the resolution test above.
    prefix = module_name + "."
    for cached_name in [
        m for m in sys.modules if m == module_name or m.startswith(prefix)
    ]:
        del sys.modules[cached_name]

    mod = importlib.import_module(module_name)
    mod_file = getattr(mod, "__file__", None)
    assert mod_file is not None, (
        f"{module_name} has no __file__ - "
        f"root duplicate package 遮蔽了 FE authority"
    )
    resolved = Path(mod_file).resolve()
    assert resolved.is_relative_to(FE_ROOT.resolve()), (
        f"{module_name} resolves to {resolved}, NOT under {FE_ROOT} - "
        f"root duplicate package 遮蔽了 FE authority"
    )
