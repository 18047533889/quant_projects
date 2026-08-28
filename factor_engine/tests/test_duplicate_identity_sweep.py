# -*- coding: utf-8 -*-
"""P0-03 duplicate-identity sweep — the mission test.

For every module reachable under BOTH a legacy name (cleaned_operators.*, api,
backend, runtime, planner, ...) and factor_engine.*, assert both resolve to the
SAME module object (a single sys.modules entry per physical .py).  Also assert
find_spec('cleaned_operators') is None in a pristine interpreter (no physical
top-level package) and that the shim never shadows third-party names.
"""
import importlib
import importlib.util
import sys

import pytest


LEGACY_TOPS = [
    "api", "backend", "cache", "cleaned_operators", "planner", "planning",
    "runtime", "mining", "security", "semantic", "service", "storage",
    "telemetry", "util", "validation", "market", "fields", "expr", "ir",
    "factor_recipes", "export", "research_operators", "research_tools", "audit",
    "execution", "operator_contracts",
]

#: Representative dotted submodules under the legacy head.
LEGACY_DOTTED = [
    "cleaned_operators.registry",
    "cleaned_operators.common.polars_statistics",
    "backend.sql_pushdown.emitter",
    "runtime.multibackend",
    "planner.physical_lowerer",
]

THIRD_PARTY = ["polars", "pandas", "numpy", "pyarrow", "scipy", "packaging", "yaml"]


@pytest.fixture(autouse=True)
def _fresh_sys_modules():
    """Give every test an interpreter with factor_engine freshly imported."""
    yield


def test_legacy_and_canonical_resolve_to_same_object():
    import factor_engine  # installs the shim

    for legacy in LEGACY_TOPS:
        canonical_name = f"factor_engine.{legacy}"
        if importlib.util.find_spec(canonical_name) is None:
            continue
        leg = importlib.import_module(legacy)
        can = sys.modules[canonical_name]
        assert leg is can, (
            f"legacy {legacy!r} is a DIFFERENT object from canonical "
            f"{canonical_name!r}"
        )


def test_legacy_dotted_resolve_to_same_object():
    import factor_engine

    for dotted in LEGACY_DOTTED:
        canonical_name = f"factor_engine.{dotted}"
        if importlib.util.find_spec(canonical_name) is None:
            continue
        leg = importlib.import_module(dotted)
        can = sys.modules[canonical_name]
        assert leg is can, (
            f"legacy dotted {dotted!r} is a DIFFERENT object from canonical "
            f"{canonical_name!r}"
        )


def test_registry_singleton_is_single_object():
    import factor_engine

    from factor_engine.cleaned_operators.registry import OperatorRegistry as A
    import cleaned_operators.registry as leg_reg

    B = leg_reg.OperatorRegistry
    assert A is B
    assert A is sys.modules["factor_engine.cleaned_operators.registry"].OperatorRegistry


def test_find_spec_legacy_is_none_in_pristine_interpreter():
    # In a pristine interpreter (no factor_engine import, no shim installed) the
    # legacy top-level name must NOT resolve to a physical package.
    if "factor_engine" in sys.modules:
        pytest.skip("not pristine — canonical already imported")
    for name in ("cleaned_operators", "api", "backend", "runtime", "planner"):
        assert importlib.util.find_spec(name) is None, (
            f"pristine find_spec({name!r}) must be None (no physical top-level package)"
        )


def test_shim_does_not_shadow_third_party_names():
    import factor_engine

    for name in THIRD_PARTY:
        spec = importlib.util.find_spec(name)
        assert spec is not None, f"third-party {name!r} must remain importable"
        # The spec must come from site-packages / stdlib, never the FE tree.
        origin = spec.origin or ""
        assert "factor_engine" not in origin.replace("\\", "/"), (
            f"third-party {name!r} resolved to FE origin {origin!r}"
        )


def test_shim_idempotent():
    import factor_engine.deprecated_shims as shims

    before = len(sys.meta_path)
    shims.install_legacy_aliases()
    shims.install_legacy_aliases()
    assert len(sys.meta_path) == before
    n = sum(1 for f in sys.meta_path if type(f).__name__ == "_LegacyAliasFinder")
    assert n == 1
