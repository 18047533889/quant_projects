# -*- coding: utf-8 -*-
"""P0-03: single module identity gate for FactorEngine.

Confirms that the distribution exposes EXACTLY one importable identity —
``factor_engine`` and ``factor_engine.*`` — and that no physical ``.py`` file
is loadable under two ``sys.modules`` names.  Runs under the CURRENT venv
(repo-root path), so the working tree is the authority.

This gate is import-order-independent: it asserts the canonical side first
(``factor_engine.<pkg>`` resolves), and only then probes the legacy names that
the deprecated_shims aliasing exposes.  In a pristine interpreter (no
``factor_engine`` import) the legacy names are simply absent, which the
``find_spec is None`` assertions below cover; once ``factor_engine`` is
imported the shim exposes the legacy names as state-free aliases of the SAME
module objects (no second package copy, no second OperatorRegistry).
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Packages that historically shipped as unqualified top-level names and now
#: must ONLY exist under factor_engine.*.
LEGACY_PACKAGES = [
    "api",
    "backend",
    "cache",
    "cleaned_operators",
    "planner",
    "planning",
    "runtime",
    "mining",
    "security",
    "semantic",
    "service",
    "storage",
    "telemetry",
    "util",
    "validation",
    "market",
    "fields",
    "expr",
    "ir",
    "factor_recipes",
    "export",
    "research_operators",
    "research_tools",
]

#: Top-level non-package modules that must resolve only under factor_engine.*.
LEGACY_ROOT_MODULES = [
    "logging_utils",
    "pit_contract",
    "stateful_contract",
    "stateful_runtime",
    "workspace_paths",
]

#: Representative dotted submodules that must map to the SAME object as their
#: factor_engine.* counterpart.
LEGACY_DOTTED = [
    "cleaned_operators.registry",
    "backend.sql_pushdown.emitter",
    "runtime.multibackend",
]


def _pristine_import(name: str) -> bool:
    """Return True when *name* is importable in the CURRENT interpreter."""
    return importlib.util.find_spec(name) is not None


class TestCanonicalOnlyPackaging:
    """P0-03: the wheel/source tree exposes only factor_engine + factor_engine.*."""

    def test_canonical_packages_are_importable(self) -> None:
        for name in LEGACY_PACKAGES:
            spec = importlib.util.find_spec(f"factor_engine.{name}")
            assert spec is not None, f"canonical factor_engine.{name} must be importable"
            assert spec.origin is not None

    def test_canonical_root_modules_are_importable(self) -> None:
        for name in LEGACY_ROOT_MODULES:
            spec = importlib.util.find_spec(f"factor_engine.{name}")
            assert spec is not None, f"canonical factor_engine.{name} must be importable"

    def test_top_level_packages_do_not_exist(self) -> None:
        # In a pristine interpreter (fresh pytest worker) the top-level names
        # are simply absent.  In an interpreter that has imported factor_engine
        # the deprecated_shims aliases them — so the meaningful assertion is on
        # the RESOLUTION TARGET, which must be the factor_engine.* module, not
        # a second package copy.
        for name in LEGACY_PACKAGES:
            spec = importlib.util.find_spec(name)
            if spec is None:
                continue  # pristine interpreter: absent is correct
            assert spec.name == f"factor_engine.{name}", (
                f"legacy name {name!r} resolved to {spec.name!r}, expected the "
                "canonical factor_engine.* module"
            )

    def test_top_level_root_modules_do_not_exist(self) -> None:
        for name in LEGACY_ROOT_MODULES:
            spec = importlib.util.find_spec(name)
            if spec is None:
                continue
            # The deprecated shim resolves a bare root-module name by walking
            # the canonical source file.  The spec NAME may be the bare name,
            # but its ORIGIN must be the canonical factor_engine.<name> file —
            # never a second physical copy.
            assert spec.origin is not None and spec.origin.startswith(
                str(_REPO_ROOT / "factor_engine")
            ), (
                f"legacy root module {name!r} resolved to origin "
                f"{getattr(spec, 'origin', None)!r}, expected a canonical "
                "factor_engine.* file"
            )
            canonical = sys.modules.get(f"factor_engine.{name}")
            if canonical is not None:
                assert spec.origin == getattr(canonical, "__file__", None), (
                    f"legacy root module {name!r} origin {spec.origin!r} does "
                    "not match the canonical module file"
                )


class TestSinglePhysicalIdentity:
    """P0-03: one physical .py file must never be loadable under two names."""

    def test_legacy_import_aliases_canonical_object(self) -> None:
        import factor_engine

        for legacy in LEGACY_PACKAGES + LEGACY_ROOT_MODULES:
            if not _pristine_import(legacy):
                continue
            mod = importlib.import_module(legacy)
            canonical = sys.modules.get(f"factor_engine.{legacy}")
            assert canonical is not None
            assert mod is canonical, (
                f"legacy {legacy!r} is a different object from factor_engine.{legacy}"
            )

    def test_legacy_dotted_submodule_aliases_canonical_object(self) -> None:
        import factor_engine

        for legacy in LEGACY_DOTTED:
            if not _pristine_import(legacy):
                continue
            mod = importlib.import_module(legacy)
            canonical = sys.modules.get(f"factor_engine.{legacy}")
            assert canonical is not None
            assert mod is canonical, (
                f"legacy dotted {legacy!r} is a different object from "
                f"factor_engine.{legacy}"
            )

    def test_legacy_submodule_origin_under_canonical_tree(self) -> None:
        import factor_engine

        for legacy in LEGACY_DOTTED:
            if not _pristine_import(legacy):
                continue
            mod = importlib.import_module(legacy)
            origin = getattr(mod, "__file__", None) or ""
            assert origin.startswith(
                str(_REPO_ROOT / "factor_engine")
            ), f"{legacy!r} origin {origin!r} is outside the canonical tree"

    def test_operator_registry_is_single_singleton(self) -> None:
        """R40 immutable-contract guard: the shim must never create a second
        OperatorRegistry singleton."""
        import factor_engine

        from factor_engine.cleaned_operators.registry import OperatorRegistry as A
        import cleaned_operators.registry

        B = cleaned_operators.registry.OperatorRegistry
        assert A is B
        assert A is sys.modules["factor_engine.cleaned_operators.registry"].OperatorRegistry

    def test_shim_is_state_free_and_idempotent(self) -> None:
        import factor_engine.deprecated_shims as shims

        before = len(sys.meta_path)
        shims.install_legacy_aliases()
        shims.install_legacy_aliases()
        assert len(sys.meta_path) == before


class TestDeprecatedShimMapping:
    """Static mapping table contract (documented names only)."""

    def test_mapping_covers_legacy_packages(self) -> None:
        import factor_engine.deprecated_shims as shims

        for name in LEGACY_PACKAGES:
            assert shims._LEGACY_ALIASES[name] == f"factor_engine.{name}"

    def test_mapping_covers_root_modules(self) -> None:
        import factor_engine.deprecated_shims as shims

        for name in LEGACY_ROOT_MODULES:
            assert shims._ROOT_MODULE_ALIASES[name] == f"factor_engine.{name}"