"""Recursive import manifest for the shipped factor_assets production surface."""

from __future__ import annotations

import importlib
import pkgutil

import factor_assets


_EXCLUDED_SUBTREES = {
    "factor_assets.similarity",
    "factor_assets.seen_index",
    "factor_assets.tests",
    "factor_assets.setup",
}


def _production_modules() -> list[str]:
    modules = []
    for module_info in pkgutil.walk_packages(
        factor_assets.__path__, prefix="factor_assets."
    ):
        name = module_info.name
        if any(
            name == excluded or name.startswith(excluded + ".")
            for excluded in _EXCLUDED_SUBTREES
        ):
            continue
        modules.append(name)
    return sorted(modules)


def test_shipped_production_packages_import_recursively() -> None:
    """Every shipped module must be importable from a clean package surface."""
    failures = {}
    for name in _production_modules():
        try:
            importlib.import_module(name)
        except Exception as exc:  # report all broken modules in one test run
            failures[name] = f"{type(exc).__name__}: {exc}"

    assert not failures, "Broken production imports:\n" + "\n".join(
        f"- {name}: {error}" for name, error in failures.items()
    )


def test_namespace_capability_markers_match_shipped_surfaces() -> None:
    """Production and incomplete namespaces must expose truthful capabilities."""
    assembly = importlib.import_module("factor_assets.assembly")
    assert assembly.FactorSetAssembler is not None
    assert not hasattr(assembly, "RESEARCH_ONLY")
    assert importlib.import_module("factor_assets.campaigns").RESEARCH_ONLY is True
