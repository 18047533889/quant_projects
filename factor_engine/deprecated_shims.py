"""P0-03 deprecated legacy-name shims (co-located, state-free).

FactorEngine ships with a SINGLE importable identity: ``factor_engine.*``
(P0-03).  The historical unqualified top-level package names (``api``,
``backend``, ``cache``, ``cleaned_operators``, ``runtime``, ``planner``, ...)
are no longer distributed.  To keep legacy ``import runtime`` /
``from api import ...`` style code working without resurrecting a SECOND
``sys.modules`` identity (which would create two ``OperatorRegistry``
singletons and violate the R40 immutable-registry contract), this module
installs a small meta-path finder that translates any unqualified legacy name
into its ``factor_engine.*`` canonical module.

Contract:
- Mappings are statically declared below (no dynamic scanning of installed
  distributions, no package metadata).
- Each mapping carries NO module state: ``factor_engine.runtime`` is imported
  (or reused) and aliased into ``sys.modules`` under the legacy name, so the
  two plain names refer to the SAME module object — one object, not two copies.
  In particular the ``OperatorRegistry`` singleton stays unique.
- The finder is installed lazily on first import of this shim module and is
  idempotent.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys

#: Per-module cache keyed by legacy name for the explicit module-alias branch
#: (the target spec and its canonical name are resolved at most once).
_AliasCache = {}

#: Static legacy-name -> factor_engine.* canonical mapping.  Keep minimal and
#: explicit; do not add names that were never top-level packages.
_LEGACY_ALIASES = {
    "api": "factor_engine.api",
    "backend": "factor_engine.backend",
    "cache": "factor_engine.cache",
    "cleaned_operators": "factor_engine.cleaned_operators",
    "docs": "factor_engine.docs",
    "evidence": "factor_engine.evidence",
    "execution": "factor_engine.execution",
    "export": "factor_engine.export",
    "expr": "factor_engine.expr",
    "factor_recipes": "factor_engine.factor_recipes",
    "fields": "factor_engine.fields",
    "ir": "factor_engine.ir",
    "market": "factor_engine.market",
    "mining": "factor_engine.mining",
    "operator_contracts": "factor_engine.operator_contracts",
    "planner": "factor_engine.planner",
    "planning": "factor_engine.planning",
    "research_operators": "factor_engine.research_operators",
    "research_tools": "factor_engine.research_tools",
    "runtime": "factor_engine.runtime",
    "security": "factor_engine.security",
    "semantic": "factor_engine.semantic",
    "service": "factor_engine.service",
    "storage": "factor_engine.storage",
    "telemetry": "factor_engine.telemetry",
    "util": "factor_engine.util",
    "validation": "factor_engine.validation",
    "audit": "factor_engine.audit",
    "modeling": "factor_engine.modeling",
}

#: Non-package top-level modules of the distribution root.  Some of these names
#: collide with a submodule of an installed package (e.g. ``logging_utils`` is
#: BOTH a top-level module and ``factor_engine.util.logging_utils``); the
#: canonical resolution for a bare ``import logging_utils`` is the top-level
#: module, which is the form that pre-P0-03 codebases imported.
_ROOT_MODULE_ALIASES = {
    "logging_utils": "factor_engine.logging_utils",
    "pit_contract": "factor_engine.pit_contract",
    "stateful_contract": "factor_engine.stateful_contract",
    "stateful_runtime": "factor_engine.stateful_runtime",
    "workspace_paths": "factor_engine.workspace_paths",
    "logging_config": "factor_engine.logging_config",
    "ohlc_contract": "factor_engine.ohlc_contract",
    "parameter_canonicalizer": "factor_engine.parameter_canonicalizer",
    "pipeline": "factor_engine.pipeline",
    "pipeline_event": "factor_engine.pipeline_event",
    "polars_backend_kind": "factor_engine.polars_backend_kind",
    "recursive_kernel": "factor_engine.recursive_kernel",
    "stateful_kernel": "factor_engine.stateful_kernel",
}


def _canonical_target(name: str) -> str | None:
    exact = _ROOT_MODULE_ALIASES.get(name)
    if exact is not None:
        return exact
    if name in _LEGACY_ALIASES:
        return _LEGACY_ALIASES[name]
    # Strip any accidentally qualified legacy-style prefix (defensive).
    if name.startswith("modeling."):
        return "factor_engine.modeling" + name[len("modeling") :]
    return None


def _canonical_head(head: str) -> str | None:
    if head in _ROOT_MODULE_ALIASES or head in _LEGACY_ALIASES:
        return _canonical_target(head)
    return "factor_engine." + head


class _AliasLoader(importlib.abc.Loader):
    """Loader whose sole job is to point ``sys.modules[legacy]`` at the
    canonical module object, then raise ImportError to stop further module
    initialization (the alias entry is already present)."""

    def __init__(self, legacy_name: str, canonical_name: str) -> None:
        self.legacy_name = legacy_name
        self.canonical_name = canonical_name

    def create_module(self, spec):
        # No module is created; the alias entry is the canonical object.
        return None

    def exec_module(self, module):
        # The module object backing ``sys.modules[legacy]`` is the canonical
        # object already registered by the finder.  Nothing to execute.
        raise ImportError(
            f"{self.legacy_name} is an alias for {self.canonical_name}; "
            "the alias was installed in sys.modules"
        )


class _LegacyAliasFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder mapping legacy unqualified names to factor_engine.*."""

    def _target_for(self, fullname: str) -> str | None:
        # A dotted name is only ever legacy when its head is a KNOWN legacy
        # package AND the remainder is a module that exists under the canonical
        # namespace; skip submodules of unrelated third-party packages that
        # merely share the head substring (e.g. ``polars.io``).
        head = fullname.split(".", 1)[0]
        if "." in fullname and head not in _LEGACY_ALIASES and head not in _ROOT_MODULE_ALIASES:
            return None
        exact = _canonical_target(fullname)
        if exact is not None:
            return exact
        if head in _LEGACY_ALIASES or head in _ROOT_MODULE_ALIASES:
            return "factor_engine." + fullname
        return None

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "deprecated_shims":
            return None
        head = fullname.split(".", 1)[0]
        if head in ("pickle", "_pickle"):
            return None
        if head == "polars":
            return None
        if fullname in _ROOT_MODULE_ALIASES:
            # Explicit root-module aliases (``logging_utils`` etc.).
            cache_key = ("_alias_target", fullname)
            if cache_key not in _AliasCache:
                canonical_name = _ROOT_MODULE_ALIASES[fullname]
                target = importlib.util.find_spec(canonical_name)
                _AliasCache[cache_key] = (
                    canonical_name,
                    target,
                )
            canonical_name, target = _AliasCache[cache_key]
            if target is not None and target.origin is not None:
                canonical_module = sys.modules.get(canonical_name)
                if canonical_module is not None:
                    # Canonical already loaded: alias it and return an alias
                    # loader spec so the core uses the SAME object and does not
                    # re-execute the file.
                    sys.modules[fullname] = canonical_module
                    return importlib.util.spec_from_loader(
                        fullname, _AliasLoader(fullname, canonical_name)
                    )
                # Canonical NOT loaded: the CURRENT import is under the legacy
                # name.  Wrap the canonical source loader so the module gets
                # registered under the canonical name in ``sys.modules`` during
                # execution, and then alias the legacy name to it on success.
                inner = importlib.machinery.SourceFileLoader(
                    canonical_name, target.origin
                )

                def _exec_and_register(inner, canonical_name, legacy_name):
                    def exec_module(self, module):
                        # The source loader works against ITS OWN spec name and
                        # its own module object; the core's ``module`` argument
                        # (created from the legacy-name spec) is only a holder.
                        # Create the canonical module from the inner loader,
                        # register it under BOTH names BEFORE exec (so
                        # dataclasses / circular imports inside the module see
                        # their canonical sys.modules entry), then exec.
                        spec = importlib.util.spec_from_loader(
                            canonical_name,
                            inner,
                            origin=inner.get_filename(),
                            is_package=(
                                module.__spec__.submodule_search_locations
                                is not None
                            ),
                        )
                        canonical_module = importlib.util.module_from_spec(spec)
                        sys.modules[canonical_name] = canonical_module
                        sys.modules[legacy_name] = canonical_module
                        try:
                            inner.exec_module(canonical_module)
                        finally:
                            # ``module`` is the core-created holder; the real
                            # module object is ``canonical_module``.  Keep the
                            # aliases pointing at the real object.
                            sys.modules[canonical_name] = canonical_module
                            sys.modules[legacy_name] = canonical_module

                    return exec_module

                exec_module_impl = _exec_and_register(
                    inner, canonical_name, fullname
                )
                if target.submodule_search_locations is not None:
                    import types as _types

                    loader = _types.SimpleNamespace(
                        create_module=lambda spec: None,
                        exec_module=lambda module: exec_module_impl(
                            None, module
                        ),
                        __init__=lambda module, name=None, spec=None: None,
                        is_package=lambda: True,
                        get_filename=lambda: target.origin,
                    )
                else:
                    import types as _types

                    loader = _types.SimpleNamespace(
                        create_module=lambda spec: None,
                        exec_module=lambda module: exec_module_impl(
                            None, module
                        ),
                        get_filename=lambda: target.origin,
                    )
                try:
                    return importlib.util.spec_from_loader(
                        fullname,
                        loader,
                        origin=target.origin,
                        is_package=(
                            target.submodule_search_locations is not None
                        ),
                    )
                except Exception:
                    return None
            return None
        target_name = self._target_for(fullname)
        if target_name is None:
            return None
        canonical = sys.modules.get(target_name)
        if canonical is None:
            try:
                __import__(target_name)
            except ImportError:
                return None
            canonical = sys.modules.get(target_name)
        if canonical is None:
            return None

        if "." in fullname:
            # Dotted alias: alias the PARENT first so a later submodule import
            # resolves through the canonical tree.
            head = fullname.split(".", 1)[0]
            canonical_head = _canonical_head(head)
            parent = sys.modules.get(canonical_head)
            if parent is not None and head not in sys.modules:
                sys.modules[head] = parent

        # Alias the full legacy name to the SAME canonical module object.
        sys.modules[fullname] = canonical
        # Provide a no-op loader spec so the CORE import machinery does not
        # re-execute the module under the legacy name (the alias entry is
        # already authoritative).
        try:
            spec = importlib.util.spec_from_loader(
                fullname,
                loader=_AliasLoader(fullname, target_name),
            )
        except Exception:
            return None
        return spec


_installed = False


def _make_finder() -> _LegacyAliasFinder:
    return _LegacyAliasFinder()


def install_legacy_aliases() -> None:
    """Register the legacy-name alias finder exactly once (idempotent).

    The finder is inserted IMMEDIATELY BEFORE ``PathFinder`` (i.e. after the
    builtin and frozen importers).  That ordering is essential: for a legacy
    DOTTED name (``cleaned_operators.registry``) the parent has already been
    aliased to the canonical package, so its ``__path__`` points at the
    canonical directory and a later ``PathFinder`` would resolve the ``.py``
    file directly and create a SECOND module object for the same file under the
    legacy name — exactly the dual-identity bug P0-03 forbids.  Intercepting
    before ``PathFinder`` hands the core one alias spec instead.  Non-legacy
    imports are untouched (the finder returns ``None`` immediately for names it
    does not own).
    """
    global _installed
    if _installed:
        return
    if not any(isinstance(f, _LegacyAliasFinder) for f in sys.meta_path):
        # Insert after builtin & frozen importers (indexes 0..2) but before
        # PathFinder.  Do not shadow ``importlib``/``sys``/frozen modules.
        # NOTE: in some interpreters the standard importer objects appear on
        # ``sys.meta_path`` as the CLASS objects themselves rather than
        # instances, so compare by identity (``is PathFinder``), not
        # ``isinstance``.
        from importlib.machinery import BuiltinImporter, FrozenImporter, PathFinder

        insert_at = len(sys.meta_path)
        for i, finder in enumerate(sys.meta_path):
            if (
                finder is BuiltinImporter
                or finder is FrozenImporter
                or type(finder) in (BuiltinImporter, FrozenImporter)
            ):
                insert_at = i + 1
            if finder is PathFinder or type(finder) is PathFinder:
                insert_at = i
                break
        finder_inst = _LegacyAliasFinder()
        sys.meta_path.insert(insert_at, finder_inst)
        _installed = True
    else:
        _installed = True


install_legacy_aliases()
__all__ = ["install_legacy_aliases", "_LEGACY_ALIASES", "_ROOT_MODULE_ALIASES"]