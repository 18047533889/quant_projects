# -*- coding: utf-8 -*-
"""Preserve declared SQL emitter implementations across load-order filtering."""
from __future__ import annotations

import ast
from pathlib import Path

from cleaned_operators.registry import OperatorRegistry

_INSTALLED = False
_DECLARED: frozenset[str] | None = None
_STRUCTURAL = frozenset({"column", "literal"})


def _declared_implemented() -> frozenset[str]:
    global _DECLARED
    if _DECLARED is not None:
        return _DECLARED
    from backend import sql_tiers

    tree = ast.parse(Path(sql_tiers.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        target = None
        value = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        if target != "SQL_IMPLEMENTED_CANONICALS" or not isinstance(value, ast.Call):
            continue
        if not value.args:
            continue
        literal = ast.literal_eval(value.args[0])
        _DECLARED = frozenset(str(name) for name in literal)
        return _DECLARED
    raise RuntimeError("unable to recover declared SQL implementation set")


def restore_declared_sql_implementations() -> None:
    from backend import sql_tiers
    from backend.sql_pushdown import emitter, sql_registry

    active = frozenset(OperatorRegistry._operators)
    runtime_markers = frozenset(
        name for name in active if "sql" in OperatorRegistry.backends_for(name)
    )
    implemented = frozenset(
        (set(_declared_implemented()) | set(runtime_markers)) & set(active)
    ) | _STRUCTURAL
    parity = frozenset(
        name for name in sql_tiers.SQL_PARITY_VERIFIED_CANONICALS
        if name in implemented
    )
    production = frozenset(
        name for name in sql_tiers.SQL_PRODUCTION_SAFE_CANONICALS
        if name in parity
    )

    sql_tiers.SQL_IMPLEMENTED_CANONICALS = implemented
    sql_tiers.SQL_CAPABLE_CANONICALS = implemented
    sql_tiers.SQL_PARITY_VERIFIED_CANONICALS = parity
    sql_tiers.SQL_PRODUCTION_SAFE_CANONICALS = production
    sql_tiers.DUCKDB_SQL_PARITY_VERIFIED = parity
    sql_tiers.DUCKDB_SQL_PRODUCTION_SAFE = production

    sql_registry.SQL_IMPLEMENTED_CANONICALS = implemented
    sql_registry.SQL_CAPABLE_CANONICALS = implemented
    sql_registry.SQL_PARITY_VERIFIED_CANONICALS = parity
    sql_registry.SQL_PRODUCTION_SAFE_CANONICALS = production
    emitter.SQL_CAPABLE_OPS = implemented


def install_declared_sql_restore() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from backend import active_capabilities

    original = active_capabilities.synchronize_active_backend_sets

    def synchronized() -> None:
        original()
        restore_declared_sql_implementations()

    active_capabilities.synchronize_active_backend_sets = synchronized
    restore_declared_sql_implementations()
    _INSTALLED = True


__all__ = ["install_declared_sql_restore", "restore_declared_sql_implementations"]
