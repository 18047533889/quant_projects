# -*- coding: utf-8 -*-
"""Remove SQL implementations whose semantics no longer match audited runtime."""
from __future__ import annotations


def apply_active_sql_contracts() -> None:
    from backend import sql_tiers
    from backend.sql_pushdown import emitter, sql_registry

    removed = frozenset({"period_lag"})
    implemented = frozenset(sql_tiers.SQL_IMPLEMENTED_CANONICALS - removed)
    parity = frozenset(sql_tiers.SQL_PARITY_VERIFIED_CANONICALS - removed)
    production = frozenset(sql_tiers.SQL_PRODUCTION_SAFE_CANONICALS - removed)

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


__all__ = ["apply_active_sql_contracts"]
