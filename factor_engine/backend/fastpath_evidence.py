# -*- coding: utf-8
"""Production fast path 证据链：production-safe 算子须绑定真实执行 parity case。"""
from __future__ import annotations

import importlib
from typing import Iterable

_META_OPS: frozenset[str] = frozenset(
    {"column", "literal", "materialized_series", "plan_ref", "if_else"}
)

# DSL 别名 → parity 代表 canonical
_EVIDENCE_ALIASES: dict[str, str] = {
    "if_else": "where",
    "rolling_beta": "ts_beta",
    "cum_std": "expanding_std",
    "WMA": "ts_decay_linear",
}

#  parametrized parity 用例来源（module, cases_attr）
_POLARS_PARITY_SOURCES: tuple[tuple[str, str], ...] = (
    ("tests.backend_parity.test_production_core_triple_parity", "MEMORY_CASES"),
    ("tests.backend_parity.test_p0_edge_cases_triple_parity", "EDGE_CASES"),
    ("tests.backend_parity.test_production_safe_bulk_parity", "POLARS_BULK_CASES"),
    ("tests.backend_parity.test_p1_pending_golden", "PENDING_GOLDEN_CASES"),
    ("tests.backend_parity.test_polars_long_no_pandas_path", "NO_PANDAS_CASES"),
    ("tests.backend_parity.test_batch2_rolling_triple_parity", "BATCH2_ROLLING_CASES"),
)

_DUCKDB_PARITY_SOURCES: tuple[tuple[str, str], ...] = (
    ("tests.backend_parity.test_production_core_triple_parity", "DUCKDB_CASES"),
    ("tests.backend_parity.test_p0_edge_cases_triple_parity", "DUCKDB_EDGE_CASES"),
    ("tests.backend_parity.test_production_safe_bulk_parity", "DUCKDB_BULK_CASES"),
    ("tests.backend_parity.test_p1_pending_golden", "PENDING_DUCKDB_CASES"),
    ("tests.backend_parity.test_batch2_rolling_triple_parity", "BATCH2_ROLLING_CASES"),
)


def _load_case_names(module_path: str, attr: str) -> frozenset[str]:
    """从测试模块加载 parity case 名称集合。"""
    mod = importlib.import_module(module_path)
    raw = getattr(mod, attr)
    return frozenset(str(name) for name, _ in raw)


def _expand_aliases(names: Iterable[str]) -> frozenset[str]:
    """将 DSL 别名展开为证据链 canonical 集合。"""
    out: set[str] = set()
    for name in names:
        out.add(name)
        for alias, target in _EVIDENCE_ALIASES.items():
            if name == target:
                out.add(alias)
    return frozenset(out)


def polars_executed_parity_canonicals() -> frozenset[str]:
    """PolarsLong 真实执行 parity case 覆盖的 canonical。"""
    out: set[str] = set()
    for mod, attr in _POLARS_PARITY_SOURCES:
        out |= _load_case_names(mod, attr)
    # polars_long_parity 历史用例
    try:
        from tests.backend_parity import test_polars_long_parity as plp

        out |= _load_case_names(plp.__name__, "POLARS_LONG_PARITY_CASES")
    except Exception:
        pass
    return _expand_aliases(out)


def duckdb_executed_parity_canonicals() -> frozenset[str]:
    """DuckDB 真实 SQL execute parity case 覆盖的 canonical。"""
    out: set[str] = set()
    for mod, attr in _DUCKDB_PARITY_SOURCES:
        out |= _load_case_names(mod, attr)
    return _expand_aliases(out)


def missing_polars_parity_evidence() -> list[str]:
    """列出 PolarsLong production-safe 但缺少 parity case 的 canonical。

    返回:
        按字母排序的缺失 canonical 名称列表。
    """
    from backend.polars_long_production import POLARS_LONG_NATIVE_PRODUCTION_SAFE

    required = POLARS_LONG_NATIVE_PRODUCTION_SAFE - _META_OPS
    have = polars_executed_parity_canonicals()
    return sorted(c for c in required if c not in have)


def missing_duckdb_parity_evidence() -> list[str]:
    """列出 DuckDB production-safe 但缺少 parity case 的 canonical。

    返回:
        按字母排序的缺失 canonical 名称列表。
    """
    from backend.sql_tiers import SQL_PRODUCTION_SAFE_CANONICALS

    required = SQL_PRODUCTION_SAFE_CANONICALS - _META_OPS
    have = duckdb_executed_parity_canonicals()
    return sorted(c for c in required if c not in have)


def evidence_summary() -> dict[str, int | list[str]]:
    """汇总 Polars/DuckDB parity 证据链覆盖情况。

    返回:
        含 required、evidence 计数及 missing 列表的摘要字典。
    """
    from backend.polars_long_production import POLARS_LONG_NATIVE_PRODUCTION_SAFE
    from backend.sql_tiers import SQL_PRODUCTION_SAFE_CANONICALS

    pol_req = POLARS_LONG_NATIVE_PRODUCTION_SAFE - _META_OPS
    duck_req = SQL_PRODUCTION_SAFE_CANONICALS - _META_OPS
    pol_have = polars_executed_parity_canonicals()
    duck_have = duckdb_executed_parity_canonicals()
    return {
        "polars_required": len(pol_req),
        "polars_evidence": len(pol_have & pol_req),
        "polars_missing": missing_polars_parity_evidence(),
        "duckdb_required": len(duck_req),
        "duckdb_evidence": len(duck_have & duck_req),
        "duckdb_missing": missing_duckdb_parity_evidence(),
    }
