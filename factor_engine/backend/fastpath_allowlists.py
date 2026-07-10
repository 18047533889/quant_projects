# -*- coding: utf-8
"""Production fast path 三层 allowlist（research / production / fastpath）。"""
from __future__ import annotations

from cleaned_operators.operator_spec import PRODUCTION_CORE_CANONICALS, build_operator_spec


def _resolve(canon: str) -> str:
    from cleaned_operators.registry import OperatorRegistry

    return OperatorRegistry._aliases.get(canon, canon)


def research_allowlist() -> frozenset[str]:
    """研究挖因子：有 runtime 实现的 canonical。"""
    from cleaned_operators.registry import OperatorRegistry

    out: set[str] = set()
    for canon in OperatorRegistry.list_canonical():
        if OperatorRegistry.backends_for(canon):
            out.add(_resolve(canon))
    return frozenset(out)


def production_allowlist() -> frozenset[str]:
    """production 可运行：``allow_in_production``。"""
    from cleaned_operators.operator_spec import iter_operator_specs

    return frozenset(s.canonical for s in iter_operator_specs() if s.allow_in_production)


def production_fastpath_allowlist(*, strict: bool = True) -> frozenset[str]:
    """production 高性能：duckdb effective production_safe 或 polars_long native production_safe。"""
    from backend.fastpath_coverage import build_fastpath_coverage_row

    base = production_allowlist() if not strict else production_allowlist()
    out: set[str] = set()
    for canon in base:
        row = build_fastpath_coverage_row(canon)
        if row.production_fast_path:
            out.add(canon)
    if strict:
        # 核心算子子集优先显式声明
        for c in PRODUCTION_CORE_CANONICALS:
            if c in {"column", "literal"}:
                continue
            row = build_fastpath_coverage_row(c)
            if row.production_fast_path:
                out.add(_resolve(c))
    return frozenset(out)


def research_fastpath_allowlist() -> frozenset[str]:
    """research 挖因子：SQL 已实现且 PolarsLong/SQL emitter 可走 fast path 的 canonical。"""
    from backend.fastpath_coverage import build_fastpath_coverage_row
    from backend.production_fastpath_tiers import resolve_polars_native_canonical
    from backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    out: set[str] = set()
    for canon in SQL_IMPLEMENTED_CANONICALS:
        if canon in {"column", "literal"}:
            continue
        from cleaned_operators.registry import OperatorRegistry

        candidates = {
            canon,
            resolve_polars_native_canonical(canon),
            OperatorRegistry._aliases.get(canon, canon),
        }
        ok = False
        for check in candidates:
            row = build_fastpath_coverage_row(check)
            if (
                row.polars_long_native_production_safe
                or (row.duckdb_sql_production_safe and row.sql_emitter_ok)
                or row.polars_long_tier == "passthrough"
            ):
                ok = True
                break
        if ok:
            out.add(canon)
            out.add(_resolve(canon))
    return frozenset(out)


def is_research_allowed(canon: str) -> bool:
    return _resolve(canon) in research_allowlist()


def is_production_allowed(canon: str) -> bool:
    spec = build_operator_spec(_resolve(canon))
    return bool(spec and spec.allow_in_production)


def is_production_fastpath_allowed(canon: str, *, strict: bool = True) -> bool:
    from backend.production_fastpath_gate import check_production_fastpath_plan_ops
    from backend.sql_pushdown.plan_fixtures import minimal_plan

    name = _resolve(canon)
    if name in {"column", "literal"}:
        return True
    try:
        plan = minimal_plan(name)
    except Exception:
        from planner.logical_plan import PlanNode

        from backend.sql_pushdown.plan_fixtures import column

        plan = PlanNode(op=name, inputs=[column("close")], attrs={"d": 3})
    result = check_production_fastpath_plan_ops(plan, strict=strict)
    return result.ok
