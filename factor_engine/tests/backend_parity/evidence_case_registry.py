# -*- coding: utf-8
"""Evidence case 注册表：parity 测试 case 名集合与六证交集（单一事实来源）。"""
from __future__ import annotations

from typing import Callable, Iterable

CaseList = list[tuple[str, Callable]]


def merge_case_lists(*lists: Iterable[tuple[str, Callable]]) -> CaseList:
    """按 canonical 名去重合并 case（先出现的 builder 保留）。"""
    out: dict[str, Callable] = {}
    for cases in lists:
        for name, builder in cases:
            out.setdefault(name, builder)
    return sorted(out.items(), key=lambda x: x[0])


def case_names(cases: CaseList) -> frozenset[str]:
    return frozenset(name for name, _ in cases)


# fusion / join 等多路径算子：evidence 须按 execution_variant 分别认证
EXECUTION_VARIANTS: dict[str, list[str]] = {
    "maximum": ["base_column_fused", "generic_binary_join"],
    "minimum": ["base_column_fused", "generic_binary_join"],
    "gt": ["base_column_fused", "generic_binary_join"],
    "lt": ["base_column_fused", "generic_binary_join"],
    "eq": ["base_column_fused", "generic_binary_join"],
    "ge": ["base_column_fused", "generic_binary_join"],
    "le": ["base_column_fused", "generic_binary_join"],
    "ne": ["base_column_fused", "generic_binary_join"],
    "add": ["nary_ts_fusion", "generic_binary_join", "nested_binary"],
    "rank": ["direct", "fused_ts_unary"],
    "rank_pct": ["direct", "fused_ts_unary"],
    "cs_pct_rank": ["direct", "fused_ts_unary"],
    "zscore": ["direct", "fused_ts_unary"],
    "normalize": ["direct", "fused_ts_unary"],
    "scale": ["direct", "fused_ts_unary"],
    "where": ["direct", "nested_conditional"],
    "WMA": ["alias_rewrite_ts_decay_linear"],
    "rolling_beta": ["alias_rewrite_ts_beta"],
    "if_else": ["alias_rewrite_where"],
}


# alias rewrite 须与 canonical 四端等价（见 test_alias_semantic_equivalence.py）
ALIAS_EQUIVALENCE_CASES: dict[str, str] = {
    "WMA": "ts_decay_linear",
    "rolling_beta": "ts_beta",
    "if_else": "where",
    "div_or_null": "protected_div",
}


def operator_evidence_meta(*, certified: frozenset[str]) -> dict[str, dict]:
    """为已认证 primitive 附加 execution_variant 元数据。"""
    meta: dict[str, dict] = {}
    for name in sorted(certified):
        variants = EXECUTION_VARIANTS.get(name)
        if variants:
            meta[name] = {"execution_variants": variants}
    return meta


def six_way_certified_names(
    *,
    polars_reference: frozenset[str],
    polars_edge: frozenset[str],
    duckdb_reference: frozenset[str],
    duckdb_real_sql: frozenset[str],
    duckdb_edge: frozenset[str],
    no_fallback: frozenset[str],
) -> frozenset[str]:
    return (
        polars_reference
        & polars_edge
        & duckdb_reference
        & duckdb_real_sql
        & duckdb_edge
        & no_fallback
    )


def build_evidence_payload(
    *,
    polars_reference: CaseList,
    polars_edge: CaseList,
    duckdb_reference: CaseList,
    duckdb_edge: CaseList,
    no_fallback: CaseList,
    duckdb_null_edge: CaseList | None = None,
    duckdb_nan_edge: CaseList | None = None,
    duckdb_inf_edge: CaseList | None = None,
    operators_meta: dict | None = None,
) -> dict:
    """由 case 列表生成 ``primitive_verified.json`` 载荷。"""
    polars_ref = case_names(polars_reference)
    polars_ed = case_names(polars_edge)
    duck_ref = case_names(duckdb_reference)
    duck_ed = case_names(duckdb_edge)
    duck_null = case_names(duckdb_null_edge) if duckdb_null_edge is not None else duck_ed
    duck_nan = case_names(duckdb_nan_edge) if duckdb_nan_edge is not None else frozenset()
    duck_inf = case_names(duckdb_inf_edge) if duckdb_inf_edge is not None else frozenset()
    no_fb = case_names(no_fallback)
    payload = {
        "schema_version": 2,
        "description": "Primitive production evidence（由 CI 测试 case 维护，禁止从 production 白名单反向生成）",
        "polars_reference_parity": sorted(polars_ref),
        "polars_edge_verified": sorted(polars_ed),
        "duckdb_reference_parity": sorted(duck_ref),
        "duckdb_real_sql_verified": sorted(duck_ref),
        "duckdb_edge_verified": sorted(duck_ed),
        "duckdb_null_edge_verified": sorted(duck_null),
        "duckdb_nan_edge_verified": sorted(duck_nan),
        "duckdb_inf_edge_verified": sorted(duck_inf),
        "no_fallback_verified": sorted(no_fb),
    }
    dual = six_way_certified_names(
        polars_reference=polars_ref,
        polars_edge=polars_ed,
        duckdb_reference=duck_ref,
        duckdb_real_sql=duck_ref,
        duckdb_edge=duck_ed,
        no_fallback=no_fb,
    )
    payload["_six_way_count"] = len(dual)
    if operators_meta is None:
        operators_meta = operator_evidence_meta(certified=dual)
    else:
        auto = operator_evidence_meta(certified=dual)
        merged = dict(operators_meta)
        for k, v in auto.items():
            merged.setdefault(k, {}).update(v)
        operators_meta = merged
    if operators_meta:
        payload["operators"] = operators_meta
    return payload
