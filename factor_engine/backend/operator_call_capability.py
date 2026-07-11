# -*- coding: utf-8
"""参数级算子 capability（按 canonical + 具体调用 + backend 判定）。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from backend.plan_params import int_mode_from_plan_node
from planner.logical_plan import PlanNode

BackendName = str  # polars_long | duckdb_sql | pandas | any


class CapabilityLevel(str, Enum):
    PRODUCTION = "production"
    RESEARCH = "research"
    FORBIDDEN = "forbidden"


@dataclass(frozen=True)
class CapabilityResult:
    level: CapabilityLevel
    reason: str = ""
    redirect_canonical: str | None = None

    @property
    def ok_production(self) -> bool:
        return self.level == CapabilityLevel.PRODUCTION


def _resolve_canon(node: PlanNode | None, canonical: str | None) -> str:
    if canonical:
        return canonical
    if node is None:
        return ""
    from cleaned_operators.registry import OperatorRegistry

    return OperatorRegistry._aliases.get(str(node.op or ""), str(node.op or ""))


def _literal_at(node: PlanNode, index: int) -> Any:
    if index >= len(node.inputs):
        return None
    child = node.inputs[index]
    if child.op != "literal":
        return None
    return child.attrs.get("value")


def _fillna_capability(node: PlanNode, *, production: bool) -> CapabilityResult:
    raw = _literal_at(node, 1)
    if raw is None:
        for key in ("method", "value", "fill_value"):
            if key in (node.attrs or {}) and node.attrs[key] is not None:
                raw = node.attrs[key]
                break
    if production:
        if raw is None:
            return CapabilityResult(CapabilityLevel.FORBIDDEN, "production 请使用 fillna_const")
        if isinstance(raw, str):
            method = raw.strip().lower()
            if method == "bfill":
                return CapabilityResult(CapabilityLevel.FORBIDDEN, "fillna(bfill) 永久禁止")
            if method == "ffill":
                return CapabilityResult(
                    CapabilityLevel.RESEARCH,
                    "production 请使用 ffill canonical，禁止 fillna(method='ffill')",
                )
        return CapabilityResult(
            CapabilityLevel.RESEARCH,
            "production 请使用 fillna_const（禁止通用 fillna canonical）",
            redirect_canonical="fillna_const",
        )
    if raw is None:
        return CapabilityResult(CapabilityLevel.FORBIDDEN, "fillna: 非 literal 填充值")
    if isinstance(raw, str):
        method = raw.strip().lower()
        if method == "bfill":
            return CapabilityResult(CapabilityLevel.FORBIDDEN, "fillna(bfill) 永久禁止")
        if method == "ffill":
            return CapabilityResult(CapabilityLevel.PRODUCTION, redirect_canonical="ffill")
        if method in {"mean", "median"}:
            return CapabilityResult(
                CapabilityLevel.RESEARCH,
                "fillna(method) 请使用 cs_fill_mean / fillna_const 等专用算子",
            )
        if method == "zero":
            return CapabilityResult(CapabilityLevel.PRODUCTION, redirect_canonical="fillna_const")
        return CapabilityResult(CapabilityLevel.RESEARCH, f"未知 fillna method {raw!r}")
    if isinstance(raw, (int, float)):
        if raw != raw or raw in (float("inf"), float("-inf")):
            return CapabilityResult(CapabilityLevel.FORBIDDEN, "fillna 填充值不能为 NaN/Inf")
        return CapabilityResult(CapabilityLevel.PRODUCTION, redirect_canonical="fillna_const")
    return CapabilityResult(CapabilityLevel.FORBIDDEN, f"fillna 不支持的填充值 {type(raw).__name__}")


def _cs_regression_capability(node: PlanNode, *, production: bool) -> CapabilityResult:
    mode = int_mode_from_plan_node(node, input_index=2, default=0)
    if mode not in {0, 1, 2}:
        return CapabilityResult(CapabilityLevel.FORBIDDEN, f"cs_regression mode={mode} 非法（须 0/1/2）")
    if production:
        from backend.production_fastpath_tiers import P1_REGRESSION_PARITY_PENDING

        if "cs_regression" in P1_REGRESSION_PARITY_PENDING:
            return CapabilityResult(
                CapabilityLevel.RESEARCH,
                f"cs_regression(mode={mode}) 第一阶段 pending，尚未 dual-backend 证据认证",
            )
    return CapabilityResult(CapabilityLevel.PRODUCTION, f"cs_regression mode={mode}")


def _winsorize_capability(node: PlanNode, *, production: bool) -> CapabilityResult:
    lo = _literal_at(node, 1)
    hi = _literal_at(node, 2)
    if lo is None and hi is None and not node.attrs:
        if production:
            return CapabilityResult(
                CapabilityLevel.RESEARCH,
                "winsorize 默认分位尚未 dual-backend 认证，请使用 group_winsorize 或显式分位",
            )
        return CapabilityResult(CapabilityLevel.RESEARCH, "winsorize 默认分位")
    return CapabilityResult(CapabilityLevel.RESEARCH, "winsorize 自定义分位 pending parity")


def _quantile_capability(node: PlanNode, *, production: bool) -> CapabilityResult:
    interp = (node.attrs or {}).get("interpolation", "linear")
    if production:
        return CapabilityResult(
            CapabilityLevel.FORBIDDEN,
            f"quantile(interpolation={interp!r}) 不在第一阶段 production（map_groups）",
        )
    return CapabilityResult(CapabilityLevel.RESEARCH, f"quantile interpolation={interp!r}")


def _scale_capability(node: PlanNode, *, backend: BackendName, production: bool) -> CapabilityResult:
    from backend.operator_evidence_schema import operator_evidence_record, supported_calls_match

    to_val = _literal_at(node, 1)
    if to_val is None:
        for key in ("to",):
            if key in (node.attrs or {}) and node.attrs[key] is not None:
                to_val = node.attrs[key]
                break
    if to_val is None:
        to_val = 1.0
    try:
        to_f = float(to_val)
    except (TypeError, ValueError):
        return CapabilityResult(CapabilityLevel.FORBIDDEN, f"scale(to={to_val!r}) 须为数值 literal")
    if production and to_f != 1.0:
        rec = operator_evidence_record("scale")
        supported = (rec or {}).get("supported_calls") if rec else None
        if not supported or not supported_calls_match(node, supported):
            return CapabilityResult(
                CapabilityLevel.RESEARCH,
                f"scale(to={to_f}) 未在 evidence supported_calls 中认证",
            )
    _ = backend
    return CapabilityResult(CapabilityLevel.PRODUCTION, f"scale(to={to_f})")


def _backend_evidence_ok(canon: str, backend: BackendName) -> bool:
    from backend.primitive_evidence import (
        DUCKDB_REAL_SQL_VERIFIED,
        NO_FALLBACK_VERIFIED,
        POLARS_REFERENCE_PARITY_VERIFIED,
    )

    if backend in {"polars_long", "polars", "polars_panel"}:
        return canon in POLARS_REFERENCE_PARITY_VERIFIED and canon in NO_FALLBACK_VERIFIED
    if backend in {"duckdb_sql", "sql", "duckdb"}:
        return canon in DUCKDB_REAL_SQL_VERIFIED
    return True


def check_operator_call_capability(
    canonical: str,
    *,
    node: PlanNode | None = None,
    args: Sequence[Any] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    backend: BackendName = "any",
    production: bool = False,
) -> CapabilityResult:
    """按 canonical + 具体调用 + backend 判定 capability。"""
    _ = (args, kwargs)
    canon = _resolve_canon(node, canonical)
    if canon == "ts_mad":
        return CapabilityResult(
            CapabilityLevel.RESEARCH,
            "ts_mad 为非标准双重滚动实现，请使用 ts_median_abs_deviation / ts_mean_abs_deviation",
        )
    if canon == "ts_product":
        return CapabilityResult(
            CapabilityLevel.RESEARCH,
            "ts_product 尚未完成 dual-backend 证据认证",
        )
    if canon == "fillna":
        if node is None:
            return CapabilityResult(CapabilityLevel.FORBIDDEN, "fillna 需要 plan 节点解析 method")
        return _fillna_capability(node, production=production)
    if canon == "scale":
        if node is None:
            return CapabilityResult(CapabilityLevel.RESEARCH, "scale 需要 plan 节点")
        return _scale_capability(node, backend=backend, production=production)
    if canon == "cs_regression":
        if node is None:
            return CapabilityResult(CapabilityLevel.RESEARCH, "cs_regression 需要 plan 节点解析 mode")
        return _cs_regression_capability(node, production=production)
    if canon == "winsorize":
        if node is None:
            return CapabilityResult(CapabilityLevel.RESEARCH, "winsorize 需要 plan 节点")
        return _winsorize_capability(node, production=production)
    if canon in {"quantile", "ts_quantile", "cs_quantile"}:
        if node is None:
            return CapabilityResult(CapabilityLevel.RESEARCH, f"{canon} 需要 plan 节点")
        return _quantile_capability(node, production=production)
    if production:
        from backend.phase1_scope import phase1_production_certified

        if not phase1_production_certified(canon):
            return CapabilityResult(
                CapabilityLevel.RESEARCH,
                f"{canon} 不在第一阶段 dual-backend 证据认证集",
            )
        if backend not in {"any", ""} and not _backend_evidence_ok(canon, backend):
            return CapabilityResult(
                CapabilityLevel.RESEARCH,
                f"{canon} 未通过 {backend} 证据认证",
            )
    return CapabilityResult(CapabilityLevel.PRODUCTION)


def operator_call_violations(
    node: PlanNode,
    *,
    canonical: str | None = None,
    production: bool = False,
) -> list[str]:
    """将 capability 结果转为 gate 违规字符串。"""
    canon = _resolve_canon(node, canonical)
    result = check_operator_call_capability(canon, node=node, production=production)
    if production and result.level == CapabilityLevel.FORBIDDEN:
        return [result.reason or f"{canon}: forbidden"]
    if production and result.level == CapabilityLevel.RESEARCH:
        return [result.reason or f"{canon}: not production"]
    if not production and result.level == CapabilityLevel.FORBIDDEN:
        return [result.reason or f"{canon}: forbidden"]
    return []


def check_plan_operator_calls(plan: Any, *, production: bool = False) -> list[str]:
    """深度遍历 plan，收集参数级 capability 违规。"""
    from backend.operator_types import check_plan_types

    _SKIP = frozenset({"column", "literal", "materialized_series", "plan_ref"})
    violations: list[str] = []
    violations.extend(check_plan_types(plan))

    def walk(node: Any) -> None:
        if not isinstance(node, PlanNode):
            return
        op = str(node.op or "")
        if not op or op in _SKIP:
            for child in node.inputs or []:
                walk(child)
            return
        from cleaned_operators.registry import OperatorRegistry

        canon = OperatorRegistry._aliases.get(op, op)
        violations.extend(
            operator_call_violations(node, canonical=canon, production=production)
        )
        for child in node.inputs or []:
            walk(child)

    walk(plan)
    return violations
