# -*- coding: utf-8 -*-
"""Point-in-Time safety audit for operator IR and logical SourceRef dependencies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ir.nodes import IRNode


class PitSafetyError(RuntimeError):
    def __init__(self, violations: list[str]):
        self.violations = violations
        super().__init__("PIT 安全审计失败: " + ", ".join(sorted(set(violations))))


@dataclass
class PitAuditReport:
    violations: list[str] = field(default_factory=list)
    checked_ops: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "violations": list(self.violations), "checked_ops": list(self.checked_ops)}


_FORWARD_FILL_OPS = frozenset({
    "ffill", "fillna_ffill", "forward_fill", "fill_forward",
    "bfill", "fillna_backfill", "backfill",
})
_POSITIVE_LAG_PARAMS: dict[str, tuple[str, int]] = {
    "ts_delay": ("n", 1), "ts_delta": ("n", 1), "ts_pct": ("d", 1),
}
_FINANCIAL_TABLES = frozenset({"StockIncome", "StockCashFlow", "StockBalance"})
_EXACT_DAILY_TABLES = frozenset({
    "DailyBar", "StockDailyBar", "BenchmarkIndexDailyBar", "SizeDaily", "EtfDailyBar",
})
_ASOF_DAILY_TABLES = frozenset({"IndustryDaily"})
_MINUTE_TABLES = frozenset({"StockMinuteBar", "MinuteBar"})


def _literal_number(node: IRNode, *, attr: str, input_index: int) -> float | None:
    raw = (node.attrs or {}).get(attr)
    if raw is None and input_index < len(node.inputs):
        child = node.inputs[input_index]
        if child.op == "literal":
            raw = (child.attrs or {}).get("value")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


def _audit_derived_field(
    field: str,
    *,
    forbid_forward_fill: bool,
    fail_on_missing: bool,
    violations: list[str],
    checked: list[str],
    source_guard: set[str],
) -> None:
    key = f"DerivedField:{field}"
    if key in source_guard:
        violations.append(f"{key}(cycle)")
        return
    source_guard.add(key)
    try:
        from runtime.derived_field_registry import load_derived_field_definition
        from api.dsl_parser import parse_factor
        from ir.analyzer import Analyzer

        definition = load_derived_field_definition(field)
        factor = parse_factor(
            definition.expression,
            name=f"pit::{field}@{definition.version}",
            surface="compat",
            dialect="lqtp",
            dialect_version="2026-07-19",
        )
        child_ir = Analyzer().lower(factor.expr).ir
        report = audit_ir(
            child_ir,
            forbid_forward_fill=forbid_forward_fill,
            fail_on_missing=fail_on_missing,
            _source_guard=source_guard,
        )
        checked.extend(f"{key}->{op}" for op in report.checked_ops)
        violations.extend(f"{key}->{item}" for item in report.violations)
    except Exception as exc:
        violations.append(f"{key}(unverifiable:{type(exc).__name__})")
    finally:
        source_guard.discard(key)


def _audit_source_ref(
    name: str,
    *,
    forbid_forward_fill: bool,
    fail_on_missing: bool,
    violations: list[str],
    checked: list[str],
    source_guard: set[str],
) -> bool:
    from api.source_ref import decode_source_ref

    spec = decode_source_ref(name)
    if spec is None:
        return False
    label = f"SourceRef[{spec.table}.{spec.field}]"
    checked.append(label)
    params, tparams = spec.params_dict(), spec.transform_params_dict()

    if spec.table in _EXACT_DAILY_TABLES:
        if spec.table == "BenchmarkIndexDailyBar" and not str(params.get("index", "")).strip():
            violations.append(f"{label}(missing_index)")
        if spec.transform is not None:
            violations.append(f"{label}(unexpected_transform={spec.transform})")
        return True
    if spec.table in _ASOF_DAILY_TABLES:
        if spec.transform not in {None, "asof_backward"}:
            violations.append(f"{label}(unsupported_transform={spec.transform})")
        return True
    if spec.table in _FINANCIAL_TABLES:
        if spec.transform not in {None, "financial_asof", "financial_lag"}:
            violations.append(f"{label}(unsupported_transform={spec.transform})")
        if spec.transform == "financial_lag":
            try:
                quarters = int(tparams.get("quarters", 1))
            except (TypeError, ValueError):
                quarters = 0
            if quarters <= 0:
                violations.append(f"{label}(financial_lag_quarters<=0)")
        return True
    if spec.table in _MINUTE_TABLES:
        if spec.transform not in {"minute_at", "minute_range", "minute_bar", "minute_resample"}:
            violations.append(f"{label}(minute_transform_required)")
            return True
        if spec.transform == "minute_at" and not str(tparams.get("hhmm", "")).strip():
            violations.append(f"{label}(missing_hhmm)")
        if spec.transform == "minute_range":
            start, end = str(tparams.get("start", "")), str(tparams.get("end", ""))
            if not start or not end or start >= end:
                violations.append(f"{label}(invalid_minute_range)")
        if spec.transform in {"minute_bar", "minute_resample"}:
            try:
                period, index = int(tparams.get("period", 1)), int(tparams.get("index", 0))
            except (TypeError, ValueError):
                period, index = 0, -1
            if period <= 0 or index < 0:
                violations.append(f"{label}(invalid_minute_period_or_index)")
        return True
    if spec.table == "Intermediate":
        try:
            from runtime.intermediate_registry import intermediate_dependency_lineage
            intermediate_dependency_lineage(str(params.get("name", "")), int(params.get("version", 0)))
        except Exception as exc:
            violations.append(f"{label}(unversioned:{type(exc).__name__})")
        return True
    if spec.table == "DerivedField":
        _audit_derived_field(
            spec.field,
            forbid_forward_fill=forbid_forward_fill,
            fail_on_missing=fail_on_missing,
            violations=violations,
            checked=checked,
            source_guard=source_guard,
        )
        return True
    if spec.table == "TurnoverBaseDaily":
        if spec.transform is not None:
            violations.append(f"{label}(unexpected_transform={spec.transform})")
        return True

    violations.append(f"{label}(unknown_availability_contract)")
    return True


def audit_ir(
    ir: IRNode,
    *,
    forbid_forward_fill: bool = False,
    fail_on_missing: bool = True,
    _source_guard: set[str] | None = None,
) -> PitAuditReport:
    from backend.cleaned_bridge import ensure_cleaned_loaded
    from cleaned_operators.operator_policy import infer_operator_policy
    from cleaned_operators.registry import OperatorRegistry

    ensure_cleaned_loaded()
    violations: list[str] = []
    checked: list[str] = []
    source_guard = _source_guard if _source_guard is not None else set()

    def walk(node: IRNode) -> None:
        if node.op == "column":
            _audit_source_ref(
                str((node.attrs or {}).get("name") or ""),
                forbid_forward_fill=forbid_forward_fill,
                fail_on_missing=fail_on_missing,
                violations=violations,
                checked=checked,
                source_guard=source_guard,
            )
            return
        if node.op in {"literal", "plan_ref"}:
            return
        checked.append(node.op)
        canon = OperatorRegistry.resolve_canonical_optional(node.op)
        op_impl = OperatorRegistry.get(canon)
        if op_impl is None:
            if fail_on_missing:
                violations.append(f"{canon}(missing_runtime)")
            return
        policy = infer_operator_policy(op_impl, canonical=canon)
        if not policy.pit_safe or policy.lag < 0:
            violations.append(canon)
        param_rule = _POSITIVE_LAG_PARAMS.get(canon)
        if param_rule is not None:
            param_name, input_index = param_rule
            value = _literal_number(node, attr=param_name, input_index=input_index)
            if value is not None and value < 1:
                violations.append(f"{canon}({param_name}={value:g})")
        if canon in {"bfill", "backfill", "fillna_backfill", "fillna_bfill", "lead", "Lead", "next"}:
            violations.append(f"{canon}(future_data)")
        if forbid_forward_fill and canon in _FORWARD_FILL_OPS:
            violations.append(f"{canon}(forward_fill)")
        for child in node.inputs:
            walk(child)

    walk(ir)
    return PitAuditReport(violations=violations, checked_ops=checked)


def assert_pit_safe(
    ir: IRNode,
    *,
    enforce: bool = True,
    forbid_forward_fill: bool = False,
    fail_on_missing: bool = True,
) -> PitAuditReport:
    report = audit_ir(ir, forbid_forward_fill=forbid_forward_fill, fail_on_missing=fail_on_missing)
    if enforce and not report.passed:
        raise PitSafetyError(report.violations)
    return report
