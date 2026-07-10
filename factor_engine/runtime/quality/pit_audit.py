# -*- coding: utf-8
"""Point-in-Time 安全审计：compile 后检查 IR 是否含未来函数 / 非因果算子。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ir.nodes import IRNode


class PitSafetyError(RuntimeError):
    """PIT 门禁未通过。"""

    def __init__(self, violations: list[str]):
        self.violations = violations
        super().__init__(
            "PIT 安全审计失败，含非因果算子: " + ", ".join(sorted(set(violations)))
        )


@dataclass
class PitAuditReport:
    """PIT 安全审计报告。"""

    violations: list[str] = field(default_factory=list)
    checked_ops: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """是否无违规算子。"""
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 友好字典。"""
        return {
            "passed": self.passed,
            "violations": list(self.violations),
            "checked_ops": list(self.checked_ops),
        }


_FORWARD_FILL_OPS = frozenset({
    "ffill",
    "fillna_ffill",
    "forward_fill",
    "fill_forward",
    "bfill",
    "fillna_backfill",
    "backfill",
})


def audit_ir(
    ir: IRNode,
    *,
    forbid_forward_fill: bool = False,
    fail_on_missing: bool = True,
) -> PitAuditReport:
    """遍历 IR，收集 ``pit_safe=False``、负 lag 或缺失 runtime 的算子。"""
    from backend.cleaned_bridge import ensure_cleaned_loaded
    from cleaned_operators.operator_policy import infer_operator_policy
    from cleaned_operators.registry import OperatorRegistry

    ensure_cleaned_loaded()
    violations: list[str] = []
    checked: list[str] = []

    def walk(node: IRNode) -> None:
        if node.op in ("column", "literal", "plan_ref"):
            return
        checked.append(node.op)
        canon = OperatorRegistry._aliases.get(node.op, node.op)
        op_impl = OperatorRegistry.get(canon)
        if op_impl is None:
            if fail_on_missing:
                violations.append(f"{canon}(missing_runtime)")
            return
        policy = infer_operator_policy(op_impl, canonical=canon)
        if not policy.pit_safe or policy.lag < 0:
            violations.append(canon)
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
    """审计 IR；``enforce=True`` 时未通过则抛 :class:`PitSafetyError`。"""
    report = audit_ir(
        ir,
        forbid_forward_fill=forbid_forward_fill,
        fail_on_missing=fail_on_missing,
    )
    if enforce and not report.passed:
        raise PitSafetyError(report.violations)
    return report
