# -*- coding: utf-8
"""向后兼容：参数级 policy 转发至 operator_call_capability。"""
from __future__ import annotations

from factor_engine.backend.operator_call_capability import (  # noqa: F401
    CapabilityLevel,
    CapabilityResult,
    check_operator_call_capability,
    check_plan_operator_calls,
    operator_call_violations,
)


def is_operator_call_production_safe(node, *, canonical: str | None = None) -> bool:
    from factor_engine.backend.operator_call_capability import check_operator_call_capability
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    canon = canonical
    if canon is None:
        canon = OperatorRegistry._aliases.get(str(getattr(node, "op", "") or ""), str(getattr(node, "op", "")))
    return check_operator_call_capability(canon, node=node, production=False).level != CapabilityLevel.FORBIDDEN


def fillna_call_production_safe(node):
    from factor_engine.backend.operator_call_capability import check_operator_call_capability

    r = check_operator_call_capability("fillna", node=node, production=False)
    ok = r.level != CapabilityLevel.FORBIDDEN
    return ok, r.reason
