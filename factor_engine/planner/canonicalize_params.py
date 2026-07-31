# -*- coding: utf-8 -*-
"""Normalize legacy parameter spellings after all plan rewrites.

Analyzer canonicalizes user-authored keyword aliases before IR construction, but
optimizer/composite rules can synthesize new ``PlanNode.attrs`` later.  This
final pass guarantees execution layers only receive canonical parameters.
"""
from __future__ import annotations

from planner.logical_plan import PlanNode


def canonicalize_plan_parameters(plan: PlanNode) -> PlanNode:
    """Return a structurally equivalent plan with canonical attrs everywhere."""
    from backend.parameter_aliases import normalize_parameter_aliases
    from cleaned_operators.registry import OperatorRegistry

    inputs = [canonicalize_plan_parameters(child) for child in plan.inputs]
    op = str(plan.op)
    attrs = dict(plan.attrs)
    if op not in {"column", "literal", "plan_ref", "materialized_series"}:
        canonical = OperatorRegistry._aliases.get(op, op)
        attrs = normalize_parameter_aliases(canonical, attrs)
        op = canonical
    return PlanNode(op=op, inputs=inputs, attrs=attrs, node_id=plan.node_id)
