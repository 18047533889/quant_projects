# -*- coding: utf-8 -*-
"""Coordinate SQL/native scan decisions with logical SourceRef I/O."""
from __future__ import annotations
from typing import Any, Iterable
from planner.logical_plan import PlanNode
from planner.sql_lowerer import lower_to_physical_plan


def _contains_source_ref(plan: PlanNode) -> bool:
    if plan.op == "column":
        try:
            from api.source_ref import decode_source_ref
            if decode_source_ref(str((plan.attrs or {}).get("name", ""))) is not None:
                return True
        except Exception:
            pass
    return any(_contains_source_ref(child) for child in plan.inputs)


def plan_is_fully_sql(plan: PlanNode) -> bool:
    return not _contains_source_ref(plan) and lower_to_physical_plan(plan).fully_sql


def should_skip_column_prefetch(plans: Iterable[PlanNode], *, input_dq_check: bool=False,
                                backend: Any | None=None) -> bool:
    if input_dq_check: return False
    nodes=list(plans)
    if not nodes: return False
    # SourceRefs require runtime source resolution even when the chosen backend
    # normally prefers a direct native scan.
    if any(_contains_source_ref(p) for p in nodes): return False
    if backend is not None and getattr(backend,"prefers_native_scan",False): return True
    return all(plan_is_fully_sql(p) for p in nodes)
