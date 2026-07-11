"""物理计划占位（尚未接入编译链；当前执行仍用 ``logical_plan.PlanNode``）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PhysicalPlan:
    op: str
    attrs: dict[str, Any] = field(default_factory=dict)
