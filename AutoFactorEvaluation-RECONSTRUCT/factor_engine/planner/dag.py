from dataclasses import dataclass, field

from .logical_plan import PlanNode


@dataclass
class FactorPlan:
    factor_name: str
    root: PlanNode


@dataclass
class DAGPlan:
    roots: list[FactorPlan]
    shared_nodes: dict[str, PlanNode] = field(default_factory=dict)
