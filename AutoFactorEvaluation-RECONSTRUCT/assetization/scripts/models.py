from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FactorCandidate:
    """GatewayPassBase candidate.json 结构 + 向后兼容字段。

    New (assetization.md): expr, config, born_timestamp, basic_info.
    Legacy compat: formula → expr, source_metadata → config.
    """

    expr: str
    config: dict[str, Any]
    born_timestamp: str = ""
    basic_info: dict[str, Any] = field(default_factory=dict)

    @property
    def formula(self) -> str:
        return self.expr

    @property
    def source_metadata(self) -> dict[str, Any]:
        return self.config

    @property
    def gateway_pass(self) -> bool:
        return True


@dataclass(frozen=True)
class FactorAsset:
    """正式因子资产对象。"""

    factor_id: str
    coordinates: dict[str, str]
    state: str  # landing / screening / materializing
    materialized_path: str | None = None


@dataclass
class PhysicalPlan:
    """优化后的物理执行计划。

    Attributes:
        dag: 执行计划的 DAG 结构（序列化 dict）。
        reused_nodes: CSE 消除的公共子表达式节点列表。
        backend: 执行后端标识（pandas / polars）。
    """

    dag: dict
    reused_nodes: list
    backend: str
    # Internal execution handles — not part of the spec, carried for execute_plan
    _plan_node: Any = field(default=None, repr=False)
    _ir_node: Any = field(default=None, repr=False)
