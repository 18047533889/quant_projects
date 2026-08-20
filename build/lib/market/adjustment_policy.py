# -*- coding: utf-8 -*-
"""AdjustmentPolicy / AdjustmentVintage —— 复权因子的 PIT 语义（R40 #230）。

后复权（RETROSPECTIVE_ADJUSTED）用**今日已知全部**公司行动的因子回写整个历史，
把未来事件（后面的除权、配股、分红）提前泄漏进历史价格。生产实时因子必须用
PIT_ADJUSTED —— 每个历史时点只用**那时已知**的复权因子（复权因子的
knowledge_time 即 AdjustmentVintage）。

硬规则：
* ``RETROSPECTIVE_ADJUSTED`` 允许 research（离线回溯），禁止 production 实时
  因子——production 用到 RETROSPECTIVE 时 hard fail 或显式拒绝；
* ``PIT_ADJUSTED`` 绑定 ``adjustment_knowledge_time``：价格数据源的每个复权
  快照只能使用 ``knowledge_time <= decision_time`` 的 AdjustmentVintage。
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any


class AdjustmentPolicy(str, enum.Enum):
    RETROSPECTIVE_ADJUSTED = "retrospective_adjusted"
    PIT_ADJUSTED = "pit_adjusted"


class AdjustmentPolicyViolation(ValueError):
    """R40 #230：production 实时因子使用后复权 → 拒绝。"""


@dataclass(frozen=True)
class AdjustmentVintage:
    """复权因子的一个 PIT 可及快照。

    ``as_of`` 是复权因子数据本身生效/公布的时点；``factor_version`` 是因子
    序列的 revision（公司行动更正触发新版本）。
    """

    as_of: str
    factor_version: str = ""
    provider_snapshot: str = ""

    def available_at(self, decision_time: str) -> bool:
        """decision_time 时该 vintage 是否已可及（as_of <= decision_time）。"""
        return decision_time >= self.as_of

    def identity_payload(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "factor_version": self.factor_version,
            "provider_snapshot": self.provider_snapshot,
        }

    def identity_hash(self) -> str:
        import hashlib

        raw = ",".join(f"{k}={v}" for k, v in sorted(self.identity_payload().items()))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def validate_adjustment_policy_for_production(policy: AdjustmentPolicy | str) -> None:
    """production 实时因子检查：RETROSPECTIVE_ADJUSTED → hard fail。"""
    p = AdjustmentPolicy(policy) if not isinstance(policy, AdjustmentPolicy) else policy
    if p is AdjustmentPolicy.RETROSPECTIVE_ADJUSTED:
        raise AdjustmentPolicyViolation(
            "production real-time factors must not use "
            "RETROSPECTIVE_ADJUSTED prices — the adjustment factor embeds "
            "future corporate actions into history. Use PIT_ADJUSTED with an "
            "explicit AdjustmentVintage(knowledge_time)."
        )


def vintage_binds_knowledge_time(vintage: AdjustmentVintage | None) -> str:
    """price data source 绑定 ``adjustment_knowledge_time`` 的辅助函数。"""
    return vintage.as_of if vintage is not None else ""


__all__ = [
    "AdjustmentPolicy",
    "AdjustmentPolicyViolation",
    "AdjustmentVintage",
    "validate_adjustment_policy_for_production",
    "vintage_binds_knowledge_time",
]
