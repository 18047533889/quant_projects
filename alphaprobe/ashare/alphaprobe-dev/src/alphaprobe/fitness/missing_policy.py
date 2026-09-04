"""FactorFitness V2.1 —— metric missing policy（plan Task 8.1）。

把「单 metric 缺失语义」从 components 分支中独立出来，统一：
- OPTIONAL 缺失 → utility 0.5（维度内该子项取中性）。
- DIAGNOSTIC 缺失 → 无 score 影响（外层跳过该子项，保持 V2 的 utility=0
  但权重归一仍按原权重语义——见 components._weighted_sum 的跳过行为）。
- REQUIRED 缺失 → 不做降级替身；是否 promote 交给 gate 层
  （fitness/funnel.py）按 MetricRequirementPolicy 判定。

模块命名独立（不塞进 contracts/calibration），避免 contracts 增重。
"""

from __future__ import annotations

from typing import Any

from alphaprobe.fitness.contracts import MetricRequirement


def requirement_fallback_utility(
    requirement: MetricRequirement,
    *,
    key: str = "",
    calibrator: Any = None,
) -> float | None:
    """缺测 metric 的降级 utility（按 requirement 语义）。

    返回 None 表示「调用方跳过该子项」（DIAGNOSTIC / REQUIRED 都不在组件层
    制造替身分）；返回数值表示该子项取该中性分（OPTIONAL → 0.5）。
    """
    if requirement == MetricRequirement.OPTIONAL:
        return 0.5
    return None


def optional_fallback_utility(
    requirement: MetricRequirement,
    *,
    key: str = "",
    calibrator: Any = None,
) -> float:
    """OPTIONAL 缺失时的明确 0.5 工具（单测直接锚定语义）。"""
    return 0.5
