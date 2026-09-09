"""
目标函数构建器：从 BenchmarkSpec 提取目标函数相关字段。

将 BenchmarkSpec 中的优化器名称、目标标识、风险口径等
封装为字典，供后续 Riskfolio-Lib 适配层使用。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict

from .benchmark_router import BenchmarkSpec


def build_objective(spec: BenchmarkSpec) -> Dict[str, object]:
    """从 BenchmarkSpec 构建目标函数描述字典。

    Args:
        spec: 优化器映射描述

    Returns:
        包含 optimizer_name、objective_id、risk_mode 和完整 params 的字典。
    """
    return {
        "optimizer_name": spec.name,
        "objective_id": spec.objective_id,
        "risk_mode": spec.risk_mode,
        "params": asdict(spec),
    }
