"""
风险模型构建器：从 BenchmarkSpec 提取风险模型相关字段。

将优化器的风险口径、目标标识、后端类型封装为字典，
供后续 Riskfolio-Lib 适配层构建实际风险模型时使用。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict

from .benchmark_router import BenchmarkSpec


def build_risk_model(spec: BenchmarkSpec) -> Dict[str, object]:
    """从 BenchmarkSpec 构建风险模型描述字典。

    Args:
        spec: 优化器映射描述

    Returns:
        包含 risk_mode、objective_id、backend 的字典。
    """
    return {
        "risk_mode": spec.risk_mode,
        "objective_id": spec.objective_id,
        "backend": spec.backend,
    }
