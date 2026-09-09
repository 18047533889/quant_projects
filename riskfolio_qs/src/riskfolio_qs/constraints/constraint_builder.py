"""
约束构建器：将 ConstraintTemplate 转换为字典格式，供优化器使用。

是约束模板层与优化器之间的桥梁。
v0.2 中，实际约束参数主要由 ParameterStore 的 YAML 配置驱动，
此模块保留用于 rule_backend 优化器的约束回退。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict

from .constraint_templates import ConstraintTemplate, get_benchmark_constraint_template


class ConstraintBuilder:
    """约束构建器：按优化器名称查找模板并转为字典。"""

    def build(self, benchmark_name: str) -> Dict[str, object]:
        """根据优化器名称构建约束参数字典。

        Args:
            benchmark_name: 优化器名称

        Returns:
            约束参数字典（来自 ConstraintTemplate 的 dataclass 转换）。
        """
        template = get_benchmark_constraint_template(benchmark_name)
        return asdict(template)
