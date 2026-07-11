"""
AutoFactorEvaluation 初始化模块。

服务启动时执行:
  1. 加载算子库（内建或外部）→ 注册到 OperatorRegistry
  2. 通过 DeepSeek 生成临时算子复杂度评分表
  3. 确保目标路径结构完整

路径原则：所有路径必须显式传入，无默认值。
"""

from __future__ import annotations

from .operator_loader import load_operators, get_registered_operators, ensure_factor_engine_path
from .complexity_builder import build_complexity_table
from .init_core import init_service, get_init_state, get_cache_path

__all__ = [
    "load_operators",
    "get_registered_operators",
    "ensure_factor_engine_path",
    "build_complexity_table",
    "init_service",
    "get_init_state",
    "get_cache_path",
]
