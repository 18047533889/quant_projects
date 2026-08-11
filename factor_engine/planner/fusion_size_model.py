# -*- coding: utf-8 -*-
"""R39-P0-PERF-021: 可测量变量的 fusion 规模模型（size model）。

旧的 ``adaptive_fusion_block_size`` 用
``expression_complexity * max(1, estimated_output_bytes / GB)`` 与
``backend_compile_budget_bytes`` 做标量比较——两个维度的量直接比大小存在
维度不一致风险（R39 §7）。本模块把 block 选择换成**可测量变量**驱动的
校准模型：

可测量变量
    sql_chars / ast_node_count / window_expression_count / projection_count /
    estimated_intermediate_bytes / output_bytes

校准目标
    min(compile_ms + execute_ms + output_materialization_ms)

block 仍取 32/64/128/256 bucket，但由模型（``choose_fusion_block_size``）选
最小化总耗时的 bucket；等耗时取更小 block（保守，避免过度融合）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

#: block bucket（R27-222 保留，取值不变）。
FUSION_BLOCKS: tuple[int, ...] = (32, 64, 128, 256)

#: 默认校准权重（单位：毫秒）。``estimate_fusion_cost`` 可用 ``weights`` 覆盖。
#:
#: 校准依据：
#:   - compile 对 AST 规模**超线性**（解析 + optimizer 多趟遍历）→ ast² 项；
#:   - compile 与 SQL 长度、window 表达式数、中间字节正相关；
#:   - execute 与 AST 节点数、window 表达式数、中间字节线性相关；
#:   - output materialization 与输出字节线性相关。
#:
#: 全部为静态标定值（诚实标注：基于经验量级，非端到端回归拟合）。
MODEL_WEIGHTS: dict[str, float] = {
    "compile_base_ms": 5.0,  # 每次 fusion group 解析/规划固定开销
    "compile_per_node_ms": 0.02,  # 每个 AST 节点线性
    "compile_quad_per_node_ms": 0.0008,  # ast² 超线性项
    "compile_per_window_ms": 0.5,  # 每个 window expression
    "compile_per_sql_char_ms": 0.0005,  # SQL 字符
    "compile_per_intermediate_gb_ms": 30.0,  # 每 GB 中间数据
    "execute_base_ms": 2.0,
    "execute_per_node_ms": 0.01,
    "execute_per_window_ms": 0.2,
    "execute_per_intermediate_gb_ms": 60.0,
    "output_base_ms": 1.0,
    "output_per_gb_ms": 80.0,  # 每 GB 输出
}


@dataclass(frozen=True)
class FusionSizeFeatures:
    """一个 fusion 单元的可测量规模变量（R39-P0-PERF-021）。

    字段全部来自真实计划（``extract_fusion_size_features`` 遍历 plan 统计），
    不是单个泛化标量。
    """

    sql_chars: int = 0
    ast_node_count: int = 0
    window_expression_count: int = 0
    projection_count: int = 0
    estimated_intermediate_bytes: int = 0
    output_bytes: int = 0
    #: 可选先验：plan cost 层已有的 compile / optimizer 模型估算。
    compile_ms_model: float | None = None
    optimizer_ms_model: float | None = None

    def scaled(self, factor: float) -> "FusionSizeFeatures":
        """线性缩放到 per-group 规模（字节/节点数按比例）。

        block 分组时，每个 group 的规模 ≈ 全组规模 × (block / root_count)。
        """
        f = max(0.0, float(factor))
        return FusionSizeFeatures(
            sql_chars=int(self.sql_chars * f),
            ast_node_count=int(self.ast_node_count * f),
            window_expression_count=int(self.window_expression_count * f),
            projection_count=int(self.projection_count * f),
            estimated_intermediate_bytes=int(self.estimated_intermediate_bytes * f),
            output_bytes=int(self.output_bytes * f),
            compile_ms_model=self.compile_ms_model,
            optimizer_ms_model=self.optimizer_ms_model,
        )


@dataclass(frozen=True)
class FusionCostModel:
    """R39-P0-PERF-021：融合总耗时 = 三个分量的和。"""

    compile_ms: float
    execute_ms: float
    output_materialization_ms: float
    optimizer_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.compile_ms + self.execute_ms + self.output_materialization_ms


def estimate_fusion_cost(
    features: FusionSizeFeatures,
    *,
    weights: dict[str, float] | None = None,
) -> FusionCostModel:
    """把可测量规模变量校准成 compile/execute/output 三个时间分量。

    ``weights`` 可部分覆盖 ``MODEL_WEIGHTS``（测试 / 标定用）。
    """
    w = {**MODEL_WEIGHTS, **(weights or {})}
    ast = max(0, features.ast_node_count)
    windows = max(0, features.window_expression_count)
    sql = max(0, features.sql_chars)
    intermediate_gb = max(0.0, features.estimated_intermediate_bytes) / (1024**3)
    out_gb = max(0.0, features.output_bytes) / (1024**3)

    compile_ms = (
        w["compile_base_ms"]
        + w["compile_per_node_ms"] * ast
        + w["compile_quad_per_node_ms"] * ast * ast
        + w["compile_per_window_ms"] * windows
        + w["compile_per_sql_char_ms"] * sql
        + w["compile_per_intermediate_gb_ms"] * intermediate_gb
    )
    if features.compile_ms_model is not None:
        compile_ms += float(features.compile_ms_model)
    optimizer_ms = float(features.optimizer_ms_model or 0.0)

    execute_ms = (
        w["execute_base_ms"]
        + w["execute_per_node_ms"] * ast
        + w["execute_per_window_ms"] * windows
        + w["execute_per_intermediate_gb_ms"] * intermediate_gb
    )
    output_materialization_ms = (
        w["output_base_ms"] + w["output_per_gb_ms"] * out_gb
    )
    return FusionCostModel(
        compile_ms=compile_ms,
        execute_ms=execute_ms,
        output_materialization_ms=output_materialization_ms,
        optimizer_ms=optimizer_ms,
    )


def choose_fusion_block_size(
    *,
    root_count: int,
    features: FusionSizeFeatures,
    weights: dict[str, float] | None = None,
) -> int:
    """在 32/64/128/256 bucket 中选使 ``compile+execute+output`` 最小的 block。

    模型：
        groups = ceil(root_count / block)
        per_group = features.scaled(block / root_count)
        cost(block) = groups * estimate_fusion_cost(per_group).total_ms

    由于 compile 对 AST 规模超线性 + 每 group 固定开销，AST 越多 → 最优
    block 越小（复杂度上升 block 单调不增）；等耗时取更小 block（保守）。
    """
    if root_count <= 0:
        return FUSION_BLOCKS[0]
    best_block: int = FUSION_BLOCKS[0]
    best_cost: float | None = None
    for block in FUSION_BLOCKS:
        groups = max(1, math.ceil(root_count / block))
        per_group = features.scaled(block / max(1, root_count))
        cost = groups * estimate_fusion_cost(per_group, weights=weights).total_ms
        if best_cost is None or cost < best_cost - 1e-9:
            best_cost = cost
            best_block = block
    return best_block
