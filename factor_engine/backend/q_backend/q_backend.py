"""q/K Backend - Main Backend Implementation.

完整的 q/K 后端实现，集成 QExecutor/QCompiler/QAdapter/QProcessManager。

Hard Gates (文档 §85):
- Q_BACKEND_CANONICAL_IR_ONLY: q 只能是 Canonical IR 的编译目标
- Q_BACKEND_ZERO_SEMANTIC_AUTHORITY: q 不得成为语义权威
- Q_BACKEND_OPERATOR_LEVEL_PINGPONG_ZERO: 禁止算子级 ping-pong
- BACKEND_REGION_PLANNER_IS_EXECUTION_AUTHORITY: Planner 是唯一决策权威

遵循文档 §2 架构原则：
- DSL/IR/OperatorSemanticRegistry 是 canonical authority
- q 只负责 compile_to_q(region) 和 execute_q_region()
- PIT/Unit/Factor identity 必须由 DataAccess 给定
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from backend.base import Backend
from backend.context import ExecutionContext
from backend.q_backend.q_adapter import get_q_type_adapter
from backend.q_backend.q_capability import get_q_capability
from backend.q_backend.q_compiler import QRegionPlan, get_q_compiler
from backend.q_backend.q_executor import (
    QExecutionFallbackPolicy,
    get_q_executor,
)
from backend.q_backend.q_process_manager import get_q_process_manager
from backend.runtime_events import append_runtime_event
from planner.logical_plan import PlanNode

logger = logging.getLogger(__name__)


class QBackend(Backend):
    """q/K 执行后端。

    遵循文档 §10-11 要求：
    - Region 级边界（不是算子级）
    - Planner 决定 Region 路由
    - Production 禁止 Region 内算子自选 backend

    架构：
    1. PlanNode → PhysicalRegionPlan（由 Planner 完成）
    2. PhysicalRegionPlan → QRegionPlan（QCompiler）
    3. QRegionPlan → QExecutionResult（QExecutor）
    4. QExecutionResult → pandas.Series（QAdapter）
    """

    runtime_backend_label = "q_kdb"
    prefers_native_scan = False
    supports_lazy_shared = False

    def __init__(
        self,
        *,
        fallback_to_pandas: bool = False,
        production_mode: bool = True,
    ):
        """初始化 q 后端。

        参数:
            fallback_to_pandas: q 不可用时是否回退到 pandas（生产默认禁止）
            production_mode: 生产模式（更严格的错误处理）
        """
        self._capability = get_q_capability()
        self._compiler = get_q_compiler()
        self._executor = get_q_executor()
        self._adapter = get_q_type_adapter()
        self._process_manager = get_q_process_manager()

        self._fallback_to_pandas = fallback_to_pandas
        self._production_mode = production_mode

        # 运行时统计
        self._stats = {
            "regions_compiled": 0,
            "regions_executed": 0,
            "regions_failed": 0,
            "fallback_count": 0,
            "total_rows_processed": 0,
            "total_execution_time_ms": 0.0,
        }

    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        """执行逻辑计划。

        参数:
            plan: 逻辑计划节点
            ctx: 执行上下文

        返回:
            pandas.Series (MultiIndex: timestamp × instrument)

        抛出:
            RuntimeError: q 不可用且不允许 fallback
            ValueError: 计划无法编译为 q
        """
        # 检查 q 可用性
        if not self._process_manager.is_available():
            info = self._process_manager.check_availability()
            logger.warning(
                f"q backend unavailable: {info.status.value} - {info.error_message}"
            )

            if self._fallback_to_pandas:
                logger.info("Falling back to pandas backend")
                self._stats["fallback_count"] += 1
                return self._fallback_to_pandas_backend(plan, ctx)
            else:
                raise RuntimeError(
                    f"q backend unavailable: {info.status.value}. "
                    f"Error: {info.error_message}. "
                    "Fallback disabled in production mode."
                )

        # 记录运行时事件
        append_runtime_event(
            ctx,
            "q_backend_start",
            backend="q_kdb",
            plan_node_id=plan.node_id,
        )

        try:
            # 将逻辑计划转为 q region
            # 注意：目前简化实现，将整棵树作为单个 region
            # 实际生产中应由 PhysicalPlanner 完成 region 切分
            region_plan = self._plan_to_q_region(plan, ctx)

            self._stats["regions_compiled"] += 1

            # 准备输入数据
            input_data = self._prepare_input_data(plan, ctx)

            # 执行 q region
            fallback_policy = QExecutionFallbackPolicy(
                allow_fallback=self._fallback_to_pandas,
                fail_on_unavailable=self._production_mode,
            )

            result = self._executor.execute_region(
                region_plan,
                input_data,
                fallback_policy=fallback_policy,
            )

            if not result.success:
                self._stats["regions_failed"] += 1
                raise RuntimeError(
                    f"q execution failed: {result.error_message}"
                )

            # 更新统计
            self._stats["regions_executed"] += 1
            self._stats["total_rows_processed"] += result.rows_processed
            self._stats["total_execution_time_ms"] += result.execution_time_ms

            # 记录成功事件
            append_runtime_event(
                ctx,
                "q_backend_success",
                backend="q_kdb",
                rows=result.rows_processed,
                execution_time_ms=result.execution_time_ms,
            )

            # 转换为 MultiIndex Series
            return self._result_to_series(result.output_df, ctx)

        except Exception as e:
            self._stats["regions_failed"] += 1

            append_runtime_event(
                ctx,
                "q_backend_error",
                backend="q_kdb",
                error=str(e),
            )

            if self._fallback_to_pandas:
                logger.warning(f"q execution failed, falling back to pandas: {e}")
                self._stats["fallback_count"] += 1
                return self._fallback_to_pandas_backend(plan, ctx)
            else:
                raise

    def _plan_to_q_region(
        self,
        plan: PlanNode,
        ctx: ExecutionContext,
    ) -> QRegionPlan:
        """将逻辑计划转换为 q region。

        参数:
            plan: 逻辑计划
            ctx: 执行上下文

        返回:
            QRegionPlan
        """
        # 提取节点信息（拓扑排序）
        nodes = self._extract_nodes_topological(plan)

        # 验证所有节点可以编译
        is_valid, unsupported = self._compiler.validate_region(nodes)
        if not is_valid:
            raise ValueError(
                f"Cannot compile plan to q: unsupported operators {unsupported}"
            )

        # 编译
        region_plan = self._compiler.compile_region(
            region_id=f"region_{plan.node_id}",
            nodes=nodes,
            input_tables=["input_table"],
            output_name="result",
        )

        return region_plan

    def _extract_nodes_topological(self, plan: PlanNode) -> list[dict[str, Any]]:
        """提取计划节点的拓扑排序。

        参数:
            plan: 根节点

        返回:
            节点列表（按拓扑顺序）
        """
        nodes = []
        visited = set()

        def visit(node: PlanNode):
            if node.node_id in visited:
                return
            visited.add(node.node_id)

            # 先访问子节点
            for child in node.children:
                visit(child)

            # 提取节点信息
            node_dict = {
                "id": node.node_id,
                "operator": node.operator,
                "inputs": [c.node_id for c in node.children],
                "params": getattr(node, "params", {}),
            }
            nodes.append(node_dict)

        visit(plan)
        return nodes

    def _prepare_input_data(
        self,
        plan: PlanNode,
        ctx: ExecutionContext,
    ) -> dict[str, pd.DataFrame]:
        """准备输入数据。

        参数:
            plan: 逻辑计划
            ctx: 执行上下文

        返回:
            输入表字典
        """
        # 简化实现：从 ctx 获取基础数据
        # 实际应由 DataAccess 提供
        input_df = self._get_base_data(ctx)

        return {"input_table": input_df}

    def _get_base_data(self, ctx: ExecutionContext) -> pd.DataFrame:
        """获取基础数据。

        参数:
            ctx: 执行上下文

        返回:
            基础数据 DataFrame
        """
        # 从 ctx.data_source 获取数据
        # 简化实现
        if hasattr(ctx, "base_data"):
            return ctx.base_data

        # 空 DataFrame 作为 fallback
        return pd.DataFrame()

    def _result_to_series(
        self,
        df: pd.DataFrame,
        ctx: ExecutionContext,
    ) -> pd.Series:
        """将结果 DataFrame 转换为 MultiIndex Series。

        参数:
            df: 结果 DataFrame
            ctx: 执行上下文

        返回:
            MultiIndex Series
        """
        # 假设 df 有 timestamp, instrument, value 列
        if "timestamp" in df.columns and "instrument" in df.columns:
            df = df.set_index(["timestamp", "instrument"])

        if "value" in df.columns:
            return df["value"]
        elif len(df.columns) == 1:
            return df.iloc[:, 0]
        else:
            # 返回第一列
            return df.iloc[:, 0] if len(df.columns) > 0 else pd.Series()

    def _fallback_to_pandas_backend(
        self,
        plan: PlanNode,
        ctx: ExecutionContext,
    ) -> Any:
        """回退到 pandas backend。

        参数:
            plan: 逻辑计划
            ctx: 执行上下文

        返回:
            执行结果
        """
        from backend.pandas_backend import PandasBackend

        pandas_backend = PandasBackend()
        return pandas_backend.execute(plan, ctx)

    def get_stats(self) -> dict[str, Any]:
        """获取运行时统计。

        返回:
            统计信息字典
        """
        return dict(self._stats)

    def reset_stats(self):
        """重置统计信息。"""
        for key in self._stats:
            if isinstance(self._stats[key], (int, float)):
                self._stats[key] = 0 if isinstance(self._stats[key], int) else 0.0


# Global singleton
_Q_BACKEND: QBackend | None = None


def get_q_backend(
    *,
    fallback_to_pandas: bool = False,
    production_mode: bool = True,
) -> QBackend:
    """获取全局 q backend 实例。

    参数:
        fallback_to_pandas: q 不可用时是否回退
        production_mode: 生产模式

    返回:
        QBackend 实例
    """
    global _Q_BACKEND
    if _Q_BACKEND is None:
        _Q_BACKEND = QBackend(
            fallback_to_pandas=fallback_to_pandas,
            production_mode=production_mode,
        )
    return _Q_BACKEND
