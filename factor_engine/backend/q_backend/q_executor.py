"""q/K Region Executor.

执行编译后的 q Region 计划（文档 §25, §67, §74）。

Hard Gates (文档 §85):
- Q_BACKEND_OPERATOR_LEVEL_PINGPONG_ZERO: 禁止算子级 ping-pong
- BACKEND_REGION_PLANNER_IS_EXECUTION_AUTHORITY: Planner 是唯一决策权威

遵循文档 §11: Production 主路径禁止 Region 内算子自选 backend。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from backend.q_backend.q_adapter import QTypeAdapter, get_q_type_adapter
from backend.q_backend.q_compiler import QRegionPlan
from backend.q_backend.q_process_manager import (
    QAvailabilityStatus,
    QProcessManager,
    get_q_process_manager,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QExecutionResult:
    """q 执行结果。"""
    region_id: str
    output_df: pd.DataFrame
    execution_time_ms: float
    rows_processed: int
    success: bool
    error_message: str | None = None


@dataclass(frozen=True)
class QExecutionFallbackPolicy:
    """q 不可用时的 fallback 策略（文档 §66, §74）。

    Production 要求：
    - 执行前就选择 certified alternative
    - 或者直接 fail
    - 禁止 runtime 偷偷换 Pandas
    """
    allow_fallback: bool = False  # Production 默认不允许 fallback
    fallback_backend: str | None = None  # certified alternative backend
    fail_on_unavailable: bool = True  # q 不可用时失败


class QExecutor:
    """q Region 执行器。

    遵循文档 §11: Planner 必须成为生产执行唯一真值。
    Production 禁止：RegionPlan 选 q → q 内某 op 再选其他 backend。
    """

    def __init__(
        self,
        process_manager: QProcessManager | None = None,
        type_adapter: QTypeAdapter | None = None,
    ):
        self.process_manager = process_manager or get_q_process_manager()
        self.type_adapter = type_adapter or get_q_type_adapter()

    def check_execution_readiness(self) -> tuple[bool, str]:
        """检查 q 是否可执行。

        返回:
            (ready, message)
        """
        info = self.process_manager.check_availability()
        if info.status == QAvailabilityStatus.AVAILABLE:
            return True, "q runtime available"
        else:
            return False, f"q unavailable: {info.status.value} - {info.error_message}"

    def execute_region(
        self,
        plan: QRegionPlan,
        input_data: dict[str, pd.DataFrame],
        *,
        fallback_policy: QExecutionFallbackPolicy | None = None,
    ) -> QExecutionResult:
        """执行 q Region 计划。

        参数:
            plan: 编译好的 q Region 计划
            input_data: 输入表 {table_name: DataFrame}
            fallback_policy: fallback 策略

        返回:
            QExecutionResult

        抛出:
            RuntimeError: q 不可用且不允许 fallback
        """
        fallback_policy = fallback_policy or QExecutionFallbackPolicy()

        # 检查 q 可用性
        ready, message = self.check_execution_readiness()
        if not ready:
            if fallback_policy.fail_on_unavailable:
                raise RuntimeError(
                    f"q execution failed: {message}. "
                    "Fallback disabled in production mode."
                )
            else:
                logger.warning(f"q unavailable: {message}")
                return QExecutionResult(
                    region_id=plan.region_id,
                    output_df=pd.DataFrame(),
                    execution_time_ms=0.0,
                    rows_processed=0,
                    success=False,
                    error_message=message,
                )

        # 验证输入
        for table_name in plan.input_tables:
            if table_name not in input_data:
                raise ValueError(f"Missing input table: {table_name}")

        try:
            start_time = time.perf_counter()

            # 获取 q 连接
            q = self.process_manager.get_connection()

            # 将输入数据加载到 q
            self._load_inputs_to_q(q, input_data)

            # 执行 q 代码
            logger.debug(f"Executing q Region {plan.region_id}:\n{plan.q_code}")
            q(plan.q_code)

            # 获取输出
            q_result = q(plan.output_table)
            output_df = self.type_adapter.q_to_pandas(q_result)

            execution_time = (time.perf_counter() - start_time) * 1000  # ms
            rows = len(output_df)

            logger.info(
                f"q Region {plan.region_id} executed: "
                f"{rows} rows, {execution_time:.2f}ms"
            )

            return QExecutionResult(
                region_id=plan.region_id,
                output_df=output_df,
                execution_time_ms=execution_time,
                rows_processed=rows,
                success=True,
            )

        except Exception as e:
            error_msg = f"q execution failed for region {plan.region_id}: {e}"
            logger.error(error_msg)

            if fallback_policy.fail_on_unavailable:
                raise RuntimeError(error_msg) from e

            return QExecutionResult(
                region_id=plan.region_id,
                output_df=pd.DataFrame(),
                execution_time_ms=0.0,
                rows_processed=0,
                success=False,
                error_message=error_msg,
            )

    def _load_inputs_to_q(
        self,
        q: Any,
        input_data: dict[str, pd.DataFrame],
    ):
        """将输入数据加载到 q workspace。

        参数:
            q: q 连接
            input_data: {table_name: DataFrame}
        """
        for table_name, df in input_data.items():
            # 转换为 q table
            q_table = self.type_adapter.pandas_to_q(
                df,
                preserve_index=True,
                zero_copy=True,  # 优化提示
            )

            # 赋值到 q workspace
            q[table_name] = q_table
            logger.debug(f"Loaded {table_name}: {len(df)} rows to q")

    def execute_batch_regions(
        self,
        plans: list[QRegionPlan],
        input_data: dict[str, pd.DataFrame],
        *,
        fallback_policy: QExecutionFallbackPolicy | None = None,
    ) -> list[QExecutionResult]:
        """批量执行多个 q Regions。

        参数:
            plans: Region 计划列表
            input_data: 共享输入数据
            fallback_policy: fallback 策略

        返回:
            执行结果列表
        """
        results = []
        for plan in plans:
            result = self.execute_region(
                plan,
                input_data,
                fallback_policy=fallback_policy,
            )
            results.append(result)

            # 失败时是否继续
            if not result.success and fallback_policy and fallback_policy.fail_on_unavailable:
                logger.error(f"Region {plan.region_id} failed, stopping batch")
                break

        return results


# Global singleton
_EXECUTOR: QExecutor | None = None


def get_q_executor() -> QExecutor:
    """获取全局 q 执行器。"""
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = QExecutor()
    return _EXECUTOR
