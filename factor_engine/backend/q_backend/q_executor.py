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

from backend.q_backend.q_adapter import (
    QResidentTableHandle,
    QTypeAdapter,
    get_q_type_adapter,
)
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
    resident_handle: QResidentTableHandle | None = None  # For residency reuse


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

        # Telemetry counters for residency tracking
        self._telemetry = {
            "python_to_q_bytes": 0,
            "q_to_python_bytes": 0,
            "resident_reuse_count": 0,
        }

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
        input_data: dict[str, pd.DataFrame | QResidentTableHandle],
        *,
        fallback_policy: QExecutionFallbackPolicy | None = None,
        return_resident_handle: bool = False,
        materialize_output: bool = True,
    ) -> QExecutionResult:
        """执行 q Region 计划。

        参数:
            plan: 编译好的 q Region 计划
            input_data: 输入表 {table_name: DataFrame or QResidentTableHandle}
            fallback_policy: fallback 策略
            return_resident_handle: 是否返回 Q-resident handle 供后续 region 复用

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

            # 将输入数据加载到 q (支持 resident handle 复用)
            self._load_inputs_to_q(q, input_data)

            # 执行 q 代码
            logger.debug(f"Executing q Region {plan.region_id}:\n{plan.q_code}")
            q(plan.q_code)

            # 获取输出。中间 region 只需保留 q-resident handle，不应无条件
            # materialize 成 pandas；这会产生无谓的 q→Python ping-pong。
            q_result = q(plan.output_table)
            should_materialize = materialize_output and not return_resident_handle
            if should_materialize:
                output_df = self.type_adapter.q_to_pandas(q_result)
                output_bytes = int(output_df.memory_usage(deep=True).sum())
                rows = len(output_df)
                self._telemetry["q_to_python_bytes"] += output_bytes
            else:
                output_df = pd.DataFrame()
                rows, output_bytes = self._resident_result_metadata(q_result)

            execution_time = (time.perf_counter() - start_time) * 1000  # ms

            logger.info(
                f"q Region {plan.region_id} executed: "
                f"{rows} rows, {execution_time:.2f}ms"
            )

            # 创建 resident handle (如果需要)
            resident_handle = None
            if return_resident_handle:
                resident_handle = QResidentTableHandle(
                    table_name=plan.output_table,
                    q_table_ref=q_result,
                    row_count=rows,
                    byte_size=int(output_bytes),
                    region_id=plan.region_id,
                    connection_id=id(q),
                )
                logger.debug(f"Created resident handle: {resident_handle}")

            return QExecutionResult(
                region_id=plan.region_id,
                output_df=output_df,
                execution_time_ms=execution_time,
                rows_processed=rows,
                success=True,
                resident_handle=resident_handle,
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

    @staticmethod
    def _resident_result_metadata(q_result: Any) -> tuple[int, int]:
        """Read bounded metadata without materializing a q result.

        PyKX table objects expose ``len`` and may expose ``nbytes``. Metadata
        is best-effort: inspection failures must not trigger a q-to-pandas
        round trip or turn a successful resident region into a failure.
        """
        try:
            rows = max(0, int(len(q_result)))
        except (TypeError, ValueError, AttributeError):
            rows = 0

        try:
            byte_size = max(0, int(getattr(q_result, "nbytes", 0)))
        except (TypeError, ValueError, AttributeError):
            byte_size = 0
        return rows, byte_size

    def _load_inputs_to_q(
        self,
        q: Any,
        input_data: dict[str, pd.DataFrame | QResidentTableHandle],
    ):
        """将输入数据加载到 q workspace。

        支持 QResidentTableHandle 复用以消除 region 间 ping-pong。

        参数:
            q: q 连接
            input_data: {table_name: DataFrame or QResidentTableHandle}
        """
        for table_name, data in input_data.items():
            # 检查是否是 Q-resident handle
            if isinstance(data, QResidentTableHandle):
                if data.connection_id is not None and data.connection_id != id(q):
                    raise RuntimeError(
                        f"Stale Q-resident handle for {data.table_name}: "
                        "owning q connection is no longer active"
                    )

                # 数据已在 Q 中，无需上传
                logger.debug(
                    f"Reusing Q-resident table {data.table_name} "
                    f"({data.row_count} rows, {data.byte_size} bytes) "
                    f"from region {data.region_id}"
                )
                # 如果名称不同，创建引用
                if data.table_name != table_name:
                    q(f"{table_name}: {data.table_name}")
                    logger.debug(f"Aliased {data.table_name} -> {table_name}")

                # 更新 telemetry
                self._telemetry["resident_reuse_count"] += 1
            else:
                # 标准 DataFrame，需要上传
                df = data
                # 转换为 q table
                q_table = self.type_adapter.pandas_to_q(
                    df,
                    preserve_index=True,
                    zero_copy=True,  # 优化提示
                )

                # 赋值到 q workspace
                q[table_name] = q_table

                # 记录上传字节数
                upload_bytes = df.memory_usage(deep=True).sum()
                self._telemetry["python_to_q_bytes"] += upload_bytes

                logger.debug(
                    f"Loaded {table_name}: {len(df)} rows "
                    f"({upload_bytes} bytes) to q"
                )

    def execute_batch_regions(
        self,
        plans: list[QRegionPlan],
        input_data: dict[str, pd.DataFrame],
        *,
        fallback_policy: QExecutionFallbackPolicy | None = None,
        enable_residency: bool = True,
    ) -> list[QExecutionResult]:
        """批量执行多个 q Regions，支持中间结果 Q-resident 复用。

        参数:
            plans: Region 计划列表
            input_data: 共享输入数据
            fallback_policy: fallback 策略
            enable_residency: 是否启用跨 region 数据复用

        返回:
            执行结果列表
        """
        results = []
        current_input = dict(input_data)  # 初始输入

        for i, plan in enumerate(plans):
            is_last = (i == len(plans) - 1)

            result = self.execute_region(
                plan,
                current_input,
                fallback_policy=fallback_policy,
                return_resident_handle=enable_residency and not is_last,
                materialize_output=is_last,
            )
            results.append(result)

            # 失败时是否继续
            if not result.success and fallback_policy and fallback_policy.fail_on_unavailable:
                logger.error(f"Region {plan.region_id} failed, stopping batch")
                break

            # 如果启用 residency 且不是最后一个 region，传递 resident handle
            if enable_residency and not is_last and result.resident_handle:
                # 下一个 region 将使用当前 region 的输出作为输入
                # 假设输出表名在下一个 region 的输入中被引用
                # 这里简化处理：将 resident handle 加入可用输入
                next_plan = plans[i + 1]
                if plan.output_table in next_plan.input_tables:
                    current_input = {plan.output_table: result.resident_handle}
                    logger.info(
                        f"Passing resident handle from {plan.region_id} "
                        f"to {next_plan.region_id} (eliminated re-upload)"
                    )
                else:
                    # 下一个 region 不需要当前输出，重置为初始输入
                    current_input = dict(input_data)

        return results


# Global singleton
_EXECUTOR: QExecutor | None = None


def get_q_executor() -> QExecutor:
    """获取全局 q 执行器。"""
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = QExecutor()
    return _EXECUTOR


def get_q_executor_telemetry() -> dict[str, int]:
    """获取全局 q 执行器的 telemetry 数据。"""
    executor = get_q_executor()
    return dict(executor._telemetry)


def reset_q_executor_telemetry():
    """重置全局 q 执行器的 telemetry 计数器。"""
    executor = get_q_executor()
    for key in executor._telemetry:
        executor._telemetry[key] = 0
