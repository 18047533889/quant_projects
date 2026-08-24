"""q/K Region Executor.

执行编译后的 q Region 计划（文档 §25, §67, §74）。

Hard Gates (文档 §85):
- Q_BACKEND_OPERATOR_LEVEL_PINGPONG_ZERO: 禁止算子级 ping-pong
- BACKEND_REGION_PLANNER_IS_EXECUTION_AUTHORITY: Planner 是唯一决策权威

遵循文档 §11: Production 主路径禁止 Region 内算子自选 backend。
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import pandas as pd

from factor_engine.backend.q_backend.q_adapter import (
    QResidentTableHandle,
    QTypeAdapter,
    get_q_type_adapter,
)
from factor_engine.backend.q_backend.q_compiler import QRegionPlan
from factor_engine.backend.q_backend.q_errors import (
    QDataUnavailableError,
    QExecutionError,
    QProcessUnavailableError,
)
from factor_engine.backend.q_backend.q_process_manager import (
    QAvailabilityStatus,
    QProcessManager,
    get_q_process_manager,
)

logger = logging.getLogger(__name__)

# QProcessManager returns the process-global PyKX connection. This lock is
# shared by every executor instance, not just one executor's callers.
_PROCESS_CONNECTION_LOCK = threading.RLock()
_LEASE_LOCK = threading.RLock()
_ACTIVE_LEASES: dict[str, tuple[str, int, set[str]]] = {}
_SHARED_WORKSPACE_LEASES: set[tuple[str, str]] = set()


def _safe_q_identifier(value: str) -> str:
    """Encode a logical name as a valid, injective q identifier."""
    encoded = "".join(
        character if ("A" <= character <= "Z" or "a" <= character <= "z" or "0" <= character <= "9")
        else f"_{byte:02x}"
        for character in value
        for byte in (ord(character),)
    )
    if not encoded or encoded[0].isdigit():
        encoded = f"v_{encoded}"
    return encoded


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
        # PyKX exposes one process-global q connection. A complete region
        # execution is the smallest safe critical section for that workspace.
        self._connection_lock = _PROCESS_CONNECTION_LOCK
        self._telemetry_lock = threading.Lock()
        self._pending_workspace_cleanup: dict[
            tuple[str, str, int], frozenset[str]
        ] = {}

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
        workspace_id: str | None = None,
        generation_id: str | None = None,
        allow_legacy_handles: bool = False,
        _defer_workspace_cleanup: bool = False,
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
        workspace_id = workspace_id or uuid.uuid4().hex
        generation_id = generation_id or uuid.uuid4().hex
        workspace_prefix = f"qe_{_safe_q_identifier(workspace_id)}_"
        # Batch regions intentionally share one namespace so resident handles can
        # flow into later regions. Direct calls sharing that workspace need an
        # execution identity: logical names alone would alias their q symbols.
        if not _defer_workspace_cleanup:
            workspace_prefix += f"{uuid.uuid4().hex}_"
        if not materialize_output and not return_resident_handle:
            raise ValueError(
                "execute_region must materialize output or return a resident handle"
            )

        # 检查 q 可用性
        try:
            ready, message = self.check_execution_readiness()
        except Exception as exc:
            raise QProcessUnavailableError(
                f"q readiness check failed for region {plan.region_id}: {exc}"
            ) from exc
        if not ready:
            raise QProcessUnavailableError(
                f"q execution failed: {message}. "
                "Runtime fallback is disabled; select another backend during planning."
            )

        missing_tables = [
            table_name for table_name in plan.input_tables if table_name not in input_data
        ]
        if missing_tables:
            raise QDataUnavailableError(
                f"Missing input tables for region {plan.region_id}: {missing_tables}"
            )

        q: Any | None = None
        resident_handle: QResidentTableHandle | None = None
        workspace_symbols_before: frozenset[str] = frozenset()
        with _LEASE_LOCK:
            shared_workspace = (
                workspace_id,
                generation_id,
            ) in _SHARED_WORKSPACE_LEASES
        try:
            with self._connection_lock:
                start_time = time.perf_counter()

                # The lock spans upload, execution, and output retrieval so
                # another execution cannot overwrite this connection's names.
                q = self.process_manager.get_connection()
                with _LEASE_LOCK:
                    existing_lease = _ACTIVE_LEASES.get(workspace_id)
                    workspace_symbols_before = frozenset(
                        existing_lease[2]
                        if shared_workspace
                        and existing_lease is not None
                        and existing_lease[:2] == (generation_id, id(q))
                        else ()
                    )
                bindings = self._load_inputs_to_q(
                    q,
                    input_data,
                    workspace_id=workspace_id,
                    generation_id=generation_id,
                    workspace_prefix=workspace_prefix,
                    allow_legacy_handles=allow_legacy_handles,
                )

                # Compiler node identifiers are logical names.  Bind every
                # emitted intermediate into this execution's namespace, not
                # only region inputs and final output, so all q symbols are
                # lease-tracked and removed with the workspace.
                compiler_symbols = {
                    node_id: f"{workspace_prefix}{_safe_q_identifier(node_id)}"
                    for node_id in plan.node_ids
                }
                output_symbol = f"{workspace_prefix}{_safe_q_identifier(plan.output_table)}"
                bindings = {
                    **bindings,
                    **compiler_symbols,
                    plan.output_table: output_symbol,
                }
                rewritten_code = self._rewrite_q_code(plan.q_code, bindings)
                with _LEASE_LOCK:
                    lease = _ACTIVE_LEASES.setdefault(
                        workspace_id, (generation_id, id(q), set())
                    )
                    if lease[:2] != (generation_id, id(q)):
                        raise QExecutionError("Execution workspace lease identity conflict")
                    lease[2].update((*compiler_symbols.values(), output_symbol))
                logger.debug(f"Executing q Region {plan.region_id}:\n{rewritten_code}")
                q(rewritten_code)
                q_result = q(output_symbol)
                should_materialize = materialize_output and not return_resident_handle
                if should_materialize:
                    output_df = self.type_adapter.q_to_pandas(q_result)
                    output_bytes = int(output_df.memory_usage(deep=True).sum())
                    rows = len(output_df)
                    self._increment_telemetry("q_to_python_bytes", output_bytes)
                else:
                    output_df = pd.DataFrame()
                    rows, output_bytes = self._resident_result_metadata(q_result)

                execution_time = (time.perf_counter() - start_time) * 1000

                if return_resident_handle:
                    with _LEASE_LOCK:
                        lease = _ACTIVE_LEASES.get(workspace_id)
                        handle_owned_symbols = frozenset(
                            lease[2].difference(workspace_symbols_before)
                            if shared_workspace and lease is not None
                            else ()
                        )
                    resident_handle = QResidentTableHandle(
                        table_name=plan.output_table,
                        q_table_ref=q_result,
                        row_count=rows,
                        byte_size=int(output_bytes),
                        region_id=plan.region_id,
                        connection_id=id(q),
                        workspace_id=workspace_id,
                        generation_id=generation_id,
                        q_symbol=output_symbol,
                        _release_callback=(
                            (
                                lambda handle: self._release_shared_symbols(
                                    handle, handle_owned_symbols
                                )
                            )
                            if shared_workspace
                            else self._release_resident_handle
                        ),
                    )
                    self._register_lease(resident_handle)

            logger.info(
                f"q Region {plan.region_id} executed: "
                f"{rows} rows, {execution_time:.2f}ms"
            )
            result = QExecutionResult(
                region_id=plan.region_id,
                output_df=output_df,
                execution_time_ms=execution_time,
                rows_processed=rows,
                success=True,
                resident_handle=resident_handle,
            )
            return result

        except (QDataUnavailableError, QProcessUnavailableError):
            raise
        except Exception as e:
            error_msg = f"q execution failed for region {plan.region_id}: {e}"
            logger.error(error_msg)

            raise QExecutionError(error_msg) from e
        finally:
            # Direct executions own their workspace.  Always release it,
            # including cancellation-like BaseExceptions, while preserving
            # any original failure from the execution body.
            if q is not None and resident_handle is None and not _defer_workspace_cleanup:
                with self._connection_lock:
                    if shared_workspace:
                        self._cleanup_workspace(
                            q,
                            workspace_id,
                            generation_id,
                            preserve_symbols=workspace_symbols_before,
                            expected_connection_id=id(q),
                        )
                    else:
                        self._cleanup_workspace(q, workspace_id, generation_id)

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

    @staticmethod
    def _rewrite_q_code(q_code: str, bindings: dict[str, str]) -> str:
        rewritten = q_code
        for logical_name in sorted(bindings, key=len, reverse=True):
            rewritten = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(logical_name)}(?![A-Za-z0-9_])",
                bindings[logical_name],
                rewritten,
            )
        return rewritten

    @staticmethod
    def _register_lease(handle: QResidentTableHandle) -> None:
        if handle.workspace_id is None or handle.generation_id is None:
            raise QExecutionError("Resident handle is missing workspace lease identity")
        with _LEASE_LOCK:
            generation, connection_id, symbols = _ACTIVE_LEASES.setdefault(
                handle.workspace_id,
                (handle.generation_id, int(handle.connection_id or 0), set()),
            )
            if generation != handle.generation_id or connection_id != int(handle.connection_id or 0):
                raise QExecutionError("Resident workspace lease identity conflict")
            symbols.add(handle.q_symbol or handle.table_name)

    def _release_batch_handle(self, handle: QResidentTableHandle) -> None:
        """Release one published batch symbol without invalidating siblings."""
        self._release_shared_symbols(handle, frozenset({handle.q_symbol or handle.table_name}))

    def _release_shared_symbols(
        self,
        handle: QResidentTableHandle,
        physical_symbols: frozenset[str],
    ) -> None:
        """Release symbols owned by one handle on a shared workspace."""
        if handle.workspace_id is None or handle.generation_id is None:
            return
        try:
            with self._connection_lock:
                q = self.process_manager.get_connection()
                if handle.connection_id != id(q):
                    raise QExecutionError("Resident workspace connection identity conflict")
                with _LEASE_LOCK:
                    lease = _ACTIVE_LEASES.get(handle.workspace_id)
                    if lease is None:
                        return
                    if lease[:2] != (handle.generation_id, handle.connection_id):
                        raise QExecutionError("Resident workspace lease identity conflict")
                    owned_symbols = physical_symbols.intersection(lease[2])
                failed: set[str] = set()
                for physical_symbol in owned_symbols:
                    try:
                        q(f"delete {physical_symbol} from `.")
                    except Exception:
                        failed.add(physical_symbol)
                succeeded = owned_symbols.difference(failed)
                with _LEASE_LOCK:
                    lease = _ACTIVE_LEASES.get(handle.workspace_id)
                    if lease is not None and lease[:2] == (
                        handle.generation_id,
                        handle.connection_id,
                    ):
                        lease[2].difference_update(succeeded)
                        if not lease[2]:
                            del _ACTIVE_LEASES[handle.workspace_id]
                            _SHARED_WORKSPACE_LEASES.discard(
                                (handle.workspace_id, handle.generation_id)
                            )
                if failed:
                    raise QExecutionError("Resident symbol cleanup failed")
        except Exception:
            object.__setattr__(handle, "_released", False)
            raise

    def _release_resident_handle(self, handle: QResidentTableHandle) -> None:
        """Release the handle's entire owned workspace, retrying on failure."""
        if handle.workspace_id is None or handle.generation_id is None:
            return
        try:
            with self._connection_lock:
                q = self.process_manager.get_connection()
                if handle.connection_id != id(q):
                    raise QExecutionError("Resident workspace connection identity conflict")
                if not self._cleanup_workspace(
                    q,
                    handle.workspace_id,
                    handle.generation_id,
                    expected_connection_id=handle.connection_id,
                ):
                    raise QExecutionError("Resident workspace cleanup failed")
        except Exception:
            # QResidentTableHandle marks itself released before invoking us.
            # Roll that state back so callers can retry an unsuccessful cleanup.
            object.__setattr__(handle, "_released", False)
            raise

    @staticmethod
    def _validate_lease(
        handle: QResidentTableHandle,
        q: Any,
        *,
        allow_legacy_handles: bool,
    ) -> None:
        if handle.connection_id is not None and handle.connection_id != id(q):
            raise QExecutionError(
                f"Stale Q-resident handle for {handle.table_name}: connection changed"
            )
        if handle.workspace_id is None or handle.generation_id is None:
            if not allow_legacy_handles:
                raise QExecutionError(
                    f"Unbound Q-resident handle rejected for {handle.table_name}"
                )
            return
        with _LEASE_LOCK:
            lease = _ACTIVE_LEASES.get(handle.workspace_id)
            expected = (handle.generation_id, id(q))
            physical_symbol = handle.q_symbol or handle.table_name
            if lease is None or lease[:2] != expected or physical_symbol not in lease[2]:
                raise QExecutionError(
                    f"Stale Q-resident handle for {handle.table_name}: lease is not active"
                )

    @staticmethod
    def _cleanup_workspace(
        q: Any,
        workspace_id: str,
        generation_id: str,
        *,
        preserve_symbols: frozenset[str] | set[str] = frozenset(),
        expected_connection_id: int | None = None,
    ) -> bool:
        """Delete unpreserved symbols, retaining failed deletions for retry."""
        connection_id = id(q) if expected_connection_id is None else expected_connection_id
        with _LEASE_LOCK:
            lease = _ACTIVE_LEASES.get(workspace_id)
            if lease is None:
                return True
            if lease[:2] != (generation_id, connection_id) or id(q) != connection_id:
                return False
            symbols = tuple(symbol for symbol in lease[2] if symbol not in preserve_symbols)
        failed: set[str] = set()
        for symbol in symbols:
            try:
                q(f"delete {symbol} from `.")
            except Exception:
                failed.add(symbol)
                logger.warning("Failed to clean q workspace symbol %s", symbol, exc_info=True)
        with _LEASE_LOCK:
            lease = _ACTIVE_LEASES.get(workspace_id)
            if lease is None:
                return not failed
            if lease[:2] != (generation_id, connection_id):
                return False
            lease[2].difference_update(set(symbols) - failed)
            if not lease[2]:
                del _ACTIVE_LEASES[workspace_id]
                _SHARED_WORKSPACE_LEASES.discard((workspace_id, generation_id))
        return not failed

    def _retry_pending_workspace_cleanup(self, q: Any) -> None:
        """Retry workspaces retained after connection or delete failures."""
        for (workspace_id, generation_id, connection_id), preserve_symbols in tuple(
            self._pending_workspace_cleanup.items()
        ):
            if self._cleanup_workspace(
                q,
                workspace_id,
                generation_id,
                preserve_symbols=preserve_symbols,
                expected_connection_id=connection_id,
            ):
                self._pending_workspace_cleanup.pop(
                    (workspace_id, generation_id, connection_id), None
                )

    def _increment_telemetry(self, key: str, amount: int = 1) -> None:
        with self._telemetry_lock:
            self._telemetry[key] += int(amount)

    def telemetry_snapshot(self) -> dict[str, int]:
        with self._telemetry_lock:
            return dict(self._telemetry)

    def reset_telemetry(self) -> None:
        with self._telemetry_lock:
            for key in self._telemetry:
                self._telemetry[key] = 0

    def _load_inputs_to_q(
        self,
        q: Any,
        input_data: dict[str, pd.DataFrame | QResidentTableHandle | Any],
        *,
        workspace_id: str,
        generation_id: str,
        workspace_prefix: str,
        allow_legacy_handles: bool,
    ) -> dict[str, str]:
        """Load inputs into an execution-owned q namespace.

        Inputs may be pandas DataFrames, ``QResidentTableHandle`` (region
        intermediate kept resident), or Arrow tables (``pyarrow.Table``).  The
        Arrow case is the R21 cross-backend boundary (``Q_TO_ARROW``): a prior
        producer region can hand the q region an Arrow table instead of forcing
        a pandas round-trip in between.
        """
        bindings: dict[str, str] = {}
        for table_name, data in input_data.items():
            symbol = f"{workspace_prefix}{_safe_q_identifier(table_name)}"
            bindings[table_name] = symbol
            if isinstance(data, QResidentTableHandle):
                self._validate_lease(data, q, allow_legacy_handles=allow_legacy_handles)
                if data.table_name != symbol:
                    q(f"{symbol}: {data.q_symbol or data.table_name}")
                self._increment_telemetry("resident_reuse_count")
            elif self._is_arrow_table(data):
                # Cross-backend q boundary (R21-TRANSFER-BOUNDARIES Q_TO_ARROW):
                # accept an Arrow table directly, not just pandas.  Zero-copy
                # remains an optimization, never a correctness requirement.
                arrow_df = data.to_pandas()
                q[symbol] = self.type_adapter.pandas_to_q(
                    arrow_df, preserve_index=True, zero_copy=True
                )
                self._increment_telemetry(
                    "python_to_q_bytes", int(arrow_df.memory_usage(deep=True).sum())
                )
            else:
                q[symbol] = self.type_adapter.pandas_to_q(
                    data, preserve_index=True, zero_copy=True
                )
                self._increment_telemetry(
                    "python_to_q_bytes", int(data.memory_usage(deep=True).sum())
                )
            with _LEASE_LOCK:
                lease = _ACTIVE_LEASES.setdefault(
                    workspace_id, (generation_id, id(q), set())
                )
                if lease[:2] != (generation_id, id(q)):
                    raise QExecutionError("Execution workspace lease identity conflict")
                lease[2].add(symbol)
        return bindings

    @staticmethod
    def _is_arrow_table(data: Any) -> bool:
        """True when ``data`` is an Arrow table (R21 Q_TO_ARROW boundary).

        Duck-typed rather than importing pyarrow at module import time so the
        q executor stays importable even when pyarrow is absent.
        """
        module = type(data).__module__ or ""
        type_name = type(data).__name__
        if module.startswith("pyarrow") and type_name == "Table":
            return True
        return False

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
        available_inputs: dict[str, pd.DataFrame | QResidentTableHandle] = dict(input_data)
        workspace_id = uuid.uuid4().hex
        generation_id = uuid.uuid4().hex
        published_handles: list[QResidentTableHandle] = []
        batch_succeeded = False
        with _LEASE_LOCK:
            _SHARED_WORKSPACE_LEASES.add((workspace_id, generation_id))

        try:
            for i, plan in enumerate(plans):
                is_last = (i == len(plans) - 1)
                region_inputs = {
                    name: available_inputs[name]
                    for name in plan.input_tables
                    if name in available_inputs
                }

                result = self.execute_region(
                    plan,
                    region_inputs,
                    fallback_policy=fallback_policy,
                    return_resident_handle=enable_residency and not is_last,
                    materialize_output=is_last or not enable_residency,
                    workspace_id=workspace_id,
                    generation_id=generation_id,
                    allow_legacy_handles=False,
                    _defer_workspace_cleanup=True,
                )
                results.append(result)

                if not is_last:
                    if enable_residency and result.resident_handle:
                        published_handles.append(result.resident_handle)
                        available_inputs[plan.output_table] = result.resident_handle
                        logger.info(
                            f"Retained resident handle from {plan.region_id} for downstream regions"
                        )
                    else:
                        available_inputs[plan.output_table] = result.output_df
            batch_succeeded = True
            return results
        finally:
            preserve_symbols = frozenset(
                {
                    handle.q_symbol or handle.table_name
                    for handle in published_handles
                }
                if batch_succeeded
                else set()
            )
            with _LEASE_LOCK:
                lease = _ACTIVE_LEASES.get(workspace_id)
                cleanup_connection_id = lease[1] if lease is not None else 0
            cleanup_key = (workspace_id, generation_id, cleanup_connection_id)
            with self._connection_lock:
                try:
                    q = self.process_manager.get_connection()
                except Exception:
                    # Keep lease ownership so a later batch/release can retry;
                    # never mask the execution exception already in flight.
                    self._pending_workspace_cleanup[cleanup_key] = preserve_symbols
                else:
                    self._retry_pending_workspace_cleanup(q)
                    if self._cleanup_workspace(
                        q,
                        workspace_id,
                        generation_id,
                        preserve_symbols=preserve_symbols,
                        expected_connection_id=cleanup_connection_id,
                    ):
                        self._pending_workspace_cleanup.pop(cleanup_key, None)
                    else:
                        self._pending_workspace_cleanup[cleanup_key] = preserve_symbols
            with _LEASE_LOCK:
                if workspace_id not in _ACTIVE_LEASES:
                    _SHARED_WORKSPACE_LEASES.discard((workspace_id, generation_id))


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
    return get_q_executor().telemetry_snapshot()


def reset_q_executor_telemetry():
    """重置全局 q 执行器的 telemetry 计数器。"""
    get_q_executor().reset_telemetry()
