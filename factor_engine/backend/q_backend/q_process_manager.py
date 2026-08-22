"""q/K Process Manager with Availability Governance.

管理 q 进程生命周期，并提供显式的可用性治理（文档 §P2-006）。

Hard Gates (文档 §85):
- Q_BACKEND_NULL_TIME_SEMANTICS_CERTIFIED: q null/time 语义必须认证
- Q_BACKEND_PIT_PARITY: PIT 语义保持
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class QAvailabilityStatus(Enum):
    """q 运行时可用性状态。"""
    AVAILABLE = "AVAILABLE"  # q 进程可用
    UNAVAILABLE = "UNAVAILABLE"  # q 进程不可用
    LICENSE_MISSING = "LICENSE_MISSING"  # 许可证缺失
    PROCESS_FAILED = "PROCESS_FAILED"  # 进程启动失败
    VERSION_MISMATCH = "VERSION_MISMATCH"  # 版本不匹配
    NOT_CHECKED = "NOT_CHECKED"  # 尚未检查


@dataclass(frozen=True)
class QProcessInfo:
    """q 进程信息。"""
    status: QAvailabilityStatus
    version: str | None = None
    process_id: int | None = None
    pykx_version: str | None = None
    error_message: str | None = None


class QProcessManager:
    """q 进程管理器。

    遵循文档 §26 要求：
    - q process/session/version/license availability 显式治理
    - PyKX zero-copy 只能做优化，不能成为 correctness 前提
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._process_info: QProcessInfo | None = None
        self._q_connection: Any | None = None
        self._checked = False

    def check_availability(self) -> QProcessInfo:
        """检查 q 运行时可用性。

        返回:
            QProcessInfo 包含详细状态
        """
        with self._lock:
            if self._checked and self._process_info is not None:
                return self._process_info

            try:
                # 尝试导入 PyKX
                import pykx as kx

                pykx_version = getattr(kx, "__version__", "unknown")

                # 检查许可证
                if not self._check_license(kx):
                    self._process_info = QProcessInfo(
                        status=QAvailabilityStatus.LICENSE_MISSING,
                        pykx_version=pykx_version,
                        error_message="q/KDB+ license not found or invalid",
                    )
                    self._checked = True
                    return self._process_info

                # 尝试初始化 q 实例
                try:
                    q = kx.q
                    version = str(q("string .z.K"))
                    process_id = os.getpid()

                    self._q_connection = q
                    self._process_info = QProcessInfo(
                        status=QAvailabilityStatus.AVAILABLE,
                        version=version,
                        process_id=process_id,
                        pykx_version=pykx_version,
                    )
                    logger.info(
                        f"q runtime available: version={version}, "
                        f"pykx={pykx_version}, pid={process_id}"
                    )

                except Exception as e:
                    self._process_info = QProcessInfo(
                        status=QAvailabilityStatus.PROCESS_FAILED,
                        pykx_version=pykx_version,
                        error_message=f"Failed to initialize q: {e}",
                    )
                    logger.warning(f"q process initialization failed: {e}")

            except ImportError as e:
                self._process_info = QProcessInfo(
                    status=QAvailabilityStatus.UNAVAILABLE,
                    error_message=f"PyKX not installed: {e}",
                )
                logger.info("PyKX not available - q backend disabled")

            self._checked = True
            return self._process_info

    def _check_license(self, kx: Any) -> bool:
        """检查 q 许可证。

        参数:
            kx: PyKX 模块

        返回:
            许可证是否有效
        """
        try:
            # PyKX licensed mode check
            if hasattr(kx, "licensed"):
                return bool(kx.licensed)

            # Fallback: try to execute a simple query
            q = kx.q
            result = q("1+1")
            return result == 2

        except Exception as e:
            logger.debug(f"License check failed: {e}")
            return False

    def get_connection(self) -> Any:
        """获取 q 连接。

        返回:
            q 连接对象

        抛出:
            RuntimeError: q 不可用时
        """
        info = self.check_availability()
        if info.status != QAvailabilityStatus.AVAILABLE:
            raise RuntimeError(
                f"q runtime unavailable: {info.status.value}. "
                f"Error: {info.error_message}"
            )
        return self._q_connection

    def is_available(self) -> bool:
        """q 是否可用。"""
        info = self.check_availability()
        return info.status == QAvailabilityStatus.AVAILABLE

    def reset(self):
        """重置管理器状态（用于测试）。"""
        with self._lock:
            self._process_info = None
            self._q_connection = None
            self._checked = False


# Global singleton
_PROCESS_MANAGER: QProcessManager | None = None


def get_q_process_manager() -> QProcessManager:
    """获取全局 q 进程管理器。"""
    global _PROCESS_MANAGER
    if _PROCESS_MANAGER is None:
        _PROCESS_MANAGER = QProcessManager()
    return _PROCESS_MANAGER


def check_q_availability() -> QProcessInfo:
    """便捷函数：检查 q 可用性。"""
    return get_q_process_manager().check_availability()


def is_q_available() -> bool:
    """便捷函数：q 是否可用。"""
    return get_q_process_manager().is_available()
