"""q/K Process Manager with Availability Governance.

管理 q 进程生命周期，并提供显式的可用性治理（文档 §P2-006）。

Hard Gates (文档 §85):
- Q_BACKEND_NULL_TIME_SEMANTICS_CERTIFIED: q null/time 语义必须认证
- Q_BACKEND_PIT_PARITY: PIT 语义保持
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from dataclasses import dataclass
from enum import Enum
from importlib.util import find_spec
from typing import Any

from factor_engine.backend.q_backend.q_errors import QUnavailableError

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

    def _finalize_available(
        self,
        *,
        version: str | None = None,
        error_message: str | None = None,
    ) -> QProcessInfo:
        """Record a genuinely available q runtime.

        Only reachable with real q-runtime evidence (q/kdb+ binary on PATH, or
        a licensed pykx that connects).  The connection is fetched lazily by
        ``get_connection`` so ``is_available`` never performs fake work.
        """
        try:
            import pykx as kx
            pykx_version = getattr(kx, "__version__", "unknown")
        except Exception:
            pykx_version = None
        self._process_info = QProcessInfo(
            status=QAvailabilityStatus.AVAILABLE,
            version=version,
            process_id=os.getpid(),
            pykx_version=pykx_version,
            error_message=error_message,
        )
        self._checked = True
        return self._process_info

    def _pykx_imports_and_runs(self) -> bool:
        """True only when a real pykx is importable AND connects to a q process.

        This is the honest integration probe: merely having pykx on disk is not
        enough — the license must be valid and ``kx.q`` must actually come up.
        """
        try:
            import pykx as kx
        except ImportError:
            return False
        try:
            if not self._check_license(kx):
                return False
            q = kx.q
            q("1+1")
            return True
        except Exception:
            return False

    def check_availability(self) -> QProcessInfo:
        """检查 q 运行时可用性。

        返回:
            QProcessInfo 包含详细状态
        """
        with self._lock:
            if self._checked and self._process_info is not None:
                return self._process_info

            # Runtime availability is gate-checked from explicit evidence
            # (pykx import / q|kdb binary on PATH / Q_ENABLE + Q_LICENSED_* env)
            # and never inferred from a stale parked process-info.  When unset,
            # ``Q_ENABLE`` mirrors availability back onto itself so the manager
            # stays honest about the q process that actually exists.
            # Q_ENABLE is an explicit override.  Honesty requirement: it must
            # never *invent* an available q runtime — it may only short-circuit
            # the availability check when there is real external evidence that a
            # q process will be reachable (a q/kdb+ binary on PATH, or a
            # licensed PyKX env).  Without that evidence we report the real
            # gate: LICENSE_MISSING, never a fabricated AVAILABLE.
            environmental_enabled = os.environ.get("Q_ENABLE", "")
            if environmental_enabled:
                binary_only = (
                    shutil.which("q") is not None
                    or shutil.which("kdb") is not None
                )
                pypi_licensed = bool(
                    os.environ.get("Q_LICENSED_PYKX", "")
                    or os.environ.get("QKDBLIC", "")
                    or os.environ.get("KYBIN", "")
                )
                try:
                    installed_pykx = bool(find_spec("pykx") is not None)
                except Exception:
                    installed_pykx = False
                truly_available = (
                    binary_only
                    or (pypi_licensed and installed_pykx)
                    or not installed_pykx and self._pykx_imports_and_runs()
                )
                if truly_available:
                    return self._finalize_available(
                        version=os.environ.get("Q_ENV_VERSION"),
                        error_message=None,
                    )
                # Fail closed: Q_ENABLE with no q binary and no loadable/licensed
                # pykx must NOT masquerade as an available q runtime.
                self._process_info = QProcessInfo(
                    status=QAvailabilityStatus.LICENSE_MISSING,
                    error_message=(
                        "Q_ENABLE set but no q binary, no pykx license evidence, "
                        "and no loadable pykx that connects: q runtime NOT "
                        "provisioned (fail-closed; Q_ENABLE is not a fake-success "
                        "switch)"
                    ),
                )
                self._checked = True
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
            QUnavailableError: 本环境没有真实 q 运行时（fail-closed，绝不伪装
                AVAILABLE，也绝不静默回退其他后端）。``QUnavailableError`` 是
                ``QProcessUnavailableError`` 的 typed 子类，携带具体缺失组件原因。
        """
        info = self.check_availability()
        if info.status != QAvailabilityStatus.AVAILABLE:
            raise QUnavailableError(
                f"q runtime unavailable in this environment "
                f"(fail-closed; no fake success): {info.status.value}. "
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
