"""将 ``PerfConfig`` 路由决策同步到算子运行时环境。"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runtime.perf_config import PerfConfig


@contextmanager
def routing_execution_scope(perf: "PerfConfig | None" = None):
    """在 execute 窗口内同步 ``FACTOR_ENGINE_USE_NUMBA`` 与 perf 配置。

    参数:
        perf: 可选性能配置；``use_numba_rolling`` 为真时临时启用 Numba。

    返回:
        上下文管理器，退出时恢复环境变量。
    """
    prev = os.environ.get("FACTOR_ENGINE_USE_NUMBA")
    if perf is not None and perf.use_numba_rolling:
        os.environ["FACTOR_ENGINE_USE_NUMBA"] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("FACTOR_ENGINE_USE_NUMBA", None)
        else:
            os.environ["FACTOR_ENGINE_USE_NUMBA"] = prev
