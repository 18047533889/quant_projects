"""执行期性能与资源参数：从环境变量读取，适配不同机器。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


BacktestExecutionEngine = Literal["python", "numpy", "numba", "auto"]


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
        # 非正数视为未配置，避免误设 0 导致下游除零或无限并行
        return v if v > 0 else default
    except ValueError:
        return default


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
        return v if v > 0 else default
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name, "").strip()
    return raw if raw else default


def _env_backtest_engine(name: str, default: BacktestExecutionEngine) -> BacktestExecutionEngine:
    raw = _env_str(name, default).lower()
    if raw in {"python", "numpy", "numba", "auto"}:
        return raw  # type: ignore[return-value]
    # 拼写错误时静默回退 default，避免启动失败；需要时可打日志
    return default


@dataclass(frozen=True)
class PerfConfig:
    """并行与内存护栏（均为可选；未设置则由调用方使用合理默认）。"""

    #: 多进程/线程 worker 数；``None`` 表示由 joblib 等使用默认
    max_workers: int | None = None
    #: 按 instrument 分块求值时每块标的数量；``None`` 表示不分块
    instrument_chunk_size: int | None = None
    #: 软内存上限（MB），供上层在分块或落盘策略中使用；未强制 GC
    max_in_memory_mb: float | None = None
    #: 是否启用多因子 CSE（也可用 ``compile_many(..., enable_cse=...)``）
    enable_cse: bool = True
    #: 中间结果保持宽表 panel，根节点再 stack（显著减少 reshape）
    panel_native: bool = True
    #: 是否在 pandas 时序 rolling 路径尝试 Numba（需安装 numba 且未禁用）
    use_numba_rolling: bool = False
    #: 回测多资产执行内核（python/numpy/numba/auto）
    backtest_execution_engine: BacktestExecutionEngine = "python"

    @classmethod
    def from_env(cls) -> "PerfConfig":
        """环境变量：

        - ``FACTOR_ENGINE_MAX_WORKERS``：正整数
        - ``FACTOR_ENGINE_INSTRUMENT_CHUNK``：正整数，按标的分块
        - ``FACTOR_ENGINE_MAX_MEMORY_MB``：正浮点，内存提示（MB）
        - ``FACTOR_ENGINE_DISABLE_CSE``：``1``/``true`` 关闭 CSE
        - ``FACTOR_ENGINE_DISABLE_PANEL_NATIVE``：``1``/``true`` 关闭 panel-native 宽表路径
        - ``FACTOR_ENGINE_USE_NUMBA``：``1``/``true`` 尝试 Numba rolling
        - ``FACTOR_BACKTEST_EXECUTION_ENGINE``：``python``/``numpy``/``numba``/``auto``，多资产回测执行内核选择
        - ``FACTOR_ENGINE_USE_MODIN``：``1``/``true`` 使 ``PandasBackend`` 使用 ``modin.pandas``（亦可用 ``build_backend(\"pandas_modin\")``）
        - ``FACTOR_ENGINE_POLARS_LAZY``：``1``/``true`` 使 ``PolarsBackend`` 使用 LazyFrame 再 ``collect``
        """
        disable_cse = os.environ.get("FACTOR_ENGINE_DISABLE_CSE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        use_numba = os.environ.get("FACTOR_ENGINE_USE_NUMBA", "").lower() in (
            "1",
            "true",
            "yes",
        )
        disable_panel = os.environ.get("FACTOR_ENGINE_DISABLE_PANEL_NATIVE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        return cls(
            max_workers=_env_int("FACTOR_ENGINE_MAX_WORKERS", None),
            instrument_chunk_size=_env_int("FACTOR_ENGINE_INSTRUMENT_CHUNK", None),
            max_in_memory_mb=_env_float("FACTOR_ENGINE_MAX_MEMORY_MB", None),
            enable_cse=not disable_cse,
            panel_native=not disable_panel,
            use_numba_rolling=use_numba,
            backtest_execution_engine=_env_backtest_engine("FACTOR_BACKTEST_EXECUTION_ENGINE", "python"),
        )
