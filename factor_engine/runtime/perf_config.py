"""执行期性能与资源参数：从环境变量读取，适配不同机器。

``PerfConfig`` 集中管理并行度、内存护栏、CSE、panel-native 路径、
Numba rolling、回测内核、算子后端选择及 SQL 读路径预算等可调参数。
所有字段均可通过 ``PerfConfig.from_env()`` 从环境变量加载。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal


OperatorBackend = Literal["auto", "polars", "pandas_numpy"]


_ENV_CACHE: PerfConfig | None = None
_ENV_CACHE_KEY: tuple[str | None, ...] | None = None


BacktestExecutionEngine = Literal["python", "numpy", "numba", "auto"]


def _env_int(name: str, default: int | None) -> int | None:
    """读取正整数环境变量；空值、非正数或非法字符串返回 ``default``。"""
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
    """读取正浮点环境变量；空值、非正数或非法字符串返回 ``default``。"""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
        return v if v > 0 else default
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    """读取字符串环境变量；空值返回 ``default``。"""
    raw = os.environ.get(name, "").strip()
    return raw if raw else default


def _env_backtest_engine(name: str, default: BacktestExecutionEngine) -> BacktestExecutionEngine:
    """读取回测执行内核环境变量；非法值静默回退 ``default``。"""
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
    #: 算子 runtime 选择：auto=有 polars 则用 polars，否则 pandas
    operator_backend: OperatorBackend = "auto"
    #: SQL/读路径行数硬上限；``None`` 表示不额外限制（仍受 data_access 默认 budget 约束）
    query_max_rows: int | None = None
    #: SQL/读路径结果字节硬上限
    query_max_result_bytes: int | None = None
    #: Phase 5 资源预算：有效内存上限（字节）；``None`` 表示自动探测
    memory_limit_bytes: int | None = None
    #: Phase 5 进程预算占有效内存比例（默认 0.75）
    process_fraction: float | None = None
    #: Phase 5 进程外保留内存（GB）；``None`` 表示按有效内存 20% 自动
    reserve_gb: float | None = None
    #: Phase 5 结果字节预算（``results.max_in_memory_bytes``）；``None`` 表示按比例
    result_budget_bytes: int | None = None
    #: Phase 5 spill 目录；``None`` 表示系统 temp
    spill_dir: str | None = None
    #: Phase 5 spill 上限；``None`` 表示自动
    spill_budget_bytes: int | None = None

    def build_resource_plan(self, config: dict | None = None) -> Any:
        """构造 Phase 5 ``ExecutionResourcePlan``（resources 配置 + 环境变量合并）。"""
        from runtime.resource_governor import (
            ExecutionResourcePlan,
            _cfg_bytes,
            effective_memory_limit_bytes,
        )

        cfg: dict = dict(config or {})
        mem_cfg = dict(cfg.get("memory") or {})
        cache_cfg = dict(cfg.get("cache") or {})
        results_cfg = dict(cfg.get("results") or {})
        spill_cfg = dict(cfg.get("spill") or {})
        if self.memory_limit_bytes is not None:
            mem_cfg["limit"] = self.memory_limit_bytes
        if self.process_fraction is not None:
            mem_cfg["process_fraction"] = self.process_fraction
        if self.reserve_gb is not None:
            mem_cfg["reserve_gb"] = self.reserve_gb
        if self.result_budget_bytes is not None:
            results_cfg["max_in_memory_bytes"] = self.result_budget_bytes
        if self.spill_dir is not None:
            spill_cfg["directory"] = self.spill_dir
        if self.spill_budget_bytes is not None:
            spill_cfg["max_size"] = self.spill_budget_bytes
        # 审计 #340：显式透传 cache 配置（含 ``resources.cache.duckdb_fraction``），
        # 否则 ``ExecutionResourcePlan.from_dict`` 读不到用户配置。
        cfg["memory"], cfg["cache"], cfg["results"], cfg["spill"] = (
            mem_cfg,
            cache_cfg,
            results_cfg,
            spill_cfg,
        )
        if not cfg.get("mode"):
            cfg["mode"] = "auto"
        return ExecutionResourcePlan.from_dict(cfg)

    def build_query_budget(self) -> Any | None:
        """构造 ``data_access.QueryBudget`` 供执行上下文使用。

        若 ``query_max_rows`` / ``query_max_result_bytes`` 任一非空则显式构造
        预算；production 模式下 ``resolve_query_budget`` 会强制显式列选择。

        Returns:
            ``QueryBudget`` 实例，或 ``data_access`` 不可导入时返回 ``None``。
        """
        try:
            from data_access.read.query_budget import QueryBudget, resolve_query_budget
        except ImportError:
            return None
        explicit: QueryBudget | None = None
        if self.query_max_rows is not None or self.query_max_result_bytes is not None:
            explicit = QueryBudget(
                max_rows=self.query_max_rows,
                max_result_bytes=self.query_max_result_bytes,
            )
        return resolve_query_budget(explicit)

    @classmethod
    def from_env(cls) -> "PerfConfig":
        """从环境变量构造 ``PerfConfig`` 实例。

        环境变量在进程内通常不会在一次执行期间变化，因此通过小型缓存
        避免每个算子/节点重复解析；测试或动态配置修改可调用
        :meth:`invalidate_env_cache`。

        支持的环境变量：

        - ``FACTOR_ENGINE_MAX_WORKERS``：正整数，多进程/线程 worker 数
        - ``FACTOR_ENGINE_INSTRUMENT_CHUNK``：正整数，按标的分块大小
        - ``FACTOR_ENGINE_MAX_MEMORY_MB``：正浮点，软内存上限（MB）
        - ``FACTOR_ENGINE_DISABLE_CSE``：``1``/``true`` 关闭多因子 CSE
        - ``FACTOR_ENGINE_DISABLE_PANEL_NATIVE``：``1``/``true`` 关闭 panel-native 宽表路径
        - ``FACTOR_ENGINE_USE_NUMBA``：``1``/``true`` 尝试 Numba rolling
        - ``FACTOR_BACKTEST_EXECUTION_ENGINE``：``python``/``numpy``/``numba``/``auto``
        - ``FACTOR_ENGINE_USE_MODIN``：``1``/``true`` 使 PandasBackend 使用 modin
        - ``FACTOR_ENGINE_POLARS_LAZY``：``1``/``true`` 使 PolarsBackend 使用 LazyFrame
        - ``FACTOR_ENGINE_OPERATOR_BACKEND``：``auto`` / ``polars`` / ``pandas_numpy``
        - ``FACTOR_ENGINE_QUERY_MAX_ROWS``：SQL 下推行数硬上限
        - ``FACTOR_ENGINE_QUERY_MAX_RESULT_BYTES``：SQL 结果字节硬上限
        """
        global _ENV_CACHE, _ENV_CACHE_KEY
        cache_names = (
            "FACTOR_ENGINE_MAX_WORKERS",
            "FACTOR_ENGINE_INSTRUMENT_CHUNK",
            "FACTOR_ENGINE_MAX_MEMORY_MB",
            "FACTOR_ENGINE_DISABLE_CSE",
            "FACTOR_ENGINE_DISABLE_PANEL_NATIVE",
            "FACTOR_ENGINE_USE_NUMBA",
            "FACTOR_BACKTEST_EXECUTION_ENGINE",
            "FACTOR_ENGINE_OPERATOR_BACKEND",
            "FACTOR_ENGINE_QUERY_MAX_ROWS",
            "FACTOR_ENGINE_QUERY_MAX_RESULT_BYTES",
            "FACTOR_ENGINE_MAX_MEMORY_BYTES",
            "FACTOR_ENGINE_PROCESS_FRACTION",
            "FACTOR_ENGINE_RESERVE_GB",
            "FACTOR_ENGINE_RESULT_BUDGET_BYTES",
            "FACTOR_ENGINE_SPILL_DIR",
            "FACTOR_ENGINE_SPILL_BUDGET_BYTES",
        )
        cache_key = tuple(os.environ.get(name) for name in cache_names)
        if _ENV_CACHE is not None and _ENV_CACHE_KEY == cache_key:
            return _ENV_CACHE
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
        op_backend = _env_str("FACTOR_ENGINE_OPERATOR_BACKEND", "auto").lower()
        if op_backend not in {"auto", "polars", "pandas_numpy"}:
            op_backend = "auto"
        _ENV_CACHE = cls(
            max_workers=_env_int("FACTOR_ENGINE_MAX_WORKERS", None),
            instrument_chunk_size=_env_int("FACTOR_ENGINE_INSTRUMENT_CHUNK", None),
            max_in_memory_mb=_env_float("FACTOR_ENGINE_MAX_MEMORY_MB", None),
            enable_cse=not disable_cse,
            panel_native=not disable_panel,
            use_numba_rolling=use_numba,
            backtest_execution_engine=_env_backtest_engine("FACTOR_BACKTEST_EXECUTION_ENGINE", "python"),
            operator_backend=op_backend,  # type: ignore[arg-type]
            query_max_rows=_env_int("FACTOR_ENGINE_QUERY_MAX_ROWS", None),
            query_max_result_bytes=_env_int("FACTOR_ENGINE_QUERY_MAX_RESULT_BYTES", None),
            memory_limit_bytes=_env_int("FACTOR_ENGINE_MAX_MEMORY_BYTES", None),
            process_fraction=_env_float("FACTOR_ENGINE_PROCESS_FRACTION", None),
            reserve_gb=_env_float("FACTOR_ENGINE_RESERVE_GB", None),
            result_budget_bytes=_env_int("FACTOR_ENGINE_RESULT_BUDGET_BYTES", None),
            spill_dir=_env_str("FACTOR_ENGINE_SPILL_DIR", "").strip() or None,
            spill_budget_bytes=_env_int("FACTOR_ENGINE_SPILL_BUDGET_BYTES", None),
        )
        _ENV_CACHE_KEY = cache_key
        return _ENV_CACHE

    @classmethod
    def invalidate_env_cache(cls) -> None:
        """清除环境配置缓存，供测试和动态配置重载使用。"""
        global _ENV_CACHE, _ENV_CACHE_KEY
        _ENV_CACHE = None
        _ENV_CACHE_KEY = None
