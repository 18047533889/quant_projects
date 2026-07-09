"""GTJA-191 最快执行路径：HybridBackend(auto) + read_auto。"""
from __future__ import annotations

from typing import Any

from lib.data_source import default_ashare_pv_data_source


def fastest_data_source(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """data_access 最快读路径：read_auto（Arrow 零拷贝 collect）。"""
    return default_ashare_pv_data_source(
        start_date=start_date,
        end_date=end_date,
        read_auto=True,
    )


def enable_fastest_read(source: Any) -> Any:
    """兼容旧调用；read_auto 已在 factory 配置中启用。"""
    return source


def build_fastest_engine(data_source: Any):
    """构建 factor_engine 最快执行栈（backend=auto, operator_backend=auto）。

    auto 路径：SQL 可下推子树走 DuckDB，其余走 Polars；SQL 编译/执行失败时
    自动降级 Polars/Pandas（见 factor_engine sql_pushdown/executor）。
    """
    from backend.factory import build_backend
    from runtime.engine import FactorEngine
    from runtime.perf_config import PerfConfig

    backend = build_backend("auto")
    perf = PerfConfig(
        operator_backend="auto",
        enable_cse=True,
        panel_native=True,
    )
    engine = FactorEngine(backend=backend, data_source=data_source)

    class _FastEngine:
        """薄包装：run 时注入 PerfConfig。"""

        __slots__ = ("_engine", "_perf")

        def __init__(self, engine, perf) -> None:
            self._engine = engine
            self._perf = perf

        def run(self, factor, **kwargs):
            plan, analysis = self._engine.compile(factor)
            ctx = self._engine._make_context(perf=self._perf)
            result = self._engine.backend.execute(plan, ctx)
            return {
                "factor": factor,
                "analysis": analysis,
                "plan": plan,
                "result": result,
            }

        @property
        def backend(self):
            return self._engine.backend

        @property
        def data_source(self):
            return self._engine.data_source

    return _FastEngine(engine, perf)
