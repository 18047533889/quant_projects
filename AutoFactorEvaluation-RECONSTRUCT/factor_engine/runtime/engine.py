"""运行时入口：``FactorEngine`` 串联 compile 与 run。

编译链（单因子，expr 模式）::

    Factor.expr  →  Analyzer.lower  →  IRNode
                 →  Optimizer       →  IRNode（可选规则）
                 →  Lowerer         →  PlanNode
                 →  PandasBackend   →  MultiIndex Series

代码模式（calc_mode="code"）::

    Factor.expr(str)  →  compile_code_function  →  code object
                      →  execute_code_function  →  MultiIndex Series

``run_many`` / CSE：多因子共享子表达式时插入 ``plan_ref`` 节点与 ``shared_result_cache``。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from api.dsl_parser import parse_factor
from api.factor import Factor
from backend.cleaned_bridge import ensure_cleaned_loaded
from backend.context import ExecutionContext
from backend.factory import build_backend
from backend.operator_loader import maybe_load_operators_from_config
from ir.analyzer import AnalysisResult, Analyzer
from logging_utils import get_logger
from planner.cse import apply_cse
from planner.dag import DAGPlan, FactorPlan
from planner.logical_plan import PlanNode
from planner.lowerer import Lowerer
from planner.optimizer import Optimizer
from runtime.config import FactorEngineConfig, load_config
from runtime.perf_config import PerfConfig
from storage.cache import CacheManager
from storage.factory import build_data_source
from storage.materializer import ParquetMaterializer
from storage.persistent_cache import PersistentCache, PersistentCacheAdapter

logger = get_logger("runtime.engine")

# 小体量运行常量
_TINY_N_INSTRUMENTS = 5   # 只取前 5 个标的
_TINY_N_DAYS = 10         # 只取前 10 个交易日


class FactorEngine:
    """因子引擎：注入后端与数据源，对 :class:`api.factor.Factor` 做编译与执行。

    参数:
      - ``tiny_run``: 小体量模式（仅取少量数据验证可运行性）。
      - ``calc_mode``: ``"expr"`` (表达式) 或 ``"code"`` (函数代码)。
      - ``factor_engine_operators``: 外部算子库路径。
      - ``cache_root``: 持久化缓存根目录。提供后引擎使用跨进程磁盘缓存，
        即使单因子逐一评估也能复用之前缓存的列数据和中间结果。
    """

    def __init__(self, backend, data_source, cache=None, *,
                 tiny_run: bool = False,
                 factor_engine_operators: str | None = None,
                 cache_root: str | Path | None = None) -> None:
        self.backend = backend
        self.data_source = data_source
        self._cache_root = cache_root

        # 持久化缓存（进程共享 / 跨进程复用）
        self._persistent_cache: PersistentCache | None = None
        if cache_root is not None:
            self._persistent_cache = PersistentCache(cache_root)
            logger.info("持久化缓存已启用: cache_root=%s", cache_root)

        # 向后兼容：若传入了 cache 对象则使用；否则用持久化缓存适配
        if cache is not None:
            self.cache = cache
        elif self._persistent_cache is not None:
            self.cache = PersistentCacheAdapter(self._persistent_cache)
        else:
            self.cache = CacheManager()

        self.tiny_run = tiny_run
        self.factor_engine_operators = factor_engine_operators
        self.analyzer = Analyzer()
        self.lowerer = Lowerer()
        self.optimizer = Optimizer()

    def _maybe_truncate_source(self, ds: Any) -> Any:
        """tiny_run 模式：包装数据源，只返回少量数据。"""
        if not self.tiny_run:
            return ds
        return _TruncatedDataSource(ds, n_instruments=_TINY_N_INSTRUMENTS, n_days=_TINY_N_DAYS)

    # ── compile (仅 expr 模式) ──

    def compile(self, factor: Factor):
        """Expr → Analyzer → Lowerer → Optimizer，返回 (plan, analysis)。"""
        started_at = time.perf_counter()
        logger.info("开始编译因子 '%s'", factor.name)
        analysis = self.analyzer.lower(factor.expr)
        logical_plan = self.lowerer.to_logical_plan(analysis.ir)
        optimized_plan = self.optimizer.optimize(logical_plan)
        logger.info(
            "完成编译因子 '%s'，lookback=%s，耗时 %.2fs",
            factor.name,
            getattr(analysis, "lookback", None),
            time.perf_counter() - started_at,
        )
        return optimized_plan, analysis

    # ── run (统一入口，支持两种 calc_mode) ──

    def run(self, factor: Factor):
        """执行因子。根据 ``factor.calc_mode`` 自动选择执行链路。"""
        if factor.calc_mode == "code":
            return self._run_code(factor)
        return self._run_expr(factor)

    def _run_expr(self, factor: Factor):
        """表达式模式：编译 + 执行。"""
        started_at = time.perf_counter()
        logger.info("开始执行因子 '%s' (calc_mode=expr)", factor.name)
        plan, analysis = self.compile(factor)
        ds = self._maybe_truncate_source(self.data_source)
        ctx = ExecutionContext(data_source=ds, cache=self.cache,
                               perf=PerfConfig(tiny_run=self.tiny_run))
        result = self.backend.execute(plan, ctx)
        non_null_count = int(result.notna().sum()) if hasattr(result, "notna") else None

        # 日志：缓存命中率
        cache_info = ""
        if self._persistent_cache is not None:
            stats = self._persistent_cache.stats()
            cache_info = f", cache_hit_rate={stats['hit_rate']}"

        logger.info(
            "完成执行因子 '%s'，结果行数=%s，非空=%s，耗时 %.2fs%s%s",
            factor.name,
            len(result),
            non_null_count,
            time.perf_counter() - started_at,
            " (tiny_run)" if self.tiny_run else "",
            cache_info,
        )
        return {
            "factor": factor,
            "analysis": analysis,
            "plan": plan,
            "result": result,
        }

    def _run_code(self, factor: Factor):
        """代码模式：编译函数源码 → 加载所需列 → 执行函数。"""
        started_at = time.perf_counter()
        logger.info("开始执行因子 '%s' (calc_mode=code)", factor.name)

        from backend.code_runner import execute_code_function, compile_code_function

        # 先编译函数以获取参数名
        code_obj, func_name = compile_code_function(factor.expr)  # expr 存的是源码字符串

        # 从源码的 AST 中提取参数名，以确定需要加载哪些数据列
        import ast as _ast
        import inspect as _inspect

        tree = _ast.parse(factor.expr, mode="exec")
        param_names: list[str] = []
        for node in _ast.walk(tree):
            if isinstance(node, _ast.FunctionDef) and node.name == func_name:
                param_names = [arg.arg for arg in node.args.args]
                break

        # 加载所需数据列（tiny_run 模式下截断）
        ds = self._maybe_truncate_source(self.data_source)
        data_columns: dict[str, Any] = {}
        for pname in param_names:
            if pname in ("tiny_run",):
                continue
            try:
                data_columns[pname] = ds.load_column(pname)
            except Exception as e:
                logger.warning("加载数据列 '%s' 失败: %s", pname, e)

        result = execute_code_function(
            factor.expr,
            data_columns,
            tiny_run=self.tiny_run,
        )

        logger.info(
            "完成执行因子 '%s' (code)，耗时 %.2fs%s",
            factor.name,
            time.perf_counter() - started_at,
            " (tiny_run)" if self.tiny_run else "",
        )
        return {
            "factor": factor,
            "result": result,
            "analysis": None,
            "plan": None,
        }

    # ── 工厂方法 ──

    @classmethod
    def from_loaded_config(cls, config: FactorEngineConfig, *,
                           cache_root: str | Path | None = None):
        """从已解析配置构造 engine 与 factor。

        启动时序（算子库加载 → 白名单构建）:
          1. 加载内建算子库 (ensure_cleaned_loaded)
          2. 加载外部算子库 (maybe_load_operators_from_config)
          3. 此时 OperatorRegistry 包含所有算子
          4. 解析因子（parse_expr 使用当前 OperatorRegistry 构建白名单）

        Args:
            config: 已解析的引擎配置。
            cache_root: 持久化缓存根目录。提供后引擎使用跨进程磁盘缓存，
                即使单因子逐一评估也能复用之前缓存的列数据和中间结果。
        """
        # 1. 先确保内建算子已加载
        ensure_cleaned_loaded()
        # 2. 再加载外部算子库（如有）
        maybe_load_operators_from_config(config)

        backend = build_backend(config.backend.type)
        data_source = build_data_source(config.data_source)

        # 持久化缓存（优先使用传入的 cache_root，其次配置中的）
        _cache_root = cache_root or (
            getattr(config.engine, "cache_root", None) if hasattr(config, "engine") else None
        )

        if config.factor.calc_mode == "code":
            from api.factor import Factor as _Factor
            from expr.literal import Literal
            factor = _Factor(
                name=config.factor.name,
                expr=Literal(config.factor.expr),
                calc_mode="code",
                freq=config.factor.freq,
                universe=config.factor.universe,
                description=config.factor.description,
            )
        else:
            factor = parse_factor(
                config.factor.expr,
                name=config.factor.name,
                freq=config.factor.freq,
                universe=config.factor.universe,
                description=config.factor.description,
            )

        logger.info(
            "配置对象加载完成: factor=%s, backend=%s, data_source=%s, "
            "calc_mode=%s, tiny_run=%s, cache_root=%s",
            factor.name,
            type(backend).__name__,
            type(data_source).__name__,
            factor.calc_mode,
            config.engine.tiny_run,
            _cache_root,
        )
        return cls(
            backend=backend,
            data_source=data_source,
            tiny_run=config.engine.tiny_run,
            factor_engine_operators=config.engine.factor_engine_operators,
            cache_root=_cache_root,
        ), factor

    @classmethod
    def from_config(cls, config_path: str | Path, *,
                    cache_root: str | Path | None = None):
        """读 YAML：建 backend、数据源、可选缓存，并解析因子表达式。"""
        logger.info("加载配置文件: %s", config_path)
        config = load_config(config_path)
        engine, factor = cls.from_loaded_config(config, cache_root=cache_root)
        return engine, factor, config

    @classmethod
    def run_from_config(cls, config_path: str | Path, *,
                        cache_root: str | Path | None = None):
        """一键从配置文件跑因子，结果里附带 config 对象。"""
        logger.info("开始从配置执行因子: %s", config_path)
        engine, factor, config = cls.from_config(config_path, cache_root=cache_root)
        result = engine.run(factor)
        result["config"] = config
        logger.info("完成从配置执行因子: %s", factor.name)
        return result

    @classmethod
    def materialize_from_config(
        cls,
        config_path: str | Path,
        *,
        output_path: str | Path,
        factor_id: str | None = None,
        author: str | None = None,
        frequency: str | None = None,
        description: str | None = None,
        expression: str | None = None,
    ):
        """一键从配置文件执行因子并落盘。

        Args:
            config_path: 配置文件路径。
            output_path: **必填**。因子值落盘根路径。
                落盘到 ``{output_path}/{factor_id}/data/{YYYY-MM-DD}.parquet``。
        """
        logger.info("开始从配置物化因子: %s", config_path)
        engine, factor, config = cls.from_config(config_path)
        materialization = config.materialization
        result = engine.materialize(
            factor,
            output_path=output_path,
            factor_id=factor_id or (materialization.factor_id if materialization else None),
            author=author or (materialization.author if materialization else None),
            frequency=frequency or (materialization.frequency if materialization else None),
            description=description or (materialization.description if materialization else None),
            expression=(
                expression
                or (materialization.expression if materialization else None)
                or config.factor.expr
            ),
        )
        result["config"] = config
        logger.info("完成从配置物化因子: %s", factor.name)
        return result

    # ── 单因子执行 + 落盘 ──

    def materialize(
        self,
        factor: Factor,
        *,
        output_path: str | Path,
        factor_id: str | None = None,
        author: str | None = None,
        frequency: str | None = None,
        description: str | None = None,
        expression: str | None = None,
    ):
        """执行单因子并将结果落盘到指定路径（按日分区 Parquet）。

        Args:
            output_path: **必填**。输出根路径。因子值落盘到
                ``{output_path}/{factor_id}/data/{YYYY-MM-DD}.parquet``。
                不提供默认值，必须显式传入。
            factor_id: 因子 ID。默认使用 ``factor.name``。
        """
        logger.info("开始落盘因子 '%s' → %s", factor.name, output_path)
        output = self.run(factor)
        materializer = ParquetMaterializer(lake_root=output_path)
        summary = materializer.materialize(
            factor_id=factor_id or factor.name,
            result=output["result"],
            ir_node=output.get("analysis") and output["analysis"].ir,
            author=author,
            frequency=frequency or factor.freq,
            description=description or factor.description,
            expression=expression,
            output_path=output_path,
        )
        output["materialization"] = {
            **summary,
            "output_path": str(output_path),
        }
        logger.info(
            "完成落盘因子 '%s'，factor_id=%s，rows_written=%s",
            factor.name,
            summary["factor_id"],
            summary["rows_written"],
        )
        return output

    # ── 多因子 ──

    def _dag_from_factors(
        self,
        factors: Sequence[Factor],
        *,
        enable_cse: bool | None = None,
        perf: PerfConfig | None = None,
    ) -> tuple[DAGPlan, dict[str, AnalysisResult]]:
        """编译多因子并可选 CSE；每个因子只 ``compile`` 一次。"""
        perf = perf or PerfConfig.from_env()
        if enable_cse is None:
            enable_cse = perf.enable_cse
        plans: list[PlanNode] = []
        names: list[str] = []
        analyses: dict[str, AnalysisResult] = {}
        for factor in factors:
            plan, analysis = self.compile(factor)
            plans.append(plan)
            names.append(factor.name)
            analyses[factor.name] = analysis
        if enable_cse and len(plans) > 0:
            new_plans, shared = apply_cse(plans)
        else:
            new_plans, shared = plans, {}
        roots = [
            FactorPlan(factor_name=n, root=r) for n, r in zip(names, new_plans, strict=True)
        ]
        return DAGPlan(roots=roots, shared_nodes=shared), analyses

    def compile_many(
        self,
        factors: Sequence[Factor],
        *,
        enable_cse: bool | None = None,
        perf: PerfConfig | None = None,
    ) -> DAGPlan:
        dag, _ = self._dag_from_factors(factors, enable_cse=enable_cse, perf=perf)
        return dag

    def run_many(
        self,
        factors: Sequence[Factor],
        *,
        perf: PerfConfig | None = None,
        enable_cse: bool | None = None,
    ) -> dict[str, Any]:
        dag, analyses = self._dag_from_factors(factors, enable_cse=enable_cse, perf=perf)
        perf = perf or PerfConfig.from_env()
        ctx = ExecutionContext(
            data_source=self.data_source,
            cache=self.cache,
            shared_result_cache={},
            perf=perf,
        )
        if ctx.shared_result_cache is not None:
            for sid, sub in dag.shared_nodes.items():
                ctx.shared_result_cache[sid] = self.backend.execute(sub, ctx)
        out: dict[str, Any] = {}
        for fp in dag.roots:
            out[fp.factor_name] = self.backend.execute(fp.root, ctx)
        return {"results": out, "dag": dag, "analyses": analyses}

    def run_many_parallel(
        self,
        factors: Sequence[Factor],
        *,
        n_jobs: int | None = None,
        perf: PerfConfig | None = None,
        enable_cse: bool | None = None,
    ) -> dict[str, Any]:
        try:
            from joblib import Parallel, delayed
        except ImportError as exc:
            raise ImportError(
                "run_many_parallel 需要 joblib：pip install 'factor-engine[parallel]'"
            ) from exc

        dag, analyses = self._dag_from_factors(factors, enable_cse=enable_cse, perf=perf)
        perf = perf or PerfConfig.from_env()
        workers = n_jobs if n_jobs is not None else perf.max_workers
        ctx = ExecutionContext(
            data_source=self.data_source,
            cache=self.cache,
            shared_result_cache={},
            perf=perf,
        )
        if ctx.shared_result_cache is not None:
            for sid, sub in dag.shared_nodes.items():
                ctx.shared_result_cache[sid] = self.backend.execute(sub, ctx)

        def _one(fp: FactorPlan):
            res = self.backend.execute(fp.root, ctx)
            return fp.factor_name, res

        raw = Parallel(n_jobs=workers, backend="threading")(
            delayed(_one)(fp) for fp in dag.roots
        )
        results = dict(raw)
        return {"results": results, "dag": dag, "analyses": analyses}


# ============================================================
# 小体量数据源包装器
# ============================================================

class _TruncatedDataSource:
    """包装 DataSource，只返回前 N 个标的和前 N 个时间点的数据。"""

    def __init__(self, inner: Any, n_instruments: int = 5, n_days: int = 10):
        self._inner = inner
        self._n_instruments = n_instruments
        self._n_days = n_days
        logger.info(
            "tiny_run 模式: 限制 %d 个标的, %d 个交易日",
            n_instruments, n_days,
        )

    def load_column(self, name: str):
        series = self._inner.load_column(name)
        return self._truncate(series)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def _truncate(self, series):
        if not hasattr(series, "index"):
            return series
        import pandas as pd
        if not isinstance(series.index, pd.MultiIndex):
            return series.head(self._n_days)
        # MultiIndex: (timestamp, instrument)
        instruments = series.index.get_level_values(1).unique()
        keep_inst = instruments[:self._n_instruments]
        mask = series.index.get_level_values(1).isin(keep_inst)
        truncated = series.loc[mask]
        # 再截取时间
        times = truncated.index.get_level_values(0).unique()
        keep_times = times[:self._n_days]
        mask2 = truncated.index.get_level_values(0).isin(keep_times)
        return truncated.loc[mask2]
