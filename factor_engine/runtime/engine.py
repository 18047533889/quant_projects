"""运行时入口：``FactorEngine`` 串联 compile 与 run。

编译链（单因子）::

    Factor.expr  →  Analyzer.lower  →  IRNode
                 →  Optimizer       →  IRNode（可选规则）
                 →  Lowerer         →  PlanNode
                 →  PandasBackend   →  MultiIndex Series

``run_many`` / CSE：多因子共享子表达式时插入 ``plan_ref`` 节点与 ``shared_result_cache``。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from api.dsl_parser import parse_factor
from api.factor import Factor
from backend.context import ExecutionContext
from backend.factory import build_backend
from ir.analyzer import AnalysisResult, Analyzer
from logging_utils import get_logger
from planner.cse import apply_cse
from planner.dag import DAGPlan, FactorPlan
from planner.logical_plan import PlanNode
from planner.lowerer import Lowerer
from planner.optimizer import Optimizer
from runtime.config import FactorEngineConfig, load_config
from runtime.perf_config import PerfConfig
from storage.cache import CacheManager, PersistentPlanCache
from storage.data_scope import compute_data_scope
from storage.factory import build_data_source
from storage.materializer import ParquetMaterializer

logger = get_logger("runtime.engine")


class FactorEngine:
    """因子引擎：注入后端与数据源，对 :class:`api.factor.Factor` 做编译与执行。"""

    def __init__(self, backend, data_source, cache=None) -> None:
        self.backend = backend  # PandasBackend / PolarsBackend / …，由 build_backend 构造
        self.data_source = data_source  # 从 parquet 等拉 MultiIndex 面板的统一入口
        self.cache = cache  # 列级缓存；无则每次 execute 全量算
        self.analyzer = Analyzer()  # Expr → IR + 依赖列分析
        self.lowerer = Lowerer()  # IR → 逻辑计划树
        self.optimizer = Optimizer()  # 计划级优化（常折叠等）

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
            new_plans, shared = apply_cse(plans)  # shared：子树 ID → 可复用 PlanNode
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
        """多因子编译：可选 **公共子表达式消除（CSE）**，重复子树只保留一份于 ``shared_nodes``。

        CSE 默认开启；可用环境变量 ``FACTOR_ENGINE_DISABLE_CSE=1`` 关闭，或传入
        ``perf=PerfConfig.from_env()`` / ``enable_cse=False``。
        """
        dag, _ = self._dag_from_factors(
            factors, enable_cse=enable_cse, perf=perf
        )
        return dag

    @classmethod
    def _build_cache(cls, config: FactorEngineConfig, data_source) -> CacheManager | None:
        if not config.engine.enable_cache and not config.engine.plan_cache_dir:
            return None
        scope = compute_data_scope(data_source)
        if config.engine.plan_cache_dir:
            return PersistentPlanCache(config.engine.plan_cache_dir, data_scope=scope)
        return CacheManager(data_scope=scope)

    @classmethod
    def from_loaded_config(cls, config: FactorEngineConfig):
        """从已解析配置构造 engine 与 factor，供 pipeline 等编排层复用。"""
        backend = build_backend(config.backend.type)
        data_source = build_data_source(config.data_source)
        cache = cls._build_cache(config, data_source)
        factor = parse_factor(
            config.factor.expr,
            name=config.factor.name,
            freq=config.factor.freq,
            universe=config.factor.universe,
            description=config.factor.description,
        )
        logger.info(
            "配置对象加载完成: factor=%s, backend=%s, data_source=%s, cache=%s",
            factor.name,
            type(backend).__name__,
            type(data_source).__name__,
            "on" if cache is not None else "off",
        )
        return cls(backend=backend, data_source=data_source, cache=cache), factor

    @classmethod
    def from_config(cls, config_path: str | Path):
        """读 YAML：建 backend、数据源、可选缓存，并解析因子表达式。"""
        logger.info("加载配置文件: %s", config_path)
        config = load_config(config_path)
        engine, factor = cls.from_loaded_config(config)
        return engine, factor, config

    @classmethod
    def run_from_config(cls, config_path: str | Path):
        """一键从配置文件跑因子，结果里附带 config 对象。"""
        logger.info("开始从配置执行因子: %s", config_path)
        engine, factor, config = cls.from_config(config_path)
        result = engine.run(factor)
        result["config"] = config
        logger.info("完成从配置执行因子: %s", factor.name)
        return result

    @classmethod
    def materialize_from_config(
        cls,
        config_path: str | Path,
        *,
        lake_root: str | Path | None = None,
        factor_id: str | None = None,
        author: str | None = None,
        frequency: str | None = None,
        description: str | None = None,
        expression: str | None = None,
    ):
        """一键从配置文件执行因子并落盘到因子湖（Parquet）。"""
        logger.info("开始从配置物化因子: %s", config_path)
        engine, factor, config = cls.from_config(config_path)
        materialization = config.materialization
        result = engine.materialize(
            factor,
            lake_root=lake_root or (materialization.lake_root if materialization else None),
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

    def _make_context(
        self,
        *,
        shared_result_cache: dict[str, Any] | None = None,
        perf: PerfConfig | None = None,
    ) -> ExecutionContext:
        from storage.long_table_source import LongTableDataSource

        prefer_long = isinstance(self.data_source, LongTableDataSource)
        return ExecutionContext(
            data_source=self.data_source,
            cache=self.cache,
            shared_result_cache=shared_result_cache,
            panel_cache={},
            materialized_series={},
            prefer_long_table=prefer_long,
            perf=perf,
        )

    @staticmethod
    def _lineage_extra(
        *,
        data_source_config: dict | None,
        snapshot_id: str | None,
        input_dq: dict | None = None,
        **more: Any,
    ) -> dict[str, Any]:
        from runtime.lineage import resolve_git_commit_hash

        extra: dict[str, Any] = {
            "data_snapshot_id": snapshot_id,
            "data_source_config": data_source_config,
            "git_commit": resolve_git_commit_hash(),
        }
        if input_dq is not None:
            extra["input_dq"] = input_dq
        extra.update(more)
        return extra

    @staticmethod
    def _prefetch_columns(data_source: Any, columns: set[str]) -> None:
        if not columns:
            return
        names = sorted(columns)
        prefetch_panels = getattr(data_source, "prefetch_panels", None)
        if callable(prefetch_panels):
            prefetch_panels(names)
            logger.info("预加载宽表 panel（prefetch_panels）: %s", names)
            return
        prefetch = getattr(data_source, "prefetch_columns", None)
        if callable(prefetch):
            prefetch(names)
            logger.info("预加载数据列（prefetch）: %s", names)
            return
        load_columns = getattr(data_source, "load_columns", None)
        if callable(load_columns):
            load_columns(names)
            logger.info("预加载数据列: %s", names)

    def run(
        self,
        factor: Factor,
        *,
        plan=None,
        analysis=None,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
    ):
        """编译后调用 ``backend.execute``，返回 factor、analysis、plan、result。"""
        started_at = time.perf_counter()
        logger.info("开始执行因子 '%s'", factor.name)
        if plan is None or analysis is None:
            plan, analysis = self.compile(factor)
        from cleaned_operators.operator_policy import effective_lookback

        history_buffer = effective_lookback(getattr(analysis, "lookback", 0))
        if history_buffer > 0:
            logger.info(
                "因子 '%s' 建议历史缓冲 >= %d bars（lookback=%s）",
                factor.name,
                history_buffer,
                getattr(analysis, "lookback", 0),
            )
        if input_dq_check and analysis.referenced_columns:
            from runtime.input_dq import assert_input_dq

            input_report = assert_input_dq(
                self.data_source,
                analysis.referenced_columns,
                raise_on_fail=input_dq_strict,
            )
            logger.info("因子 '%s' 输入 DQ 通过", factor.name)
        else:
            input_report = None
        self._prefetch_columns(self.data_source, analysis.referenced_columns)
        ctx = self._make_context()
        result = self.backend.execute(plan, ctx)
        non_null_count = int(result.notna().sum()) if hasattr(result, "notna") else None
        logger.info(
            "完成执行因子 '%s'，结果行数=%s，非空=%s，耗时 %.2fs",
            factor.name,
            len(result),
            non_null_count,
            time.perf_counter() - started_at,
        )
        out = {
            "factor": factor,
            "analysis": analysis,
            "plan": plan,
            "result": result,
        }
        if input_report is not None:
            out["input_dq"] = input_report.to_dict()
        return out

    @staticmethod
    def _data_snapshot_id_from_config(data_source_config: dict | None) -> str | None:
        if not data_source_config:
            return None
        from runtime.lineage import hash_data_source_config

        return hash_data_source_config(data_source_config)

    def materialize(
        self,
        factor: Factor,
        *,
        lake_root: str | Path | None = None,
        factor_id: str | None = None,
        author: str | None = None,
        frequency: str | None = None,
        description: str | None = None,
        expression: str | None = None,
        dq_check: bool = False,
        dq_strict: bool = True,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        data_source_config: dict | None = None,
        write_metadata: bool = True,
        resume_materialize: bool = False,
        isolate_partition_failures: bool = True,
    ):
        """执行单因子并将结果落盘到 factor lake（Parquet）。"""
        logger.info("开始落盘因子 '%s'", factor.name)
        output = self.run(
            factor,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
        )
        from backend.cleaned_bridge import ensure_cleaned_loaded
        from cleaned_operators.operator_policy import (
            compute_operator_catalog_hash,
            effective_lookback,
        )
        from runtime.lineage import build_run_lineage

        ensure_cleaned_loaded()
        from storage.catalog import compute_ir_hash

        op_hash = compute_operator_catalog_hash()
        analysis = output["analysis"]
        snapshot_id = self._data_snapshot_id_from_config(data_source_config)
        lineage = build_run_lineage(
            factor_id=factor_id or factor.name,
            factor_name=factor.name,
            ast_hash=compute_ir_hash(analysis.ir),
            operator_catalog_hash=op_hash,
            expression=expression or getattr(factor, "description", None),
            lookback=effective_lookback(getattr(analysis, "lookback", 0)),
            referenced_columns=analysis.referenced_columns,
            result=output["result"],
            extra=self._lineage_extra(
                data_source_config=data_source_config,
                snapshot_id=snapshot_id,
                input_dq=output.get("input_dq"),
            ),
        )
        materializer = ParquetMaterializer(lake_root=lake_root)
        summary = materializer.materialize(
            factor_id=factor_id or factor.name,
            result=output["result"],
            ir_node=output["analysis"].ir,
            author=author,
            frequency=frequency or factor.freq,
            description=description or factor.description,
            expression=expression,
            dq_check=dq_check,
            dq_strict=dq_strict,
            run_lineage={**lineage.to_dict(), "factor_id": factor_id or factor.name},
            write_metadata=write_metadata,
            data_snapshot_id=snapshot_id,
            data_source_config=data_source_config,
            resume=resume_materialize,
            isolate_partition_failures=isolate_partition_failures,
        )
        output["materialization"] = {
            **summary,
            "lake_root": str(materializer.lake_root),
        }
        logger.info(
            "完成落盘因子 '%s'，factor_id=%s，rows_written=%s",
            factor.name,
            summary["factor_id"],
            summary["rows_written"],
        )
        return output

    def materialize_clickhouse(
        self,
        factor: Factor,
        *,
        factor_id: str | None = None,
        table: str = "factor_values",
        timestamp_column: str = "trade_date",
        instrument_column: str = "instrument",
        factor_version: str = "",
        data_source_config: dict | None = None,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        ch_host: str | None = None,
        ch_port: int | None = None,
        ch_database: str | None = None,
        ch_username: str | None = None,
        ch_password: str | None = None,
        ch_secure: bool | None = None,
        ensure_table: bool = True,
    ):
        """执行因子并将结果写入 ClickHouse（ReplacingMergeTree 长表）。"""
        from storage.clickhouse_materializer import ClickHouseMaterializer

        logger.info("开始 ClickHouse 落盘因子 '%s'", factor.name)
        output = self.run(
            factor,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
        )
        snapshot_id = self._data_snapshot_id_from_config(data_source_config)
        mat = ClickHouseMaterializer(
            table=table,
            timestamp_column=timestamp_column,
            instrument_column=instrument_column,
            host=ch_host,
            port=ch_port,
            database=ch_database,
            username=ch_username,
            password=ch_password,
            secure=ch_secure,
        )
        summary = mat.materialize(
            factor_id=factor_id or factor.name,
            result=output["result"],
            factor_version=factor_version,
            data_snapshot_id=snapshot_id,
            ensure_table=ensure_table,
        )
        output["clickhouse_materialization"] = {
            "factor_id": summary.factor_id,
            "table": summary.table,
            "rows_written": summary.rows_written,
            "database": summary.database,
        }
        logger.info(
            "完成 ClickHouse 落盘因子 '%s'，rows_written=%s",
            factor.name,
            summary.rows_written,
        )
        return output

    def run_many(
        self,
        factors: Sequence[Factor],
        *,
        perf: PerfConfig | None = None,
        enable_cse: bool | None = None,
    ) -> dict[str, Any]:
        """多因子求值：先执行 ``DAGPlan.shared_nodes``，再各因子根；含 ``plan_ref`` 时必须用此入口。"""
        dag, analyses = self._dag_from_factors(
            factors, enable_cse=enable_cse, perf=perf
        )
        perf = perf or PerfConfig.from_env()
        all_cols: set[str] = set()
        for analysis in analyses.values():
            all_cols |= analysis.referenced_columns
        self._prefetch_columns(self.data_source, all_cols)
        ctx = self._make_context(shared_result_cache={}, perf=perf)
        # 先算共享子式，再算各因子根，避免重复执行相同子树
        if ctx.shared_result_cache is not None:
            for sid, sub in dag.shared_nodes.items():
                ctx.shared_result_cache[sid] = self.backend.execute(sub, ctx)
        out: dict[str, Any] = {}
        for fp in dag.roots:
            out[fp.factor_name] = self.backend.execute(fp.root, ctx)
        return {
            "results": out,
            "dag": dag,
            "analyses": analyses,
        }

    def run_many_parallel(
        self,
        factors: Sequence[Factor],
        *,
        n_jobs: int | None = None,
        perf: PerfConfig | None = None,
        enable_cse: bool | None = None,
    ) -> dict[str, Any]:
        """在 ``run_many`` 基础上对**各因子根**并行求值（共享子式仍先串行算完）。"""
        try:
            from joblib import Parallel, delayed
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "run_many_parallel 需要 joblib：pip install 'factor-engine[parallel]'"
            ) from exc

        dag, analyses = self._dag_from_factors(
            factors, enable_cse=enable_cse, perf=perf
        )
        perf = perf or PerfConfig.from_env()
        workers = n_jobs if n_jobs is not None else perf.max_workers
        all_cols: set[str] = set()
        for analysis in analyses.values():
            all_cols |= analysis.referenced_columns
        self._prefetch_columns(self.data_source, all_cols)
        ctx = self._make_context(shared_result_cache={}, perf=perf)
        if ctx.shared_result_cache is not None:
            for sid, sub in dag.shared_nodes.items():
                ctx.shared_result_cache[sid] = self.backend.execute(sub, ctx)

        def _one(fp: FactorPlan):
            res = self.backend.execute(fp.root, ctx)
            return fp.factor_name, res

        # 默认 threading：与 DataFrame 共享内存，避免多进程序列化大面板；CPU 极重时仍可改策略
        raw = Parallel(n_jobs=workers, backend="threading")(
            delayed(_one)(fp) for fp in dag.roots
        )
        results = dict(raw)
        return {"results": results, "dag": dag, "analyses": analyses}

    def with_data_source(self, data_source, *, fresh_cache: bool = False) -> FactorEngine:
        """返回共享 backend 的新引擎实例（用于增量时间窗口）。

        ``fresh_cache=True`` 时丢弃内存层子计划缓存；磁盘持久化缓存按新
        ``data_scope`` 隔离，仍可命中同窗口历史结果。
        """
        if self.cache is None:
            return FactorEngine(backend=self.backend, data_source=data_source, cache=None)
        scope = compute_data_scope(data_source)
        if isinstance(self.cache, PersistentPlanCache):
            new_cache = self.cache.with_scope(scope, clear_memory=fresh_cache)
            return FactorEngine(backend=self.backend, data_source=data_source, cache=new_cache)
        if fresh_cache:
            return FactorEngine(
                backend=self.backend,
                data_source=data_source,
                cache=CacheManager(data_scope=scope),
            )
        if getattr(self.cache, "data_scope", None) != scope:
            return FactorEngine(
                backend=self.backend,
                data_source=data_source,
                cache=CacheManager(data_scope=scope),
            )
        return FactorEngine(backend=self.backend, data_source=data_source, cache=self.cache)

    def run_incremental(
        self,
        factor: Factor,
        *,
        factor_id: str | None = None,
        since: str | None = None,
        end_date: str | None = None,
        lookback_extra: int = 5,
        recompute_tail_bars: int | None = None,
        lake_root: str | Path | None = None,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        market: str | None = None,
    ):
        """增量执行：按 watermark 加载 lookback 窗口，仅输出/落盘 tail 区间。"""
        from runtime.incremental import (
            build_incremental_plan,
            slice_factor_result_for_incremental,
        )
        from storage.materializer import ParquetMaterializer
        from storage.time_window import narrow_data_source_for_window
        from storage.trading_calendar import infer_market

        fid = factor_id or factor.name
        materializer = ParquetMaterializer(lake_root=lake_root)
        watermark = materializer.catalog.get_watermark(fid)

        dataset = getattr(self.data_source, "dataset", None)
        resolved_market = market or infer_market(
            universe=getattr(factor, "universe", None),
            dataset=str(dataset) if dataset else None,
        )

        plan, analysis = self.compile(factor)
        inc = build_incremental_plan(
            factor_id=fid,
            analysis_lookback=getattr(analysis, "lookback", 0),
            watermark=watermark,
            since=since,
            end_date=end_date,
            lookback_extra=lookback_extra,
            recompute_tail_bars=recompute_tail_bars,
            market=resolved_market,
        )

        if inc.is_full_run:
            scoped = self
        else:
            narrowed = narrow_data_source_for_window(
                self.data_source,
                start_date=inc.load_start,
                end_date=inc.load_end,
            )
            # 持久化缓存按 data_scope 隔离；内存缓存需 fresh_cache 避免跨窗口命中
            use_fresh = not isinstance(self.cache, PersistentPlanCache)
            scoped = self.with_data_source(narrowed, fresh_cache=use_fresh)

        logger.info(
            "增量因子 '%s' plan: full=%s load=[%s,%s] output=[%s,%s] lookback=%d",
            factor.name,
            inc.is_full_run,
            inc.load_start,
            inc.load_end,
            inc.output_start,
            inc.output_end,
            inc.lookback_bars,
        )

        output = scoped.run(
            factor,
            plan=plan,
            analysis=analysis,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
        )
        sliced = slice_factor_result_for_incremental(output["result"], inc)
        output["result"] = sliced
        output["incremental"] = {**inc.to_dict(), "market": resolved_market}
        output["analysis"] = analysis
        output["plan"] = plan
        return output

    def materialize_incremental(
        self,
        factor: Factor,
        *,
        factor_id: str | None = None,
        since: str | None = None,
        end_date: str | None = None,
        lookback_extra: int = 5,
        recompute_tail_bars: int | None = None,
        lake_root: str | Path | None = None,
        dq_check: bool = False,
        dq_strict: bool = True,
        author: str | None = None,
        frequency: str | None = None,
        description: str | None = None,
        expression: str | None = None,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        data_source_config: dict | None = None,
        write_metadata: bool = True,
        resume_materialize: bool = False,
        isolate_partition_failures: bool = True,
    ):
        """增量执行 + 落盘 upsert + lineage。"""
        from backend.cleaned_bridge import ensure_cleaned_loaded
        from cleaned_operators.operator_policy import compute_operator_catalog_hash
        from runtime.lineage import build_run_lineage
        from storage.catalog import compute_ir_hash

        output = self.run_incremental(
            factor,
            factor_id=factor_id,
            since=since,
            end_date=end_date,
            lookback_extra=lookback_extra,
            recompute_tail_bars=recompute_tail_bars,
            lake_root=lake_root,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
        )

        if output["result"] is None or len(output["result"]) == 0:
            logger.warning("增量因子 '%s' 无新输出，跳过落盘", factor.name)
            output["materialization"] = {
                "factor_id": factor_id or factor.name,
                "rows_written": 0,
                "skipped": True,
                "incremental": output.get("incremental"),
            }
            return output

        ensure_cleaned_loaded()
        fid = factor_id or factor.name
        op_hash = compute_operator_catalog_hash()
        analysis = output["analysis"]
        snapshot_id = self._data_snapshot_id_from_config(data_source_config)
        lineage = build_run_lineage(
            factor_id=fid,
            factor_name=factor.name,
            ast_hash=compute_ir_hash(analysis.ir),
            operator_catalog_hash=op_hash,
            expression=expression or getattr(factor, "description", None),
            lookback=output["incremental"]["lookback_bars"],
            referenced_columns=analysis.referenced_columns,
            result=output["result"],
            extra=self._lineage_extra(
                data_source_config=data_source_config,
                snapshot_id=snapshot_id,
                input_dq=output.get("input_dq"),
                mode="incremental",
                incremental=output["incremental"],
            ),
        )

        materializer = ParquetMaterializer(lake_root=lake_root)
        summary = materializer.materialize(
            factor_id=fid,
            result=output["result"],
            ir_node=analysis.ir,
            author=author,
            frequency=frequency or factor.freq,
            description=description or factor.description,
            expression=expression,
            dq_check=dq_check,
            dq_strict=dq_strict,
            run_lineage={**lineage.to_dict(), "factor_id": fid},
            write_metadata=write_metadata,
            data_snapshot_id=snapshot_id,
            data_source_config=data_source_config,
            resume=resume_materialize,
            isolate_partition_failures=isolate_partition_failures,
        )
        output["materialization"] = {
            **summary,
            "lake_root": str(materializer.lake_root),
            "incremental": output.get("incremental"),
        }
        return output
