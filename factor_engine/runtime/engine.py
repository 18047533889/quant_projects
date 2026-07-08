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

import pandas as pd

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
from runtime.env_bootstrap import bootstrap_runtime_env
from runtime.perf_config import PerfConfig
from storage.cache import CacheManager, PersistentPlanCache
from storage.data_scope import compute_data_scope
from storage.factory import build_data_source
from storage.materializer import ParquetMaterializer

logger = get_logger("runtime.engine")


class FactorEngine:
    """因子引擎：注入后端与数据源，对 :class:`api.factor.Factor` 做编译与执行。"""

    def __init__(self, backend, data_source, cache=None) -> None:
        bootstrap_runtime_env()
        self.backend = backend  # PandasBackend / PolarsBackend / …，由 build_backend 构造
        self.data_source = data_source  # 从 parquet 等拉 MultiIndex 面板的统一入口
        self.cache = cache  # 列级缓存；无则每次 execute 全量算
        self.analyzer = Analyzer()  # Expr → IR + 依赖列分析
        self.lowerer = Lowerer()  # IR → 逻辑计划树
        self.optimizer = Optimizer()  # 计划级优化（常折叠等）

    def compile(self, factor: Factor, *, pit_enforce: bool = False, pit_forbid_forward_fill: bool = False):
        """Expr → Analyzer → Lowerer → Optimizer，返回 (plan, analysis)。"""
        started_at = time.perf_counter()
        logger.info("开始编译因子 '%s'", factor.name)
        analysis = self.analyzer.lower(factor.expr)
        if pit_enforce:
            from runtime.pit_audit import assert_pit_safe

            assert_pit_safe(
                analysis.ir,
                enforce=True,
                forbid_forward_fill=pit_forbid_forward_fill,
            )
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
        from runtime.config_runtime import resolve_run_kwargs

        opts = resolve_run_kwargs(config)
        result = engine.run(
            factor,
            auto_warmup=opts.auto_warmup,
            trim_warmup=opts.trim_warmup,
            market=opts.market,
            input_dq_check=opts.input_dq_check,
            input_dq_strict=opts.input_dq_strict,
            input_dq_thresholds=opts.input_dq_thresholds,
            pit_enforce=opts.pit_enforce,
            pit_forbid_forward_fill=opts.pit_forbid_forward_fill,
        )
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
        from runtime.config_runtime import resolve_materialize_kwargs

        opts = resolve_materialize_kwargs(
            config,
            lake_root_override=str(lake_root) if lake_root is not None else None,
            factor_id_override=factor_id,
            author_override=author,
            frequency_override=frequency,
            description_override=description,
            expression_override=expression,
        )
        result = engine.materialize(
            factor,
            lake_root=opts.lake_root,
            factor_id=opts.factor_id,
            author=opts.author,
            frequency=opts.frequency,
            description=opts.description,
            expression=opts.expression,
            auto_warmup=opts.auto_warmup,
            trim_warmup=opts.trim_warmup,
            market=opts.market,
            dq_check=opts.dq_check,
            dq_strict=opts.dq_strict,
            dq_thresholds=opts.dq_thresholds,
            input_dq_check=opts.input_dq_check,
            input_dq_strict=opts.input_dq_strict,
            input_dq_thresholds=opts.input_dq_thresholds,
            preserve_invalid_rows=opts.preserve_invalid_rows,
            value_dtype=opts.value_dtype,
            write_target=opts.write_target,
            data_source_config=opts.data_source_config,
            pit_enforce=opts.pit_enforce,
            pit_forbid_forward_fill=opts.pit_forbid_forward_fill,
            clickhouse_table=opts.clickhouse_table,
            ch_host=opts.clickhouse_host,
            ch_port=opts.clickhouse_port,
            ch_database=opts.clickhouse_database,
            ch_username=opts.clickhouse_username,
            ch_password=opts.clickhouse_password,
            ch_secure=opts.clickhouse_secure,
            ch_ensure_table=opts.ch_ensure_table,
        )
        result["config"] = config
        logger.info("完成从配置物化因子: %s", factor.name)
        return result

    @classmethod
    def materialize_incremental_from_config(
        cls,
        config_path: str | Path,
        *,
        lake_root: str | Path | None = None,
        factor_id: str | None = None,
        since: str | None = None,
        end_date: str | None = None,
        lookback_extra: int | None = None,
        recompute_tail_bars: int | None = None,
    ):
        """从 YAML 配置执行增量物化（watermark + lookback + upsert）。"""
        logger.info("开始从配置增量物化因子: %s", config_path)
        engine, factor, config = cls.from_config(config_path)
        from runtime.config_runtime import resolve_materialize_kwargs

        opts = resolve_materialize_kwargs(
            config,
            lake_root_override=str(lake_root) if lake_root is not None else None,
            factor_id_override=factor_id,
            since_override=since,
            end_date_override=end_date,
            lookback_extra_override=lookback_extra,
            recompute_tail_bars_override=recompute_tail_bars,
        )
        result = engine.materialize_incremental(
            factor,
            lake_root=opts.lake_root,
            factor_id=opts.factor_id,
            since=opts.since,
            end_date=opts.end_date,
            lookback_extra=opts.lookback_extra,
            recompute_tail_bars=opts.recompute_tail_bars,
            author=opts.author,
            frequency=opts.frequency,
            description=opts.description,
            expression=opts.expression,
            input_dq_check=opts.input_dq_check,
            input_dq_strict=opts.input_dq_strict,
            input_dq_thresholds=opts.input_dq_thresholds,
            auto_warmup=opts.auto_warmup,
            trim_warmup=opts.trim_warmup,
            market=opts.market,
            dq_check=opts.dq_check,
            dq_strict=opts.dq_strict,
            dq_thresholds=opts.dq_thresholds,
            preserve_invalid_rows=opts.preserve_invalid_rows,
            value_dtype=opts.value_dtype,
            write_target=opts.write_target,
            data_source_config=opts.data_source_config,
            pit_enforce=opts.pit_enforce,
            pit_forbid_forward_fill=opts.pit_forbid_forward_fill,
            isolate_partition_failures=opts.isolate_partition_failures,
            resume_materialize=opts.resume_materialize,
            clickhouse_table=opts.clickhouse_table,
            ch_host=opts.clickhouse_host,
            ch_port=opts.clickhouse_port,
            ch_database=opts.clickhouse_database,
            ch_username=opts.clickhouse_username,
            ch_password=opts.clickhouse_password,
            ch_secure=opts.clickhouse_secure,
            ch_ensure_table=opts.ch_ensure_table,
        )
        result["config"] = config
        logger.info("完成从配置增量物化因子: %s", factor.name)
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
    def _resolve_parquet_write_target(write_target: str) -> str:
        """多目标物化时 ParquetMaterializer 使用的 write_target。"""
        target = str(write_target or "local").lower()
        if target == "staging_clickhouse":
            return "staging"
        if target == "clickhouse":
            return "clickhouse"
        return target

    @staticmethod
    def _needs_clickhouse_write(write_target: str) -> bool:
        return str(write_target or "local").lower() in ("clickhouse", "staging_clickhouse")

    @staticmethod
    def _append_clickhouse_to_summary(
        summary: dict[str, Any],
        *,
        factor_id: str,
        result: pd.Series,
        ast_hash: str,
        write_target: str,
        data_snapshot_id: str | None = None,
        clickhouse_table: str | None = None,
        preserve_invalid_rows: bool = False,
        value_dtype: str = "float32",
        write_metadata: bool = True,
        ensure_table: bool = True,
        dq_check: bool = False,
        dq_strict: bool = True,
        dq_thresholds=None,
        ch_host: str | None = None,
        ch_port: int | None = None,
        ch_database: str | None = None,
        ch_username: str | None = None,
        ch_password: str | None = None,
        ch_secure: bool | None = None,
    ) -> dict[str, Any]:
        """在 Parquet/staging 落盘后追加 ClickHouse 写入，返回更新后的 summary。"""
        from storage.clickhouse_materializer import ClickHouseMaterializer
        from storage.exceptions import DualWriteError

        ch_mat = ClickHouseMaterializer(
            table=clickhouse_table or "factor_values",
            host=ch_host,
            port=ch_port,
            database=ch_database,
            username=ch_username,
            password=ch_password,
            secure=ch_secure,
        )
        partial = dict(summary)
        partial["write_target"] = str(write_target).lower()
        partial["primary_write_completed"] = True
        try:
            ch_summary = ch_mat.materialize(
                factor_id=factor_id,
                result=result,
                factor_version=ast_hash[:16],
                data_snapshot_id=data_snapshot_id,
                ensure_table=ensure_table,
                dq_check=dq_check,
                dq_strict=dq_strict,
                dq_thresholds=dq_thresholds,
                preserve_invalid_rows=preserve_invalid_rows,
                value_dtype=value_dtype,
                write_metadata=write_metadata,
            )
        except Exception as exc:
            partial["partial_write"] = True
            partial["clickhouse_error"] = str(exc)
            raise DualWriteError(
                f"ClickHouse 写入失败（主存储可能已成功）: factor_id={factor_id}, error={exc}",
                summary=partial,
                cause=exc,
            ) from exc

        merged = dict(partial)
        merged["clickhouse"] = {
            "factor_id": ch_summary.factor_id,
            "table": ch_summary.table,
            "rows_written": ch_summary.rows_written,
            "database": ch_summary.database,
        }
        if ch_summary.dq_report is not None:
            merged["clickhouse_dq_report"] = ch_summary.dq_report
        merged["partial_write"] = False
        return merged

    def _dual_write_clickhouse(
        self,
        materializer: ParquetMaterializer,
        summary: dict[str, Any],
        *,
        factor_id: str,
        result: pd.Series,
        ast_hash: str,
        write_target: str,
        data_snapshot_id: str | None = None,
        clickhouse_table: str | None = None,
        preserve_invalid_rows: bool = False,
        value_dtype: str = "float32",
        write_metadata: bool = True,
        ensure_table: bool = True,
        dq_check: bool = False,
        dq_strict: bool = True,
        dq_thresholds=None,
        ch_host: str | None = None,
        ch_port: int | None = None,
        ch_database: str | None = None,
        ch_username: str | None = None,
        ch_password: str | None = None,
        ch_secure: bool | None = None,
    ) -> dict[str, Any]:
        """ClickHouse 双写；若主存储延迟水位线，成功后再 commit、失败则 abort。"""
        from storage.exceptions import DualWriteError

        deferred = bool(summary.get("watermark_deferred"))
        try:
            merged = self._append_clickhouse_to_summary(
                summary,
                factor_id=factor_id,
                result=result,
                ast_hash=ast_hash,
                write_target=write_target,
                data_snapshot_id=data_snapshot_id,
                clickhouse_table=clickhouse_table,
                preserve_invalid_rows=preserve_invalid_rows,
                value_dtype=value_dtype,
                write_metadata=write_metadata,
                ensure_table=ensure_table,
                dq_check=dq_check,
                dq_strict=dq_strict,
                dq_thresholds=dq_thresholds,
                ch_host=ch_host,
                ch_port=ch_port,
                ch_database=ch_database,
                ch_username=ch_username,
                ch_password=ch_password,
                ch_secure=ch_secure,
            )
        except DualWriteError as exc:
            partial = exc.summary or summary
            if deferred:
                materializer.abort_deferred_materialization(
                    partial,
                    error=str(exc.cause or exc),
                )
            raise
        if deferred:
            return materializer.commit_deferred_materialization(merged)
        return merged

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
        input_dq_thresholds=None,
        auto_warmup: bool = False,
        trim_warmup: bool = True,
        market: str | None = None,
        pit_enforce: bool = False,
        pit_forbid_forward_fill: bool = False,
    ):
        """编译后调用 ``backend.execute``，返回 factor、analysis、plan、result。

        ``auto_warmup=True`` 时根据 lookback 向前扩展数据源加载窗口，计算后
        ``trim_warmup`` 为真则裁剪回用户请求的 start_date。
        """
        started_at = time.perf_counter()
        logger.info("开始执行因子 '%s'", factor.name)
        if plan is None or analysis is None:
            plan, analysis = self.compile(
                factor,
                pit_enforce=pit_enforce,
                pit_forbid_forward_fill=pit_forbid_forward_fill,
            )
        from cleaned_operators.operator_policy import (
            bars_per_day,
            effective_lookback,
            infer_source_bar_freq,
        )

        source_bar_freq = infer_source_bar_freq(
            self.data_source,
            fallback=getattr(factor, "freq", None),
        )
        history_buffer = effective_lookback(
            getattr(analysis, "lookback", 0),
            factor_freq=getattr(factor, "freq", None),
            source_bar_freq=source_bar_freq,
        )
        s_bpd = bars_per_day(source_bar_freq)
        warmup_calendar_bars = history_buffer
        if s_bpd > 1:
            warmup_calendar_bars = max(1, (history_buffer + s_bpd - 1) // s_bpd)
        run_window = None
        engine_to_use = self

        if auto_warmup and history_buffer > 0:
            from runtime.run_window import build_full_run_window, extract_source_date_bounds
            from storage.time_window import narrow_data_source_for_window, slice_series_time_window
            from storage.trading_calendar import get_trading_calendar, infer_market

            req_start, req_end = extract_source_date_bounds(self.data_source)
            dataset = getattr(self.data_source, "dataset", None)
            resolved_market = market or infer_market(
                universe=getattr(factor, "universe", None),
                dataset=str(dataset) if dataset else None,
            )
            cal = get_trading_calendar(resolved_market)
            run_window = build_full_run_window(
                requested_start=req_start,
                requested_end=req_end,
                lookback_bars=warmup_calendar_bars,
                trim_output=trim_warmup,
                calendar=cal,
            )
            if (
                run_window.actual_load_start is not None
                and run_window.actual_load_start != req_start
            ):
                narrowed = narrow_data_source_for_window(
                    self.data_source,
                    start_date=run_window.actual_load_start,
                    end_date=run_window.actual_load_end,
                )
                use_fresh = not isinstance(self.cache, PersistentPlanCache)
                engine_to_use = self.with_data_source(narrowed, fresh_cache=use_fresh)
                logger.info(
                    "因子 '%s' auto_warmup: load_start %s → %s（warmup_bars=%d）",
                    factor.name,
                    req_start,
                    run_window.actual_load_start,
                    run_window.warmup_bars,
                )
        elif history_buffer > 0:
            logger.info(
                "因子 '%s' 建议历史缓冲 >= %d bars（lookback=%s）；可设 auto_warmup=True",
                factor.name,
                history_buffer,
                getattr(analysis, "lookback", 0),
            )

        if input_dq_check and analysis.referenced_columns:
            from runtime.input_dq import assert_input_dq

            input_report = assert_input_dq(
                engine_to_use.data_source,
                analysis.referenced_columns,
                raise_on_fail=input_dq_strict,
                thresholds=input_dq_thresholds,
            )
            logger.info("因子 '%s' 输入 DQ 通过", factor.name)
        else:
            input_report = None
        engine_to_use._prefetch_columns(engine_to_use.data_source, analysis.referenced_columns)
        ctx = engine_to_use._make_context()
        result = engine_to_use.backend.execute(plan, ctx)

        if run_window is not None and run_window.trim_output and run_window.requested_start:
            from storage.time_window import slice_series_time_window

            trim_start = pd.Timestamp(run_window.requested_start)
            trim_end = (
                pd.Timestamp(run_window.requested_end) if run_window.requested_end else None
            )
            # 日内源保留 timestamp 精度（不按 normalize 截断）
            if s_bpd > 1:
                trim_start = pd.Timestamp(run_window.requested_start)
            result = slice_series_time_window(
                result,
                start=trim_start,
                end=trim_end,
            )

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
        if run_window is not None:
            out["run_window"] = run_window.to_dict()
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
        dq_thresholds=None,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        input_dq_thresholds=None,
        data_source_config: dict | None = None,
        write_metadata: bool = True,
        resume_materialize: bool = False,
        isolate_partition_failures: bool = True,
        auto_warmup: bool = False,
        trim_warmup: bool = True,
        market: str | None = None,
        preserve_invalid_rows: bool = False,
        value_dtype: str = "float32",
        write_target: str = "local",
        pit_enforce: bool = False,
        pit_forbid_forward_fill: bool = False,
        ch_host: str | None = None,
        ch_port: int | None = None,
        ch_database: str | None = None,
        ch_username: str | None = None,
        ch_password: str | None = None,
        ch_secure: bool | None = None,
        clickhouse_table: str | None = None,
        ch_ensure_table: bool = True,
    ):
        """执行单因子并落盘（Parquet / staging / ClickHouse 或多目标组合）。"""
        target = str(write_target or "local").lower()
        logger.info("开始落盘因子 '%s'，write_target=%s", factor.name, target)
        output = self.run(
            factor,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
            input_dq_thresholds=input_dq_thresholds,
            auto_warmup=auto_warmup,
            trim_warmup=trim_warmup,
            market=market,
            pit_enforce=pit_enforce,
            pit_forbid_forward_fill=pit_forbid_forward_fill,
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
                run_window=output.get("run_window"),
            ),
        )
        materializer = ParquetMaterializer(lake_root=lake_root)
        ast_hash = compute_ir_hash(analysis.ir)
        parquet_target = self._resolve_parquet_write_target(target)

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
            dq_thresholds=dq_thresholds,
            run_lineage={**lineage.to_dict(), "factor_id": factor_id or factor.name},
            write_metadata=write_metadata,
            data_snapshot_id=snapshot_id,
            data_source_config=data_source_config,
            resume=resume_materialize,
            isolate_partition_failures=isolate_partition_failures,
            preserve_invalid_rows=preserve_invalid_rows,
            value_dtype=value_dtype,
            write_target=parquet_target,
            defer_watermark=self._needs_clickhouse_write(target),
        )

        if self._needs_clickhouse_write(target):
            summary = self._dual_write_clickhouse(
                materializer,
                summary,
                factor_id=factor_id or factor.name,
                result=output["result"],
                ast_hash=ast_hash,
                write_target=target,
                data_snapshot_id=snapshot_id,
                clickhouse_table=clickhouse_table,
                preserve_invalid_rows=preserve_invalid_rows,
                value_dtype=value_dtype,
                write_metadata=write_metadata,
                ensure_table=ch_ensure_table,
                ch_host=ch_host,
                ch_port=ch_port,
                ch_database=ch_database,
                ch_username=ch_username,
                ch_password=ch_password,
                ch_secure=ch_secure,
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
        table: str | None = None,
        timestamp_column: str = "trade_date",
        instrument_column: str = "instrument",
        factor_version: str = "",
        author: str | None = None,
        frequency: str | None = None,
        description: str | None = None,
        expression: str | None = None,
        data_source_config: dict | None = None,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        input_dq_thresholds=None,
        auto_warmup: bool = False,
        trim_warmup: bool = True,
        market: str | None = None,
        dq_check: bool = False,
        dq_strict: bool = True,
        dq_thresholds=None,
        preserve_invalid_rows: bool = False,
        value_dtype: str = "float32",
        pit_enforce: bool = False,
        pit_forbid_forward_fill: bool = False,
        ch_host: str | None = None,
        ch_port: int | None = None,
        ch_database: str | None = None,
        ch_username: str | None = None,
        ch_password: str | None = None,
        ch_secure: bool | None = None,
        ensure_table: bool = True,
        lake_root: str | Path | None = None,
    ):
        """执行因子并写入 ClickHouse（catalog + watermark + CH，与 materialize 同构）。"""
        if timestamp_column != "trade_date" or instrument_column != "instrument":
            logger.warning(
                "materialize_clickhouse 列名映射请改用 ClickHouseMaterializer；"
                "当前委托 materialize(write_target=clickhouse)"
            )
        output = self.materialize(
            factor,
            factor_id=factor_id,
            lake_root=lake_root,
            author=author,
            frequency=frequency,
            description=description,
            expression=expression,
            data_source_config=data_source_config,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
            input_dq_thresholds=input_dq_thresholds,
            auto_warmup=auto_warmup,
            trim_warmup=trim_warmup,
            market=market,
            dq_check=dq_check,
            dq_strict=dq_strict,
            dq_thresholds=dq_thresholds,
            preserve_invalid_rows=preserve_invalid_rows,
            value_dtype=value_dtype,
            pit_enforce=pit_enforce,
            pit_forbid_forward_fill=pit_forbid_forward_fill,
            write_target="clickhouse",
            clickhouse_table=table,
            ch_host=ch_host,
            ch_port=ch_port,
            ch_database=ch_database,
            ch_username=ch_username,
            ch_password=ch_password,
            ch_secure=ch_secure,
            ch_ensure_table=ensure_table,
        )
        mat = output.get("materialization") or {}
        ch = mat.get("clickhouse") or {}
        output["clickhouse_materialization"] = {
            "factor_id": ch.get("factor_id") or mat.get("factor_id"),
            "table": ch.get("table"),
            "rows_written": ch.get("rows_written", mat.get("rows_written", 0)),
            "database": ch.get("database"),
            "write_target": "clickhouse",
            "dq_report": mat.get("clickhouse_dq_report") or mat.get("dq_report"),
            "run_id": (mat.get("run_lineage") or {}).get("run_id"),
        }
        return output

    @classmethod
    def materialize_clickhouse_from_config(cls, config_path: str | Path):
        """从 YAML 配置执行因子并写入 ClickHouse。"""
        engine, factor, config = cls.from_config(config_path)
        from runtime.config_runtime import resolve_materialize_kwargs

        opts = resolve_materialize_kwargs(config)
        return engine.materialize_clickhouse(
            factor,
            factor_id=opts.factor_id,
            table=opts.clickhouse_table,
            expression=opts.expression,
            data_source_config=opts.data_source_config,
            auto_warmup=opts.auto_warmup,
            trim_warmup=opts.trim_warmup,
            market=opts.market,
            dq_check=opts.dq_check,
            dq_strict=opts.dq_strict,
            dq_thresholds=opts.dq_thresholds,
            input_dq_check=opts.input_dq_check,
            input_dq_strict=opts.input_dq_strict,
            input_dq_thresholds=opts.input_dq_thresholds,
            preserve_invalid_rows=opts.preserve_invalid_rows,
            value_dtype=opts.value_dtype,
            pit_enforce=opts.pit_enforce,
            pit_forbid_forward_fill=opts.pit_forbid_forward_fill,
            ch_host=opts.clickhouse_host,
            ch_port=opts.clickhouse_port,
            ch_database=opts.clickhouse_database,
            ch_username=opts.clickhouse_username,
            ch_password=opts.clickhouse_password,
            ch_secure=opts.clickhouse_secure,
        )

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
        input_dq_thresholds=None,
        market: str | None = None,
        pit_enforce: bool = False,
        pit_forbid_forward_fill: bool = False,
        auto_warmup: bool = False,
        trim_warmup: bool = True,
    ):
        """增量执行：按 watermark 加载 lookback 窗口，仅输出/落盘 tail 区间。"""
        from cleaned_operators.operator_policy import infer_source_bar_freq
        from runtime.incremental import (
            build_incremental_plan,
            slice_factor_result_for_incremental,
        )
        from storage.materializer import ParquetMaterializer
        from storage.time_window import narrow_data_source_for_window
        from storage.trading_calendar import get_trading_calendar, infer_market

        fid = factor_id or factor.name
        materializer = ParquetMaterializer(lake_root=lake_root)
        watermark = materializer.catalog.get_watermark(fid)

        dataset = getattr(self.data_source, "dataset", None)
        resolved_market = market or infer_market(
            universe=getattr(factor, "universe", None),
            dataset=str(dataset) if dataset else None,
        )
        source_bar_freq = infer_source_bar_freq(
            self.data_source,
            fallback=getattr(factor, "freq", None),
        )
        factor_freq = getattr(factor, "freq", None)

        plan, analysis = self.compile(
            factor,
            pit_enforce=pit_enforce,
            pit_forbid_forward_fill=pit_forbid_forward_fill,
        )
        cal = get_trading_calendar(resolved_market)
        inc = build_incremental_plan(
            factor_id=fid,
            analysis_lookback=getattr(analysis, "lookback", 0),
            watermark=watermark,
            since=since,
            end_date=end_date,
            lookback_extra=lookback_extra,
            recompute_tail_bars=recompute_tail_bars,
            market=resolved_market,
            calendar=cal,
            factor_freq=factor_freq,
            source_bar_freq=source_bar_freq,
        )

        if inc.is_full_run:
            scoped = self
        else:
            narrowed = narrow_data_source_for_window(
                self.data_source,
                start_date=inc.load_start,
                end_date=inc.load_end,
                bar_freq=source_bar_freq,
            )
            # 持久化缓存按 data_scope 隔离；内存缓存需 fresh_cache 避免跨窗口命中
            use_fresh = not isinstance(self.cache, PersistentPlanCache)
            scoped = self.with_data_source(narrowed, fresh_cache=use_fresh)

        logger.info(
            "增量因子 '%s' plan: full=%s load=[%s,%s] output=[%s,%s] lookback=%d freq=%s/%s",
            factor.name,
            inc.is_full_run,
            inc.load_start,
            inc.load_end,
            inc.output_start,
            inc.output_end,
            inc.lookback_bars,
            factor_freq,
            source_bar_freq,
        )

        output = scoped.run(
            factor,
            plan=plan,
            analysis=analysis,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
            input_dq_thresholds=input_dq_thresholds,
            auto_warmup=auto_warmup and inc.is_full_run,
            trim_warmup=trim_warmup,
            market=resolved_market,
            pit_enforce=False,
            pit_forbid_forward_fill=False,
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
        input_dq_thresholds=None,
        data_source_config: dict | None = None,
        write_metadata: bool = True,
        resume_materialize: bool = False,
        isolate_partition_failures: bool = True,
        market: str | None = None,
        preserve_invalid_rows: bool = False,
        value_dtype: str = "float32",
        write_target: str = "local",
        dq_thresholds=None,
        pit_enforce: bool = False,
        pit_forbid_forward_fill: bool = False,
        auto_warmup: bool = False,
        trim_warmup: bool = True,
        clickhouse_table: str | None = None,
        ch_host: str | None = None,
        ch_port: int | None = None,
        ch_database: str | None = None,
        ch_username: str | None = None,
        ch_password: str | None = None,
        ch_secure: bool | None = None,
        ch_ensure_table: bool = True,
    ):
        """增量执行 + 落盘 upsert + lineage（支持 staging_clickhouse / clickhouse 双写）。"""
        from backend.cleaned_bridge import ensure_cleaned_loaded
        from cleaned_operators.operator_policy import compute_operator_catalog_hash
        from runtime.lineage import build_run_lineage
        from storage.catalog import compute_ir_hash

        target = str(write_target or "local").lower()
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
            input_dq_thresholds=input_dq_thresholds,
            market=market,
            pit_enforce=pit_enforce,
            pit_forbid_forward_fill=pit_forbid_forward_fill,
            auto_warmup=auto_warmup,
            trim_warmup=trim_warmup,
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
        ast_hash = compute_ir_hash(analysis.ir)
        parquet_target = self._resolve_parquet_write_target(target)
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
            dq_thresholds=dq_thresholds,
            run_lineage={**lineage.to_dict(), "factor_id": fid},
            write_metadata=write_metadata,
            data_snapshot_id=snapshot_id,
            data_source_config=data_source_config,
            resume=resume_materialize,
            isolate_partition_failures=isolate_partition_failures,
            preserve_invalid_rows=preserve_invalid_rows,
            value_dtype=value_dtype,
            write_target=parquet_target,
            defer_watermark=self._needs_clickhouse_write(target),
        )
        if self._needs_clickhouse_write(target):
            summary = self._dual_write_clickhouse(
                materializer,
                summary,
                factor_id=fid,
                result=output["result"],
                ast_hash=ast_hash,
                write_target=target,
                data_snapshot_id=snapshot_id,
                clickhouse_table=clickhouse_table,
                preserve_invalid_rows=preserve_invalid_rows,
                value_dtype=value_dtype,
                write_metadata=write_metadata,
                ensure_table=ch_ensure_table,
                dq_check=dq_check,
                dq_strict=dq_strict,
                dq_thresholds=dq_thresholds,
                ch_host=ch_host,
                ch_port=ch_port,
                ch_database=ch_database,
                ch_username=ch_username,
                ch_password=ch_password,
                ch_secure=ch_secure,
            )
        output["materialization"] = {
            **summary,
            "lake_root": str(materializer.lake_root),
            "incremental": output.get("incremental"),
        }
        return output
