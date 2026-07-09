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

    def __init__(self, backend, data_source, cache=None, *, run_mode: str | None = None) -> None:
        bootstrap_runtime_env()
        from runtime.production_policy import resolve_run_mode

        self.backend = backend  # PandasBackend / PolarsBackend / …，由 build_backend 构造
        self.data_source = data_source  # 从 parquet 等拉 MultiIndex 面板的统一入口
        self.cache = cache  # 列级缓存；无则每次 execute 全量算
        self.run_mode = resolve_run_mode(run_mode)
        FactorEngine._sync_production_env(self.run_mode)
        self.analyzer = Analyzer()  # Expr → IR + 依赖列分析
        self.lowerer = Lowerer()  # IR → 逻辑计划树
        self.optimizer = Optimizer()  # 计划级优化（常折叠等）

    @staticmethod
    def _sync_production_env(run_mode: str | None = None) -> None:
        """production 模式同步 data_access 读路径硬策略。"""
        import os

        from runtime.production_policy import PRODUCTION_MODE, is_production_mode

        if is_production_mode(run_mode):
            os.environ["FACTOR_ENGINE_RUN_MODE"] = PRODUCTION_MODE
            os.environ["QUANT_PRODUCTION_MODE"] = "1"
        else:
            os.environ["FACTOR_ENGINE_RUN_MODE"] = "research"

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
        from runtime.production_policy import assert_production_plan_ops

        assert_production_plan_ops(
            optimized_plan,
            mode=self.run_mode,
            context=f"compile:{factor.name}",
        )
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
            new_plans, shared = apply_cse(plans)
            from planner.rolling_cse import apply_rolling_cse

            new_plans, shared = apply_rolling_cse(new_plans, existing_shared=shared)
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

    def analyze_batch(
        self,
        factors: Sequence[Factor],
        *,
        enable_cse: bool | None = None,
        perf: PerfConfig | None = None,
    ) -> dict[str, Any]:
        """编译多因子并返回 DAG + 列依赖图 ``batch_graph``。"""
        from planner.dependency_graph import build_factor_batch_graph

        dag, analyses = self._dag_from_factors(
            factors, enable_cse=enable_cse, perf=perf
        )
        graph = build_factor_batch_graph(factors, analyses)
        return {
            "dag": dag,
            "analyses": analyses,
            "batch_graph": graph.to_dict(),
        }

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
        return cls(
            backend=backend,
            data_source=data_source,
            cache=cache,
            run_mode=config.run.mode,
        ), factor

    @classmethod
    def from_config(cls, config_path: str | Path, *, profile: str | None = None):
        """读 YAML：建 backend、数据源、可选缓存，并解析因子表达式。"""
        logger.info("加载配置文件: %s", config_path)
        config = load_config(config_path, profile=profile)
        engine, factor = cls.from_loaded_config(config)
        return engine, factor, config

    @classmethod
    def run_from_config(cls, config_path: str | Path, *, profile: str | None = None):
        """一键从配置文件跑因子，结果里附带 config 对象。"""
        logger.info("开始从配置执行因子: %s", config_path)
        engine, factor, config = cls.from_config(config_path, profile=profile)
        from runtime.config_runtime import resolve_run_kwargs

        opts = resolve_run_kwargs(config)
        run_kwargs = opts.to_run_kwargs()
        if config.pipeline.batched_engine or config.run.mode == "production":
            batch = engine.run_many([factor], **run_kwargs)
            fp = next(r for r in batch["dag"].roots if r.factor_name == factor.name)
            result = {
                "factor": factor,
                "analysis": batch["analyses"][factor.name],
                "plan": fp.root,
                "result": batch["results"][factor.name],
                "config": config,
            }
            if batch.get("input_dq") is not None:
                result["input_dq"] = batch["input_dq"]
        else:
            result = engine.run(factor, **run_kwargs)
            result["config"] = config
        logger.info("完成从配置执行因子: %s", factor.name)
        return result

    @classmethod
    def _run_factor_with_resolved_kwargs(
        cls,
        engine: FactorEngine,
        factor: Factor,
        config: FactorEngineConfig,
        path: str | Path,
        *,
        pipeline_overrides: Any | None = None,
    ) -> dict[str, Any]:
        from runtime.config_runtime import resolve_run_kwargs_for_pipeline

        opts = resolve_run_kwargs_for_pipeline(config, pipeline_overrides)
        one = engine.run(factor, **opts.to_run_kwargs())
        one["config"] = config
        one["config_path"] = str(path)
        return one

    @classmethod
    def _run_many_batch_from_config_group(
        cls,
        group: list[tuple[FactorEngine, Factor, FactorEngineConfig, str | Path]],
        *,
        enable_cse: bool | None,
        parallel: bool,
        n_jobs: int | None,
        pipeline_overrides: Any | None,
        results: dict[str, Any],
        runs: dict[str, Any],
        configs: dict[str, FactorEngineConfig],
    ) -> None:
        from runtime.config_runtime import config_run_batch_key, resolve_run_kwargs_for_pipeline

        batches: dict[
            tuple[Any, ...],
            list[tuple[FactorEngine, Factor, FactorEngineConfig, str | Path]],
        ] = {}
        for item in group:
            batches.setdefault(
                config_run_batch_key(item[2], pipeline=pipeline_overrides),
                [],
            ).append(item)

        for batch in batches.values():
            if len(batch) == 1:
                engine, factor, config, path = batch[0]
                one = cls._run_factor_with_resolved_kwargs(
                    engine,
                    factor,
                    config,
                    path,
                    pipeline_overrides=pipeline_overrides,
                )
                results[factor.name] = one["result"]
                runs[factor.name] = one
                configs[factor.name] = config
                continue

            base_engine, _, base_config, _ = batch[0]
            opts = resolve_run_kwargs_for_pipeline(base_config, pipeline_overrides)
            run_kwargs = {
                "enable_cse": enable_cse,
                **opts.to_run_kwargs(),
            }
            factors = [f for _, f, _, _ in batch]
            if parallel:
                batch_out = base_engine.run_many_parallel(
                    factors,
                    n_jobs=n_jobs,
                    **run_kwargs,
                )
            else:
                batch_out = base_engine.run_many(factors, **run_kwargs)
            results.update(batch_out["results"])
            for engine, factor, config, path in batch:
                configs[factor.name] = config
                runs[factor.name] = {
                    "factor": factor,
                    "analysis": batch_out["analyses"][factor.name],
                    "result": batch_out["results"][factor.name],
                    "config": config,
                    "config_path": str(path),
                }

    @classmethod
    def run_many_from_config(
        cls,
        config_paths: Sequence[str | Path],
        *,
        enable_cse: bool | None = None,
        parallel: bool = False,
        n_jobs: int | None = None,
        profile: str | None = None,
        pipeline_overrides: Any | None = None,
    ) -> dict[str, Any]:
        """多 YAML 批量 run；按数据源作用域分组，再按 run kwargs 子分组 batch。"""
        from runtime.config_runtime import config_data_scope_key

        loaded: list[tuple[FactorEngine, Factor, FactorEngineConfig, str | Path]] = []
        for path in config_paths:
            engine, factor, config = cls.from_config(path, profile=profile)
            loaded.append((engine, factor, config, path))
        if not loaded:
            return {"results": {}, "runs": {}, "configs": {}}

        groups: dict[str, list[tuple[FactorEngine, Factor, FactorEngineConfig, str | Path]]] = {}
        for item in loaded:
            key = config_data_scope_key(item[2])
            groups.setdefault(key, []).append(item)

        results: dict[str, Any] = {}
        runs: dict[str, Any] = {}
        configs: dict[str, FactorEngineConfig] = {}

        for group in groups.values():
            cls._run_many_batch_from_config_group(
                group,
                enable_cse=enable_cse,
                parallel=parallel,
                n_jobs=n_jobs,
                pipeline_overrides=pipeline_overrides,
                results=results,
                runs=runs,
                configs=configs,
            )

        return {"results": results, "runs": runs, "configs": configs}

    @classmethod
    def run_many_from_config_parallel(
        cls,
        config_paths: Sequence[str | Path],
        *,
        enable_cse: bool | None = None,
        n_jobs: int | None = None,
        profile: str | None = None,
        pipeline_overrides: Any | None = None,
    ) -> dict[str, Any]:
        """多 YAML 批量 run；根节点并行（共享子式仍串行）。"""
        return cls.run_many_from_config(
            config_paths,
            enable_cse=enable_cse,
            parallel=True,
            n_jobs=n_jobs,
            profile=profile,
            pipeline_overrides=pipeline_overrides,
        )

    @classmethod
    def _materialize_one_from_config_item(
        cls,
        engine: FactorEngine,
        factor: Factor,
        config: FactorEngineConfig,
        path: str | Path,
        opts: Any | None = None,
        *,
        pipeline_overrides: Any | None = None,
    ) -> tuple[str, dict[str, Any]]:
        from runtime.config_runtime import resolve_materialize_kwargs_for_pipeline

        if opts is None:
            opts = resolve_materialize_kwargs_for_pipeline(config, pipeline_overrides)
        out = engine.materialize(factor, **opts.to_engine_materialize_kwargs())
        out["config"] = config
        out["config_path"] = str(path)
        return factor.name, out

    @classmethod
    def _materialize_batch_from_config_group(
        cls,
        group: list[tuple[FactorEngine, Factor, FactorEngineConfig, str | Path]],
        outputs: dict[str, Any],
        *,
        batch_run: bool,
        enable_cse: bool | None,
        pipeline_overrides: Any | None,
        parallel: bool,
        n_jobs: int | None,
    ) -> None:
        from runtime.config_runtime import (
            config_materialize_batch_key,
            resolve_materialize_kwargs_for_pipeline,
        )
        from runtime.materialize_service import (
            can_batch_materialize_compute,
            execute_materialize_from_resolved,
        )

        batches: dict[
            tuple[Any, ...],
            list[tuple[FactorEngine, Factor, FactorEngineConfig, str | Path]],
        ] = {}
        for item in group:
            batches.setdefault(
                config_materialize_batch_key(item[2], pipeline=pipeline_overrides),
                [],
            ).append(item)

        for batch in batches.values():
            base_opts = resolve_materialize_kwargs_for_pipeline(batch[0][2], pipeline_overrides)
            use_batch_run = batch_run and len(batch) > 1 and can_batch_materialize_compute(base_opts)
            if use_batch_run:
                engine, _, _, _ = batch[0]
                run_kwargs = {"enable_cse": enable_cse, **base_opts.to_run_kwargs()}
                factors = [f for _, f, _, _ in batch]
                if parallel:
                    run_out = engine.run_many_parallel(
                        factors,
                        n_jobs=n_jobs,
                        **run_kwargs,
                    )
                else:
                    run_out = engine.run_many(factors, **run_kwargs)
                for _, factor, config, path in batch:
                    opts = resolve_materialize_kwargs_for_pipeline(config, pipeline_overrides)
                    output = {
                        "factor": factor,
                        "analysis": run_out["analyses"][factor.name],
                        "result": run_out["results"][factor.name],
                    }
                    out = execute_materialize_from_resolved(engine, factor, output, opts)
                    out["config"] = config
                    out["config_path"] = str(path)
                    out["batched_run"] = True
                    outputs[factor.name] = out
                continue

            for engine, factor, config, path in batch:
                name, out = cls._materialize_one_from_config_item(
                    engine,
                    factor,
                    config,
                    path,
                    pipeline_overrides=pipeline_overrides,
                )
                outputs[name] = out

    @classmethod
    def materialize_many_from_config(
        cls,
        config_paths: Sequence[str | Path],
        *,
        batch_run: bool = True,
        enable_cse: bool | None = None,
        profile: str | None = None,
        pipeline_overrides: Any | None = None,
        parallel: bool = False,
        n_jobs: int | None = None,
    ) -> dict[str, Any]:
        """多 YAML 批量物化；同 scope + 物化参数一致时共享 run_many。"""
        from runtime.config_runtime import config_data_scope_key

        loaded: list[tuple[FactorEngine, Factor, FactorEngineConfig, str | Path]] = []
        for path in config_paths:
            engine, factor, config = cls.from_config(path, profile=profile)
            loaded.append((engine, factor, config, path))
        if not loaded:
            return {"materializations": {}}

        groups: dict[str, list[tuple[FactorEngine, Factor, FactorEngineConfig, str | Path]]] = {}
        for item in loaded:
            key = config_data_scope_key(item[2])
            groups.setdefault(key, []).append(item)

        outputs: dict[str, Any] = {}
        for group in groups.values():
            cls._materialize_batch_from_config_group(
                group,
                outputs,
                batch_run=batch_run,
                enable_cse=enable_cse,
                pipeline_overrides=pipeline_overrides,
                parallel=parallel,
                n_jobs=n_jobs,
            )
        return {"materializations": outputs}

    @classmethod
    def materialize_many_from_config_parallel(
        cls,
        config_paths: Sequence[str | Path],
        *,
        batch_run: bool = True,
        enable_cse: bool | None = None,
        n_jobs: int | None = None,
        profile: str | None = None,
        pipeline_overrides: Any | None = None,
    ) -> dict[str, Any]:
        """多 YAML 批量物化；共享 run_many 时根节点并行。"""
        return cls.materialize_many_from_config(
            config_paths,
            batch_run=batch_run,
            enable_cse=enable_cse,
            profile=profile,
            pipeline_overrides=pipeline_overrides,
            parallel=True,
            n_jobs=n_jobs,
        )

    def materialize_sharded(
        self,
        factors: Sequence[Factor],
        *,
        factor_ids: Sequence[str] | None = None,
        shard_by: str = "factor_id",
        shard_index: int = 0,
        shard_count: int = 1,
        run_many_batch: bool = True,
        **materialize_kwargs: Any,
    ) -> dict[str, Any]:
        """分片物化。

        - ``factor_id``：按因子 ID 哈希选取子集（默认）
        - ``asset_bucket``：按 bucket 编号取模，缩小读范围（全量因子）
        - ``time_month``：按 ``YYYY-MM`` 哈希，缩小时间窗口（全量因子）
        """
        allowed = {"factor_id", "asset_bucket", "time_month"}
        if shard_by not in allowed:
            raise ValueError(f"不支持的 shard_by={shard_by!r}，可选 {sorted(allowed)}")

        ids = list(factor_ids) if factor_ids is not None else [f.name for f in factors]
        if len(ids) != len(factors):
            raise ValueError("factor_ids 长度必须与 factors 一致")

        mat = dict(materialize_kwargs)
        shard_meta: dict[str, Any] = {"shard_by": shard_by}
        engine = self

        if shard_by == "factor_id":
            from runtime.shard_materialize import shard_factor_ids

            selected_ids = set(
                shard_factor_ids(
                    ids,
                    shard_index=shard_index,
                    shard_count=shard_count,
                )
            )
            selected_pairs = [
                (f, fid)
                for f, fid in zip(factors, ids)
                if fid in selected_ids
            ]
            if not selected_pairs:
                return {
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "shard_by": shard_by,
                    "materializations": {},
                    "factor_ids": [],
                }
            sel_factors = [p[0] for p in selected_pairs]
            sel_ids = [p[1] for p in selected_pairs]
        else:
            sel_factors = list(factors)
            sel_ids = list(ids)
            ds = self.data_source
            if shard_by == "asset_bucket":
                from runtime.shard_materialize import shard_bucket_values

                bucket_count = int(mat.pop("bucket_count", 64))
                buckets = shard_bucket_values(
                    bucket_count,
                    shard_index=shard_index,
                    shard_count=shard_count,
                )
                shard_meta["bucket_values"] = buckets
                if buckets and ds is not None and hasattr(ds, "params"):
                    import copy

                    scoped = copy.copy(ds)
                    scoped.params = {**getattr(ds, "params", {}), "bucket_values": buckets}
                    scoped._column_cache = {}
                    scoped._panel_cache = {}
                    engine = self.with_data_source(scoped, fresh_cache=True)
            elif shard_by == "time_month":
                from runtime.shard_materialize import (
                    month_date_bounds,
                    month_keys_between,
                    shard_time_months,
                )

                months = mat.get("time_months")
                if months is None:
                    months = month_keys_between(
                        mat.get("since"),
                        mat.get("end_date"),
                    )
                selected_months = shard_time_months(
                    months,
                    shard_index=shard_index,
                    shard_count=shard_count,
                )
                shard_meta["time_months"] = selected_months
                if not selected_months:
                    return {
                        "shard_index": shard_index,
                        "shard_count": shard_count,
                        "shard_by": shard_by,
                        "materializations": {},
                        "factor_ids": [],
                        **shard_meta,
                    }
                start, end = month_date_bounds(selected_months)
                if ds is not None:
                    import copy

                    scoped = copy.copy(ds)
                    scoped.start_date = start
                    scoped.end_date = end
                    if hasattr(scoped, "_column_cache"):
                        scoped._column_cache = {}
                    if hasattr(scoped, "_panel_cache"):
                        scoped._panel_cache = {}
                    engine = self.with_data_source(scoped, fresh_cache=True)

        outputs: dict[str, Any] = {}
        run_kw = {
            "input_dq_check": mat.pop("input_dq_check", False),
            "input_dq_strict": mat.pop("input_dq_strict", True),
            "input_dq_thresholds": mat.pop("input_dq_thresholds", None),
            "auto_warmup": mat.pop("auto_warmup", False),
            "trim_warmup": mat.pop("trim_warmup", True),
            "market": mat.pop("market", None),
            "pit_enforce": mat.pop("pit_enforce", False),
            "pit_forbid_forward_fill": mat.pop("pit_forbid_forward_fill", False),
        }
        if run_many_batch and len(sel_factors) > 1:
            run_out = engine.run_many(sel_factors, **run_kw)
            from runtime.materialize_service import execute_materialize

            for factor, fid in zip(sel_factors, sel_ids):
                output = {
                    "factor": factor,
                    "analysis": run_out["analyses"][factor.name],
                    "result": run_out["results"][factor.name],
                }
                out = execute_materialize(
                    engine,
                    factor,
                    output,
                    target=str(mat.get("write_target", "local")),
                    lake_root=mat.get("lake_root"),
                    staging_dataset=mat.get("staging_dataset", "factor_lake_staging"),
                    factor_id=fid,
                    author=mat.get("author"),
                    frequency=mat.get("frequency"),
                    description=mat.get("description"),
                    expression=mat.get("expression"),
                    dq_check=mat.get("dq_check", False),
                    dq_strict=mat.get("dq_strict", True),
                    dq_thresholds=mat.get("dq_thresholds"),
                    write_metadata=mat.get("write_metadata", True),
                    data_source_config=mat.get("data_source_config"),
                    resume_materialize=mat.get("resume_materialize", False),
                    isolate_partition_failures=mat.get(
                        "isolate_partition_failures", True
                    ),
                    preserve_invalid_rows=mat.get("preserve_invalid_rows", False),
                    value_dtype=str(mat.get("value_dtype", "float32")),
                    clickhouse_table=mat.get("clickhouse_table"),
                    ch_ensure_table=mat.get("ch_ensure_table", True),
                    ch_host=mat.get("ch_host"),
                    ch_port=mat.get("ch_port"),
                    ch_database=mat.get("ch_database"),
                    ch_username=mat.get("ch_username"),
                    ch_password=mat.get("ch_password"),
                    ch_secure=mat.get("ch_secure"),
                    storage_format=str(mat.get("storage_format", "long")),
                    partition_columns=mat.get("partition_columns"),
                )
                out["batched_run"] = True
                out["shard_index"] = shard_index
                out["shard_count"] = shard_count
                out["shard_by"] = shard_by
                outputs[factor.name] = out.get("materialization", out)
        else:
            for factor, fid in zip(sel_factors, sel_ids):
                mk = {**run_kw, **mat, "factor_id": fid}
                for key in ("time_months", "bucket_count"):
                    mk.pop(key, None)
                out = engine.materialize(factor, **mk)
                out["shard_index"] = shard_index
                out["shard_count"] = shard_count
                out["shard_by"] = shard_by
                outputs[factor.name] = out

        return {
            "shard_index": shard_index,
            "shard_count": shard_count,
            "shard_by": shard_by,
            "factor_ids": sel_ids,
            "materializations": outputs,
            **shard_meta,
        }

    def materialize_matrix(
        self,
        factors: Sequence[Factor],
        *,
        factor_ids: Sequence[str] | None = None,
        universe: str,
        frequency: str = "1d",
        matrix_root: str | Path | None = None,
        partition_columns: list[str] | None = None,
        value_dtype: str = "float32",
        **run_kwargs: Any,
    ) -> dict[str, Any]:
        """批量计算多因子并写入 factor_matrix 宽表（训练/回测加速格式）。"""
        from runtime.matrix_service import execute_materialize_matrix

        return execute_materialize_matrix(
            self,
            factors,
            factor_ids=factor_ids,
            universe=universe,
            frequency=frequency,
            matrix_root=matrix_root,
            partition_columns=partition_columns,
            value_dtype=value_dtype,
            **run_kwargs,
        )

    def plan_incremental_from_event(
        self,
        event: Any,
        *,
        lake_root: str | Path | None = None,
        end_date: str | None = None,
        lookback_extra: int = 5,
        market: str | None = None,
    ) -> dict[str, Any]:
        """数据列更新事件 → 受影响因子增量重算计划。"""
        from runtime.incremental_event_service import plan_incremental_from_event

        return plan_incremental_from_event(
            event,
            lake_root=lake_root,
            end_date=end_date,
            lookback_extra=lookback_extra,
            market=market,
        )

    def materialize_incremental_from_event(
        self,
        event: Any,
        *,
        lake_root: str | Path | None = None,
        end_date: str | None = None,
        lookback_extra: int = 5,
        market: str | None = None,
        dry_run: bool = False,
        **materialize_kwargs: Any,
    ) -> dict[str, Any]:
        """数据列更新事件 → 受影响因子自动增量物化。"""
        from runtime.incremental_event_service import materialize_incremental_from_event

        return materialize_incremental_from_event(
            self,
            event,
            lake_root=lake_root,
            end_date=end_date,
            lookback_extra=lookback_extra,
            market=market,
            dry_run=dry_run,
            **materialize_kwargs,
        )

    @classmethod
    def materialize_incremental_many_from_config(
        cls,
        config_paths: Sequence[str | Path],
        *,
        profile: str | None = None,
        pipeline_overrides: Any | None = None,
    ) -> dict[str, Any]:
        """多 YAML 批量增量物化；逐配置 resolve + materialize_incremental。"""
        from runtime.incremental_event_service import (
            materialize_incremental_many_from_config,
        )

        return materialize_incremental_many_from_config(
            cls,
            config_paths,
            profile=profile,
            pipeline_overrides=pipeline_overrides,
        )

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
        result = engine.materialize(factor, **opts.to_engine_materialize_kwargs())
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
            factor, **opts.to_incremental_materialize_kwargs()
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
        from cache.session import ExecutionCacheSession
        from storage.long_table_source import LongTableDataSource

        prefer_long = isinstance(self.data_source, LongTableDataSource)
        effective_perf = perf or PerfConfig.from_env()
        base = ExecutionContext(
            data_source=self.data_source,
            cache=self.cache,
            shared_result_cache=shared_result_cache,
            panel_cache={},
            materialized_series={},
            prefer_long_table=prefer_long,
            perf=effective_perf,
            query_budget=effective_perf.build_query_budget(),
        )
        if self.cache is None and shared_result_cache is None:
            return base
        session = ExecutionCacheSession(
            plan_cache=self.cache,
            shared_result_cache=shared_result_cache,
            panel_cache={},
        )
        return session.wrap_context(base)

    @staticmethod
    def _resolve_lineage_expression(factor: Factor, expression: str | None) -> str | None:
        from runtime.lineage_service import resolve_lineage_expression

        return resolve_lineage_expression(factor, expression)

    @staticmethod
    def _composite_lineage_from_source(data_source: Any) -> dict[str, Any]:
        from runtime.lineage_service import composite_lineage_from_source

        return composite_lineage_from_source(data_source)

    @staticmethod
    def _lineage_extra(
        *,
        data_source_config: dict | None,
        snapshot_id: str | None,
        input_dq: dict | None = None,
        data_source: Any = None,
        **more: Any,
    ) -> dict[str, Any]:
        from runtime.lineage_service import build_lineage_extra

        return build_lineage_extra(
            data_source_config=data_source_config,
            snapshot_id=snapshot_id,
            input_dq=input_dq,
            data_source=data_source,
            **more,
        )

    @staticmethod
    def _data_snapshot_id_from_config(data_source_config: dict | None) -> str | None:
        from runtime.lineage_service import data_snapshot_id_from_config

        return data_snapshot_id_from_config(data_source_config)

    @staticmethod
    def _resolve_parquet_write_target(write_target: str) -> str:
        from runtime.materialize_service import resolve_parquet_write_target

        return resolve_parquet_write_target(write_target)

    @staticmethod
    def _needs_clickhouse_write(write_target: str) -> bool:
        from runtime.materialize_service import needs_clickhouse_write

        return needs_clickhouse_write(write_target)

    def _append_clickhouse_to_summary(self, summary: dict[str, Any], **kwargs) -> dict[str, Any]:
        from runtime import dual_write_service

        return dual_write_service.append_clickhouse_to_summary(summary, **kwargs)

    def _dual_write_clickhouse(self, materializer, summary: dict[str, Any], **kwargs) -> dict[str, Any]:
        from runtime import dual_write_service

        return dual_write_service.dual_write_clickhouse(materializer, summary, **kwargs)

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

    @staticmethod
    def _prepare_batch_data(
        data_source: Any,
        columns: set[str],
        *,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        input_dq_thresholds=None,
        data_snapshot_id: str | None = None,
    ) -> Any | None:
        """批量 input_dq + 单次 prefetch（run_many 快路径）。"""
        if not columns:
            return None
        from storage.read_session import DataSourceReadSession

        session = DataSourceReadSession(data_source)
        input_report = session.prepare_batch(
            columns,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
            input_dq_thresholds=input_dq_thresholds,
            data_snapshot_id=data_snapshot_id,
        )
        if input_dq_check:
            logger.info("批量输入 DQ 通过，列数=%d", len(columns))
        return input_report

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
        from runtime.production_policy import (
            assert_no_stub_operators,
            assert_production_factors,
            assert_production_run_flags,
        )

        assert_production_factors([factor], mode=self.run_mode, context="run")
        if plan is None or analysis is None:
            plan, analysis = self.compile(
                factor,
                pit_enforce=pit_enforce,
                pit_forbid_forward_fill=pit_forbid_forward_fill,
            )
        assert_production_run_flags(
            mode=self.run_mode,
            input_dq_check=input_dq_check,
            auto_warmup=auto_warmup,
            pit_enforce=pit_enforce,
            context="run",
        )
        assert_no_stub_operators(plan, mode=self.run_mode)
        from runtime.warmup_service import prepare_run_warmup

        warmup = prepare_run_warmup(
            self,
            factor,
            analysis,
            auto_warmup=auto_warmup,
            trim_warmup=trim_warmup,
            market=market,
        )
        run_window = warmup.run_window
        engine_to_use = warmup.engine
        s_bpd = warmup.bars_per_day

        input_report = None
        from planner.sql_io import plan_is_fully_sql

        skip_prefetch = plan_is_fully_sql(plan) and not input_dq_check
        if analysis.referenced_columns and not skip_prefetch:
            from storage.read_session import DataSourceReadSession

            session = DataSourceReadSession(engine_to_use.data_source)
            if input_dq_check:
                input_report = session.prepare_batch(
                    analysis.referenced_columns,
                    input_dq_check=True,
                    input_dq_strict=input_dq_strict,
                    input_dq_thresholds=input_dq_thresholds,
                )
                logger.info("因子 '%s' 输入 DQ 通过", factor.name)
            else:
                session.prefetch(analysis.referenced_columns)
        elif skip_prefetch:
            logger.debug("因子 '%s' fully_sql，跳过列 prefetch", factor.name)
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
        staging_dataset: str = "factor_lake_staging",
        storage_format: str = "long",
        partition_columns: list[str] | None = None,
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
        from runtime.materialize_service import execute_materialize

        return execute_materialize(
            self,
            factor,
            output,
            target=target,
            lake_root=lake_root,
            staging_dataset=staging_dataset,
            factor_id=factor_id,
            author=author,
            frequency=frequency,
            description=description,
            expression=expression,
            dq_check=dq_check,
            dq_strict=dq_strict,
            dq_thresholds=dq_thresholds,
            write_metadata=write_metadata,
            data_source_config=data_source_config,
            resume_materialize=resume_materialize,
            isolate_partition_failures=isolate_partition_failures,
            preserve_invalid_rows=preserve_invalid_rows,
            value_dtype=value_dtype,
            clickhouse_table=clickhouse_table,
            ch_ensure_table=ch_ensure_table,
            ch_host=ch_host,
            ch_port=ch_port,
            ch_database=ch_database,
            ch_username=ch_username,
            ch_password=ch_password,
            ch_secure=ch_secure,
            lineage_mode="full",
            storage_format=storage_format,
            partition_columns=partition_columns,
        )

    def materialize_clickhouse(
        self,
        factor: Factor,
        *,
        factor_id: str | None = None,
        table: str | None = None,
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
        staging_dataset: str = "factor_lake_staging",
        timestamp_column: str | None = None,
        instrument_column: str | None = None,
    ):
        """执行因子并写入 ClickHouse（catalog + watermark + CH，与 materialize 同构）。"""
        if timestamp_column is not None or instrument_column is not None:
            logger.warning(
                "materialize_clickhouse 的 timestamp_column/instrument_column 已废弃；"
                "列映射请配置 ClickHouseMaterializer（默认 trade_date/instrument）。"
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
            staging_dataset=staging_dataset,
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
        auto_warmup: bool = False,
        trim_warmup: bool = True,
        market: str | None = None,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        input_dq_thresholds=None,
        pit_enforce: bool = False,
        pit_forbid_forward_fill: bool = False,
    ) -> dict[str, Any]:
        """多因子求值：先执行 ``DAGPlan.shared_nodes``，再各因子根；含 ``plan_ref`` 时必须用此入口。

        ``auto_warmup`` / ``pit_enforce`` 需按因子逐个 ``run()``（lookback / PIT 编译差异）。
        ``input_dq_check`` 在快路径上合并全量依赖列后批量校验，仍保留 CSE。
        """
        from runtime.batch_service import execute_run_many

        return execute_run_many(
            self,
            factors,
            perf=perf,
            enable_cse=enable_cse,
            auto_warmup=auto_warmup,
            trim_warmup=trim_warmup,
            market=market,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
            input_dq_thresholds=input_dq_thresholds,
            pit_enforce=pit_enforce,
            pit_forbid_forward_fill=pit_forbid_forward_fill,
        )

    def run_many_parallel(
        self,
        factors: Sequence[Factor],
        *,
        n_jobs: int | None = None,
        perf: PerfConfig | None = None,
        enable_cse: bool | None = None,
        auto_warmup: bool = False,
        trim_warmup: bool = True,
        market: str | None = None,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        input_dq_thresholds=None,
        pit_enforce: bool = False,
        pit_forbid_forward_fill: bool = False,
    ) -> dict[str, Any]:
        """在 ``run_many`` 基础上对**各因子根**并行求值（共享子式仍先串行算完）。"""
        from runtime.batch_service import execute_run_many_parallel

        return execute_run_many_parallel(
            self,
            factors,
            n_jobs=n_jobs,
            perf=perf,
            enable_cse=enable_cse,
            auto_warmup=auto_warmup,
            trim_warmup=trim_warmup,
            market=market,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
            input_dq_thresholds=input_dq_thresholds,
            pit_enforce=pit_enforce,
            pit_forbid_forward_fill=pit_forbid_forward_fill,
        )

    def with_data_source(self, data_source, *, fresh_cache: bool = False) -> FactorEngine:
        """返回共享 backend 的新引擎实例（用于增量时间窗口）。

        ``fresh_cache=True`` 时丢弃内存层子计划缓存；磁盘持久化缓存按新
        ``data_scope`` 隔离，仍可命中同窗口历史结果。
        """
        if self.cache is None:
            return FactorEngine(
                backend=self.backend,
                data_source=data_source,
                cache=None,
                run_mode=self.run_mode,
            )
        scope = compute_data_scope(data_source)
        if isinstance(self.cache, PersistentPlanCache):
            new_cache = self.cache.with_scope(scope, clear_memory=fresh_cache)
            return FactorEngine(
                backend=self.backend,
                data_source=data_source,
                cache=new_cache,
                run_mode=self.run_mode,
            )
        if fresh_cache:
            return FactorEngine(
                backend=self.backend,
                data_source=data_source,
                cache=CacheManager(data_scope=scope),
                run_mode=self.run_mode,
            )
        if getattr(self.cache, "data_scope", None) != scope:
            return FactorEngine(
                backend=self.backend,
                data_source=data_source,
                cache=CacheManager(data_scope=scope),
                run_mode=self.run_mode,
            )
        return FactorEngine(
            backend=self.backend,
            data_source=data_source,
            cache=self.cache,
            run_mode=self.run_mode,
        )

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
        staging_dataset: str = "factor_lake_staging",
    ):
        """增量执行 + 落盘 upsert + lineage（支持 staging_clickhouse / clickhouse 双写）。"""
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

        from runtime.materialize_service import execute_materialize

        output = execute_materialize(
            self,
            factor,
            output,
            target=target,
            lake_root=lake_root,
            staging_dataset=staging_dataset,
            factor_id=factor_id,
            author=author,
            frequency=frequency,
            description=description,
            expression=expression,
            dq_check=dq_check,
            dq_strict=dq_strict,
            dq_thresholds=dq_thresholds,
            write_metadata=write_metadata,
            data_source_config=data_source_config,
            resume_materialize=resume_materialize,
            isolate_partition_failures=isolate_partition_failures,
            preserve_invalid_rows=preserve_invalid_rows,
            value_dtype=value_dtype,
            clickhouse_table=clickhouse_table,
            ch_ensure_table=ch_ensure_table,
            ch_host=ch_host,
            ch_port=ch_port,
            ch_database=ch_database,
            ch_username=ch_username,
            ch_password=ch_password,
            ch_secure=ch_secure,
            lineage_mode="incremental",
        )
        output["materialization"]["incremental"] = output.get("incremental")
        return output
