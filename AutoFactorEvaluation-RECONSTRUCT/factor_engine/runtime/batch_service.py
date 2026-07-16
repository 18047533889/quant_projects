"""多因子批跑编排：run_many / run_many_parallel 核心逻辑。

本模块实现 ``FactorEngine.run_many`` 与 ``run_many_parallel`` 的执行体，
负责 CSE 共享子树物化、批量 input_dq、依赖图分层调度及 production 审计。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from api.factor import Factor
from ir.analyzer import AnalysisResult
from runtime.perf_config import PerfConfig
from runtime.production_policy import (
    assert_production_fastpath_runtime,
    assert_production_factors,
    assert_production_run_flags,
    summarize_pandas_fallbacks,
)

if TYPE_CHECKING:
    from runtime.engine import FactorEngine


def _materialize_shared_subplan(
    backend: Any,
    sub: Any,
    ctx: Any,
    sid: str,
) -> None:
    """CSE 共享子树：优先 lazy-only 编译，否则 eager execute 写入 ``shared_result_cache``。"""
    if (
        getattr(sub, "op", None) == "literal"
        and getattr(backend, "supports_lazy_shared", False)
    ):
        return
    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    runtime["polars_long_shared_sid"] = sid
    ctx.runtime_stats = runtime  # type: ignore[attr-defined]
    if getattr(backend, "supports_lazy_shared", False):
        compile_lazy = getattr(backend, "compile_lazy_shared", None)
        if callable(compile_lazy) and compile_lazy(sub, ctx, sid=str(sid)):
            return
    if ctx.shared_result_cache is not None:
        ctx.shared_result_cache[sid] = backend.execute(sub, ctx)


def _clear_polars_long_shared_sid(ctx: Any) -> None:
    """清除执行上下文中 Polars long 共享子树的 sid 标记。"""
    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    runtime.pop("polars_long_shared_sid", None)
    ctx.runtime_stats = runtime  # type: ignore[attr-defined]


def _execute_root_with_path(
    backend: Any,
    plan: Any,
    ctx: Any,
    *,
    run_mode: str | None = None,
    factor_name: str = "",
) -> tuple[Any, dict[str, Any]]:
    """执行单个因子根计划并快照 backend 路径摘要。

    为每个因子根创建独立 ``runtime_stats`` 副本，避免路径统计互相污染；
    production 模式下校验 fast path runtime 合规性。

    Returns:
        ``(result, backend_path_dict)`` 元组。
    """
    from dataclasses import replace

    from backend.path_summary import snapshot_backend_path
    from backend.polars_long_backend import _fresh_long_runtime

    parent_runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    local_runtime = _fresh_long_runtime(parent_runtime)
    for key in (
        "events",
        "latest",
        "backend",
        "shared_long_lazy_hits",
        "shared_long_lazy_op_stats",
        "shared_lazy_compile_failures",
        "shared_lazy_compile_failed_sid",
        "shared_lazy_compile_failed_op",
        "shared_lazy_compile_error_type",
    ):
        if key in parent_runtime:
            local_runtime[key] = parent_runtime[key]
    local_ctx = replace(ctx, runtime_stats=local_runtime)
    result = backend.execute(plan, local_ctx)
    path = snapshot_backend_path(getattr(local_ctx, "runtime_stats", None))
    audit_ctx = f"run_many:{factor_name}" if factor_name else "run_many"
    assert_production_fastpath_runtime(local_ctx, mode=run_mode, context=audit_ctx)
    return result, path


def _attach_batch_backend_paths(batch_out: dict[str, Any], paths: dict[str, dict[str, Any]]) -> None:
    """将各因子的 backend 路径写入批跑输出并生成汇总。"""
    if not paths:
        return
    from backend.path_summary import summarize_batch_backend_paths

    batch_out["backend_paths"] = paths
    batch_out["backend_path_summary"] = summarize_batch_backend_paths(paths)


def _maybe_prepare_batch_data(
    engine: "FactorEngine",
    dag: Any,
    analyses: dict[str, AnalysisResult],
    *,
    input_dq_check: bool,
    input_dq_strict: bool,
    input_dq_thresholds,
) -> Any | None:
    """合并全量依赖列后批量 prefetch 与 input_dq（run_many 快路径）。

    若计划可走 fully_sql / native_scan 则跳过 prefetch。
    """
    all_cols: set[str] = set()
    for analysis in analyses.values():
        all_cols |= analysis.referenced_columns
    all_plans = [fp.root for fp in dag.roots] + list(dag.shared_nodes.values())
    from planner.sql_io import should_skip_column_prefetch

    if all_cols and not should_skip_column_prefetch(
        all_plans,
        input_dq_check=input_dq_check,
        backend=engine.backend,
    ):
        return engine._prepare_batch_data(
            engine.data_source,
            all_cols,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
            input_dq_thresholds=input_dq_thresholds,
        )
    return None


def execute_run_many(
    engine: "FactorEngine",
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
    """``FactorEngine.run_many`` 实现体：多因子 DAG 串行求值。

    流程：编译多因子 DAG（可选 CSE）→ 批量 prefetch/input_dq →
    物化 ``shared_nodes`` → 按依赖图分层执行各因子根。

    ``auto_warmup`` 或 ``pit_enforce`` 为真时退化为逐因子 ``run()``，
    因 lookback / PIT 编译差异无法共享快路径。

    Args:
        engine: 因子引擎实例。
        factors: 待求值因子序列。
        perf: 性能配置；缺省从环境变量加载。
        enable_cse: 是否启用公共子表达式消除。
        auto_warmup: 是否自动扩展 warmup 窗口。
        trim_warmup: warmup 后是否裁剪回用户请求区间。
        market: 市场标识（warmup 日历）。
        input_dq_check: 是否校验输入列质量。
        input_dq_strict: 输入 DQ 失败是否中断。
        input_dq_thresholds: 输入 DQ 阈值。
        pit_enforce: 是否强制 PIT 安全审计。
        pit_forbid_forward_fill: PIT 审计是否禁止前向填充。

    Returns:
        含 ``results``、``dag``、``analyses`` 及可选 ``batch_graph``、
        ``input_dq``、``backend_paths`` 等的字典。
    """
    assert_production_run_flags(
        mode=engine.run_mode,
        input_dq_check=input_dq_check,
        auto_warmup=auto_warmup,
        pit_enforce=pit_enforce,
        context="run_many",
    )
    assert_production_factors(factors, mode=engine.run_mode, context="run_many")

    use_per_factor_run = auto_warmup or pit_enforce
    if use_per_factor_run:
        out: dict[str, Any] = {}
        analyses: dict[str, AnalysisResult] = {}
        all_fallbacks: list[dict[str, str]] = []
        backend_paths: dict[str, dict[str, Any]] = {}
        from backend.path_summary import snapshot_from_run_output

        for factor in factors:
            one = engine.run(
                factor,
                auto_warmup=auto_warmup,
                trim_warmup=trim_warmup,
                market=market,
                input_dq_check=input_dq_check,
                input_dq_strict=input_dq_strict,
                input_dq_thresholds=input_dq_thresholds,
                pit_enforce=pit_enforce,
                pit_forbid_forward_fill=pit_forbid_forward_fill,
            )
            out[factor.name] = one["result"]
            analyses[factor.name] = one["analysis"]
            backend_paths[factor.name] = snapshot_from_run_output(one)
            fb = one.get("production_pandas_fallbacks")
            if fb:
                all_fallbacks.extend(dict(x) for x in fb if isinstance(x, dict))
        dag, _ = engine._dag_from_factors(factors, enable_cse=enable_cse, perf=perf)
        batch_out: dict[str, Any] = {"results": out, "dag": dag, "analyses": analyses}
        _attach_batch_backend_paths(batch_out, backend_paths)
        if all_fallbacks:
            batch_out["production_pandas_fallbacks"] = all_fallbacks
        return batch_out

    dag, analyses = engine._dag_from_factors(
        factors, enable_cse=enable_cse, perf=perf
    )
    perf = perf or PerfConfig.from_env()
    input_report = _maybe_prepare_batch_data(
        engine,
        dag,
        analyses,
        input_dq_check=input_dq_check,
        input_dq_strict=input_dq_strict,
        input_dq_thresholds=input_dq_thresholds,
    )
    ctx = engine._make_context(shared_result_cache={}, perf=perf)
    from backend.routing_env import routing_execution_scope
    from planner.dependency_graph import build_factor_batch_graph

    batch_graph = build_factor_batch_graph(factors, analyses)
    root_by_name = {fp.factor_name: fp for fp in dag.roots}
    with routing_execution_scope(perf):
        if ctx.shared_result_cache is not None:
            for sid, sub in dag.shared_nodes.items():
                _materialize_shared_subplan(engine.backend, sub, ctx, sid)
            _clear_polars_long_shared_sid(ctx)
        out: dict[str, Any] = {}
        backend_paths: dict[str, dict[str, Any]] = {}
        for layer in batch_graph.parallel_layers:
            for name in layer:
                fp = root_by_name.get(name)
                if fp is not None:
                    result, path = _execute_root_with_path(
                        engine.backend, fp.root, ctx, run_mode=engine.run_mode, factor_name=name
                    )
                    out[name] = result
                    backend_paths[name] = path
        for fp in dag.roots:
            if fp.factor_name not in out:
                result, path = _execute_root_with_path(
                    engine.backend, fp.root, ctx, run_mode=engine.run_mode, factor_name=fp.factor_name
                )
                out[fp.factor_name] = result
                backend_paths[fp.factor_name] = path
    batch_out: dict[str, Any] = {
        "results": out,
        "dag": dag,
        "analyses": analyses,
    }
    _attach_batch_backend_paths(batch_out, backend_paths)
    if input_report is not None:
        batch_out["input_dq"] = input_report.to_dict()
    if len(factors) > 1:
        batch_out["batch_graph"] = batch_graph.to_dict()
    if dag.shared_nodes:
        from planner.rolling_cache import summarize_rolling_cache

        batch_out["rolling_cache"] = summarize_rolling_cache(dag.shared_nodes)
    if dag.roots:
        from backend.operator_cost import estimate_plan_cost

        batch_out["plan_costs"] = {
            fp.factor_name: estimate_plan_cost(fp.root) for fp in dag.roots
        }
        from planner.cost_summary import summarize_plans

        batch_out["cost_summary"] = summarize_plans(
            {fp.factor_name: fp.root for fp in dag.roots}
        )
        from planner.scheduling_hints import derive_scheduling_hints

        batch_out["scheduling_hints"] = derive_scheduling_hints(
            batch_out["cost_summary"]
        )
    assert_production_fastpath_runtime(ctx, mode=engine.run_mode, context="run_many")
    fallbacks = summarize_pandas_fallbacks(ctx)
    if fallbacks:
        batch_out["production_pandas_fallbacks"] = fallbacks
    from backend.path_summary import summarize_lazy_caches

    lazy_cache = summarize_lazy_caches(ctx)
    if lazy_cache:
        batch_out["lazy_cache_summary"] = lazy_cache
    return batch_out


def execute_run_many_parallel(
    engine: "FactorEngine",
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
    """``FactorEngine.run_many_parallel`` 实现体：根节点层内并行。

    共享子树仍串行物化；同一依赖层内的因子根通过 joblib 线程池并行。
    ``auto_warmup`` / ``pit_enforce`` 为真时退化为 ``execute_run_many``。

    Args:
        engine: 因子引擎实例。
        factors: 待求值因子序列。
        n_jobs: 并行 worker 数；缺省取 ``perf.max_workers``。
        其余参数同 ``execute_run_many``。

    Returns:
        与 ``execute_run_many`` 结构相同的批跑结果字典。

    Raises:
        ImportError: 未安装 joblib。
    """
    assert_production_run_flags(
        mode=engine.run_mode,
        input_dq_check=input_dq_check,
        auto_warmup=auto_warmup,
        pit_enforce=pit_enforce,
        context="run_many_parallel",
    )
    assert_production_factors(factors, mode=engine.run_mode, context="run_many_parallel")

    if auto_warmup or pit_enforce:
        return execute_run_many(
            engine,
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

    try:
        from joblib import Parallel, delayed
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "run_many_parallel 需要 joblib：pip install 'factor-engine[parallel]'"
        ) from exc

    dag, analyses = engine._dag_from_factors(
        factors, enable_cse=enable_cse, perf=perf
    )
    perf = perf or PerfConfig.from_env()
    workers = n_jobs if n_jobs is not None else perf.max_workers
    input_report = _maybe_prepare_batch_data(
        engine,
        dag,
        analyses,
        input_dq_check=input_dq_check,
        input_dq_strict=input_dq_strict,
        input_dq_thresholds=input_dq_thresholds,
    )
    ctx = engine._make_context(shared_result_cache={}, perf=perf)
    from backend.routing_env import routing_execution_scope
    from planner.dependency_graph import build_factor_batch_graph

    batch_graph = build_factor_batch_graph(factors, analyses)
    root_by_name = {fp.factor_name: fp for fp in dag.roots}

    def _one(fp):
        result, path = _execute_root_with_path(
            engine.backend, fp.root, ctx, run_mode=engine.run_mode, factor_name=fp.factor_name
        )
        return fp.factor_name, result, path

    with routing_execution_scope(perf):
        if ctx.shared_result_cache is not None:
            for sid, sub in dag.shared_nodes.items():
                _materialize_shared_subplan(engine.backend, sub, ctx, sid)
            _clear_polars_long_shared_sid(ctx)

        results: dict[str, Any] = {}
        backend_paths: dict[str, dict[str, Any]] = {}
        for layer in batch_graph.parallel_layers:
            fps = [root_by_name[n] for n in layer if n in root_by_name]
            if len(fps) <= 1:
                for fp in fps:
                    result, path = _execute_root_with_path(
                        engine.backend, fp.root, ctx, run_mode=engine.run_mode, factor_name=fp.factor_name
                    )
                    results[fp.factor_name] = result
                    backend_paths[fp.factor_name] = path
            else:
                raw = Parallel(n_jobs=workers, backend="threading")(
                    delayed(_one)(fp) for fp in fps
                )
                for name, result, path in raw:
                    results[name] = result
                    backend_paths[name] = path
        for fp in dag.roots:
            if fp.factor_name not in results:
                result, path = _execute_root_with_path(
                    engine.backend, fp.root, ctx, run_mode=engine.run_mode, factor_name=fp.factor_name
                )
                results[fp.factor_name] = result
                backend_paths[fp.factor_name] = path
    parallel_out: dict[str, Any] = {
        "results": results,
        "dag": dag,
        "analyses": analyses,
    }
    _attach_batch_backend_paths(parallel_out, backend_paths)
    if input_report is not None:
        parallel_out["input_dq"] = input_report.to_dict()
    if len(factors) > 1:
        parallel_out["batch_graph"] = batch_graph.to_dict()
    assert_production_fastpath_runtime(ctx, mode=engine.run_mode, context="run_many_parallel")
    fallbacks = summarize_pandas_fallbacks(ctx)
    if fallbacks:
        parallel_out["production_pandas_fallbacks"] = fallbacks
    from backend.path_summary import summarize_lazy_caches

    lazy_cache = summarize_lazy_caches(ctx)
    if lazy_cache:
        parallel_out["lazy_cache_summary"] = lazy_cache
    return parallel_out
