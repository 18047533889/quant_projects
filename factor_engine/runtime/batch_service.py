"""多因子批跑编排：run_many / run_many_parallel 核心逻辑。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from api.factor import Factor
from ir.analyzer import AnalysisResult
from runtime.perf_config import PerfConfig
from runtime.production_policy import assert_production_run_flags, assert_production_factors

if TYPE_CHECKING:
    from runtime.engine import FactorEngine


def _maybe_prepare_batch_data(
    engine: "FactorEngine",
    dag: Any,
    analyses: dict[str, AnalysisResult],
    *,
    input_dq_check: bool,
    input_dq_strict: bool,
    input_dq_thresholds,
) -> Any | None:
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
    """``FactorEngine.run_many`` 实现体。"""
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
            fb = one.get("production_pandas_fallbacks")
            if fb:
                all_fallbacks.extend(dict(x) for x in fb if isinstance(x, dict))
        dag, _ = engine._dag_from_factors(factors, enable_cse=enable_cse, perf=perf)
        batch_out: dict[str, Any] = {"results": out, "dag": dag, "analyses": analyses}
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
                runtime = dict(getattr(ctx, "runtime_stats", None) or {})
                runtime["polars_long_shared_sid"] = sid
                ctx.runtime_stats = runtime  # type: ignore[attr-defined]
                ctx.shared_result_cache[sid] = engine.backend.execute(sub, ctx)
        out: dict[str, Any] = {}
        for layer in batch_graph.parallel_layers:
            for name in layer:
                fp = root_by_name.get(name)
                if fp is not None:
                    out[name] = engine.backend.execute(fp.root, ctx)
        for fp in dag.roots:
            if fp.factor_name not in out:
                out[fp.factor_name] = engine.backend.execute(fp.root, ctx)
    batch_out: dict[str, Any] = {
        "results": out,
        "dag": dag,
        "analyses": analyses,
    }
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
    from runtime.production_policy import summarize_pandas_fallbacks

    fallbacks = summarize_pandas_fallbacks(ctx)
    if fallbacks:
        batch_out["production_pandas_fallbacks"] = fallbacks
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
    """``FactorEngine.run_many_parallel`` 实现体。"""
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
        res = engine.backend.execute(fp.root, ctx)
        return fp.factor_name, res

    with routing_execution_scope(perf):
        if ctx.shared_result_cache is not None:
            for sid, sub in dag.shared_nodes.items():
                runtime = dict(getattr(ctx, "runtime_stats", None) or {})
                runtime["polars_long_shared_sid"] = sid
                ctx.runtime_stats = runtime  # type: ignore[attr-defined]
                ctx.shared_result_cache[sid] = engine.backend.execute(sub, ctx)

        results: dict[str, Any] = {}
        for layer in batch_graph.parallel_layers:
            fps = [root_by_name[n] for n in layer if n in root_by_name]
            if len(fps) <= 1:
                for fp in fps:
                    results[fp.factor_name] = engine.backend.execute(fp.root, ctx)
            else:
                raw = Parallel(n_jobs=workers, backend="threading")(
                    delayed(_one)(fp) for fp in fps
                )
                results.update(dict(raw))
        for fp in dag.roots:
            if fp.factor_name not in results:
                results[fp.factor_name] = engine.backend.execute(fp.root, ctx)
    parallel_out: dict[str, Any] = {
        "results": results,
        "dag": dag,
        "analyses": analyses,
    }
    if input_report is not None:
        parallel_out["input_dq"] = input_report.to_dict()
    if len(factors) > 1:
        parallel_out["batch_graph"] = batch_graph.to_dict()
    from runtime.production_policy import summarize_pandas_fallbacks

    fallbacks = summarize_pandas_fallbacks(ctx)
    if fallbacks:
        parallel_out["production_pandas_fallbacks"] = fallbacks
    return parallel_out
