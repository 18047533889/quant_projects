"""多因子批跑编排：run_many / run_many_parallel 核心逻辑。

本模块实现 ``FactorEngine.run_many`` 与 ``run_many_parallel`` 的执行体，
负责 CSE 共享子树物化、批量 input_dq、依赖图分层调度及 production 审计。
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Sequence

from api.factor import Factor
from ir.analyzer import AnalysisResult
from logging_utils import get_logger
from runtime.perf_config import PerfConfig
from runtime.production_policy import (
    assert_production_fastpath_runtime,
    assert_production_factors,
    assert_production_run_flags,
    summarize_pandas_fallbacks,
)

logger = get_logger("runtime.batch_service")

if TYPE_CHECKING:
    from runtime.engine import FactorEngine


def materialize_shared_nodes_parallel(
    dag: Any,
    backend: Any,
    ctx: Any,
    *,
    max_workers: int | None = None,
) -> None:
    """R27-165/231：独立 shared nodes **按依赖并行**物化（不再串行循环）。

    两个彼此完全独立的 shared nodes 现在并行；R27-002。共享子树物化本身
    ``_materialize_shared_subplan`` 线程安全（每 sid 独立写入
    ``ctx.shared_result_cache``）。有资源约束时 ``max_workers`` 限制并发。

    R27-006：本函数保留「一次性物化全部 shared」的入口（老路径兼容），真正的
    DAG-ready 调度（某 shared predecessor ready 就立刻运行）由
    :mod:`runtime.adaptive_batch_scheduler` 实现。
    """
    shared = getattr(dag, "shared_nodes", None) or {}
    if isinstance(shared, dict):
        items = list(shared.items())
    else:
        items = list(shared)
    if not items:
        return
    if len(items) == 1:
        sid, sub = items[0]
        _materialize_shared_subplan(backend, sub, ctx, sid)
        return
    try:
        from concurrent.futures import ThreadPoolExecutor, wait

        def _one(pair):
            sid, sub = pair
            _materialize_shared_subplan(backend, sub, ctx, sid)

        workers = max_workers or min(4, len(items))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="r27-shared") as pool:
            futures = [pool.submit(_one, pair) for pair in items]
            done, _ = wait(futures)
            for f in done:
                f.result()  # 失败如实冒泡（共享物化失败不静默）
    except ImportError:
        for sid, sub in items:
            _materialize_shared_subplan(backend, sub, ctx, sid)


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
        value = backend.execute(sub, ctx)
        ctx.shared_result_cache[sid] = value


def _release_consumed_sids(ctx: Any, root: Any) -> None:
    """Phase 5 R5：root 执行完，对其消费的共享 sid 引用计数减一，归零立即释放。

    共享子树默认只在 batch 结束时随 ctx 一起释放；这里让大 panel CSE 在最后一个
    消费者完成时立刻逐出，而不是常驻整个 batch。
    """
    if ctx.shared_result_cache is None:
        return
    from cache.session import ExecutionCacheSession
    from planner.cse import collect_consumed_sids

    consumed = collect_consumed_sids(root)
    if not consumed:
        return
    refcounts = getattr(ctx, "_cse_refcounts", None)
    if refcounts is None:
        return
    cache = getattr(ctx, "expression_cache", None)
    for sid in consumed:
        remaining = refcounts.get(sid, 1) - 1
        refcounts[sid] = remaining
        if remaining <= 0:
            if cache is not None:
                cache.release(sid)
            else:
                ctx.shared_result_cache.pop(sid, None)


def _setup_cse_refcounts(ctx: Any, roots: list[Any]) -> None:
    """初始化 CSE 引用计数表（挂在 ctx 上）。

    R13 NEW-P1-76: refcount-init failure is NEVER silent.  When a batch shares
    subplans via CSE, ``_release_root_cse`` relies on ``ctx._cse_refcounts`` to
    evict shared panels as consumers finish; if we cannot install the table the
    shared panels would never be released and a large batch can drift into OOM.
    Failing loudly beats silently continuing with an unreleased shared panel.
    """
    from planner.cse import cse_consumer_counts

    roots_plans = [getattr(fp, "root", fp) for fp in roots]
    counts = cse_consumer_counts(roots_plans)
    if counts:
        try:
            ctx._cse_refcounts = counts  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 — attribute install failure is a run-time fault
            raise RuntimeError(
                "CSE memory lifecycle: failed to install the shared-subplan "
                f"refcount table on the execution context ({type(exc).__name__}: "
                f"{exc}) — shared panels would never be released; refusing to run "
                "with CSE memory accounting disabled (R13 NEW-P1-76)"
            ) from exc


def _clear_polars_long_shared_sid(ctx: Any) -> None:
    """清除执行上下文中 Polars long 共享子树的 sid 标记。"""
    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    runtime.pop("polars_long_shared_sid", None)
    ctx.runtime_stats = runtime  # type: ignore[attr-defined]


def validate_cse_refcount_integrity(dag: Any, ctx: Any) -> dict[str, Any]:
    """R20-253..319: 共享 CSE DAG 引用完整性审计。

    校验：
    - ``every plan_ref sid exists``：每个 root 消费的 plan_ref sid 都存在于
      ``dag.shared_nodes``（无悬空引用）；
    - ``consumer count 准确``：``ctx._cse_refcounts`` 与
      ``planner.cse.cse_consumer_counts`` 一致（无虚高/虚低）；
    - ``no orphan``：``dag.shared_nodes`` 里没有任何 root 消费的 sid（孤儿共享
      子树不会被释放 → 内存泄漏）。

    返回 ``{"ok": bool, "issues": list[str]}``。生产模式下孤儿/悬空引用抛错
    （fail closed）。
    """
    from planner.cse import collect_consumed_sids, cse_consumer_counts

    issues: list[str] = []
    shared_nodes = getattr(dag, "shared_nodes", None) or {}
    roots = getattr(dag, "roots", None) or []
    roots_plans = [getattr(fp, "root", fp) for fp in roots]
    consumed: set[str] = set()
    for root in roots_plans:
        consumed |= set(collect_consumed_sids(root))
    for sid in sorted(consumed):
        if sid not in shared_nodes:
            issues.append(f"plan_ref sid={sid!r} has no matching shared node (dangling)")
    for sid in sorted(shared_nodes):
        if sid not in consumed:
            issues.append(f"shared sid={sid!r} is orphaned (no root consumes it)")
    counts = cse_consumer_counts(roots_plans)
    refcounts = getattr(ctx, "_cse_refcounts", None) or {}
    for sid, expected in counts.items():
        actual = refcounts.get(sid, 0)
        if actual != expected:
            issues.append(
                f"sid={sid!r} consumer count mismatch expected={expected} actual={actual}"
            )
    report = {"ok": not issues, "issues": issues, "n_shared": len(shared_nodes), "n_consumed": len(consumed)}
    if issues:
        try:
            from runtime.production_policy import is_production_mode

            mode = getattr(ctx, "run_mode", None)
            if is_production_mode(mode):
                raise RuntimeError(
                    "CSE refcount integrity violated (R20-253..319):\n  "
                    + "\n  ".join(issues)
                )
        except RuntimeError:
            raise
        except Exception:  # noqa: BLE001
            pass
    return report


#: R20-241..252 worker-isolation contract for objects shared by parallel roots.
#:
#: - ``backend`` / ``engine.data_source`` / ``ctx``: IMMUTABLE after the shared
#:   subplans are materialized.  Workers only *read* these (never mutate the
#:   backend instance or the data source); per-root mutable state lives on the
#:   ``local_ctx`` copy.
#: - ``ctx.shared_result_cache`` / ``shared_long_lazy_cache`` /
#:   ``materialized_long_lazy`` / ``materialized_series`` / ``panel_cache``:
#:   THREAD-LOCAL mutable views — each root gets an independent shallow copy so
#:   parallel workers cannot race on panel/materialization dictionaries.
#: - ``ctx.runtime_stats``: THREAD-LOCAL; a fresh dict per root so backend route
#:   telemetry (``plan_backend_route`` / ``polars_long_fallback_reason`` /
#:   ``last_operator_backend_route``) is root-local and never overwrites another
#:   factor's telemetry (R20-241).
#: - process environment / registries / global governor: PROCESS-LOCAL
#:   (shared read-only); any per-run override must be scoped inside
#:   ``routing_execution_scope`` / ``ExecutionResourceScope``.
_ROOT_LOCAL_TELEMETRY_KEYS = (
    "plan_backend_route",
    "polars_long_shared_sid",
    "used_polars_long_path",
    "used_polars_long_native",
    "polars_long_fallback_reason",
    "polars_long_fallback_plan_op",
    "polars_long_fallback_exception_type",
    "polars_long_last_native",
)


def _assert_no_native_certified_fallback(
    plan: Any,
    local_ctx: Any,
    *,
    run_mode: str | None,
    factor_name: str,
) -> None:
    """R20-241..252 invariant NATIVE_CERTIFIED_BUT_FALLBACK_EXECUTED == 0.

    When EVERY operator in a root plan is dual-backend production certified, a
    fallback to the pandas path is a semantic-preservation violation — the
    native path was certified yet the backend executed a fallback.  Production
    fails closed; research records the event for telemetry.
    """
    runtime = getattr(local_ctx, "runtime_stats", None) or {}
    if runtime.get("polars_long_fallback_reason") is None:
        return
    try:
        from planner.composite_lowering import collect_plan_ops

        ops = set(collect_plan_ops(plan))
    except Exception:  # noqa: BLE001 - unanalyzable plan: skip invariant
        return
    if not ops:
        return
    try:
        from backend.primitive_evidence import PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE

        native_safe = PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
    except Exception:  # noqa: BLE001
        return
    if ops.issubset(native_safe):
        from runtime.production_policy import is_production_mode

        if is_production_mode(run_mode):
            raise RuntimeError(
                f"NATIVE_CERTIFIED_BUT_FALLBACK_EXECUTED != 0: factor={factor_name!r} "
                f"plan ops {sorted(ops)} are all dual-backend certified, yet the "
                "runtime executed a polars_long fallback "
                f"reason={runtime.get('polars_long_fallback_reason')!r} "
                f"op={runtime.get('polars_long_fallback_plan_op')!r}.  Native-certified "
                "plans must never fall back in production (R20-241..252)."
            )


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
    production 模式下校验 fast path runtime 合规性 + NATIVE_CERTIFIED_BUT_
    FALLBACK_EXECUTED == 0 不变量。

    Returns:
        ``(result, backend_path_dict)`` 元组。
    """
    from dataclasses import replace

    from backend.path_summary import snapshot_backend_path
    from backend.polars_long_backend import _fresh_long_runtime

    parent_runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    local_runtime = _fresh_long_runtime(parent_runtime)
    # R20-241: shared-subplan materialization telemetry must NOT leak into a
    # root's path summary — drop the shared/route keys before execution so the
    # root-local backend route is authoritative for THIS factor.
    for key in _ROOT_LOCAL_TELEMETRY_KEYS:
        local_runtime.pop(key, None)
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
    local_ctx = replace(
        ctx,
        runtime_stats=local_runtime,
        # Shared subplans are fully materialized before workers start. Give
        # each root an independent mutable view so backend execution cannot
        # race on panel/materialization dictionaries.
        shared_result_cache=dict(ctx.shared_result_cache or {})
        if ctx.shared_result_cache is not None
        else None,
        shared_long_lazy_cache=dict(ctx.shared_long_lazy_cache or {})
        if ctx.shared_long_lazy_cache is not None
        else None,
        materialized_long_lazy=dict(ctx.materialized_long_lazy or {})
        if ctx.materialized_long_lazy is not None
        else None,
        materialized_series=dict(ctx.materialized_series or {})
        if ctx.materialized_series is not None
        else None,
        panel_cache={},
    )
    result = backend.execute(plan, local_ctx)
    path = snapshot_backend_path(getattr(local_ctx, "runtime_stats", None))
    audit_ctx = f"run_many:{factor_name}" if factor_name else "run_many"
    assert_production_fastpath_runtime(local_ctx, mode=run_mode, context=audit_ctx)
    _assert_no_native_certified_fallback(
        plan, local_ctx, run_mode=run_mode, factor_name=factor_name
    )
    return result, path


def _cluster_factors_by_cost(
    engine: "FactorEngine",
    factors: Sequence[Factor],
    analyses: dict[str, AnalysisResult],
) -> list[list[Factor]]:
    """Phase 5 P1-4：按成本把因子聚成 waves。

    - ``full_history``：需要全历史（lookback 覆盖全窗）
    - ``long``：lookback 覆盖 2/3 窗
    - ``medium``：lookback 覆盖 1/3 窗
    - ``short``：其余

    避免一个 full-history 因子把整批 union 窗口拉成全历史（拖累 999 个短因子）。
    """
    from runtime.run_window import extract_source_date_bounds
    from runtime.warmup_service import prepare_run_warmup

    clusters: list[list[Factor]] = []
    buckets: dict[str, list[Factor]] = {"full_history": [], "long": [], "medium": [], "short": []}
    total = 252
    try:
        start, end = extract_source_date_bounds(engine.data_source)
        if start and end:
            import pandas as pd

            total = max(1, len(pd.bdate_range(start, end)))
    except Exception:
        pass
    for factor in factors:
        analysis = analyses.get(factor.name)
        rw = None
        if analysis is not None:
            try:
                wctx = prepare_run_warmup(
                    engine,
                    factor,
                    analysis,
                    auto_warmup=True,
                    trim_warmup=True,
                    market=None,
                )
                rw = wctx.run_window
            except Exception:
                # R20-234..240: warmup/history resolver 异常在 production 下不能
                # 默认为 short/no-warmup——unresolved history requirement 会静默
                # 让全历史因子退化成短窗，改变其 semantic history anchor。production
                # fail closed（抛错），research 保留 fallback 聚类行为。
                from runtime.production_policy import is_production_mode

                if is_production_mode(engine.run_mode):
                    raise
                rw = None
        if rw is not None and rw.full_history_required:
            buckets["full_history"].append(factor)
            continue
        lookback = int(getattr(analysis, "lookback", None) or 0) if analysis else 0
        ratio = lookback / total if total > 0 else 0.0
        if ratio >= 0.9:
            buckets["long"].append(factor)
        elif ratio >= 0.5:
            buckets["medium"].append(factor)
        else:
            buckets["short"].append(factor)
    for name in ("short", "medium", "long", "full_history"):
        if buckets[name]:
            clusters.append(buckets[name])
    return clusters


def _handle_result(
    policy: str,
    sink: Any,
    out: dict[str, Any],
    name: str,
    result: Any,
    path: dict[str, Any],
    backend_paths: dict[str, dict[str, Any]],
) -> None:
    """按 result_policy 处理单个因子结果（R4）。

    - ``return``：全部结果驻留 ``out``（旧行为）
    - ``sink``：立即交给 ``sink(name, result)``（通常落盘/DQ），**不**驻留
    - ``yield``/``materialize``：与 ``return`` 相同累积（生成器形态由 run_many_iter 提供）
    """
    if policy == "sink" and sink is not None:
        sink(name, result)
        backend_paths[name] = path
        return
    out[name] = result
    backend_paths[name] = path


def _attach_batch_backend_paths(batch_out: dict[str, Any], paths: dict[str, dict[str, Any]]) -> None:
    """将各因子的 backend 路径写入批跑输出并生成汇总。"""
    if not paths:
        return
    from backend.path_summary import summarize_batch_backend_paths

    batch_out["backend_paths"] = paths
    batch_out["backend_path_summary"] = summarize_batch_backend_paths(paths)


def _transition_telemetry(paths: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """R31-104/103：backend transition + conversion 观测（第一等 KPI）。

    每个因子根的 ``primary_route`` 变化计一次 transition；conversion bytes 用
    批跑结果面板的近似（rows × 8B），真实 conversion 由后端路径标记（SQL→client /
    polars→pandas 等）标注。
    """
    routes: list[str] = []
    for path in paths.values():
        route = path.get("primary_route") or path.get("backend") or ""
        if route:
            routes.append(str(route))
    transitions = 0
    for i in range(1, len(routes)):
        if routes[i] != routes[i - 1]:
            transitions += 1
    sql_paths = sum(1 for p in paths.values() if p.get("used_sql_pushdown"))
    polars_native = sum(1 for p in paths.values() if p.get("used_polars_long_native"))
    fallback_paths = sum(1 for p in paths.values() if p.get("polars_long_fallback_reason"))
    return {
        "backend_transition_count": transitions,
        "factor_count": len(routes),
        "distinct_routes": len(set(routes)),
        "routes": sorted(set(routes)),
        "sql_pushdown_factors": sql_paths,
        "polars_native_factors": polars_native,
        "fallback_factors": fallback_paths,
        "avg_transitions_per_factor": round(transitions / max(1, len(routes)), 3),
    }


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


def _batch_source_bars_per_day(engine: Any) -> int:
    """批跑数据源的日内 bar 数（intraday trim 精度用）。"""
    from cleaned_operators.operator_policy import bars_per_day, infer_source_bar_freq

    try:
        return bars_per_day(infer_source_bar_freq(engine.data_source))
    except Exception:
        return 1


def _trim_batch_result(result: Any, run_window: Any, *, bars_per_day: int) -> Any:
    """按因子请求区间裁剪批跑输出（与 ``run`` 的 warmup trim 对齐）。"""
    if result is None or run_window is None:
        return result
    if not (run_window.trim_output and run_window.requested_start):
        return result
    import pandas as pd

    from storage.time_window import slice_series_time_window

    trim_start = pd.Timestamp(run_window.requested_start)
    trim_end = (
        pd.Timestamp(run_window.requested_end)
        if run_window.requested_end
        else None
    )
    if bars_per_day > 1:
        # 日内源保留 timestamp 精度（不按 normalize 截断）
        trim_start = pd.Timestamp(run_window.requested_start)
    return slice_series_time_window(result, start=trim_start, end=trim_end)


def _maybe_prepare_batch_warmup(
    engine: "FactorEngine",
    factors: Sequence[Factor],
    analyses: dict[str, AnalysisResult],
    *,
    auto_warmup: bool,
    trim_warmup: bool,
    market: str | None,
) -> tuple[Any, dict[str, Any]]:
    """跨因子合并共享 warmup 加载窗口，一次性 narrow 数据源。

    返回 ``(engine_to_use, per_factor_run_windows)``：
        - ``engine_to_use``：窄化到「所有因子最大 lookback / 最早 full-history 起点」
          的引擎；``auto_warmup=False`` 或无窗口需求时原样返回原引擎。
        - ``per_factor_run_windows``：``factor.name -> RunWindow``，供执行后按各自
          请求区间裁剪输出（batch warmup = 一次读数，各因子独立 trim）。

    这样 ``auto_warmup=True`` 不再让 ``run_many`` / ``run_many_parallel``
    退化为逐因子 ``run()``；PIT 审计由编译期 ``assert_pit_safe`` 负责，也与执行解耦。
    """
    if not auto_warmup:
        return engine, {}
    from cleaned_operators.operator_policy import bars_per_day, infer_source_bar_freq
    from runtime.run_window import RunWindow, extract_source_date_bounds
    from runtime.warmup_service import _switch_engine_to_window, prepare_run_warmup

    per_windows: dict[str, Any] = {}
    union_start: str | None = None
    union_end: str | None = None
    any_full_history = False
    for factor in factors:
        analysis = analyses.get(factor.name)
        if analysis is None:
            continue
        wctx = prepare_run_warmup(
            engine,
            factor,
            analysis,
            auto_warmup=True,
            trim_warmup=trim_warmup,
            market=market,
        )
        rw = wctx.run_window
        if rw is None:
            continue
        per_windows[factor.name] = rw
        any_full_history = any_full_history or bool(rw.full_history_required)
        if rw.actual_load_start is not None and (
            union_start is None or rw.actual_load_start < union_start
        ):
            union_start = rw.actual_load_start
        if rw.actual_load_end is not None and (
            union_end is None or rw.actual_load_end > union_end
        ):
            union_end = rw.actual_load_end
    if union_start is None:
        return engine, per_windows

    requested_start, requested_end = extract_source_date_bounds(engine.data_source)
    union = RunWindow(
        requested_start=requested_start,
        requested_end=requested_end,
        actual_load_start=union_start,
        actual_load_end=union_end,
        warmup_bars=0,
        trim_output=trim_warmup,
        full_history_required=any_full_history,
    )
    source_bar_freq = infer_source_bar_freq(engine.data_source)
    engine_to_use = _switch_engine_to_window(
        engine,
        union,
        source_bar_freq=source_bar_freq,
        bars_per_day=bars_per_day(source_bar_freq),
    )
    if engine_to_use is engine:
        # 实际加载起点 == 请求起点 → 无需 narrow，也无需裁剪
        return engine, per_windows
    logger.info(
        "batch auto_warmup: union load_start=%s load_end=%s full_history=%s factors=%d",
        union_start,
        union_end,
        any_full_history,
        len(per_windows),
    )
    return engine_to_use, per_windows


def choose_execution_mode(
    factors: Sequence[Factor],
    analyses: dict[str, AnalysisResult],
    dag: Any,
) -> str:
    """R33 §39：Auto Execution Mode——小任务不并行，大任务不逐 root。

    依据 factor count + IR node count + source count + native coverage 选择：
        DIRECT_VECTOR：极少量简单因子 → serial fused（不建 future / 不占 lease）。
        ADAPTIVE_DAG ：真正需要多 source / barrier / 资源约束的完整调度。
    记录在 batch_out，供 hard gate R33_SMALL_BATCH_AUTO_OVERHEAD_GATE_PASS 消费。
    """
    n_factors = len(factors)
    node_count = 0
    for a in (analyses or {}).values():
        node_count += max(1, int(getattr(a, "node_count", 0) or getattr(a, "ir_nodes", 0) or 1))
    source_count = max(1, len(getattr(dag, "roots", ()) or ()))
    # DIRECT_VECTOR：≤4 因子且总 IR 节点少（调度开销 > 计算节省）。
    if n_factors <= 4 and node_count <= 400:
        return "DIRECT_VECTOR"
    return "ADAPTIVE_DAG"


def _execute_run_many_scheduler(
    engine: "FactorEngine",
    factors: Sequence[Factor],
    dag: Any,
    analyses: dict[str, AnalysisResult],
    *,
    perf: PerfConfig,
    engine_to_use: "FactorEngine",
    per_windows: dict[str, Any],
    input_report: Any,
    run_mode: str | None,
    source_bars_per_day: int,
    result_policy: str,
    sink: Any,
    enable_cse: bool,
    max_concurrency: int | None = None,
    n_jobs: int | None = None,
    input_dq_check: bool = False,
    input_dq_strict: bool = True,
    input_dq_thresholds=None,
) -> dict[str, Any]:
    """R31-P0-001：**默认生产执行链** = BatchCompiler → PhysicalPlanner →
    AdaptiveBatchScheduler → StreamingSink。

    - shared CSE 与 root 统一进 scheduler 的 topological ready queue + admission
      + as_completed（不再是「shared 先全算 → layer loop」双 control plane）。
    - 每个 task 经 ResourceBroker token admission；完成即 sink/release（CSE
      refcount 归零立即释放）。
    - ``backend_paths`` 逐 root 捕获（production fastpath 校验在
      ``_execute_root_with_path`` 内照常进行）。
    - ``FACTOR_ENGINE_LAYER_LOOP=1`` 时降级旧 layer-loop（reference/compat/
      debug mode），保证研究可对照。
    """
    from concurrent.futures import Future
    import threading

    from backend.routing_env import routing_execution_scope
    from runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
    from runtime.resource_telemetry import record_resource_telemetry

    ctx = engine_to_use._make_context(shared_result_cache={}, perf=perf)
    ctx.runtime_stats = record_resource_telemetry(ctx.runtime_stats)
    scheduler = AdaptiveBatchScheduler(
        max_concurrency=max_concurrency,
    )
    # R33-P0-001..006：BatchDataRequest —— 从 analyses/plan 提取（multi-source）
    # → 每 source scope 一次 ScanCost（真实 time_range + instruments）→ 喂 read
    # wave / IO token / admission。估算失败记录 degraded planning（不静默）。
    from planner.batch_data_request import build_batch_data_request

    batch_request = build_batch_data_request(
        engine_to_use, analyses=analyses, dag=dag, ctx=ctx
    )
    # R33-P0-017：scheduler 路径**不再** full-union prefetch（``_maybe_prepare_
    # batch_data`` 是 layer-loop 专属）。read wave 是唯一 prefetch 路径，input DQ
    # 按 wave 在 scan 后执行。
    input_report = None
    plan = scheduler.plan(
        dag,
        analyses,
        enable_cse=enable_cse,
        scope_scan_cost_map=batch_request.scan_cost_map,
        ctx=ctx,
    )
    batch_request_meta = batch_request.to_dict()
    # R33-P0-061/§31：batch-global physical route（batch 级成本 = time-to-durable-
    # commit，不是 operator 数）。记录进 runtime_stats（runtime evidence）。
    try:
        from backend.plan_cost_router import plan_batch_route, record_batch_route

        root_plans = {
            fp.factor_name: fp.root for fp in dag.roots
        }
        batch_route = plan_batch_route(
            root_plans,
            ctx,
            scan_cost_map=batch_request.scan_cost_map,
            shared_roots=len(dag.shared_nodes or {}),
            factor_count=len(factors),
        )
        record_batch_route(ctx, batch_route)
        batch_route_meta = batch_route.to_dict()
    except Exception:
        batch_route_meta = {}
    out: dict[str, Any] = {}
    backend_paths: dict[str, dict[str, Any]] = {}
    paths_lock = threading.Lock()
    results_lock = threading.Lock()

    def _execute_root(task) -> Any:
        result, path = _execute_root_with_path(
            engine_to_use.backend,
            task.node_ref.root if getattr(task.node_ref, "root", None) is not None else task.node_ref,
            ctx,
            run_mode=run_mode,
            factor_name=task.factor_name,
        )
        if per_windows and task.factor_name in per_windows:
            result = _trim_batch_result(
                result, per_windows[task.factor_name], bars_per_day=source_bars_per_day
            )
        with paths_lock:
            backend_paths[task.factor_name] = path
        return result

    def _handle(name: str, result: Any) -> None:
        with results_lock:
            _handle_result(
                result_policy, sink, out, name, result,
                backend_paths.get(name, {}), backend_paths,
            )

    with routing_execution_scope(perf):
        if ctx.shared_result_cache is not None:
            _setup_cse_refcounts(ctx, dag.roots)
            validate_cse_refcount_integrity(dag, ctx)
        # R33 §39：Auto Execution Mode——小批量走 serial fused（DIRECT_VECTOR），
        # 不建 future / 不占 lease（scheduler overhead > compute savings）。
        mode = choose_execution_mode(factors, analyses, dag)
        if mode == "DIRECT_VECTOR":
            run_stats = scheduler.run_serial_fused(
                plan,
                backend=engine_to_use.backend,
                ctx=ctx,
                execute_root=_execute_root,
                result_handler=_handle,
            )
        else:
            run_stats = scheduler.run(
                plan,
                backend=engine_to_use.backend,
                ctx=ctx,
                execute_root=_execute_root,
                result_handler=_handle,
                input_dq_check=input_dq_check,
                input_dq_strict=input_dq_strict,
                input_dq_thresholds=input_dq_thresholds,
            )
        if isinstance(run_stats.get("scheduler_stats"), dict):
            run_stats["scheduler_stats"]["auto_execution_mode"] = mode
        else:
            run_stats["scheduler_stats"] = {"auto_execution_mode": mode}
        run_stats["auto_execution_mode"] = mode
    batch_out: dict[str, Any] = {
        "results": out,
        "dag": dag,
        "analyses": analyses,
        "executor": "adaptive_batch_scheduler",
        "scheduler_stats": run_stats,
        "batch_data_request": batch_request_meta,
        "batch_physical_route": batch_route_meta,
    }
    _attach_batch_backend_paths(batch_out, backend_paths)
    # R31-104/103：backend transition + conversion 是第一等 telemetry。
    batch_out["backend_transition_telemetry"] = _transition_telemetry(backend_paths)
    if input_report is not None:
        batch_out["input_dq"] = input_report.to_dict()
    if len(factors) > 1:
        from planner.dependency_graph import build_factor_batch_graph

        batch_graph = build_factor_batch_graph(factors, analyses)
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
    if per_windows:
        batch_out["warmup_windows"] = {
            name: rw.to_dict() for name, rw in per_windows.items()
        }
    assert_production_fastpath_runtime(ctx, mode=run_mode, context="run_many:scheduler")
    fallbacks = summarize_pandas_fallbacks(ctx)
    if fallbacks:
        batch_out["production_pandas_fallbacks"] = fallbacks
    from backend.path_summary import summarize_lazy_caches

    lazy_cache = summarize_lazy_caches(ctx)
    if lazy_cache:
        batch_out["lazy_cache_summary"] = lazy_cache
    ctx.runtime_stats = record_resource_telemetry(ctx.runtime_stats, finalize=True)
    batch_out["resource_telemetry"] = dict(ctx.runtime_stats.get("resource") or {})
    return batch_out


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
    result_policy: str = "return",
    sink: Any = None,
    warmup_clusters: bool = False,
) -> dict[str, Any]:
    """``FactorEngine.run_many`` 实现体：多因子 DAG 串行求值。

    流程：编译多因子 DAG（可选 CSE）→ 批量 prefetch/input_dq →
    物化 ``shared_nodes`` → 按依赖图分层执行各因子根。

    PIT 审计在编译期完成（``assert_pit_safe``），不退化批跑；``auto_warmup``
    通过 ``_maybe_prepare_batch_warmup`` 合并共享 union 加载窗口，一次读数、
    各因子独立 trim——都不再退化为逐因子 ``run()``。

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
    from runtime.production_policy import is_production_mode

    pit_enforce = bool(pit_enforce or is_production_mode(engine.run_mode))
    assert_production_run_flags(
        mode=engine.run_mode,
        input_dq_check=input_dq_check,
        auto_warmup=auto_warmup,
        pit_enforce=pit_enforce,
        context="run_many",
    )
    assert_production_factors(factors, mode=engine.run_mode, context="run_many")

    # Phase 5 P1-4：warmup 按成本聚类——先编译拿 analyses，按 lookback 分 waves，
    # 每个 wave 单独 union 窗口，避免一个 full-history 因子拖累整批。
    if warmup_clusters and auto_warmup and len(factors) > 1:
        _dag, analyses = engine._dag_from_factors(
            factors,
            enable_cse=enable_cse,
            perf=perf,
            pit_enforce=pit_enforce,
            pit_forbid_forward_fill=pit_forbid_forward_fill,
        )
        waves = _cluster_factors_by_cost(engine, factors, analyses)
        if len(waves) > 1:
            merged: dict[str, Any] = {"results": {}, "analyses": analyses}
            for wave in waves:
                partial = execute_run_many(
                    engine,
                    wave,
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
                    result_policy=result_policy,
                    sink=sink,
                )
                merged["results"].update(partial.get("results") or {})
                for key in ("input_dq", "batch_graph", "backend_paths", "plan_costs"):
                    if key in partial and key not in merged:
                        merged[key] = partial[key]
            merged["warmup_clustered"] = True
            merged["warmup_waves"] = [
                [getattr(f, "name", "") for f in wave] for wave in waves
            ]
            return merged

    # PIT 审计在编译期做（assert_pit_safe 是纯审计、不改计划），与执行解耦，
    # 因此 production 的 pit_enforce 不再让 run_many 退化为逐因子 run()。
    dag, analyses = engine._dag_from_factors(
        factors,
        enable_cse=enable_cse,
        perf=perf,
        pit_enforce=pit_enforce,
        pit_forbid_forward_fill=pit_forbid_forward_fill,
    )
    perf = perf or PerfConfig.from_env()
    # batch warmup：跨因子合并共享加载窗口，一次读数，各因子独立 trim。
    engine_to_use, per_windows = _maybe_prepare_batch_warmup(
        engine,
        factors,
        analyses,
        auto_warmup=auto_warmup,
        trim_warmup=trim_warmup,
        market=market,
    )
    run_mode = engine_to_use.run_mode
    input_report = _maybe_prepare_batch_data(
        engine_to_use,
        dag,
        analyses,
        input_dq_check=input_dq_check,
        input_dq_strict=input_dq_strict,
        input_dq_thresholds=input_dq_thresholds,
    )
    source_bars_per_day = _batch_source_bars_per_day(engine_to_use)
    # R31-P0-001：默认主执行链 = AdaptiveBatchScheduler（BatchCompiler →
    # PhysicalPlanner → Scheduler → StreamingSink）。旧 layer-loop 仅保留为
    # compatibility / reference / debug mode（``FACTOR_ENGINE_LAYER_LOOP=1``）。
    if os.environ.get("FACTOR_ENGINE_LAYER_LOOP") != "1":
        return _execute_run_many_scheduler(
            engine,
            factors,
            dag,
            analyses,
            perf=perf,
            engine_to_use=engine_to_use,
            per_windows=per_windows,
            input_report=input_report,
            run_mode=run_mode,
            source_bars_per_day=source_bars_per_day,
            result_policy=result_policy,
            sink=sink,
            enable_cse=bool(enable_cse if enable_cse is not None else perf.enable_cse),
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
            input_dq_thresholds=input_dq_thresholds,
        )

    ctx = engine_to_use._make_context(shared_result_cache={}, perf=perf)
    from runtime.resource_telemetry import record_resource_telemetry

    ctx.runtime_stats = record_resource_telemetry(ctx.runtime_stats)
    from backend.routing_env import routing_execution_scope
    from planner.dependency_graph import build_factor_batch_graph

    batch_graph = build_factor_batch_graph(factors, analyses)
    root_by_name = {fp.factor_name: fp for fp in dag.roots}

    def _run_root(fp) -> tuple[Any, Any]:
        result, path = _execute_root_with_path(
            engine_to_use.backend, fp.root, ctx, run_mode=run_mode, factor_name=fp.factor_name
        )
        if per_windows and fp.factor_name in per_windows:
            result = _trim_batch_result(
                result, per_windows[fp.factor_name], bars_per_day=source_bars_per_day
            )
        return result, path

    with routing_execution_scope(perf):
        if ctx.shared_result_cache is not None:
            materialize_shared_nodes_parallel(dag, engine_to_use.backend, ctx)
            _clear_polars_long_shared_sid(ctx)
            _setup_cse_refcounts(ctx, dag.roots)
            validate_cse_refcount_integrity(dag, ctx)
        out: dict[str, Any] = {}
        backend_paths: dict[str, dict[str, Any]] = {}
        for layer in batch_graph.parallel_layers:
            for name in layer:
                fp = root_by_name.get(name)
                if fp is not None:
                    result, path = _run_root(fp)
                    _handle_result(result_policy, sink, out, name, result, path, backend_paths)
            # Phase 5 R5：本层全部消费完成，引用计数归零的共享子树立即释放
            for name in layer:
                fp = root_by_name.get(name)
                if fp is not None:
                    _release_consumed_sids(ctx, fp.root)
        for fp in dag.roots:
            if fp.factor_name not in out and fp.factor_name not in backend_paths:
                result, path = _run_root(fp)
                _handle_result(result_policy, sink, out, fp.factor_name, result, path, backend_paths)
                _release_consumed_sids(ctx, fp.root)
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
    if per_windows:
        batch_out["warmup_windows"] = {
            name: rw.to_dict() for name, rw in per_windows.items()
        }
    assert_production_fastpath_runtime(ctx, mode=run_mode, context="run_many")
    fallbacks = summarize_pandas_fallbacks(ctx)
    if fallbacks:
        batch_out["production_pandas_fallbacks"] = fallbacks
    from backend.path_summary import summarize_lazy_caches

    lazy_cache = summarize_lazy_caches(ctx)
    if lazy_cache:
        batch_out["lazy_cache_summary"] = lazy_cache
    from runtime.resource_telemetry import record_resource_telemetry

    ctx.runtime_stats = record_resource_telemetry(ctx.runtime_stats, finalize=True)
    batch_out["resource_telemetry"] = dict(ctx.runtime_stats.get("resource") or {})
    return batch_out


def execute_run_many_iter(
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
    precompiled: tuple | None = None,
):
    """``FactorEngine.run_many_iter`` 实现体：生成器，每算完一个因子根立即 yield。

    共享子树照常先物化；每个 root 完成后 ``yield (name, result, path)``，同时做
    CSE 引用计数释放。调用方每次取走一个结果后即可落盘/筛选并释放，内存峰值
    不随因子数线性增长。

    ``precompiled=(dag, analyses)`` 时跳过重复编译（供 materialize_many 预编译
    后流式物化使用）。
    """
    from runtime.production_policy import is_production_mode

    pit_enforce = bool(pit_enforce or is_production_mode(engine.run_mode))
    if precompiled is not None:
        dag, analyses = precompiled
    else:
        dag, analyses = engine._dag_from_factors(
            factors,
            enable_cse=enable_cse,
            perf=perf,
            pit_enforce=pit_enforce,
            pit_forbid_forward_fill=pit_forbid_forward_fill,
        )
    perf = perf or PerfConfig.from_env()
    engine_to_use, per_windows = _maybe_prepare_batch_warmup(
        engine,
        factors,
        analyses,
        auto_warmup=auto_warmup,
        trim_warmup=trim_warmup,
        market=market,
    )
    run_mode = engine_to_use.run_mode
    input_report = _maybe_prepare_batch_data(
        engine_to_use,
        dag,
        analyses,
        input_dq_check=input_dq_check,
        input_dq_strict=input_dq_strict,
        input_dq_thresholds=input_dq_thresholds,
    )
    ctx = engine_to_use._make_context(shared_result_cache={}, perf=perf)
    from backend.routing_env import routing_execution_scope
    from planner.dependency_graph import build_factor_batch_graph

    batch_graph = build_factor_batch_graph(factors, analyses)
    root_by_name = {fp.factor_name: fp for fp in dag.roots}
    source_bars_per_day = _batch_source_bars_per_day(engine_to_use)

    with routing_execution_scope(perf):
        if ctx.shared_result_cache is not None:
            materialize_shared_nodes_parallel(dag, engine_to_use.backend, ctx)
            _clear_polars_long_shared_sid(ctx)
            _setup_cse_refcounts(ctx, dag.roots)
            validate_cse_refcount_integrity(dag, ctx)
        seen: set[str] = set()
        for layer in batch_graph.parallel_layers:
            for name in layer:
                fp = root_by_name.get(name)
                if fp is None:
                    continue
                result, path = _execute_root_with_path(
                    engine_to_use.backend, fp.root, ctx, run_mode=run_mode, factor_name=fp.factor_name
                )
                if per_windows and fp.factor_name in per_windows:
                    result = _trim_batch_result(
                        result, per_windows[fp.factor_name], bars_per_day=source_bars_per_day
                    )
                seen.add(fp.factor_name)
                _release_consumed_sids(ctx, fp.root)
                yield fp.factor_name, result, path
        for fp in dag.roots:
            if fp.factor_name in seen:
                continue
            result, path = _execute_root_with_path(
                engine_to_use.backend, fp.root, ctx, run_mode=run_mode, factor_name=fp.factor_name
            )
            if per_windows and fp.factor_name in per_windows:
                result = _trim_batch_result(
                    result, per_windows[fp.factor_name], bars_per_day=source_bars_per_day
                )
            _release_consumed_sids(ctx, fp.root)
            yield fp.factor_name, result, path


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
    result_policy: str = "return",
    sink: Any = None,
) -> dict[str, Any]:
    """``FactorEngine.run_many_parallel`` 实现体：根节点层内并行。

    共享子树仍串行物化；同一依赖层内的因子根通过 joblib 线程池并行。
    PIT 在编译期审计、``auto_warmup`` 走共享 union 窗口——两者都不再退化。

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
    from runtime.production_policy import is_production_mode

    pit_enforce = bool(pit_enforce or is_production_mode(engine.run_mode))
    assert_production_run_flags(
        mode=engine.run_mode,
        input_dq_check=input_dq_check,
        auto_warmup=auto_warmup,
        pit_enforce=pit_enforce,
        context="run_many_parallel",
    )
    assert_production_factors(factors, mode=engine.run_mode, context="run_many_parallel")

    # R27-166：并行 root 路径改用 ``concurrent.futures`` 线程池 as_completed
    # 流式 sink（不再依赖 joblib，也避免整层 raw list burst memory）。

    # PIT 在编译期审计；auto_warmup 走共享 union 窗口 —— 两者都不再退化为逐因子 run()。
    dag, analyses = engine._dag_from_factors(
        factors,
        enable_cse=enable_cse,
        perf=perf,
        pit_enforce=pit_enforce,
        pit_forbid_forward_fill=pit_forbid_forward_fill,
    )
    perf = perf or PerfConfig.from_env()
    workers = n_jobs if n_jobs is not None else perf.max_workers
    # Phase 5 R1/R15：统一 ExecutionResourcePlan（CPU+RAM 双约束），并包裹
    # ExecutionResourceScope 在退出时恢复线程/内存设置，避免共享 mutable 状态污染。
    # R20-132..137：production 下资源治理坏了绝不能「无约束继续跑」——plan 解析
    # 失败必须 hard fail；research 才回退 ``resource_scope=None`` 并告警。
    try:
        from runtime.execution_resources import resource_plan
        from runtime.resource_governor import ExecutionResourceScope

        plan = resource_plan(n_jobs=workers)
        workers = plan.n_jobs
        resource_scope = ExecutionResourceScope(
            plan,
            duckdb_threads=plan.duckdb_threads,
            strict=is_production_mode(engine.run_mode),
        )
        logger.info(
            "execution_resources: jobs=%d duckdb_threads=%d total_runnable=%d",
            plan.n_jobs,
            plan.duckdb_threads,
            plan.total_runnable,
        )
    except Exception:
        if is_production_mode(engine.run_mode):
            raise
        resource_scope = None
        logger.warning(
            "execution_resources: resource plan resolution failed; running "
            "unconstrained (research only)",
            exc_info=True,
        )
    engine_to_use, per_windows = _maybe_prepare_batch_warmup(
        engine,
        factors,
        analyses,
        auto_warmup=auto_warmup,
        trim_warmup=trim_warmup,
        market=market,
    )
    run_mode = engine_to_use.run_mode
    input_report = _maybe_prepare_batch_data(
        engine_to_use,
        dag,
        analyses,
        input_dq_check=input_dq_check,
        input_dq_strict=input_dq_strict,
        input_dq_thresholds=input_dq_thresholds,
    )
    source_bars_per_day = _batch_source_bars_per_day(engine_to_use)
    # R31-P0-001：production parallel 默认 = AdaptiveBatchScheduler（资源 scope
    # 仍包裹以统一线程/内存设置）。旧 layer-loop 仅 compat/debug（env 显式）。
    if os.environ.get("FACTOR_ENGINE_LAYER_LOOP") != "1":
        _enter_scope = (lambda: resource_scope.__enter__()) if resource_scope is not None else (lambda: None)
        _exit_scope = (
            (lambda: resource_scope.__exit__(None, None, None))
            if resource_scope is not None
            else (lambda: None)
        )
        _enter_scope()
        try:
            return _execute_run_many_scheduler(
                engine,
                factors,
                dag,
                analyses,
                perf=perf,
                engine_to_use=engine_to_use,
                per_windows=per_windows,
                input_report=input_report,
                run_mode=run_mode,
                source_bars_per_day=source_bars_per_day,
                result_policy=result_policy,
                sink=sink,
                enable_cse=bool(enable_cse if enable_cse is not None else perf.enable_cse),
                max_concurrency=workers,
                input_dq_check=input_dq_check,
                input_dq_strict=input_dq_strict,
                input_dq_thresholds=input_dq_thresholds,
                n_jobs=workers,
            )
        finally:
            _exit_scope()

    ctx = engine_to_use._make_context(shared_result_cache={}, perf=perf)
    from runtime.resource_telemetry import record_resource_telemetry

    ctx.runtime_stats = record_resource_telemetry(ctx.runtime_stats)
    from backend.routing_env import routing_execution_scope
    from planner.dependency_graph import build_factor_batch_graph

    batch_graph = build_factor_batch_graph(factors, analyses)
    root_by_name = {fp.factor_name: fp for fp in dag.roots}

    def _one(fp):
        result, path = _execute_root_with_path(
            engine_to_use.backend, fp.root, ctx, run_mode=run_mode, factor_name=fp.factor_name
        )
        if per_windows and fp.factor_name in per_windows:
            result = _trim_batch_result(
                result, per_windows[fp.factor_name], bars_per_day=source_bars_per_day
            )
        return fp.factor_name, result, path

    _enter_scope = (lambda: resource_scope.__enter__()) if resource_scope is not None else (lambda: None)
    # R27-166 修复：手工调 ``__exit__`` 必须带 exc_type/exc/tb 三参（with 语句
    # 自动传；这里 finally 手工调用，原 ``lambda *a`` 展开成 0 参调用必炸）。
    _exit_scope = (
        (lambda: resource_scope.__exit__(None, None, None))
        if resource_scope is not None
        else (lambda: None)
    )
    _enter_scope()
    try:
        with routing_execution_scope(perf):
            if ctx.shared_result_cache is not None:
                materialize_shared_nodes_parallel(dag, engine_to_use.backend, ctx)
                _clear_polars_long_shared_sid(ctx)
                _setup_cse_refcounts(ctx, dag.roots)
            validate_cse_refcount_integrity(dag, ctx)

            results: dict[str, Any] = {}
            backend_paths: dict[str, dict[str, Any]] = {}
            for layer in batch_graph.parallel_layers:
                fps = [root_by_name[n] for n in layer if n in root_by_name]
                if len(fps) <= 1:
                    for fp in fps:
                        result, path = _execute_root_with_path(
                            engine_to_use.backend, fp.root, ctx, run_mode=run_mode, factor_name=fp.factor_name
                        )
                        if per_windows and fp.factor_name in per_windows:
                            result = _trim_batch_result(
                                result, per_windows[fp.factor_name], bars_per_day=source_bars_per_day
                            )
                        _handle_result(result_policy, sink, results, fp.factor_name, result, path, backend_paths)
                else:
                    # R27-166/249：as_completed → 立即 sink/release，不再等整层
                    # raw list 形成（避免 layer result burst memory，R27-103）。
                    from concurrent.futures import ThreadPoolExecutor, as_completed

                    with ThreadPoolExecutor(
                        max_workers=workers, thread_name_prefix="r27-root"
                    ) as pool:
                        futures = {pool.submit(_one, fp): fp for fp in fps}
                        for future in as_completed(futures):
                            name, result, path = future.result()
                            _handle_result(
                                result_policy, sink, results, name, result, path, backend_paths
                            )
                # Phase 5 R5：本层完成，引用计数归零的共享子树立即释放
                for fp in fps:
                    _release_consumed_sids(ctx, fp.root)
            for fp in dag.roots:
                if fp.factor_name not in results and fp.factor_name not in backend_paths:
                    result, path = _execute_root_with_path(
                        engine_to_use.backend, fp.root, ctx, run_mode=run_mode, factor_name=fp.factor_name
                    )
                    if per_windows and fp.factor_name in per_windows:
                        result = _trim_batch_result(
                            result, per_windows[fp.factor_name], bars_per_day=source_bars_per_day
                        )
                    _handle_result(result_policy, sink, results, fp.factor_name, result, path, backend_paths)
                    _release_consumed_sids(ctx, fp.root)
    finally:
        _exit_scope()
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
    if per_windows:
        parallel_out["warmup_windows"] = {
            name: rw.to_dict() for name, rw in per_windows.items()
        }
    assert_production_fastpath_runtime(ctx, mode=run_mode, context="run_many_parallel")
    fallbacks = summarize_pandas_fallbacks(ctx)
    if fallbacks:
        parallel_out["production_pandas_fallbacks"] = fallbacks
    from backend.path_summary import summarize_lazy_caches

    lazy_cache = summarize_lazy_caches(ctx)
    if lazy_cache:
        parallel_out["lazy_cache_summary"] = lazy_cache
    from runtime.resource_telemetry import record_resource_telemetry

    ctx.runtime_stats = record_resource_telemetry(ctx.runtime_stats, finalize=True)
    parallel_out["resource_telemetry"] = dict(ctx.runtime_stats.get("resource") or {})
    return parallel_out
