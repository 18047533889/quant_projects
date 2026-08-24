"""多因子批跑编排：run_many / run_many_parallel 核心逻辑。

本模块实现 ``FactorEngine.run_many`` 与 ``run_many_parallel`` 的执行体，
负责 CSE 共享子树物化、批量 input_dq、依赖图分层调度及 production 审计。
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Sequence

from factor_engine.api.factor import Factor
from factor_engine.ir.analyzer import AnalysisResult
from logging_utils import get_logger
from factor_engine.runtime.perf_config import PerfConfig
from factor_engine.runtime.production_policy import (
    assert_production_fastpath_runtime,
    assert_production_factors,
    assert_production_run_flags,
    summarize_pandas_fallbacks,
)
from factor_engine.runtime.readonly_overlay_map import ReadOnlyOverlayMap

logger = get_logger("factor_engine.runtime.batch_service")

if TYPE_CHECKING:
    from factor_engine.runtime.engine import FactorEngine

#: R39 Gate-01：adaptive-scheduler 主路径不允许发生 legacy full-union batch
#: prefetch（``engine._prepare_batch_data``）。该计数由 ``_maybe_prepare_batch_data``
#: 每次真正执行 union prefetch 时 +1；scheduler 路径根本不调用它，因此必须为 0。
_legacy_union_prefetch_count = 0


def _bump_legacy_union_prefetch_count(n: int = 1) -> None:
    global _legacy_union_prefetch_count
    _legacy_union_prefetch_count += n


def legacy_union_prefetch_count() -> int:
    """进程级 legacy full-union prefetch 计数（R39 Gate-01 探针）。"""
    return _legacy_union_prefetch_count


def reset_legacy_union_prefetch_count() -> None:
    """测试用：清零进程级计数。"""
    global _legacy_union_prefetch_count
    _legacy_union_prefetch_count = 0


def materialize_shared_nodes_parallel(
    dag: Any,
    backend: Any,
    ctx: Any,
    *,
    max_workers: int | None = None,
    executor: Any = None,
) -> None:
    """R27-165/231 + R42-014：独立 shared nodes **按依赖并行**物化。

    两个彼此完全独立的 shared nodes 现在并行；R27-002。共享子树物化本身
    ``_materialize_shared_subplan`` 线程安全（每 sid 独立写入
    ``ctx.shared_result_cache``）。有资源约束时 ``max_workers`` 限制并发。

    R42-014：优先使用传入的 ``executor``（long-lived pool），避免每次创建临时
    ThreadPoolExecutor。若无 executor 传入，fallback 到临时线程池（兼容旧调用）。

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

    def _one(pair):
        sid, sub = pair
        _materialize_shared_subplan(backend, sub, ctx, sid)

    # R42-014：优先使用传入的 long-lived executor
    if executor is not None:
        from concurrent.futures import wait
        futures = [executor.submit(_one, pair) for pair in items]
        done, _ = wait(futures)
        for f in done:
            f.result()  # 失败如实冒泡
        return

    # Fallback：创建临时线程池（兼容未传 executor 的旧路径）
    try:
        from concurrent.futures import ThreadPoolExecutor, wait

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
) -> bool:
    """Materialize shared CSE state and report whether eager execution occurred."""
    if (
        getattr(sub, "op", None) == "literal"
        and getattr(backend, "supports_lazy_shared", False)
    ):
        return False
    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    runtime["polars_long_shared_sid"] = sid
    ctx.runtime_stats = runtime  # type: ignore[attr-defined]
    if getattr(backend, "supports_lazy_shared", False):
        compile_lazy = getattr(backend, "compile_lazy_shared", None)
        if callable(compile_lazy) and compile_lazy(sub, ctx, sid=str(sid)):
            return False
    if ctx.shared_result_cache is not None:
        value = backend.execute(sub, ctx)
        # R36 P0-021（§104/106）：经 GovernedBufferStore 写入（byte 预算 + 记账 +
        # 冷淘汰），不再 raw dict 写入绕过治理。store 拒绝时回退 ExpressionCache
        # governed 路径；两者都不可用才允许 raw（研究降级），production 直接拒绝。
        store = getattr(ctx, "shared_buffers", None)
        if store is not None:
            # R38 P0-029（§12）：put 不再返回裸 bool——必须处理 REFUSED/RECOMPUTE。
            # P0-033（§14）：主链真实传入 recompute_cost（算子 cost 代理），LRU
            # 逐出时才能做 spill-vs-recompute 成本决策（高成本 spill / 低成本 drop）。
            # R42-015/016：从 CSECertificateStore 读取 recompute_cost/next_use_distance/future_consumers。
            recompute_ms = 0.0
            next_use_dist = 0
            future_cons = 1
            cert_store = getattr(ctx, "cse_certificate_store", None)
            if cert_store is not None:
                cert = cert_store.get(sid)
                if cert is not None:
                    recompute_ms = float(cert.recompute_cost_ms)
                    future_cons = int(cert.consumer_count)
                    # R42-016: reuse_distance 三元组取 min（最近一次复用）
                    if cert.reuse_distance and len(cert.reuse_distance) > 0:
                        next_use_dist = min(cert.reuse_distance)

            # Fallback: estimate_plan_cost（若 certificate 缺失）
            if recompute_ms <= 0.0:
                try:
                    from factor_engine.backend.operator_cost import estimate_plan_cost

                    recompute_ms = float(
                        estimate_plan_cost(sub).get("total_work", 0.0) or 0.0
                    )
                except Exception:
                    recompute_ms = 0.0

            res = store.put(
                sid,
                value,
                recompute_cost_ms=recompute_ms,
                next_use_distance=next_use_dist,
                future_consumers=future_cons,
            )
            if res.status in ("MEMORY", "SPILLED"):
                return True
            # REFUSED / RECOMPUTE：shared buffer 缺失会让 downstream plan_ref
            # KeyError —— production 必须 fail-closed，不能 return 成功。
            from factor_engine.runtime.production_policy import is_production_mode

            if is_production_mode(getattr(ctx, "run_mode", None)):
                raise RuntimeError(
                    f"governed buffer put for CSE sid={sid} returned "
                    f"{res.status}: {res.reason} (R38-P0-029: put refusal must "
                    "not be silently ignored in production)"
                )
            # research：显式降级走 expression_cache，并记录 telemetry。
            runtime = dict(getattr(ctx, "runtime_stats", None) or {})
            runtime["cse_buffer_put_refused"] = runtime.get("cse_buffer_put_refused", 0) + 1
            ctx.runtime_stats = runtime  # type: ignore[attr-defined]
        cache = getattr(ctx, "expression_cache", None)
        if cache is not None and getattr(cache, "set", None) is not None:
            cache.set(sid, value)
            return True
        # R37-P0-035：raw dict 写入只在 research 降级路径允许；production 下
        # store/ExpressionCache 都不可用 = 治理缺失，必须 fail-closed（§31.3：
        # 不允许 raw dict 绕过资源账本）。
        from factor_engine.runtime.production_policy import is_production_mode

        if is_production_mode(getattr(ctx, "run_mode", None)):
            raise RuntimeError(
                "R37-P0-035 fail-closed: no governed buffer store available in "
                f"production for CSE sid={sid}; raw shared_result_cache write is "
                "governance bypass"
            )
        # research 降级：显式 warning 语义的 raw 写入（telemetry 已在上面累计）。
        ctx.shared_result_cache[sid] = value
        return True
    return False


def _release_consumed_sids(ctx: Any, root: Any) -> None:
    """Phase 5 R5：root 执行完，对其消费的共享 sid 引用计数减一，归零立即释放。

    共享子树默认只在 batch 结束时随 ctx 一起释放；这里让大 panel CSE 在最后一个
    消费者完成时立刻逐出，而不是常驻整个 batch。
    """
    if ctx.shared_result_cache is None:
        return
    from factor_engine.cache.session import ExecutionCacheSession
    from factor_engine.planner.cse import collect_consumed_sids

    consumed = collect_consumed_sids(root)
    if not consumed:
        return
    refcounts = getattr(ctx, "_cse_refcounts", None)
    if refcounts is None:
        return
    cache = getattr(ctx, "expression_cache", None)
    store = getattr(ctx, "shared_buffers", None)
    for sid in consumed:
        remaining = refcounts.get(sid, 1) - 1
        refcounts[sid] = remaining
        if remaining <= 0:
            if store is not None:
                # R36 P0-023：backing ref 与 accounting 同时释放（governed store）。
                store.release(sid)
            elif cache is not None:
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
    from factor_engine.planner.cse import cse_consumer_counts

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
    from factor_engine.planner.cse import collect_consumed_sids, cse_consumer_counts

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
            from factor_engine.runtime.production_policy import is_production_mode

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
        from factor_engine.planner.composite_lowering import collect_plan_ops

        ops = set(collect_plan_ops(plan))
    except Exception:  # noqa: BLE001 - unanalyzable plan: skip invariant
        return
    if not ops:
        return
    try:
        from factor_engine.backend.primitive_evidence import PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE

        native_safe = PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
    except Exception:  # noqa: BLE001
        return
    if ops.issubset(native_safe):
        from factor_engine.runtime.production_policy import is_production_mode

        if is_production_mode(run_mode):
            raise RuntimeError(
                f"NATIVE_CERTIFIED_BUT_FALLBACK_EXECUTED != 0: factor={factor_name!r} "
                f"plan ops {sorted(ops)} are all dual-backend certified, yet the "
                "runtime executed a polars_long fallback "
                f"reason={runtime.get('polars_long_fallback_reason')!r} "
                f"op={runtime.get('polars_long_fallback_plan_op')!r}.  Native-certified "
                "plans must never fall back in production (R20-241..252)."
            )


def _make_root_local_overlays(ctx: Any) -> tuple[dict[str, Any], int]:
    """R39-P0-PERF-013：为 root 创建独立 overlay 写层，不复制共享 base。

    老路径用 ``dict(ctx.shared_result_cache)`` 等 4 个全量拷贝给每个 root 隔离
    可变视图。这里用 :class:`ReadOnlyOverlayMap`：base 只读、每 root 独立 local
    写层、读 miss 才查 base —— 把 O(shared_keys) 的 hash 拷贝降为 O(1) 的
    overlay 创建（R39 Gate-09）。

    返回 ``(overlays, entry_copy_count)``。``entry_copy_count`` 恒为 0（没有任何
    共享条目被拷贝进 root 上下文），就是 Gate-09 的进程级证明量之一。panel_cache
    保持每 root 独立空 dict（这是真正 per-root 的缓存，语义不变）。
    """
    shared = (
        ReadOnlyOverlayMap(ctx.shared_result_cache, {})
        if ctx.shared_result_cache is not None
        else None
    )
    shared_long = (
        ReadOnlyOverlayMap(ctx.shared_long_lazy_cache, {})
        if ctx.shared_long_lazy_cache is not None
        else None
    )
    mat_long = (
        ReadOnlyOverlayMap(ctx.materialized_long_lazy, {})
        if ctx.materialized_long_lazy is not None
        else None
    )
    mat_series = (
        ReadOnlyOverlayMap(ctx.materialized_series, {})
        if ctx.materialized_series is not None
        else None
    )
    return {
        "shared_result_cache": shared,
        "shared_long_lazy_cache": shared_long,
        "materialized_long_lazy": mat_long,
        "materialized_series": mat_series,
        "panel_cache": {},
    }, 0


def _execute_root_with_path(
    backend: Any,
    plan: Any,
    ctx: Any,
    *,
    run_mode: str | None = None,
    factor_name: str = "",
    physical_optimization: Any = None,
    physical_root_id: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """执行单个因子根计划并快照 backend 路径摘要。

    为每个因子根创建独立 ``runtime_stats`` 副本，避免路径统计互相污染；
    production 模式下校验 fast path runtime 合规性 + NATIVE_CERTIFIED_BUT_
    FALLBACK_EXECUTED == 0 不变量。

    Returns:
        ``(result, backend_path_dict)`` 元组。
    """
    from dataclasses import replace

    from factor_engine.backend.path_summary import snapshot_backend_path
    from factor_engine.backend.polars_long_backend import _fresh_long_runtime

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
    overlays, entry_copy_count = _make_root_local_overlays(ctx)
    # R39-P0-PERF-013：per-root overlay 不拷贝共享 base；``dict_entry_copy_count``
    # 为 0 表示本轮 root 上下文创建没有做任何共享条目拷贝（Gate-09 telemetry）。
    local_runtime["dict_entry_copy_count"] = entry_copy_count
    local_ctx = replace(
        ctx,
        runtime_stats=local_runtime,
        **overlays,
    )
    if physical_optimization is None:
        result = backend.execute(plan, local_ctx)
    else:
        # Local import avoids the engine -> batch_service execution-time cycle.
        from factor_engine.runtime.engine import _execute_ready_single_region_plan

        result = _execute_ready_single_region_plan(
            physical_optimization,
            plan,
            backend,
            local_ctx,
            logical_root_id=(
                physical_root_id
                if physical_root_id is not None
                else str(getattr(plan, "node_id", ""))
            ),
        )
    path = snapshot_backend_path(getattr(local_ctx, "runtime_stats", None))
    physical_meta = dict(
        (getattr(local_ctx, "runtime_stats", None) or {}).get("physical_plan") or {}
    )
    if physical_meta:
        path["physical_plan"] = physical_meta
        path.update(
            planned_backend=physical_meta["planned_backend"],
            actual_backend=physical_meta["actual_backend"],
            physical_region_id=physical_meta["region_id"],
        )
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
    *,
    plan: Any | None = None,
) -> list[list[Factor]]:
    """Phase 5 P1-4：按扫描成本把因子聚成 waves（R39-P0-PERF-015）。

    老实现用固定 lookback 比例桶（short/medium/long/full_history）。R39 改为
    :class:`runtime.batch_warmup_plan.BatchWarmupPlan` 的扫描成本感知分组：
    合并两组的额外 superset scan bytes < 分开扫描的重复 scan/decode 成本时才
    合并（min scan bytes + open/decode + duplicated compute），而非 lookback ratio。

    ``full_history`` 保持独立组（special-case，避免 full-history 因子把整批
    union 窗口拉成全历史）。未进 plan 的因子（无 analysis / 无 warmup 窗口）
    落到末尾兜底组，等价老 short 桶。
    """
    if plan is None:
        from factor_engine.runtime.batch_warmup_plan import compute_batch_warmup_plan

        plan = compute_batch_warmup_plan(
            analyses,
            None,
            engine,
            factors=factors,
            auto_warmup=True,
            trim_warmup=True,
            market=None,
            strict=True,
        )
    name_to_factor: dict[str, Factor] = {}
    for factor in factors:
        name = getattr(factor, "name", None)
        if name:
            name_to_factor[name] = factor
    clusters: list[list[Factor]] = []
    covered: set[str] = set()
    for group in plan.groups:
        members = [
            name_to_factor[n] for n in group.factor_names if n in name_to_factor
        ]
        if members:
            clusters.append(members)
            covered.update(m.name for m in members)
    uncovered = [f for f in factors if f.name not in covered]
    if uncovered:
        clusters.append(uncovered)
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
    from factor_engine.backend.path_summary import summarize_batch_backend_paths

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
    """合并全量依赖列后批量 prefetch 与 input_dq（layer-loop 专属）。

    R39-P0-PERF-001：本函数是 **legacy full-union prefetch**。adaptive-scheduler
    主路径（``resolve_batch_execution_control_plane`` 返回 ``adaptive_scheduler``）
    不得调用它 —— read wave 是唯一读取控制面。仅在 layer-loop 兼容路径调用；
    每次真正执行 union prefetch 时累计 ``legacy_union_prefetch_count``（Gate-01）。

    若计划可走 fully_sql / native_scan 则跳过 prefetch。
    """
    all_cols: set[str] = set()
    for analysis in analyses.values():
        all_cols |= analysis.referenced_columns
    all_plans = [fp.root for fp in dag.roots] + list(dag.shared_nodes.values())
    from factor_engine.planner.sql_io import should_skip_column_prefetch

    if all_cols and not should_skip_column_prefetch(
        all_plans,
        input_dq_check=input_dq_check,
        backend=engine.backend,
    ):
        _bump_legacy_union_prefetch_count(1)
        return engine._prepare_batch_data(
            engine.data_source,
            all_cols,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
            input_dq_thresholds=input_dq_thresholds,
        )
    return None


def resolve_batch_execution_control_plane(
    factors: Sequence[Factor] | None = None,
    analyses: dict[str, AnalysisResult] | None = None,
    dag: Any = None,
    *,
    engine: "FactorEngine" | None = None,
    input_dq_check: bool = False,
) -> str:
    """R39-P0-PERF-001：决定本批的执行 control plane。

    返回：
        - ``"adaptive_scheduler"``：默认生产执行链（BatchCompiler →
          PhysicalPlanner → AdaptiveBatchScheduler → StreamingSink）。此时
          **不允许** legacy full-union prefetch（``_maybe_prepare_batch_data``），
          read wave 是唯一读取控制面。
        - ``"legacy"``：``FACTOR_ENGINE_LAYER_LOOP=1`` 时的 layer-loop 兼容/
          reference/debug 路径，保留 full-union prefetch。

    未来可在这里扩展其它 control plane（microbatch 等），因此单独成函数。
    """
    if os.environ.get("FACTOR_ENGINE_LAYER_LOOP") == "1":
        return "legacy"
    return "adaptive_scheduler"


def _batch_source_bars_per_day(engine: Any) -> int:
    """批跑数据源的日内 bar 数（intraday trim 精度用）。"""
    from factor_engine.cleaned_operators.operator_policy import bars_per_day, infer_source_bar_freq

    try:
        return bars_per_day(infer_source_bar_freq(engine.data_source))
    except Exception:
        return 1


def _trim_batch_result(result: Any, run_window: Any, *, bars_per_day: int) -> Any:
    """按因子请求区间裁剪批跑输出（与 ``run`` 的 warmup trim 对齐）。

    R39-P0-PERF-017：native backend 若已在终端结果上挂 ``_output_slice``（位置
    切片视图），这里**不再二次切片**（writer 直接消费 ``(BufferRef, slice)``）。
    否则优先用位置 ``.iloc`` 视图（连续 block 零拷贝，``np.shares_memory`` 可验
    证）替代 boolean-mask copy；无法定位位置时回退旧
    :func:`storage.time_window.slice_series_time_window` 语义。
    """
    if result is None or run_window is None:
        return result
    if not (run_window.trim_output and run_window.requested_start):
        return result
    from factor_engine.planner.output_slice import (
        apply_output_slice,
        carried_output_slice,
        compute_output_slice,
    )

    # Backend 已切好视图 —— 不再重切（PERF-017）。
    carried = carried_output_slice(result)
    if carried is not None:
        return result
    out_slice = compute_output_slice(result, run_window, bars_per_day=bars_per_day)
    if out_slice is not None:
        return apply_output_slice(result, out_slice)
    import pandas as pd

    from factor_engine.storage.time_window import slice_series_time_window

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
    plan: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """跨因子合并共享 warmup 加载窗口，一次性 narrow 数据源。

    返回 ``(engine_to_use, per_factor_run_windows)``：
        - ``engine_to_use``：窄化到「所有因子最大 lookback / 最早 full-history 起点」
          的引擎；``auto_warmup=False`` 或无窗口需求时原样返回原引擎。
        - ``per_factor_run_windows``：``factor.name -> RunWindow``，供执行后按各自
          请求区间裁剪输出（batch warmup = 一次读数，各因子独立 trim）。

    R39-P0-PERF-014：``plan``（:class:`runtime.batch_warmup_plan.BatchWarmupPlan`）
    由调用方一次性计算，本函数只消费 ``plan.per_factor``，不再逐因子调用
    ``prepare_run_warmup``。未传 plan 时回退自行计算（保持兼容）。

    这样 ``auto_warmup=True`` 不再让 ``run_many`` / ``run_many_parallel``
    退化为逐因子 ``run()``；PIT 审计由编译期 ``assert_pit_safe`` 负责，也与执行解耦。
    """
    if not auto_warmup:
        return engine, {}
    if plan is None:
        from factor_engine.runtime.batch_warmup_plan import compute_batch_warmup_plan

        plan = compute_batch_warmup_plan(
            analyses,
            None,
            engine,
            factors=factors,
            auto_warmup=auto_warmup,
            trim_warmup=trim_warmup,
            market=market,
            strict=True,
        )
    from factor_engine.cleaned_operators.operator_policy import bars_per_day, infer_source_bar_freq
    from factor_engine.runtime.run_window import RunWindow, extract_source_date_bounds
    from factor_engine.runtime.warmup_service import _switch_engine_to_window

    per_windows: dict[str, Any] = dict(plan.per_factor)
    union_start: str | None = None
    union_end: str | None = None
    any_full_history = False
    for rw in per_windows.values():
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

    from factor_engine.backend.routing_env import routing_execution_scope
    from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
    from factor_engine.runtime.resource_telemetry import record_resource_telemetry

    ctx = engine_to_use._make_context(shared_result_cache={}, perf=perf)
    ctx.runtime_stats = record_resource_telemetry(ctx.runtime_stats)
    scheduler = AdaptiveBatchScheduler(
        max_concurrency=max_concurrency,
    )
    # R36 P0-016/017（§54/56）：FE batch run 是唯一资源权威——把 Safe Envelope
    # 应用到 DA governor（max_total_reserved_memory / scan inflight），DA 不再
    # 独立决定全局内存（no double admission）。
    try:
        from factor_engine.runtime.host_resource_coordinator import get_host_coordinator

        da_env = get_host_coordinator().apply_da_envelope()
        runtime = dict(ctx.runtime_stats or {})
        runtime["host_coordinator_da_envelope"] = da_env
        ctx.runtime_stats = runtime  # type: ignore[attr-defined]
    except Exception:
        pass
    # R38 P0-011（§7）：进程级固定 cadence 控制循环（幂等）——scheduler 只读
    # last_decision，绝不自行 tick controller（stable/cooldown 与 loop 次数解耦）。
    try:
        from factor_engine.runtime.resource_autopilot_service import start_resource_autopilot

        autopilot = start_resource_autopilot()
        runtime = dict(ctx.runtime_stats or {})
        runtime["resource_autopilot"] = autopilot.summary()
        ctx.runtime_stats = runtime  # type: ignore[attr-defined]
    except Exception:
        pass
    # R33-P0-001..006：BatchDataRequest —— 从 analyses/plan 提取（multi-source）
    # → 每 source scope 一次 ScanCost（真实 time_range + instruments）→ 喂 read
    # wave / IO token / admission。估算失败记录 degraded planning（不静默）。
    from factor_engine.planner.batch_data_request import build_batch_data_request

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
    # Some in-memory/reference sources lower directly to ROOT tasks and therefore
    # produce no SOURCE_SCAN read wave. Keep the scheduler as the control plane,
    # but coalesce that batch's declared fields into one read-session load so
    # DIRECT_VECTOR roots do not reload each dependency independently. This also
    # preserves input-DQ output for the no-wave scheduler path.
    if (
        input_dq_check
        and batch_request.fields
        and not any(getattr(wave, "columns", ()) for wave in plan.read_waves.waves)
    ):
        input_report = engine_to_use._prepare_batch_data(
            engine_to_use.data_source,
            set(batch_request.fields),
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
            input_dq_thresholds=input_dq_thresholds,
        )
    else:
        input_report = None
    batch_request_meta = batch_request.to_dict()
    if run_mode == "production":
        try:
            from factor_engine.planner.batch_global_optimizer import optimize_batch_global

            root_plans = {fp.factor_name: fp.root for fp in dag.roots}
            from factor_engine.runtime.engine import _admit_ready_single_region_batch

            physical_optimization = optimize_batch_global(
                root_plans,
                dict(dag.shared_nodes or {}),
                {},
                ctx,
            )
            physical_optimization = _admit_ready_single_region_batch(
                physical_optimization,
                tuple(root_plans),
            )
            physical_by_factor = {
                name: physical_optimization for name in root_plans
            }
            from factor_engine.runtime.engine import (
                _physical_backend_for_region,
                _physical_plan_telemetry,
            )

            admitted_region = physical_optimization.physical_plan.regions[0]
            execution_backend = _physical_backend_for_region(
                engine_to_use.backend, admitted_region.backend
            )
            physical_plan_meta = _physical_plan_telemetry(
                physical_optimization,
                actual_backend=str(
                    getattr(execution_backend, "runtime_backend_label", "")
                    or admitted_region.backend.value
                ),
            )
            runtime = dict(ctx.runtime_stats or {})
            runtime["physical_plan"] = physical_plan_meta
            ctx.runtime_stats = runtime
        except ImportError:
            raise
    else:
        physical_optimization = None
        physical_by_factor = {}
        physical_plan_meta = {}
        execution_backend = engine_to_use.backend

    if run_mode == "production":
        batch_route_meta = {}
    else:
        try:
            from factor_engine.backend.plan_cost_router import plan_batch_route, record_batch_route

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
            execution_backend,
            task.node_ref.root if getattr(task.node_ref, "root", None) is not None else task.node_ref,
            ctx,
            run_mode=run_mode,
            factor_name=task.factor_name,
            physical_optimization=physical_by_factor.get(task.factor_name),
            physical_root_id=task.factor_name,
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

    materialized_shared_sids: set[str] = set()
    materialized_shared_lock = threading.Lock()

    def _materialize_shared(sid: str, node: Any) -> bool:
        eagerly_materialized = _materialize_shared_subplan(
            execution_backend, node, ctx, sid
        )
        if eagerly_materialized:
            with materialized_shared_lock:
                materialized_shared_sids.add(str(sid))
        return eagerly_materialized

    # R36 P0-029：run 级专用峰值采样器（start→stop 只统计本 run 窗口的
    # process family PSS/RSS，不混入 lifetime peak）。
    from factor_engine.runtime.run_peak_sampler import RunPeakSampler

    run_peak_sampler = RunPeakSampler(interval_s=0.5)
    run_peak_sampler.start()
    run_stats: dict[str, Any] = {}
    try:
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
                    backend=execution_backend,
                    ctx=ctx,
                    execute_root=_execute_root,
                    materialize_shared=_materialize_shared,
                    result_handler=_handle,
                    input_dq_check=input_dq_check,
                    input_dq_strict=input_dq_strict,
                    input_dq_thresholds=input_dq_thresholds,
                )
            else:
                run_stats = scheduler.run(
                    plan,
                    backend=execution_backend,
                    ctx=ctx,
                    execute_root=_execute_root,
                    materialize_shared=_materialize_shared,
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
            if input_report is None and scheduler._input_dq_reports:
                from factor_engine.runtime.input_dq import InputDQReport

                input_report = InputDQReport(
                    columns=[
                        column
                        for report in scheduler._input_dq_reports
                        for column in report.columns
                    ]
                )
    finally:
        peak = run_peak_sampler.stop()
        ctx.runtime_stats = record_resource_telemetry(
            ctx.runtime_stats, finalize=True, run_peak=peak
        )
        run_stats["run_peak"] = peak
    if physical_plan_meta:
        physical_plan_meta["materialization_count"] = len(materialized_shared_sids)
        runtime = dict(ctx.runtime_stats or {})
        runtime["physical_plan"] = physical_plan_meta
        ctx.runtime_stats = runtime
        for path in backend_paths.values():
            path_physical = path.get("physical_plan")
            if isinstance(path_physical, dict):
                path_physical["materialization_count"] = len(materialized_shared_sids)
    batch_out: dict[str, Any] = {
        "results": out,
        "dag": dag,
        "analyses": analyses,
        "executor": "adaptive_batch_scheduler",
        "scheduler_stats": run_stats,
        "batch_data_request": batch_request_meta,
        "batch_physical_route": batch_route_meta,
        "physical_plan": physical_plan_meta,
    }
    _attach_batch_backend_paths(batch_out, backend_paths)
    # R31-104/103：backend transition + conversion 是第一等 telemetry。
    batch_out["backend_transition_telemetry"] = _transition_telemetry(backend_paths)
    if input_report is not None:
        batch_out["input_dq"] = input_report.to_dict()
    if len(factors) > 1:
        from factor_engine.planner.dependency_graph import build_factor_batch_graph

        batch_graph = build_factor_batch_graph(factors, analyses)
        batch_out["batch_graph"] = batch_graph.to_dict()
    if dag.shared_nodes:
        from factor_engine.planner.rolling_cache import summarize_rolling_cache

        batch_out["rolling_cache"] = summarize_rolling_cache(dag.shared_nodes)
    if dag.roots:
        from factor_engine.backend.operator_cost import estimate_plan_cost

        batch_out["plan_costs"] = {
            fp.factor_name: estimate_plan_cost(fp.root) for fp in dag.roots
        }
        from factor_engine.planner.cost_summary import summarize_plans

        batch_out["cost_summary"] = summarize_plans(
            {fp.factor_name: fp.root for fp in dag.roots}
        )
        from factor_engine.planner.scheduling_hints import derive_scheduling_hints

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
    from factor_engine.backend.path_summary import summarize_lazy_caches

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
    from factor_engine.runtime.production_policy import is_production_mode

    pit_enforce = bool(pit_enforce or is_production_mode(engine.run_mode))
    assert_production_run_flags(
        mode=engine.run_mode,
        input_dq_check=input_dq_check,
        auto_warmup=auto_warmup,
        pit_enforce=pit_enforce,
        context="run_many",
    )
    assert_production_factors(factors, mode=engine.run_mode, context="run_many")

    # 编译一次（CSE DAG + analyses），聚类与主执行共享同一份编译结果；同时消除
    # 老代码在 warmup_clusters 分支里的重复 ``_dag_from_factors`` 二次编译。
    dag, analyses = engine._dag_from_factors(
        factors,
        enable_cse=enable_cse,
        perf=perf,
        pit_enforce=pit_enforce,
        pit_forbid_forward_fill=pit_forbid_forward_fill,
    )
    perf = perf or PerfConfig.from_env()
    # R39-P0-PERF-014：warmup 规划（analysis → RunWindow → 扫描成本分组）全 batch
    # 只算一次，成本聚类与 batch warmup 共享同一份结果（替代每因子多次
    # ``prepare_run_warmup``）。
    warmup_plan = None
    if auto_warmup:
        from factor_engine.runtime.batch_warmup_plan import compute_batch_warmup_plan

        warmup_plan = compute_batch_warmup_plan(
            analyses,
            dag,
            engine,
            factors=factors,
            auto_warmup=auto_warmup,
            trim_warmup=trim_warmup,
            market=market,
            strict=True,
        )

    # Phase 5 P1-4：warmup 按扫描成本聚类（R39-P0-PERF-015），避免一个
    # full-history 因子把整批 union 窗口拉成全历史（拖累其余短因子）。
    if warmup_clusters and auto_warmup and len(factors) > 1:
        waves = _cluster_factors_by_cost(engine, factors, analyses, plan=warmup_plan)
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
    # batch warmup：跨因子合并共享加载窗口，一次读数，各因子独立 trim。
    engine_to_use, per_windows = _maybe_prepare_batch_warmup(
        engine,
        factors,
        analyses,
        auto_warmup=auto_warmup,
        trim_warmup=trim_warmup,
        market=market,
        plan=warmup_plan,
    )
    run_mode = engine_to_use.run_mode
    # R39-P0-PERF-001：先决定 control plane。adaptive-scheduler 路径**禁止**
    # legacy full-union prefetch（``_maybe_prepare_batch_data``）—— read wave 是
    # 唯一读取控制面（Gate-01：``legacy_union_prefetch_count == 0``）；只有
    # layer-loop 兼容路径保留 union prefetch。
    control_plane = resolve_batch_execution_control_plane(
        factors, analyses, dag, engine=engine, input_dq_check=input_dq_check
    )
    if control_plane == "adaptive_scheduler":
        input_report = None
        legacy_union_prefetch = 0
    else:
        input_report = _maybe_prepare_batch_data(
            engine_to_use,
            dag,
            analyses,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
            input_dq_thresholds=input_dq_thresholds,
        )
        legacy_union_prefetch = 1 if input_report is not None else 0
    source_bars_per_day = _batch_source_bars_per_day(engine_to_use)
    # R31-P0-001：默认主执行链 = AdaptiveBatchScheduler（BatchCompiler →
    # PhysicalPlanner → Scheduler → StreamingSink）。旧 layer-loop 仅保留为
    # compatibility / reference / debug mode（``FACTOR_ENGINE_LAYER_LOOP=1``）。
    if control_plane == "adaptive_scheduler":
        batch_out = _execute_run_many_scheduler(
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
        batch_out["legacy_union_prefetch_count"] = legacy_union_prefetch
        batch_out["control_plane"] = control_plane
        return batch_out

    ctx = engine_to_use._make_context(shared_result_cache={}, perf=perf)
    from factor_engine.runtime.resource_telemetry import record_resource_telemetry

    ctx.runtime_stats = record_resource_telemetry(ctx.runtime_stats)
    from factor_engine.backend.routing_env import routing_execution_scope
    from factor_engine.planner.dependency_graph import build_factor_batch_graph

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
        "control_plane": control_plane,
        "legacy_union_prefetch_count": legacy_union_prefetch,
    }
    _attach_batch_backend_paths(batch_out, backend_paths)
    if input_report is not None:
        batch_out["input_dq"] = input_report.to_dict()
    if len(factors) > 1:
        batch_out["batch_graph"] = batch_graph.to_dict()
    if dag.shared_nodes:
        from factor_engine.planner.rolling_cache import summarize_rolling_cache

        batch_out["rolling_cache"] = summarize_rolling_cache(dag.shared_nodes)
    if dag.roots:
        from factor_engine.backend.operator_cost import estimate_plan_cost

        batch_out["plan_costs"] = {
            fp.factor_name: estimate_plan_cost(fp.root) for fp in dag.roots
        }
        from factor_engine.planner.cost_summary import summarize_plans

        batch_out["cost_summary"] = summarize_plans(
            {fp.factor_name: fp.root for fp in dag.roots}
        )
        from factor_engine.planner.scheduling_hints import derive_scheduling_hints

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
    from factor_engine.backend.path_summary import summarize_lazy_caches

    lazy_cache = summarize_lazy_caches(ctx)
    if lazy_cache:
        batch_out["lazy_cache_summary"] = lazy_cache
    from factor_engine.runtime.resource_telemetry import record_resource_telemetry

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
    from factor_engine.runtime.production_policy import is_production_mode

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
    # R39-P0-PERF-014：warmup 规划只算一次，供 batch warmup 消费。
    warmup_plan = None
    if auto_warmup:
        from factor_engine.runtime.batch_warmup_plan import compute_batch_warmup_plan

        warmup_plan = compute_batch_warmup_plan(
            analyses,
            dag,
            engine,
            factors=factors,
            auto_warmup=auto_warmup,
            trim_warmup=trim_warmup,
            market=market,
            strict=True,
        )
    engine_to_use, per_windows = _maybe_prepare_batch_warmup(
        engine,
        factors,
        analyses,
        auto_warmup=auto_warmup,
        trim_warmup=trim_warmup,
        market=market,
        plan=warmup_plan,
    )
    run_mode = engine_to_use.run_mode
    # iter 是 layer-loop 生成器（无 adaptive scheduler 路径），保留 legacy
    # full-union prefetch（PERF-001 只 gate scheduler 路径）。
    input_report = _maybe_prepare_batch_data(
        engine_to_use,
        dag,
        analyses,
        input_dq_check=input_dq_check,
        input_dq_strict=input_dq_strict,
        input_dq_thresholds=input_dq_thresholds,
    )
    ctx = engine_to_use._make_context(shared_result_cache={}, perf=perf)
    from factor_engine.backend.routing_env import routing_execution_scope
    from factor_engine.planner.dependency_graph import build_factor_batch_graph

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
    from factor_engine.runtime.production_policy import is_production_mode

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
        from factor_engine.runtime.execution_resources import resource_plan
        from factor_engine.runtime.resource_governor import ExecutionResourceScope

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
    # R39-P0-PERF-014：warmup 规划（analysis → RunWindow → 扫描成本分组）只算一次。
    warmup_plan = None
    if auto_warmup:
        from factor_engine.runtime.batch_warmup_plan import compute_batch_warmup_plan

        warmup_plan = compute_batch_warmup_plan(
            analyses,
            dag,
            engine,
            factors=factors,
            auto_warmup=auto_warmup,
            trim_warmup=trim_warmup,
            market=market,
            strict=True,
        )
    engine_to_use, per_windows = _maybe_prepare_batch_warmup(
        engine,
        factors,
        analyses,
        auto_warmup=auto_warmup,
        trim_warmup=trim_warmup,
        market=market,
        plan=warmup_plan,
    )
    run_mode = engine_to_use.run_mode
    # R39-P0-PERF-001：先决定 control plane。adaptive-scheduler 路径**禁止**
    # legacy full-union prefetch（``_maybe_prepare_batch_data``）—— read wave 是
    # 唯一读取控制面（Gate-01：``legacy_union_prefetch_count == 0``）。
    control_plane = resolve_batch_execution_control_plane(
        factors, analyses, dag, engine=engine, input_dq_check=input_dq_check
    )
    if control_plane == "adaptive_scheduler":
        input_report = None
        legacy_union_prefetch = 0
    else:
        input_report = _maybe_prepare_batch_data(
            engine_to_use,
            dag,
            analyses,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
            input_dq_thresholds=input_dq_thresholds,
        )
        legacy_union_prefetch = 1 if input_report is not None else 0
    source_bars_per_day = _batch_source_bars_per_day(engine_to_use)
    # R31-P0-001：production parallel 默认 = AdaptiveBatchScheduler（资源 scope
    # 仍包裹以统一线程/内存设置）。旧 layer-loop 仅 compat/debug（env 显式）。
    if control_plane == "adaptive_scheduler":
        _enter_scope = (lambda: resource_scope.__enter__()) if resource_scope is not None else (lambda: None)
        _exit_scope = (
            (lambda: resource_scope.__exit__(None, None, None))
            if resource_scope is not None
            else (lambda: None)
        )
        _enter_scope()
        try:
            batch_out = _execute_run_many_scheduler(
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
            batch_out["legacy_union_prefetch_count"] = legacy_union_prefetch
            batch_out["control_plane"] = control_plane
            return batch_out
        finally:
            _exit_scope()

    ctx = engine_to_use._make_context(shared_result_cache={}, perf=perf)
    from factor_engine.runtime.resource_telemetry import record_resource_telemetry

    ctx.runtime_stats = record_resource_telemetry(ctx.runtime_stats)
    from factor_engine.backend.routing_env import routing_execution_scope
    from factor_engine.planner.dependency_graph import build_factor_batch_graph

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
        "control_plane": control_plane,
        "legacy_union_prefetch_count": legacy_union_prefetch,
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
    from factor_engine.backend.path_summary import summarize_lazy_caches

    lazy_cache = summarize_lazy_caches(ctx)
    if lazy_cache:
        parallel_out["lazy_cache_summary"] = lazy_cache
    from factor_engine.runtime.resource_telemetry import record_resource_telemetry

    ctx.runtime_stats = record_resource_telemetry(ctx.runtime_stats, finalize=True)
    parallel_out["resource_telemetry"] = dict(ctx.runtime_stats.get("resource") or {})
    return parallel_out
