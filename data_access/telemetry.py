"""
data_access.telemetry —— per-operator 查询计数 + 慢查询告警

职责
    1. 每次 execute_arrow / execute_df 调用时计一次（按 operator + op 类型）
    2. 查询耗时超阈值时 WARNING 日志（不阻塞业务）
    3. 每个 operator 的累计查询次数超过配额时 WARNING 日志（也不阻塞）

设计要点
    - 计数器是内存里的 dict，按当前进程生命周期计（不跨进程聚合）；
      跨进程聚合请消费 data_access_audit.jsonl（已记录 read/write/publish）。
    - 阈值从环境变量读，有合理兜底：
        QUANT_SLOW_QUERY_MS        慢查询阈值（毫秒），默认 5000
        QUANT_OPERATOR_QUERY_QUOTA 进程内单 operator 查询次数软上限，默认 10000
      「软上限」意味着超了只日志不抛异常——原因：
        a) 批回测 / 因子研究动不动几万次查询，硬上限会误伤
        b) 真正恶意的查询应该在 SQL 层 (store.sql) 禁词拦下来
        c) 日志 + 告警是可观测手段，业务侧按需关停
    - 对慢查询：只记 `SQL 前 200 字符` 到 logging，不暴露 params 值；
      需要调查时走 audit_log 或 DuckDB 的 EXPLAIN。

非职责
    - 不写文件（审计已经写到 data_access_audit.jsonl）
    - 不抛异常（阻塞查询不是 telemetry 的事）
    - 不做 rate-limiting（没有共识该怎么 throttle）

维护人：quant 基础平台组    最后更新：2026-04-20
"""

from __future__ import annotations

import logging
import os
import threading
from collections import defaultdict
from dataclasses import dataclass, field

from .namespace import resolve_operator


logger = logging.getLogger("data_access.telemetry")

# 默认阈值；覆盖见上面 docstring
_DEFAULT_SLOW_QUERY_MS = 5000.0
_DEFAULT_OPERATOR_QUOTA = 10000


def _slow_query_ms() -> float:
    raw = os.environ.get("QUANT_SLOW_QUERY_MS")
    if not raw:
        return _DEFAULT_SLOW_QUERY_MS
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_SLOW_QUERY_MS
    return value if value > 0 else _DEFAULT_SLOW_QUERY_MS


def _operator_quota() -> int:
    raw = os.environ.get("QUANT_OPERATOR_QUERY_QUOTA")
    if not raw:
        return _DEFAULT_OPERATOR_QUOTA
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_OPERATOR_QUOTA
    return value if value > 0 else _DEFAULT_OPERATOR_QUOTA


def _explain_slow_query_enabled() -> bool:
    return os.environ.get("QUANT_EXPLAIN_SLOW_QUERY", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _explain_slow_query_ms() -> float:
    raw = os.environ.get("QUANT_EXPLAIN_QUERY_MS")
    if not raw:
        return _slow_query_ms()
    try:
        value = float(raw)
    except ValueError:
        return _slow_query_ms()
    return value if value > 0 else _slow_query_ms()


def _profile_slow_query_enabled() -> bool:
    return os.environ.get("QUANT_PROFILE_SLOW_QUERY", "").lower() in {
        "1",
        "true",
        "yes",
    }


def maybe_log_slow_query_plan(
    engine: Any,
    *,
    elapsed_ms: float,
    sql: str,
    params: Any,
    op: str,
    audit_dataset: str | None = None,
) -> None:
    """慢查询可选 EXPLAIN；默认关闭（QUANT_EXPLAIN_SLOW_QUERY=1 启用）。"""
    if not _explain_slow_query_enabled():
        return
    if elapsed_ms < _explain_slow_query_ms():
        return
    try:
        explain_fn = getattr(engine, "explain", None)
        if not callable(explain_fn):
            return
        plan = explain_fn(sql, params)
        operator = resolve_operator()
        logger.warning(
            "慢查询 EXPLAIN operator=%s op=%s elapsed_ms=%.1f plan=%s",
            operator,
            op,
            elapsed_ms,
            (plan[:2000] + "…") if len(plan) > 2000 else plan,
        )
        if audit_dataset:
            from . import audit

            audit.record(
                op="explain",
                dataset=audit_dataset,
                ok=True,
                elapsed_ms=elapsed_ms,
                extra={"sql_op": op, "plan": plan[:4000]},
            )
        if _profile_slow_query_enabled():
            profile_fn = getattr(engine, "profile_analyze", None)
            if callable(profile_fn):
                profile = profile_fn(sql, params)
                logger.warning(
                    "慢查询 PROFILE operator=%s op=%s elapsed_ms=%.1f profile=%s",
                    operator,
                    op,
                    elapsed_ms,
                    (profile[:2000] + "…") if len(profile) > 2000 else profile,
                )
                if audit_dataset:
                    from . import audit

                    audit.record(
                        op="profile",
                        dataset=audit_dataset,
                        ok=True,
                        elapsed_ms=elapsed_ms,
                        extra={"sql_op": op, "profile": profile[:8000]},
                    )
    except Exception:  # noqa: BLE001 — 诊断失败不能影响主路径
        pass


@dataclass
class OperatorCounters:
    """单 operator 的累计指标；进程生命周期计数。"""

    total_queries: int = 0
    slow_queries: int = 0
    total_elapsed_ms: float = 0.0
    # 防止同一分位多次重复告警：已经触发过 over-quota 告警的标记位
    quota_alert_fired: bool = False
    # 80% 告警用：防止刷屏
    quota_warning_fired: bool = False


# 关键设计：一把 lock 保护所有 operator 的 counters。粒度粗但查询路径上只改 int，
# 锁持有时间 < 1us，压测量级（10k QPS）下可忽略。
_counters_lock = threading.Lock()
_counters: dict[str, OperatorCounters] = defaultdict(OperatorCounters)


def record_polars_scan(
    *,
    dataset: str,
    elapsed_ms: float,
    paths_count: int = 0,
) -> None:
    """scan_polars 惰性扫描建图完成时调一下（不含下游 .collect() 耗时）。"""
    record_query(
        elapsed_ms=elapsed_ms,
        sql=f"polars_scan dataset={dataset} paths={paths_count}",
        op="polars_scan",
    )


def record_query(
    *,
    elapsed_ms: float,
    sql: str,
    op: str = "query",
) -> None:
    """execute_arrow / execute_df / scan_polars 每次调用时调一下。

    线程安全，不抛异常（telemetry 失败不能影响主路径）。
    """
    try:
        operator = resolve_operator()
        slow_threshold = _slow_query_ms()
        quota = _operator_quota()

        with _counters_lock:
            counters = _counters[operator]
            counters.total_queries += 1
            counters.total_elapsed_ms += elapsed_ms
            is_slow = elapsed_ms >= slow_threshold
            if is_slow:
                counters.slow_queries += 1

            # 查询配额告警：80% 和 100% 各一次
            if (
                not counters.quota_warning_fired
                and counters.total_queries >= int(quota * 0.8)
                and counters.total_queries < quota
            ):
                counters.quota_warning_fired = True
                should_warn_quota = True
                warn_stage = "80%"
            elif not counters.quota_alert_fired and counters.total_queries >= quota:
                counters.quota_alert_fired = True
                should_warn_quota = True
                warn_stage = "100%"
            else:
                should_warn_quota = False
                warn_stage = ""

            total_queries_snapshot = counters.total_queries

        # 锁外打日志，避免 I/O 持锁
        if is_slow:
            logger.warning(
                "慢查询 operator=%s op=%s elapsed_ms=%.1f threshold_ms=%.1f sql=%r",
                operator,
                op,
                elapsed_ms,
                slow_threshold,
                sql[:200],
            )
        if should_warn_quota:
            logger.warning(
                "查询配额 %s operator=%s total_queries=%d quota=%d "
                "（软上限，只告警不阻塞；超配额可能意味着循环读或未命中缓存）",
                warn_stage,
                operator,
                total_queries_snapshot,
                quota,
            )
    except Exception:  # noqa: BLE001 — telemetry 不能把主查询搞挂
        # 不用 logger 打 exc_info，避免 logger 本身挂掉又触发递归
        pass


def get_counters_snapshot() -> dict[str, OperatorCounters]:
    """拿当前进程的计数器快照。主要给测试 / 调试 / 监控拉取点用。"""
    with _counters_lock:
        # 浅拷贝 dataclass 值；OperatorCounters 是个简单 dataclass，field=default 不依赖外部引用
        return {
            op: OperatorCounters(
                total_queries=c.total_queries,
                slow_queries=c.slow_queries,
                total_elapsed_ms=c.total_elapsed_ms,
                quota_alert_fired=c.quota_alert_fired,
                quota_warning_fired=c.quota_warning_fired,
            )
            for op, c in _counters.items()
        }


def reset_counters() -> None:
    """主要给测试用。重置所有 operator 的计数器。"""
    with _counters_lock:
        _counters.clear()
