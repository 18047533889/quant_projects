"""R25 §111 —— Production Startup Gate。

production 启动前：
    registry load → ContractIR audit == 0 blocking → security context valid →
    credential provider valid → legacy home fallback unused → critical calendar
    authoritative → cache permissions safe → source snapshot provider reachable。

任一 critical fail → startup fail。绝不在另一台 server 默默读错误目录。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

logger = logging.getLogger("data_access.startup_gate")


@dataclass
class StartupGateResult:
    passed: bool
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "problems": list(self.problems)}


def _legacy_home_fallback_in_use(store: Any = None) -> list[str]:
    """§53：production 下 legacy ``/home/shw/...`` fallback 未显式配置 → 启动失败。"""
    from data_access.cos.mirror import known_cos_mirror_local_roots

    used: list[str] = []
    for root in known_cos_mirror_local_roots():
        if "/home/shw/" in str(root).lower():
            used.append(str(root))
    return used


def _contract_ir_blocking_problems(store: Any) -> list[str]:
    """ContractIR audit 返回 blocking problems（§23：mirror layout != COS contract 等）。"""
    try:
        ir = store.contract_ir()
        if ir is None:
            return []
        return list(ir.audit() or ())
    except Exception as exc:
        return [f"ContractIR compile failed: {exc}"]


def _security_policy_explicit(store: Any = None) -> list[str]:
    """R26-P0-007/020：production 下必须显式 principal + policy，不能回退 DEFAULT。"""
    from data_access.security.policy import production_security_configured
    from data_access.read.query_budget import is_strict_semantics

    if not is_strict_semantics():
        return []
    if not production_security_configured():
        return [
            "production 未配置显式 principal + policy（DATA_ACCESS_PRINCIPAL_ID + "
            "DATA_ACCESS_ALLOWED_DATASETS）。缺配置 ≠ 本地超级用户（R26-P0-007）。"
        ]
    return []


def _critical_calendar_authoritative(store: Any = None) -> list[str]:
    """R26-P0-020：critical calendar 必须 authoritative（不 COALESCE same-day）。

    R29-P0：不再调错签名的 ``compile_available_from_result(dataset)``（需要
    knowledge/availability），而是**真实加载** ashare/us 日历并对一个边界日期跑
    ``next_trading_day`` + ``next_session_open`` smoke probe——证明「已注入的日历
    真的能推出下一交易日/下一开盘点」，而不是只 import 一下。
    """
    from data_access.read.query_budget import is_strict_semantics

    if not is_strict_semantics():
        return []
    if store is None:
        return []
    from datetime import date, datetime, timedelta

    from data_access.read.session_calendar import compile_available_from

    problems: list[str] = []
    for market in ("ashare", "us"):
        try:
            cal = store.get_calendar(market)
        except Exception as exc:
            problems.append(f"critical calendar {market} 加载失败：{exc}")
            continue
        if cal is None:
            problems.append(
                f"critical calendar {market} 未注入/未加载（production 必须提供真实"
                "交易所日历，禁止 same_day fallback）"
            )
            continue
        if not getattr(cal, "has_data", False):
            problems.append(f"critical calendar {market} 无交易日数据")
            continue
        if getattr(cal, "source", None) == "fallback":
            problems.append(
                f"critical calendar {market} 是 fallback 兜底日历（非权威，禁止 "
                "look-ahead 语义依赖）"
            )
            continue
        # 真实 smoke probe：最近一个周六 → next_trading_day 必须返回交易日。
        today = date.today()
        sat = today - timedelta(days=(today.weekday() - 5) % 7)
        try:
            nxt = cal.next_trading_day(sat)
        except Exception as exc:
            problems.append(f"critical calendar {market} next_trading_day 失败：{exc}")
            continue
        if nxt is None or not cal.is_trading_day(nxt):
            problems.append(
                f"critical calendar {market} smoke probe 失败："
                f"next_trading_day({sat})={nxt}（非交易日）"
            )
            continue
        # next_session_open availability 必须编译成 authoritative（不降级）。
        try:
            res = compile_available_from(
                datetime(nxt.year, nxt.month, nxt.day, 0, 0),
                "next_session_open",
                calendar=cal,
            )
            if getattr(res, "authoritative", True) is False:
                problems.append(
                    f"critical calendar {market} next_session_open 非 authoritative"
                    f"（{getattr(res, 'degradation_reason', 'degraded')}）"
                )
        except Exception as exc:
            problems.append(
                f"critical calendar {market} next_session_open probe 失败：{exc}"
            )
    return problems


def _source_snapshot_provider_available(store: Any = None) -> list[str]:
    """R26-P0-020：需要 remote authoritative source 的 production service，snapshot
    provider 不可用 → 启动失败。

    R29-P0 #199：SourceSnapshotResolver 已是 Store 主读链必选组件——production
    strict 下 resolver 未注入直接 fail（不再是「resolver=None 直接通过」的 no-op）。
    resolver 存在时对 critical dataset 做真实 ``resolve()`` smoke probe（拉 publisher
    manifest / exact object set），而不是只 import 一个异常类型就返回成功。
    """
    from data_access.read.query_budget import is_strict_semantics

    if not is_strict_semantics():
        return []
    if store is None:
        return []
    pipeline = getattr(store, "_pipeline", None)
    resolver = getattr(pipeline, "_resolver", None) if pipeline is not None else None
    if resolver is None:
        return [
            "production 未注入 SourceSnapshotResolver：无法解析权威 source snapshot"
            "（R29-P0 #199，resolver 是主读链必选组件）"
        ]
    # 真实 smoke probe：对关键 generation/remote 数据集走一次 resolve()。
    problems: list[str] = []
    for ds_name in ("factor_matrix", "factor_lake"):
        try:
            store._registry.get(ds_name)
        except Exception:
            continue  # 未注册的部署不强制
        try:
            snap = resolver.resolve(ds_name)
        except Exception as exc:
            problems.append(
                f"source snapshot smoke probe 失败（{ds_name}）："
                f"{type(exc).__name__}: {exc}"
            )
            continue
        if snap is None:
            problems.append(f"source snapshot smoke probe 返回 None（{ds_name}）")
    return problems


def _single_worker_contract(store: Any = None) -> list[str]:
    """R26-P1-017：production workers>1 且无共享 limiter → 启动失败（不只 warning）。"""
    from data_access.read.query_budget import is_strict_semantics

    if not is_strict_semantics():
        return []
    import os

    workers = os.environ.get("DATA_ACCESS_WORKERS", "")
    single = os.environ.get("DATA_ACCESS_SINGLE_WORKER", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if workers and workers.isdigit() and int(workers) > 1 and not single:
        return [
            "DATA_ACCESS_WORKERS>1 且未设 DATA_ACCESS_SINGLE_WORKER=1：进程级 "
            "ResourceGovernor 是 single-worker contract（R26-P1-017，production fail）"
        ]
    return []


def _cache_permissions_ok(store: Any = None) -> list[str]:
    """§28/§111：cache 权限安全（不泄 secret；权限过宽 fail-closed）。"""
    try:
        from data_access.cos.remote import ensure_cache_root_secure

        ensure_cache_root_secure()
        return []
    except Exception as exc:
        return [f"cache permissions unsafe: {exc}"]


def _credential_provider_valid(store: Any = None) -> list[str]:
    """CredentialProvider 可解析（不打印 secret）。"""
    try:
        from data_access.security.credentials import _global_credential_provider

        provider = _global_credential_provider()
        if provider is None:
            return ["no CredentialProvider configured"]
        provider.resolve()
        return []
    except Exception as exc:
        return [f"CredentialProvider invalid: {type(exc).__name__}"]


def run_startup_gate(
    store: Any,
    *,
    production: bool | None = None,
    checks: Sequence[Callable[[], list[str]]] | None = None,
) -> StartupGateResult:
    """运行 production startup gate。

    ``production`` 缺省用 ``is_strict_semantics()``；非 production 只收集不失败
    （research 可跑，但问题会被列出）。
    """
    from data_access.read.query_budget import is_strict_semantics

    effective_production = (
        production if production is not None else is_strict_semantics()
    )
    problems: list[str] = []
    runner = checks if checks is not None else (
        _contract_ir_blocking_problems,
        _credential_provider_valid,
        _legacy_home_fallback_in_use,
        _cache_permissions_ok,
        _security_policy_explicit,
        _critical_calendar_authoritative,
        _source_snapshot_provider_available,
        _single_worker_contract,
    )
    for check in runner:
        try:
            problems.extend(check(store))
        except Exception as exc:  # pragma: no cover
            problems.append(f"{getattr(check, '__name__', 'check')}: {exc}")

    if effective_production:
        # production：任一 critical fail → startup fail。
        if problems:
            raise RuntimeError(
                "Production startup gate failed:\n  " + "\n  ".join(problems)
            )
        return StartupGateResult(passed=True)
    # research：收集问题不失败。
    for p in problems:
        logger.warning("startup gate (research, non-blocking): %s", p)
    return StartupGateResult(passed=True, problems=problems)
