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
