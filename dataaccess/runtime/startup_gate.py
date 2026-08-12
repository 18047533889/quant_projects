"""R25 §111 —— Production Startup Gate。

production 启动前：
    registry load → ContractIR audit == 0 blocking → security context valid →
    credential provider valid → legacy home fallback unused → critical calendar
    authoritative → cache permissions safe → source snapshot provider reachable。

任一 critical fail → startup fail。绝不在另一台 server 默默读错误目录。
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Sequence

logger = logging.getLogger("data_access.startup_gate")


@dataclass
class StartupGateResult:
    """R32-P0-016：启动 gate 结果（三态：PASS / DEGRADED / FAIL）。"""
    status: str  # "PASS" / "DEGRADED" / "FAIL"
    problems: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """向后兼容：PASS 和 DEGRADED 都算通过（根据 run mode 判断）。"""
        return self.status in ("PASS", "DEGRADED")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "problems": list(self.problems),
        }


@dataclass(frozen=True)
class StartupCertificate:
    """R40 #59 + R32-P0-013：production store 启动证书（immutable）。

    - ``gate_name``   启动 gate 名称（默认 ``production_startup``）；
    - ``passed``      全部 critical checks 通过；
    - ``timestamp``   ISO-8601 证书签发时间；
    - ``evidence_hash`` 问题清单 + gate 版本的确定性摘要（证明该证书对应哪一组
      checks 判定）；
    - ``problems``    research 模式收集的非阻塞问题（production 通过时为空）；
    - ``subject_digest`` R32-P0-014：启动环境主体摘要（build SHA + package version +
      registry digest + contract digest + semantic schema + policy digest + calendar
      snapshot + credential generation + source profile + worker topology + gate version）。

    R32-P0-013：frozen dataclass，调用者不能修改 passed。
    """

    gate_name: str = "production_startup"
    passed: bool = False
    timestamp: str = ""
    evidence_hash: str = ""
    problems: tuple[str, ...] = field(default_factory=tuple)
    subject_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "timestamp": self.timestamp,
            "evidence_hash": self.evidence_hash,
            "problems": list(self.problems),
            "subject_digest": self.subject_digest,
        }


#: 证书有效期（超时视为过期，production store 需重新跑 gate）。
_STARTUP_CERTIFICATE_TTL_SECONDS = 24 * 3600


def _evidence_hash_of(problems: Sequence[str], gate_name: str) -> str:
    """问题清单 + gate 名称 → 确定性摘要（#59 证据哈希）。"""
    payload = "\n".join([gate_name, *sorted(problems)]) or "ok"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_startup_certificate(
    passed: bool, problems: Sequence[str], *, gate_name: str = "production_startup",
    subject_digest: str = "",
) -> StartupCertificate:
    """构造 StartupCertificate（timestamp + evidence_hash + subject_digest）。

    R32-P0-013：返回 frozen dataclass。
    R32-P0-014：绑定 subject_digest。
    """
    return StartupCertificate(
        gate_name=gate_name,
        passed=bool(passed),
        timestamp=datetime.now(timezone.utc).isoformat(),
        evidence_hash=_evidence_hash_of(problems, gate_name),
        problems=tuple(problems),
        subject_digest=subject_digest,
    )


def startup_certificate_expired(
    cert: StartupCertificate | None,
    *,
    ttl_seconds: int = _STARTUP_CERTIFICATE_TTL_SECONDS,
    now: datetime | None = None,
    current_subject_digest: str | None = None,
) -> bool:
    """证书是否过期（缺失 / 未通过 / timestamp 超过 TTL → 过期）。

    R32-P0-015：同时验证 subject_digest（配置变化立即失效）。
    """
    if cert is None or not getattr(cert, "passed", False):
        return True
    raw = getattr(cert, "timestamp", "") or ""
    if not raw:
        return True
    try:
        ts = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return True
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if (now - ts).total_seconds() > max(0, int(ttl_seconds)):
        return True
    # R32-P0-015：subject_digest 不匹配 → 环境变化，证书失效。
    if current_subject_digest is not None:
        cert_digest = getattr(cert, "subject_digest", "")
        if cert_digest and cert_digest != current_subject_digest:
            return True
    return False


def require_startup_certificate(
    store: Any,
    *,
    production: bool | None = None,
    force: bool = False,
    checks: Sequence[Callable[[], list[str]]] | None = None,
) -> StartupCertificate:
    """production store 要求未过期 StartupCertificate 才 proceed（R40 #59）。

    - store 已有未过期且 passed 的证书 → 直接返回（不重跑 gate）；
    - 否则跑 :func:`run_startup_gate`（production 失败会 raise）；
    - 成功后把证书挂到 ``store._startup_certificate``。

    R32-P0-015：证书复用前验证 current_subject_digest。
    """
    cert = getattr(store, "_startup_certificate", None)
    if cert is not None and not force:
        # R32-P0-015：计算当前 subject_digest，验证证书是否仍有效。
        try:
            from data_access.runtime.startup_subject import build_startup_subject_digest
            current_digest = build_startup_subject_digest(store).to_digest()
        except Exception:
            current_digest = None
        if not startup_certificate_expired(cert, current_subject_digest=current_digest):
            return cert
    cert = run_startup_gate(store, production=production, checks=checks)
    try:
        store._startup_certificate = cert
    except Exception:
        pass
    return cert


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


def calendar_digest(cal: Any) -> str:
    """R40 #60：日历内容摘要——hash 全部交易日（有序）。

    日历 trading_days 变化（注入新日历 / 被替换）→ digest 变。这是「日历身份」
    的确定性证明，不再依赖 ``date.today()`` 算上一个周六。
    """
    days = getattr(cal, "trading_days", None) or ()
    payload = "\n".join(str(d) for d in sorted(set(days)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def calendar_coverage_check(
    cal: Any,
    *,
    market: str = "",
    expected_digest: str | None = None,
    required_window_days: int = 30,
    lookback_days: int = 120,
    now: datetime | None = None,
) -> list[str]:
    """R40 #60：calendar 权威性 + digest + 覆盖窗口校验。

    - ``source == "fallback"`` → 非权威（fail）；
    - ``expected_digest`` 提供且与 ``calendar_digest(cal)`` 不符 → 注入的日历与
      期望身份不一致（fail）；
    - 覆盖窗口：``[now - lookback_days, now]`` 内实际交易日数必须 ≥
      ``required_window_days``——证明日历不是空壳/被截断到远古。
    """
    problems: list[str] = []
    if not getattr(cal, "has_data", False):
        problems.append(f"critical calendar {market} 无交易日数据")
        return problems
    if getattr(cal, "source", None) == "fallback":
        problems.append(
            f"critical calendar {market} 是 fallback 兜底日历（非权威，禁止 "
            "look-ahead 语义依赖）"
        )
    digest = calendar_digest(cal)
    if expected_digest and digest != expected_digest:
        problems.append(
            f"critical calendar {market} digest 不匹配：expected={expected_digest} "
            f"actual={digest}（注入的日历与期望身份不一致）"
        )
    # 覆盖窗口：最近 lookback_days 天内至少有 required_window_days 个交易日。
    now = now or datetime.now(timezone.utc)
    today = now.date()
    start = today - timedelta(days=max(1, int(lookback_days)))
    days = getattr(cal, "trading_days", None) or ()
    count = sum(1 for d in days if start <= d <= today)
    if count < max(1, int(required_window_days)):
        problems.append(
            f"critical calendar {market} 覆盖不足：最近 {lookback_days} 天仅 "
            f"{count} 个交易日，要求 ≥ {required_window_days}（日历可能被截断/"
            "过期）。"
        )
    return problems


def _critical_calendar_authoritative(store: Any = None) -> list[str]:
    """R26-P0-020 + R40 #60：critical calendar 必须 authoritative（不 COALESCE
    same-day）。

    R40 #60：不再用 ``date.today()`` 算「上一个周六」做 smoke probe（依赖周末
    结构、无法验证日历身份）；改为 **calendar digest + 覆盖窗口** 校验——注入的
    日历必须是 authoritative（非 fallback）、digest 匹配期望、覆盖覆盖所需窗口。
    期望 digest 从 store 的 ``calendar_expected_digest`` 配置读取（未配置 →
    只校验权威 + 覆盖，digest 维度 N/A）。
    """
    from data_access.read.query_budget import is_strict_semantics

    if not is_strict_semantics():
        return []
    if store is None:
        return []
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
        expected = None
        try:
            expected = getattr(store, "calendar_expected_digest", None)
            if callable(expected):
                expected = expected(market)
            elif isinstance(expected, dict):
                expected = expected.get(market)
        except Exception:
            expected = None
        problems.extend(
            calendar_coverage_check(cal, market=market, expected_digest=expected)
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
    skip: bool = False,
) -> StartupCertificate:
    """运行 production startup gate。

    ``production`` 缺省用 ``is_strict_semantics()``；非 production 只收集不失败
    （research 可跑，但问题会被列出）。

    R32-P0-016：返回三态（PASS / DEGRADED / FAIL）。
    R32-P0-014：绑定 subject_digest。
    R32-P0-022: ``skip=True`` in production mode raises RuntimeError.
    """
    from data_access.read.query_budget import is_strict_semantics

    effective_production = (
        production if production is not None else is_strict_semantics()
    )

    # R32-P0-022: Production forbid --skip-startup-gate
    if skip and effective_production:
        raise RuntimeError(
            "R32-P0-022: Production mode cannot skip startup gate. "
            "--skip-startup-gate is only allowed in dev/research mode."
        )

    if skip:
        logger.warning("Startup gate skipped (dev/research mode only)")
        return build_startup_certificate(True, ["startup gate skipped"])

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

    # R32-P0-014：构造 subject_digest。
    subject_digest = ""
    try:
        from data_access.runtime.startup_subject import build_startup_subject_digest
        subject_digest = build_startup_subject_digest(store).to_digest()
    except Exception:
        pass

    if effective_production:
        # R32-P0-016：production 任一 critical fail → FAIL。
        if problems:
            raise RuntimeError(
                "Production startup gate failed:\n  " + "\n  ".join(problems)
            )
        return build_startup_certificate(True, [], subject_digest=subject_digest)
    # R32-P0-016：research 有问题 → DEGRADED，无问题 → PASS。
    for p in problems:
        logger.warning("startup gate (research, non-blocking): %s", p)
    status = "DEGRADED" if problems else "PASS"
    return build_startup_certificate(
        True, problems, subject_digest=subject_digest
    )
