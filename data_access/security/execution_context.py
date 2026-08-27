"""R26-P0-005 —— request-scoped 嵌套执行上下文（ContextVar）。

HTTP 顶层解析 ``X-API-Key → DataPrincipal → AccessPolicy`` 后，把
principal / authorizer / credential_provider / request_id / run_mode /
security_digest 放进 ``DataAccessExecutionContext``，用 ``execution_scope``
context manager 写入 ContextVar。Store 的 **所有嵌套读**（calendar / universe /
factor catalog / factor tags / dependent datasets / remote credential /
cache scope / audit / lineage）都通过 ``current_*`` 读取同一上下文，
自动继承请求级授权，杜绝：

    - 顶层授权 dataset A，但内部嵌套读仍用 process principal；
    - 并发 API key A/B 之间身份串扰（ContextVar 是任务/线程隔离的）。

**绝不**在每个 request 里 ``set_authorizer()`` 改全局状态（P0-005 §「不要每个
request 改全局 set_authorizer」）。
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from data_access.core.exceptions import ValidationError

_execution_ctx_var: ContextVar["DataAccessExecutionContext | None"] = ContextVar(
    "data_access_execution_ctx", default=None
)


@dataclass(frozen=True)
class DataAccessExecutionContext:
    """一个请求的统一执行上下文（R26-P0-005）。

    任一字段为 None 表示「未指定」，消费方回退到 process 级默认。
    ``security_digest`` 由 ``digest()`` 计算（principal + policy + run_mode），
    GovernedFrame 用它校验 frame 是否来自当前安全上下文（P0-023）。
    """

    principal: Any = None
    authorizer: Any = None
    access_policy: Any = None
    credential_provider: Any = None
    request_id: str | None = None
    run_mode: Any = None
    security_digest: str | None = None
    # P0-6：principal 类型（HUMAN / TEAM / SERVICE）。None = 未指定（回退 process 级）。
    # 只有 SERVICE（后台任务）允许回退到 server 的 global/env/service credential 链；
    # HUMAN/TEAM 无 principal-scoped credential 时 fail-closed（见 cos.remote）。
    principal_type: str | None = None

    def __post_init__(self) -> None:
        if self.security_digest is not None and not isinstance(self.security_digest, str):
            raise ValidationError("security_digest 必须是字符串")

    def digest(self) -> str:
        """计算当前上下文的 security digest（principal + policy + run_mode）。"""
        payload = {
            "principal_id": (
                getattr(self.principal, "principal_id", None)
                if self.principal is not None
                else None
            ),
            "policy_digest": (
                getattr(self.access_policy, "digest", None)()
                if self.access_policy is not None
                and callable(getattr(self.access_policy, "digest", None))
                else None
            ),
            "run_mode": (
                getattr(self.run_mode, "value", None)
                if self.run_mode is not None
                else None
            ),
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

    def to_dict(self) -> dict[str, Any]:
        return {
            "principal_id": (
                getattr(self.principal, "principal_id", None)
                if self.principal is not None
                else None
            ),
            "request_id": self.request_id,
            "run_mode": getattr(self.run_mode, "value", None) if self.run_mode else None,
            "security_digest": self.security_digest or self.digest(),
        }


def current_execution_context() -> "DataAccessExecutionContext | None":
    """当前线程/任务的执行上下文（无则 None → 回退 process 级）。"""
    return _execution_ctx_var.get()


def current_principal() -> Any:
    """当前请求 principal；无上下文时 None（调用方回退 store 级）。"""
    ctx = _execution_ctx_var.get()
    return ctx.principal if ctx is not None else None


def current_authorizer() -> Any:
    """当前请求 authorizer；无上下文时 None（调用方回退 store 级）。"""
    ctx = _execution_ctx_var.get()
    return ctx.authorizer if ctx is not None else None


def current_credential_provider() -> Any:
    ctx = _execution_ctx_var.get()
    return ctx.credential_provider if ctx is not None else None


def current_principal_type() -> str | None:
    """当前请求 principal 类型（HUMAN / TEAM / SERVICE）；无上下文时 None。"""
    ctx = _execution_ctx_var.get()
    if ctx is None:
        return None
    return ctx.principal_type


def current_security_digest() -> str | None:
    ctx = _execution_ctx_var.get()
    if ctx is None:
        return None
    return ctx.security_digest or ctx.digest()


@contextmanager
def execution_scope(ctx: "DataAccessExecutionContext | None"):
    """进入请求作用域；退出时恢复上级上下文（嵌套自动继承）。"""
    token = _execution_ctx_var.set(ctx)
    try:
        yield
    finally:
        _execution_ctx_var.reset(token)


def resolve_execution_context(
    *,
    principal: Any = None,
    authorizer: Any = None,
    access_policy: Any = None,
    credential_provider: Any = None,
    request_id: str | None = None,
    run_mode: Any = None,
    principal_type: str | None = None,
) -> DataAccessExecutionContext:
    """构造一个执行上下文，缺省字段从当前上下文/环境回退。

    优先级：显式参数 > 当前嵌套上下文 > 进程级（authorizer/principal）。
    """
    outer = _execution_ctx_var.get()
    if access_policy is None:
        access_policy = (
            getattr(authorizer, "policy", None)
            if authorizer is not None
            else getattr(outer, "access_policy", None) if outer else None
        )
    if principal is None:
        principal = getattr(outer, "principal", None) if outer else None
    if authorizer is None:
        authorizer = getattr(outer, "authorizer", None) if outer else None
    if credential_provider is None:
        credential_provider = (
            getattr(outer, "credential_provider", None) if outer else None
        )
    if run_mode is None:
        run_mode = getattr(outer, "run_mode", None) if outer else None
    if principal_type is None:
        principal_type = getattr(outer, "principal_type", None) if outer else None
    ctx = DataAccessExecutionContext(
        principal=principal,
        authorizer=authorizer,
        access_policy=access_policy,
        credential_provider=credential_provider,
        request_id=request_id,
        run_mode=run_mode,
        principal_type=principal_type,
    )
    return DataAccessExecutionContext(
        principal=ctx.principal,
        authorizer=ctx.authorizer,
        access_policy=ctx.access_policy,
        credential_provider=ctx.credential_provider,
        request_id=ctx.request_id,
        run_mode=ctx.run_mode,
        principal_type=ctx.principal_type,
        security_digest=ctx.digest(),
    )


__all__ = [
    "DataAccessExecutionContext",
    "execution_scope",
    "current_execution_context",
    "current_principal",
    "current_authorizer",
    "current_credential_provider",
    "current_principal_type",
    "current_security_digest",
    "resolve_execution_context",
]
