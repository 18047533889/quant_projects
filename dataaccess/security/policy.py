# -*- coding: utf-8 -*-
"""DatasetAuthorizer + 默认授权器（R24 P0-S2 §4 / §26）。

## 职责

在 backend 选择之前执行逻辑授权：

    authorize_dataset(principal, dataset, action="dataset:read")

任何一层 deny 都必须 deny，且 **AccessDenied 不得 fallback 到更高身份**
（§4「AccessDenied 不能 fallback」）：受限 httpfs credential 403 → 不准
读 ~/.cos.yaml 换 credential、不准换 profile、直接抛 ``AuthorizationError``。

可以 backend fallback 的只有能力类错误（httpfs extension unavailable、
S3 API capability unsupported、endpoint incompatibility）。
"""
from __future__ import annotations

import os
import threading
from typing import Protocol, runtime_checkable

from data_access.core.exceptions import AccessDeniedError
from data_access.security.principal import (
    ACTION_DATASET_LIST,
    ACTION_DATASET_READ,
    DATASET_SCOPED_ACTIONS,
    AccessPolicy,
    DataPrincipal,
    DEFAULT_ACCESS_POLICY,
    DEFAULT_LOCAL_PRINCIPAL,
)


@runtime_checkable
class DatasetAuthorizer(Protocol):
    """逻辑授权器（§26 接口）。``authorize`` 拒绝时抛 AuthorizationError。"""

    def authorize(
        self,
        principal: DataPrincipal,
        dataset: str,
        action: str = ACTION_DATASET_READ,
    ) -> None:
        """校验 (principal, dataset, action)；拒绝抛 AuthorizationError。"""
        ...


class DefaultAuthorizer:
    """基于 AccessPolicy 的最小授权器。

    未配置 policy 时回退 ``DEFAULT_ACCESS_POLICY``（放行已注册 dataset，拒绝
    uri:read）。production/strict 下「既未配置 principal 也未配置 policy」也
    fail-closed 拒绝——不能把「没配置」当成「全放行」。
    """

    def __init__(
        self,
        *,
        policy: AccessPolicy | None = None,
        principal: DataPrincipal | None = None,
        strict_default_deny: bool | None = None,
    ) -> None:
        self._policy = policy or DEFAULT_ACCESS_POLICY
        self._principal = principal or DEFAULT_LOCAL_PRINCIPAL
        # None = 每次调用按当前 strict 语义实时判定（避免进程级缓存把某个测试/
        # 会话的 production 状态永久烘焙进授权器）。
        self._strict_default_deny = strict_default_deny

    def _is_strict(self) -> bool:
        if self._strict_default_deny is not None:
            return self._strict_default_deny
        return _is_strict_semantics()

    @property
    def policy(self) -> AccessPolicy:
        return self._policy

    @property
    def principal(self) -> DataPrincipal:
        return self._principal

    def authorize(
        self,
        principal: DataPrincipal,
        dataset: str,
        action: str = ACTION_DATASET_READ,
    ) -> None:
        principal = principal or self._principal
        if not self._policy.allows_action(action):
            self._raise(principal, dataset, action)
        # dataset:list 只校验 action 权限（不涉及具体数据集）。
        if action == ACTION_DATASET_LIST:
            return
        if action not in DATASET_SCOPED_ACTIONS:
            self._raise(principal, dataset, action)
        if not self._policy.allows_dataset(dataset):
            self._raise(principal, dataset, action)
        # uri:read 需要显式 allow_uri_read（§7 / §24 production/strict 默认拒绝；
        # research 保持旧 dev 行为，仍受 path whitelist 约束）。
        if action == "uri:read" and not self._policy.allow_uri_read and self._is_strict():
            self._raise(principal, dataset, action)

    def _raise(self, principal: DataPrincipal, dataset: str, action: str) -> None:
        # 外部错误信息必须脱敏（P1-S6 §8）：不输出 allowed list / 完整路径。
        # 内部安全日志（audit）可另记 principal/dataset/policy_decision。
        raise AccessDeniedError(
            f"resource is not authorized "
            f"(principal={principal.principal_id}, action={action})"
        )


# ---- 全局（store / HTTP 共享）----

_global_lock = threading.Lock()
_global_authorizer: DatasetAuthorizer | None = None


def set_authorizer(authorizer: DatasetAuthorizer | None) -> None:
    global _global_authorizer
    with _global_lock:
        _global_authorizer = authorizer


def get_authorizer() -> DatasetAuthorizer:
    """进程内共享授权器（store 入口 / HTTP scopes 消费同一份）。"""
    global _global_authorizer
    if _global_authorizer is None:
        with _global_lock:
            if _global_authorizer is None:
                _global_authorizer = _authorizer_from_env()
    return _global_authorizer


def _authorizer_from_env() -> DefaultAuthorizer:
    """从 env 构造默认授权器（server 级策略，§27 多服务器部署）。

    - ``DATA_ACCESS_PRINCIPAL_ID``：当前 server principal（缺省 "local"）
    - ``DATA_ACCESS_ALLOWED_DATASETS``：逗号分隔的允许数据集（缺省全部）
    - ``DATA_ACCESS_ALLOW_URI_READ``：是否放行 uri:read（production 默认 False）
    """
    principal = _principal_from_env()
    allowed_raw = os.environ.get("DATA_ACCESS_ALLOWED_DATASETS", "").strip()
    allow_uri = (
        os.environ.get("DATA_ACCESS_ALLOW_URI_READ", "").lower()
        in {"1", "true", "yes", "on"}
    )
    if allowed_raw:
        allowed = frozenset(p.strip() for p in allowed_raw.split(",") if p.strip())
        policy = AccessPolicy(
            allowed_datasets=allowed,
            allowed_actions=frozenset(DATASET_SCOPED_ACTIONS),
            allow_uri_read=allow_uri,
        )
        return DefaultAuthorizer(policy=policy, principal=principal)
    return DefaultAuthorizer(policy=DEFAULT_ACCESS_POLICY, principal=principal)


def _principal_from_env() -> DataPrincipal:
    pid = os.environ.get("DATA_ACCESS_PRINCIPAL_ID", "").strip()
    if not pid:
        return DEFAULT_LOCAL_PRINCIPAL
    server = os.environ.get("DATA_ACCESS_SERVER_ID", "").strip() or None
    roles = tuple(
        r.strip()
        for r in os.environ.get("DATA_ACCESS_PRINCIPAL_ROLES", "").split(",")
        if r.strip()
    )
    return DataPrincipal(principal_id=pid, roles=roles, server_id=server)


def _is_strict_semantics() -> bool:
    try:
        from data_access.read.query_budget import is_strict_semantics

        return is_strict_semantics()
    except Exception:
        return True


__all__ = [
    "DatasetAuthorizer",
    "DefaultAuthorizer",
    "set_authorizer",
    "get_authorizer",
]
