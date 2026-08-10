# -*- coding: utf-8 -*-
"""HTTP API key → principal 映射（R24 P0-S5 §7）。

不能一把 API Key = 整台服务器权限。``api_key hash -> principal_id -> roles ->
dataset scopes -> factor scopes`` 由部署配置文件/环境注入，不要求上 OAuth。

配置来源：
    ``DATA_ACCESS_API_PRINCIPALS``：JSON，格式
    {
      "<api_key_hash_sha256>": {
        "principal_id": "server-a",
        "server_id": "server-a",
        "allowed_datasets": ["ashare_stock_daily", ...],
        "allowed_factor_namespaces": [],
        "allow_uri_read": false,
        "allow_metadata_sensitive": false
      },
      ...
    }

    key 全量 hash 用 ``hash_api_key(key)``（sha256 hex）。未匹配的 key → 拒绝。
    未配置映射时（开发）→ 默认 principal（全部 dataset、无 uri:read）。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any

from data_access.security.principal import (
    ACTION_FACTOR_METADATA_SENSITIVE,
    ACTION_URI_READ,
    ALL_ACTIONS,
    AccessPolicy,
    DataPrincipal,
)


def hash_api_key(api_key: str) -> str:
    """key 全量 hash（sha256 hex）——存储/比较只留 hash，不留明文。"""
    return hashlib.sha256(str(api_key).encode("utf-8")).hexdigest()


class ApiPrincipalRegistry:
    """api key hash → (DataPrincipal, AccessPolicy) 的只读 registry。"""

    def __init__(self, principals: dict[str, dict[str, Any]] | None = None) -> None:
        self._by_hash: dict[str, dict[str, Any]] = dict(principals or {})

    @classmethod
    def from_env(cls, raw: str | None = None) -> "ApiPrincipalRegistry":
        if raw is None:
            raw = os.environ.get("DATA_ACCESS_API_PRINCIPALS", "").strip()
        if not raw:
            return cls()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"DATA_ACCESS_API_PRINCIPALS 不是合法 JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError("DATA_ACCESS_API_PRINCIPALS 顶层必须是 mapping")
        return cls(payload)

    def resolve(
        self, api_key: str
    ) -> tuple[DataPrincipal, AccessPolicy]:
        """按 api key 解析 (principal, policy)。未命中 → 拒绝（返回 None, None）。"""
        key_hash = hash_api_key(api_key)
        entry = self._by_hash.get(key_hash)
        if entry is None:
            return None, None
        allowed = frozenset(
            str(x) for x in (entry.get("allowed_datasets") or []) if x
        )
        namespaces = frozenset(
            str(x) for x in (entry.get("allowed_factor_namespaces") or []) if x
        )
        allow_uri = bool(entry.get("allow_uri_read"))
        allow_sensitive = bool(entry.get("allow_metadata_sensitive"))
        actions = set(ALL_ACTIONS)
        if not allow_uri:
            actions.discard(ACTION_URI_READ)
        if not allow_sensitive:
            actions.discard(ACTION_FACTOR_METADATA_SENSITIVE)
        policy = AccessPolicy(
            allowed_datasets=allowed,
            allowed_factor_namespaces=namespaces,
            allowed_actions=frozenset(actions),
            allow_uri_read=allow_uri,
        )
        principal = DataPrincipal(
            principal_id=str(entry.get("principal_id") or "unknown"),
            roles=tuple(str(r) for r in (entry.get("roles") or []) if r),
            server_id=str(entry.get("server_id") or "").strip() or None,
        )
        return principal, policy

    def visible_datasets(self, api_key: str, all_datasets: list[str]) -> list[str]:
        """按 principal 可见性过滤数据集列表（T-S09）。"""
        principal, policy = self.resolve(api_key)
        if policy is None:
            return []
        if not policy.allowed_datasets:
            return list(all_datasets)
        return [d for d in all_datasets if d in policy.allowed_datasets]


_api_lock = threading.Lock()
_api_registry: ApiPrincipalRegistry | None = None


def get_api_principal_registry() -> ApiPrincipalRegistry:
    global _api_registry
    if _api_registry is None:
        with _api_lock:
            if _api_registry is None:
                _api_registry = ApiPrincipalRegistry.from_env()
    return _api_registry


def reset_api_principal_registry() -> None:
    global _api_registry
    with _api_lock:
        _api_registry = None


__all__ = [
    "hash_api_key",
    "ApiPrincipalRegistry",
    "get_api_principal_registry",
    "reset_api_principal_registry",
]
