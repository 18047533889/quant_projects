"""data_access.r30.cache_hierarchy —— R30-P1-013：L0-L4 缓存层级模型。

把 DA 全部缓存层显式建模成 ``CacheLayer`` 列表（L0 request/plan → L1 source
block → L2 process metadata/query → L3 local SSD/COS → L4 remote source）。

**关键契约（freshness 单一事实源）**：所有层都必须绑定 security scope + snapshot
identity——任何一层都不允许自己「猜 freshness」（例如 L2 不能只看 manifest epoch
就假定数据没变，必须同时绑 security scope + snapshot identity）。``cache_scope_id``
把 ``layer.level + identity_parts + security_scope(store)`` 折叠成稳定 digest，
保证同一份数据在不同 principal/policy 下是不同缓存 scope，绝不共享缓存项。

完全 additive，不修改既有 read/query_cache.py / runtime/cache_manager.py。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from data_access.r30._shared import security_scope, stable_digest_full

__all__ = [
    "CACHE_HIERARCHY",
    "CacheLayer",
    "cache_scope_id",
    "invalidation_policy",
]


@dataclass(frozen=True)
class CacheLayer:
    """一层缓存的声明式模型。

    ``security_scope``    该层绑定的授权维度（principal/policy/run_mode 或
                           credential scope）；
    ``snapshot_scope``    该层绑定的快照维度（snapshot_id / manifest epoch /
                           object etag/checksum / source generation）；
    ``invalidation_trigger`` 该层**唯一**的失效触发（不允许各层自行猜 freshness）。
    """

    level: str
    purpose: str
    identity: str
    security_scope: str
    snapshot_scope: str
    ttl: str | None = None
    max_bytes: int | None = None
    eviction: str = "LRU"
    invalidation_trigger: str = ""


CACHE_HIERARCHY: list[CacheLayer] = [
    CacheLayer(
        level="L0",
        purpose="request/plan cache",
        identity="request+security+snapshot",
        security_scope="principal+policy+run_mode",
        snapshot_scope="request snapshot_id",
        ttl=None,
        max_bytes=None,
        eviction="session 结束",
        invalidation_trigger="session 结束",
    ),
    CacheLayer(
        level="L1",
        purpose="DataReadSession source-block cache",
        identity="dataset+snapshot+security+price_basis",
        security_scope="principal+policy+run_mode",
        snapshot_scope="source snapshot identity",
        ttl=None,
        max_bytes=256 * 1024 * 1024,
        eviction="LRU",
        invalidation_trigger="session exit / 超限 evict",
    ),
    CacheLayer(
        level="L2",
        purpose="process metadata/query cache",
        identity="dataset+manifest_epoch+security",
        security_scope="principal+policy+run_mode",
        snapshot_scope="manifest epoch",
        ttl="短 TTL",
        max_bytes=None,
        eviction="LRU/TTL",
        invalidation_trigger="manifest epoch 变化",
    ),
    CacheLayer(
        level="L3",
        purpose="local SSD/COS cache",
        identity="object uri+etag+credential_scope",
        security_scope="credential_scope",
        snapshot_scope="object etag/checksum",
        ttl="持久",
        max_bytes=None,
        eviction="容量/LRU",
        invalidation_trigger="etag/checksum 变化",
    ),
    CacheLayer(
        level="L4",
        purpose="remote source",
        identity="source generation",
        security_scope="source credential scope",
        snapshot_scope="source generation",
        ttl="持久",
        max_bytes=None,
        eviction="无",
        invalidation_trigger="manifest pointer",
    ),
]


def invalidation_policy(layer: CacheLayer) -> dict[str, Any]:
    """返回一层的失效策略；所有层强制绑定 security + snapshot（单事实源）。"""
    return {
        "level": layer.level,
        "purpose": layer.purpose,
        "identity": layer.identity,
        "security_scope": layer.security_scope,
        "snapshot_scope": layer.snapshot_scope,
        "ttl": layer.ttl,
        "max_bytes": layer.max_bytes,
        "eviction": layer.eviction,
        "invalidation_trigger": layer.invalidation_trigger,
        "security_bound": True,
        "snapshot_bound": True,
        "freshness_authority": "security scope + snapshot identity（单事实源，"
        "禁止各层各自猜 freshness）",
    }


def cache_scope_id(layer: CacheLayer, identity_parts: Iterable[Any], store: Any) -> str:
    """折叠成缓存 scope digest：``layer.level + identity_parts + security_scope``。

    ``identity_parts`` 必须**包含 snapshot identity**（快照维度由调用方携带，
    与层声明一致）；security scope 由 ``security_scope(store)`` 折叠进去。
    不同 principal/policy 得不同 digest → 绝不跨 scope 共享缓存项。
    """
    scope = security_scope(store)
    parts: list[str] = [layer.level]
    parts.extend(str(p) for p in identity_parts)
    if scope is not None:
        parts.append(scope)
    return stable_digest_full(*parts)
