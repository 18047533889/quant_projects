"""data_access.r30._shared —— R30 层共享工具（身份摘要 / 安全范围 / 版本常量）。

R30 全维度成熟度层是完全 **additive** 的：不修改既有 store/read/runtime 任何文件，
所有新对象放在本包。本模块提供各 R30 模块共用的：

    1. ``stable_digest(*parts)`` —— strict canonical identity encoder 的完整 SHA-256 摘要；
    2. ``security_scope(store)`` —— 当前执行上下文的安全范围标识（principal +
       access_policy + run_mode 折叠），进 cache / execution identity；
    3. 版本治理常量（R30-P1-028）：API / CONTRACT_SCHEMA / REGISTRY_SCHEMA /
       SEMANTIC_SCHEMA / STORAGE_FORMAT 版本。
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from data_access.core.exceptions import ValidationError
from data_access.core.identity_encoder import CanonicalIdentityEncoder


_STRICT_IDENTITY_ENCODER = CanonicalIdentityEncoder(strict=True)


def _stable_digest(*parts: Any) -> str:
    """Encode ordered digest parts through the canonical strict identity path."""
    try:
        return _STRICT_IDENTITY_ENCODER.hash_identity(parts, bits=256)
    except ValidationError as exc:
        raise ValueError(f"unsupported type in stable digest: {exc}") from exc


def stable_digest(*parts: Any) -> str:
    """Return a strict canonical SHA-256 digest of the ordered parts."""
    return _stable_digest(*parts)


def stable_digest_full(*parts: Any) -> str:
    """Return a strict canonical SHA-256 digest of the ordered parts."""
    return _stable_digest(*parts)


def security_scope(store: Any) -> str | None:
    """折叠当前执行上下文的授权范围 → 稳定标识。

    进 R30 cache key 与 execution identity（R30-P1-018）：同一份数据在
    不同 principal/policy 下是不同 scope，绝不共享缓存项。从 Store 的既有
    ``_effective_security_digest`` 取（不存在则 None → 调用方按未绑定处理）。
    """
    fn = getattr(store, "_effective_security_digest", None)
    if callable(fn):
        try:
            return fn()
        except Exception:
            return None
    return None


# ---- R30-P1-028 版本治理拆分：不把所有兼容性都用 Python package 0.x.y 表达 ----

API_VERSION = "1.0"
"""REST/服务 API 版本。破坏性 API 变更时递增。"""

CONTRACT_SCHEMA_VERSION = "2"
"""RuntimeDatasetContract / DataRequest / PreparedRead 的 schema 版本。"""

REGISTRY_SCHEMA_VERSION = "3"
"""Dataset Registry / typed Contract IR 的 JSON schema 版本。"""

SEMANTIC_SCHEMA_VERSION = "1"
"""SemanticFieldCatalog / UnitSpec / ConceptId 语义元数据版本。"""

STORAGE_FORMAT_VERSION = "1"
"""dataaccess 自有持久格式（manifest / metadata_plane / partition index）。"""


def version_gate() -> dict[str, str]:
    """把全部版本常量折叠成一个 dict（进 ExperimentDataSnapshot / lineage）。"""
    return {
        "api": API_VERSION,
        "contract_schema": CONTRACT_SCHEMA_VERSION,
        "registry_schema": REGISTRY_SCHEMA_VERSION,
        "semantic_schema": SEMANTIC_SCHEMA_VERSION,
        "storage_format": STORAGE_FORMAT_VERSION,
    }


def sorted_items(mapping: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
    return sorted(mapping.items(), key=lambda kv: str(kv[0]))


__all__ = [
    "API_VERSION",
    "CONTRACT_SCHEMA_VERSION",
    "REGISTRY_SCHEMA_VERSION",
    "SEMANTIC_SCHEMA_VERSION",
    "STORAGE_FORMAT_VERSION",
    "security_scope",
    "sorted_items",
    "stable_digest",
    "stable_digest_full",
    "version_gate",
]
