"""data_access.r30._shared —— R30 层共享工具（身份摘要 / 安全范围 / 版本常量）。

R30 全维度成熟度层是完全 **additive** 的：不修改既有 store/read/runtime 任何文件，
所有新对象放在本包。本模块提供各 R30 模块共用的：

    1. ``stable_digest(*parts)`` —— 确定性 sha256 摘要（列表输入先 sort/str 化）；
    2. ``security_scope(store)`` —— 当前执行上下文的安全范围标识（principal +
       access_policy + run_mode 折叠），进 cache / execution identity；
    3. 版本治理常量（R30-P1-028）：API / CONTRACT_SCHEMA / REGISTRY_SCHEMA /
       SEMANTIC_SCHEMA / STORAGE_FORMAT 版本。
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping


def stable_digest(*parts: Any) -> str:
    """把任意可散列部件折叠成确定性的 16 字节 sha256。

    列表/dict 输入按 ``sorted(repr())`` 规范化，保证：
      - 同一输入多次调用 → 同一 digest；
      - 仅顺序不同的 list → 同一 digest（除非顺序有意义，请显式包 tuple）。
    """
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, Mapping):
            text = "|".join(
                f"{stable_digest(k)}={stable_digest(v)}"
                for k, v in sorted(part.items(), key=lambda kv: str(kv[0]))
            )
            h.update(f"{{{text}}}".encode("utf-8"))
        elif isinstance(part, (set, frozenset)):
            # Sets/frozensets have no order - sort for stability
            items = sorted(
                (stable_digest(x) for x in part),
                key=lambda d: d,
            )
            h.update(("[" + ",".join(items) + "]").encode("utf-8"))
        elif isinstance(part, (list, tuple)):
            # Lists/tuples have meaningful order - preserve it
            items = [stable_digest(x) for x in part]
            h.update(("[" + ",".join(items) + "]").encode("utf-8"))
        elif isinstance(part, bytes):
            h.update(part)
        else:
            h.update(repr(part).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:16]


def stable_digest_full(*parts: Any) -> str:
    """与 ``stable_digest`` 相同的折叠逻辑，但返回完整 64 字符 sha256。"""
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, Mapping):
            text = "|".join(
                f"{stable_digest_full(k)}={stable_digest_full(v)}"
                for k, v in sorted(part.items(), key=lambda kv: str(kv[0]))
            )
            h.update(f"{{{text}}}".encode("utf-8"))
        elif isinstance(part, (set, frozenset)):
            # Sets/frozensets have no order - sort for stability
            items = sorted((stable_digest_full(x) for x in part), key=lambda d: d)
            h.update(("[" + ",".join(items) + "]").encode("utf-8"))
        elif isinstance(part, (list, tuple)):
            # Lists/tuples have meaningful order - preserve it
            items = [stable_digest_full(x) for x in part]
            h.update(("[" + ",".join(items) + "]").encode("utf-8"))
        elif isinstance(part, bytes):
            h.update(part)
        else:
            h.update(repr(part).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


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
