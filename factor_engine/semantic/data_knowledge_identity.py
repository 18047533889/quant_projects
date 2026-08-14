# -*- coding: utf-8 -*-
"""R37-P0-011：DataKnowledgeIdentity —— 数据知识身份的统一组合对象。

R37 §6 要求：cache key / FactorId / CSE identity / checkpoint identity /
materialization manifest / lineage / evidence / incremental invalidation 全部
消费同一个数据知识身份。当前实现把这些维度散落在
``FactorSemanticIdentity`` / ``SourceVintageSpec`` / ``DataScope`` 等对象中，
本模块提供一个**单一组合** dataclass，把散落的维度收敛成可哈希、可序列化、
可比较的统一 identity。

设计要点：
- frozen dataclass，``to_key()`` 输出 canonical JSON 字符串（可作 cache/checkpoint
  key），``digest()`` 输出 SHA-256 hex（可作 FactorId 维度）。
- 任一字段变化 => key/digest 变化：不同 snapshot/universe/calendar/price_basis/
  revision 的同一公式不是同一数据知识。
- 不重复制造 truth：字段值从上游 ``FactorSemanticIdentity`` / DA snapshot 等
  权威来源抽取；本模块只做组合 + 哈希。

Wire-in（R37-P0-011 要求进入 cache/checkpoint/lineage）：
    from semantic.data_knowledge_identity import DataKnowledgeIdentity
    dki = DataKnowledgeIdentity.from_factor_identity(identity, snapshot_meta=...)
    cache_key_suffix = dki.to_key()
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DataKnowledgeIdentity:
    """统一数据知识身份（R37-P0-011）。

    ``universe`` 语义：可同时保留命名 universe 标签与 resolved membership hash
    —— 两者任一变化都会改 digest（P0-012：universe 必须是 PIT 对象，成员列表
    随 valid_time 变化）。
    """

    dataset_id: str = ""
    snapshot_id: str = ""
    schema_epoch: str = ""
    market: str = ""
    calendar_id: str = ""
    universe_snapshot_id: str = ""
    universe_membership_hash: str = ""
    price_basis: str = ""
    corporate_action_vintage: str = ""
    source_revision_id: str = ""
    availability_policy_version: str = ""
    timezone: str = ""
    #: 附加维度（supplier/period_identity/flow 等），排序后进 key
    extra: tuple[tuple[str, str], ...] = ()

    # -- 构造 --

    @classmethod
    def from_factor_identity(cls, identity: Any,
                             snapshot_meta: Mapping[str, Any] | None = None,
                             **extra: str) -> "DataKnowledgeIdentity":
        """从 ``FactorSemanticIdentity`` + DA snapshot metadata 组合。

        ``identity`` 只需有 ``to_dict()``（FactorSemanticIdentity 满足）。
        ``snapshot_meta`` 可带 snapshot_id / schema_epoch / source_revision_id。
        """
        d = {}
        if identity is not None:
            try:
                d = identity.to_dict() if hasattr(identity, "to_dict") else dict(identity)
            except Exception:
                d = {}
        snap = dict(snapshot_meta or {})
        extras = tuple(sorted((k, str(v)) for k, v in extra.items()))
        return cls(
            dataset_id=str(d.get("source_contract_hash", "") or d.get("dataset", "")),
            snapshot_id=str(snap.get("snapshot_id", "") or d.get("source_scope_hash", "")),
            schema_epoch=str(snap.get("schema_epoch", "")),
            market=str(d.get("market", "") or ""),
            calendar_id=str(d.get("calendar", "") or ""),
            universe_snapshot_id=str(d.get("universe", "") or ""),
            universe_membership_hash=str(d.get("universe_membership_hash", "") or ""),
            price_basis=str(d.get("price_basis", "") or ""),
            corporate_action_vintage=str(snap.get("corporate_action_vintage", "")),
            source_revision_id=str(snap.get("source_revision_id", "")
                                   or d.get("source_dependency_hash", "")),
            availability_policy_version=str(d.get("pit_policy", "") or ""),
            timezone=str(d.get("timezone", "") or ""),
            extra=extras,
        )

    # -- 哈希 --

    def to_key(self) -> str:
        """canonical JSON 字符串——可作 cache/checkpoint 后缀 key。"""
        payload = asdict(self)
        payload["extra"] = sorted(payload["extra"])
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def digest(self) -> str:
        """SHA-256 hex (full 256-bit) —— 可作 FactorId / lineage 维度。

        DA-ID-P0-002: 使用完整 256-bit hash 避免 FactorId 命名空间碰撞。
        64-bit 在 ~4B 因子时有 50% 碰撞概率（生日悖论），生产环境不可接受。
        """
        return hashlib.sha256(self.to_key().encode("utf-8")).hexdigest()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, DataKnowledgeIdentity) and self.to_key() == other.to_key()

    def __hash__(self) -> int:
        return hash(self.to_key())


def compute_data_knowledge_identity(
    identity: Any, snapshot_meta: Mapping[str, Any] | None = None, **extra: str,
) -> DataKnowledgeIdentity:
    """便捷工厂。"""
    return DataKnowledgeIdentity.from_factor_identity(identity, snapshot_meta, **extra)
