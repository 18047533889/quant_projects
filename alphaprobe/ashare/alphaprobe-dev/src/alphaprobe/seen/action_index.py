"""ActionSeenIndex（§39）。

action_key = sha256(parent_signal_id + action_type +
normalized_payload_json(sort_keys) + grammar_version)。
seen_actions UNIQUE(action_key)；try_record(action_key, ...) -> (first_time: bool)。
payload 归一：数字/列表 json sort_keys。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from alphaprobe.seen.store import SeenStore


def normalize_payload(payload: Any) -> Any:
    """payload 归一：数字排序、dict sort_keys、递归。"""
    if isinstance(payload, dict):
        return {str(k): normalize_payload(v) for k, v in sorted(payload.items())}
    if isinstance(payload, (list, tuple, set, frozenset)):
        items = [normalize_payload(v) for v in payload]
        try:
            return sorted(items, key=repr)
        except TypeError:
            return items
    return payload


def action_key(
    parent_signal_id: str,
    action_type: str,
    payload: Any,
    grammar_version: str = "v1",
) -> str:
    """§39 action_key = sha256(parent + type + normalized_payload + grammar_version)。"""
    payload_json = json.dumps(normalize_payload(payload), sort_keys=True, default=str)
    material = f"{parent_signal_id}|{action_type}|{payload_json}|{grammar_version}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class ActionSeenIndex:
    """§39：跨轮/跨 Miner 的 action 去重。"""

    def __init__(self, store: SeenStore) -> None:
        self._store = store

    def try_record(
        self,
        action_key: str,
        payload: Any,
        *,
        parent_signal_id: str | None = None,
        action_type: str | None = None,
        grammar_version: str = "v1",
    ) -> bool:
        """记录 action；返回 first_time（True=首次见到）。"""
        payload_json = json.dumps(normalize_payload(payload), sort_keys=True, default=str)
        return self._store.try_record_action(
            action_key,
            parent_signal_id=parent_signal_id,
            action_type=action_type,
            payload_json=payload_json,
            grammar_version=grammar_version,
        )

    def build_and_record(
        self,
        parent_signal_id: str,
        action_type: str,
        payload: Any,
        *,
        grammar_version: str = "v1",
    ) -> tuple[str, bool]:
        """构建 action_key + 记录。返回 (action_key, first_time)。"""
        key = action_key(parent_signal_id, action_type, payload, grammar_version)
        first_time = self.try_record(
            key, payload,
            parent_signal_id=parent_signal_id,
            action_type=action_type,
            grammar_version=grammar_version,
        )
        return key, first_time

    def count(self) -> int:
        return self._store.count_actions()