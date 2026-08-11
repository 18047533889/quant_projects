# -*- coding: utf-8 -*-
"""R40 #213：operator capability payload hash 不再回退 repr(payload)。

production capability identity 不允许 repr fallback：typed serializer error
→ capability infrastructure error（CapabilityInfrastructureError），绝不返回
基于 ``repr()`` 的伪 identity。
"""
from __future__ import annotations

import hashlib

import pytest

from backend.operator_capability import (
    CapabilityInfrastructureError,
    _safe_payload_hash,
    _sql_contract,
)


class _Unserializable:
    def __repr__(self) -> str:  # repr works, but not JSON-serializable
        return "UNSERIALIZABLE-MARKER"


def test_happy_path_stable_hash() -> None:
    payload = {"canonical": "ts_mean", "source": "test", "n": 5}
    h1 = _safe_payload_hash(payload)
    h2 = _safe_payload_hash(dict(payload))
    assert isinstance(h1, str) and len(h1) == 16
    assert h1 == h2  # sort_keys + stable separators


def test_unserializable_payload_raises_not_fallback() -> None:
    """负控：不可序列化 payload → 抛 CapabilityInfrastructureError。

    绝不允许悄悄回退到 sha256(repr(payload))——那会掩盖 payload 的 identity 缺陷，
    并让 capability 缓存 key 依赖 repr 稳定性。
    """
    payload = {"canonical": "ts_mean", "extra": _Unserializable()}
    with pytest.raises(CapabilityInfrastructureError):
        _safe_payload_hash(payload)

    # 显式负控：确认旧行为（repr fallback）已被移除——绝不返回 repr 的 hash。
    fallback = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:16]
    try:
        _safe_payload_hash(payload)
    except CapabilityInfrastructureError:
        pass  # 期望路径
    else:  # pragma: no cover
        pytest.fail("repr fallback 仍生效——#213 未修复")


def test_sql_contract_hash_still_works() -> None:
    """集成：正常 SQL contract 仍可计算稳定 identity hash（不回归 #363）。"""
    contract = _sql_contract("ts_mean")
    assert isinstance(contract, dict)
    h = _safe_payload_hash(contract)
    assert isinstance(h, str) and len(h) == 16
