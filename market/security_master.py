# -*- coding: utf-8 -*-
"""SecurityMasterId —— 永久证券主键（R40 #227）。

ticker string 不能当永久主键：ticker 会改名（rename）、symbol 会被复用
（一个代码退市后另一家公司挂上同一代码）。本模块引入：

* :class:`SecurityMasterId` —— 一个证券的**永久、跨 rename 稳定**的身份
  （UUID / 交易所分配的永久代码）；
* :class:`SymbolValidityInterval` —— 一个 symbol 在哪个区间映射到该
  SecurityMasterId；
* :class:`SecurityMaster` —— 极小的自包含内存 registry，用
  ``(market, symbol, trade_date)`` 解析出 stable SecurityMasterId。

cache / factor lake 的 key 应使用 ``security_master_id`` 而不是 ticker string，
使 symbol rename 后历史输出仍映射到同一证券身份。本模块刻意自包含、不碰
dataaccess/security 的真实存储（真实存储由 dataaccess 簇负责）。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SecurityMasterId:
    """一个证券的永久身份。``value`` 是规范化的永久代码。"""

    market: str
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "market", str(self.market or "").strip().lower())
        object.__setattr__(self, "value", str(self.value or "").strip())
        if not self.value:
            raise ValueError("SecurityMasterId.value must be non-empty")

    def __str__(self) -> str:
        return f"{self.market}:sec:{self.value}"

    @classmethod
    def generate(cls, market: str, seed: str) -> "SecurityMasterId":
        """从 seed 生成稳定的永久 id（确定性，跨进程一致）。"""
        digest = hashlib.sha256(f"{market}|{seed}".encode("utf-8")).hexdigest()[:20]
        return cls(market=market, value=f"SEC-{digest}")


@dataclass(frozen=True)
class SymbolValidityInterval:
    """一个 symbol 映射到某 SecurityMasterId 的有效区间。"""

    symbol: str
    valid_from: str
    valid_to_exclusive: str | None = None  # None = 至今有效

    def contains(self, trade_date: Any) -> bool:
        import pandas as pd

        d = pd.Timestamp(trade_date)
        if self.valid_from and d < pd.Timestamp(self.valid_from):
            return False
        if self.valid_to_exclusive and d >= pd.Timestamp(self.valid_to_exclusive):
            return False
        return True


@dataclass(frozen=True)
class _SecurityRecord:
    security_master_id: SecurityMasterId
    intervals: tuple[SymbolValidityInterval, ...]


class SecurityMaster:
    """自包含的内存证券主档 registry。

    ``register`` 把一串 (symbol, valid_from, valid_to_exclusive) 区间绑到一个
    SecurityMasterId；``resolve(market, symbol, trade_date)`` 返回该时点有效的
    SecurityMasterId，无匹配返回 ``None``。
    """

    def __init__(self) -> None:
        self._records: dict[str, list[_SecurityRecord]] = {}

    def register(
        self,
        security_master_id: SecurityMasterId,
        intervals: list[SymbolValidityInterval],
    ) -> None:
        key = str(security_master_id.market)
        self._records.setdefault(key, []).append(
            _SecurityRecord(security_master_id, tuple(intervals))
        )

    def resolve(self, market: str, symbol: str, trade_date: Any) -> SecurityMasterId | None:
        key = str(market or "").strip().lower()
        sym = str(symbol or "").strip()
        for record in self._records.get(key, ()):
            for interval in record.intervals:
                if interval.symbol == sym and interval.contains(trade_date):
                    return record.security_master_id
        return None

    def symbols_for(self, security_master_id: SecurityMasterId) -> tuple[str, ...]:
        """该证券历史用过的所有 symbol（rename 追踪）。"""
        key = str(security_master_id.market)
        out: set[str] = set()
        for record in self._records.get(key, ()):
            if record.security_master_id == security_master_id:
                for interval in record.intervals:
                    out.add(interval.symbol)
        return tuple(sorted(out))


__all__ = [
    "SecurityMaster",
    "SecurityMasterId",
    "SymbolValidityInterval",
]
