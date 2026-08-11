# -*- coding: utf-8 -*-
"""PriceGridContract —— 每市场 tick size / 价格网格契约（R40 #231）。

涨跌停算子里 ``tick_tolerance: float = 0.005`` 是硬编码（0.5%），忽略了市场
实际的报价单位网格：A 股 0.01 元（价格 < 0.01 有特殊网格）、US 是 0.01 美元
等。operator 的 ``tick_tolerance`` 参数必须受 ``declared_tick_policy_bound``
约束（上限校验），默认值从 PriceGridContract 获取，而不是写死 0.005。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PriceGridContract:
    """一个市场的价格网格契约。

    ``base_tick`` 是该市场大部分价格区间的报价单位（A 股 0.01 元、US 0.01 美元）。
    ``declared_tick_policy_bound`` 是 operator 的 ``tick_tolerance`` 允许的
    **上限**（超过即拒绝——容差必须与市场网格一致，不能任意放大）。
    ``tick_size(instrument, trade_date)`` 可按 instrument / date 特判（如
    A 股低价股 0.001 元网格），默认返回 ``base_tick``。
    """

    market: str
    base_tick: float = 0.01
    declared_tick_policy_bound: float = 0.005
    currency: str = "CNY"

    def tick_size(self, instrument: Any = None, trade_date: Any = None) -> float:
        """返回指定 instrument / trade_date 的报价单位（默认 base_tick）。"""
        del instrument, trade_date
        return float(self.base_tick)

    def validate_tick_tolerance(self, tick_tolerance: float) -> float:
        """校验 operator 的 ``tick_tolerance`` 不超 ``declared_tick_policy_bound``。"""
        t = float(tick_tolerance)
        if t < 0:
            raise ValueError("tick_tolerance must be non-negative")
        if t > float(self.declared_tick_policy_bound):
            raise ValueError(
                f"tick_tolerance={t!r} exceeds declared_tick_policy_bound="
                f"{self.declared_tick_policy_bound!r} for market {self.market!r} — "
                "the tolerance must stay within the market's price-grid policy"
            )
        return t

    def default_tick_tolerance(self) -> float:
        return float(self.declared_tick_policy_bound)


ASHARE_PRICE_GRID = PriceGridContract(
    market="ashare", base_tick=0.01, declared_tick_policy_bound=0.005, currency="CNY",
)
US_PRICE_GRID = PriceGridContract(
    market="us", base_tick=0.01, declared_tick_policy_bound=0.005, currency="USD",
)

_PRICE_GRIDS = {"ashare": ASHARE_PRICE_GRID, "us": US_PRICE_GRID}


def price_grid_for_market(market: str) -> PriceGridContract:
    key = str(market or "").strip().lower()
    if key in {"a_share", "cn", "china"}:
        key = "ashare"
    try:
        return _PRICE_GRIDS[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"no PriceGridContract for market {market!r}") from exc


__all__ = [
    "ASHARE_PRICE_GRID",
    "PriceGridContract",
    "US_PRICE_GRID",
    "price_grid_for_market",
]
