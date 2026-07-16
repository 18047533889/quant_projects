# -*- coding: utf-8 -*-
"""市场交易日历：A 股 / 美股增量窗口偏移（替代纯 pandas bdate_range）。"""

from __future__ import annotations

import bisect
from typing import Literal

import pandas as pd

Market = Literal["ashare", "us", "pandas"]

_CALENDAR_CACHE: dict[str, TradingCalendar] = {}


class TradingCalendar:
    """有序交易日列表，支持相对 bar 偏移。
    
    参数:
        trading_days: 见函数签名
    """

    def __init__(self, trading_days: list[pd.Timestamp | str]) -> None:
        """初始化实例。
        
        参数:
            trading_days: 见函数签名
        
        返回:
            无
        """
        days = sorted({pd.Timestamp(d).normalize() for d in trading_days})
        if not days:
            raise ValueError("TradingCalendar 需要至少一个交易日")
        self._days = days

    @property
    def days(self) -> list[pd.Timestamp]:
        """days。
        
        参数:
            无
        
        返回:
            list[pd.Timestamp]
        """
        return list(self._days)

    def _anchor_index(self, base: pd.Timestamp) -> int:
        """_anchor_index。
        
        参数:
            base: 见函数签名
        
        返回:
            int
        """
        base = pd.Timestamp(base).normalize()
        pos = bisect.bisect_right(self._days, base) - 1
        return max(0, min(pos, len(self._days) - 1))

    def offset(self, base: str | pd.Timestamp, n: int) -> pd.Timestamp:
        """相对 base 偏移 n 个交易日（n>0 向前，n<0 向后）。
        
        参数:
            base: 见函数签名
            n: 见函数签名
        
        返回:
            pd.Timestamp
        """
        if n == 0:
            return pd.Timestamp(base).normalize()
        pos = self._anchor_index(base)
        target = pos + int(n)
        if target < 0:
            return self._days[0]
        if target >= len(self._days):
            return self._days[-1]
        return self._days[target]


def register_trading_calendar(market: str, calendar: TradingCalendar) -> None:
    """注册市场交易日历到进程内缓存。
    
    参数:
        market: 市场标识（ashare/us）
        calendar: 交易日历实例
    
    返回:
        无
    """
    _CALENDAR_CACHE[str(market).lower()] = calendar


def clear_trading_calendar_cache() -> None:
    """清空进程内交易日历缓存。
    
    参数:
        无
    
    返回:
        无
    """
    _CALENDAR_CACHE.clear()


def infer_market(
    *,
    universe: str | None = None,
    dataset: str | None = None,
) -> str | None:
    """从 universe 或 dataset 名称推断市场标识。
    
    参数:
        universe: 标的池标识（可选）
        dataset: data_access 数据集名称（可选）
    
    返回:
        str | None
    """
    u = str(universe or "").upper()
    if any(k in u for k in ("ASHARE", "A_SHARE", "A股", "CN_")):
        return "ashare"
    if any(k in u for k in ("US_", "MASSIVE", "NYSE", "NASDAQ")):
        return "us"
    ds = str(dataset or "").lower()
    if ds.startswith("ashare_"):
        return "ashare"
    if ds.startswith("us_"):
        return "us"
    return None


def _load_calendar_from_data_access(market: str) -> TradingCalendar | None:
    """从 data_access 加载指定市场的交易日历。
    
    参数:
        market: 市场标识（ashare/us）
    
    返回:
        TradingCalendar | None
    """
    try:
        import os
        import sys

        from workspace_paths import quant_projects_root

        root = str(quant_projects_root())
        if root not in sys.path:
            sys.path.insert(0, root)
        os.environ.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")
        from data_access import get_store

        store = get_store()
        if market == "ashare":
            frame = store.read_frame(
                "ashare_calendar",
                columns=["TradeDate", "IsTradeDay"],
            )
            mask = frame["IsTradeDay"].astype(bool)
            days = pd.to_datetime(frame.loc[mask, "TradeDate"]).dt.normalize()
        elif market == "us":
            frame = store.read_frame(
                "us_calendar",
                columns=["trade_date", "is_trading_day"],
            )
            mask = frame["is_trading_day"].astype(bool)
            days = pd.to_datetime(frame.loc[mask, "trade_date"]).dt.normalize()
        else:
            return None
        if days.empty:
            return None
        return TradingCalendar(days.tolist())
    except Exception:
        return None


def get_trading_calendar(market: str | None) -> TradingCalendar | None:
    """获取市场交易日历，不可用时返回 None。
    
    参数:
        market: 市场标识（ashare/us）
    
    返回:
        TradingCalendar | None
    """
    if not market:
        return None
    key = str(market).lower()
    if key in _CALENDAR_CACHE:
        return _CALENDAR_CACHE[key]
    loaded = _load_calendar_from_data_access(key)
    if loaded is not None:
        _CALENDAR_CACHE[key] = loaded
    return loaded


def trading_day_offset(
    base: str | pd.Timestamp,
    n: int,
    *,
    calendar: TradingCalendar | None = None,
) -> pd.Timestamp:
    """统一交易日偏移入口（可指定日历或回退 pandas bdate）。
    
    参数:
        base: 见函数签名
        n: 见函数签名
        calendar: 交易日历实例（可选）
    
    返回:
        pd.Timestamp
    """
    if calendar is not None:
        return calendar.offset(base, n)
    ts = pd.Timestamp(base).normalize()
    if n == 0:
        return ts
    if n > 0:
        rng = pd.bdate_range(ts + pd.Timedelta(days=1), periods=n)
        return pd.Timestamp(rng[-1])
    rng = pd.bdate_range(end=ts - pd.Timedelta(days=1), periods=-n)
    return pd.Timestamp(rng[0])
