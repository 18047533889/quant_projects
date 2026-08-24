# -*- coding: utf-8 -*-
"""市场交易日历：A 股 / 美股增量窗口偏移（替代纯 pandas bdate_range）。"""

from __future__ import annotations

import bisect
from typing import Literal

import pandas as pd

Market = Literal["ashare", "us", "pandas"]

_CALENDAR_CACHE: dict[str, TradingCalendar] = {}


class CalendarUnavailableError(RuntimeError):
    """R32-P0-001: 真实交易日历不可用（production 禁止伪装成普通工作日）。"""


class CalendarCoverageError(ValueError):
    """R32-P0-002/003: 偏移超出日历 coverage / anchor 不满足精确交易日。

    production 缺历史日历会变成看似合法的 warmup 日期 —— 默认抛错，``clamp=True``
    仅用于显式 UI/研究用途。
    """


class TradingCalendar:
    """有序交易日列表，支持相对 bar 偏移。

    R32-P0-003 anchor policy:
      - ``exact_trade_day``（生产默认）：``offset(base, 0)`` 只接受本身就是交易日
        的 base，否则抛 ``CalendarCoverageError``（绝不把周末/节假日返回成原日期）；
      - ``previous_trade_day``：非交易日向前取最近交易日；
      - ``next_trade_day``：非交易日向后取最近交易日。

    R32-P0-002: ``offset`` 超出左/右 coverage 默认抛 ``CalendarCoverageError``；
    ``clamp=True`` 才 clamp 到首/末日（仅显式 UI/研究用途）。

    参数:
        trading_days: 见函数签名
        anchor_policy: anchor policy（exact_trade_day/previous_trade_day/next_trade_day）
        source: 日历来源（data_access / explicit）
        snapshot: 数据快照 id（可选）
        version: 日历版本（可选）
        timezone: 日历时区（可选）
    """

    def __init__(
        self,
        trading_days: list[pd.Timestamp | str],
        *,
        anchor_policy: str = "exact_trade_day",
        source: str | None = None,
        snapshot: str | None = None,
        version: str | None = None,
        timezone: str | None = None,
    ) -> None:
        """初始化实例。

        参数:
            trading_days: 见函数签名
            anchor_policy: anchor policy（可选）
            source: 日历来源（可选）
            snapshot: 数据快照 id（可选）
            version: 日历版本（可选）
            timezone: 日历时区（可选）

        返回:
            无
        """
        days = sorted({pd.Timestamp(d).normalize() for d in trading_days})
        if not days:
            raise ValueError("TradingCalendar 需要至少一个交易日")
        if anchor_policy not in {"exact_trade_day", "previous_trade_day", "next_trade_day"}:
            raise ValueError(
                f"anchor_policy must be exact_trade_day|previous_trade_day|"
                f"next_trade_day, got {anchor_policy!r}"
            )
        self._days = days
        self._anchor_policy = str(anchor_policy)
        self._source = str(source) if source else None
        self._snapshot = str(snapshot) if snapshot else None
        self._version = str(version) if version else None
        self._timezone = str(timezone) if timezone else None

    @property
    def anchor_policy(self) -> str:
        """anchor_policy。

        参数:
            无

        返回:
            str
        """
        return self._anchor_policy

    @property
    def source(self) -> str | None:
        """source。

        参数:
            无

        返回:
            str | None
        """
        return self._source

    @property
    def snapshot(self) -> str | None:
        """snapshot。

        参数:
            无

        返回:
            str | None
        """
        return self._snapshot

    @property
    def version(self) -> str | None:
        """version。

        参数:
            无

        返回:
            str | None
        """
        return self._version

    @property
    def timezone(self) -> str | None:
        """timezone。

        参数:
            无

        返回:
            str | None
        """
        return self._timezone

    def metadata(self) -> dict:
        """R32 §2: 日历来源 / snapshot / 版本追踪。"""
        return {
            "source": self._source,
            "snapshot": self._snapshot,
            "version": self._version,
            "timezone": self._timezone,
            "anchor_policy": self._anchor_policy,
            "days": len(self._days),
            "first_day": str(self._days[0]),
            "last_day": str(self._days[-1]),
        }

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
        """R32-P0-003: 按 anchor_policy 定位 base 在日历中的索引。

        参数:
            base: 见函数签名

        返回:
            int
        """
        base = pd.Timestamp(base).normalize()
        pos = bisect.bisect_right(self._days, base) - 1
        if self._anchor_policy == "exact_trade_day":
            if pos < 0 or self._days[pos] != base:
                raise CalendarCoverageError(
                    f"base={base.date()} is not a trading day and anchor_policy="
                    f"exact_trade_day (production default). Use previous_trade_day/"
                    f"next_trade_day anchor or clamp=True for UI/research only."
                )
            return pos
        if self._anchor_policy == "previous_trade_day":
            if pos < 0:
                raise CalendarCoverageError(
                    f"base={base.date()} precedes calendar coverage "
                    f"[{self._days[0].date()} .. {self._days[-1].date()}]"
                )
            return pos
        # next_trade_day
        if self._days[pos] < base:
            pos += 1
        if pos >= len(self._days):
            raise CalendarCoverageError(
                f"base={base.date()} exceeds calendar coverage "
                f"[{self._days[0].date()} .. {self._days[-1].date()}]"
            )
        return pos

    def offset(
        self,
        base: str | pd.Timestamp,
        n: int,
        *,
        clamp: bool = False,
    ) -> pd.Timestamp:
        """相对 base 偏移 n 个交易日（n>0 向前，n<0 向后）。

        R32-P0-002: 超出 coverage 默认抛 ``CalendarCoverageError``（production
        incremental 禁止 clamp —— 缺历史日历会变成看似合法的 warmup 日期）；
        ``clamp=True`` 仅用于显式 UI/研究用途。

        参数:
            base: 见函数签名
            n: 见函数签名
            clamp: 越界是否 clamp 到首/末日（可选，默认 False）

        返回:
            pd.Timestamp
        """
        base_ts = pd.Timestamp(base).normalize()
        if n == 0:
            # R32-P0-003: anchor policy —— exact 下非交易日抛错，不再直接返回原日期。
            pos = self._anchor_index(base_ts)
            return self._days[pos]
        pos = self._anchor_index(base_ts)
        target = pos + int(n)
        if target < 0 or target >= len(self._days):
            if clamp:
                return self._days[0] if target < 0 else self._days[-1]
            raise CalendarCoverageError(
                f"calendar offset {n} from {base_ts.date()} lands outside coverage "
                f"[{self._days[0].date()} .. {self._days[-1].date()}]; production "
                f"incremental must not clamp (would fabricate a warmup date)."
            )
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

    R32-P0-001: production 下真实日历不可用必须 fail-closed（抛
    ``CalendarUnavailableError``），绝不静默返回 None 让调用方退化成 pandas
    bdate_range 普通工作日。research 下保留返回 None（调用方按需
    ``allow_approximate_calendar`` 走 bdate）。

    参数:
        market: 市场标识（ashare/us）

    返回:
        TradingCalendar | None
    """
    try:
        import os
        import sys

        from factor_engine.util.workspace_paths import quant_projects_root

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
        return TradingCalendar(
            days.tolist(),
            source="data_access",
            snapshot=str(getattr(store, "snapshot_id", "") or None) or None,
            version=str(getattr(store, "version", "") or None) or None,
        )
    except CalendarUnavailableError:
        raise
    except Exception as exc:
        from factor_engine.runtime.production_policy import is_production_mode

        if is_production_mode():
            raise CalendarUnavailableError(
                f"real trading calendar for market={market!r} unavailable from "
                f"data_access ({exc!r}); production must NOT fall back to plain "
                f"Mon-Fri business days (R32-P0-001)."
            ) from exc
        return None


def get_trading_calendar(
    market: str | None,
    *,
    allow_approximate_calendar: bool = False,
) -> TradingCalendar | None:
    """获取市场交易日历。

    R32-P0-001: production 下真实日历不可用且未显式 ``allow_approximate_calendar``
    时抛 ``CalendarUnavailableError``。research 下不可用返回 None（调用方走
    bdate fallback）。记录 calendar source / snapshot / version（§2）。

    参数:
        market: 市场标识（ashare/us）
        allow_approximate_calendar: 是否允许近似工作日历（仅显式 research/UI，可选）

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
    elif loaded is None and not allow_approximate_calendar:
        from factor_engine.runtime.production_policy import is_production_mode

        if is_production_mode():
            raise CalendarUnavailableError(
                f"trading calendar for market={market!r} is unavailable and "
                f"allow_approximate_calendar is not set; production refuses the "
                f"silent bdate fallback (R32-P0-001)."
            )
    return loaded


def trading_day_offset(
    base: str | pd.Timestamp,
    n: int,
    *,
    calendar: TradingCalendar | None = None,
    clamp: bool = False,
    allow_approximate_calendar: bool = False,
) -> pd.Timestamp:
    """统一交易日偏移入口（可指定日历或回退 pandas bdate）。

    R32-P0-001/002: production 下 calendar 不可用且未显式
    ``allow_approximate_calendar`` 时抛 ``CalendarUnavailableError``；offset 越界
    默认抛 ``CalendarCoverageError``，``clamp=True`` 仅研究/UI。

    参数:
        base: 见函数签名
        n: 见函数签名
        calendar: 交易日历实例（可选）
        clamp: 越界是否 clamp（可选，默认 False）
        allow_approximate_calendar: 是否允许 bdate 近似（可选，默认 False）

    返回:
        pd.Timestamp
    """
    if calendar is not None:
        return calendar.offset(base, n, clamp=clamp)
    ts = pd.Timestamp(base).normalize()
    if n == 0:
        return ts
    from factor_engine.runtime.production_policy import is_production_mode

    if is_production_mode() and not allow_approximate_calendar:
        raise CalendarUnavailableError(
            f"trading calendar unavailable in production for offset(base={base}, "
            f"n={n}); silent pandas bdate fallback is forbidden (R32-P0-001). "
            f"Pass a real calendar or set allow_approximate_calendar=True."
        )
    if n > 0:
        rng = pd.bdate_range(ts + pd.Timedelta(days=1), periods=n)
        return pd.Timestamp(rng[-1])
    rng = pd.bdate_range(end=ts - pd.Timedelta(days=1), periods=-n)
    return pd.Timestamp(rng[0])
