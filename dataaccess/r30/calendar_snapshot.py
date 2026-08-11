"""data_access.r30.calendar_snapshot —— R30-P0-008 全量 CalendarSnapshot digest。

既有 ``Store.calendar_snapshot_id()`` 只折叠 ``market:timezone:count:first:last``
（粗 digest）——中间某个交易日 / early-close 日期变了，count/first/last 全不变，
旧快照身份不变，日历缓存/可见时点全被带偏。本模块提供**全量**日历快照：

    - ``CalendarSnapshot``：独立对象，从 ``MarketCalendar`` / ``MarketSession``
      提取**全部交易日 + 全部 session open/close（含午休段）+ early-close 日期与
      提前收市时间 + timezone + 日历 source/version**，折叠成稳定 digest；
    - ``freeze_calendar_world(store)``：显式冻结 calendar 世界（进 job 时用）。

完全不修改 store.py / session_calendar.py；只消费稳定接口 + 防御性 getattr。
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from data_access.r30._shared import stable_digest, stable_digest_full

# 日历数据集按市场映射（source/version token 探针用；未知市场走 store._calendars）。
_CALENDAR_DATASETS = {
    "ashare": "ashare_calendar",
    "us": "us_calendar",
}


def _to_iso(value: Any) -> str:
    """把 date / datetime / 其它值折叠成稳定字符串（datetime 用完整 iso）。"""
    if isinstance(value, _dt.datetime):
        return value.isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())
        except Exception:
            pass
    return str(value)


def canonical(value: Any) -> str:
    """把任意值折叠成确定性**标量**字符串（稳定 digest 的输入）。

    ``data_access.r30._shared.stable_digest_full`` 对 list/tuple/set 输入有
    预置运算符优先级缺陷（``"[" + ",".join(items) + "]".encode()`` 变成
    ``str + bytes`` 抛 TypeError），所以任何集合都必须先折叠成标量再交给它。
    tuple 保序（session 段序有意义）；list/set 排序（集合序无关）。
    """
    if isinstance(value, Mapping):
        entries = "|".join(
            f"{canonical(k)}={canonical(v)}"
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        )
        return "{" + entries + "}"
    if isinstance(value, (list, tuple, set, frozenset)):
        if isinstance(value, tuple):
            parts = [canonical(x) for x in value]
        else:
            parts = sorted(canonical(x) for x in value)
        return "[" + "|".join(parts) + "]"
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, _dt.datetime):
        return value.isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    if value is None:
        return "None"
    return repr(value)


def _hhmm(value: Any) -> str:
    """time → 'HH:MM'；非 time 对象回退 repr。"""
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%H:%M")
        except Exception:
            pass
    return str(value)


def session_schedule(session: Any) -> tuple[tuple[str, str, str, int], ...]:
    """把 ``MarketSession.segments`` 序列化成稳定可散列形式（含午休段）。

    返回 ``((name, start_hhmm, end_hhmm, offset_min), ...)``；无 session /
    无 segments → ``()``。午休段（A 股 morning+afternoon）自然都在 segments 里。
    """
    if session is None:
        return ()
    segs = getattr(session, "segments", None) or ()
    out: list[tuple[str, str, str, int]] = []
    for seg in segs:
        name = str(getattr(seg, "name", "") or "")
        start = getattr(seg, "start", None)
        end = getattr(seg, "end", None)
        offset = int(getattr(seg, "offset_min", 0) or 0)
        out.append((name, _hhmm(start), _hhmm(end), offset))
    return tuple(out)


def early_close_dates(session: Any) -> tuple[str, ...]:
    """``MarketSession.early_close_dates`` → 排序后的 ISO 日期元组。"""
    if session is None:
        return ()
    dates = getattr(session, "early_close_dates", None) or ()
    return tuple(sorted(_to_iso(d) for d in dates if d is not None))


def early_close_time(session: Any) -> str | None:
    """``MarketSession.early_close_time`` → 'HH:MM' 或 None。"""
    if session is None:
        return None
    ec = getattr(session, "early_close_time", None)
    if ec is None:
        return None
    return _hhmm(ec)


def _calendar_source_version(store: Any, market: str, calendar: Any) -> str:
    """日历 source/version：``calendar.source`` +（有 manifest 时）数据版本 token。

    source/version 必须覆盖：既有 ``calendar_snapshot_id`` 只 hash 日历内容，不
    管日历**数据**版本——registry.yaml 没变但 ``ashare_calendar.parquet`` 更新了，
    内容变但 source 不变。有 manifest 时把 source_epoch / manifest_generation_id
    并进 version；没有则只保留 source（内容已由 digest 覆盖）。
    """
    source = str(getattr(calendar, "source", "") or "unknown")
    dataset = _CALENDAR_DATASETS.get(str(market).strip().lower())
    if not dataset:
        return source
    mv = None
    fn = getattr(store, "manifest_version", None)
    if callable(fn):
        try:
            mv = fn(dataset)
        except Exception:
            mv = None
    if isinstance(mv, dict) and mv.get("has_manifest"):
        token = (
            mv.get("manifest_generation_id")
            or mv.get("source_epoch")
            or mv.get("manifest_epoch")
            or mv.get("manifest_built_epoch")
        )
        if token is not None:
            return f"{source}:{token}"
    return source


def _known_markets(store: Any) -> list[str]:
    """store 里已注入的日历市场；无注入时回退 ['ashare', 'us']。"""
    cals = getattr(store, "_calendars", None)
    if isinstance(cals, dict) and cals:
        return sorted(str(k) for k in cals)
    return ["ashare", "us"]


@dataclass(frozen=True)
class CalendarSnapshot:
    """一个市场的**全量**交易日历快照身份（R30-P0-008）。

    与既有 ``Store.calendar_snapshot_id()``（market:tz:count:first:last 粗 digest）
    不同：本对象覆盖**全部交易日**、**全部 session open/close（含午休段）**、
    **early-close 日期与提前收市时间**、timezone、日历 source/version。任何中间
    元素变化都会改变 ``snapshot_id``。
    """

    market: str
    source_version: str
    timezone: str
    trading_days: tuple[str, ...]
    sessions: tuple[tuple[str, str, str, int], ...]
    early_close: tuple[str, ...]
    snapshot_id: str
    built_at: _dt.datetime = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc))

    # ---- 构建 ----

    @classmethod
    def build(
        cls,
        market: str,
        calendar: Any,
        source_version: str | None = None,
    ) -> "CalendarSnapshot":
        """从 ``MarketCalendar`` 构建全量快照。

        ``source_version`` 缺省用 ``calendar.source``；调用方（如 from_store）
        可传入更细的 source/version token（含日历数据版本）。
        """
        market = str(market)
        timezone = str(getattr(calendar, "timezone", None) or "UTC")
        src = str(getattr(calendar, "source", "") or "unknown")
        if source_version is None:
            source_version = src

        days = getattr(calendar, "trading_days", None) or ()
        trading_days = tuple(sorted(_to_iso(d) for d in days if d is not None))

        session = getattr(calendar, "session", None)
        sessions = session_schedule(session)
        ec_dates = early_close_dates(session)
        ec_time = early_close_time(session)
        early_close: list[str] = list(ec_dates)
        if ec_time is not None:
            early_close.append(f"early_close_time:{ec_time}")
        early_close_t = tuple(early_close)

        snapshot_id = stable_digest_full(
            canonical(market),
            canonical(timezone),
            canonical(trading_days),
            canonical(sessions),
            canonical(early_close_t),
            canonical(source_version),
        )
        return cls(
            market=market,
            source_version=source_version,
            timezone=timezone,
            trading_days=trading_days,
            sessions=sessions,
            early_close=early_close_t,
            snapshot_id=snapshot_id,
        )

    @classmethod
    def from_store(cls, store: Any, market: str) -> "CalendarSnapshot | None":
        """从 store 取该市场的日历并构建快照；get_calendar 返回 None → None。"""
        get = getattr(store, "get_calendar", None)
        if not callable(get):
            return None
        try:
            calendar = get(market)
        except Exception:
            return None
        if calendar is None:
            return None
        sv = _calendar_source_version(store, market, calendar)
        return cls.build(market, calendar, source_version=sv)

    # ---- 访问 ----

    def digest(self) -> str:
        """快照身份（与 snapshot_id 相同，语义别名）。"""
        return self.snapshot_id

    @property
    def first_trading_day(self) -> str | None:
        return self.trading_days[0] if self.trading_days else None

    @property
    def last_trading_day(self) -> str | None:
        return self.trading_days[-1] if self.trading_days else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "market": self.market,
            "source_version": self.source_version,
            "timezone": self.timezone,
            "trading_day_count": len(self.trading_days),
            "first_trading_day": self.first_trading_day,
            "last_trading_day": self.last_trading_day,
            "trading_days": list(self.trading_days),
            "sessions": [list(s) for s in self.sessions],
            "early_close": list(self.early_close),
            "built_at": self.built_at.isoformat(),
        }


def freeze_calendar_world(store: Any) -> dict[str, CalendarSnapshot]:
    """冻结 calendar 世界并返回当前各市场 ``CalendarSnapshot`` 的 dict。

    进入 job 时调用：显式 ``store.lock_calendars()``（若可调用）→ PIT 世界不可变；
    然后为 store 里每个已知市场构建全量快照。返回 ``{market: CalendarSnapshot}``。
    """
    lock = getattr(store, "lock_calendars", None)
    if callable(lock):
        try:
            lock()
        except Exception:
            pass
    out: dict[str, CalendarSnapshot] = {}
    for market in _known_markets(store):
        snap = CalendarSnapshot.from_store(store, market)
        if snap is not None:
            out[snap.market] = snap
    return out


__all__ = [
    "CalendarSnapshot",
    "canonical",
    "early_close_dates",
    "early_close_time",
    "freeze_calendar_world",
    "session_schedule",
]
