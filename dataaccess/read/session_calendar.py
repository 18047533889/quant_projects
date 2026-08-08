"""
data_access.read.session_calendar —— 交易所 session / 交易日历（时区感知）

为什么需要
    #3 依赖。旧实现把 ``next_trading_day`` 翻译成 SQL 严格大于 ``decision > knowledge``，
    在日频 A 股上近似可用，但：
      - 周五 / 节假日前公告、美股 filing timestamp、盘前/盘后 filing、
        early close、分钟级决策都会出问题；
      - 分钟聚合（#5）里 QuoteTime 存 UTC，但 A 股实际 bar 标签是
        北京时间 09:31–11:30、13:01–15:00（共 240 根，午休无 bar），
        旧 ``minute - 570`` 用自然时钟差，午后多算 90 分钟、且 09:30 不是第一根。

本模块提供：
    ``MarketSession``  —— 一个市场的交易时段（含午休），可把 HH:MM / 本地时间
        换算成 session elapsed bar index，也可生成 SQL 时区/elapsed 表达式。
    ``MarketCalendar`` —— 交易日历（可注入交易日序列，或从 registry 的
        ashare_calendar / us_calendar 数据集惰性加载），提供下一/前一交易日、
        ``available_from`` 编译。
    ``get_market_session`` / ``get_market_calendar`` —— 进程内单例。

字典事实（COS_ashare_lqtp_data_dictionary.md）：
    - A 股 QuoteTime 存 UTC，北京时间 = UTC+8；
    - CST 时段 09:31–11:30 与 13:01–15:00（共 240 个分钟标签；午休无 bar；
      无 09:30/13:00 标签；集合竞价无独立 bar，开盘价体现在首根分钟）；
    - A 股 StockList/Status/Industry/TopTen/ETFList/IndexList/IndexConstituent
      含周末自然日文件；行情/估值主要是交易日。

维护人：quant 基础平台组    最后更新：2026-08-08
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from data_access.core.exceptions import ValidationError

# (start, end) 是闭区间分钟标签；offset_min 是该时段第一根 bar 的 session
# elapsed 下标。
_ASHARE_SEGMENTS = (
    ("morning", _dt.time(9, 31), _dt.time(11, 30), 0),
    ("afternoon", _dt.time(13, 1), _dt.time(15, 0), 120),
)
_US_SEGMENTS = (
    ("continuous", _dt.time(9, 30), _dt.time(16, 0), 0),
)


@dataclass(frozen=True)
class SessionSegment:
    """一个连续交易子时段。``start``/``end`` 是闭区间本地时间标签。"""

    name: str
    start: _dt.time
    end: _dt.time
    offset_min: int  # 本时段第一根 bar 的 session elapsed 下标

    @property
    def bar_count(self) -> int:
        return _minutes_between(self.start, self.end) + 1

    def contains(self, t: _dt.time) -> bool:
        return self.start <= t <= self.end


@dataclass(frozen=True)
class MarketSession:
    """一个市场的完整交易时段定义（含午休）。

    #P0-8：支持 date-specific early close（美股 half day，如 13:00 收市）。
    ``early_close_dates`` 命中时，当日收盘时间 = ``early_close_time``；否则用
    各 segment 的常规 end。
    """

    market: str
    timezone: str
    segments: tuple[SessionSegment, ...]
    early_close_dates: frozenset[_dt.date] = frozenset()
    early_close_time: _dt.time | None = None

    # ---- 纯 Python 侧 ----

    @property
    def total_bars(self) -> int:
        return sum(s.bar_count for s in self.segments)

    def is_early_close(self, d: _dt.date) -> bool:
        """该日是否提前收市（early close / half day）。"""
        return d in self.early_close_dates

    def close_on(self, d: _dt.date) -> _dt.time:
        """该日收盘时间：early close 日取 early_close_time，否则末段 end。"""
        if d in self.early_close_dates and self.early_close_time is not None:
            return self.early_close_time
        if self.segments:
            return self.segments[-1].end
        return _dt.time(0, 0)

    def bar_count_on(self, d: _dt.date) -> int:
        """该日 bar 数：early close 日按提前收盘计算（最后一根 <= early_close）。"""
        if d in self.early_close_dates and self.early_close_time is not None:
            total = 0
            for seg in self.segments:
                if seg.start <= self.early_close_time:
                    end = min(seg.end, self.early_close_time)
                    total += _minutes_between(seg.start, end) + 1
            return max(total, 1)
        return self.total_bars

    def elapsed_index(self, t: _dt.time) -> int | None:
        """把本地时间标签映射到 session elapsed bar index（0 = 当日第一根）。

        不在任何时段内（午休 / 盘前盘后）返回 None。
        """
        for seg in self.segments:
            if seg.contains(t):
                return seg.offset_min + _minutes_between(seg.start, t)
        return None

    def hhmm_elapsed_index(self, hhmm: str) -> int | None:
        try:
            h, m = (int(p) for p in str(hhmm).split(":", 1))
            return self.elapsed_index(_dt.time(h, m))
        except (ValueError, TypeError):
            return None

    def contains_hhmm(self, hhmm: str) -> bool:
        try:
            h, m = (int(p) for p in str(hhmm).split(":", 1))
            return any(s.contains(_dt.time(h, m)) for s in self.segments)
        except (ValueError, TypeError):
            return False

    # ---- SQL 侧（DuckDB） ----

    def sql_local_time_expr(self, col: str) -> str:
        """把 UTC 列转成交易所本地时间的 SQL 表达式（DuckDB timezone 函数）。

        DuckDB 的 ``timezone(tz, ts)`` 要求 ts 为 TIMESTAMP（naive 视为 UTC），
        返回带时区语义的 TIMESTAMPTZ；再对 TIMESTAMPTZ 用 ``strftime`` 取
        本地 HH:MM 时会按 session 时区渲染，正好得到交易所本地标签。
        """
        if self.timezone and self.timezone.upper() not in {"", "UTC", "ETC/UTC"}:
            return f"timezone('{self.timezone}', {_q(col)})"
        return _q(col)

    def sql_elapsed_expr(self, local_min_expr: str) -> str:
        """由「本地时钟分钟数」表达式计算 session elapsed index 的 SQL。

        A 股：elapsed = local_min - 571 - (90 if local_min >= 780 else 0)
        （571 = 09:31 的第一根；780 = 13:00；90 = 午休 90 分钟）。
        多时段时按段拼接 CASE。
        """
        parts = []
        for seg in self.segments:
            lo = seg.start.hour * 60 + seg.start.minute
            hi = seg.end.hour * 60 + seg.end.minute
            within = f"{local_min_expr} BETWEEN {lo} AND {hi}"
            parts.append(
                f"WHEN {within} THEN {local_min_expr} - {lo} + {seg.offset_min}"
            )
        if not parts:
            return f"GREATEST({local_min_expr} - 570, 0)"
        return f"CASE {' '.join(parts)} ELSE -1 END"


def _q(name: str) -> str:
    return f'"{str(name).replace(chr(34), chr(34) * 2)}"'


def _minutes_between(a: _dt.time, b: _dt.time) -> int:
    return (b.hour * 60 + b.minute) - (a.hour * 60 + a.minute)


def build_ashare_session() -> MarketSession:
    return MarketSession(
        market="ashare",
        timezone="Asia/Shanghai",
        segments=tuple(
            SessionSegment(name, s, e, off) for (name, s, e, off) in _ASHARE_SEGMENTS
        ),
    )


def build_us_session(
    early_close_dates: Iterable[_dt.date] | None = None,
) -> MarketSession:
    """#P0-8 美股 session；``early_close_dates`` 提供提前收市日（13:00 收市）。"""
    return MarketSession(
        market="us",
        timezone="America/New_York",
        segments=tuple(
            SessionSegment(name, s, e, off) for (name, s, e, off) in _US_SEGMENTS
        ),
        early_close_dates=frozenset(early_close_dates or ()),
        early_close_time=_dt.time(13, 0),
    )


def load_us_early_close_dates(store: Any) -> frozenset[_dt.date]:
    """#P0-8 从 ``us_is_early_close`` 数据集加载提前收市日期（half day）。

    数据集不存在/读不到时返回空集（session 回退常规 16:00 收市）。
    用 ``mode="dimension"`` 读 STATIC 维表。
    """
    try:
        ds = store.registry.get("us_is_early_close")
    except Exception:
        return frozenset()
    time_col = getattr(ds, "time_column", None) or getattr(ds, "instrument_column", None)
    if not time_col:
        return frozenset()
    try:
        tbl = store.read_arrow(
            "us_is_early_close", columns=[time_col], limit=50_000, mode="dimension"
        )
    except Exception:
        return frozenset()
    if tbl is None or tbl.num_rows == 0:
        return frozenset()
    out: set[_dt.date] = set()
    for v in tbl.column(0).to_pylist():
        d = _as_date(v)
        if d is not None:
            out.add(d)
    return frozenset(out)


def get_market_session(market: str | None) -> MarketSession | None:
    """按市场名取 session 定义；无法识别返回 None（调用方用旧行为）。

    美股 session 不在此处注入 early-close 数据（需要 store 才能读
    ``us_is_early_close``）；需要时用 ``store.get_market_session(market)``
    或 ``get_market_session_with_early_close(market, store)``。
    """
    if not market:
        return None
    key = str(market).strip().lower()
    if key.startswith("ashare") or key == "a":
        return build_ashare_session()
    if key in {"us", "usa", "am", "nyse"}:
        return build_us_session()
    return None


def get_market_session_with_early_close(
    market: str | None, store: Any = None
) -> MarketSession | None:
    """#P0-8 取 session 并注入美股 early-close 日期（store 提供时）。"""
    if not market:
        return None
    key = str(market).strip().lower()
    if key in {"us", "usa", "am", "nyse"}:
        dates = load_us_early_close_dates(store) if store is not None else frozenset()
        return build_us_session(early_close_dates=dates)
    return get_market_session(market)


class MarketCalendar:
    """一个市场的交易日历（本地日期，不含时区）。

    ``source`` 标记数据来源（registry / explicit / fallback），供缓存与
    production fail-closed 判断（#P0-5）：fallback 日历不得被当作权威日历。
    """

    def __init__(
        self,
        market: str,
        *,
        trading_days: Sequence[_dt.date] | None = None,
        timezone: str | None = None,
        holidays: set[_dt.date] | None = None,
        source: str = "unknown",
        session: "MarketSession | None" = None,
    ) -> None:
        self.market = market
        days = sorted(set(trading_days or ()))
        self.trading_days: tuple[_dt.date, ...] = tuple(days)
        self._day_set = set(days)
        # #P0-8 允许注入带 early-close 数据的 session（美股 half day）。
        session = session if session is not None else get_market_session(market)
        self.session = session
        self.timezone = timezone or (session.timezone if session else None) or "UTC"
        self.source = source

    @classmethod
    def from_business_days(
        cls,
        market: str,
        *,
        start: _dt.date | None = None,
        end: _dt.date | None = None,
        holidays: set[_dt.date] | None = None,
        max_years: int = 30,
    ) -> "MarketCalendar":
        """没有真实日历数据的兜底：纯周末休市 + 显式节假日。

        **注意**：这只是兜底，春节/国庆等法定假日不在默认表里。真实部署请
        注入交易所日历（trading_days= 或 registry 的 ashare_calendar）。
        ``source="fallback"``——不得被当作权威日历缓存（#P0-5）。
        """
        import pandas as pd

        start = start or (_dt.date.today() - _dt.timedelta(days=365 * max_years))
        end = end or (_dt.date.today() + _dt.timedelta(days=365))
        if start > end:
            start, end = end, start
        bdays = pd.bdate_range(start, end)
        days = [d.date() for d in bdays]
        if holidays:
            days = [d for d in days if d not in holidays]
        return cls(market, trading_days=days, source="fallback")

    @property
    def has_data(self) -> bool:
        return bool(self.trading_days)

    def is_trading_day(self, d: _dt.date) -> bool:
        return d in self._day_set

    def next_trading_day(self, d: _dt.date) -> _dt.date | None:
        """严格大于 d 的下一个交易日。"""
        for day in self.trading_days:
            if day > d:
                return day
        return None

    def previous_trading_day(self, d: _dt.date) -> _dt.date | None:
        """严格小于 d 的上一个交易日。"""
        prev: _dt.date | None = None
        for day in self.trading_days:
            if day < d:
                prev = day
            else:
                break
        return prev

    def available_from(
        self, knowledge: Any, availability: str = "next_trading_day"
    ) -> Any:
        """把 knowledge 时间编译成「数据真正可用」的时间。

        #P0-7 细粒度 availability：
        - same_instant / same_day / effective_date_only ：可用 = knowledge 本身
        - next_bar / after_close_next_open              ：下一交易日（严格）
        - next_session_open / session                   ：按 session 边界：
          盘前 → 当日 session 起点；盘中/盘后 → 下一 session 起点
        - next_trading_day                              ：下一交易日 00:00
          （knowledge 是日期时）或 下一交易日 session 起点（datetime 时）
        """
        av = str(availability or "same_day").lower()
        if av in {"same_day", "same_instant", "effective_date_only"}:
            return knowledge
        kdate = _as_date(knowledge)
        if kdate is None:
            # 无法识别的 knowledge 时间：回退严格 >=（保守可见）
            return knowledge
        if av in {"session", "next_session_open"}:
            # knowledge 携带时刻时按 session 边界判断
            if isinstance(knowledge, _dt.datetime):
                t = knowledge.time()
                if self.session is not None:
                    for seg in self.session.segments:
                        if t < seg.start:
                            # 该 session 已开盘但 knowledge 在其前 → 当日该段起点
                            return _combine(kdate, seg.start)
                        if seg.contains(t):
                            # 盘中 → 下一段/下一日开盘
                            nxt = self._next_segment_start(seg.name)
                            if nxt is not None:
                                return _combine(kdate, nxt)
                            td = self.next_trading_day(kdate)
                            if td is not None:
                                return _combine(td, self._first_start())
                            return knowledge
            td = self.next_trading_day(kdate)
            return _combine(td, self._first_start()) if td is not None else knowledge
        # next_trading_day（默认）
        td = self.next_trading_day(kdate)
        if td is None:
            return knowledge
        if isinstance(knowledge, _dt.datetime):
            return _combine(td, self._first_start())
        return td

    def _first_start(self) -> _dt.time:
        if self.session is not None and self.session.segments:
            return self.session.segments[0].start
        return _dt.time(0, 0)

    def _next_segment_start(self, seg_name: str) -> _dt.time | None:
        if self.session is None:
            return None
        names = [s.name for s in self.session.segments]
        idx = names.index(seg_name) if seg_name in names else -1
        if 0 <= idx < len(self.session.segments) - 1:
            return self.session.segments[idx + 1].start
        return None

    def sql_next_trading_day_join(self, alias: str = "_cal") -> str:
        """生成一个 (knowledge_date -> next_trading_day) 的 VALUES CTE。

        供 read_joined 把 ``decision > knowledge`` 升级为
        ``decision >= available_from``（日期级）。
        """
        if not self.trading_days:
            raise ValidationError("交易日历为空，无法编译 next_trading_day")
        pairs: list[str] = []
        for d in self.trading_days:
            nxt = self.next_trading_day(d)
            if nxt is None:
                continue
            pairs.append(f"(DATE '{d.isoformat()}', DATE '{nxt.isoformat()}')")
        if not pairs:
            raise ValidationError("交易日历无可用的 next_trading_day 映射")
        return (
            f"SELECT * FROM (VALUES {', '.join(pairs)}) "
            f"AS {alias}(_kd, _next_td)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "timezone": self.timezone,
            "source": self.source,
            "trading_days": len(self.trading_days),
            "first": self.trading_days[0].isoformat() if self.trading_days else None,
            "last": self.trading_days[-1].isoformat() if self.trading_days else None,
        }


def _as_date(value: Any) -> _dt.date | None:
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str):
        try:
            return _dt.date.fromisoformat(str(value)[:10])
        except ValueError:
            return None
    try:
        return _dt.date.fromisoformat(str(value)[:10])
    except Exception:
        return None


def _combine(d: _dt.date, t: _dt.time) -> _dt.datetime:
    return _dt.datetime(d.year, d.month, d.day, t.hour, t.minute, t.second)


def _load_calendar_from_registry(store: Any, market: str) -> tuple[_dt.date, ...] | None:
    """尝试从 registry 的 ashare_calendar / us_calendar 数据集加载真实交易日。

    #P0-4：日历文件可能包含自然日（周末/节假日）——必须按交易日标志过滤：
    A股 ``IsTradeDay=True``，美股 ``is_trading_day=True``。用 ``mode="dimension"``
    读 STATIC 维表，避免 production/strict 下被面板契约拒绝。
    失败（未注册 / 读不到 / 无数据）返回 None，调用方决定回退或 fail-closed。
    """
    dataset = "ashare_calendar" if market == "ashare" else "us_calendar"
    try:
        ds = store.registry.get(dataset)
    except Exception:
        return None
    time_col = ds.time_column or ds.instrument_column
    if not time_col:
        return None
    schema = dict(getattr(ds, "schema", None) or {})
    flag_col = None
    for cand in ("IsTradeDay", "is_trading_day"):
        if cand in schema:
            flag_col = cand
            break
    cols = [time_col] + ([flag_col] if flag_col else [])
    try:
        tbl = store.read_arrow(dataset, columns=cols, limit=100_000, mode="dimension")
    except Exception:
        return None
    if tbl is None or tbl.num_rows == 0:
        return None
    time_vals = tbl.column(0).to_pylist()
    if flag_col:
        flag_vals = tbl.column(1).to_pylist()
        days = [
            _as_date(v)
            for v, flag in zip(time_vals, flag_vals)
            if flag is True and v is not None
        ]
    else:
        days = [_as_date(v) for v in time_vals if v is not None]
    return [d for d in days if d is not None]


def _strict_calendar_mode() -> bool:
    from data_access.read.query_budget import is_strict_semantics

    return is_strict_semantics()


# ---- 进程内单例 ----

_calendars: dict[str, MarketCalendar] = {}


def get_market_calendar(
    market: str,
    *,
    store: Any = None,
    trading_days: Sequence[_dt.date] | None = None,
    holidays: set[_dt.date] | None = None,
    force_reload: bool = False,
) -> MarketCalendar:
    """进程级 MarketCalendar 单例。

    优先使用显式 ``trading_days``；否则尝试从 ``store`` 的日历数据集加载；
    都没有时回退纯工作日日历（含显式 ``holidays``）。

    #P0-5：fallback（business-day）日历**不得污染** authoritative 缓存键——
    缓存里出现 fallback 后，后续带真实 store 的调用可能误命中。fallback 缓存到
    独立的 ``{market}::fallback`` 键。production/strict 下真实日历不可用
    直接 fail-closed，不静默退回 business-day。
    """
    key = str(market).strip().lower()
    if not force_reload and not trading_days:
        cached = _calendars.get(key)
        if cached is not None:
            return cached
    if trading_days:
        cal = MarketCalendar(key, trading_days=trading_days, source="explicit")
        _calendars[key] = cal
        return cal
    if store is not None:
        loaded = _load_calendar_from_registry(store, key)
        if loaded:
            cal = MarketCalendar(
                key,
                trading_days=loaded,
                source="registry",
                session=get_market_session_with_early_close(key, store),
            )
            _calendars[key] = cal
            return cal
    if _strict_calendar_mode():
        raise ValidationError(
            f"market={key!r} 真实交易日历不可用（registry 无 {key}_calendar 或读不到），"
            "production/strict fail-closed：禁止静默退回 business-day 日历。"
        )
    # fallback：独立缓存键，绝不当权威日历
    fkey = f"{key}::fallback"
    cached = _calendars.get(fkey)
    if cached is not None:
        return cached
    cal = MarketCalendar.from_business_days(key, holidays=holidays)
    _calendars[fkey] = cal
    return cal


def reset_calendars() -> None:
    """清空日历缓存（测试用）。"""
    _calendars.clear()


__all__ = [
    "SessionSegment",
    "MarketSession",
    "MarketCalendar",
    "get_market_session",
    "get_market_session_with_early_close",
    "get_market_calendar",
    "load_us_early_close_dates",
    "build_ashare_session",
    "build_us_session",
    "reset_calendars",
]
