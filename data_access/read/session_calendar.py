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
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from data_access.core.exceptions import AvailabilityLatencyError, ValidationError

# #7 唯一 Market canonicalizer：明确 alias → canonical market。未知值直接报错，
# 绝不能 ``else → us``。
_MARKET_ALIASES = {
    "ashare": "ashare", "a_share": "ashare", "a": "ashare", "cn": "ashare",
    "us": "us", "usa": "us", "nyse": "us", "nasdaq": "us", "am": "us",
    "us_stock": "us",
}


def canonicalize_market(market: str | None) -> str | None:
    """把市场名 canonicalize 到 ``ashare`` / ``us``；未知值 fail-closed。

    #7 旧代码 ``_calendar_dataset_for`` 对任何非 ashare 的 market（``europe`` /
    ``abc`` / ``uss``）静默返回 ``us_calendar``——把未知市场当美国市场，PIT /
    session availability / 日历缓存全部被带偏。现在只接受明确 alias。
    """
    if market is None:
        return None
    key = str(market).strip().lower()
    canon = _MARKET_ALIASES.get(key)
    if canon is None:
        raise ValidationError(
            f"未知市场名 {market!r}。仅支持：ashare/a_share/cn（A股）与 "
            "us/usa/nyse/nasdaq（美股）；不能把未知市场静默当美国市场处理。"
        )
    return canon

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

    def effective_segments_on(self, d: _dt.date) -> tuple[SessionSegment, ...]:
        """该日生效的 segment 集合：early-close 日按提前收盘裁剪，否则原样。

        #7 date-specific session 操作（``next_bar`` / ``elapsed`` / ``close`` /
        ``latency``）必须先构造 effective segments，再消费同一套——不能再拿常规
        09:30–16:00 映射 half-day（实际 13:00 收市）的 13:01。
        """
        if d not in self.early_close_dates or self.early_close_time is None:
            return self.segments
        ec = self.early_close_time
        out: list[SessionSegment] = []
        for seg in self.segments:
            if seg.start > ec:
                continue
            end = min(seg.end, ec)
            out.append(SessionSegment(seg.name, seg.start, end, seg.offset_min))
        return tuple(out)

    def elapsed_index(
        self, t: _dt.time, *, on: _dt.date | None = None
    ) -> int | None:
        """把本地时间标签映射到 session elapsed bar index（0 = 当日第一根）。

        ``on`` 提供日期时按该日 effective segments 映射（early-close 日 13:00
        之后不再映射到当天）。不在任何时段内（午休 / 盘前盘后）返回 None。
        """
        segs = self.effective_segments_on(on) if on is not None else self.segments
        for seg in segs:
            if seg.contains(t):
                return seg.offset_min + _minutes_between(seg.start, t)
        return None

    def hhmm_elapsed_index(
        self, hhmm: str, *, on: _dt.date | None = None
    ) -> int | None:
        try:
            h, m = (int(p) for p in str(hhmm).split(":", 1))
            return self.elapsed_index(_dt.time(h, m), on=on)
        except (ValueError, TypeError):
            return None

    def contains_hhmm(self, hhmm: str, *, on: _dt.date | None = None) -> bool:
        try:
            h, m = (int(p) for p in str(hhmm).split(":", 1))
            segs = self.effective_segments_on(on) if on is not None else self.segments
            return any(s.contains(_dt.time(h, m)) for s in segs)
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

    用 ``mode="dimension"`` 读 STATIC 维表。

    #P0-final closure 6：production/strict 下「读不到」**不是**「没有 early
    close」——数据集未注册 / 缺时间列 / 读取失败 / 空结果一律 fail-closed
    （提权成 ValidationError）。否则 July 3 / Thanksgiving 后一天等 half-day
    上 ``after_close_next_open / session availability`` 会按 16:00 正常收盘，
    产生错误的可见时点。research 保持宽容返回空集。
    """
    strict = _strict_calendar_mode()
    try:
        ds = store.registry.get("us_is_early_close")
    except Exception:
        if strict:
            raise ValidationError(
                "us_is_early_close 未注册，无法加载美股 early-close 合约"
                "（production fail-closed：禁止把'没读到'当'没有 early close'）"
            )
        return frozenset()
    time_col = getattr(ds, "time_column", None) or getattr(ds, "instrument_column", None)
    if not time_col:
        if strict:
            raise ValidationError(
                "us_is_early_close 缺时间列，无法加载 early-close 合约"
                "（production fail-closed）"
            )
        return frozenset()
    try:
        tbl = store.read_arrow(
            "us_is_early_close", columns=[time_col], limit=50_000, mode="dimension"
        )
    except Exception:
        if strict:
            raise ValidationError(
                "读取 us_is_early_close 失败（production fail-closed："
                "禁止把'读不到'当'没有 early close'）"
            )
        return frozenset()
    if tbl is None or tbl.num_rows == 0:
        if strict:
            raise ValidationError(
                "us_is_early_close 为空，无法证明美股无 early-close 日"
                "（production fail-closed）"
            )
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
        self,
        knowledge: Any,
        availability: str = "next_trading_day",
        *,
        time_representation: str | None = None,
    ) -> Any:
        """把 knowledge 时间编译成「数据真正可用」的时间。

        #P1-final closure 5：委托给统一 ``compile_available_from``（本模块唯一
        IR）——Python helper / SQL join / PhysicalPlan 全消费同一份编译逻辑。
        ``next_bar`` 现在计算**当前 session 的下一根 bar**（盘中 +1 bar、段末
        下一段起点、收盘/非交易日下一交易日第一根），不再落到通用的
        next_trading_day 分支。

        #11/#12：knowledge 带时刻时**先把 UTC timestamp 转成交易所本地
        时区**再判断 session 边界（美股 filing timestamp 是实质 PIT bug）；
        非交易日绝不生成「当天的开盘」。

        R24 P0-PIT2 §11：``time_representation="date_label"``（filing_date /
        PubDate）按日期标签处理，不做时区换算。
        """
        return compile_available_from(
            knowledge,
            availability,
            calendar=self,
            latency=None,
            time_representation=time_representation,
        )

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
        # #32 每个交易日线性找 next 是 O(N²)；直接 zip(相邻交易日) 即 O(N)。
        pairs: list[str] = []
        days = self.trading_days
        for d, nxt in zip(days, days[1:]):
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


def _add_minutes(t: _dt.time, minutes: int) -> _dt.time:
    minutes %= 1440
    total = (t.hour * 60 + t.minute + minutes) % 1440
    return _dt.time(total // 60, total % 60, t.second)


def _session_first_start(session: "MarketSession | None") -> _dt.time:
    """session 第一根 bar 的本地时间（无 session → 00:00）。"""
    if session is not None and session.segments:
        return session.segments[0].start
    return _dt.time(0, 0)


def _session_next_segment_start(
    session: "MarketSession | None", seg_name: str
) -> _dt.time | None:
    """当前 segment 之后下一个 segment 的起点（无下一段 → None）。"""
    if session is None or not session.segments:
        return None
    names = [s.name for s in session.segments]
    idx = names.index(seg_name) if seg_name in names else -1
    if 0 <= idx < len(session.segments) - 1:
        return session.segments[idx + 1].start
    return None


def _apply_latency(base: Any, latency: int | None, bar_interval_minutes: int) -> Any:
    """#P1-final closure 5 availability_latency：在编译出的 available_from 上
    叠加额外可见性延迟（latency 以 bar 计，默认 1 分钟/bar）。date 结果先转
    当日 00:00 datetime 再叠加——延迟必须落在时刻上。

    非零 latency 无法解析或无法应用时抛 typed error；禁止把未延迟的
    ``available_from`` 当成成功结果返回，否则会提前暴露 PIT 数据。
    """
    try:
        if latency is None:
            return base
        if isinstance(latency, bool) or not isinstance(latency, int) or latency < 0:
            raise TypeError("latency must be a non-negative integer or None")
        if latency == 0:
            return base
        if (
            isinstance(bar_interval_minutes, bool)
            or not isinstance(bar_interval_minutes, int)
            or bar_interval_minutes <= 0
        ):
            raise TypeError("bar_interval_minutes must be a positive integer")
        minutes = latency * bar_interval_minutes
        if isinstance(base, _dt.datetime):
            return base + _dt.timedelta(minutes=minutes)
        if isinstance(base, _dt.date):
            return _dt.datetime(base.year, base.month, base.day, 0, 0) + _dt.timedelta(
                minutes=minutes
            )
        return base + _dt.timedelta(minutes=minutes)
    except Exception as exc:
        raise AvailabilityLatencyError(
            f"availability_latency={latency!r} 无法应用到 available_from={base!r}"
        ) from exc


def _raise_right_boundary(availability: str, kdate: _dt.date) -> None:
    """#14 右边界 fail-closed：日历覆盖不到 knowledge 时禁止「无法证明 ⇒ 现在可用」。

    对 PIT 恰好是反方向：无法证明「数据什么时候可用」就把它当「现在就可用了」，
    会让最新一期数据在真正可见前就被查出来（false-negative 边界）。strict 直接
    抛错，调用方决定补日历数据或显式放宽。
    """
    raise ValidationError(
        f"availability={availability!r}：{kdate.isoformat()} 超出交易日历覆盖"
        "（next_trading_day 返回 None），无法证明可见时点（strict fail-closed）。"
        "请补充日历数据到更晚的日期，或对非 strict 语义使用 same_day/effective_date_only。"
    )


def compile_available_from(
    knowledge: Any,
    availability: str,
    *,
    calendar: "MarketCalendar | None" = None,
    latency: int | None = None,
    bar_interval_minutes: int = 1,
    strict: bool | None = None,
    time_representation: str | None = None,
) -> Any:
    """**统一 AvailabilityCompiler**：把 knowledge 编译成「数据真正可用」的
    ``available_from``。Python helper（``read_cos_events_asof``）、SQL join
    （``_session_avail_sql``）与 PhysicalPlan 组合读全部消费同一个 IR。

    语义：
        - same_instant / same_day / effective_date_only ：knowledge 本身
        - next_bar        ：当前 session 的**下一根 bar**——
            · 盘中 → knowledge + bar_interval（仍在该 segment 内 → 当段下一根）
            · 段末 → 下一段起点（A 股 11:30 → 13:01）
            · 收盘 / 非交易日 / 周末 / 节假日 → 下一交易日第一根
            · 午休 / 盘后 → 下一段起点或下一交易日；盘前 → 当日第一根
        - next_session_open / session ：按 session 边界（盘前→当日段起点；
          盘中→下一段/下一日；收盘→下一交易日）
        - next_trading_day ：下一交易日（knowledge 是日期 → 日期；带时刻 →
          下一交易日 session 开盘）
        - after_close_next_open ：盘后披露 → 下一交易日 session 开盘

    R24 P0-PIT6 §15：calendar-required availability（next_bar / next_session_open /
    next_trading_day / after_close_next_open / session）在**真实日历加载失败时
    production/strict 必须 hard fail**，禁止 ``logger.warning → fallback same_day``。
    Research 需要显式 ``allow_temporal_degradation``（见 ``MarketCalendar.available_from``
    调用方）才回退 knowledge，且 lineage 要标 ``temporal_semantics_degraded=true``。

    R24 P0-PIT2 §11：``time_representation="date_label"`` 时 knowledge 按日期标签
    处理（不做时区换算）。``latency``（bar 数）叠加在结果上：额外可见性延迟。
    """
    return compile_available_from_result(
        knowledge,
        availability,
        calendar=calendar,
        latency=latency,
        bar_interval_minutes=bar_interval_minutes,
        strict=strict,
        time_representation=time_representation,
    ).available_from


def compile_available_from_result(
    knowledge: Any,
    availability: str,
    *,
    calendar: "MarketCalendar | None" = None,
    latency: int | None = None,
    bar_interval_minutes: int = 1,
    strict: bool | None = None,
    time_representation: str | None = None,
    calendar_snapshot_id: str | None = None,
) -> "AvailabilityResult":
    """R25 P0-005：AvailabilityCompiler 的结构化版本，返回 ``AvailabilityResult``。

    生产（authoritative=False → 调用方拒绝）+ 研究（可降级，但 lineage 必须记录
    ``degradation_reason``）。关键顺序（R25 §8）：
        1. 先决定 strict（缺省用 ``is_strict_semantics()``）；
        2. calendar-required availability（next_bar / next_session_open /
           next_trading_day / after_close_next_open / session）+ calendar 缺失
           → strict 抛 ``CalendarUnavailableError``；research 返回
           ``AvailabilityResult(authoritative=False, degradation_reason=...)``，
           **绝不** return raw knowledge 当成已可用。
    """
    from data_access.contract.temporal_axis import AvailabilityResult
    from data_access.core.exceptions import CalendarUnavailableError

    av = str(availability or "same_day").lower()
    if av in {"same_instant", "same_day", "effective_date_only"}:
        return AvailabilityResult(
            available_from=_apply_latency(knowledge, latency, bar_interval_minutes),
            authoritative=True,
            calendar_snapshot_id=calendar_snapshot_id,
        )
    if strict is None:
        strict = _strict_calendar_mode()
    if calendar is None or not getattr(calendar, "has_data", False):
        # R25 P0-005：calendar-required availability 必须真实日历。strict 下
        # 日历不可用 = 无法证明可见时点 → 抛 CalendarUnavailableError（不是裸
        # ValidationError，供调用方按 §113 reject）。research 返回 degraded
        # AvailabilityResult（authoritative=False + degradation_reason）。
        if strict:
            raise CalendarUnavailableError(
                f"availability={av!r} 需要交易日历，但日历不可用"
                "（calendar=None / 无交易日）。production/strict fail-closed："
                "禁止 same_day fallback（look-ahead）。请补齐日历数据。"
            )
        return AvailabilityResult(
            available_from=knowledge,
            authoritative=False,
            calendar_snapshot_id=calendar_snapshot_id,
            degradation_reason=f"availability={av!r} requires calendar but calendar unavailable",
        )
    session = getattr(calendar, "session", None)
    kdate, ktime = _to_local_date_time(
        knowledge, calendar.timezone, time_representation=time_representation
    )
    if kdate is None:
        if strict:
            raise CalendarUnavailableError(
                f"compile_available_from({availability!r}): knowledge 无法解析成"
                f"交易所本地日期（{knowledge!r}），无法证明可见时点（strict fail-closed）"
            )
        return AvailabilityResult(
            available_from=knowledge,
            authoritative=False,
            calendar_snapshot_id=calendar_snapshot_id,
            degradation_reason=f"knowledge {knowledge!r} unresolvable to local date",
        )
    if av == "next_bar":
        base = _compile_next_bar(knowledge, kdate, ktime, calendar, session, strict=strict)
    elif av in {"session", "next_session_open"}:
        base = _compile_next_session_open(
            knowledge, kdate, ktime, calendar, session, strict=strict
        )
    elif av == "after_close_next_open":
        # 盘后披露：knowledge 在任一时刻，可用 = 下一交易日的 session 开盘
        td = calendar.next_trading_day(kdate)
        if td is not None and session is not None and session.segments:
            base = _combine(td, _session_first_start(session))
        else:
            if td is None and strict:
                _raise_right_boundary(av, kdate)
            base = knowledge
            if td is None:
                return AvailabilityResult(
                    available_from=knowledge,
                    authoritative=False,
                    calendar_snapshot_id=calendar_snapshot_id,
                    degradation_reason=f"availability={av!r}: calendar right boundary unreachable",
                )
    else:  # next_trading_day（默认）
        td = calendar.next_trading_day(kdate)
        if td is None:
            if strict:
                _raise_right_boundary(av, kdate)
            return AvailabilityResult(
                available_from=knowledge,
                authoritative=False,
                calendar_snapshot_id=calendar_snapshot_id,
                degradation_reason=f"availability={av!r}: calendar right boundary unreachable",
            )
        elif ktime is not None and session is not None and session.segments:
            base = _combine(td, _session_first_start(session))
        else:
            base = td
    return AvailabilityResult(
        available_from=_apply_latency(base, latency, bar_interval_minutes),
        authoritative=True,
        calendar_snapshot_id=calendar_snapshot_id,
    )


def _segment_after(
    segs: Sequence[SessionSegment], seg_name: str
) -> _dt.time | None:
    """effective segments 中某 segment 之后下一 segment 的起点（无 → None）。"""
    names = [s.name for s in segs]
    idx = names.index(seg_name) if seg_name in names else -1
    if 0 <= idx < len(segs) - 1:
        return segs[idx + 1].start
    return None


def _compile_next_bar(
    knowledge: Any,
    kdate: _dt.date,
    ktime: _dt.time | None,
    calendar: "MarketCalendar",
    session: "MarketSession | None",
    *,
    strict: bool = False,
) -> Any:
    """next_bar：当前 session 的下一根 bar（含午休跨段 / 收盘跨日）。

    #7 消费该日 **effective segments**（``session.effective_segments_on(kdate)``）：
    early-close 日（美股 half-day，实际 13:00 收市）knowledge=13:00 的 next_bar
    必须跳到下一交易日第一根，而不是常规 segment 里的 13:01。
    """
    if session is None or not session.segments:
        td = calendar.next_trading_day(kdate)
        if td is not None:
            return _combine(td, _dt.time(0, 0))
        if strict:
            _raise_right_boundary("next_bar", kdate)
        return knowledge
    segs = session.effective_segments_on(kdate)
    if not segs:
        td = calendar.next_trading_day(kdate)
        if td is not None:
            return _combine(td, _session_first_start(session))
        if strict:
            _raise_right_boundary("next_bar", kdate)
        return knowledge
    if ktime is None:
        # 只有日期：默认当天开盘后第一根（当天是交易日）→ 当日第一根
        if calendar.is_trading_day(kdate):
            return _combine(kdate, segs[0].start)
        td = calendar.next_trading_day(kdate)
        if td is None:
            if strict:
                _raise_right_boundary("next_bar", kdate)
            return knowledge
        return _combine(td, segs[0].start)
    if not calendar.is_trading_day(kdate):
        # 非交易日（周末/节假日）：绝无「当天开盘」，直接下一交易日第一根
        td = calendar.next_trading_day(kdate)
        if td is None:
            if strict:
                _raise_right_boundary("next_bar", kdate)
            return knowledge
        return _combine(td, segs[0].start)
    for seg in segs:
        if seg.contains(ktime):
            nxt = _add_minutes(ktime, 1)  # 分钟级 session 默认 1 根 = 1 分钟
            if nxt <= seg.end:
                return _combine(kdate, nxt)
            # 段末最后一根 → 下一段起点（午休跨段）；无下一段 → 下一交易日
            nxt_seg = _segment_after(segs, seg.name)
            if nxt_seg is not None:
                return _combine(kdate, nxt_seg)
            td = calendar.next_trading_day(kdate)
            if td is None:
                if strict:
                    _raise_right_boundary("next_bar", kdate)
                return knowledge
            return _combine(td, segs[0].start)
    # 不在任何 effective segment 内：盘前 → 当日第一根；午休/盘后 → 下一段/下一交易日
    if ktime < segs[0].start:
        return _combine(kdate, segs[0].start)
    for seg in segs:
        if seg.start > ktime:
            return _combine(kdate, seg.start)
    td = calendar.next_trading_day(kdate)
    if td is None:
        if strict:
            _raise_right_boundary("next_bar", kdate)
        return knowledge
    return _combine(td, segs[0].start)


def _compile_next_session_open(
    knowledge: Any,
    kdate: _dt.date,
    ktime: _dt.time | None,
    calendar: "MarketCalendar",
    session: "MarketSession | None",
    *,
    strict: bool = False,
) -> Any:
    """next_session_open / session：按 session 边界映射（early-close 日消费
    effective segments，与 ``next_bar`` 同一套）。"""
    if session is None or not session.segments:
        td = calendar.next_trading_day(kdate)
        if td is not None:
            return _combine(td, _dt.time(0, 0))
        if strict:
            _raise_right_boundary("next_session_open", kdate)
        return knowledge
    segs = session.effective_segments_on(kdate)
    if not segs:
        td = calendar.next_trading_day(kdate)
        if td is not None:
            return _combine(td, _session_first_start(session))
        if strict:
            _raise_right_boundary("next_session_open", kdate)
        return knowledge
    if ktime is not None:
        if not calendar.is_trading_day(kdate):
            td = calendar.next_trading_day(kdate)
            if td is None:
                if strict:
                    _raise_right_boundary("next_session_open", kdate)
                return knowledge
            return _combine(td, segs[0].start)
        for seg in segs:
            if ktime < seg.start:
                # 该 session 已开盘但 knowledge 在其前 → 当日该段起点
                return _combine(kdate, seg.start)
            if seg.contains(ktime):
                nxt = _segment_after(segs, seg.name)
                if nxt is not None:
                    return _combine(kdate, nxt)
                td = calendar.next_trading_day(kdate)
                if td is None:
                    if strict:
                        _raise_right_boundary("next_session_open", kdate)
                    return knowledge
                return _combine(td, segs[0].start)
    td = calendar.next_trading_day(kdate)
    if td is None:
        if strict:
            _raise_right_boundary("next_session_open", kdate)
        return knowledge
    return _combine(td, segs[0].start)


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


def _to_local_date_time(
    value: Any,
    timezone: str | None,
    *,
    time_representation: str | None = None,
) -> tuple[_dt.date | None, _dt.time | None]:
    """把 knowledge 归一化成交易所本地 (date, time)。

    #11：带时区的 datetime 先用 ``ZoneInfo(market timezone)`` 转成本地时间；
    naive datetime 按字典约定视为 **UTC 存储**，同样先转本地（A 股 QuoteTime /
    美股 filing timestamp 都存 UTC）。date 无时刻 → (d, None)。

    R24 P0-PIT2 §11 / T-P07：``time_representation="date_label"`` 时，naive
    datetime（如 ``filing_date = 2024-05-27 00:00:00``）是**日期标签伪装成
    timestamp**——绝不当 UTC instant 转纽约（会提前一天变成 2024-05-26
    20:00）。date_label 直接用其日期；只有 ``instant`` 才做时区换算。
    """
    if not isinstance(value, _dt.datetime):
        d = _as_date(value)
        return (d, None)
    if time_representation == "date_label":
        # 日期标签：忽略时刻（通常 00:00），直接用日期——不做时区换算。
        return value.date(), None
    tz = None
    tz_name = (timezone or "").strip()
    if tz_name and tz_name.upper() not in {"", "UTC", "ETC/UTC"}:
        try:
            from zoneinfo import ZoneInfo

            tz = ZoneInfo(tz_name)
        except Exception:
            tz = None
    if value.tzinfo is not None:
        if tz is not None:
            value = value.astimezone(tz)
        local = value.replace(tzinfo=None)
    elif tz is not None:
        try:
            from zoneinfo import ZoneInfo

            local = value.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz).replace(
                tzinfo=None
            )
        except Exception:
            local = value
    else:
        local = value
    return local.date(), local.time()


def _combine(d: _dt.date, t: _dt.time) -> _dt.datetime:
    return _dt.datetime(d.year, d.month, d.day, t.hour, t.minute, t.second)


def _calendar_dataset_for(market: str) -> str:
    """market → registry 日历数据集名（缓存 key / manifest token 共用）。

    #7 未知 market fail-closed（经 ``canonicalize_market``），不再静默 us_calendar。
    """
    canon = canonicalize_market(market)
    if canon is None:
        raise ValidationError(
            f"无法为 market={market!r} 定位交易日历数据集（market 未知）"
        )
    return "ashare_calendar" if canon == "ashare" else "us_calendar"


def _calendar_source_token(store: Any, market: str) -> str | None:
    """日历数据集当前的 source_epoch（用于缓存 key 失效判断，#31）。

    日历数据更新 → source_epoch 变化 → 缓存 key 变化 → 自动 reload。无
    manifest 返回 None（调用方用 store 注册表指纹兜底）。
    """
    try:
        token = store.manifest_version(_calendar_dataset_for(market))
    except Exception:
        return None
    return token.get("source_epoch") or token.get("manifest_epoch")


def _calendar_file_token(store: Any, market: str) -> str | None:
    """无 manifest 时的日历数据版本 token：日历数据集文件的 path+size+mtime_ns。

    #7 旧退路是 ``store.registry_fingerprint()``——registry 是**配置**版本不是
    **数据**版本：``us_calendar.parquet`` 更新了、registry.yaml 没变、无 manifest
    时，长驻 worker 的 cache key 不变，一直拿旧 trading days（holiday/临时休市/
    PIT 都被带偏）。现在按实际文件 stat 出 token；remote/读不到返回 None 兜底。
    """
    try:
        dataset = _calendar_dataset_for(market)
        ds = store.registry.get(dataset)
    except Exception:
        return None
    root = getattr(ds, "root", None) or getattr(ds, "root_template", None)
    if not root:
        return None
    try:
        from data_access.registry.paths import resolve_namespace_path

        root_path = Path(resolve_namespace_path(str(root))).expanduser()
    except Exception:
        return None
    if not root_path.is_dir():
        return None  # remote/未同步：无法本地 stat，兜底 registry_fingerprint
    glob = getattr(ds, "glob", None) or "*.parquet"
    parts: list[str] = []
    try:
        for p in sorted(root_path.glob(glob)):
            try:
                st = p.stat()
                parts.append(f"{p.name}:{st.st_size}:{st.st_mtime_ns}")
            except OSError:
                continue
    except OSError:
        return None
    if not parts:
        return None
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return f"files:{digest[:24]}:{len(parts)}"


def _load_calendar_from_registry(store: Any, market: str) -> tuple[_dt.date, ...] | None:
    """尝试从 registry 的 ashare_calendar / us_calendar 数据集加载真实交易日。

    #P0-4：日历文件可能包含自然日（周末/节假日）——必须按交易日标志过滤：
    A股 ``IsTradeDay=True``，美股 ``is_trading_day=True``。用 ``mode="dimension"``
    读 STATIC 维表，避免 production/strict 下被面板契约拒绝。
    失败（未注册 / 读不到 / 无数据）返回 None，调用方决定回退或 fail-closed。

    #P0-final closure 7：strict 下 calendar 契约**必须**能证明是
    trading-days-only——schema 或物理列里有 ``IsTradeDay / is_trading_day`` 标志。
    缺标志 → fail-closed（不能再把自然日当交易日、然后当成权威日历用）。
    """
    dataset = _calendar_dataset_for(market)
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
    # schema 未声明 flag 时，probe 物理列（mode="dimension" 读首行看列名）。
    if flag_col is None:
        for cand in ("IsTradeDay", "is_trading_day"):
            try:
                probe = store.read_arrow(
                    dataset, columns=[time_col, cand], limit=1, mode="dimension"
                )
                if probe is not None and cand in (probe.column_names or ()):
                    flag_col = cand
                    break
            except Exception:
                continue
        if flag_col is None and _strict_calendar_mode():
            raise ValidationError(
                f"calendar 数据集 {dataset!r} 缺少交易日标志列 "
                "(IsTradeDay/is_trading_day)，strict fail-closed："
                "不能把自然日/节假日当交易日。请在 schema 或物理数据声明标志列。"
            )
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

    #31：缓存 key 不只按 market——同一 market 的 Store A / Store B 日历数据可能
    不同，日历更新后也应失效。key = ``{market}:{source_token}``：
        - 显式 trading_days → 内容指纹（长度+首末日）
        - store registry 且日历数据集有 manifest → source_epoch
        - store registry 无 manifest → 日历文件 snapshot（path+size+mtime_ns），
          #7 不再用 registry_fingerprint（那是配置版本，不是数据版本）
    日历数据更新 → token 变化 → key 变化 → 自动 reload。

    #7 market 未知 fail-closed：未知名不得静默当美国市场。
    """
    key = canonicalize_market(market)
    if key is None:
        raise ValidationError("market 未提供，无法加载交易日历")
    if trading_days:
        days_sorted = sorted(set(trading_days))
        token = (
            f"explicit:{len(days_sorted)}:"
            f"{days_sorted[0].isoformat() if days_sorted else '-'}:"
            f"{days_sorted[-1].isoformat() if days_sorted else '-'}"
        )
    elif store is not None:
        token = _calendar_source_token(store, key)
        if token is None:
            # #7 无 manifest → 文件 snapshot（path+size+mtime_ns）做数据版本 token，
            # 不用 registry_fingerprint（配置版本，数据更新了它不变 → 长驻 worker
            # 一直拿旧 trading days）。remote 读不到本地文件时返回 None 走兜底。
            token = _calendar_file_token(store, key)
        if token is None:
            try:
                token = f"store:{store.registry_fingerprint()[:16]}"
            except Exception:
                token = None
    else:
        token = None
    cache_key = f"{key}:{token}" if token else key
    if not force_reload and not trading_days:
        cached = _calendars.get(cache_key)
        if cached is not None:
            return cached
    if trading_days:
        cal = MarketCalendar(key, trading_days=trading_days, source="explicit")
        _calendars[cache_key] = cal
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
            _calendars[cache_key] = cal
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
    "compile_available_from",
    "compile_available_from_result",
]
