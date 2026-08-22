"""
data_access.read.aggregation —— 分钟→日频聚合下推（AggregationSpec / AggregationBundle）

背景
    日频因子里的分钟聚合（09:31~10:00 volume、14:30~15:00 return、minute_at
    等）以前在 FactorEngine 里 Pandas 做：整份分钟数据过 Arrow/Pandas 再
    groupby。本模块把聚合下沉到 DuckDB（#14）：只把聚合结果拉回来。

A 股分钟线事实（COS_ashare_lqtp_data_dictionary.md）
    - ``QuoteTime`` 存 **UTC**，北京时间 = UTC+8（01:31Z = 09:31 CST）；
    - CST 时段 **09:31–11:30** 与 **13:01–15:00**，共 240 根；**无 09:30/13:00
      bar**（集合竞价无独立 bar，开盘价体现在首根分钟）；
    - ``minute_of_day`` 的桶号是 **session elapsed bar index**（09:31=0 …
      11:30=119，13:01=120 … 15:00=239），不是自然时钟分钟差（午后多算 90 分钟）。

因此 ``AggregationSpec`` 绑定 ``market``/``timezone`` 后：
    - minute_at("09:31") 匹配北京 09:31（先 ``timezone('Asia/Shanghai', t)``），
      而不是 UTC 09:31；
    - minute_of_day 用 session elapsed index 分桶（#5）。

Multi-aggregation coalescing（#6）
    ``aggregate_minute_bundle``：同一 (dataset, time_range, instruments) 只
    **scan 一次**，每个输出列用 ``agg(val) FILTER (WHERE <spec 条件>)`` 在
    同一条 GROUP BY 里算出几十列（morning_volume / morning_return /
    close_30m_vol / intraday_high / range ...）。A 股单日分钟约 122.76 万行，
    重复扫描代价极高，一次 scan 收益巨大。

治理（#7）
    两个聚合入口都接受 ``query_budget``，走 ``deadline_ms`` + ``enforce_arrow_budget``
    + ``audit.record``，并返回带 ``DataSnapshot`` 的 ``ReadHandle``（lineage 复现）。

维护人：quant 基础平台组    最后更新：2026-08-08
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from data_access.core.exceptions import ValidationError
from data_access.read.minute_filter import FilterSignature, hhmm_to_minutes

_VALID_AGGREGATIONS = frozenset({"minute_at", "minute_range", "minute_of_day"})
_METRIC_FUNCS = {
    "first": "arg_min(value, ts)",      # 每 (date, instrument) 最早分钟的值
    "last": "arg_max(value, ts)",       # 最晚分钟的值
    "min": "min(value)",
    "max": "max(value)",
    "sum": "sum(value)",
    "mean": "avg(value)",
    "count": "count(value)",
}


def _semantic_metric(field: str) -> str:
    """按字段名推断聚合度量（open→first, high→max, low→min, volume/amount→sum）。"""
    key = field.lower()
    if key.endswith("open"):
        return "first"
    if key.endswith("high"):
        return "max"
    if key.endswith("low"):
        return "min"
    if key.endswith("volume") or key.endswith("amount"):
        return "sum"
    return "last"


@dataclass(frozen=True)
class AggregationSpec:
    """一次分钟→日聚合的完整规格。

    参数:
        aggregation: minute_at / minute_range / minute_of_day
        start: HH:MM（minute_range 下界，闭区间）
        end: HH:MM（minute_range 上界，闭区间）
        hhmm: HH:MM（minute_at 指定分钟）
        metric: first/last/min/max/sum/mean/count；缺省按字段语义推断
        period: 分钟桶宽（minute_of_day 用，按 session elapsed index 分桶）
        index: 取第几个桶（0 为最新，与 FE minute_bar 对齐）
        market: ashare/us（决定时区与 session 时段；None 保持旧行为不换算）
        timezone: 显式时区（优先于 market 推断，如 Asia/Shanghai）
    """

    aggregation: str = "minute_range"
    start: str | None = None
    end: str | None = None
    hhmm: str | None = None
    metric: str | None = None
    period: int = 5
    index: int = 0
    market: str | None = None
    timezone: str | None = None

    def __post_init__(self) -> None:
        """R29-P0：直接构造（不经 ``parse_aggregation_spec``）也强制校验。

        封 programmatic bypass：``AggregationSpec(aggregation="abc", period=-3,
        index=-8, market="mars", timezone="xxx")`` 不能再绕过 parser 的严格
        检查——与 TemporalJoinSpec 的 invariant 下沉同理。空默认构造
        （``AggregationSpec()`` = minute_range、无边界）保持合法（AggregationItem
        的 default_factory 依赖它）；跨字段边界只在显式提供了字段时校验。
        """
        agg = self.aggregation
        if agg not in _VALID_AGGREGATIONS:
            raise ValidationError(
                f"聚合 '{agg}' 不支持；合法: {sorted(_VALID_AGGREGATIONS)}"
            )
        if self.market is not None and str(self.market).strip().lower() not in {
            "ashare",
            "us",
        }:
            raise ValidationError(f"聚合 market={self.market!r} 非法（应为 ashare/us）")
        if self.metric is not None and self.metric not in _METRIC_FUNCS:
            raise ValidationError(
                f"不支持的分钟聚合 metric '{self.metric}'；合法: {sorted(_METRIC_FUNCS)}"
            )
        # 与 parser 同语义：strict int（禁 bool/小数截断/负数）、HH:MM、IANA 时区。
        object.__setattr__(
            self, "period", _strict_int(self.period, context="聚合 period", minimum=1)
        )
        object.__setattr__(
            self, "index", _strict_int(self.index, context="聚合 index", minimum=0)
        )
        object.__setattr__(
            self, "timezone", _validate_timezone(self.timezone, context="聚合 timezone")
        )
        object.__setattr__(self, "start", _parse_hhmm(self.start, context="聚合 start"))
        object.__setattr__(self, "end", _parse_hhmm(self.end, context="聚合 end"))
        object.__setattr__(self, "hhmm", _parse_hhmm(self.hhmm, context="聚合 hhmm"))
        # 跨字段：只在显式提供了边界时校验（空默认保持合法）。
        if agg == "minute_range" and (self.start is None) != (self.end is None):
            raise ValidationError("minute_range 需要 start 和 end（二者必须同时提供）")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValidationError(
                f"minute_range start={self.start} 必须早于 end={self.end}"
            )
        if (
            agg == "minute_at"
            and self.hhmm is None
            and (self.start is not None or self.end is not None)
        ):
            raise ValidationError("minute_at 需要 hhmm")

    def effective_metric(self, field: str) -> str:
        m = self.metric or _semantic_metric(field)
        if m not in _METRIC_FUNCS:
            raise ValidationError(
                f"不支持的分钟聚合 metric '{m}'；合法: {sorted(_METRIC_FUNCS)}"
            )
        return m

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregation": self.aggregation,
            "start": self.start,
            "end": self.end,
            "hhmm": self.hhmm,
            "metric": self.metric,
            "period": self.period,
            "index": self.index,
            "market": self.market,
            "timezone": self.timezone,
        }


def _parse_hhmm(value: Any, *, context: str) -> str | None:
    r"""#P0-56 HH:MM 严格格式：``^([01]\d|2[0-3]):[0-5]\d$``。"""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError(f"{context} 必须是 HH:MM 字符串，收到 {value!r}")
    text = value.strip()
    import re as _re

    if not _re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", text):
        raise ValidationError(f"{context} 必须是 HH:MM（00:00~23:59），收到 {text!r}")
    return text


def _strict_int(value: Any, *, context: str, minimum: int) -> int:
    """#P0-56 严格整数：禁止 ``int("5.5")`` 静默截断 / bool 混入 / 负数静默修正。"""
    if isinstance(value, bool):
        raise ValidationError(f"{context} 必须是整数，不能是 bool")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{context} 必须是整数，收到 {value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValidationError(f"{context} 必须是整数，收到 {value!r}")
    if out < minimum:
        raise ValidationError(f"{context} 必须 >= {minimum}，收到 {out}")
    return out


def _validate_timezone(tz: str | None, *, context: str) -> str | None:
    """#P0-55 timezone 只接受合法 IANA 时区（ZoneInfo 可解析）。"""
    if tz is None:
        return None
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        ZoneInfo(tz)
    except ZoneInfoNotFoundError as exc:
        raise ValidationError(f"{context} 不是合法 IANA 时区: {tz!r}") from exc
    return tz


def _sql_string_literal(value: str) -> str:
    """#P0-55 SQL 字符串字面量转义（timezone 不再直接 f-string 插入）。"""
    return "'" + value.replace("'", "''") + "'"


def parse_aggregation_spec(raw: Any, *, field: str) -> AggregationSpec:
    """把 str / dict / AggregationSpec 归一化。

    #P0-56 严格校验：period>0 / index>=0 / HH:MM 格式 / market 枚举 / timezone
    IANA 合法 / unknown-key reject——禁止 silent clamp（``max(1, period)`` 之类的
    静默修正把 ``period=-100`` 悄悄变 1）。
    """
    if isinstance(raw, AggregationSpec):
        return raw
    if isinstance(raw, str):
        return AggregationSpec(aggregation=raw, metric=_semantic_metric(field))
    if not isinstance(raw, Mapping):
        raise ValidationError("聚合规格必须是 str / dict / AggregationSpec")
    _KNOWN_KEYS = {
        "aggregation",
        "start",
        "end",
        "hhmm",
        "metric",
        "period",
        "index",
        "market",
        "timezone",
    }
    unknown = sorted(set(raw) - _KNOWN_KEYS)
    if unknown:
        raise ValidationError(
            f"聚合规格（field={field}）未知 key {unknown}；应为 {sorted(_KNOWN_KEYS)}"
        )
    agg = str(raw.get("aggregation", "minute_range"))
    if agg not in _VALID_AGGREGATIONS:
        raise ValidationError(
            f"聚合 '{agg}' 不支持；合法: {sorted(_VALID_AGGREGATIONS)}"
        )
    market = raw.get("market")
    if market is not None:
        market = str(market).strip().lower()
        if market not in {"ashare", "us"}:
            raise ValidationError(f"聚合 market={market!r} 非法（应为 ashare/us）")
    timezone = _validate_timezone(
        str(raw["timezone"]).strip() if raw.get("timezone") is not None else None,
        context=f"聚合（field={field}）.timezone",
    )
    period = _strict_int(
        raw.get("period", 5), context=f"聚合（field={field}）.period", minimum=1
    )
    index = _strict_int(
        raw.get("index", 0), context=f"聚合（field={field}）.index", minimum=0
    )
    return AggregationSpec(
        aggregation=agg,
        start=_parse_hhmm(raw.get("start"), context=f"聚合（field={field}）.start"),
        end=_parse_hhmm(raw.get("end"), context=f"聚合（field={field}）.end"),
        hhmm=_parse_hhmm(raw.get("hhmm"), context=f"聚合（field={field}）.hhmm"),
        metric=str(raw["metric"]) if raw.get("metric") is not None else None,
        period=period,
        index=index,
        market=market,
        timezone=timezone,
    )


@dataclass(frozen=True)
class AggregationItem:
    """#6 一个聚合输出：``field`` + ``spec`` + ``output_name``。"""

    field: str
    spec: Any = field(default_factory=lambda: AggregationSpec())
    output_name: str | None = None

    def effective_output_name(self, spec: AggregationSpec) -> str:
        if self.output_name:
            return self.output_name
        base = str(self.field).lower()
        if spec.aggregation == "minute_at":
            return f"{base}_{str(spec.hhmm or '').replace(':', '')}"
        if spec.aggregation == "minute_range":
            return (
                f"{base}_{str(spec.start or '').replace(':', '')}_"
                f"{str(spec.end or '').replace(':', '')}"
            )
        return f"{base}_{spec.aggregation}"


@dataclass(frozen=True)
class ParsedAggregationItem:
    """R39-PERF-033：已 parse 的 bundle 条目（AggregationItem + 归一化后的 spec）。

    从 ``aggregate_minute_bundle`` 的校验阶段一路传给 SQL builder，避免
    ``_build_bundle_sql`` 对每个 item 重复 ``parse_aggregation_spec``。frozen 保证
    bundle 内多次复用安全。
    """

    item: AggregationItem
    spec: AggregationSpec


def _quote_ident(name: str) -> str:
    return f'"{name.replace(chr(34), chr(34) * 2)}"'


def _session_timezone(market: str | None, timezone: str | None) -> str | None:
    """确定生效时区：显式 timezone 优先，其次 market session 时区。"""
    if timezone and timezone.strip():
        return timezone.strip()
    if market:
        from data_access.read.session_calendar import get_market_session

        session = get_market_session(market)
        if session is not None:
            return session.timezone
    return None


def _local_time_expr(
    t_col: str,
    market: str | None,
    timezone: str | None,
    *,
    time_is_tz: bool = False,
) -> str:
    """把 UTC 时间列转成交易所本地时间的 SQL 表达式（#5）。

    ``t_col`` 是原始列名（未引用），内部统一 ``_quote_ident``。
    - ``time_is_tz=True``（列是 TIMESTAMPTZ，如字典 ``timestamp[ms,tz=UTC]``）：
      直接 ``timezone(tz, col)``。
    - ``time_is_tz=False``（naive TIMESTAMP，按约定存 UTC）：先用
      ``col AT TIME ZONE 'UTC'`` 把它解释成 UTC 的 TIMESTAMPTZ，再
      ``timezone(tz, ...)``——不依赖 session timezone。
    """
    tz = _session_timezone(market, timezone)
    if tz and tz.upper() not in {"UTC", "ETC/UTC"}:
        # #P0-55 tz 已经过 ZoneInfo 校验；SQL 里仍走统一字符串字面量转义，
        # 不直接 f-string 插值（防止含单引号的异常时区名拼接进 SQL）。
        tz_literal = _sql_string_literal(tz)
        if time_is_tz:
            return f"timezone({tz_literal}, {_quote_ident(t_col)})"
        return (
            f"timezone({tz_literal}, {_quote_ident(t_col)} AT TIME ZONE 'UTC')"
        )
    return _quote_ident(t_col)


def _minute_from_local_expr(local_expr: str) -> str:
    """PERF-034：把「本地时间表达式/列名」包成整数分钟表达式。

    与 ``strftime('%H:%M')`` 同语义（分钟向下取整）：09:30:59 → 570（排除），
    09:31:00 → 571（包含）。``local_expr`` 应为已经过时区换算的表达式或 CTE 列名
    （如 ``_local``），避免每列重复应用 timezone()。
    """
    return f"(EXTRACT(HOUR FROM {local_expr}) * 60 + EXTRACT(MINUTE FROM {local_expr}))"


def _elapsed_expr(
    local_min_expr: str, market: str | None, timezone: str | None
) -> str:
    """session elapsed bar index（0 = 当日第一根 bar）。"""
    if market:
        from data_access.read.session_calendar import get_market_session

        session = get_market_session(market)
        if session is not None:
            return session.sql_elapsed_expr(local_min_expr)
    # 旧行为（无 market）：09:30=570 分钟为 0 号桶
    return f"GREATEST({local_min_expr} - 570, 0)"


def _session_total_bars(market: str | None) -> int:
    if market:
        from data_access.read.session_calendar import get_market_session

        session = get_market_session(market)
        if session is not None:
            return session.total_bars
    return 240


def _spec_filter_sql(
    spec: AggregationSpec,
    *,
    minute_expr: str,
    elapsed_expr: str,
    market: str | None,
) -> str | None:
    """构建单个 spec 的 FILTER 条件；None 表示不过滤（全时段聚合）。

    PERF-034：minute_at / minute_range 改用整数分钟比较（``minute_expr >= 571``），
    不再 ``strftime('%H:%M')`` 字符串比较；边界语义逐字相同（分钟向下取整）。
    PERF-035：经 ``FilterSignature`` 规范化，相同窗口的签名可去重共享。
    整数分钟/桶号都是校验过的整数值，直接内联 SQL 字面量（无注入风险），不再
    产生绑定参数。
    """
    return FilterSignature.canonicalize(
        spec, session_total_bars=_session_total_bars(market)
    ).to_sql(minute_expr=minute_expr, elapsed_expr=elapsed_expr)


def build_minute_aggregation_sql(
    *,
    time_column: str,
    instrument_column: str,
    value_column: str,
    field: str,
    spec: AggregationSpec,
    time_is_tz: bool = False,
) -> tuple[str, list[Any]]:
    """构建分钟→日聚合的 DuckDB SQL；返回 (sql, params)。

    输出列：``ts``（日期）+ ``inst`` + ``value``。
    ``spec.market``/``spec.timezone`` 提供时把 UTC QuoteTime 转本地再取 HH:MM
    （#5）；``minute_of_day`` 用 session elapsed index 分桶。
    ``time_is_tz``：时间列是否为 TIMESTAMPTZ（字典 ``timestamp[ms,tz=UTC]``）。
    """
    metric = spec.effective_metric(field)
    inst = _quote_ident(instrument_column)
    val = _quote_ident(value_column)
    market = getattr(spec, "market", None)
    timezone = getattr(spec, "timezone", None)

    # PERF-034：timezone() 只在 _raw 的 _local 应用一次；整数分钟在 _clock 计算
    # 一次。minute_at / minute_range 过滤用整数分钟（_minute），不再 strftime。
    local = _local_time_expr(time_column, market, timezone, time_is_tz=time_is_tz)
    date_expr = f"CAST(_local AS DATE)"
    minute_expr = _minute_from_local_expr("_local")
    cond = _spec_filter_sql(
        spec, minute_expr="_minute", elapsed_expr="_el", market=market
    )
    # ``_raw`` 把 value 列投影为 ``_v``（避免 read_parquet 宽列名直接出现在 CTE
    # 里的重复/转义问题），聚合引用 ``_v`` 而不是原始列名。
    agg_expr = _METRIC_FUNCS[metric].replace("value", "_v").replace("ts", "_local")

    raw = (
        f"SELECT {local} AS _local, {inst} AS _inst, {val} AS _v "
        f"FROM read_parquet(?)"
    )
    # _el（session elapsed）只有 minute_of_day 需要；minute_at/range 不生成，避免
    # 每行重复 EXTRACT。
    if spec.aggregation == "minute_of_day":
        elapsed = _elapsed_expr(minute_expr, market, timezone)
        clock = (
            f"SELECT _inst, {date_expr} AS _tsd, {minute_expr} AS _minute, "
            f"{elapsed} AS _el, _local, _v FROM _raw"
        )
    else:
        clock = (
            f"SELECT _inst, {date_expr} AS _tsd, {minute_expr} AS _minute, "
            f"_local, _v FROM _raw"
        )
    sel = f"_tsd AS ts, _inst AS inst, {agg_expr} AS value"
    if cond:
        sql = (
            f"WITH _raw AS ({raw}), _clock AS ({clock}) "
            f"SELECT {sel} FROM _clock WHERE {cond} GROUP BY _tsd, _inst"
        )
    else:
        sql = (
            f"WITH _raw AS ({raw}), _clock AS ({clock}) "
            f"SELECT {sel} FROM _clock GROUP BY _tsd, _inst"
        )
    return sql, []


def _build_bundle_sql(
    *,
    from_clause: str,
    predicate_where: str,
    time_column: str,
    instrument_column: str,
    parsed_items: Sequence[ParsedAggregationItem],
    market: str | None,
    timezone: str | None,
    time_is_tz: bool = False,
) -> tuple[str, list[Any]]:
    """#6 一次 scan 多聚合：单条 GROUP BY，每输出列 ``agg(val) FILTER (WHERE ...)``。

    PERF-033：``parsed_items`` 由上层（``aggregate_minute_bundle``）已 parse 一次
    传入，这里不再重复 parse（旧实现这里又 ``parse_aggregation_spec`` 一遍）。
    PERF-034：timezone() 只在 ``_raw`` 的 ``_local`` 应用一次；整数分钟在 ``_clock``
    算一次（``_minute``），minute_at / minute_range 过滤用整数比较，不再
    ``strftime('%H:%M')`` 字符串比较。
    PERF-035：同一 ``FilterSignature`` 的多个输出共享一个 ``_fN`` 条件列——过滤
    条件只在 ``_scan`` CTE 定义一次，各输出列 ``FILTER (WHERE _fN)`` 复用。

    返回 (sql, params)。整数分钟/桶号都是校验过的整数值，直接内联 SQL 字面量
    （无注入风险），因此 params 恒为 []；真正的绑定参数只剩扫描路径 + 扫描谓词
    （由调用方拼接）。
    """
    inst = _quote_ident(instrument_column)
    local = _local_time_expr(time_column, market, timezone, time_is_tz=time_is_tz)
    date_expr = f"CAST(_local AS DATE)"
    minute_expr = _minute_from_local_expr("_local")
    # _el（session elapsed）只有 minute_of_day 项需要；否则不生成，避免每行重复
    # EXTRACT（PERF-034）。
    need_elapsed = any(p.spec.aggregation == "minute_of_day" for p in parsed_items)
    if need_elapsed:
        elapsed = _elapsed_expr(minute_expr, market, timezone)
    session_total_bars = _session_total_bars(market)

    # 去重 value 列（多 item 复用同一物理列时只投影一次）
    cols: list[str] = []
    seen_cols: dict[str, str] = {}
    for parsed in parsed_items:
        item = parsed.item
        col_alias = seen_cols.get(item.field)
        if col_alias is None:
            col_alias = f"_v{len(seen_cols)}"
            seen_cols[item.field] = col_alias
            cols.append(f"{_quote_ident(item.field)} AS {col_alias}")

    # PERF-035：同一 filter signature 只生成一次 _fN 条件列，多输出共享。
    sig_alias: dict[FilterSignature, str] = {}
    filter_cols: list[str] = []
    sig_list: list[tuple[ParsedAggregationItem, FilterSignature | None]] = []
    for parsed in parsed_items:
        sig = FilterSignature.canonicalize(
            parsed.spec, session_total_bars=session_total_bars
        )
        cond = sig.to_sql(minute_expr="_minute", elapsed_expr="_el")
        if cond is not None:
            if sig not in sig_alias:
                alias = f"_f{len(sig_alias)}"
                sig_alias[sig] = alias
                filter_cols.append(f"({cond}) AS {alias}")
            sig_list.append((parsed, sig))
        else:
            sig_list.append((parsed, None))

    value_aliases = ", ".join(seen_cols.values())
    # _raw（timezone 一次）→ _clock（整数分钟 / elapsed 一次）→ _scan（过滤条件
    # 一次）。DuckDB 不允许同一 SELECT 列表内引用别名，因此派生值各占一层 CTE。
    # ``_scan`` 里 ``({cond}) AS _fN`` 引用 ``_clock`` 的 _minute/_el（不同 CTE，
    # 允许）；外层 ``FILTER (WHERE _fN)`` 复用这些条件列。
    raw = (
        f"SELECT {local} AS _local, {inst} AS _inst, {', '.join(cols)} "
        f"FROM {from_clause} {predicate_where}".strip()
    )
    if need_elapsed:
        clock = (
            f"SELECT _inst, {date_expr} AS _tsd, {minute_expr} AS _minute, "
            f"{elapsed} AS _el, _local, {value_aliases} FROM _raw"
        )
    else:
        clock = (
            f"SELECT _inst, {date_expr} AS _tsd, {minute_expr} AS _minute, "
            f"_local, {value_aliases} FROM _raw"
        )
    scan = (
        f"SELECT _tsd, _inst, _local, {value_aliases}"
        f"{', ' + ', '.join(filter_cols) if filter_cols else ''} FROM _clock"
    )

    select_parts: list[str] = []
    for parsed, sig in sig_list:
        item, spec = parsed.item, parsed.spec
        out = item.effective_output_name(spec)
        col_alias = seen_cols[item.field]
        metric = spec.effective_metric(item.field)
        agg = _METRIC_FUNCS[metric].replace("value", col_alias).replace("ts", "_local")
        if sig is not None and sig in sig_alias:
            select_parts.append(
                f"{agg} FILTER (WHERE {sig_alias[sig]}) AS {_quote_ident(out)}"
            )
        else:
            select_parts.append(f"{agg} AS {_quote_ident(out)}")

    sql = (
        f"WITH _raw AS ({raw}), _clock AS ({clock}), _scan AS ({scan}) "
        f"SELECT _tsd AS ts, _inst AS inst, {', '.join(select_parts)} "
        f"FROM _scan GROUP BY _tsd, _inst"
    )
    return sql, []


def _resolve_budget(store: Any, dataset: str, query_budget: Any) -> Any:
    from data_access.read.query_budget import (
        merge_dataset_policy,
        resolve_query_budget,
    )

    base = resolve_query_budget(query_budget)
    ds = store._registry.get(dataset)
    return merge_dataset_policy(base, ds.query_policy)


def _audit_and_handle(
    store: Any,
    dataset: str,
    table: Any,
    *,
    paths: list[str],
    params: dict[str, Any],
    elapsed_ms: float,
    budget: Any,
    kind: str,
    spec_info: dict[str, Any] | None = None,
    time_range: tuple[Any, Any] | None = None,
    instrument_filter: Sequence[str] | None = None,
) -> Any:
    """budget 强制 + 审计 + 构建带 snapshot 的 ReadHandle（#7）。

    #P1-43 ``spec_info`` 把真实聚合请求（field/spec/output_name/market/timezone/
    transform hash）并入审计——仅靠 snapshot 无法复现「这列日频数字怎么从分钟
    数据聚出来的」。
    #P1-final closure 9：ReadLineage 不再 ``columns=(), time_range=None``——
    field / time_range / instruments 全部写入，复现信息完整。
    """
    from data_access.core import audit as _audit
    from data_access.read.query_budget import enforce_arrow_budget
    from data_access.read.read_contract import (
        ReadLineage,
        ReadStats,
        build_data_snapshot,
        lineage_params,
    )
    from data_access.read.read_handle import ReadHandle

    enforce_arrow_budget(budget, table, elapsed_ms=elapsed_ms)
    ds = store._registry.get(dataset)
    snapshot = build_data_snapshot(
        dataset=dataset,
        registry_hash=store.registry_fingerprint(),
        schema=getattr(ds, "schema", None),
        paths=paths,
        params=params if getattr(ds, "params_schema", None) else None,
    )
    lineage = ReadLineage(
        dataset=dataset,
        columns=tuple(spec_info.get("fields") or ()) if spec_info else (),
        time_range=time_range,
        # #P1-final closure：None（全市场）与 []（空池）必须保留区别。
        instrument_filter=(
            tuple(instrument_filter)
            if instrument_filter is not None
            else None
        ),
        # #P1-final closure：params canonicalize 成不可变 tuple。
        params=lineage_params(params),
    )
    stats = ReadStats(rows=table.num_rows, bytes=table.nbytes, elapsed_ms=elapsed_ms)
    extra: dict[str, Any] = {"aggregation": True}
    if spec_info:
        extra["aggregation_spec"] = spec_info
    _audit.record(
        op=kind,
        dataset=dataset,
        ok=True,
        rows=table.num_rows,
        paths=paths[:5] if paths else None,
        params=params or None,
        elapsed_ms=elapsed_ms,
        extra=extra,
    )
    return ReadHandle(table=table, snapshot=snapshot, stats=stats, lineage=lineage)


_RESULT_MODES = ("arrow", "relation", "arrow_stream")


def _validate_result_mode(
    result_mode: str,
) -> Literal["arrow", "relation", "arrow_stream"]:
    """PERF-037：result_mode 白名单校验（fail-closed，不静默降级）。"""
    if result_mode not in _RESULT_MODES:
        raise ValidationError(
            f"result_mode 必须为 {sorted(_RESULT_MODES)}，收到 {result_mode!r}"
        )
    return result_mode  # type: ignore[return-value]


def _audit_lazy_aggregate(
    store: Any,
    dataset: str,
    *,
    paths: list[str],
    params: dict[str, Any],
    elapsed_ms: float,
    kind: str,
    spec_info: dict[str, Any] | None = None,
    time_range: tuple[Any, Any] | None = None,
    instrument_filter: Sequence[str] | None = None,
    result_mode: str = "lazy",
) -> None:
    """PERF-037：relation / arrow_stream 模式的审计（无物化表，rows=None）。

    lazy 结果没有 ``table.num_rows``，这里只记录「聚合请求已下发」+ lineage/
    snapshot 参数（下游物化时由 RelationHandle/ManagedBatchReader 自己再上报真实
    行数/字节）。不做 ``enforce_arrow_budget``——物化发生在调用方，由调用方负责
    治理。
    """
    from data_access.core import audit as _audit
    from data_access.read.read_contract import (
        ReadLineage,
        build_data_snapshot,
        lineage_params,
    )

    ds = store._registry.get(dataset)
    snapshot = build_data_snapshot(
        dataset=dataset,
        registry_hash=store.registry_fingerprint(),
        schema=getattr(ds, "schema", None),
        paths=paths,
        params=params if getattr(ds, "params_schema", None) else None,
    )
    lineage = ReadLineage(
        dataset=dataset,
        columns=tuple(spec_info.get("fields") or ()) if spec_info else (),
        time_range=time_range,
        instrument_filter=(
            tuple(instrument_filter)
            if instrument_filter is not None
            else None
        ),
        params=lineage_params(params),
    )
    extra: dict[str, Any] = {"aggregation": True, "result_mode": result_mode}
    if spec_info:
        extra["aggregation_spec"] = spec_info
    _audit.record(
        op=kind,
        dataset=dataset,
        ok=True,
        rows=None,
        paths=paths[:5] if paths else None,
        params=params or None,
        elapsed_ms=elapsed_ms,
        extra=extra,
    )


def aggregate_minute_to_daily(
    store: Any,
    dataset: str,
    field: str,
    spec: Any,
    *,
    time_range: tuple[Any, Any] | None = None,
    instrument_filter: Sequence[str] | None = None,
    params: Mapping[str, Any] | None = None,
    query_budget: Any = None,
    result_mode: Literal["arrow", "relation", "arrow_stream"] = "arrow",
) -> Any:
    """分钟→日聚合（DuckDB 内完成，不物化整份分钟数据）。

    ``result_mode``（PERF-037）：
        - ``"arrow"``（默认）：物化为 Arrow Table，返回带 ``DataSnapshot`` 的
          ``ReadHandle``（``.to_arrow()`` / ``.to_pandas()``，走 budget/deadline/
          audit）；
        - ``"relation"``：返回惰性 DuckDB Relation（不物化，调用方稍后
          ``.to_arrow_table()`` / ``.df()`` / ``.fetchall()``）。跳过 arrow budget
          强制——物化治理由调用方负责；
        - ``"arrow_stream"``：返回 ``ManagedBatchReader``（流式 RecordBatch，低
          内存峰值），同样跳过物化前的 arrow budget。

    默认 ``"arrow"``，既有调用方/测试零行为变化。整条 分钟→日 链路（PERF-036）在
    单条 DuckDB SQL 内完成（filter → daily aggregate），不物化整份分钟数据。
    """
    import pyarrow as pa

    from data_access.read.formats import format_adapter_for_dataset
    from data_access.read.predicate import Predicate, compile_predicate

    result_mode = _validate_result_mode(result_mode)
    ds = store._registry.get(dataset)
    if not ds.time_column or not ds.instrument_column:
        raise ValidationError(f"分钟数据集 '{dataset}' 未声明 time/instrument 列")
    # #P0-7 public 聚合入口走统一读前语义门禁（temporal contract / event cutoff /
    # required_filters / allowed values）——不再让 aggregation 成为绕过 gate 的
    # 第四条 read path。
    store._prepare_read_request(
        dataset,
        columns=[field],
        time_range=time_range,
        mode="auto",
        params=dict(params or {}),
    )
    agg = parse_aggregation_spec(spec, field=field)
    budget = _resolve_budget(store, dataset, query_budget)

    import time as _time

    paths = store._prepare_dataset_read(
        ds,
        time_range=time_range,
        params=dict(params or {}),
        instrument_filter=instrument_filter,
    )
    # #P1-final closure 9 max_scan_files：聚合不能绕过 dataset/query 的扫描文件
    # 预算——与普通 read 一致，路径解析后对匹配文件数做硬限制。
    store._enforce_scan_files(budget, paths)
    if not paths:
        return _audit_and_handle(
            store, dataset, pa.table({"ts": [], "inst": [], "value": []}),
            paths=[], params=dict(params or {}), elapsed_ms=0.0, budget=budget,
            kind="aggregate",
            spec_info={"field": field, "fields": [field], "spec": agg.to_dict()},
            time_range=time_range,
            instrument_filter=instrument_filter,
        )

    adapter = format_adapter_for_dataset(ds)
    path_param = paths if len(paths) > 1 else paths[0]
    from_clause = adapter.build_from_clause(
        path_param,
        hive_partitioning=ds.hive_partitioning,
        union_by_name=ds.union_by_name,
    )
    time_type = str((ds.schema or {}).get(ds.time_column or "", "")).lower()
    time_is_tz = "tz=" in time_type
    pred = Predicate(
        time_range=time_range,
        instrument_filter=instrument_filter,
        time_column_is_timestamp=("timestamp" in time_type or "datetime" in time_type),
    )
    compiled = compile_predicate(
        pred, time_column=ds.time_column, instrument_column=ds.instrument_column
    )

    sql, agg_params = build_minute_aggregation_sql(
        time_column=ds.time_column,
        instrument_column=ds.instrument_column,
        value_column=field,
        field=field,
        spec=agg,
        time_is_tz=time_is_tz,
    )
    inner = f"SELECT * FROM {from_clause} {compiled.where_sql}".strip()
    sql = sql.replace("FROM read_parquet(?)", f"FROM ({inner}) AS _m")

    # PERF-037：结果形态按 result_mode 分流（默认 arrow 物化，行为与旧版一致）。
    exec_params = [path_param, *compiled.params, *agg_params]
    start = _time.perf_counter()
    if result_mode == "relation":
        rel = store._engine.relation(sql, exec_params)
        elapsed_ms = (_time.perf_counter() - start) * 1000
        _audit_lazy_aggregate(
            store, dataset, paths=paths, params=dict(params or {}),
            elapsed_ms=elapsed_ms, kind="aggregate",
            spec_info={"field": field, "fields": [field], "spec": agg.to_dict()},
            time_range=time_range, instrument_filter=instrument_filter,
            result_mode=result_mode,
        )
        return rel
    if result_mode == "arrow_stream":
        reader = store._engine.execute_reader(
            sql, exec_params, deadline_ms=budget.max_elapsed_ms
        )
        elapsed_ms = (_time.perf_counter() - start) * 1000
        _audit_lazy_aggregate(
            store, dataset, paths=paths, params=dict(params or {}),
            elapsed_ms=elapsed_ms, kind="aggregate",
            spec_info={"field": field, "fields": [field], "spec": agg.to_dict()},
            time_range=time_range, instrument_filter=instrument_filter,
            result_mode=result_mode,
        )
        return reader
    # arrow：物化 + budget/deadline/audit + ReadHandle（现状）
    table = store._engine.execute_arrow(
        sql, exec_params, deadline_ms=budget.max_elapsed_ms
    )
    elapsed_ms = (_time.perf_counter() - start) * 1000
    return _audit_and_handle(
        store, dataset, table, paths=paths, params=dict(params or {}),
        elapsed_ms=elapsed_ms, budget=budget, kind="aggregate",
        spec_info={"field": field, "fields": [field], "spec": agg.to_dict()},
        time_range=time_range,
        instrument_filter=instrument_filter,
    )


def aggregate_minute_bundle(
    store: Any,
    dataset: str,
    items: Sequence[AggregationItem],
    *,
    time_range: tuple[Any, Any] | None = None,
    instrument_filter: Sequence[str] | None = None,
    params: Mapping[str, Any] | None = None,
    query_budget: Any = None,
    market: str | None = None,
    timezone: str | None = None,
    result_mode: Literal["arrow", "relation", "arrow_stream"] = "arrow",
) -> Any:
    """#6 一次 scan 多聚合：同一 (dataset, time_range, instruments) 只扫一次，
    单条 SQL 产出几十列（每列 ``agg FILTER (WHERE spec)``）。

    ``items``：``AggregationItem(field, spec, output_name)`` 列表。
    ``market``/``timezone``：未在 item.spec 上声明时作为全局默认。
    ``result_mode``（PERF-037）：与 ``aggregate_minute_to_daily`` 相同——
    ``"arrow"``（默认，返回带 ``DataSnapshot`` 的 ``ReadHandle``）/ ``"relation"``
    （惰性 DuckDB Relation）/ ``"arrow_stream"``（``ManagedBatchReader`` 流）。

    PERF-033：items 在入口 parse 一次（``ParsedAggregationItem``）后一路传给 SQL
    builder，builder 不再重复 parse。PERF-035：同一 minute-window 的多个输出共享
    一个 SQL 过滤条件列。
    """
    import pyarrow as pa

    from data_access.read.formats import format_adapter_for_dataset
    from data_access.read.predicate import Predicate, compile_predicate

    result_mode = _validate_result_mode(result_mode)
    if not items:
        raise ValidationError("aggregate_minute_bundle: items 不能为空")
    ds = store._registry.get(dataset)
    if not ds.time_column or not ds.instrument_column:
        raise ValidationError(f"分钟数据集 '{dataset}' 未声明 time/instrument 列")
    # #P0-7 与 aggregate_minute_to_daily 同 gate
    store._prepare_read_request(
        dataset,
        columns=[i.field for i in items],
        time_range=time_range,
        mode="auto",
        params=dict(params or {}),
    )
    budget = _resolve_budget(store, dataset, query_budget)

    # #P0-9 输出列名必须唯一：两个 item 同名输出 → DuckDB 会产生重复列，语义不明。
    # PERF-033：parse 一次成 ParsedAggregationItem，一路传给 SQL builder，避免
    # builder 内再 parse 一遍。
    parsed_items = [
        ParsedAggregationItem(
            item=item, spec=parse_aggregation_spec(item.spec, field=item.field)
        )
        for item in items
    ]
    out_names = [p.item.effective_output_name(p.spec) for p in parsed_items]
    # PERF-033：Counter 去重 O(K)，不再 ``list.count`` O(K²)。错误消息逐字不变。
    dup = {name for name, count in Counter(out_names).items() if count > 1}
    if dup:
        raise ValidationError(
            f"aggregate_minute_bundle 输出列名重复: {sorted(dup)}。"
            "每个 AggregationItem 必须用 output_name 区分，否则聚合结果列语义不明。"
        )

    # #P0-8 同一 bundle 必须 clock-compatible：各 item 自己声明的 market/timezone
    # 不一致时不能编译到同一条 SQL（时区换算/时段不同会串味）。
    declared_markets = {
        p.spec.market for p in parsed_items if p.spec.market is not None
    }
    declared_timezones = {
        p.spec.timezone for p in parsed_items if p.spec.timezone is not None
    }
    if market is not None:
        declared_markets.add(market)
    if timezone is not None:
        declared_timezones.add(timezone)
    if len(declared_markets) > 1:
        raise ValidationError(
            f"aggregate_minute_bundle 的 items 声明了不一致的 market: {sorted(declared_markets)}；"
            "同一 bundle 必须 clock-compatible（同一 market/timezone）。"
        )
    if len(declared_timezones) > 1:
        raise ValidationError(
            f"aggregate_minute_bundle 的 items 声明了不一致的 timezone: {sorted(declared_timezones)}；"
            "同一 bundle 必须 clock-compatible（同一 market/timezone）。"
        )

    import time as _time

    # #P0-8 已校验所有 item 的 market 一致；effective = 全局 market 或 item 声明值。
    # （提前计算：空路径分支的 spec_info 也要用，不能等 SQL 分支再算。）
    effective_market = market or next(
        (p.spec.market for p in parsed_items if p.spec.market is not None),
        None,
    )

    paths = store._prepare_dataset_read(
        ds,
        time_range=time_range,
        params=dict(params or {}),
        instrument_filter=instrument_filter,
    )
    # #P1-final closure 9 max_scan_files：bundle 聚合同样强制扫描文件预算。
    store._enforce_scan_files(budget, paths)
    if not paths:
        cols = ["ts", "inst"]
        for p in parsed_items:
            cols.append(p.item.effective_output_name(p.spec))
        return _audit_and_handle(
            store, dataset, pa.table({c: [] for c in cols}),
            paths=[], params=dict(params or {}), elapsed_ms=0.0, budget=budget,
            kind="aggregate",
            spec_info={
                "bundle": True,
                "fields": [p.item.field for p in parsed_items],
                "items": [
                    {
                        "field": p.item.field,
                        "output_name": p.item.effective_output_name(p.spec),
                        "spec": p.spec.to_dict(),
                    }
                    for p in parsed_items
                ],
                "market": effective_market,
                "timezone": timezone,
            },
            time_range=time_range,
            instrument_filter=instrument_filter,
        )

    adapter = format_adapter_for_dataset(ds)
    path_param = paths if len(paths) > 1 else paths[0]
    from_clause = adapter.build_from_clause(
        path_param,
        hive_partitioning=ds.hive_partitioning,
        union_by_name=ds.union_by_name,
    )
    time_type = str((ds.schema or {}).get(ds.time_column or "", "")).lower()
    time_is_tz = "tz=" in time_type
    pred = Predicate(
        time_range=time_range,
        instrument_filter=instrument_filter,
        time_column_is_timestamp=("timestamp" in time_type or "datetime" in time_type),
    )
    compiled = compile_predicate(
        pred, time_column=ds.time_column, instrument_column=ds.instrument_column
    )

    sql, agg_params = _build_bundle_sql(
        from_clause=from_clause,
        predicate_where=compiled.where_sql,
        time_column=ds.time_column,
        instrument_column=ds.instrument_column,
        parsed_items=parsed_items,
        market=effective_market,
        timezone=timezone,
        time_is_tz=time_is_tz,
    )

    spec_info = {
        "bundle": True,
        "fields": [p.item.field for p in parsed_items],
        "items": [
            {
                "field": p.item.field,
                "output_name": p.item.effective_output_name(p.spec),
                "spec": p.spec.to_dict(),
            }
            for p in parsed_items
        ],
        "market": effective_market,
        "timezone": timezone,
    }
    # PERF-037：结果形态按 result_mode 分流（默认 arrow 物化，行为与旧版一致）。
    exec_params = [path_param, *compiled.params, *agg_params]
    start = _time.perf_counter()
    if result_mode == "relation":
        rel = store._engine.relation(sql, exec_params)
        elapsed_ms = (_time.perf_counter() - start) * 1000
        _audit_lazy_aggregate(
            store, dataset, paths=paths, params=dict(params or {}),
            elapsed_ms=elapsed_ms, kind="aggregate", spec_info=spec_info,
            time_range=time_range, instrument_filter=instrument_filter,
            result_mode=result_mode,
        )
        return rel
    if result_mode == "arrow_stream":
        reader = store._engine.execute_reader(
            sql, exec_params, deadline_ms=budget.max_elapsed_ms
        )
        elapsed_ms = (_time.perf_counter() - start) * 1000
        _audit_lazy_aggregate(
            store, dataset, paths=paths, params=dict(params or {}),
            elapsed_ms=elapsed_ms, kind="aggregate", spec_info=spec_info,
            time_range=time_range, instrument_filter=instrument_filter,
            result_mode=result_mode,
        )
        return reader
    table = store._engine.execute_arrow(
        sql, exec_params, deadline_ms=budget.max_elapsed_ms
    )
    elapsed_ms = (_time.perf_counter() - start) * 1000
    return _audit_and_handle(
        store, dataset, table, paths=paths, params=dict(params or {}),
        elapsed_ms=elapsed_ms, budget=budget, kind="aggregate",
        spec_info=spec_info,
        time_range=time_range,
        instrument_filter=instrument_filter,
    )
