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

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from data_access.core.exceptions import ValidationError

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


def _local_min_expr(
    t_col: str,
    market: str | None,
    timezone: str | None,
    *,
    time_is_tz: bool = False,
) -> str:
    loc = _local_time_expr(t_col, market, timezone, time_is_tz=time_is_tz)
    return f"(EXTRACT(HOUR FROM {loc}) * 60 + EXTRACT(MINUTE FROM {loc}))"


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
    hhmm_expr: str,
    elapsed_expr: str,
    market: str | None,
) -> tuple[str | None, list[Any]]:
    """构建单个 spec 的 FILTER 条件；None 表示不过滤（全时段聚合）。"""
    if spec.aggregation == "minute_at":
        if not spec.hhmm:
            raise ValidationError("minute_at 需要 hhmm")
        return f"{hhmm_expr} = ?", [spec.hhmm]
    if spec.aggregation == "minute_range":
        if not spec.start or not spec.end:
            raise ValidationError("minute_range 需要 start 和 end")
        if spec.start >= spec.end:
            raise ValidationError("minute_range start 必须早于 end")
        return (
            f"{hhmm_expr} >= ? AND {hhmm_expr} <= ?",
            [spec.start, spec.end],
        )
    if spec.aggregation == "minute_of_day":
        period = max(1, spec.period)
        total = _session_total_bars(market)
        # index=0 → 最新桶；桶号从 session 尾部向前数
        last_slot = max(0, (total - 1) // period)
        slot = max(0, last_slot - max(0, spec.index))
        return (
            f"{elapsed_expr} >= {slot * period} AND "
            f"{elapsed_expr} < {(slot + 1) * period}",
            [],
        )
    raise ValidationError(f"不支持的聚合: {spec.aggregation}")


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

    local = _local_time_expr(time_column, market, timezone, time_is_tz=time_is_tz)
    hhmm_expr = f"strftime({local}, '%H:%M')"
    date_expr = f"CAST({local} AS DATE)"
    local_min = _local_min_expr(time_column, market, timezone, time_is_tz=time_is_tz)
    elapsed = _elapsed_expr(local_min, market, timezone)
    params: list[Any] = []
    agg_expr = _METRIC_FUNCS[metric].replace("value", val).replace("ts", local)

    if spec.aggregation == "minute_at":
        cond, p = _spec_filter_sql(spec, hhmm_expr=hhmm_expr, elapsed_expr=elapsed, market=market)
        params.extend(p)
        sel = f"{date_expr} AS ts, {inst} AS inst, {agg_expr} AS value"
        group = f"{date_expr}, {inst}"
    elif spec.aggregation == "minute_range":
        cond, p = _spec_filter_sql(spec, hhmm_expr=hhmm_expr, elapsed_expr=elapsed, market=market)
        params.extend(p)
        sel = f"{date_expr} AS ts, {inst} AS inst, {agg_expr} AS value"
        group = f"{date_expr}, {inst}"
    elif spec.aggregation == "minute_of_day":
        cond, p = _spec_filter_sql(spec, hhmm_expr=hhmm_expr, elapsed_expr=elapsed, market=market)
        params.extend(p)
        sel = f"{date_expr} AS ts, {inst} AS inst, {agg_expr} AS value"
        group = f"{date_expr}, {inst}"
    else:  # pragma: no cover
        raise ValidationError(f"不支持的聚合: {spec.aggregation}")

    # minute_of_day：FILTER 已把桶号（session elapsed）限定到目标桶，直接按
    # (date, inst) 聚合该桶——而不是按 elapsed 分组再挑最后一分钟。
    if spec.aggregation == "minute_of_day":
        sel = f"{date_expr} AS ts, {inst} AS inst, {agg_expr} AS value"
        sql = (
            f"SELECT {sel} FROM (SELECT * FROM read_parquet(?)) "
            f"WHERE {cond} GROUP BY {date_expr}, {inst}"
        )
    else:
        if cond:
            sql = (
                f"SELECT {sel} FROM (SELECT * FROM read_parquet(?)) "
                f"WHERE {cond} GROUP BY {group}"
            )
        else:
            sql = (
                f"SELECT {sel} FROM (SELECT * FROM read_parquet(?)) "
                f"GROUP BY {group}"
            )
    return sql, params


def _build_bundle_sql(
    *,
    from_clause: str,
    predicate_where: str,
    time_column: str,
    instrument_column: str,
    items: Sequence[AggregationItem],
    market: str | None,
    timezone: str | None,
    time_is_tz: bool = False,
) -> tuple[str, list[Any]]:
    """#6 一次 scan 多聚合：单条 GROUP BY，每输出列 ``agg(val) FILTER (WHERE ...)``。

    返回 (sql, params)；params 顺序 = 扫描谓词参数 → 各 item 的 FILTER 参数。
    派生列（_hhmm/_el/_local/_tsd）在 ``_scan`` CTE 内各算一次，外层只引用别名。
    """
    inst = _quote_ident(instrument_column)
    local = _local_time_expr(time_column, market, timezone, time_is_tz=time_is_tz)
    hhmm_expr = f"strftime({local}, '%H:%M')"
    date_expr = f"CAST({local} AS DATE)"
    local_min = _local_min_expr(time_column, market, timezone, time_is_tz=time_is_tz)
    elapsed = _elapsed_expr(local_min, market, timezone)

    cols: list[str] = []
    select_parts: list[str] = []
    filter_params: list[Any] = []

    parsed: list[tuple[AggregationItem, AggregationSpec]] = []
    for item in items:
        parsed.append((item, parse_aggregation_spec(item.spec, field=item.field)))
    # 去重 value 列（多 item 复用同一物理列时只投影一次）
    seen_cols: dict[str, str] = {}
    for idx, (item, spec) in enumerate(parsed):
        col_alias = seen_cols.get(item.field)
        if col_alias is None:
            col_alias = f"_v{len(seen_cols)}"
            seen_cols[item.field] = col_alias
            cols.append(f"{_quote_ident(item.field)} AS {col_alias}")

    # _scan CTE：派生列算一次；read_parquet(?) 在 SQL 文本最前 → 参数顺序
    # [path, predicate, filters] 天然正确（WITH 在最前）。
    base = (
        f"SELECT {inst} AS _inst, {date_expr} AS _tsd, {hhmm_expr} AS _hhmm, "
        f"{elapsed} AS _el, {local} AS _local, {', '.join(cols)} "
        f"FROM {from_clause} {predicate_where}".strip()
    )
    for idx, (item, spec) in enumerate(parsed):
        out = item.effective_output_name(spec)
        col_alias = seen_cols[item.field]
        metric = spec.effective_metric(item.field)
        agg = _METRIC_FUNCS[metric].replace("value", col_alias).replace("ts", "_local")
        cond, p = _spec_filter_sql(
            spec, hhmm_expr="_hhmm", elapsed_expr="_el", market=market
        )
        if cond:
            select_parts.append(f"{agg} FILTER (WHERE {cond}) AS {_quote_ident(out)}")
            filter_params.extend(p)
        else:
            select_parts.append(f"{agg} AS {_quote_ident(out)}")

    sql = (
        f"WITH _scan AS ({base}) "
        f"SELECT _tsd AS ts, _inst AS inst, {', '.join(select_parts)} "
        f"FROM _scan GROUP BY _tsd, _inst"
    )
    return sql, filter_params


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
) -> Any:
    """分钟→日聚合（DuckDB 内完成，不物化整份分钟数据）。

    返回带 ``DataSnapshot`` 的 ``ReadHandle``：``.to_arrow()`` 得到
    Arrow Table（``ts`` DATE + ``inst`` + ``value``），``.to_pandas()`` 同理。
    走完整 QueryBudget / deadline / audit（#7）。
    """
    import pyarrow as pa

    from data_access.read.formats import format_adapter_for_dataset
    from data_access.read.predicate import Predicate, compile_predicate

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

    start = _time.perf_counter()
    table = store._engine.execute_arrow(
        sql, [path_param, *compiled.params, *agg_params], deadline_ms=budget.max_elapsed_ms
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
) -> Any:
    """#6 一次 scan 多聚合：同一 (dataset, time_range, instruments) 只扫一次，
    单条 SQL 产出几十列（每列 ``agg FILTER (WHERE spec)``）。

    ``items``：``AggregationItem(field, spec, output_name)`` 列表。
    ``market``/``timezone``：未在 item.spec 上声明时作为全局默认。
    返回带 ``DataSnapshot`` 的 ``ReadHandle``（budget/deadline/audit，见 #7）。
    """
    import pyarrow as pa

    from data_access.read.formats import format_adapter_for_dataset
    from data_access.read.predicate import Predicate, compile_predicate

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
    parsed_items = [
        (item, parse_aggregation_spec(item.spec, field=item.field)) for item in items
    ]
    out_names = [item.effective_output_name(spec) for item, spec in parsed_items]
    dup = {name for name in out_names if out_names.count(name) > 1}
    if dup:
        raise ValidationError(
            f"aggregate_minute_bundle 输出列名重复: {sorted(dup)}。"
            "每个 AggregationItem 必须用 output_name 区分，否则聚合结果列语义不明。"
        )

    # #P0-8 同一 bundle 必须 clock-compatible：各 item 自己声明的 market/timezone
    # 不一致时不能编译到同一条 SQL（时区换算/时段不同会串味）。
    declared_markets = {
        spec.market for _, spec in parsed_items if spec.market is not None
    }
    declared_timezones = {
        spec.timezone for _, spec in parsed_items if spec.timezone is not None
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
        (spec.market for _, spec in parsed_items if spec.market is not None),
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
        for item, spec in parsed_items:
            cols.append(item.effective_output_name(spec))
        return _audit_and_handle(
            store, dataset, pa.table({c: [] for c in cols}),
            paths=[], params=dict(params or {}), elapsed_ms=0.0, budget=budget,
            kind="aggregate",
            spec_info={
                "bundle": True,
                "fields": [item.field for item, _ in parsed_items],
                "items": [
                    {
                        "field": item.field,
                        "output_name": item.effective_output_name(spec),
                        "spec": spec.to_dict(),
                    }
                    for item, spec in parsed_items
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
        items=items,
        market=effective_market,
        timezone=timezone,
        time_is_tz=time_is_tz,
    )

    start = _time.perf_counter()
    table = store._engine.execute_arrow(
        sql, [path_param, *compiled.params, *agg_params], deadline_ms=budget.max_elapsed_ms
    )
    elapsed_ms = (_time.perf_counter() - start) * 1000
    return _audit_and_handle(
        store, dataset, table, paths=paths, params=dict(params or {}),
        elapsed_ms=elapsed_ms, budget=budget, kind="aggregate",
        spec_info={
            "bundle": True,
            "fields": [item.field for item, _ in parsed_items],
            "items": [
                {
                    "field": item.field,
                    "output_name": item.effective_output_name(spec),
                    "spec": spec.to_dict(),
                }
                for item, spec in parsed_items
            ],
            "market": effective_market,
            "timezone": timezone,
        },
        time_range=time_range,
        instrument_filter=instrument_filter,
    )
