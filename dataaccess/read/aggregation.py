"""
data_access.read.aggregation —— 分钟→日频聚合下推（AggregationSpec）

背景
    日频因子里的分钟聚合（09:30~10:00 volume、14:30~15:00 return、minute_at
    等）以前在 FactorEngine 里 Pandas 做：整份分钟数据过 Arrow/Pandas 再
    groupby。本模块把聚合下沉到 DuckDB（#14）：只把聚合结果拉回来。

用法
    spec = AggregationSpec(
        aggregation="minute_range",
        start="09:30", end="10:00",
        metric="sum",            # first/last/min/max/sum/mean/close(默认 last)
    )
    table = aggregate_minute_to_daily(store, "ashare_stock_minute", "Volume", spec,
                                      time_range=..., instrument_filter=...)

设计要点
    1. 支持 minute_at / minute_range / minute_of_day / vwap（amount/volume 组合）。
    2. 语义聚合映射（与 FactorEngine _minute_semantic_aggregate 对齐）：
       open→first, high→max, low→min, volume/amount→sum, 其余→last。
    3. 结果在 DuckDB 内完成，不物化整份分钟数据。

维护人：quant 基础平台组    最后更新：2026-08-08
"""

from __future__ import annotations

from dataclasses import dataclass
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
        period: 分钟桶宽（minute_of_day 用，把 09:30 起按 N 分钟分桶）
        index: 取第几个桶（0 为最新，与 FE minute_bar 对齐）
    """

    aggregation: str = "minute_range"
    start: str | None = None
    end: str | None = None
    hhmm: str | None = None
    metric: str | None = None
    period: int = 5
    index: int = 0

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
        }


def parse_aggregation_spec(raw: Any, *, field: str) -> AggregationSpec:
    """把 str / dict / AggregationSpec 归一化。"""
    if isinstance(raw, AggregationSpec):
        return raw
    if isinstance(raw, str):
        return AggregationSpec(aggregation=raw, metric=_semantic_metric(field))
    if not isinstance(raw, Mapping):
        raise ValidationError("聚合规格必须是 str / dict / AggregationSpec")
    agg = str(raw.get("aggregation", "minute_range"))
    if agg not in _VALID_AGGREGATIONS:
        raise ValidationError(
            f"聚合 '{agg}' 不支持；合法: {sorted(_VALID_AGGREGATIONS)}"
        )
    return AggregationSpec(
        aggregation=agg,
        start=str(raw["start"]) if raw.get("start") is not None else None,
        end=str(raw["end"]) if raw.get("end") is not None else None,
        hhmm=str(raw["hhmm"]) if raw.get("hhmm") is not None else None,
        metric=str(raw["metric"]) if raw.get("metric") is not None else None,
        period=int(raw.get("period", 5)),
        index=int(raw.get("index", 0)),
    )


def _quote_ident(name: str) -> str:
    return f'"{name.replace(chr(34), chr(34) * 2)}"'


def build_minute_aggregation_sql(
    *,
    time_column: str,
    instrument_column: str,
    value_column: str,
    field: str,
    spec: AggregationSpec,
) -> tuple[str, list[Any]]:
    """构建分钟→日聚合的 DuckDB SQL；返回 (sql, params)。

    输出列：``ts``（日期）+ ``inst`` + ``value``。
    """
    metric = spec.effective_metric(field)
    t = _quote_ident(time_column)
    inst = _quote_ident(instrument_column)
    val = _quote_ident(value_column)
    hhmm_expr = f"strftime({t}, '%H:%M')"
    date_expr = f"CAST({t} AS DATE)"
    params: list[Any] = []
    agg_expr = (
        _METRIC_FUNCS[metric]
        .replace("value", val)
        .replace("ts", t)
    )

    if spec.aggregation == "minute_at":
        if not spec.hhmm:
            raise ValidationError("minute_at 需要 hhmm")
        where = f"WHERE {hhmm_expr} = ?"
        params.append(spec.hhmm)
        agg = agg_expr
        group = date_expr
        sel = f"{date_expr} AS ts, {inst} AS inst, {agg} AS value"
    elif spec.aggregation == "minute_range":
        if not spec.start or not spec.end:
            raise ValidationError("minute_range 需要 start 和 end")
        if spec.start >= spec.end:
            raise ValidationError("minute_range start 必须早于 end")
        where = f"WHERE {hhmm_expr} >= ? AND {hhmm_expr} <= ?"
        params.extend([spec.start, spec.end])
        agg = agg_expr
        sel = f"{date_expr} AS ts, {inst} AS inst, {agg} AS value"
    elif spec.aggregation == "minute_of_day":
        period = max(1, spec.period)
        offset = max(0, spec.index)
        # 09:30 (570 分钟) 为当日 0 号桶；把盘中分钟映射到桶号
        minute_expr = f"(EXTRACT(HOUR FROM {t}) * 60 + EXTRACT(MINUTE FROM {t}))"
        slot_expr = f"GREATEST(({minute_expr} - 570), 0) // {period}"
        where = "WHERE 1=1"
        sel = (
            f"{date_expr} AS ts, {inst} AS inst, {slot_expr} AS slot, "
            f"{agg_expr} AS value"
        )
        group = f"{date_expr}, {inst}, {slot_expr}"
    else:  # pragma: no cover
        raise ValidationError(f"不支持的聚合: {spec.aggregation}")

    if spec.aggregation == "minute_of_day":
        sub = (
            f"SELECT {sel} FROM (SELECT * FROM read_parquet(?)) {where} "
            f"GROUP BY {group}"
        )
        sql = (
            f"SELECT ts, inst, value FROM ("
            f"SELECT ts, inst, value, "
            f"ROW_NUMBER() OVER (PARTITION BY ts, inst ORDER BY slot DESC) AS _rn "
            f"FROM ({sub})) WHERE _rn = 1 + {offset}"
        )
    else:
        sql = f"SELECT {sel} FROM (SELECT * FROM read_parquet(?)) {where} GROUP BY {date_expr}, {inst}"
    return sql, params


def aggregate_minute_to_daily(
    store: Any,
    dataset: str,
    field: str,
    spec: Any,
    *,
    time_range: tuple[Any, Any] | None = None,
    instrument_filter: Sequence[str] | None = None,
    params: Mapping[str, Any] | None = None,
) -> Any:
    """分钟→日聚合（DuckDB 内完成，不物化整份分钟数据）。

    返回 Arrow Table：``ts``（DATE）+ ``inst`` + ``value``。``dataset`` 需是
    已注册的分钟数据集（time_column/instrument_column 取 registry）。
    """
    from data_access.read.predicate import Predicate, compile_predicate

    ds = store._registry.get(dataset)
    if not ds.time_column or not ds.instrument_column:
        raise ValidationError(f"分钟数据集 '{dataset}' 未声明 time/instrument 列")
    agg = parse_aggregation_spec(spec, field=field)
    paths = store._prepare_dataset_read(
        ds,
        time_range=time_range,
        params=dict(params or {}),
        instrument_filter=instrument_filter,
    )
    if not paths:
        import pyarrow as pa

        return pa.table({"ts": [], "inst": [], "value": []})

    # 外层 read_parquet(?) 用路径；time/instrument 过滤由 predicate 下推
    from data_access.read.formats import format_adapter_for_dataset

    adapter = format_adapter_for_dataset(ds)
    path_param = paths if len(paths) > 1 else paths[0]
    from_clause = adapter.build_from_clause(
        path_param,
        hive_partitioning=ds.hive_partitioning,
        union_by_name=ds.union_by_name,
    )
    time_type = str((ds.schema or {}).get(ds.time_column or "", "")).lower()
    pred = Predicate(
        time_range=time_range,
        instrument_filter=instrument_filter,
        time_column_is_timestamp=("timestamp" in time_type or "datetime" in time_type),
    )
    compiled = compile_predicate(pred, time_column=ds.time_column, instrument_column=ds.instrument_column)

    sql, agg_params = build_minute_aggregation_sql(
        time_column=ds.time_column,
        instrument_column=ds.instrument_column,
        value_column=field,
        field=field,
        spec=agg,
    )
    # 把 predicate WHERE 并进 read_parquet 的 SELECT；参数顺序按 SQL 文本出现次序：
    # path（?）→ 谓词参数 → 聚合参数（hhmm/start/end 在外层 WHERE）
    inner = f"SELECT * FROM {from_clause} {compiled.where_sql}".strip()
    sql = sql.replace("FROM read_parquet(?)", f"FROM ({inner}) AS _m")
    return store._engine.execute_arrow(
        sql, [path_param, *compiled.params, *agg_params], deadline_ms=None
    )
