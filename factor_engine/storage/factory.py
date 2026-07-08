from __future__ import annotations

from typing import Any

from .composite_source import CompositeDataSource
from .cleaned_parquet_source import CleanedParquetSource
from .data_access_source import DataAccessSource
from .kline_parquet_source import KlineParquetSource
from .parquet_source import ParquetSource
from workspace_paths import resolve_path


def _extract_source_spec(config: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(config, dict):
        raw = dict(config)
        source_type = raw.pop("type")
        return str(source_type), raw

    source_type = getattr(config, "type", None)
    if source_type is None:
        raise ValueError("Data source config must include a 'type'")

    options = dict(getattr(config, "options", {}) or {})
    return str(source_type), options


def _pop_option(
    options: dict[str, Any],
    *names: str,
    default: Any = None,
    required: bool = False,
) -> Any:
    for name in names:
        if name in options and options[name] is not None:
            return options.pop(name)
    if required:
        joined = ", ".join(names)
        raise ValueError(f"Missing required data source option: {joined}")
    return default


def _ensure_no_extra_options(source_type: str, options: dict[str, Any]) -> None:
    if not options:
        return
    unexpected = ", ".join(sorted(options))
    raise ValueError(f"Unsupported options for data source '{source_type}': {unexpected}")


def _wrap_long_table(source: Any, *, long_table: bool) -> Any:
    if not long_table:
        return source
    from .long_table_source import LongTableDataSource

    return LongTableDataSource(source)


def _normalize_path(value: Any) -> str:
    return str(resolve_path(str(value)))


def _attach_bar_freq(source: Any, bar_freq: str) -> Any:
    """为 DataSource 挂载 bar 频率（供 warmup / lookback 换算）。"""
    if bar_freq and not getattr(source, "bar_freq", None):
        source.bar_freq = bar_freq
    return source


def _apply_date_range_to_source_config(
    config: Any,
    *,
    start_date: str | None,
    end_date: str | None,
) -> Any:
    """将 composite 顶层的 ``start_date``/``end_date`` 下发到各子源（子源已显式设置则不覆盖）。"""
    if not isinstance(config, dict) or (start_date is None and end_date is None):
        return config
    cfg = dict(config)
    if start_date is not None and cfg.get("start_date") is None:
        cfg["start_date"] = start_date
    if end_date is not None and cfg.get("end_date") is None:
        cfg["end_date"] = end_date
    if str(cfg.get("type", "")).lower() == "composite":
        subs = cfg.get("sources")
        if isinstance(subs, dict):
            cfg["sources"] = {
                str(name): _apply_date_range_to_source_config(
                    sub,
                    start_date=start_date,
                    end_date=end_date,
                )
                for name, sub in subs.items()
            }
    return cfg


def build_data_source(config: Any):
    """根据运行时配置构造单数据源或组合数据源实例。"""

    # YAML 可为 dict 或已解析的 dataclass；统一成 (type, options) 再分支
    source_type, options = _extract_source_spec(config)
    if source_type == "composite":
        anchor_source = _pop_option(options, "anchor", "anchor_source", required=True)
        anchor_column = _pop_option(options, "anchor_column", required=True)
        raw_sources = _pop_option(options, "sources", required=True)
        joins = _pop_option(options, "joins", default=None)
        aliases = _pop_option(options, "aliases", default=None)
        allow_unqualified_anchor_columns = bool(
            _pop_option(options, "allow_unqualified_anchor_columns", default=True)
        )
        start_date = _pop_option(options, "start_date", default=None)
        end_date = _pop_option(options, "end_date", default=None)

        if not isinstance(raw_sources, dict) or not raw_sources:
            raise ValueError("Composite data source requires a non-empty 'sources' mapping")

        built_sources = {
            str(name): build_data_source(
                _apply_date_range_to_source_config(
                    source_config,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
            for name, source_config in raw_sources.items()
        }
        source = CompositeDataSource(
            anchor_source=str(anchor_source),
            anchor_column=str(anchor_column),
            sources=built_sources,
            joins=joins,
            aliases=aliases,
            allow_unqualified_anchor_columns=allow_unqualified_anchor_columns,
        )
        _ensure_no_extra_options(source_type, options)
        return source

    max_files = _pop_option(options, "max_files", default=None)
    start_date = _pop_option(options, "start_date", default=None)
    end_date = _pop_option(options, "end_date", default=None)
    fields = _pop_option(options, "fields", "field_mapping", default=None)
    bar_freq = str(_pop_option(options, "bar_freq", default="1d"))
    long_table = bool(_pop_option(options, "long_table", default=False))

    if source_type == "long_table":
        inner = _pop_option(options, "inner", "source", required=True)
        _ensure_no_extra_options(source_type, options)
        return _wrap_long_table(build_data_source(inner), long_table=True)

    if source_type == "data_access":
        dataset = _pop_option(options, "dataset", required=True)
        instrument_filter = _pop_option(options, "instrument_filter", default=None)
        normalize_timestamp = _pop_option(options, "normalize_timestamp", default=None)
        timestamp_unit = _pop_option(options, "timestamp_unit", default=None)
        params = _pop_option(options, "params", default=None)
        kind = _pop_option(options, "kind", default=None)
        if kind is not None:
            merged = dict(params or {})
            merged["kind"] = str(kind)
            params = merged
        _pop_option(options, "root", default=None)  # 忽略旧配置残留
        source = DataAccessSource(
            dataset=str(dataset),
            fields=fields,
            start_date=start_date,
            end_date=end_date,
            instrument_filter=instrument_filter,
            normalize_timestamp=normalize_timestamp,
            timestamp_unit=timestamp_unit,
            params=params,
        )
        _ensure_no_extra_options(source_type, options)
        return _wrap_long_table(_attach_bar_freq(source, bar_freq), long_table=long_table)

    if source_type == "clickhouse":
        table = _pop_option(options, "table", required=True)
        timestamp_column = _pop_option(
            options, "timestamp_col", "timestamp_column", default="trade_date"
        )
        instrument_column = _pop_option(
            options, "instrument_col", "instrument_column", default="instrument"
        )
        instrument_filter = _pop_option(options, "instrument_filter", default=None)
        from .clickhouse_source import ClickHouseSource

        source = ClickHouseSource(
            table=str(table),
            timestamp_column=str(timestamp_column),
            instrument_column=str(instrument_column),
            fields=fields,
            start_date=start_date,
            end_date=end_date,
            instrument_filter=instrument_filter,
            host=_pop_option(options, "host", default=None),
            port=_pop_option(options, "port", default=None),
            database=_pop_option(options, "database", default=None),
            username=_pop_option(options, "username", default=None),
            password=_pop_option(options, "password", default=None),
            secure=_pop_option(options, "secure", default=None),
        )
        _ensure_no_extra_options(source_type, options)
        return _wrap_long_table(_attach_bar_freq(source, bar_freq), long_table=long_table)

    root = _normalize_path(_pop_option(options, "root", required=True))
    recursive = bool(_pop_option(options, "recursive", default=True))

    if source_type == "parquet_kline":
        # KlineParquetSource 使用 ``instrument_column`` / ``timestamp_column`` 命名（与 YAML 中 instrument_col 别名兼容）
        source = KlineParquetSource(
            root=root,
            instrument_column=_pop_option(
                options, "instrument_col", "instrument_column", default="ticker"
            ),
            timestamp_column=_pop_option(
                options, "timestamp_col", "timestamp_column", default="window_start"
            ),
            fields=fields,
            max_files=max_files,
            timestamp_unit=_pop_option(options, "timestamp_unit", default="ns"),
            start_date=start_date,
            end_date=end_date,
            bar_freq=bar_freq,
        )
    elif source_type in {"multi_parquet", "parquet"}:
        source = ParquetSource(
            root=root,
            timestamp_column=_pop_option(
                options, "timestamp_col", "timestamp_column", required=True
            ),
            instrument_column=_pop_option(
                options, "instrument_col", "instrument_column", required=True
            ),
            fields=fields,
            max_files=max_files,
            timestamp_unit=_pop_option(options, "timestamp_unit", default=None),
            start_date=start_date,
            end_date=end_date,
            recursive=recursive,
        )
    elif source_type == "intraday_daily":
        inner_cfg = _pop_option(options, "source", required=True)
        if not isinstance(inner_cfg, dict) or "type" not in inner_cfg:
            raise ValueError("intraday_daily 需要嵌套 source: {type: ..., ...}")
        raw_features = _pop_option(
            options,
            "features",
            default=("last_close", "sum_volume", "vwap"),
        )
        features = tuple(str(x) for x in raw_features)
        inner = build_data_source(inner_cfg)
        from runtime.intraday_aggregator import IntradayAggregatedDataSource

        source = IntradayAggregatedDataSource(inner=inner, features=features)
        _ensure_no_extra_options(source_type, options)
        return _attach_bar_freq(source, bar_freq)

    elif source_type == "cleaned_parquet":
        source = CleanedParquetSource(
            root=root,
            timestamp_col=_pop_option(options, "timestamp_col", "timestamp_column", default=None),
            instrument_col=_pop_option(options, "instrument_col", "instrument_column", default=None),
            fields=fields,
            max_files=max_files,
            timestamp_unit=_pop_option(options, "timestamp_unit", default=None),
            start_date=start_date,
            end_date=end_date,
            recursive=recursive,
        )
    else:
        raise ValueError(f"Unsupported data_source.type: {source_type}")

    _ensure_no_extra_options(source_type, options)
    return _wrap_long_table(_attach_bar_freq(source, bar_freq), long_table=long_table)
