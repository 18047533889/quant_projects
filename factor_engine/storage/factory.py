from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .composite_source import CompositeDataSource
from .cleaned_parquet_source import CleanedParquetSource
from .data_access_source import DataAccessSource
from .kline_parquet_source import KlineParquetSource
from .parquet_source import ParquetSource
from workspace_paths import resolve_path


@dataclass(frozen=True)
class DataSourceBuildContext:
    """Production policy injected from the engine's run config into every source.

    Built by ``FactorEngine.from_loaded_config`` / the pipeline layer from the
    parsed ``RunConfig`` and threaded recursively through Composite /
    LongTable / Intraday children, so a production run constructs
    ``DataAccessSource`` with the hard gates ON *without* the YAML author having
    to remember them per-source (R10 #1).  Research runs simply leave the gates
    off; a research build is never upgraded to production here.

    Per-source options in the source spec (``run_mode`` / ``production`` /
    ``strict_unknown_fields`` / ``enforce_mining_gate`` /
    ``snapshot_now_only`` / ``mining_coverage_threshold``) are allowed and take
    precedence for that source — but a *production* context is a floor: the
    source constructor itself refuses ``strict_unknown_fields=False`` under a
    production run (R10 #3).
    """

    run_mode: str | None = None
    market: str | None = None
    calendar_id: str | None = None
    timezone: str | None = None
    pit_enforce: bool = False
    enforce_mining_gate: bool = False
    #: ``"snapshot_now_only"`` maps to ``snapshot_now_only=True`` on the source.
    snapshot_policy: str | None = None
    #: maps to ``mining_coverage_threshold`` on the source.
    coverage_policy: float | None = None

    @property
    def production(self) -> bool:
        if self.run_mode:
            from runtime.production_policy import is_production_mode

            return bool(is_production_mode(self.run_mode))
        return False

    @property
    def snapshot_now_only(self) -> bool:
        return self.snapshot_policy == "snapshot_now_only"


def _ctx_default(ctx: DataSourceBuildContext | None) -> DataSourceBuildContext:
    return ctx if ctx is not None else DataSourceBuildContext()


def _extract_source_spec(config: Any) -> tuple[str, dict[str, Any]]:
    """从 dict 或 dataclass 配置提取 (type, options)。
    
    参数:
        config: 数据源或运行时配置
    
    返回:
        tuple[str, dict[str, Any]]
    """
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
    """从 options 弹出首个非空配置项。
    
    参数:
        options: 数据源配置选项字典
        default: 见函数签名
        required: 见函数签名（可选）
        *names: 逻辑列名列表（可选）
    
    返回:
        Any
    """
    for name in names:
        if name in options and options[name] is not None:
            return options.pop(name)
    if required:
        joined = ", ".join(names)
        raise ValueError(f"Missing required data source option: {joined}")
    return default


def _ensure_no_extra_options(source_type: str, options: dict[str, Any]) -> None:
    """校验数据源配置无多余未识别选项。
    
    参数:
        source_type: 数据源类型字符串
        options: 数据源配置选项字典
    
    返回:
        无
    """
    if not options:
        return
    unexpected = ", ".join(sorted(options))
    raise ValueError(f"Unsupported options for data source '{source_type}': {unexpected}")


def _wrap_long_table(source: Any, *, long_table: bool) -> Any:
    """按需用 LongTableDataSource 包装内层数据源。
    
    参数:
        source: 见函数签名
        long_table: 是否包装为长表数据源（可选）
    
    返回:
        Any
    """
    if not long_table:
        return source
    from .long_table_source import LongTableDataSource

    return LongTableDataSource(source)


def _normalize_path(value: Any) -> str:
    """将路径解析为工作区绝对路径字符串。
    
    参数:
        value: 缓存值
    
    返回:
        str
    """
    return str(resolve_path(str(value)))


def _attach_bar_freq(source: Any, bar_freq: str) -> Any:
    """为数据源挂载 bar 频率属性。
    
    参数:
        source: 见函数签名
        bar_freq: bar 频率字符串
    
    返回:
        Any
    """
    if bar_freq and not getattr(source, "bar_freq", None):
        source.bar_freq = bar_freq
    return source


def _apply_date_range_to_source_config(
    config: Any,
    *,
    start_date: str | None,
    end_date: str | None,
) -> Any:
    """将顶层日期范围递归下发到 composite 子源。
    
    参数:
        config: 数据源或运行时配置
        start_date: 起始日期（可选）
        end_date: 结束日期（可选）
    
    返回:
        Any
    """
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


def build_data_source(config: Any, *, build_context: DataSourceBuildContext | None = None):
    """根据运行时配置构造数据源或组合数据源实例。

    参数:
        config: 数据源或运行时配置
        build_context: 引擎级 DataSourceBuildContext（R10 #1）；把 run_mode /
            production / mining 门禁递归注入所有子 source。None 时不注入，
            保持旧的纯 research 行为。

    返回:
        DataSource 实例
    """

    # YAML 可为 dict 或已解析的 dataclass；统一成 (type, options) 再分支
    ctx = _ctx_default(build_context)
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
                ),
                build_context=ctx,
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
        return _wrap_long_table(build_data_source(inner, build_context=ctx), long_table=True)

    if source_type == "data_access":
        dataset = _pop_option(options, "dataset", required=True)
        instrument_filter = _pop_option(options, "instrument_filter", default=None)
        normalize_timestamp = _pop_option(options, "normalize_timestamp", default=None)
        timestamp_unit = _pop_option(options, "timestamp_unit", default=None)
        params = _pop_option(options, "params", default=None)
        semantic_filters = _pop_option(options, "semantic_filters", default=None)
        read_mode = _pop_option(options, "read_mode", default="panel")
        read_auto = _pop_option(options, "read_auto", default=None)
        kind = _pop_option(options, "kind", default=None)
        if kind is not None:
            merged = dict(params or {})
            merged["kind"] = str(kind)
            params = merged
        _pop_option(options, "root", default=None)  # 忽略旧配置残留
        # R10 #1: production policy comes from the engine build context; the
        # source spec may still override per-source.  These options are popped
        # (not left for _ensure_no_extra_options to reject).
        run_mode = _pop_option(options, "run_mode", default=None)
        production = _pop_option(options, "production", default=None)
        strict_unknown_fields = _pop_option(
            options, "strict_unknown_fields", default=None
        )
        enforce_mining_gate = _pop_option(options, "enforce_mining_gate", default=None)
        snapshot_now_only = _pop_option(options, "snapshot_now_only", default=None)
        mining_coverage_threshold = _pop_option(
            options, "mining_coverage_threshold", default=None
        )
        # #收官轮 P0（Integration）：pit_enforce 从 engine build context 注入——
        # 之前 ``config.pit.enforce`` 只进 DataSourceBuildContext、factory 不取、
        # 构造 DataAccessSource 时不传，四层 PIT gate「有安全门、主通道没经过」。
        pit_enforce = _pop_option(options, "pit_enforce", default=None)
        # run_mode: context default, source override wins.  The constructor
        # derives ``production``/``strict_unknown_fields`` from the resolved
        # run_mode (R10 #3 makes production a floor the source cannot lower),
        # so the factory only propagates the mining/snapshot/PIT gates that have
        # no run_mode-derivable default.
        if run_mode is None:
            run_mode = ctx.run_mode
        if enforce_mining_gate is None:
            enforce_mining_gate = ctx.enforce_mining_gate
        if snapshot_now_only is None:
            snapshot_now_only = ctx.snapshot_now_only
        if mining_coverage_threshold is None:
            mining_coverage_threshold = ctx.coverage_policy
        if pit_enforce is None:
            pit_enforce = ctx.pit_enforce
        source = DataAccessSource(
            dataset=str(dataset),
            fields=fields,
            start_date=start_date,
            end_date=end_date,
            instrument_filter=instrument_filter,
            normalize_timestamp=normalize_timestamp,
            timestamp_unit=timestamp_unit,
            read_auto=read_auto,
            params=params,
            semantic_filters=semantic_filters,
            read_mode=str(read_mode),
            run_mode=run_mode,
            production=production,
            strict_unknown_fields=strict_unknown_fields,
            enforce_mining_gate=bool(enforce_mining_gate),
            snapshot_now_only=bool(snapshot_now_only),
            mining_coverage_threshold=mining_coverage_threshold,
            pit_enforce=bool(pit_enforce),
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
        inner = build_data_source(inner_cfg, build_context=ctx)
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
