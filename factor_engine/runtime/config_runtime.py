# -*- coding: utf-8
"""运行时配置解析：pipeline / engine / CLI 共用，避免 prod profile 语义漂移。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from factor_engine.runtime.config import FactorEngineConfig, MaterializationConfig
from factor_engine.runtime.dq_profiles import resolve_input_dq_thresholds, resolve_output_dq_thresholds


def _production_mode(config: FactorEngineConfig) -> bool:
    return str(config.run.mode).lower() == "production"


def _effective_bool(
    config: FactorEngineConfig,
    *,
    config_flag: bool,
    cli_flag: bool | None,
    production_default: bool,
) -> bool:
    """CLI 显式 True 优先；否则 config；production 模式可兜底开启。"""
    if cli_flag is True:
        return True
    if config_flag:
        return True
    return production_default and _production_mode(config)


@dataclass(frozen=True)
class ResolvedRunKwargs:
    """解析后的 ``FactorEngine.run`` / ``run_many`` 运行参数。

    R10-P0-025: ``market`` is the resolved MarketID (``ashare`` / ``us``);
    ``calendar_id`` is the trading-calendar identity (``SSE`` / ``SZSE`` /
    ``NYSE`` / ``NASDAQ`` / ...).  The old code stored ``config.run.calendar``
    into the ``market`` field — two different concepts sharing one variable, so
    a warmup / universe / market-specific semantic could not tell them apart.
    """

    auto_warmup: bool
    trim_warmup: bool
    market: str | None
    input_dq_check: bool
    input_dq_strict: bool
    input_dq_thresholds: Any
    pit_enforce: bool
    pit_forbid_forward_fill: bool
    calendar_id: str | None = None

    def to_run_kwargs(self) -> dict[str, Any]:
        """``FactorEngine.run`` / ``run_many`` 参数字典。"""
        return {
            "auto_warmup": self.auto_warmup,
            "trim_warmup": self.trim_warmup,
            "market": self.market,
            "input_dq_check": self.input_dq_check,
            "input_dq_strict": self.input_dq_strict,
            "input_dq_thresholds": self.input_dq_thresholds,
            "pit_enforce": self.pit_enforce,
            "pit_forbid_forward_fill": self.pit_forbid_forward_fill,
        }

    def with_calendar_id(self, calendar_id: str | None) -> "ResolvedRunKwargs":
        """Attach the resolved calendar identity (distinct from ``market``)."""
        return replace(self, calendar_id=calendar_id)


@dataclass(frozen=True)
class ResolvedMaterializeKwargs:
    """解析后的 ``FactorEngine.materialize`` / ``materialize_incremental`` 参数。"""

    lake_root: str | None
    factor_id: str | None
    author: str | None
    frequency: str | None
    description: str | None
    expression: str | None
    auto_warmup: bool
    trim_warmup: bool
    market: str | None
    dq_check: bool
    dq_strict: bool
    dq_thresholds: Any
    input_dq_check: bool
    input_dq_strict: bool
    input_dq_thresholds: Any
    preserve_invalid_rows: bool
    value_dtype: str
    write_target: str
    data_source_config: dict[str, Any]
    pit_enforce: bool
    pit_forbid_forward_fill: bool
    isolate_partition_failures: bool
    clickhouse_table: str | None
    clickhouse_host: str | None
    clickhouse_port: int | None
    clickhouse_database: str | None
    clickhouse_username: str | None
    clickhouse_password: str | None
    clickhouse_secure: bool | None
    ch_ensure_table: bool
    resume_materialize: bool
    staging_dataset: str
    storage_format: str
    partition_columns: tuple[str, ...] | None
    since: str | None
    end_date: str | None
    lookback_extra: int
    recompute_tail_bars: int | None

    def to_engine_materialize_kwargs(self) -> dict[str, Any]:
        """``FactorEngine.materialize(**kwargs)`` 参数字典。"""
        return {
            "lake_root": self.lake_root,
            "factor_id": self.factor_id,
            "author": self.author,
            "frequency": self.frequency,
            "description": self.description,
            "expression": self.expression,
            "auto_warmup": self.auto_warmup,
            "trim_warmup": self.trim_warmup,
            "market": self.market,
            "dq_check": self.dq_check,
            "dq_strict": self.dq_strict,
            "dq_thresholds": self.dq_thresholds,
            "input_dq_check": self.input_dq_check,
            "input_dq_strict": self.input_dq_strict,
            "input_dq_thresholds": self.input_dq_thresholds,
            "preserve_invalid_rows": self.preserve_invalid_rows,
            "value_dtype": self.value_dtype,
            "write_target": self.write_target,
            "data_source_config": self.data_source_config,
            "pit_enforce": self.pit_enforce,
            "pit_forbid_forward_fill": self.pit_forbid_forward_fill,
            "resume_materialize": self.resume_materialize,
            "isolate_partition_failures": self.isolate_partition_failures,
            "clickhouse_table": self.clickhouse_table,
            "ch_host": self.clickhouse_host,
            "ch_port": self.clickhouse_port,
            "ch_database": self.clickhouse_database,
            "ch_username": self.clickhouse_username,
            "ch_password": self.clickhouse_password,
            "ch_secure": self.clickhouse_secure,
            "ch_ensure_table": self.ch_ensure_table,
            "staging_dataset": self.staging_dataset,
            "storage_format": self.storage_format,
            "partition_columns": list(self.partition_columns)
            if self.partition_columns
            else None,
        }

    def to_run_kwargs(self) -> dict[str, Any]:
        """``FactorEngine.run`` / ``run_many`` 共用 run 段参数字典。"""
        return {
            "auto_warmup": self.auto_warmup,
            "trim_warmup": self.trim_warmup,
            "market": self.market,
            "input_dq_check": self.input_dq_check,
            "input_dq_strict": self.input_dq_strict,
            "input_dq_thresholds": self.input_dq_thresholds,
            "pit_enforce": self.pit_enforce,
            "pit_forbid_forward_fill": self.pit_forbid_forward_fill,
        }

    def to_incremental_materialize_kwargs(self) -> dict[str, Any]:
        """``FactorEngine.materialize_incremental(**kwargs)`` 参数字典。"""
        out = self.to_engine_materialize_kwargs()
        out.update(
            {
                "since": self.since,
                "end_date": self.end_date,
                "lookback_extra": self.lookback_extra,
                "recompute_tail_bars": self.recompute_tail_bars,
            }
        )
        return out


@dataclass(frozen=True)
class PipelineConfigOverrides:
    """pipeline / CLI 对 resolve_* 的统一覆盖（目录 batch 与 engine batch 共用）。"""

    dq_check: bool = False
    dq_strict: bool = True
    input_dq_check: bool = False
    input_dq_strict: bool = True
    write_target: str | None = None
    preserve_invalid_rows: bool | None = None
    resume_materialize: bool = False
    since: str | None = None
    end_date: str | None = None
    lookback_extra: int | None = None
    recompute_tail_bars: int | None = None

    def to_resolve_run_kwargs(self) -> dict[str, Any]:
        """转为 ``resolve_run_kwargs`` 可接收的 CLI 覆盖参数字典。"""
        return {
            "cli_input_dq_check": True if self.input_dq_check else None,
            "cli_input_dq_strict": self.input_dq_strict if self.input_dq_check else None,
        }

    def to_resolve_materialize_kwargs(self) -> dict[str, Any]:
        """转为 ``resolve_materialize_kwargs`` 可接收的 CLI 覆盖参数字典。"""
        out = self.to_resolve_run_kwargs()
        out.update(
            {
                "cli_dq_check": True if self.dq_check else None,
                "cli_dq_strict": self.dq_strict if self.dq_check else None,
                "write_target_override": self.write_target,
                "preserve_invalid_rows_override": self.preserve_invalid_rows,
                "since_override": self.since,
                "end_date_override": self.end_date,
                "lookback_extra_override": self.lookback_extra,
                "recompute_tail_bars_override": self.recompute_tail_bars,
                "resume_materialize_override": self.resume_materialize or None,
            }
        )
        return out


def resolve_run_kwargs_for_pipeline(
    config: FactorEngineConfig,
    pipeline: PipelineConfigOverrides | None = None,
) -> ResolvedRunKwargs:
    """结合 pipeline CLI 覆盖解析 run 参数。"""
    extra = pipeline.to_resolve_run_kwargs() if pipeline is not None else {}
    return resolve_run_kwargs(config, **extra)


def resolve_materialize_kwargs_for_pipeline(
    config: FactorEngineConfig,
    pipeline: PipelineConfigOverrides | None = None,
) -> ResolvedMaterializeKwargs:
    """结合 pipeline CLI 覆盖解析物化参数。"""
    extra = pipeline.to_resolve_materialize_kwargs() if pipeline is not None else {}
    return resolve_materialize_kwargs(config, **extra)


def _input_dq_thresholds_key(th: Any) -> tuple[Any, ...] | None:
    if th is None:
        return None
    return (
        th.min_rows,
        th.min_non_null_ratio,
        th.min_instruments,
        th.max_inf_ratio,
    )


def _output_dq_thresholds_key(th: Any) -> tuple[Any, ...] | None:
    if th is None:
        return None
    return (
        th.min_coverage,
        th.max_nan_ratio,
        th.max_inf_ratio,
        th.max_abs_value,
        th.min_rows,
        th.min_instruments_per_day,
    )


def build_data_source_config(config: FactorEngineConfig) -> dict[str, Any]:
    """从引擎配置提取数据源 type + options 字典。"""
    return {"type": config.data_source.type, **config.data_source.options}


def config_data_scope_key(config: FactorEngineConfig) -> str:
    """配置对应的数据源作用域键（run_many 分组用）。"""
    from factor_engine.storage.data_scope import compute_data_scope
    from factor_engine.storage.factory import DataSourceBuildContext, build_data_source

    build_context = DataSourceBuildContext(
        run_mode=getattr(config.run, "mode", None),
        market=getattr(config.run, "market", None),
        calendar_id=getattr(config.run, "calendar", None),
        pit_enforce=bool(getattr(config.pit, "enforce", False)),
    )
    ds = build_data_source(config.data_source, build_context=build_context)
    return compute_data_scope(ds)


def config_run_batch_key(
    config: FactorEngineConfig,
    *,
    pipeline: PipelineConfigOverrides | None = None,
) -> tuple[Any, ...]:
    """同 data_scope 内 run_many 子分组键（可 hash）。"""
    opts = resolve_run_kwargs_for_pipeline(config, pipeline)
    return (
        opts.auto_warmup,
        opts.trim_warmup,
        opts.market,
        opts.calendar_id,  # R10-P0-025: market and calendar are separate keys
        opts.input_dq_check,
        opts.input_dq_strict,
        _input_dq_thresholds_key(opts.input_dq_thresholds),
        opts.pit_enforce,
        opts.pit_forbid_forward_fill,
    )


def config_materialize_batch_key(
    config: FactorEngineConfig,
    *,
    pipeline: PipelineConfigOverrides | None = None,
) -> tuple[Any, ...]:
    """同 data_scope 内物化 batch 键（不含 factor_id/author 等因子专属字段）。"""
    opts = resolve_materialize_kwargs_for_pipeline(config, pipeline)
    return (
        opts.lake_root,
        opts.write_target,
        opts.staging_dataset,
        opts.preserve_invalid_rows,
        opts.value_dtype,
        opts.dq_check,
        opts.dq_strict,
        _output_dq_thresholds_key(opts.dq_thresholds),
        *config_run_batch_key(config, pipeline=pipeline),
        opts.isolate_partition_failures,
        opts.clickhouse_table,
        opts.clickhouse_host,
        opts.clickhouse_port,
        opts.clickhouse_database,
        opts.clickhouse_username,
        opts.clickhouse_password,
        opts.clickhouse_secure,
        opts.ch_ensure_table,
        opts.resume_materialize,
        opts.storage_format,
        opts.partition_columns,
        opts.since,
        opts.end_date,
        opts.lookback_extra,
        opts.recompute_tail_bars,
    )


def production_run_flags_equal(a: FactorEngineConfig, b: FactorEngineConfig) -> bool:
    """两配置 production run 开关是否一致（同组 run_many 前提）。"""
    return config_run_batch_key(a) == config_run_batch_key(b)


def _resolve_market(config: FactorEngineConfig) -> tuple[str | None, str | None]:
    """Resolve ``(market_id, calendar_id)`` — R10-P0-025.

    ``market`` is a MarketID (``ashare`` / ``us``), ``calendar_id`` is the
    trading-calendar identity (``SSE`` / ``SZSE`` / ``NYSE`` / ``NASDAQ``).
    When only a calendar is configured the MarketID is inferred from it, so the
    two concepts are never conflated in one variable.
    """
    explicit = str(getattr(config.run, "market", None) or "").strip()
    calendar_id = str(config.run.calendar or "").strip() or None
    if explicit:
        return explicit, calendar_id
    cal = (calendar_id or "").lower()
    if cal in {"ashare", "a_share", "cn", "china", "sse", "szse"}:
        return "ashare", calendar_id
    if cal in {"us", "usa", "nyse", "nasdaq"}:
        return "us", calendar_id
    return None, calendar_id


def resolve_run_kwargs(
    config: FactorEngineConfig,
    *,
    cli_input_dq_check: bool | None = None,
    cli_input_dq_strict: bool | None = None,
) -> ResolvedRunKwargs:
    """从 YAML 配置解析 run 段参数（含 production 默认与 CLI 覆盖）。"""
    prod = _production_mode(config)
    input_dq_check = _effective_bool(
        config,
        config_flag=config.dq.strict,
        cli_flag=cli_input_dq_check,
        production_default=prod,
    )
    input_dq_strict = (
        cli_input_dq_strict
        if cli_input_dq_strict is not None
        else config.dq.input_strict
    )
    market_id, calendar_id = _resolve_market(config)
    return ResolvedRunKwargs(
        auto_warmup=_effective_bool(
            config,
            config_flag=config.run.auto_warmup,
            cli_flag=None,
            production_default=True,
        ),
        trim_warmup=config.run.trim_warmup,
        market=market_id,
        input_dq_check=input_dq_check,
        input_dq_strict=input_dq_strict,
        input_dq_thresholds=resolve_input_dq_thresholds(config.dq.profile),
        pit_enforce=config.pit.enforce,
        pit_forbid_forward_fill=config.pit.forbid_forward_fill,
        calendar_id=calendar_id,
    )


def resolve_materialize_kwargs(
    config: FactorEngineConfig,
    *,
    cli_dq_check: bool | None = None,
    cli_dq_strict: bool | None = None,
    cli_input_dq_check: bool | None = None,
    cli_input_dq_strict: bool | None = None,
    lake_root_override: str | None = None,
    factor_id_override: str | None = None,
    author_override: str | None = None,
    frequency_override: str | None = None,
    description_override: str | None = None,
    expression_override: str | None = None,
    write_target_override: str | None = None,
    preserve_invalid_rows_override: bool | None = None,
    since_override: str | None = None,
    end_date_override: str | None = None,
    lookback_extra_override: int | None = None,
    recompute_tail_bars_override: int | None = None,
    resume_materialize_override: bool | None = None,
) -> ResolvedMaterializeKwargs:
    """从 YAML 配置解析物化段参数（含增量窗口与 ClickHouse 连接）。"""
    run = resolve_run_kwargs(
        config,
        cli_input_dq_check=cli_input_dq_check,
        cli_input_dq_strict=cli_input_dq_strict,
    )
    prod = _production_mode(config)
    mat: MaterializationConfig | None = config.materialization
    inc = mat.incremental if mat else None

    dq_check = _effective_bool(
        config,
        config_flag=config.dq.strict,
        cli_flag=cli_dq_check,
        production_default=prod,
    )
    dq_strict = cli_dq_strict if cli_dq_strict is not None else (config.dq.strict or prod)

    return ResolvedMaterializeKwargs(
        lake_root=lake_root_override or (mat.lake_root if mat else None),
        factor_id=factor_id_override or (mat.factor_id if mat else None),
        author=author_override or (mat.author if mat else None),
        frequency=frequency_override or (mat.frequency if mat else None),
        description=description_override or (mat.description if mat else None),
        expression=expression_override or (mat.expression if mat else None) or config.factor.expr,
        auto_warmup=run.auto_warmup,
        trim_warmup=run.trim_warmup,
        market=run.market,
        dq_check=dq_check,
        dq_strict=dq_strict,
        dq_thresholds=resolve_output_dq_thresholds(config.dq.profile),
        input_dq_check=run.input_dq_check,
        input_dq_strict=run.input_dq_strict,
        input_dq_thresholds=run.input_dq_thresholds,
        preserve_invalid_rows=(
            preserve_invalid_rows_override
            if preserve_invalid_rows_override is not None
            else bool(mat.preserve_invalid_rows) if mat else False
        ),
        value_dtype=str(mat.value_dtype) if mat else "float32",
        write_target=write_target_override or (str(mat.target) if mat else "local"),
        data_source_config=build_data_source_config(config),
        pit_enforce=run.pit_enforce,
        pit_forbid_forward_fill=run.pit_forbid_forward_fill,
        isolate_partition_failures=bool(mat.isolate_partition_failures) if mat else True,
        clickhouse_table=mat.clickhouse_table if mat else None,
        clickhouse_host=mat.clickhouse_host if mat else None,
        clickhouse_port=mat.clickhouse_port if mat else None,
        clickhouse_database=mat.clickhouse_database if mat else None,
        clickhouse_username=mat.clickhouse_username if mat else None,
        clickhouse_password=mat.clickhouse_password if mat else None,
        clickhouse_secure=mat.clickhouse_secure if mat else None,
        ch_ensure_table=bool(mat.ch_ensure_table) if mat else True,
        resume_materialize=(
            resume_materialize_override
            if resume_materialize_override is not None
            else bool(mat.resume_materialize) if mat else False
        ),
        staging_dataset=str(mat.staging_dataset) if mat else "factor_lake_staging",
        storage_format=str(mat.storage_format) if mat else "long",
        partition_columns=mat.partition_columns if mat else None,
        since=since_override or (inc.since if inc else None),
        end_date=end_date_override or (inc.end_date if inc else None),
        lookback_extra=(
            lookback_extra_override
            if lookback_extra_override is not None
            else int(inc.lookback_extra) if inc else 5
        ),
        recompute_tail_bars=(
            recompute_tail_bars_override
            if recompute_tail_bars_override is not None
            else (inc.recompute_tail_bars if inc else None)
        ),
    )
