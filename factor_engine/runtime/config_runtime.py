# -*- coding: utf-8
"""运行时配置解析：pipeline / engine / CLI 共用，避免 prod profile 语义漂移。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from runtime.config import FactorEngineConfig, MaterializationConfig
from runtime.dq_profiles import resolve_input_dq_thresholds, resolve_output_dq_thresholds


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
    auto_warmup: bool
    trim_warmup: bool
    market: str | None
    input_dq_check: bool
    input_dq_strict: bool
    input_dq_thresholds: Any
    pit_enforce: bool
    pit_forbid_forward_fill: bool


@dataclass(frozen=True)
class ResolvedMaterializeKwargs:
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
    since: str | None
    end_date: str | None
    lookback_extra: int
    recompute_tail_bars: int | None


def build_data_source_config(config: FactorEngineConfig) -> dict[str, Any]:
    return {"type": config.data_source.type, **config.data_source.options}


def resolve_run_kwargs(
    config: FactorEngineConfig,
    *,
    cli_input_dq_check: bool | None = None,
    cli_input_dq_strict: bool | None = None,
) -> ResolvedRunKwargs:
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
    return ResolvedRunKwargs(
        auto_warmup=config.run.auto_warmup,
        trim_warmup=config.run.trim_warmup,
        market=config.run.calendar,
        input_dq_check=input_dq_check,
        input_dq_strict=input_dq_strict,
        input_dq_thresholds=resolve_input_dq_thresholds(config.dq.profile),
        pit_enforce=config.pit.enforce,
        pit_forbid_forward_fill=config.pit.forbid_forward_fill,
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
