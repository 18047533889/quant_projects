"""Strongly typed FactorEngine YAML configuration."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml
from workspace_paths import default_factor_lake_root, resolve_path

_PROFILES_DIR = Path(__file__).resolve().parent.parent / "examples" / "profiles"

@dataclass(frozen=True)
class FactorDefinitionConfig:
    name: str
    expr: str
    freq: str = "1d"
    universe: str | None = None
    description: str | None = None
    surface: str = "daily"
    dialect: str = "native"
    dialect_version: str | None = None

@dataclass(frozen=True)
class DataSourceConfig:
    type: str
    options: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class BackendConfig:
    type: str = "auto"

@dataclass(frozen=True)
class EngineConfig:
    enable_cache: bool = True
    plan_cache_dir: str | None = None

@dataclass(frozen=True)
class RunConfig:
    mode: str = "research"
    auto_warmup: bool = False
    trim_warmup: bool = True
    # R10-P0-025: ``market`` is the MarketID (ashare/us); ``calendar`` is the
    # trading-calendar identity (SSE/SZSE/NYSE/NASDAQ).  They are resolved
    # separately (``runtime.config_runtime._resolve_market``) and never merged
    # into one variable.
    market: str | None = None
    calendar: str | None = None

@dataclass(frozen=True)
class DQConfig:
    profile: str | None = None
    strict: bool = False
    input_strict: bool = True

@dataclass(frozen=True)
class PITConfig:
    enforce: bool = False
    forbid_forward_fill: bool = False

@dataclass(frozen=True)
class IncrementalConfig:
    since: str | None = None
    end_date: str | None = None
    lookback_extra: int = 5
    recompute_tail_bars: int | None = None

@dataclass(frozen=True)
class MaterializationConfig:
    lake_root: str | None = None
    factor_id: str | None = None
    author: str | None = None
    frequency: str | None = None
    description: str | None = None
    expression: str | None = None
    target: str = "local"
    preserve_invalid_rows: bool = False
    value_dtype: str = "float32"
    isolate_partition_failures: bool = True
    resume_materialize: bool = False
    ch_ensure_table: bool = True
    incremental: IncrementalConfig | None = None
    clickhouse_table: str | None = None
    clickhouse_host: str | None = None
    clickhouse_port: int | None = None
    clickhouse_database: str | None = None
    clickhouse_username: str | None = None
    clickhouse_password: str | None = None
    clickhouse_secure: bool | None = None
    staging_dataset: str = "factor_lake_staging"
    storage_format: str = "long"
    partition_columns: tuple[str, ...] | None = None

@dataclass(frozen=True)
class PipelineConfig:
    batched_engine: bool = False

@dataclass(frozen=True)
class FactorEngineConfig:
    factor: FactorDefinitionConfig
    data_source: DataSourceConfig
    backend: BackendConfig = field(default_factory=BackendConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    materialization: MaterializationConfig | None = None
    run: RunConfig = field(default_factory=RunConfig)
    dq: DQConfig = field(default_factory=DQConfig)
    pit: PITConfig = field(default_factory=PITConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

def _parse_partition_columns(raw: Any) -> tuple[str, ...] | None:
    if raw is None: return None
    if isinstance(raw, str): raw=[raw]
    if not isinstance(raw,(list,tuple)): raise ValueError("materialization.partition_columns 必须是字符串列表")
    cols=tuple(str(c) for c in raw if str(c)); return cols or None

def _resolve_optional_path(value: Any, *, base_dir: Path) -> str | None:
    if value is None: return None
    text=str(value).strip()
    return str(resolve_path(text,base_dir=base_dir)) if text else None

def _deep_merge(base: dict[str,Any], override: dict[str,Any]) -> dict[str,Any]:
    merged=dict(base)
    for key,value in override.items():
        merged[key]=_deep_merge(merged[key],value) if key in merged and isinstance(merged[key],dict) and isinstance(value,dict) else value
    return merged

def _resolve_profile_path(profile: str) -> Path:
    name=str(profile).strip()
    if not name: raise ValueError("profile name must be non-empty")
    path=_PROFILES_DIR/f"{name}.yaml"
    if not path.is_file(): raise FileNotFoundError(f"Unknown profile '{name}': {path}")
    return path

def load_profile(profile: str) -> dict[str,Any]:
    path=_resolve_profile_path(profile); payload=yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload,dict): raise ValueError(f"Profile file must be a mapping: {path}")
    return payload

def _factor_dialect(factor_payload: dict[str,Any]) -> tuple[str,str,str|None]:
    surface=str(factor_payload.get("surface") or factor_payload.get("dsl_surface") or "daily")
    dialect=str(factor_payload.get("dialect") or "native").lower(); version=factor_payload.get("dialect_version")
    if surface=="lqtp" and dialect=="native": dialect="lqtp"
    if dialect not in {"native","lqtp"}: raise ValueError(f"unsupported factor.dialect={dialect!r}")
    if dialect=="lqtp":
        from api.lqtp_compat import DEFAULT_LQTP_DIALECT_VERSION
        version=str(version or DEFAULT_LQTP_DIALECT_VERSION)
        if version!=DEFAULT_LQTP_DIALECT_VERSION: raise ValueError(f"unsupported factor.dialect_version={version!r}; supported={DEFAULT_LQTP_DIALECT_VERSION!r}")
    elif version is not None: raise ValueError("factor.dialect_version is only valid when factor.dialect='lqtp'")
    return surface,dialect,version

def load_config(path: str|Path, *, profile: str|None=None) -> FactorEngineConfig:
    config_path=Path(path); payload=yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload,dict): raise ValueError(f"Config must be a mapping: {config_path}")
    profile_name=profile or payload.get("profile")
    if profile_name: payload=_deep_merge(load_profile(str(profile_name)),payload)
    base_dir=config_path.parent.resolve()
    fp=payload.get("factor",{}); dsp=payload.get("data_source",{}); bp=payload.get("backend",{}); ep=payload.get("engine",{}); rp=payload.get("run",{}); dqp=payload.get("dq",{}); pp=payload.get("pit",{}); pipe=payload.get("pipeline",{}); mp=payload.get("materialization",payload.get("materialize"))
    if not isinstance(fp,dict) or "name" not in fp or "expr" not in fp: raise ValueError("Config must include factor.name and factor.expr")
    if not isinstance(dsp,dict) or "type" not in dsp: raise ValueError("Config must include data_source.type")
    canonical_surface,dialect,version=_factor_dialect(fp)
    # FactorEngine.from_loaded_config in the current public API still consumes
    # one parser-surface argument. Preserve the orthogonal metadata on the config
    # while bridging LQTP configs through the legacy surface token until that API
    # is removed.
    parser_surface="lqtp" if dialect=="lqtp" else canonical_surface
    factor_config=FactorDefinitionConfig(name=str(fp["name"]),expr=str(fp["expr"]),freq=str(fp.get("freq","1d")),universe=fp.get("universe"),description=fp.get("description"),surface=parser_surface,dialect=dialect,dialect_version=version)
    data_source_config=DataSourceConfig(type=str(dsp["type"]),options={k:v for k,v in dsp.items() if k!="type"})
    backend_config=BackendConfig(type=str(bp.get("type","auto")))
    engine_config=EngineConfig(enable_cache=bool(ep.get("enable_cache",True)),plan_cache_dir=_resolve_optional_path(ep.get("plan_cache_dir"),base_dir=base_dir))
    materialization_config=None
    if mp is not None:
        if not isinstance(mp,dict): raise ValueError("Config materialization section must be a mapping")
        inc=mp.get("incremental")
        materialization_config=MaterializationConfig(
            lake_root=_resolve_optional_path(mp.get("lake_root"),base_dir=base_dir) or str(default_factor_lake_root()),factor_id=mp.get("factor_id"),author=mp.get("author"),frequency=mp.get("frequency"),description=mp.get("description"),expression=mp.get("expression"),target=str(mp.get("target","local")),preserve_invalid_rows=bool(mp.get("preserve_invalid_rows",False)),value_dtype=str(mp.get("value_dtype","float32")),isolate_partition_failures=bool(mp.get("isolate_partition_failures",True)),resume_materialize=bool(mp.get("resume_materialize",False)),ch_ensure_table=bool(mp.get("ch_ensure_table",True)),
            incremental=IncrementalConfig(since=inc.get("since"),end_date=inc.get("end_date"),lookback_extra=int(inc.get("lookback_extra",5)),recompute_tail_bars=inc.get("recompute_tail_bars")) if isinstance(inc,dict) else None,
            clickhouse_table=mp.get("clickhouse_table"),clickhouse_host=mp.get("clickhouse_host"),clickhouse_port=mp.get("clickhouse_port"),clickhouse_database=mp.get("clickhouse_database"),clickhouse_username=mp.get("clickhouse_username"),clickhouse_password=mp.get("clickhouse_password"),clickhouse_secure=mp.get("clickhouse_secure"),staging_dataset=str(mp.get("staging_dataset","factor_lake_staging")),storage_format=str(mp.get("storage_format","long")),partition_columns=_parse_partition_columns(mp.get("partition_columns")))
    return FactorEngineConfig(factor=factor_config,data_source=data_source_config,backend=backend_config,engine=engine_config,materialization=materialization_config,
        run=RunConfig(mode=str(rp.get("mode","research")),auto_warmup=bool(rp.get("auto_warmup",False)),trim_warmup=bool(rp.get("trim_warmup",True)),market=rp.get("market"),calendar=rp.get("calendar")),
        dq=DQConfig(profile=dqp.get("profile"),strict=bool(dqp.get("strict",False)),input_strict=bool(dqp.get("input_strict",True))),
        pit=PITConfig(enforce=bool(pp.get("enforce",False)),forbid_forward_fill=bool(pp.get("forbid_forward_fill",False))),pipeline=PipelineConfig(batched_engine=bool(pipe.get("batched_engine",False))))
