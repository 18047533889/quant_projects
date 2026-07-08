from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
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


@dataclass(frozen=True)
class DataSourceConfig:
    type: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendConfig:
    type: str = "pandas"


@dataclass(frozen=True)
class EngineConfig:
    enable_cache: bool = True
    plan_cache_dir: str | None = None


@dataclass(frozen=True)
class RunConfig:
    """运行模式：research 宽松 / production 严格（warmup、DQ 等）。"""

    mode: str = "research"
    auto_warmup: bool = False
    trim_warmup: bool = True
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
    """增量物化参数（与 ``materialize_incremental`` 对齐）。"""

    since: str | None = None
    end_date: str | None = None
    lookback_extra: int = 5
    recompute_tail_bars: int | None = None


@dataclass(frozen=True)
class MaterializationConfig:
    """因子物化到数据湖的元信息（与 ``FactorEngine.materialize*`` 配套）。"""

    lake_root: str | None = None
    factor_id: str | None = None
    author: str | None = None
    frequency: str | None = None
    description: str | None = None
    expression: str | None = None
    # local | staging | both | clickhouse | staging_clickhouse
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


def _resolve_optional_path(value: Any, *, base_dir: Path) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return str(resolve_path(text, base_dir=base_dir))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_profile_path(profile: str) -> Path:
    name = str(profile).strip()
    if not name:
        raise ValueError("profile name must be non-empty")
    path = _PROFILES_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Unknown profile '{name}': {path}")
    return path


def load_profile(profile: str) -> dict[str, Any]:
    path = _resolve_profile_path(profile)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Profile file must be a mapping: {path}")
    return payload


def load_config(path: str | Path, *, profile: str | None = None) -> FactorEngineConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    profile_name = profile or payload.get("profile")
    if profile_name:
        profile_payload = load_profile(str(profile_name))
        payload = _deep_merge(profile_payload, payload)

    base_dir = config_path.parent.resolve()

    factor_payload = payload.get("factor", {})
    data_source_payload = payload.get("data_source", {})
    backend_payload = payload.get("backend", {})
    engine_payload = payload.get("engine", {})
    run_payload = payload.get("run", {})
    dq_payload = payload.get("dq", {})
    pit_payload = payload.get("pit", {})
    materialization_payload = payload.get("materialization", payload.get("materialize"))

    if "name" not in factor_payload or "expr" not in factor_payload:
        raise ValueError("Config must include factor.name and factor.expr")

    if "type" not in data_source_payload:
        raise ValueError("Config must include data_source.type")

    factor_config = FactorDefinitionConfig(
        name=factor_payload["name"],
        expr=factor_payload["expr"],
        freq=factor_payload.get("freq", "1d"),
        universe=factor_payload.get("universe"),
        description=factor_payload.get("description"),
    )
    data_source_config = DataSourceConfig(
        type=data_source_payload["type"],
        options={key: value for key, value in data_source_payload.items() if key != "type"},
    )
    backend_config = BackendConfig(type=backend_payload.get("type", "pandas"))
    engine_config = EngineConfig(
        enable_cache=engine_payload.get("enable_cache", True),
        plan_cache_dir=_resolve_optional_path(engine_payload.get("plan_cache_dir"), base_dir=base_dir),
    )
    materialization_config = None
    if materialization_payload is not None:
        if not isinstance(materialization_payload, dict):
            raise ValueError("Config materialization section must be a mapping")
        materialization_config = MaterializationConfig(
            lake_root=_resolve_optional_path(materialization_payload.get("lake_root"), base_dir=base_dir)
            or str(default_factor_lake_root()),
            factor_id=materialization_payload.get("factor_id"),
            author=materialization_payload.get("author"),
            frequency=materialization_payload.get("frequency"),
            description=materialization_payload.get("description"),
            expression=materialization_payload.get("expression"),
            target=str(materialization_payload.get("target", "local")),
            preserve_invalid_rows=bool(materialization_payload.get("preserve_invalid_rows", False)),
            value_dtype=str(materialization_payload.get("value_dtype", "float32")),
            isolate_partition_failures=bool(
                materialization_payload.get("isolate_partition_failures", True)
            ),
            resume_materialize=bool(materialization_payload.get("resume_materialize", False)),
            ch_ensure_table=bool(materialization_payload.get("ch_ensure_table", True)),
            incremental=(
                IncrementalConfig(
                    since=inc_payload.get("since"),
                    end_date=inc_payload.get("end_date"),
                    lookback_extra=int(inc_payload.get("lookback_extra", 5)),
                    recompute_tail_bars=inc_payload.get("recompute_tail_bars"),
                )
                if isinstance(inc_payload := materialization_payload.get("incremental"), dict)
                else None
            ),
            clickhouse_table=materialization_payload.get("clickhouse_table"),
            clickhouse_host=materialization_payload.get("clickhouse_host"),
            clickhouse_port=materialization_payload.get("clickhouse_port"),
            clickhouse_database=materialization_payload.get("clickhouse_database"),
            clickhouse_username=materialization_payload.get("clickhouse_username"),
            clickhouse_password=materialization_payload.get("clickhouse_password"),
            clickhouse_secure=materialization_payload.get("clickhouse_secure"),
        )

    run_config = RunConfig(
        mode=str(run_payload.get("mode", "research")),
        auto_warmup=bool(run_payload.get("auto_warmup", False)),
        trim_warmup=bool(run_payload.get("trim_warmup", True)),
        calendar=run_payload.get("calendar"),
    )
    dq_config = DQConfig(
        profile=dq_payload.get("profile"),
        strict=bool(dq_payload.get("strict", False)),
        input_strict=bool(dq_payload.get("input_strict", True)),
    )
    pit_config = PITConfig(
        enforce=bool(pit_payload.get("enforce", False)),
        forbid_forward_fill=bool(pit_payload.get("forbid_forward_fill", False)),
    )

    return FactorEngineConfig(
        factor=factor_config,
        data_source=data_source_config,
        backend=backend_config,
        engine=engine_config,
        materialization=materialization_config,
        run=run_config,
        dq=dq_config,
        pit=pit_config,
    )
