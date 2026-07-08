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
class MaterializationConfig:
    """因子物化到数据湖的元信息（与 ``FactorEngine.materialize*`` 配套）。"""

    lake_root: str | None = None
    factor_id: str | None = None
    author: str | None = None
    frequency: str | None = None
    description: str | None = None
    expression: str | None = None


@dataclass(frozen=True)
class FactorEngineConfig:
    factor: FactorDefinitionConfig
    data_source: DataSourceConfig
    backend: BackendConfig = field(default_factory=BackendConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    materialization: MaterializationConfig | None = None


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
        )

    return FactorEngineConfig(
        factor=factor_config,
        data_source=data_source_config,
        backend=backend_config,
        engine=engine_config,
        materialization=materialization_config,
    )
