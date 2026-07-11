from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

import yaml

from workspace_paths import default_factor_lake_root, resolve_path


@dataclass(frozen=True)
class FactorDefinitionConfig:
    name: str
    expr: str
    calc_mode: str = "expr"  # "expr" | "code"
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
    tiny_run: bool = False
    factor_engine_operators: str | None = None  # 外部算子库绝对路径
    cache_root: str | None = None  # 持久化缓存根目录（跨进程共享）


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


def load_config(path: str | Path) -> FactorEngineConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text()) or {}
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

    calc_mode = factor_payload.get("calc_mode", "expr")
    if calc_mode not in ("expr", "code"):
        raise ValueError(f"calc_mode must be 'expr' or 'code', got: {calc_mode}")
    factor_config = FactorDefinitionConfig(
        name=factor_payload["name"],
        expr=factor_payload.get("expr", "") if calc_mode == "code" else factor_payload["expr"],
        calc_mode=calc_mode,
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
        tiny_run=engine_payload.get("tiny_run", False),
        factor_engine_operators=engine_payload.get("factor_engine_operators"),
        cache_root=engine_payload.get("cache_root"),
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
