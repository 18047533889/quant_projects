from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from workspace_paths import default_factor_evaluation_root, default_factor_lake_root


@dataclass(frozen=True)
class MetaConfig:
    factor_id: str
    run_name: str | None = None
    primary_horizon: int | None = None
    description: str | None = None


@dataclass(frozen=True)
class SourceConfig:
    factor_lake_root: str
    market_data_path: str
    market_timestamp_col: str = "timestamp"
    market_symbol_col: str = "symbol"
    market_price_col: str = "open"
    universe_path: str | None = None
    universe_timestamp_col: str = "timestamp"
    universe_symbol_col: str = "symbol"
    universe_membership_col: str | None = None
    universe_include_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunConfig:
    start: str | None = None
    end: str | None = None
    horizons: tuple[int, ...] = (1, 5, 10, 20)
    n_quantiles: int = 10
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99
    standardize: bool = True
    min_assets_per_date: int = 20
    direction: float = 1.0
    annualization_factor: int = 252


@dataclass(frozen=True)
class OutputConfig:
    root: str | None = None
    save_artifacts: bool = True


@dataclass(frozen=True)
class FactorEvaluationConfig:
    meta: MetaConfig
    source: SourceConfig
    run: RunConfig = field(default_factory=RunConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _as_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} 必须是 object")
    return dict(value)


def _as_list(value: Any, *, field_name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} 必须是数组")
    return list(value)


def _normalize_scalar_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _require_non_empty_text(value: Any, *, field_name: str) -> str:
    text = _normalize_scalar_text(value)
    if text is None or not text.strip():
        raise ValueError(f"{field_name} 不能为空")
    return text.strip()


def _resolve_path_text(
    value: Any,
    *,
    field_name: str,
    base_dir: Path,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{field_name} 必须是路径字符串")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} 不能为空字符串")
    expanded = Path(os.path.expandvars(os.path.expanduser(text)))
    if not expanded.is_absolute():
        expanded = (base_dir / expanded).resolve()
    return str(expanded)


def _normalize_horizons(value: Any) -> tuple[int, ...]:
    if value is None:
        return (1, 5, 10, 20)
    horizons = []
    for raw in _as_list(value, field_name="run.horizons"):
        horizon = int(raw)
        if horizon <= 0:
            raise ValueError("run.horizons 必须全部为正整数")
        horizons.append(horizon)
    unique = tuple(sorted(set(horizons)))
    if not unique:
        raise ValueError("run.horizons 不能为空")
    return unique


def _slugify_run_name(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or None


def load_config(path: str | Path) -> FactorEvaluationConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("配置文件顶层必须是 object")

    base_dir = config_path.resolve().parent
    meta_payload = _as_mapping(payload.get("meta"), field_name="meta")
    source_payload = _as_mapping(payload.get("source"), field_name="source")
    run_payload = _as_mapping(payload.get("run"), field_name="run")
    output_payload = _as_mapping(payload.get("output"), field_name="output")

    factor_id = _require_non_empty_text(meta_payload.get("factor_id"), field_name="meta.factor_id")
    horizons = _normalize_horizons(run_payload.get("horizons"))
    primary_horizon = meta_payload.get("primary_horizon")
    if primary_horizon is not None:
        primary_horizon = int(primary_horizon)
        if primary_horizon not in horizons:
            raise ValueError("meta.primary_horizon 必须包含在 run.horizons 中")

    meta = MetaConfig(
        factor_id=factor_id,
        run_name=_slugify_run_name(_normalize_scalar_text(meta_payload.get("run_name"))),
        primary_horizon=primary_horizon,
        description=_normalize_scalar_text(meta_payload.get("description")),
    )

    factor_lake_root = _resolve_path_text(
        source_payload.get("factor_lake_root"),
        field_name="source.factor_lake_root",
        base_dir=base_dir,
    ) or str(default_factor_lake_root())
    market_data_path = _resolve_path_text(
        source_payload.get("market_data_path"),
        field_name="source.market_data_path",
        base_dir=base_dir,
    )
    if market_data_path is None:
        raise ValueError("source.market_data_path 不能为空")

    source = SourceConfig(
        factor_lake_root=factor_lake_root,
        market_data_path=market_data_path,
        market_timestamp_col=str(source_payload.get("market_timestamp_col") or "timestamp"),
        market_symbol_col=str(source_payload.get("market_symbol_col") or "symbol"),
        market_price_col=str(source_payload.get("market_price_col") or "open"),
        universe_path=_resolve_path_text(
            source_payload.get("universe_path"),
            field_name="source.universe_path",
            base_dir=base_dir,
        ),
        universe_timestamp_col=str(source_payload.get("universe_timestamp_col") or "timestamp"),
        universe_symbol_col=str(source_payload.get("universe_symbol_col") or "symbol"),
        universe_membership_col=_normalize_scalar_text(source_payload.get("universe_membership_col")),
        universe_include_values=tuple(
            _require_non_empty_text(v, field_name="source.universe_include_values")
            for v in _as_list(source_payload.get("universe_include_values"), field_name="source.universe_include_values")
        ),
    )

    run = RunConfig(
        start=_normalize_scalar_text(run_payload.get("start")),
        end=_normalize_scalar_text(run_payload.get("end")),
        horizons=horizons,
        n_quantiles=int(run_payload.get("n_quantiles", 10)),
        winsorize_lower=float(run_payload.get("winsorize_lower", 0.01)),
        winsorize_upper=float(run_payload.get("winsorize_upper", 0.99)),
        standardize=bool(run_payload.get("standardize", True)),
        min_assets_per_date=int(run_payload.get("min_assets_per_date", 20)),
        direction=float(run_payload.get("direction", 1.0)),
        annualization_factor=int(run_payload.get("annualization_factor", 252)),
    )
    if run.n_quantiles < 2:
        raise ValueError("run.n_quantiles 至少为 2")
    if run.min_assets_per_date < 2:
        raise ValueError("run.min_assets_per_date 至少为 2")
    if not 0.0 <= run.winsorize_lower < run.winsorize_upper <= 1.0:
        raise ValueError("winsorize 分位数配置非法")

    default_output_root = str(default_factor_evaluation_root(factor_lake_root=source.factor_lake_root))
    output = OutputConfig(
        root=_resolve_path_text(
            output_payload.get("root") or default_output_root,
            field_name="output.root",
            base_dir=base_dir,
        ),
        save_artifacts=bool(output_payload.get("save_artifacts", True)),
    )
    return FactorEvaluationConfig(meta=meta, source=source, run=run, output=output)