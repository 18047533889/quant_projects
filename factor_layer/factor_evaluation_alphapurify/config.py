from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]

DEFAULT_TARGET_FACTORS: tuple[str, ...] = (
    "day_aggs_v1_fundamental_asset_scale_rank_2016_2025_v1",
    "day_aggs_v1_fundamental_book_strength_rank_2016_2025_v1",
    "day_aggs_v1_fundamental_cash_reinvestment_rank_2016_2025_v1",
)

DEFAULT_EXPOSURE_FACTORS: tuple[str, ...] = (
    "day_aggs_v1_fundamental_float_tightness_rank_2016_2025_v1",
    "day_aggs_v1_fundamental_operating_cashflow_rank_2016_2025_v1",
    "day_aggs_v1_fundamental_operating_margin_proxy_rank_2016_2025_v1",
)


@dataclass(frozen=True)
class AlphaPurifyAdapterConfig:
    data_root: str = str(REPO_ROOT / "0431-测试数据")
    output_root: str = str(REPO_ROOT / "workspace_data" / "alphapurify" / "outputs_exposures_factoranalyzer")
    year: int | None = None
    min_symbols_per_day: int = 10
    target_factors: tuple[str, ...] = DEFAULT_TARGET_FACTORS
    exposure_factors: tuple[str, ...] = DEFAULT_EXPOSURE_FACTORS
    save_config_snapshot: bool = True
    database_module: str | None = None
    exposures_module: str | None = None
    factor_analyzer_module: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("配置文件顶层必须是 object")
    return dict(value)


def _resolve_path(value: Any, *, field_name: str, base_dir: Path) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{field_name} 必须是路径字符串")
    raw = str(value).strip()
    if not raw:
        raise ValueError(f"{field_name} 不能为空")
    path = Path(os.path.expandvars(os.path.expanduser(raw)))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _normalize_factor_names(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        raise ValueError(f"{field_name} 不能为空")
    if not isinstance(value, list):
        raise ValueError(f"{field_name} 必须是字符串数组")
    items: list[str] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{field_name}[{index}] 必须是非空字符串")
        items.append(raw.strip())
    if not items:
        raise ValueError(f"{field_name} 不能为空")
    return tuple(items)


def _normalize_module_name(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} 必须是字符串或 null")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空字符串")
    return normalized


def _validate(config: AlphaPurifyAdapterConfig) -> None:
    data_root = Path(config.data_root)
    if not data_root.exists() or not data_root.is_dir():
        raise ValueError(f"data_root 不存在或不是目录: {config.data_root}")
    if config.min_symbols_per_day <= 0:
        raise ValueError("min_symbols_per_day 必须大于 0")
    if config.year is not None and config.year <= 0:
        raise ValueError("year 必须是正整数或 null")
    if not config.target_factors:
        raise ValueError("target_factors 不能为空")
    if not config.exposure_factors:
        raise ValueError("exposure_factors 不能为空")


def load_config(path: str | Path) -> AlphaPurifyAdapterConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    payload = _as_mapping(payload)
    base_dir = config_path.resolve().parent

    target_factors = payload.get("target_factors")
    exposure_factors = payload.get("exposure_factors")

    config = AlphaPurifyAdapterConfig(
        data_root=_resolve_path(payload.get("data_root"), field_name="data_root", base_dir=base_dir)
        or str(REPO_ROOT / "0431-测试数据"),
        output_root=_resolve_path(payload.get("output_root"), field_name="output_root", base_dir=base_dir)
        or str(REPO_ROOT / "workspace_data" / "alphapurify" / "outputs_exposures_factoranalyzer"),
        year=int(payload["year"]) if payload.get("year") is not None else None,
        min_symbols_per_day=int(payload.get("min_symbols_per_day", 10)),
        target_factors=_normalize_factor_names(target_factors, field_name="target_factors")
        if target_factors is not None
        else DEFAULT_TARGET_FACTORS,
        exposure_factors=_normalize_factor_names(exposure_factors, field_name="exposure_factors")
        if exposure_factors is not None
        else DEFAULT_EXPOSURE_FACTORS,
        save_config_snapshot=bool(payload.get("save_config_snapshot", True)),
        database_module=_normalize_module_name(payload.get("database_module"), field_name="database_module"),
        exposures_module=_normalize_module_name(payload.get("exposures_module"), field_name="exposures_module"),
        factor_analyzer_module=_normalize_module_name(
            payload.get("factor_analyzer_module"), field_name="factor_analyzer_module"
        ),
    )
    _validate(config)
    return config
