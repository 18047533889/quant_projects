"""GTJA-191 落值 YAML 与路径约定（factor_engine materialize）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from lib.data_source import ASHARE_PV_FIELDS, production_date_range
from lib.paths import PACKAGE_ROOT

AUTHOR = "gtja191"
UNIVERSE = "ASHARE_ALL"
FREQUENCY = "1d"
CONFIG_DIR = PACKAGE_ROOT / "examples" / "materialize"


def default_lake_root() -> Path:
    env = __import__("os").environ.get("GTJA191_LAKE_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (PACKAGE_ROOT.parent / "data" / "factors" / "lake" / "gtja191").resolve()


def default_plan_cache_dir() -> Path:
    env = __import__("os").environ.get("GTJA191_PLAN_CACHE_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (PACKAGE_ROOT.parent / "data" / "factors" / "plan_cache" / "gtja191").resolve()


def build_materialize_config(
    factor_name: str,
    dsl_formula: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    lake_root: str | Path | None = None,
    factor_id: str | None = None,
    write_target: str = "local",
) -> dict[str, Any]:
    """单因子 factor_engine 落值 YAML 配置体。"""
    start, end = production_date_range(start_date=start_date, end_date=end_date)
    lake = str(lake_root or default_lake_root())
    fid = factor_id or factor_name
    data_source: dict[str, Any] = {
        "type": "data_access",
        "dataset": "ashare_stock_daily",
        "read_auto": True,
        "start_date": start,
        "end_date": end,
        "fields": dict(ASHARE_PV_FIELDS),
    }
    return {
        "factor": {
            "name": factor_name,
            "expr": dsl_formula,
            "freq": FREQUENCY,
            "universe": UNIVERSE,
            "description": f"GTJA-191 {factor_name}",
        },
        "data_source": data_source,
        "backend": {"type": "pandas"},
        "engine": {
            "enable_cache": True,
            "plan_cache_dir": str(default_plan_cache_dir()),
        },
        "materialization": {
            "lake_root": lake,
            "factor_id": fid,
            "author": AUTHOR,
            "frequency": FREQUENCY,
            "expression": dsl_formula,
            "target": write_target,
            "preserve_invalid_rows": True,
            "value_dtype": "float32",
        },
    }


def materialize_config_path(factor_name: str) -> Path:
    return CONFIG_DIR / f"{factor_name}.yaml"
