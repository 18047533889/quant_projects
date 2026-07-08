# -*- coding: utf-8
"""示例 YAML 已全部迁移至 data_access；本模块校验 factory 可解析且无 legacy type。"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from storage.factory import build_data_source

EXAMPLES_ROOT = Path(__file__).resolve().parent.parent / "examples"
LEGACY_TYPES = frozenset({"cleaned_parquet", "multi_parquet", "parquet_kline"})


def _iter_example_yaml_paths() -> list[Path]:
    paths: list[Path] = []
    for pattern in ("configs/**/*.yaml", "profiles/*.yaml", "*.yaml"):
        paths.extend(EXAMPLES_ROOT.glob(pattern))
    return sorted({p for p in paths if p.is_file()})


def _collect_source_types(node: object) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        if "type" in node:
            found.add(str(node["type"]).lower())
        for value in node.values():
            found |= _collect_source_types(value)
    elif isinstance(node, list):
        for item in node:
            found |= _collect_source_types(item)
    return found


@pytest.mark.parametrize("yaml_path", _iter_example_yaml_paths(), ids=lambda p: p.name)
def test_example_yaml_has_no_legacy_data_source_type(yaml_path: Path):
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    ds = raw.get("data_source") if isinstance(raw, dict) else None
    if not isinstance(ds, dict):
        pytest.skip("no data_source block")
    types = _collect_source_types(ds)
    legacy = types & LEGACY_TYPES
    assert not legacy, f"{yaml_path}: legacy types {legacy}"


@pytest.mark.parametrize(
    "yaml_path",
    [
        p
        for p in _iter_example_yaml_paths()
        if p.name
        in {
            "us_stocks_sip_quotes_v1.yaml",
            "us_stocks_sip_trades_v1.yaml",
            "config_driven_factor.yaml",
            "notebook_config_smoke.yaml",
            "day_aggs_v1_fundamental_cash_reinvestment_rank.yaml",
        }
    ],
    ids=lambda p: p.name,
)
def test_example_yaml_builds_data_source(yaml_path: Path):
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    ds = raw["data_source"]
    source = build_data_source(ds)
    assert source is not None
