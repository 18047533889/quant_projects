# -*- coding: utf-8
"""examples/profiles 因子模板可加载性 + data_source factory 构建。"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from factor_engine.runtime.config import load_config
from factor_engine.storage.factory import build_data_source

PROFILES_ROOT = Path(__file__).resolve().parent.parent / "examples" / "profiles"


def _factor_profiles() -> list[Path]:
    paths: list[Path] = []
    for path in sorted(PROFILES_ROOT.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and raw.get("factor"):
            paths.append(path)
    return paths


@pytest.mark.parametrize("profile_path", _factor_profiles(), ids=lambda p: p.name)
def test_factor_profile_loads(profile_path: Path):
    loaded = load_config(profile_path)
    assert loaded.factor.name
    assert loaded.factor.expr
    assert loaded.data_source.type in {
        "data_access",
        "composite",
        "clickhouse",
    }


@pytest.mark.parametrize("profile_path", _factor_profiles(), ids=lambda p: p.name)
def test_factor_profile_builds_data_source(profile_path: Path):
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    source = build_data_source(raw["data_source"])
    assert source is not None
