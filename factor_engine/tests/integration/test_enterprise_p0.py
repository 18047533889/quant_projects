# -*- coding: utf-8
"""企业级 P0/P1 改造单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.config import load_config
from runtime.config_runtime import resolve_run_kwargs
from storage.factor_schema import FACTOR_LAKE_DATASET_SCHEMA, FACTOR_VALUE_COLUMNS


def test_production_auto_warmup_default_without_yaml_flag(tmp_path):
    config = tmp_path / "factor.yaml"
    config.write_text(
        """
run:
  mode: production
dq:
  strict: true
factor:
  name: x
  expr: rank(close)
data_source:
  type: parquet
  root: /tmp
""".strip()
        + "\n",
        encoding="utf-8",
    )
    loaded = load_config(config)
    opts = resolve_run_kwargs(loaded)
    assert opts.auto_warmup is True


def test_factor_schema_matches_materializer_metadata():
    from storage.materializer import METADATA_COLUMNS

    assert set(FACTOR_VALUE_COLUMNS) == {"datetime", "asset", "value", *METADATA_COLUMNS}
    assert set(FACTOR_LAKE_DATASET_SCHEMA) >= {"datetime", "asset", "value"}
