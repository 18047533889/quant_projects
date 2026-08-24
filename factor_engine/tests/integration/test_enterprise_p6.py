# -*- coding: utf-8
"""Phase 18：PipelineConfigOverrides / 并行物化 batch / 增量批量 / batch key 分组。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from factor_engine.runtime.config_runtime import (
    PipelineConfigOverrides,
    config_materialize_batch_key,
    config_run_batch_key,
    resolve_materialize_kwargs_for_pipeline,
)
from factor_engine.runtime.engine import FactorEngine


def _write_factor_yaml(tmp_path: Path, name: str, expr: str, root: Path) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        f"""
factor:
  name: {name}
  expr: {expr}
data_source:
  type: parquet_kline
  root: {root}
  timestamp_column: datetime
  instrument_column: asset
  fields:
    close: close
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _seed_parquet(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "asset": ["A", "A"],
            "close": [1.0, 2.0],
        }
    ).to_parquet(root / "data.parquet")


def test_pipeline_overrides_affect_batch_keys(tmp_path):
    root = tmp_path / "data"
    _seed_parquet(root)
    cfg_path = _write_factor_yaml(tmp_path, "bk", "close", root)
    _, _, config = FactorEngine.from_config(cfg_path)

    base_run = config_run_batch_key(config)
    with_dq = config_run_batch_key(
        config,
        pipeline=PipelineConfigOverrides(input_dq_check=True),
    )
    assert base_run != with_dq

    base_mat = config_materialize_batch_key(config)
    with_target = config_materialize_batch_key(
        config,
        pipeline=PipelineConfigOverrides(write_target="staging"),
    )
    assert base_mat != with_target


def test_resolve_materialize_kwargs_for_pipeline_applies_write_target(tmp_path):
    root = tmp_path / "data"
    _seed_parquet(root)
    cfg_path = _write_factor_yaml(tmp_path, "wt", "close", root)
    _, _, config = FactorEngine.from_config(cfg_path)
    opts = resolve_materialize_kwargs_for_pipeline(
        config,
        PipelineConfigOverrides(write_target="staging_clickhouse"),
    )
    assert opts.write_target == "staging_clickhouse"


def test_materialize_many_from_config_parallel_uses_run_many_parallel(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _seed_parquet(root)
    cfg_a = _write_factor_yaml(tmp_path, "mpa", "close", root)
    cfg_b = _write_factor_yaml(tmp_path, "mpb", "close * 2", root)
    seen: dict[str, bool] = {"parallel": False}

    def _fake_materialize_many_from_config(*args, **kwargs):
        seen["parallel"] = kwargs.get("parallel") is True
        return {"materializations": {}}

    monkeypatch.setattr(
        FactorEngine,
        "materialize_many_from_config",
        staticmethod(_fake_materialize_many_from_config),
    )
    FactorEngine.materialize_many_from_config_parallel([cfg_a, cfg_b], n_jobs=2)
    assert seen["parallel"] is True


def test_materialize_incremental_many_from_config(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _seed_parquet(root)
    cfg_a = _write_factor_yaml(tmp_path, "ia", "close", root)
    cfg_b = _write_factor_yaml(tmp_path, "ib", "close * 2", root)
    calls: list[str] = []

    def _fake_materialize_incremental(self, factor, **kwargs):
        calls.append(factor.name)
        return {"incremental": {"factor_id": factor.name}, "result": pd.Series([1.0])}

    monkeypatch.setattr(FactorEngine, "materialize_incremental", _fake_materialize_incremental)
    out = FactorEngine.materialize_incremental_many_from_config([cfg_a, cfg_b])
    assert sorted(calls) == ["ia", "ib"]
    assert set(out["materializations"]) == {"ia", "ib"}


def test_run_many_from_config_passes_pipeline_overrides(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _seed_parquet(root)
    cfg_a = _write_factor_yaml(tmp_path, "oa", "close", root)
    cfg_b = _write_factor_yaml(tmp_path, "ob", "close", root)
    captured: dict[str, object] = {}

    def _fake_group(cls, group, **kwargs):
        captured["pipeline_overrides"] = kwargs.get("pipeline_overrides")
        return None

    monkeypatch.setattr(FactorEngine, "_run_many_batch_from_config_group", classmethod(_fake_group))
    overrides = PipelineConfigOverrides(input_dq_check=True)
    FactorEngine.run_many_from_config([cfg_a, cfg_b], pipeline_overrides=overrides)
    assert captured["pipeline_overrides"] == overrides
