# -*- coding: utf-8
"""pipeline 目录 batch 与 kwargs 映射测试。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml

from factor_engine.pipeline import (
    _directory_batch_eligible,
    _directory_incremental_batch_eligible,
    run_config_directory,
)
from factor_engine.runtime.engine import FactorEngine


def _write_kline_config(config_dir: Path, data_root: Path, name: str, expr: str) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "factor": {"name": name, "expr": expr},
                "data_source": {
                    "type": "parquet_kline",
                    "root": str(data_root),
                    "instrument_column": "asset",
                    "timestamp_column": "datetime",
                    "fields": {"close": "close"},
                },
                "backend": {"type": "pandas"},
                "engine": {"enable_cache": False},
            }
        ),
        encoding="utf-8",
    )
    return path


def _seed_data(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "asset": ["A", "A"],
            "close": [1.0, 2.0],
        }
    ).to_parquet(root / "data.parquet")


def test_directory_batch_eligible_requires_two_configs(tmp_path):
    root = tmp_path / "data"
    _seed_data(root)
    cfg_dir = tmp_path / "cfgs"
    p1 = _write_kline_config(cfg_dir, root, "a", "close")
    assert not _directory_batch_eligible(
        [p1],
        profile=None,
        materialize=False,
        incremental=False,
        n_jobs=1,
        max_retries=0,
    )
    p2 = _write_kline_config(cfg_dir, root, "b", "close * 2")
    assert _directory_batch_eligible(
        [p1, p2],
        profile=None,
        materialize=False,
        incremental=False,
        n_jobs=1,
        max_retries=0,
    )


def test_directory_batch_eligible_with_cli_dq_override(tmp_path):
    root = tmp_path / "data"
    _seed_data(root)
    cfg_dir = tmp_path / "cfgs"
    p1 = _write_kline_config(cfg_dir, root, "a", "close")
    p2 = _write_kline_config(cfg_dir, root, "b", "close * 2")
    assert _directory_batch_eligible(
        [p1, p2],
        profile=None,
        materialize=False,
        incremental=False,
        n_jobs=1,
        max_retries=0,
    )


def test_directory_incremental_batch_eligible(tmp_path):
    root = tmp_path / "data"
    _seed_data(root)
    cfg_dir = tmp_path / "cfgs"
    p1 = _write_kline_config(cfg_dir, root, "a", "close")
    p2 = _write_kline_config(cfg_dir, root, "b", "close * 2")
    assert not _directory_incremental_batch_eligible(
        [p1],
        incremental=True,
        n_jobs=1,
        max_retries=0,
    )
    assert _directory_incremental_batch_eligible(
        [p1, p2],
        incremental=True,
        n_jobs=1,
        max_retries=0,
    )


def test_run_config_directory_uses_engine_batch(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _seed_data(root)
    cfg_dir = tmp_path / "cfgs"
    _write_kline_config(cfg_dir, root, "fa", "close")
    _write_kline_config(cfg_dir, root, "fb", "close * 2")

    called = {"run_many": 0, "single": 0}

    def _fake_run_many_from_config(cls, paths, **kwargs):
        called["run_many"] += 1
        runs = {}
        for path in paths:
            cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            name = cfg["factor"]["name"]
            runs[name] = {
                "factor": MagicMock(name=name),
                "analysis": MagicMock(lookback=0, has_ts_op=False, has_cs_op=False, referenced_columns=set(), ir=MagicMock(op="x")),
                "plan": MagicMock(op="root", inputs=[]),
                "result": pd.Series([1.0], name=name),
            }
        return {"runs": runs, "results": {}, "configs": {}}

    def _fake_single(*args, **kwargs):
        called["single"] += 1
        return "x", {"status": "success", "config_name": "x", "factor_name": "x", "mode": "run"}

    monkeypatch.setattr(FactorEngine, "run_many_from_config", classmethod(_fake_run_many_from_config))
    monkeypatch.setattr("factor_engine.pipeline._run_single_config_file", _fake_single)

    out = run_config_directory(cfg_dir, output_root=tmp_path / "out", materialize=False, n_jobs=1)
    assert called["run_many"] == 1
    assert called["single"] == 0
    assert out["summary"]["configs_success"] == 2
    assert all(item.get("batched_engine") for item in out["results"])


def test_run_config_directory_batch_with_dq_check_override(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _seed_data(root)
    cfg_dir = tmp_path / "cfgs"
    _write_kline_config(cfg_dir, root, "fa", "close")
    _write_kline_config(cfg_dir, root, "fb", "close * 2")

    captured: dict[str, object] = {}

    def _fake_run_many_from_config(cls, paths, **kwargs):
        captured["pipeline_overrides"] = kwargs.get("pipeline_overrides")
        runs = {}
        for path in paths:
            cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            name = cfg["factor"]["name"]
            runs[name] = {
                "factor": MagicMock(name=name),
                "analysis": MagicMock(
                    lookback=0,
                    has_ts_op=False,
                    has_cs_op=False,
                    referenced_columns=set(),
                    ir=MagicMock(op="x"),
                ),
                "plan": MagicMock(op="root", inputs=[]),
                "result": pd.Series([1.0], name=name),
            }
        return {"runs": runs, "results": {}, "configs": {}}

    monkeypatch.setattr(FactorEngine, "run_many_from_config", classmethod(_fake_run_many_from_config))
    run_config_directory(
        cfg_dir,
        output_root=tmp_path / "out",
        materialize=False,
        n_jobs=1,
        input_dq_check=True,
    )
    overrides = captured.get("pipeline_overrides")
    assert overrides is not None
    assert overrides.input_dq_check is True
