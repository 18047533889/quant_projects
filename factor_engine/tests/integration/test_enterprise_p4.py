# -*- coding: utf-8
"""run_many_from_config 子分组 / write_factor_series / 公开 API 测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from runtime.config_runtime import config_run_batch_key
from runtime.engine import FactorEngine


def _write_factor_yaml(
    tmp_path: Path,
    name: str,
    expr: str,
    root: Path,
    *,
    trim_warmup: bool | None = None,
) -> Path:
    run_block = ""
    if trim_warmup is not None:
        run_block = f"""
run:
  trim_warmup: {"true" if trim_warmup else "false"}
"""
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        f"""
factor:
  name: {name}
  expr: {expr}
{run_block}
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


def test_config_run_batch_key_distinguishes_trim_warmup(tmp_path):
    root = tmp_path / "data"
    _seed_parquet(root)
    cfg_a = _write_factor_yaml(tmp_path, "a", "close", root, trim_warmup=True)
    cfg_b = _write_factor_yaml(tmp_path, "b", "close", root, trim_warmup=False)
    _, _, config_a = FactorEngine.from_config(cfg_a)
    _, _, config_b = FactorEngine.from_config(cfg_b)
    assert config_run_batch_key(config_a) != config_run_batch_key(config_b)


def test_run_many_from_config_subgroups_by_run_flags(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _seed_parquet(root)
    cfg_a = _write_factor_yaml(tmp_path, "fa", "close", root, trim_warmup=True)
    cfg_b = _write_factor_yaml(tmp_path, "fb", "close * 2", root, trim_warmup=True)
    cfg_c = _write_factor_yaml(tmp_path, "fc", "close + 1", root, trim_warmup=False)

    batch_calls: list[list[str]] = []

    def _fake_run_many(self, factors, **kwargs):
        names = [f.name for f in factors]
        batch_calls.append(names)
        return {
            "results": {n: pd.Series([1.0], name=n) for n in names},
            "analyses": {n: {"ir": None} for n in names},
        }

    single_calls: list[str] = []

    def _fake_run(self, factor, **kwargs):
        single_calls.append(factor.name)
        return {
            "factor": factor,
            "analysis": {"ir": None},
            "result": pd.Series([1.0], name=factor.name),
        }

    monkeypatch.setattr(FactorEngine, "run_many", _fake_run_many)
    monkeypatch.setattr(FactorEngine, "run", _fake_run)

    out = FactorEngine.run_many_from_config([cfg_a, cfg_b, cfg_c])
    assert set(out["results"]) == {"fa", "fb", "fc"}
    assert batch_calls == [["fa", "fb"]]
    assert single_calls == ["fc"]


def test_clickhouse_write_target_write_factor_series(monkeypatch):
    from storage.write_targets import ClickHouseWriteTarget

    captured: dict = {}

    class FakeSummary:
        factor_id = "f1"
        rows_written = 2
        table = "factor_values"
        database = "default"
        dq_report = {"passed": True}

    class FakeCHMat:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def materialize(self, **kwargs):
            captured["materialize"] = kwargs
            return FakeSummary()

    monkeypatch.setattr(
        "storage.clickhouse_materializer.ClickHouseMaterializer",
        FakeCHMat,
    )
    target = ClickHouseWriteTarget(table="fv", host="localhost")
    series = pd.Series(
        [1.0, 2.0],
        index=pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-01"), "A"),
                (pd.Timestamp("2024-01-02"), "A"),
            ],
            names=["datetime", "asset"],
        ),
    )
    out = target.write_factor_series("f1", series, factor_version="v1")
    assert out["rows_written"] == 2
    assert captured["materialize"]["factor_id"] == "f1"


def test_storage_exports_write_targets():
    from storage import (
        ClickHouseWriteTarget,
        LocalParquetWriteTarget,
        StagingWriteTarget,
        resolve_write_target,
    )

    assert resolve_write_target("local").name == "local"
    assert isinstance(resolve_write_target("staging"), StagingWriteTarget)
    assert isinstance(resolve_write_target("clickhouse"), ClickHouseWriteTarget)
