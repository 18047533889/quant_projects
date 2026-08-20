# -*- coding: utf-8
"""Phase 16：物化 batch run_many / ResolvedMaterializeKwargs / parallel from_config。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from runtime.config_runtime import config_materialize_batch_key, resolve_materialize_kwargs
from runtime.engine import FactorEngine


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


def test_resolved_materialize_kwargs_includes_resume_and_isolate(tmp_path):
    root = tmp_path / "data"
    _seed_parquet(root)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"""
factor:
  name: f1
  expr: close
data_source:
  type: parquet_kline
  root: {root}
  timestamp_column: datetime
  instrument_column: asset
  fields:
    close: close
materialization:
  resume_materialize: true
  isolate_partition_failures: false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _, _, config = FactorEngine.from_config(cfg)
    opts = resolve_materialize_kwargs(config)
    kw = opts.to_engine_materialize_kwargs()
    assert kw["resume_materialize"] is True
    assert kw["isolate_partition_failures"] is False


def test_to_incremental_materialize_kwargs_includes_since(tmp_path):
    root = tmp_path / "data"
    _seed_parquet(root)
    cfg = tmp_path / "inc.yaml"
    cfg.write_text(
        f"""
factor:
  name: inc1
  expr: close
data_source:
  type: parquet_kline
  root: {root}
  timestamp_column: datetime
  instrument_column: asset
  fields:
    close: close
materialization:
  incremental:
    since: "2024-01-01"
    lookback_extra: 3
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _, _, config = FactorEngine.from_config(cfg)
    opts = resolve_materialize_kwargs(config)
    kw = opts.to_incremental_materialize_kwargs()
    assert kw["since"] == "2024-01-01"
    assert kw["lookback_extra"] == 3
    assert kw["resume_materialize"] is False


def test_materialize_many_from_config_batches_run_many(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _seed_parquet(root)
    cfg_a = _write_factor_yaml(tmp_path, "ma", "close", root)
    cfg_b = _write_factor_yaml(tmp_path, "mb", "close * 2", root)

    run_many_calls: list[list[str]] = []
    materialize_calls: list[str] = []
    execute_calls: list[str] = []

    def _fake_run_many(self, factors, **kwargs):
        names = [f.name for f in factors]
        run_many_calls.append(names)
        return {
            "results": {n: pd.Series([1.0], name=n) for n in names},
            "analyses": {n: type("A", (), {"ir": None, "lookback": 0})() for n in names},
        }

    def _fake_materialize(self, factor, **kwargs):
        materialize_calls.append(factor.name)
        return {"materialization": {"factor_id": factor.name, "rows_written": 1}}

    def _fake_execute(engine, factor, output, opts, **kw):
        execute_calls.append(factor.name)
        return {
            "materialization": {"factor_id": factor.name, "rows_written": 1},
            "factor": factor,
            "result": output["result"],
        }

    monkeypatch.setattr(FactorEngine, "run_many", _fake_run_many)
    monkeypatch.setattr(FactorEngine, "materialize", _fake_materialize)
    monkeypatch.setattr(
        "runtime.materialize_service.execute_materialize_from_resolved",
        _fake_execute,
    )

    out = FactorEngine.materialize_many_from_config([cfg_a, cfg_b], batch_run=True)
    assert run_many_calls == [["ma", "mb"]]
    assert materialize_calls == []
    assert execute_calls == ["ma", "mb"]
    assert out["materializations"]["ma"].get("batched_run") is True


def test_run_many_from_config_parallel_delegates(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _seed_parquet(root)
    cfg_a = _write_factor_yaml(tmp_path, "pa", "close", root)
    cfg_b = _write_factor_yaml(tmp_path, "pb", "close", root)
    seen: dict[str, bool] = {"parallel": False}

    def _fake_run_many_from_config(*args, **kwargs):
        seen["parallel"] = kwargs.get("parallel") is True
        return {"results": {}, "runs": {}, "configs": {}}

    monkeypatch.setattr(FactorEngine, "run_many_from_config", _fake_run_many_from_config)
    FactorEngine.run_many_from_config_parallel([cfg_a, cfg_b], n_jobs=2)
    assert seen["parallel"] is True
