# -*- coding: utf-8 -*-
"""运行时环境引导测试。"""
from __future__ import annotations

import os
from pathlib import Path

from factor_engine.runtime.env_bootstrap import bootstrap_runtime_env
from workspace_paths import quant_projects_root


def test_bootstrap_loads_env_example_keys(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("CLICKHOUSE_HOST=testhost.example\n", encoding="utf-8")
    monkeypatch.delenv("CLICKHOUSE_HOST", raising=False)
    monkeypatch.setattr("workspace_paths.quant_projects_root", lambda: tmp_path)
    from workspace_paths import load_env_file

    load_env_file(path=env_file, override=True)
    assert os.environ.get("CLICKHOUSE_HOST") == "testhost.example"


def test_bootstrap_does_not_override_existing(monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_HOST", "keep")
    root = quant_projects_root()
    env_path = root / ".env"
    if env_path.is_file():
        from workspace_paths import load_env_file

        load_env_file(path=env_path)
    assert os.environ.get("CLICKHOUSE_HOST") == "keep"


def test_engine_calls_bootstrap(monkeypatch):
    called = {"n": 0}

    def _fake():
        called["n"] += 1

    monkeypatch.setattr("factor_engine.runtime.engine.bootstrap_runtime_env", _fake)
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.backend.pandas_backend import PandasBackend
    from tests.helpers import InMemorySeriesSource
    import pandas as pd

    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=2), ["A"]],
        names=["timestamp", "instrument"],
    )
    src = InMemorySeriesSource(data={"close": pd.Series([1.0, 2.0], index=idx)})
    FactorEngine(backend=PandasBackend(), data_source=src)
    assert called["n"] == 1
