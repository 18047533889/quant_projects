from __future__ import annotations

from pathlib import Path

import pytest

from storage.factory import build_data_source
from workspace_paths import resolve_path


def test_build_data_source_expands_tilde_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    data_root = tmp_path / "quant_projects" / "data" / "kline"
    data_root.mkdir(parents=True)

    source = build_data_source(
        {
            "type": "parquet_kline",
            "root": "~/quant_projects/data/kline",
            "instrument_col": "ticker",
            "timestamp_col": "window_start",
        }
    )
    assert Path(source.root).resolve() == data_root.resolve()


def test_resolve_path_relative_to_quant_projects(monkeypatch):
    from workspace_paths import quant_projects_root

    root = quant_projects_root()
    resolved = resolve_path("data/us_stock/massive_data/StockDailyBar")
    assert resolved == (root / "data/us_stock/massive_data/StockDailyBar").resolve()
