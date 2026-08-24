from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_canonical_fields import get_table_meta
from factor_engine.util.workspace_paths import quant_projects_root, resolve_path, repo_root


def test_get_table_meta_stock_daily_bar_per_market():
    us = get_table_meta("StockDailyBar", "us_stock")
    ash = get_table_meta("StockDailyBar", "ashare")
    assert us["market"] == "us_stock"
    assert ash["market"] == "ashare"
    assert us["domain_root"] == ash["domain_root"] == "price_volume"


def test_resolve_path_expands_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    p = tmp_path / "quant_projects" / "data" / "x"
    p.mkdir(parents=True)
    assert resolve_path("~/quant_projects/data/x") == p.resolve()


def test_quant_projects_root_default():
    assert quant_projects_root() == repo_root().parent
