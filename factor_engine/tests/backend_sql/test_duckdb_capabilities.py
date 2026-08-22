# -*- coding: utf-8
"""DuckDB 能力探测。"""
from __future__ import annotations

import pytest

pytest.importorskip("duckdb")

from backend.sql_pushdown.duckdb_capabilities import probe_duckdb_capabilities


def test_duckdb_capability_probe_runs():
    report = probe_duckdb_capabilities()
    assert report.version
    assert "corr_window" in report.features
