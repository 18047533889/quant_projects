"""quality/contracts.py + quality/cli.py 单测。"""
from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data_access.quality.contracts import QualityOptions, run_quality_checks
from data_access.quality.cli import main


def _bad_table() -> pa.Table:
    return pa.table({
        "datetime": ["2024-01-02", "2024-01-02", "2024-01-03"],
        "symbol": ["AAPL", "AAPL", "MSFT"],
        "close": [150.5, 380.1, float("nan")],
        "report_period": ["2024-01-02", "2024-01-02", "2024-01-03"],
        "publish_time": ["2024-01-01", "2024-01-01", "2024-01-04"],
    })


def test_quality_detects_issues():
    report = run_quality_checks(
        "test",
        _bad_table(),
        options=QualityOptions(
            required_columns=("datetime", "symbol", "close"),
            primary_key=("datetime", "symbol"),
            null_ratio_max=0.1,
            finite_columns=("close",),
            time_column="datetime",
            instrument_column="symbol",
            check_duplicate_timestamp=True,
            check_pit_leakage=True,
        ),
    )
    assert not report.passed
    text = " | ".join(report.failures)
    assert "duplicate primary keys" in text
    assert "non-finite" in text
    assert "PIT leakage" in text
    assert "duplicate (time, instrument)" in text


def test_quality_clean_passes():
    report = run_quality_checks(
        "test",
        pa.table({"a": [1.0, 2.0], "b": ["x", "y"]}),
        options=QualityOptions(finite_columns=("a",), min_rows=2),
    )
    assert report.passed


def test_quality_range_and_null():
    report = run_quality_checks(
        "test",
        pa.table({"close": [10.0, 500.0], "note": [None, "x"]}),
        options=QualityOptions(
            range={"close": (0.0, 100.0)},
            null_ratio_max=0.5,
        ),
    )
    assert not report.passed
    text = " | ".join(report.failures)
    assert "range violations" in text


def test_cli_reads_via_read_uri(tmp_path, monkeypatch):
    pq.write_table(_bad_table(), str(tmp_path / "data.parquet"))
    monkeypatch.setenv("DATA_ACCESS_READ_URI_ROOTS", str(tmp_path))
    from data_access import reset_store

    reset_store()
    rc = main([
        str(tmp_path / "data.parquet"),
        "--time-column", "datetime",
        "--instrument-column", "symbol",
        "--primary-key", "datetime",
        "--primary-key", "symbol",
        "--finite", "close",
        "--check", "duplicate_timestamp,pit_leakage",
    ])
    assert rc == 1


def test_cli_clean_passes(tmp_path, monkeypatch):
    pq.write_table(
        pa.table({"datetime": ["2024-01-02"], "symbol": ["AAPL"], "close": [150.5]}),
        str(tmp_path / "ok.parquet"),
    )
    monkeypatch.setenv("DATA_ACCESS_READ_URI_ROOTS", str(tmp_path))
    from data_access import reset_store

    reset_store()
    rc = main([
        str(tmp_path / "ok.parquet"),
        "--time-column", "datetime",
        "--instrument-column", "symbol",
        "--finite", "close",
    ])
    assert rc == 0
