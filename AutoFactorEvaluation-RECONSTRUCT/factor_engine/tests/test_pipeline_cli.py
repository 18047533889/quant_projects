from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
yaml = pytest.importorskip("yaml")


def _write_kline_day(root: Path, day: str, rows: list[tuple[str, float]]) -> None:
    frame = pd.DataFrame(
        {
            "ticker": [ticker for ticker, _ in rows],
            "window_start": [pd.Timestamp(day, tz="UTC").value for _ in rows],
            "close": [close for _, close in rows],
            "open": [close for _, close in rows],
            "high": [close + 1 for _, close in rows],
            "low": [close - 1 for _, close in rows],
            "volume": [1000.0 for _ in rows],
            "transactions": [10.0 for _ in rows],
        }
    )
    target = root / pd.Timestamp(day).strftime("%Y") / pd.Timestamp(day).strftime("%m")
    target.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target / f"{day}.parquet", index=False)


def _write_config(path: Path, data_root: Path, factor_name: str, expr: str) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "factor": {
                    "name": factor_name,
                    "expr": expr,
                },
                "data_source": {
                    "type": "parquet_kline",
                    "root": str(data_root),
                    "instrument_column": "ticker",
                    "timestamp_column": "window_start",
                    "fields": {"close": "close"},
                },
                "backend": {"type": "pandas"},
                "engine": {"enable_cache": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _run_cli(*args: str) -> dict:
    script_path = Path(__file__).resolve().parents[1] / "run_pipeline.py"
    completed = subprocess.run(
        [sys.executable, str(script_path), *args],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(script_path.parent),
    )
    return json.loads(completed.stdout)


def test_pipeline_cli_runs_single_config(tmp_path: Path):
    data_root = tmp_path / "day_aggs_v1"
    _write_kline_day(data_root, "2024-01-01", [("AAA", 10.0), ("BBB", 20.0)])
    _write_kline_day(data_root, "2024-01-02", [("AAA", 11.0), ("BBB", 19.0)])
    _write_kline_day(data_root, "2024-01-03", [("AAA", 12.0), ("BBB", 18.0)])

    config_path = tmp_path / "mom_rank.yaml"
    _write_config(config_path, data_root, "mom_rank", 'rank(ts_mean(col("close"), 2))')

    output_root = tmp_path / "cli_output"
    payload = _run_cli(
        "config",
        str(config_path),
        "--output-root",
        str(output_root),
        "--log-level",
        "WARNING",
    )

    assert payload["mode"] == "config"
    assert payload["summary"]["configs_total"] == 1
    assert payload["summary"]["configs_success"] == 1
    assert (output_root / "run_summary.json").exists()
    assert (output_root / "config_snapshot.yaml").exists()


def test_pipeline_cli_runs_config_directory(tmp_path: Path):
    data_root = tmp_path / "day_aggs_v1"
    _write_kline_day(data_root, "2024-01-01", [("AAA", 10.0), ("BBB", 20.0)])
    _write_kline_day(data_root, "2024-01-02", [("AAA", 11.0), ("BBB", 19.0)])
    _write_kline_day(data_root, "2024-01-03", [("AAA", 12.0), ("BBB", 18.0)])

    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_config(config_dir / "mom_rank.yaml", data_root, "mom_rank", 'rank(ts_mean(col("close"), 2))')
    _write_config(config_dir / "delay_close.yaml", data_root, "delay_close", 'ts_delay(col("close"), 1)')

    output_root = tmp_path / "cli_batch_output"
    payload = _run_cli(
        "config-dir",
        str(config_dir),
        "--output-root",
        str(output_root),
        "--log-level",
        "WARNING",
    )

    assert payload["mode"] == "config-dir"
    assert payload["summary"]["configs_total"] == 2
    assert payload["summary"]["configs_success"] == 2
    assert (output_root / "run_summary.json").exists()
    assert (output_root / "config_snapshots" / "mom_rank.yaml").exists()
    assert (output_root / "config_snapshots" / "delay_close.yaml").exists()