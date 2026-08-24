from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
yaml = pytest.importorskip("yaml")

from factor_engine.pipeline import run_config_directory, run_from_config


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
    frame.to_parquet(root / f"{day}.parquet", index=False)


def _write_config(config_path: Path, data_root: Path, *, name: str, expr: str) -> None:
    config_path.write_text(
        yaml.safe_dump(
            {
                "factor": {
                    "name": name,
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
            }
        ),
        encoding="utf-8",
    )


def test_pipeline_run_from_config_writes_summary_and_result_json(tmp_path: Path):
    data_root = tmp_path / "day_aggs_v1" / "2024" / "01"
    data_root.mkdir(parents=True)
    _write_kline_day(data_root, "2024-01-01", [("AAA", 10.0), ("BBB", 20.0)])
    _write_kline_day(data_root, "2024-01-02", [("AAA", 11.0), ("BBB", 19.0)])
    _write_kline_day(data_root, "2024-01-03", [("AAA", 12.0), ("BBB", 18.0)])

    config_path = tmp_path / "single_factor.yaml"
    _write_config(config_path, tmp_path / "day_aggs_v1", name="mom_2_rank", expr='rank(ts_mean(col("close"), 2))')

    output_root = tmp_path / "pipeline_output"
    out = run_from_config(config_path, output_root=output_root)

    assert out["summary"]["configs_total"] == 1
    assert out["summary"]["configs_success"] == 1
    assert out["results"][0]["status"] == "success"
    assert out["results"][0]["analysis"]["lookback"] == 2
    assert out["results"][0]["result"]["row_count"] == 6
    assert (output_root / "run_summary.json").exists()
    assert (output_root / "config_snapshot.yaml").exists()
    assert (output_root / "results" / "single_factor.json").exists()


def test_pipeline_run_config_directory_writes_batch_summary(tmp_path: Path):
    data_root = tmp_path / "day_aggs_v1" / "2024" / "01"
    data_root.mkdir(parents=True)
    _write_kline_day(data_root, "2024-01-01", [("AAA", 10.0), ("BBB", 20.0)])
    _write_kline_day(data_root, "2024-01-02", [("AAA", 11.0), ("BBB", 19.0)])
    _write_kline_day(data_root, "2024-01-03", [("AAA", 12.0), ("BBB", 18.0)])

    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    _write_config(config_dir / "mom_rank.yaml", tmp_path / "day_aggs_v1", name="mom_rank", expr='rank(ts_mean(col("close"), 2))')
    _write_config(config_dir / "vol.yaml", tmp_path / "day_aggs_v1", name="vol_2", expr='ts_std_dev(col("close"), 2)')

    output_root = tmp_path / "batch_output"
    out = run_config_directory(config_dir, output_root=output_root)

    assert out["summary"]["configs_total"] == 2
    assert out["summary"]["configs_success"] == 2
    assert len(out["results"]) == 2
    assert (output_root / "run_summary.json").exists()
    assert (output_root / "config_snapshots" / "mom_rank.yaml").exists()
    assert (output_root / "config_snapshots" / "vol.yaml").exists()
    assert (output_root / "results" / "mom_rank.json").exists()
    assert (output_root / "results" / "vol.json").exists()