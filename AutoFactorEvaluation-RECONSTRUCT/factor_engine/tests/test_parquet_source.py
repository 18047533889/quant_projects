from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from storage.parquet_source import ParquetSource


def _write_day_file(root: Path, day: str) -> None:
    day_ts = pd.Timestamp(day)
    target_dir = root / day_ts.strftime("%Y") / day_ts.strftime("%m")
    target_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "window_start": [pd.Timestamp(day, tz="UTC").value],
            "close": [1.0],
        }
    )
    frame.to_parquet(target_dir / f"{day}.parquet", index=False)


def test_selected_files_respects_requested_date_range(tmp_path: Path):
    root = tmp_path / "day_aggs_v1"
    for day in ["2015-12-31", "2016-01-04", "2025-12-31", "2026-01-02"]:
        _write_day_file(root, day)

    source = ParquetSource(
        root=root,
        timestamp_column="window_start",
        instrument_column="ticker",
        start_date="2016-01-01",
        end_date="2025-12-31 23:59:59",
    )

    selected = source._selected_files()

    assert [path.name for path in selected] == [
        "2016-01-04.parquet",
        "2025-12-31.parquet",
    ]


def test_load_column_falls_back_when_batch_read_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """PR1 之后：批量读失败应自动降级到逐文件读，保证单点坏 parquet 不影响整体。

    旧版本测试的是 `pd.read_parquet` 级的重试（已由 DuckDB 内部 IO 接管，
    且 `data_access/tests/unit/test_retry.py` 已覆盖 OSError 重试语义）。
    这里改为验证 ParquetSource 自身的 batch→per-file 容错路径。
    """
    root = tmp_path / "day_aggs_v1"
    _write_day_file(root, "2024-01-02")

    source = ParquetSource(
        root=root,
        timestamp_column="window_start",
        instrument_column="ticker",
    )

    call_count = {"batch": 0, "per_file": 0}
    original_batch = source._duckdb_read_batch
    original_per_file = source._duckdb_read_per_file

    def flaky_batch(*args, **kwargs):
        call_count["batch"] += 1
        raise RuntimeError("Repetition level histogram size mismatch")

    def counted_per_file(*args, **kwargs):
        call_count["per_file"] += 1
        return original_per_file(*args, **kwargs)

    monkeypatch.setattr(source, "_duckdb_read_batch", flaky_batch)
    monkeypatch.setattr(source, "_duckdb_read_per_file", counted_per_file)

    series = source.load_column("close")

    assert len(series) == 1
    assert call_count["batch"] == 1
    assert call_count["per_file"] == 1


def test_resolve_instrument_column_symbol_to_ticker(tmp_path: Path):
    root = tmp_path / "StockDailyBar"
    root.mkdir()
    frame = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "TradeDate": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "Close": [10.0, 11.0],
        }
    )
    frame.to_parquet(root / "2024-01-01.parquet", index=False)

    source = ParquetSource(
        root=root,
        timestamp_column="TradeDate",
        instrument_column="Symbol",
    )
    series = source.load_column("Close")
    assert len(series) == 2
    assert source.instrument_column == "Ticker"