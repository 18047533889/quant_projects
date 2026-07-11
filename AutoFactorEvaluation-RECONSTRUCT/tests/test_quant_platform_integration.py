from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa

from integrations.quant_platform import (
    execute_factor_formula,
    execute_factor_on_frame,
    materialize_factor_to_staging,
)


def _market_frame(periods: int = 12, symbols: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=periods, freq="B")
    assets = [f"S{i:03d}" for i in range(symbols)]
    rows = []
    for asset_idx, asset in enumerate(assets):
        for date_idx, date in enumerate(dates):
            close = 100.0 + asset_idx * 5.0 + date_idx
            rows.append(
                {
                    "TradeDate": date,
                    "Symbol": asset,
                    "Open": close - 0.5,
                    "High": close + 1.0,
                    "Low": close - 1.0,
                    "Close": close,
                    "PreClose": close - 1.0,
                    "Volume": 1_000 + date_idx,
                    "Amount": close * (1_000 + date_idx),
                    "Return": close / (close - 1.0) - 1.0,
                    "Vwap": close + 0.1,
                }
            )
    return pd.DataFrame(rows)


class _FakeStore:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.snapshot = SimpleNamespace(snapshot_id="snapshot-test-001")
        self.dataset = SimpleNamespace(
            name="ashare_stock_daily",
            kind="static",
            time_column="TradeDate",
            instrument_column="Symbol",
            schema={
                "TradeDate": "date",
                "Symbol": "string",
                "Open": "double",
                "High": "double",
                "Low": "double",
                "Close": "double",
                "PreClose": "double",
                "Volume": "int",
                "Amount": "double",
                "Return": "double",
                "Vwap": "double",
            },
        )
        self.upsert_calls = []
        self.publish_calls = []

    def describe_dataset(self, dataset, params=None, instrument_filter=None):
        assert dataset == "ashare_stock_daily"
        return self.snapshot

    def get_dataset(self, dataset):
        assert dataset == "ashare_stock_daily"
        return self.dataset

    def read_result(
        self,
        dataset,
        *,
        columns=None,
        time_range=None,
        instrument_filter=None,
        **params,
    ):
        assert dataset == "ashare_stock_daily"
        frame = self.frame.copy()
        if time_range:
            start, end = time_range
            if start is not None:
                frame = frame[frame["TradeDate"] >= pd.Timestamp(start)]
            if end is not None:
                frame = frame[frame["TradeDate"] <= pd.Timestamp(end)]
        if instrument_filter:
            frame = frame[frame["Symbol"].isin(instrument_filter)]
        if columns:
            frame = frame[list(columns)]
        return SimpleNamespace(
            table=pa.Table.from_pandas(frame, preserve_index=False),
            snapshot=self.snapshot,
        )

    def upsert(self, dataset, table, **kwargs):
        self.upsert_calls.append((dataset, table, kwargs))
        return {"rows": table.num_rows, "dataset": dataset}

    def publish_from_staging(self, source, target, **kwargs):
        self.publish_calls.append((source, target, kwargs))
        return {"source": source, "target": target}


def test_vendored_factor_engine_is_removed():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "factor_engine").exists()


def test_current_factor_engine_executes_on_in_memory_frame():
    frame = _market_frame().rename(
        columns={
            "TradeDate": "datetime",
            "Symbol": "asset",
            "Close": "close",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Volume": "volume",
            "Vwap": "vwap",
        }
    )
    execution = execute_factor_on_frame(
        "ts_mean(close, 3) / close - 1",
        frame,
        factor_name="integration_frame_factor",
    )
    assert len(execution.result) == 36
    assert isinstance(execution.result.index, pd.MultiIndex)
    assert np.isfinite(execution.result.to_numpy(dtype=float)).sum() > 0
    assert execution.snapshot_id.startswith("in_memory:")


def test_data_access_source_executes_and_records_snapshot(monkeypatch):
    import data_access

    store = _FakeStore(_market_frame())
    monkeypatch.setattr(data_access, "get_store", lambda: store)
    execution = execute_factor_formula(
        "ts_mean(close, 2)",
        factor_name="integration_data_access_factor",
        market="ashare",
        dataset="ashare_stock_daily",
        start_date="2024-01-02",
        end_date="2024-01-31",
    )
    assert len(execution.result) == 36
    assert execution.snapshot_id == "snapshot-test-001"
    assert np.isfinite(execution.result.to_numpy(dtype=float)).sum() > 0


def test_factor_staging_write_uses_data_access_upsert(monkeypatch):
    import data_access

    store = _FakeStore(_market_frame())
    monkeypatch.setattr(data_access, "get_store", lambda: store)
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=3), ["S001", "S002"]],
        names=["datetime", "asset"],
    )
    series = pd.Series(np.arange(len(idx), dtype=float), index=idx, name="value")
    summary = materialize_factor_to_staging(
        series,
        factor_id="integration_factor",
        snapshot_id="snapshot-test-001",
        publish=True,
    )
    assert summary["rows"] == len(series)
    assert store.upsert_calls
    dataset, table, kwargs = store.upsert_calls[0]
    assert dataset == "factor_lake_staging"
    assert table.num_rows == len(series)
    assert kwargs["upsert_on"] == ["datetime", "asset"]
    assert kwargs["partition_by"] == ["year"]
    assert store.publish_calls == [
        (
            "factor_lake_staging",
            "factor_lake",
            {"factor_id": "integration_factor"},
        )
    ]
