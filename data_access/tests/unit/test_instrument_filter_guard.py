"""instrument_filter 与 schema 一致性守卫。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_access import reset_store
from data_access.engine import DuckDBEngine
from data_access.exceptions import ValidationError
from data_access.registry import load_registry
from data_access.store import DataAccessStore


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    reset_store()


def test_instrument_filter_rejected_when_column_missing_from_schema(tmp_path):
    root = tmp_path / "bars"
    root.mkdir()
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02"]),
            "close": [1.0],
        }
    ).to_parquet(root / "2024-01-02.parquet")

    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(
        f"""
streaming_bars:
  kind: static
  access_mode: published
  layout: plain
  root: {root}
  glob: "**/*.parquet"
  time_column: timestamp
  instrument_column: symbol
  hive_partitioning: false
  union_by_name: true
  schema:
    timestamp: timestamp
    close: double
""".strip()
        + "\n",
        encoding="utf-8",
    )
    store = DataAccessStore(load_registry(cfg), DuckDBEngine())
    with pytest.raises(ValidationError, match="不支持 instrument_filter"):
        store.read_arrow(
            "streaming_bars",
            columns=["close"],
            instrument_filter=["AAPL"],
        )
