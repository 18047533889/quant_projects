# -*- coding: utf-8 -*-
"""staging 行级删除与 Kline prefetch 测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pd = pytest.importorskip("pandas")


def test_delete_staging_rows_after_watermark(tmp_path, monkeypatch):
    staging_root = tmp_path / "staging" / "ns" / "factor_lake" / "factors" / "f1"
    part = staging_root / "year=2024"
    part.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "asset": ["A", "A", "A"],
            "value": [1.0, 2.0, 3.0],
            "year": [2024, 2024, 2024],
        }
    )
    df.to_parquet(part / "data.parquet", index=False)

    class FakeStore:
        def resolve_dataset_path(self, dataset, **params):
            assert dataset == "factor_lake_staging"
            return staging_root

        def delete_rows(self, dataset, **kwargs):
            from factor_engine.storage.staging_loader import _delete_staging_rows_local

            return _delete_staging_rows_local(factor_id=kwargs.get("factor_id", "f1"), **{
                k: v for k, v in kwargs.items() if k in ("start", "end", "after")
            })

    import sys
    from types import ModuleType

    fake_mod = ModuleType("data_access")
    fake_mod.get_store = lambda: FakeStore()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "data_access", fake_mod)

    from factor_engine.storage.staging_loader import delete_staging_rows

    result = delete_staging_rows("f1", after="2024-01-02")
    assert result["rows_deleted"] == 2
    kept = pd.read_parquet(part / "data.parquet")
    assert len(kept) == 1
    assert kept.iloc[0]["datetime"] == pd.Timestamp("2024-01-02")


def test_kline_prefetch_columns_batch(tmp_path):
    pytest.importorskip("pyarrow")
    root = tmp_path / "kline"
    root.mkdir()
    for day, val in [("2024-01-02", 1.0), ("2024-01-03", 2.0)]:
        pd.DataFrame(
            {
                "ticker": ["A"],
                "window_start": [pd.Timestamp(day)],
                "close": [val],
                "open": [val + 0.1],
            }
        ).to_parquet(root / f"{day}.parquet", index=False)

    from factor_engine.storage.kline_parquet_source import KlineParquetSource

    src = KlineParquetSource(
        root=str(root),
        fields={"close": "close", "open": "open"},
        normalize_timestamp=True,
    )
    src.prefetch_columns(["close", "open"])
    assert "close" in src._column_cache
    assert "open" in src._column_cache
    assert len(src._column_cache["close"]) == 2
