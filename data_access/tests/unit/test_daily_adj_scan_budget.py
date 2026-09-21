"""An optimizer date slice must prune unrelated daily partitions before admission."""
from datetime import date

import pandas as pd

from data_access.registry.loader import load_registry
from data_access.store import DataAccessStore
from data_access.read.query_budget import QueryBudget
from data_access.core.engine import DuckDBEngine


def test_adjusted_daily_date_slice_fits_single_file_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("ASHARE_PARQUET_ROOT", str(tmp_path))
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    root = tmp_path / "StockDailyBarAdj"
    root.mkdir()
    for day, value in [(1, 11.), (2, 22.), (3, 33.)]:
        pd.DataFrame({"TradeDate": [date(2024, 1, day)], "Symbol": ["a.SZ"],
                      "AdjVwap": [value]}).to_parquet(
                          root / f"2024-01-{day:02d}.parquet", index=False)
    engine = DuckDBEngine(threads=1)
    try:
        store = DataAccessStore(registry=load_registry(), engine=engine)
        out = store.read("ashare_stock_daily_adj", columns=["TradeDate", "Symbol", "AdjVwap"],
                         time_range=("2024-01-02", "2024-01-02"),
                         instrument_filter=["a.SZ"],
                         query_budget=QueryBudget(max_scan_files=1)).to_pandas()
        assert out["AdjVwap"].tolist() == [22.]
    finally:
        engine.close()
