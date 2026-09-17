"""Retest exact catalog source rows with an adequate observed cross-section."""
import sys
from data_access import get_store
from smoke_catalog import main

store = get_store()
table = store.read_arrow(
    "ashare_stock_daily_adj", columns=["Symbol"],
    time_range=("2026-04-30", "2026-04-30"), limit=64,
)
symbols = list(dict.fromkeys(table.column("Symbol").to_pylist()))
assert 22 <= len(symbols) <= 64
for source_row in (1845, 1890, 1939):
    sys.argv = [
        "smoke_catalog.py", "--offset", str(source_row - 2), "--limit", "1",
        "--batch-size", "1", "--symbols", ",".join(symbols),
        "--output", f"optional-cs-real-row{source_row}.jsonl.gz",
    ]
    main()
