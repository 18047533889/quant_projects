#!/usr/bin/env python3
"""Create minimal A-share StockDailyBar parquet for smoke tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Symbols observed in LQTP RunFactor output for 2024Q1.
SAFE_LQTP_SYMBOLS = [
    "000001.SZ", "000004.SZ", "000006.SZ", "000008.SZ", "000011.SZ",
    "000014.SZ", "000016.SZ", "000021.SZ", "000023.SZ", "000026.SZ",
    "000027.SZ", "000028.SZ", "000029.SZ", "000030.SZ", "000031.SZ",
    "000032.SZ", "000034.SZ", "000035.SZ", "000036.SZ", "000037.SZ",
    "000039.SZ", "000040.SZ", "000042.SZ", "000045.SZ", "000048.SZ",
    "000049.SZ", "000050.SZ", "000055.SZ", "000056.SZ", "000058.SZ",
    "000059.SZ", "000060.SZ", "000061.SZ", "000062.SZ", "000063.SZ",
    "000066.SZ", "000068.SZ", "000069.SZ", "000070.SZ", "000078.SZ",
    "000088.SZ", "000089.SZ", "000090.SZ", "000096.SZ", "000099.SZ",
    "000100.SZ", "000151.SZ", "000153.SZ", "000155.SZ", "000156.SZ",
]


def build_smoke_panel(
    *,
    start: str = "2024-01-02",
    end: str = "2024-03-29",
    n_symbols: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    symbols = SAFE_LQTP_SYMBOLS[:n_symbols]

    rows: list[dict] = []
    for symbol in symbols:
        price = 10.0 + rng.random() * 20.0
        for dt in dates:
            ret = rng.normal(0.0, 0.02)
            open_px = price
            close_px = max(0.5, price * (1.0 + ret))
            high_px = max(open_px, close_px) * (1.0 + abs(rng.normal(0, 0.005)))
            low_px = min(open_px, close_px) * (1.0 - abs(rng.normal(0, 0.005)))
            volume = int(rng.integers(1_000_000, 10_000_000))
            amount = volume * close_px
            pre_close = price
            rows.append(
                {
                    "TradeDate": dt.date(),
                    "Symbol": symbol,
                    "Open": float(open_px),
                    "High": float(high_px),
                    "Low": float(low_px),
                    "Close": float(close_px),
                    "PreClose": float(pre_close),
                    "Volume": volume,
                    "Amount": float(amount),
                    "Return": float(ret),
                    "Vwap": float((high_px + low_px + close_px) / 3.0),
                    "Factor": 1.0,
                    "IsSuspend": 0,
                }
            )
            price = close_px
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Write smoke StockDailyBar parquet")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start", default="2024-01-02")
    parser.add_argument("--end", default="2024-03-29")
    parser.add_argument("--n-symbols", type=int, default=50)
    args = parser.parse_args()

    frame = build_smoke_panel(
        start=args.start,
        end=args.end,
        n_symbols=args.n_symbols,
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke_2024Q1.parquet"
    frame.to_parquet(out_path, index=False)
    print(f"wrote {len(frame):,} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
