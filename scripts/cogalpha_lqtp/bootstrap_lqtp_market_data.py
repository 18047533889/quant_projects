#!/usr/bin/env python3
"""Bootstrap local StockDailyBar parquet from LQTP when COS mirror is unavailable."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cogalpha_lqtp.lqtp_client import (  # noqa: E402
    DEFAULT_SERVER,
    login,
    run_factor_formula,
)

# LQTP field -> parquet column
LQTP_COLUMNS: dict[str, str] = {
    "StockDailyBar.Open": "Open",
    "StockDailyBar.High": "High",
    "StockDailyBar.Low": "Low",
    "StockDailyBar.Close": "Close",
    "StockDailyBar.PreClose": "PreClose",
    "StockDailyBar.Volume": "Volume",
    "StockDailyBar.Amount": "Amount",
    "StockDailyBar.Vwap": "Vwap",
}


def _yyyymmdd(text: str) -> int:
    return int(text.replace("-", ""))


def _year_ranges(start: str, end: str) -> list[tuple[int, int, int]]:
    start_i = _yyyymmdd(start)
    end_i = _yyyymmdd(end)
    start_year = int(start[:4])
    end_year = int(end[:4])
    out: list[tuple[int, int, int]] = []
    for year in range(start_year, end_year + 1):
        y0 = year * 10000 + 101
        y1 = year * 10000 + 1231
        b = max(start_i, y0)
        e = min(end_i, y1)
        if b <= e:
            out.append((year, b, e))
    return out


def _fetch_column(
    *,
    token: str,
    formula: str,
    begin: int,
    end: int,
    server: str,
) -> dict[int, dict[str, float]]:
    resp = run_factor_formula(
        token=token,
        formula=formula,
        begin_date=begin,
        end_date=end,
        warmup=0,
        analyze=False,
        server=server,
    )
    if resp.error:
        raise RuntimeError(resp.error)
    out: dict[int, dict[str, float]] = defaultdict(dict)
    for point in resp.values:
        td = int(point.trade_date)
        for item in point.values:
            out[td][str(item.symbol)] = float(item.value)
    return out


def bootstrap(
    *,
    out_root: Path,
    start: str,
    end: str,
    token: str,
    server: str,
    force: bool,
) -> dict[str, int]:
    bar_dir = out_root / "StockDailyBar"
    bar_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    for _year, begin, end_i in _year_ranges(start, end):
        print(f"fetch year chunk {begin}-{end_i}")
        day_maps: dict[int, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        for formula, col in LQTP_COLUMNS.items():
            col_map = _fetch_column(
                token=token,
                formula=formula,
                begin=begin,
                end=end_i,
                server=server,
            )
            for td, sym_map in col_map.items():
                for sym, val in sym_map.items():
                    day_maps[td][sym][col] = val

        for td in sorted(day_maps):
            fname = f"{str(td)[0:4]}-{str(td)[4:6]}-{str(td)[6:8]}.parquet"
            path = bar_dir / fname
            if path.exists() and not force:
                skipped += 1
                continue
            rows: list[dict] = []
            for sym, cols in day_maps[td].items():
                if "Close" not in cols:
                    continue
                rows.append(
                    {
                        "TradeDate": pd.Timestamp(str(td)).date(),
                        "Symbol": sym,
                        "Open": cols.get("Open", cols["Close"]),
                        "High": cols.get("High", cols["Close"]),
                        "Low": cols.get("Low", cols["Close"]),
                        "Close": cols["Close"],
                        "PreClose": cols.get("PreClose", cols["Close"]),
                        "Volume": int(cols.get("Volume", 0)),
                        "Amount": float(cols.get("Amount", 0.0)),
                        "Return": 0.0,
                        "Vwap": float(cols.get("Vwap", cols["Close"])),
                        "Factor": 1.0,
                        "IsSuspend": 0,
                    }
                )
            if rows:
                pd.DataFrame(rows).to_parquet(path, index=False)
                written += 1
        day_maps.clear()

    return {"written": written, "skipped": skipped, "dir": str(bar_dir)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap StockDailyBar from LQTP")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--server", default=os.getenv("LQTP_SERVER", DEFAULT_SERVER))
    parser.add_argument("--username", default=os.getenv("LQTP_USERNAME", ""))
    parser.add_argument("--password", default=os.getenv("LQTP_PASSWORD", ""))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_root = args.out_root or Path(
        os.getenv("ASHARE_PARQUET_ROOT", str(ROOT / "data/a_share/lqtp_data"))
    )
    auth = login(args.server, args.username, args.password)
    stats = bootstrap(
        out_root=out_root,
        start=args.start,
        end=args.end,
        token=auth.access_token,
        server=args.server,
        force=args.force,
    )
    meta = {"start": args.start, "end": args.end, **stats}
    (out_root / "bootstrap_lqtp_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
