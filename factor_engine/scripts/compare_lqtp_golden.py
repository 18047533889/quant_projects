#!/usr/bin/env python3
"""Compare FactorEngine output with an external LQTP golden result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _normalize(frame: pd.DataFrame, *, value_col: str) -> pd.DataFrame:
    aliases = {
        "timestamp": ["timestamp", "datetime", "TradeDate", "trade_date", "date"],
        "instrument": ["instrument", "asset", "Symbol", "Ticker", "ticker"],
    }
    out = frame.copy()
    for target, candidates in aliases.items():
        if target not in out.columns:
            found = next((c for c in candidates if c in out.columns), None)
            if found is None:
                raise ValueError(f"missing {target} column; tried {candidates}")
            out = out.rename(columns={found: target})
    if value_col not in out.columns:
        raise ValueError(f"missing value column {value_col!r}")
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="raise")
    out["instrument"] = out["instrument"].astype(str)
    return out[["timestamp", "instrument", value_col]].sort_values(["timestamp", "instrument"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("factor_engine", type=Path)
    ap.add_argument("lqtp_golden", type=Path)
    ap.add_argument("--fe-value", default="value")
    ap.add_argument("--gold-value", default="value")
    ap.add_argument("--rtol", type=float, default=1e-10)
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("--output", type=Path, default=Path("lqtp_golden_parity.json"))
    args = ap.parse_args()

    fe = _normalize(_read(args.factor_engine), value_col=args.fe_value).rename(columns={args.fe_value: "fe"})
    gold = _normalize(_read(args.lqtp_golden), value_col=args.gold_value).rename(columns={args.gold_value: "gold"})
    merged = fe.merge(gold, on=["timestamp", "instrument"], how="outer", indicator=True)
    shape_ok = bool((merged["_merge"] == "both").all())
    both = merged[merged["_merge"] == "both"].copy()
    fe_null = both["fe"].isna().to_numpy(); gold_null = both["gold"].isna().to_numpy()
    null_mask_ok = bool(np.array_equal(fe_null, gold_null))
    valid = ~(fe_null | gold_null)
    a = both.loc[valid, "fe"].to_numpy(dtype=float); b = both.loc[valid, "gold"].to_numpy(dtype=float)
    close = np.isclose(a, b, rtol=args.rtol, atol=args.atol, equal_nan=True) if len(a) else np.array([], dtype=bool)
    value_ok = bool(close.all()) if len(close) else True
    abs_err = np.abs(a - b) if len(a) else np.array([])
    denom = np.maximum(np.abs(b), args.atol) if len(b) else np.array([])
    rel_err = abs_err / denom if len(abs_err) else np.array([])
    report = {
        "schema_version": "factor_engine.lqtp_golden_parity.v1",
        "shape_ok": shape_ok,
        "null_mask_ok": null_mask_ok,
        "value_ok": value_ok,
        "passed": bool(shape_ok and null_mask_ok and value_ok),
        "rows_factor_engine": int(len(fe)),
        "rows_lqtp": int(len(gold)),
        "rows_compared": int(valid.sum()),
        "left_only": int((merged["_merge"] == "left_only").sum()),
        "right_only": int((merged["_merge"] == "right_only").sum()),
        "null_mask_mismatches": int(np.count_nonzero(fe_null != gold_null)),
        "value_mismatches": int(np.count_nonzero(~close)) if len(close) else 0,
        "max_abs_error": float(abs_err.max()) if len(abs_err) else 0.0,
        "max_rel_error": float(rel_err.max()) if len(rel_err) else 0.0,
        "rtol": args.rtol,
        "atol": args.atol,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
