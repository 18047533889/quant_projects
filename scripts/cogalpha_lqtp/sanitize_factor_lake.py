#!/usr/bin/env python3
"""Drop ±inf / non-finite values from factor_lake parquets; report empties."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK = ROOT / "data/cogalpha_lqtp_production"


def sanitize_one(path: Path) -> dict:
    df = pd.read_parquet(path)
    if "value" not in df.columns:
        return {"path": str(path), "status": "skip_no_value"}
    before = len(df)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    arr = df["value"].to_numpy(dtype="float64", copy=False)
    mask = np.isfinite(arr)
    n_inf = int((~mask & df["value"].notna().to_numpy()).sum()) if before else 0
    # also count explicit inf via isinf on original numeric
    n_bad = int((~mask).sum())
    cleaned = df.loc[mask].copy()
    after = len(cleaned)
    if after == 0:
        return {
            "path": str(path),
            "status": "empty_after_sanitize",
            "before": before,
            "removed": before,
            "inf_or_nonfinite": n_bad,
        }
    if after < before:
        cleaned.to_parquet(path, index=False)
        return {
            "path": str(path),
            "status": "sanitized",
            "before": before,
            "after": after,
            "removed": before - after,
            "inf_or_nonfinite": n_bad,
        }
    return {"path": str(path), "status": "ok", "rows": before, "inf_or_nonfinite": n_inf}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    lake = args.work_dir / "factor_lake"
    names = sorted(p.name for p in lake.iterdir() if (p / "values.parquet").is_file())
    if args.only:
        only = set(args.only)
        names = [n for n in names if n in only]
    report = []
    for name in names:
        try:
            report.append(sanitize_one(lake / name / "values.parquet"))
        except Exception as exc:  # noqa: BLE001
            report.append({"path": name, "status": "error", "error": str(exc)})
    out = {
        "generated_at": datetime.now().isoformat(),
        "n": len(report),
        "sanitized": sum(1 for r in report if r.get("status") == "sanitized"),
        "empty": sum(1 for r in report if r.get("status") == "empty_after_sanitize"),
        "errors": sum(1 for r in report if r.get("status") == "error"),
        "details": [r for r in report if r.get("status") not in {"ok"}],
    }
    path = args.work_dir / "factor_lake_sanitize_report.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"sanitize done n={out['n']} sanitized={out['sanitized']} "
        f"empty={out['empty']} errors={out['errors']} -> {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
