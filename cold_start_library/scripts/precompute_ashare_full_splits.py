#!/usr/bin/env python3
"""按 AlphaPROBE split 一劳永逸预计算冷启动因子值 + train/valid/test IC。

默认覆盖：
  train 2016-01-01 ~ 2021-12-31
  valid 2022-01-01 ~ 2023-12-31
  test  2024-01-01 ~ 2026-07-31

策略：每个因子在 full span 上只 run 一次，落 float16 panel，再切片算三段 IC。
AlphaPROBE 训练期可通过 value_cache 免重算冷启动种子。

输出：
  data/ashare/value_cache/factors/<hash>/panel.npz + meta.json
  data/ashare/backend_v9_core.yaml   # metrics.train/valid/test 更新
  data/ashare/full_split_precompute_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FE = ROOT.parent / "factor_engine"
DA = ROOT.parent / "data_access"
for p in (str(SRC), str(FE), str(DA)):
    if p not in sys.path:
        sys.path.insert(0, p)

_LOCAL_ASHARE = Path.home() / "quant_projects" / "data" / "a_share" / "lqtp_data"

SPLITS = {
    "train": ("2016-01-01", "2021-12-31"),
    "valid": ("2022-01-01", "2023-12-31"),
    "test": ("2024-01-01", "2026-07-31"),
}


class _RetDecimalProxy:
    def __init__(self, inner: Any, scale: float = 1.0 / 10000.0) -> None:
        self._inner = inner
        self._scale = float(scale)

    def load_column(self, name: str):
        series = self._inner.load_column(name)
        if name in {"ret", "Return"}:
            return series * self._scale
        return series

    def load_columns(self, names: list[str]):
        out = self._inner.load_columns(names)
        for name in names:
            if name in {"ret", "Return"} and name in out:
                out[name] = out[name] * self._scale
        return out

    def __getattr__(self, item: str):
        return getattr(self._inner, item)


def build_engine(start: str, end: str, *, use_valuation: bool = True):
    from backend.factory import build_backend
    from runtime.engine import FactorEngine
    from storage.factory import build_data_source
    from api.mining_integration import _VALUATION_FIELD_ALIASES

    os.environ.setdefault("ASHARE_PARQUET_ROOT", str(_LOCAL_ASHARE))
    pv_fields = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "vwap": "Vwap",
        "amount": "Amount",
        "ret": "Return",
        "pre_close": "PreClose",
        "preclose": "PreClose",
        "factor": "Factor",
    }
    pv_cfg: dict[str, Any] = {
        "type": "parquet",
        "root": str(_LOCAL_ASHARE / "StockDailyBar"),
        "timestamp_col": "TradeDate",
        "instrument_col": "Symbol",
        "fields": pv_fields,
        "start_date": start,
        "end_date": end,
    }
    if use_valuation:
        val_cfg = {
            "type": "parquet",
            "root": str(_LOCAL_ASHARE / "StockValuationDaily"),
            "timestamp_col": "TradeDate",
            "instrument_col": "Symbol",
            "fields": {
                "pe": "PeRatio",
                "pb": "PbRatio",
                "turnover_ratio": "TurnoverRatio",
                "market_cap": "MarketCap",
                "circulating_market_cap": "CirculatingMarketCap",
            },
            "start_date": start,
            "end_date": end,
        }
        cfg: dict[str, Any] = {
            "type": "composite",
            "anchor": "pv",
            "anchor_column": "close",
            "sources": {"pv": pv_cfg, "valuation": val_cfg},
            "joins": {"valuation": "asof_backward"},
            "aliases": dict(_VALUATION_FIELD_ALIASES),
        }
        raw = build_data_source(cfg)
    else:
        raw = build_data_source(pv_cfg)
    ds = _RetDecimalProxy(raw)
    eng = FactorEngine(backend=build_backend("pandas"), data_source=ds, run_mode="research")
    return eng, ds


def make_fwd_matrix(ds, label_days: int):
    vwap = ds.load_column("vwap")
    fwd = vwap.groupby(level=1).shift(-label_days) / vwap - 1.0
    wide = fwd.unstack(level=1)
    return wide.index, wide.columns, wide.to_numpy(dtype=float)


def panel_ic(f_mat: np.ndarray, y_mat: np.ndarray, *, min_cs: int = 30) -> tuple[float | None, float | None]:
    ics: list[float] = []
    n = min(f_mat.shape[0], y_mat.shape[0])
    for i in range(n):
        a = f_mat[i]
        b = y_mat[i]
        mask = np.isfinite(a) & np.isfinite(b)
        if int(mask.sum()) < min_cs:
            continue
        aa = a[mask] - a[mask].mean()
        bb = b[mask] - b[mask].mean()
        da = float(np.dot(aa, aa))
        db = float(np.dot(bb, bb))
        if da < 1e-18 or db < 1e-18:
            continue
        c = float(np.dot(aa, bb) / math.sqrt(da * db))
        if np.isfinite(c):
            ics.append(c)
    if not ics:
        return None, None
    arr = np.asarray(ics, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
    icir = float(mean / std) if std and np.isfinite(std) and std > 1e-12 else None
    return mean, icir


def slice_mask(dates: pd.Index, start: str, end: str) -> np.ndarray:
    d = pd.to_datetime(pd.Index(dates))
    return (d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", type=Path, default=ROOT / "data" / "ashare" / "backend_v9_core.yaml")
    ap.add_argument("--cache-root", type=Path, default=ROOT / "data" / "ashare" / "value_cache")
    ap.add_argument("--full-start", default="2016-01-01")
    ap.add_argument("--full-end", default="2026-07-31")
    ap.add_argument("--label-days", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-valuation", action="store_true")
    ap.add_argument("--metrics-only", action="store_true", help="不算/不落值，只刷三段 IC（仍需 run）")
    ap.add_argument("--skip-existing-cache", action="store_true", help="已有 panel 则跳过计算")
    args = ap.parse_args()

    from api.dsl_parser import parse_factor
    from cold_start_library.runtime.value_cache import (
        load_factor_meta,
        load_factor_panel,
        save_factor_panel,
        dsl_hash,
    )

    payload = yaml.safe_load(args.yaml.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = list(payload.get("entries") or [])
    if args.limit > 0:
        entries = entries[: args.limit]
    print(f"[load] entries={len(entries)} yaml={args.yaml}", flush=True)

    use_val = not args.no_valuation
    print(f"[engine] {args.full_start}..{args.full_end} valuation={use_val}", flush=True)
    eng, ds = build_engine(args.full_start, args.full_end, use_valuation=use_val)
    dates, stocks, y_full = make_fwd_matrix(ds, args.label_days)
    print(f"[panel] days={len(dates)} stocks={len(stocks)}", flush=True)

    split_masks = {k: slice_mask(dates, a, b) for k, (a, b) in SPLITS.items()}
    for k, m in split_masks.items():
        print(f"[split] {k}: {int(m.sum())} days ({SPLITS[k][0]}..{SPLITS[k][1]})", flush=True)

    args.cache_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.cache_root / "full_split_progress.jsonl"
    summary_path = ROOT / "data" / "ashare" / "full_split_precompute_summary.json"

    done = 0
    skipped = 0
    failed = 0
    t_all = time.time()

    with progress_path.open("a", encoding="utf-8") as prog:
        for i, entry in enumerate(entries):
            fid = str(entry.get("factor_id") or f"idx_{i}")
            expr = " ".join(str(entry.get("expr") or "").split())
            if not expr:
                continue

            # resume: if meta has full metrics, skip
            meta = load_factor_meta(args.cache_root, expr)
            if args.skip_existing_cache and meta and meta.get("metrics", {}).get("train"):
                entry["metrics"] = {
                    **(entry.get("metrics") or {}),
                    **meta["metrics"],
                    "values_cached": True,
                    "expr_hash": meta.get("expr_hash"),
                    "cache_span": {"start": args.full_start, "end": args.full_end},
                }
                skipped += 1
                if (i + 1) % 50 == 0:
                    print(f"[skip] {i+1}/{len(entries)} skipped={skipped}", flush=True)
                continue

            t0 = time.time()
            try:
                fac = parse_factor(expr, name="cs", surface="compat")
                result = eng.run(fac)["result"]
                s = pd.to_numeric(result, errors="coerce")
                wide = s.unstack(level=1).reindex(index=dates, columns=stocks)
                f_mat = wide.to_numpy(dtype=float)
                finite_ratio = float(np.isfinite(f_mat).mean()) if f_mat.size else 0.0

                metrics: dict[str, Any] = {
                    "label_days": args.label_days,
                    "finite_ratio": finite_ratio,
                    "cache_span": {"start": args.full_start, "end": args.full_end},
                }
                for name, mask in split_masks.items():
                    if not mask.any():
                        metrics[name] = {"ic": None, "icir": None, "abs_ic": None, "n_days": 0}
                        continue
                    ic, icir = panel_ic(f_mat[mask], y_full[mask])
                    metrics[name] = {
                        "ic": ic,
                        "icir": icir,
                        "abs_ic": abs(ic) if ic is not None else None,
                        "n_days": int(mask.sum()),
                        "window": {"start": SPLITS[name][0], "end": SPLITS[name][1]},
                    }

                # primary metrics keep train IC for backward compat
                train_m = metrics.get("train") or {}
                metrics["ic"] = train_m.get("ic")
                metrics["icir"] = train_m.get("icir")
                metrics["abs_ic"] = train_m.get("abs_ic")

                if not args.metrics_only:
                    save_factor_panel(
                        args.cache_root,
                        factor_id=fid,
                        expr=expr,
                        dates=dates,
                        stocks=stocks,
                        values=f_mat,
                        metrics=metrics,
                    )
                    metrics["values_cached"] = True
                    metrics["expr_hash"] = dsl_hash(expr)
                entry["metrics"] = metrics
                done += 1
                row = {
                    "factor_id": fid,
                    "status": "pass",
                    "elapsed_sec": round(time.time() - t0, 3),
                    "finite_ratio": finite_ratio,
                    "train_ic": train_m.get("ic"),
                    "valid_ic": (metrics.get("valid") or {}).get("ic"),
                    "test_ic": (metrics.get("test") or {}).get("ic"),
                }
                prog.write(json.dumps(row, ensure_ascii=False) + "\n")
                prog.flush()
            except Exception as exc:
                failed += 1
                err = {"factor_id": fid, "status": "fail", "error": f"{type(exc).__name__}: {exc}"}
                prog.write(json.dumps(err, ensure_ascii=False) + "\n")
                prog.flush()

            if (i + 1) % 10 == 0 or i + 1 == len(entries):
                elapsed = time.time() - t_all
                rate = (done + failed) / max(elapsed, 1e-6)
                eta = (len(entries) - i - 1) / max(rate, 1e-9) / 3600
                print(
                    f"[precompute] {i+1}/{len(entries)} done={done} fail={failed} skip={skipped} "
                    f"rate={rate*3600:.1f}/h eta={eta:.1f}h",
                    flush=True,
                )

    # Always re-read yaml and merge metrics by factor_id so concurrent
    # library expansions (new entries) are not wiped by this job.
    full = yaml.safe_load(args.yaml.read_text(encoding="utf-8"))
    full_entries = list(full.get("entries") or [])
    by_id = {str(e.get("factor_id")): e for e in entries}
    for j, e in enumerate(full_entries):
        fid = str(e.get("factor_id"))
        if fid in by_id and by_id[fid].get("metrics"):
            full_entries[j]["metrics"] = by_id[fid]["metrics"]
    # keep any brand-new ids computed in this run but not yet on disk
    disk_ids = {str(e.get("factor_id")) for e in full_entries}
    for e in entries:
        fid = str(e.get("factor_id"))
        if fid not in disk_ids and e.get("metrics"):
            full_entries.append(e)
    payload = full
    payload["entries"] = full_entries

    payload["values_precomputed"] = True
    payload["value_cache_root"] = str(args.cache_root)
    payload["metrics_splits"] = {k: {"start": a, "end": b} for k, (a, b) in SPLITS.items()}
    payload["source"] = (
        f"V9+expand; value_cache full-span {args.full_start}..{args.full_end}; "
        f"splits=train/valid/test"
    )
    with args.yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False, width=120)

    summary = {
        "entries": len(payload.get("entries") or []),
        "done": done,
        "failed": failed,
        "skipped": skipped,
        "cache_root": str(args.cache_root),
        "full_span": {"start": args.full_start, "end": args.full_end},
        "splits": SPLITS,
        "label_days": args.label_days,
        "elapsed_sec": round(time.time() - t_all, 1),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"[done] yaml updated + cache -> {args.cache_root}", flush=True)
    return 0 if failed < len(entries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
