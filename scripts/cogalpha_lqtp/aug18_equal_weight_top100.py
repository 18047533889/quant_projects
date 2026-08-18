#!/usr/bin/env python3
"""Materialize RankIC>2% DSL factors for 2026-08-18 ± 5 sessions, equal-weight, Top100."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.cogalpha_lqtp.ast_translator import _replace_func_calls, dsl_to_lqtp  # noqa: E402
from scripts.cogalpha_lqtp.lqtp_client import (  # noqa: E402
    LqtpTokenManager,
    factor_values_to_long_df,
    run_factor_formula,
)


def _split_top_level_args(text: str) -> list[str]:
    args: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in text:
        if ch == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        args.append(tail)
    return args


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _normalize_symbol(symbol: str) -> str:
    text = str(symbol).strip()
    if "." in text:
        return text
    if text.isdigit():
        return f"{text.zfill(6)}.SZ"
    return text


_TRUE_RANGE = (
    "where((high - low) > abs(high - delay(close, 1)), "
    "where((high - low) > abs(low - delay(close, 1)), high - low, abs(low - delay(close, 1))), "
    "where(abs(high - delay(close, 1)) > abs(low - delay(close, 1)), "
    "abs(high - delay(close, 1)), abs(low - delay(close, 1))))"
)


def _rewrite_binary_minmax(text: str) -> str:
    """LQTP has no max/min; rewrite 2-arg calls to where()."""
    import re

    fn_pat = re.compile(r"\b(min|max|minimum|maximum)\s*\(")

    def _scan_balanced(src: str, start: int) -> int:
        depth = 0
        i = start
        while i < len(src):
            if src[i] == "(":
                depth += 1
            elif src[i] == ")":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return len(src)

    out = text
    for _ in range(128):
        m = fn_pat.search(out)
        if not m:
            break
        open_idx = m.end() - 1
        close_idx = _scan_balanced(out, open_idx)
        inner = out[open_idx + 1 : close_idx - 1]
        args = _split_top_level_args(inner)
        if len(args) != 2:
            break
        a, b = args[0], args[1]
        if m.group(1) in {"min", "minimum"}:
            repl = f"where(({a}) < ({b}), {a}, {b})"
        else:
            repl = f"where(({a}) > ({b}), {a}, {b})"
        out = out[: m.start()] + repl + out[close_idx:]
    return out


def rewrite_dsl_for_lqtp(dsl: str) -> str:
    def atr(args: list[str]) -> str | None:
        if len(args) < 4:
            return None
        return f"ema({_TRUE_RANGE}, {args[3]})"

    def rv(args: list[str]) -> str | None:
        if len(args) != 3:
            return None
        n = args[2]
        return f"safe_div(ts_sum(close * volume, {n}), ts_sum(volume, {n}))"

    def pow_repl(args: list[str]) -> str | None:
        if len(args) != 2:
            return None
        base, exp = args[0].strip(), args[1].strip()
        if exp in {"2", "2.0"}:
            return f"(({base}) * ({base}))"
        if exp in {"0.5", "1/2"}:
            return f"sqrt({base})"
        return None

    out = dsl
    out = _replace_func_calls(out, "ATR_WILDER", atr)
    out = _replace_func_calls(out, "ATR", atr)
    out = _replace_func_calls(out, "rolling_vwap", rv)
    out = _replace_func_calls(out, "pow", pow_repl)
    out = _replace_func_calls(out, "power", pow_repl)
    out = dsl_to_lqtp(out)
    out = _rewrite_binary_minmax(out)
    import re

    def divide_repl(args: list[str]) -> str | None:
        if len(args) != 2:
            return None
        return f"safe_div({args[0]}, {args[1]})"

    out = _replace_func_calls(out, "divide", divide_repl)
    out = re.sub(r"\bvolume\b", "(volume * 1.0)", out)
    return out


def _lqtp_to_keep_df(resp_values: Any) -> pd.DataFrame:
    long_df = factor_values_to_long_df(resp_values)
    if long_df.empty:
        return pd.DataFrame(columns=["datetime", "asset", "value"])
    long_df["datetime"] = pd.to_datetime(long_df["trade_date"].astype(str), format="%Y%m%d")
    long_df["asset"] = long_df["symbol"].map(_normalize_symbol)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    keep = set(pd.to_datetime(KEEP_DATES))
    long_df = long_df[long_df["datetime"].isin(keep)]
    long_df = long_df.dropna(subset=["value"])
    long_df = long_df[np.isfinite(long_df["value"].to_numpy(dtype="float64", copy=False))]
    return long_df[["datetime", "asset", "value"]]

KEEP_DATES = [
    "2026-08-11",
    "2026-08-12",
    "2026-08-13",
    "2026-08-14",
    "2026-08-17",
    "2026-08-18",
]
SIGNAL_DATE = "2026-08-17"
PREV_DATE = "2026-08-14"
TRADE_DATE = "2026-08-18"

TECH_SW_L1 = {"电子", "计算机", "通信", "传媒"}
CONSUMER_SW_L1 = {
    "食品饮料",
    "家用电器",
    "商贸零售",
    "商业贸易",
    "社会服务",
    "休闲服务",
    "美容护理",
    "纺织服饰",
    "纺织服装",
    "农林牧渔",
    "轻工制造",
}
SW_L1_ALIAS = {
    "商业贸易": "商贸零售",
    "纺织服装": "纺织服饰",
    "休闲服务": "社会服务",
}

SKIP_PYTHON_ONLY = {
    "factor_adaptive_volume_smoothness",
    "factor_downside_vol_liquidity_momentum",
    "factor_liquidity_adaptive_momentum",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _catalog_index(catalog: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for e in catalog:
        out[e["function_name"]] = e
        disp = e.get("manifest_display_name")
        if disp:
            out.setdefault(str(disp), e)
    return out


def resolve_dsl_jobs(work: Path) -> list[dict[str, str]]:
    man = _load_json(work / "factor_rankic_screening_manifest.json")["factors"]
    cat = _load_json(work / "screening_reeval_catalog.json")
    idx = _catalog_index(cat)
    jobs: list[dict[str, str]] = []
    for rec in man:
        name = rec["display_name"]
        if name in SKIP_PYTHON_ONLY:
            continue
        entry = idx.get(name)
        if not entry:
            continue
        dsl = (entry.get("lqtp_formula") or entry.get("dsl") or "").strip()
        if not dsl:
            continue
        jobs.append({"name": name, "dsl": dsl})
    return jobs


def materialize_batch(
    *,
    jobs: list[dict[str, str]],
    lake_root: Path,
    start: str,
    end: str,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    _ = start, end
    _load_dotenv(ROOT / ".env")
    lake_root.mkdir(parents=True, exist_ok=True)
    progress_path = lake_root / "materialize_progress.json"
    progress = {"ok": [], "failed": {}, "via": {}}
    if progress_path.exists():
        progress = _load_json(progress_path)
    ok = set(progress.get("ok", []))
    failed: dict[str, str] = dict(progress.get("failed", {}))
    via: dict[str, str] = dict(progress.get("via", {}))

    todo = jobs[:limit] if limit else jobs
    pending = []
    for job in todo:
        name = job["name"]
        out_path = lake_root / name / "values.parquet"
        if out_path.exists() and out_path.stat().st_size > 1000 and not force:
            ok.add(name)
            failed.pop(name, None)
            continue
        pending.append(job)

    n_reuse = len(ok.intersection(j["name"] for j in todo))
    print(f"reuse={n_reuse} pending={len(pending)}", flush=True)
    if not pending:
        progress = {"ok": sorted(ok), "failed": failed, "via": via}
        _save_json(progress_path, progress)
        return progress

    user = os.environ.get("LQTP_USERNAME", "").strip()
    pwd = os.environ.get("LQTP_PASSWORD", "")
    server = os.environ.get("LQTP_SERVER", "110.42.223.26:50051").strip()
    if not user or not pwd:
        raise RuntimeError("LQTP_USERNAME/LQTP_PASSWORD missing in env")
    auth = LqtpTokenManager(username=user, password=pwd, server=server)

    for i, job in enumerate(pending, 1):
        name = job["name"]
        formula = rewrite_dsl_for_lqtp(job["dsl"])
        print(f"lqtp {i}/{len(pending)} {name}", flush=True)
        t1 = time.time()
        try:
            resp = run_factor_formula(
                token=auth.token,
                formula=formula,
                begin_date=20260811,
                end_date=20260818,
                warmup=252,
                analyze=False,
                server=server,
                factor_name=name,
            )
            if resp.error:
                raise RuntimeError(resp.error)
            long_df = _lqtp_to_keep_df(resp.values)
            if long_df.empty:
                raise RuntimeError("empty after date slice")
            out_dir = lake_root / name
            out_dir.mkdir(parents=True, exist_ok=True)
            long_df.to_parquet(out_dir / "values.parquet", index=False)
            ok.add(name)
            failed.pop(name, None)
            via[name] = "lqtp"
            print(
                f"  OK {name} rows={len(long_df)} dates={int(long_df['datetime'].nunique())} "
                f"wall={time.time() - t1:.1f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            failed[name] = str(exc)[:500]
            print(f"  FAIL {name}: {exc}", flush=True)
        progress = {"ok": sorted(ok), "failed": failed, "via": via}
        _save_json(progress_path, progress)
    return progress


def _load_sw_l1(parquet_root: Path, trade_date: str) -> pd.DataFrame:
    path = parquet_root / "StockIndustry" / f"{trade_date}.parquet"
    ind = pd.read_parquet(path)
    ind = ind[ind["IndustrySource"].astype(str) == "sw_l1"].copy()
    ind["asset"] = ind["Symbol"].map(_normalize_symbol)
    ind["sw_l1"] = ind["IndustryName"].astype(str).map(_canon_sw_l1)
    return ind[["asset", "sw_l1"]].drop_duplicates("asset")


def _load_bars(parquet_root: Path, trade_date: str) -> pd.DataFrame:
    path = parquet_root / "StockDailyBar" / f"{trade_date}.parquet"
    bar = pd.read_parquet(path)
    bar["asset"] = bar["Symbol"].map(_normalize_symbol)
    close = pd.to_numeric(bar["Close"], errors="coerce")
    pre = pd.to_numeric(bar["PreClose"], errors="coerce")
    bar["ret"] = close / pre.replace(0.0, pd.NA) - 1.0
    bar["close"] = close
    bar["vwap"] = pd.to_numeric(bar["Vwap"], errors="coerce")
    sus = bar["IsSuspend"] if "IsSuspend" in bar.columns else 0
    bar["is_suspend"] = sus.fillna(False).astype(bool)
    return bar[["asset", "ret", "close", "vwap", "is_suspend"]]


def _canon_sw_l1(name: str) -> str:
    text = str(name).strip()
    if text.endswith("I") and not text.endswith("II"):
        text = text[:-1]
    return SW_L1_ALIAS.get(text, text)


def _style(name: str) -> str:
    key = _canon_sw_l1(name)
    if key in TECH_SW_L1:
        return "科技"
    if key in CONSUMER_SW_L1:
        return "消费"
    return "其他"


def _cs_rank_pct(s: pd.Series) -> pd.Series:
    return s.rank(method="average", pct=True)


def build_combo(lake_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    names: list[str] = []
    for path in sorted(lake_root.glob("*/values.parquet")):
        name = path.parent.name
        if name.startswith("_"):
            continue
        df = pd.read_parquet(path, columns=["datetime", "asset", "value"])
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()
        df["factor"] = name
        frames.append(df)
        names.append(name)
    if not frames:
        raise RuntimeError("no factor values in lake")
    long = pd.concat(frames, ignore_index=True)
    long["cs_rank"] = long.groupby(["datetime", "factor"], sort=False)["value"].transform(_cs_rank_pct)
    combo = (
        long.groupby(["datetime", "asset"], sort=False)
        .agg(combo=("cs_rank", "mean"), n_factors=("cs_rank", "size"))
        .reset_index()
    )
    min_n = max(1, int(round(0.5 * len(names))))
    combo = combo[combo["n_factors"] >= min_n].copy()
    combo["combo_rank"] = combo.groupby("datetime", sort=False)["combo"].rank(
        method="first", ascending=False
    )
    combo["n_universe"] = combo.groupby("datetime", sort=False)["asset"].transform("size")
    combo.attrs["n_factors_used"] = len(names)
    combo.attrs["min_n"] = min_n
    return combo


def topk(combo: pd.DataFrame, date: str, k: int = 100) -> pd.DataFrame:
    d = pd.Timestamp(date)
    part = combo[combo["datetime"] == d].sort_values("combo_rank")
    return part.head(k).copy()


def _mix(top: pd.DataFrame, industry: pd.DataFrame) -> pd.DataFrame:
    m = top.merge(industry, on="asset", how="left")
    m["sw_l1"] = m["sw_l1"].fillna("未知")
    m["style"] = m["sw_l1"].map(_style)
    return m


def _style_counts(m: pd.DataFrame) -> dict[str, int]:
    vc = m["style"].value_counts()
    return {k: int(vc.get(k, 0)) for k in ("科技", "消费", "其他")}


def _industry_counts(m: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in m["sw_l1"].value_counts().items()}


def analyze(*, lake_root: Path, parquet_root: Path, out_dir: Path, k: int = 100) -> dict[str, Any]:
    combo = build_combo(lake_root)
    n_factors = int(combo.attrs.get("n_factors_used", 0))
    industry_by_date = {d: _load_sw_l1(parquet_root, d) for d in KEEP_DATES}
    bars_by_date = {d: _load_bars(parquet_root, d) for d in KEEP_DATES}

    daily: dict[str, Any] = {}
    top_frames: dict[str, pd.DataFrame] = {}
    for d in KEEP_DATES:
        t = _mix(topk(combo, d, k=k), industry_by_date[d])
        t["date"] = d
        top_frames[d] = t
        daily[d] = {
            "style": _style_counts(t),
            "industry": _industry_counts(t),
            "mean_combo": float(t["combo"].mean()),
        }

    old = top_frames[PREV_DATE]
    new = top_frames[SIGNAL_DATE]
    old_set = set(old["asset"])
    new_set = set(new["asset"])
    entered = new[~new["asset"].isin(old_set)].copy()
    exited = old[~old["asset"].isin(new_set)].copy()
    stayed = new[new["asset"].isin(old_set)].copy()

    ret18 = bars_by_date[TRADE_DATE][["asset", "ret"]].rename(columns={"ret": "ret_0818"})
    # market-wide 8.18 style returns
    mkt = bars_by_date[TRADE_DATE].merge(industry_by_date[TRADE_DATE], on="asset", how="left")
    mkt = mkt[~mkt["is_suspend"]].copy()
    mkt["style"] = mkt["sw_l1"].fillna("未知").map(_style)

    def _ew_ret(df: pd.DataFrame, assets: set[str]) -> float:
        part = ret18[ret18["asset"].isin(assets)]["ret_0818"]
        return float(part.mean()) if len(part) else float("nan")

    def _style_ew(df: pd.DataFrame) -> dict[str, float]:
        g = df.groupby("style")["ret"].mean()
        return {k: float(g.get(k)) for k in ("科技", "消费", "其他") if k in g.index}

    # attach 8.18 return onto 8.17 book
    new_ret = new.merge(ret18, on="asset", how="left")
    old_ret = old.merge(ret18, on="asset", how="left")
    entered_ret = entered.merge(ret18, on="asset", how="left")
    exited_ret = exited.merge(ret18, on="asset", how="left")

    summary = {
        "n_factors_used": n_factors,
        "top_k": k,
        "keep_dates": KEEP_DATES,
        "hypothesis": (
            "8.17 收盘等权因子 Top100 相对 8.14 已减科技/增消费，"
            "对应 8.18 量化调仓日"
        ),
        "daily_style_counts": daily,
        "rotation_8_14_to_8_17": {
            "overlap": len(old_set & new_set),
            "entered": len(entered),
            "exited": len(exited),
            "entered_style": _style_counts(entered),
            "exited_style": _style_counts(exited),
            "entered_industry": _industry_counts(entered),
            "exited_industry": _industry_counts(exited),
            "old_style": _style_counts(old),
            "new_style": _style_counts(new),
            "delta_style": {
                s: _style_counts(new)[s] - _style_counts(old)[s] for s in ("科技", "消费", "其他")
            },
        },
        "ret_20260818": {
            "market_style_ew": _style_ew(mkt),
            "book_8_14_signal": _ew_ret(ret18, old_set),
            "book_8_17_signal": _ew_ret(ret18, new_set),
            "entered_8_17": _ew_ret(ret18, set(entered["asset"])),
            "exited_8_17": _ew_ret(ret18, set(exited["asset"])),
            "stayed": _ew_ret(ret18, set(stayed["asset"])),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    combo.to_parquet(out_dir / "combo_scores.parquet", index=False)
    all_top = pd.concat(top_frames.values(), ignore_index=True)
    all_top.to_csv(out_dir / "top100_by_date.csv", index=False)
    entered.assign(side="enter_on_0817").to_csv(out_dir / "entered_0817_vs_0814.csv", index=False)
    exited.assign(side="exit_on_0817").to_csv(out_dir / "exited_0817_vs_0814.csv", index=False)
    new_ret.to_csv(out_dir / "top100_0817_with_0818_ret.csv", index=False)
    old_ret.to_csv(out_dir / "top100_0814_with_0818_ret.csv", index=False)
    entered_ret.to_csv(out_dir / "entered_with_0818_ret.csv", index=False)
    exited_ret.to_csv(out_dir / "exited_with_0818_ret.csv", index=False)
    _save_json(out_dir / "rotation_summary.json", summary)

    # compact top100 tables for canvas
    def _slim(df: pd.DataFrame) -> list[dict[str, Any]]:
        cols = ["combo_rank", "asset", "sw_l1", "style", "combo", "n_factors"]
        extra = [c for c in ("ret_0818",) if c in df.columns]
        part = df[cols + extra].copy()
        part["combo_rank"] = part["combo_rank"].astype(int)
        part["combo"] = part["combo"].round(4)
        if "ret_0818" in part.columns:
            part["ret_0818"] = part["ret_0818"].round(4)
        return part.to_dict(orient="records")

    slim = {
        d: _slim(top_frames[d]) for d in KEEP_DATES
    }
    slim["top100_0817_with_ret"] = _slim(new_ret)
    slim["entered"] = _slim(entered_ret)
    slim["exited"] = _slim(exited_ret)
    _save_json(out_dir / "top100_slim.json", slim)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/cogalpha_lqtp_production")
    parser.add_argument(
        "--lake-root",
        type=Path,
        default=ROOT / "data/cogalpha_lqtp_production/factor_lake_aug18_window",
    )
    parser.add_argument(
        "--parquet-root",
        type=Path,
        default=ROOT / "data/a_share/lqtp_data",
    )
    parser.add_argument("--start", default="2025-08-01")
    parser.add_argument("--end", default="2026-08-18")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--k", type=int, default=100)
    args = parser.parse_args()

    jobs = resolve_dsl_jobs(args.work_dir)
    print(f"dsl jobs={len(jobs)}", flush=True)
    if not args.analyze_only:
        progress = materialize_batch(
            jobs=jobs,
            lake_root=args.lake_root,
            start=args.start,
            end=args.end,
            limit=args.limit,
            force=args.force,
        )
        print(
            f"materialize ok={len(progress['ok'])} failed={len(progress['failed'])}",
            flush=True,
        )
        if progress["failed"]:
            print("failed:", json.dumps(progress["failed"], ensure_ascii=False)[:2000], flush=True)

    n_ok = len(list(args.lake_root.glob("*/values.parquet")))
    if n_ok == 0:
        print("no materialized factors; skip analyze", flush=True)
        return 1
    out_dir = args.lake_root / "_combo"
    summary = analyze(
        lake_root=args.lake_root,
        parquet_root=args.parquet_root,
        out_dir=out_dir,
        k=args.k,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
