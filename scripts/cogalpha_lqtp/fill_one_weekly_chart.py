#!/usr/bin/env python3
"""Ultra-low-memory: fill charts for ONE weekly-dug factor then exit.

Routes:
  1) optional COS cache melt (usually skipped --skip-cache)
  2) DSL materialize HALF-YEAR chunks in child processes (FE never accumulates)
  3) DuckDB concat + analyze + render (no extended extras)

One factor = one invocation. Do NOT batch-parallelize.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "cogalpha_lqtp")]

from scripts.cogalpha_lqtp.backtest_weekly_dug_panels import _materialize_long  # noqa: E402
from scripts.cogalpha_lqtp.eval_lake_fast import analyze_factor_parquet_duckdb  # noqa: E402
from scripts.cogalpha_lqtp.memory_utils import (  # noqa: E402
    ensure_memory_floor,
    read_mem_available_gb,
    release_memory,
    wait_for_memory,
)
from scripts.cogalpha_lqtp.report_html import render_factor_report  # noqa: E402

WORK = ROOT / "data/cogalpha_lqtp_production"
CHUNK_SCRIPT = ROOT / "scripts/cogalpha_lqtp/materialize_halfyear_chunk.py"


def display_factor_name(row: dict[str, Any]) -> str:
    """Strip dig-date prefixes from display names."""
    raw = str(row.get("display_name") or row.get("factor_id") or row.get("name") or "").strip()
    m = re.match(r"^(alpha|ext|cand|alphasage)_(?:\d{8,14}_)?(.+)$", raw, re.I)
    if m:
        return f"{m.group(1).lower()}_{m.group(2)}"
    return raw


def _has_charts(html: str) -> bool:
    return (
        "data:image/png;base64," in html
        and 'alt="Daily RankIC"' in html
        and ("Decile cumulative" in html or "Group PnL" in html)
        and ("Cumulative long-short" in html or "Cumulative LS" in html)
    )


def _flip_lake(lake: Path) -> None:
    import duckdb

    tmp = lake.with_suffix(".flip.parquet")
    con = duckdb.connect()
    try:
        con.execute(
            f"""
            COPY (
              SELECT trade_date, symbol, -value AS value
              FROM read_parquet('{lake.as_posix()}')
            ) TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        con.close()
    tmp.replace(lake)


def _find_cache(work: Path, fid: str) -> list[Path]:
    h = re.search(r"([0-9a-f]{8})$", fid, re.I)
    if not h:
        return []
    hx = h.group(1).lower()
    root = work / "candidate_pool_neutral_cache"
    prefer = ["evoalpha", "alphasage", "pool_a", "pool_b", "evoalpha_raw", "alphasage_raw"]
    hits: list[Path] = []
    ordered = [root / s for s in prefer if (root / s).is_dir()]
    ordered += [p for p in root.iterdir() if p.is_dir() and p not in ordered] if root.exists() else []
    for src in ordered:
        for d in src.iterdir():
            if d.is_dir() and hx in d.name and any(d.glob("*.parquet")):
                hits.append(d)
    return hits


def _abs(x: Any) -> float:
    try:
        v = abs(float(x))
        return v if v == v else 0.0
    except (TypeError, ValueError):
        return 0.0


def _has_cs_rank(dsl: str) -> bool:
    """True if DSL uses cross-sectional rank( (not ts_rank)."""
    d = re.sub(r"ts_rank\s*\(", "", str(dsl).lower())
    return bool(re.search(r"(?<![a-z_])rank\s*\(", d))


def _year_windows() -> list[tuple[str, str, str]]:
    """Calendar-year windows snapped to local trading days (avoid holiday COS holes)."""
    bar = ROOT / "data/a_share/lqtp_data/StockDailyBar"
    by_year: dict[int, list[str]] = {}
    if bar.is_dir():
        for p in bar.glob("*.parquet"):
            if p.name.endswith(".manifest.json"):
                continue
            try:
                y = int(p.stem[:4])
            except ValueError:
                continue
            by_year.setdefault(y, []).append(p.stem)
    out: list[tuple[str, str, str]] = []
    for y in range(2019, 2027):
        days = sorted(by_year.get(y) or [])
        if days:
            start, end = days[0], days[-1]
            if y == 2026:
                end = min(end, "2026-06-30")
            out.append((f"{y}", start, end))
        else:
            end = f"{y}-12-31" if y < 2026 else "2026-06-30"
            out.append((f"{y}", f"{y}-01-02", end))
    return out


def materialize_dsl_by_halfyear(work: Path, dname: str, dsl: str, *, min_avail_gb: float) -> Path:
    """Each year runs in a fresh child process so FE RAM returns to the OS."""
    lake_dir = work / "factor_lake" / dname
    lake_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = lake_dir / "_year_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_files: list[Path] = []
    env = os.environ.copy()
    env.setdefault("MALLOC_ARENA_MAX", "2")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("FACTOR_ENGINE_MAX_WORKERS", "1")
    env.setdefault("FACTOR_ENGINE_DISABLE_CSE", "1")
    env.setdefault("FACTOR_ENGINE_DISABLE_PANEL_NATIVE", "1")
    env.setdefault("FACTOR_ENGINE_RESERVE_GB", "8")
    env.setdefault("FACTOR_ENGINE_MAX_MEMORY_MB", "12000")
    env.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    env.setdefault("FACTOR_ENGINE_OPERATOR_BACKEND", "polars")
    env.setdefault(
        "FACTOR_ENGINE_SPILL_DIR",
        str(work / "_fe_spill"),
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "scripts" / "cogalpha_lqtp"), str(ROOT / "ashare_lqtp_kit" / "protos")]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )

    for tag, start, end in _year_windows():
        part = parts_dir / f"{tag}.parquet"
        if part.exists() and part.stat().st_size > 1000:
            print(f"  {tag} reuse size={part.stat().st_size} avail={read_mem_available_gb():.1f}G", flush=True)
            part_files.append(part)
            continue

        # Wait for comfortable headroom before spawning FE.
        wait_for_memory(required_gb=min_avail_gb, reserve_gb=0.0, timeout_sec=1800)
        avail = read_mem_available_gb()
        if avail < min_avail_gb:
            raise MemoryError(f"abort before {tag}: avail={avail:.1f}G < {min_avail_gb:.1f}G")
        print(f"  {tag} spawn {start}->{end} avail={avail:.1f}G", flush=True)
        cmd = [
            sys.executable,
            str(CHUNK_SCRIPT),
            "--name",
            dname,
            "--dsl",
            dsl,
            "--start",
            start,
            "--end",
            end,
            "--out",
            str(part),
            "--work-dir",
            str(work),
            "--min-avail-gb",
            str(min_avail_gb),
        ]
        # Stream child output; kill if host free RAM collapses mid-run.
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"    {line.rstrip()}", flush=True)
            # soft guard: if free RAM tanked, stop this factor (leave parts for resume)
            if read_mem_available_gb() < 10.0:
                print(f"  ABORT_MEM_GUARD avail={read_mem_available_gb():.1f}G kill child", flush=True)
                proc.kill()
                proc.wait(timeout=30)
                raise MemoryError(f"mem guard tripped during {tag}")
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"chunk {tag} failed rc={rc}")
        if not part.exists() or part.stat().st_size < 1000:
            raise RuntimeError(f"chunk {tag} missing/empty output")
        part_files.append(part)
        # let OS reclaim child pages
        time.sleep(2)
        release_memory()
        gc.collect()
        print(f"  {tag} ok size={part.stat().st_size} avail={read_mem_available_gb():.1f}G", flush=True)

    import duckdb

    lake = lake_dir / "values.parquet"
    wait_for_memory(required_gb=12.0, reserve_gb=0.0, timeout_sec=600)
    glob = (parts_dir / "*.parquet").as_posix()
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.execute("SET memory_limit='1GB'")
        cols = {c.lower() for c in pq.read_schema(part_files[0]).names}
        if {"datetime", "asset", "value"}.issubset(cols):
            sql = f"""
            COPY (
              SELECT
                CAST(strftime(CAST(datetime AS DATE), '%Y%m%d') AS INTEGER) AS trade_date,
                CAST(asset AS VARCHAR) AS symbol,
                CAST(value AS DOUBLE) AS value
              FROM read_parquet('{glob}')
              WHERE value IS NOT NULL
            ) TO '{lake.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        elif {"trade_date", "symbol", "value"}.issubset(cols):
            sql = f"""
            COPY (
              SELECT
                CAST(replace(CAST(trade_date AS VARCHAR), '-', '') AS INTEGER) AS trade_date,
                CAST(symbol AS VARCHAR) AS symbol,
                CAST(value AS DOUBLE) AS value
              FROM read_parquet('{glob}')
              WHERE value IS NOT NULL
            ) TO '{lake.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        else:
            raise RuntimeError(f"unsupported half-part schema: {sorted(cols)}")
        con.execute(sql)
    finally:
        con.close()
    release_memory()
    gc.collect()
    return lake


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="display_name e.g. ext_abe447a6")
    ap.add_argument("--work-dir", type=Path, default=WORK)
    ap.add_argument("--cache-only", action="store_true", help="fail if no usable cache")
    ap.add_argument("--skip-cache", action="store_true", help="skip COS cache melt (go DSL year chunks)")
    ap.add_argument(
        "--allow-weak-cache",
        action="store_true",
        help="use COS cache even if IC << target (charts only; may be wrong panel)",
    )
    ap.add_argument(
        "--allow-cs-rank",
        action="store_true",
        help="allow FE materialize for cross-sectional rank( (high RAM; default refuse)",
    )
    ap.add_argument(
        "--min-avail-gb",
        type=float,
        default=18.0,
        help="refuse to spawn FE chunk unless MemAvailable >= this (default 18)",
    )
    args = ap.parse_args()
    work = args.work_dir
    name = args.name.strip()
    weekly = json.loads((work / "reports/weekly_dug_neutral_rankic.json").read_text())
    row = None
    for r in weekly.get("selected") or []:
        if display_factor_name(r) == name or str(r.get("factor_id")) == name:
            row = r
            break
    if row is None:
        print(f"NOT_FOUND {name}", flush=True)
        return 2

    dname = display_factor_name(row)
    out = work / "reports_weekly_dug" / f"{dname}.html"
    if out.exists() and _has_charts(out.read_text(errors="ignore")):
        print(f"SKIP_ALREADY {dname}", flush=True)
        return 0

    print(f"START {dname} mem={read_mem_available_gb():.1f}G", flush=True)
    fwd = work / "lqtp_fwd_vwap_returns_cache.parquet"
    lake = work / "factor_lake" / dname / "values.parquet"
    target_ic = _abs(row.get("display_rank_ic") or row.get("platform_mean_ic") or row.get("mean_rank_ic"))
    route = ""

    # --- cache path ---
    if not args.skip_cache:
        caches = _find_cache(work, str(row.get("factor_id") or dname))
        for cdir in caches[:1]:
            print(f"TRY_CACHE {cdir} mem={read_mem_available_gb():.1f}G", flush=True)
            wait_for_memory(required_gb=5.0, reserve_gb=12.0, timeout_sec=1800)
            ensure_memory_floor(8.0)
            wide = sorted(cdir.glob("*.parquet"), key=lambda p: p.stem)
            _materialize_long(wide_paths=wide, out_path=lake, sign_flip=False)
            release_memory()
            gc.collect()
            analysis = analyze_factor_parquet_duckdb(
                factor_path=lake, fwd_returns_path=fwd, return_kind="vwap_to_vwap"
            )
            ic = float(analysis.get("mean_rank_ic") or 0.0)
            release_memory(analysis)
            gc.collect()
            min_ok = max(0.012, target_ic * 0.55) if target_ic > 0 else 0.012
            if args.allow_weak_cache:
                min_ok = 0.005
            if ic == ic and abs(ic) >= min_ok:
                if ic < 0:
                    _flip_lake(lake)
                route = f"cache:{cdir.parent.name}/{cdir.name}"
                break
            print(f"CACHE_WEAK ic={ic:.4f} target={target_ic:.4f}", flush=True)
            route = ""
            try:
                lake.unlink(missing_ok=True)
            except OSError:
                pass
    else:
        print("SKIP_CACHE", flush=True)

    if not route:
        if args.cache_only:
            print("FAIL cache-only no usable cache", flush=True)
            return 3
        dsl = str(row.get("lqtp_formula") or row.get("dsl") or "").strip()
        if not dsl:
            print("FAIL no dsl", flush=True)
            return 4
        if _has_cs_rank(dsl) and not args.allow_cs_rank:
            print(
                "SKIP_CS_RANK: cross-sectional rank() FE needs ~20GB+ even for 1 month; "
                "refuse by default. Re-run with --allow-cs-rank after FE spill is proven.",
                flush=True,
            )
            return 5
        print(f"TRY_DSL_YEARLY mem={read_mem_available_gb():.1f}G", flush=True)
        lake = materialize_dsl_by_halfyear(
            work, dname, dsl, min_avail_gb=float(args.min_avail_gb)
        )
        route = "dsl_yearly"
        analysis = analyze_factor_parquet_duckdb(
            factor_path=lake, fwd_returns_path=fwd, return_kind="vwap_to_vwap"
        )
        ic = float(analysis.get("mean_rank_ic") or 0.0)
        release_memory(analysis)
        if ic == ic and ic < 0:
            _flip_lake(lake)

    # --- analyze + render (no extended) ---
    wait_for_memory(required_gb=5.0, reserve_gb=12.0, timeout_sec=1800)
    ensure_memory_floor(8.0)
    analysis = analyze_factor_parquet_duckdb(
        factor_path=lake, fwd_returns_path=fwd, return_kind="vwap_to_vwap"
    )
    mic = float(analysis.get("mean_rank_ic") or 0.0)
    if mic == mic and mic < 0:
        analysis["mean_rank_ic"] = abs(mic)
        if analysis.get("rank_icir") is not None:
            analysis["rank_icir"] = abs(float(analysis["rank_icir"]))

    dsl = str(row.get("lqtp_formula") or row.get("dsl") or "").strip()
    label = str(row.get("label") or row.get("source_label") or "本周新挖")
    meta = {
        "engine": "local_dsl" if dsl else "local_panel",
        "eval_route": "local_panel_values",
        "values_path": str(lake),
        "date_range": ["2019-01-01", "2026-06-30"],
        "formula": dsl,
        "eval_engine": "duckdb_panel",
        "return_kind": "vwap_to_vwap",
        "source_label": label,
        "platform_submit": str(row.get("platform_submit") or "submitted"),
        "chart_route": route,
    }
    render_factor_report(
        factor_name=dname,
        dsl=dsl,
        python_code="",
        analysis=analysis,
        backtest_rows=None,
        out_path=out,
        eval_mode="duckdb_panel_vwap_to_vwap_t1t2_realto",
        materialize_meta=meta,
        engine="local_dsl" if dsl else "local_panel",
        eval_route="local_panel_values",
        work_dir=work,
        annotation={
            "factor_id": dname,
            "function_name": dname,
            "source": label,
            "formula_display": dsl,
            "note": "面板回测（含 DSL）",
        },
    )
    text = out.read_text(encoding="utf-8", errors="ignore")
    for brand in ("evoalpha", "EvoAlpha", "alphasage", "AlphaSage", "cogalpha", "CogAlpha", "lizhuo"):
        text = text.replace(brand, "cand")
    out.write_text(text, encoding="utf-8")
    ok = _has_charts(text)
    print(
        f"DONE {dname} ok={ok} route={route} ic={analysis.get('mean_rank_ic')} "
        f"size={out.stat().st_size} mem={read_mem_available_gb():.1f}G",
        flush=True,
    )
    release_memory(analysis)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
