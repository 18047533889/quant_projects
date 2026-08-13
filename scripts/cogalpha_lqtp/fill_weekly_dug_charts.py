#!/usr/bin/env python3
"""Fill missing RankIC / decile / LS charts for weekly-dug HTML reports.

Memory-safe sequential pipeline:
  resolve COS wide-cache OR materialize DSL → long lake parquet → DuckDB analyze
  (+ extended metrics) → render_factor_report under reports_weekly_dug/

Does NOT drop soft-gate (|IC|<0.02) rows from weekly JSON.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "cogalpha_lqtp")]

from scripts.cogalpha_lqtp.backtest_weekly_dug_panels import _materialize_long  # noqa: E402
from scripts.cogalpha_lqtp.crawl_candidate_pool_neutral_rankic import (  # noqa: E402
    display_factor_name,
    patch_screening_html,
)
from scripts.cogalpha_lqtp.data_access_panel import ashare_materialize_data_source_config  # noqa: E402
from scripts.cogalpha_lqtp.eval_extensions import (  # noqa: E402
    compute_extended_eval,
    ensure_eval_aux_cache,
    merge_extended_into_analysis,
)
from scripts.cogalpha_lqtp.eval_lake_fast import analyze_factor_parquet_duckdb  # noqa: E402
from scripts.cogalpha_lqtp.materialize import materialize_factor  # noqa: E402
from scripts.cogalpha_lqtp.memory_utils import (  # noqa: E402
    ensure_memory_floor,
    read_mem_available_gb,
    release_memory,
    wait_for_memory,
)
from scripts.cogalpha_lqtp.report_html import render_factor_report  # noqa: E402

WORK = ROOT / "data/cogalpha_lqtp_production"


def _has_charts(html: str) -> bool:
    return (
        "data:image/png;base64," in html
        and 'alt="Daily RankIC"' in html
        and ("alt=\"Decile cumulative return\"" in html or 'alt="Group PnL"' in html)
        and ("alt=\"Cumulative long-short\"" in html or 'alt="Cumulative LS"' in html)
    )


def _hash(fid: str) -> str:
    m = re.search(r"([0-9a-f]{8})$", str(fid), re.I)
    return m.group(1).lower() if m else ""


def _find_cache_dirs(cache_root: Path, fid: str) -> list[Path]:
    h = _hash(fid)
    if not h or not cache_root.exists():
        return []
    hits: list[Path] = []
    prefer = ["evoalpha", "alphasage", "pool_a", "pool_b", "evoalpha_raw", "alphasage_raw"]
    ordered = [cache_root / s for s in prefer if (cache_root / s).is_dir()]
    ordered += [p for p in cache_root.iterdir() if p.is_dir() and p not in ordered]
    for src in ordered:
        for d in src.iterdir():
            if d.is_dir() and h in d.name and list(d.glob("*.parquet")):
                hits.append(d)
    return hits


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


def _abs(x: Any) -> float:
    try:
        v = abs(float(x))
        return v if v == v else 0.0
    except (TypeError, ValueError):
        return 0.0


def ensure_lake(
    *,
    work: Path,
    row: dict[str, Any],
    dname: str,
    start: str,
    end: str,
    prefer_cache: bool,
) -> tuple[Path, str]:
    """Return (lake_path, route)."""
    lake = work / "factor_lake" / dname / "values.parquet"
    target_ic = _abs(row.get("display_rank_ic") or row.get("platform_mean_ic") or row.get("mean_rank_ic"))
    fwd = work / "lqtp_fwd_vwap_returns_cache.parquet"

    if prefer_cache:
        caches = _find_cache_dirs(work / "candidate_pool_neutral_cache", str(row.get("factor_id") or dname))
        # try at most 2 cache dirs to save RAM/time; fall back to DSL if IC mismatch
        for cdir in caches[:2]:
            wait_for_memory(required_gb=6.0, reserve_gb=10.0, timeout_sec=900)
            ensure_memory_floor(8.0)
            wide = sorted(cdir.glob("*.parquet"), key=lambda p: p.stem)
            _materialize_long(wide_paths=wide, out_path=lake, sign_flip=False)
            release_memory()
            analysis = analyze_factor_parquet_duckdb(
                factor_path=lake, fwd_returns_path=fwd, return_kind="vwap_to_vwap"
            )
            ic = float(analysis.get("mean_rank_ic") or 0.0)
            release_memory(analysis)
            min_ok = max(0.012, target_ic * 0.55) if target_ic > 0 else 0.012
            if ic == ic and abs(ic) >= min_ok:
                if ic < 0:
                    _flip_lake(lake)
                return lake, f"cache:{cdir.parent.name}/{cdir.name}"
            print(f"  cache weak ic={ic:.4f} vs target={target_ic:.4f} ({cdir.name})", flush=True)

    dsl = str(row.get("lqtp_formula") or row.get("dsl") or "").strip()
    if not dsl:
        raise RuntimeError(f"no dsl and no usable cache for {dname}")
    wait_for_memory(required_gb=10.0, reserve_gb=10.0, timeout_sec=1200)
    ensure_memory_floor(10.0)
    cfg = ashare_materialize_data_source_config(start_date=start, end_date=end)
    # rank()/cs ops need universe=ALL on full-market panels
    path = materialize_factor(
        factor_id=dname,
        dsl=dsl,
        data_source_cfg=cfg,
        lake_root=work / "factor_lake",
        universe="ALL",
    )
    # drop FE caches/stores before analyze
    release_memory()
    gc.collect()
    # ensure positive IC orientation for display
    analysis = analyze_factor_parquet_duckdb(
        factor_path=path, fwd_returns_path=fwd, return_kind="vwap_to_vwap"
    )
    ic = float(analysis.get("mean_rank_ic") or 0.0)
    release_memory(analysis)
    if ic == ic and ic < 0:
        _flip_lake(path)
    return path, "dsl_materialize"


def render_one(
    *,
    work: Path,
    row: dict[str, Any],
    lake: Path,
    ind: Path | None,
    mcap: Path | None,
    extended: bool = True,
) -> dict[str, Any]:
    dname = display_factor_name(row)
    dsl = str(row.get("lqtp_formula") or row.get("dsl") or "").strip()
    fwd = work / "lqtp_fwd_vwap_returns_cache.parquet"
    wait_for_memory(required_gb=6.0, reserve_gb=10.0, timeout_sec=600)
    ensure_memory_floor(6.0)
    analysis = analyze_factor_parquet_duckdb(
        factor_path=lake, fwd_returns_path=fwd, return_kind="vwap_to_vwap"
    )
    if extended:
        try:
            extended_res = compute_extended_eval(
                factor_path=lake,
                fwd_returns_path=fwd,
                industry_path=ind if ind and ind.exists() else None,
                market_cap_path=mcap if mcap and mcap.exists() else None,
            )
            analysis = merge_extended_into_analysis(analysis, extended_res)
        except Exception as exc:  # noqa: BLE001
            analysis = dict(analysis)
            analysis["extended_eval_error"] = str(exc)

    # keep display IC positive
    mic = float(analysis.get("mean_rank_ic") or 0.0)
    if mic == mic and mic < 0:
        analysis["mean_rank_ic"] = abs(mic)
        if analysis.get("rank_icir") is not None:
            analysis["rank_icir"] = abs(float(analysis["rank_icir"]))

    out = work / "reports_weekly_dug" / f"{dname}.html"
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
            "note": "面板回测（含 DSL）" if dsl else "面板回测",
        },
    )
    # scrub brands in display text only
    text = out.read_text(encoding="utf-8", errors="ignore")
    for brand in ("evoalpha", "EvoAlpha", "alphasage", "AlphaSage", "cogalpha", "CogAlpha", "lizhuo"):
        text = text.replace(brand, "cand")
    out.write_text(text, encoding="utf-8")

    patch = {
        "display_name": dname,
        "report_href": f"reports_weekly_dug/{dname}.html",
        "mean_daily_coverage": analysis.get("mean_daily_coverage"),
        "ls_mean_one_way_turnover": analysis.get("ls_mean_one_way_turnover"),
        "ls_mean_daily_cost": analysis.get("ls_mean_daily_cost"),
        "long_short_return_sum": analysis.get("long_short_return")
        or analysis.get("long_short_return_sum"),
        "charts_filled_at": datetime.now(timezone.utc).isoformat(),
        "charts_ok": _has_charts(text),
        "panel_mean_rank_ic": analysis.get("mean_rank_ic"),
        "panel_rank_icir": analysis.get("rank_icir"),
        "panel_long_short_sharpe": analysis.get("long_short_sharpe"),
    }
    release_memory(analysis)
    return patch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path, default=WORK)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--prefer-cache", action="store_true", default=True)
    ap.add_argument("--no-prefer-cache", action="store_false", dest="prefer_cache")
    ap.add_argument("--force", action="store_true", help="rebuild even if charts exist")
    ap.add_argument("--min-mem-gb", type=float, default=8.0)
    ap.add_argument(
        "--no-extended",
        action="store_true",
        help="skip rolling-IC / heatmap extras (saves RAM; still renders RankIC/decile/LS)",
    )
    args = ap.parse_args()

    work = args.work_dir
    weekly_path = work / "reports/weekly_dug_neutral_rankic.json"
    weekly = json.loads(weekly_path.read_text(encoding="utf-8"))
    selected = list(weekly.get("selected") or [])
    only = set(args.only) if args.only else None

    need: list[dict[str, Any]] = []
    for r in selected:
        dname = display_factor_name(r)
        if only and dname not in only and str(r.get("factor_id")) not in only:
            continue
        html = work / "reports_weekly_dug" / f"{dname}.html"
        if html.exists() and not args.force and _has_charts(html.read_text(encoding="utf-8", errors="ignore")):
            continue
        need.append(r)
    if args.limit and args.limit > 0:
        need = need[: args.limit]

    print(
        f"fill charts n={len(need)} mem={read_mem_available_gb():.1f}G prefer_cache={args.prefer_cache}",
        flush=True,
    )
    print("ensure aux cache ...", flush=True)
    aux = ensure_eval_aux_cache(work, start=args.start, end=args.end)
    ind = Path(aux["industry"]) if aux.get("industry") else None
    mcap = Path(aux["market_cap"]) if aux.get("market_cap") else None

    results: list[dict[str, Any]] = []
    by_name = {display_factor_name(r): r for r in selected}
    for i, row in enumerate(need, 1):
        dname = display_factor_name(row)
        t0 = time.time()
        print(f"[{i}/{len(need)}] {dname} mem={read_mem_available_gb():.1f}G", flush=True)
        try:
            ensure_memory_floor(args.min_mem_gb)
            lake, route = ensure_lake(
                work=work,
                row=row,
                dname=dname,
                start=args.start,
                end=args.end,
                prefer_cache=args.prefer_cache,
            )
            print(f"  lake via {route} -> {lake}", flush=True)
            patch = render_one(
                work=work,
                row=row,
                lake=lake,
                ind=ind,
                mcap=mcap,
                extended=not args.no_extended,
            )
            # merge into selected row
            tgt = by_name.get(dname) or row
            tgt.update(patch)
            tgt["display_name"] = dname
            results.append({"name": dname, "ok": True, "route": route, "sec": round(time.time() - t0, 1)})
            print(
                f"  OK charts={patch.get('charts_ok')} sec={results[-1]['sec']} "
                f"mem={read_mem_available_gb():.1f}G",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"name": dname, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAIL {exc}", flush=True)
        finally:
            release_memory()
            gc.collect()

    weekly["selected"] = selected
    weekly["n_selected"] = len(selected)
    weekly["charts_fill"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_attempted": len(need),
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_fail": sum(1 for r in results if not r.get("ok")),
        "results": results,
    }
    weekly_path.write_text(json.dumps(weekly, ensure_ascii=False, indent=2), encoding="utf-8")
    (work / "reports/weekly_dug_charts_fill.json").write_text(
        json.dumps(weekly["charts_fill"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    class A:
        work_dir = work
        html = work / "reports/factor_rankic_screening_index.html"
        threshold = float(weekly.get("threshold") or 0.02)

    patch_screening_html(A(), payload=weekly)
    print("done", weekly["charts_fill"]["n_ok"], "/", weekly["charts_fill"]["n_attempted"], flush=True)
    return 0 if weekly["charts_fill"]["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
