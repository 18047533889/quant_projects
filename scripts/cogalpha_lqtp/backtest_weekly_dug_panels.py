#!/usr/bin/env python3
"""Local backtest for weekly-dug COS value panels (no formulas → no platform submit).

Memory-safe sequential pipeline:
  wide yearly cache → long factor_lake parquet → DuckDB RankIC/LS (+ optional extended)
  → per-factor HTML under reports_weekly_dug/
Then refresh the weekly section on the screening index (brand-neutral labels).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cogalpha_lqtp.crawl_candidate_pool_neutral_rankic import (  # noqa: E402
    patch_screening_html,
)
from scripts.cogalpha_lqtp.eval_lake_fast import (  # noqa: E402
    analyze_factor_parquet_duckdb,
    _apply_extended_eval,
)
from scripts.cogalpha_lqtp.memory_utils import (  # noqa: E402
    ensure_memory_floor,
    read_mem_available_gb,
    release_memory,
    wait_for_memory,
)
from scripts.cogalpha_lqtp.report_html import render_factor_report  # noqa: E402

DEFAULT_WORK = ROOT / "data/cogalpha_lqtp_production"
SOURCE_LABEL = {
    "evoalpha": "外部中性化候选 A",
    "alphasage": "外部中性化候选 B",
    "pool_a": "外部中性化候选 A",
    "pool_b": "外部中性化候选 B",
}


def _materialize_long(
    *,
    wide_paths: list[Path],
    out_path: Path,
    sign_flip: bool,
) -> dict[str, Any]:
    """Stream year files into a long parquet without holding all years in RAM."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    n_rows = 0
    years: list[int] = []
    try:
        for path in wide_paths:
            ensure_memory_floor(4.0)
            df = pq.read_table(path).to_pandas(date_as_object=True, ignore_metadata=True)
            if "date" not in df.columns:
                raise ValueError(f"missing date in {path}")
            long = df.melt(id_vars=["date"], var_name="symbol", value_name="value")
            del df
            long = long.dropna(subset=["value"])
            long["trade_date"] = pd.to_datetime(long["date"]).dt.strftime("%Y%m%d").astype(np.int32)
            long["symbol"] = long["symbol"].astype(str)
            long["value"] = long["value"].astype(np.float64)
            if sign_flip:
                long["value"] = -long["value"]
            table = pa.Table.from_pandas(
                long[["trade_date", "symbol", "value"]],
                preserve_index=False,
            )
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
            writer.write_table(table)
            n_rows += table.num_rows
            years.append(int(path.stem))
            del long, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    return {"n_rows": n_rows, "years": years, "path": str(out_path)}


def _wide_paths(cache: Path, source: str, factor_id: str) -> list[Path]:
    d = cache / source / factor_id
    return sorted(d.glob("*.parquet"), key=lambda p: p.stem)


def backtest_one(
    *,
    work: Path,
    row: dict[str, Any],
    fwd: Path,
    cache: Path,
    report_dir: Path,
    extended: bool,
    force: bool,
) -> dict[str, Any]:
    fid = str(row["factor_id"])
    source = str(row.get("source") or "pool_a")
    label = SOURCE_LABEL.get(source, "外部中性化候选")
    t0 = time.time()
    wait_for_memory(required_gb=6.0, reserve_gb=8.0, timeout_sec=300.0)
    paths = _wide_paths(cache, source, fid)
    if not paths:
        return {"factor_id": fid, "ok": False, "error": "missing_cache"}

    lake_path = work / "factor_lake" / fid / "values.parquet"
    sign_flip = bool(row.get("sign_flipped") or (float(row.get("mean_rank_ic") or 0) < 0))
    if force or not lake_path.exists() or lake_path.stat().st_size <= 0:
        mat = _materialize_long(wide_paths=paths, out_path=lake_path, sign_flip=sign_flip)
    else:
        mat = {"n_rows": None, "years": [int(p.stem) for p in paths], "path": str(lake_path)}

    ensure_memory_floor(4.0)
    analysis = analyze_factor_parquet_duckdb(
        factor_path=lake_path,
        fwd_returns_path=fwd,
        return_kind="vwap_to_vwap",
    )
    if extended:
        try:
            analysis = _apply_extended_eval(
                analysis,
                factor_path=lake_path,
                fwd=fwd,
                payload={
                    "work_dir": str(work),
                    "name": fid,
                    "extended_eval": True,
                },
            )
        except Exception as exc:  # noqa: BLE001
            analysis = dict(analysis)
            analysis["extended_eval_error"] = str(exc)

    mean_ic = float(analysis.get("mean_rank_ic", float("nan")))
    # if still negative somehow, flip lake once more
    if mean_ic == mean_ic and mean_ic < 0:
        # negate in place via duckdb copy
        tmp = lake_path.with_suffix(".flip.parquet")
        import duckdb

        con = duckdb.connect()
        try:
            con.execute(
                f"""
                COPY (
                  SELECT trade_date, symbol, -value AS value
                  FROM read_parquet('{lake_path.as_posix()}')
                ) TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        finally:
            con.close()
        tmp.replace(lake_path)
        sign_flip = not sign_flip
        analysis = analyze_factor_parquet_duckdb(
            factor_path=lake_path,
            fwd_returns_path=fwd,
            return_kind="vwap_to_vwap",
        )
        mean_ic = float(analysis.get("mean_rank_ic", float("nan")))

    meta = {
        "engine": "local_panel",
        "eval_route": "local_panel_values",
        "values_path": str(lake_path),
        "date_range": ["2019-01-01", "2026-06-30"],
        "formula": "",
        "eval_engine": "duckdb_panel",
        "return_kind": "vwap_to_vwap",
        "source_label": label,
        "ic_sign_flipped": sign_flip,
        "platform_submit": "skipped_no_formula",
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / f"{fid}.html"
    render_factor_report(
        factor_name=fid,
        dsl="",
        python_code="",
        analysis=analysis,
        backtest_rows=None,
        out_path=html_path,
        eval_mode="duckdb_panel_vwap_to_vwap_t1t2_realto",
        materialize_meta=meta,
        engine="local_panel",
        eval_route="local_panel_values",
        work_dir=work,
        annotation={
            "factor_id": fid,
            "function_name": fid,
            "source": label,
            "note": "面板回测（无公式，未提交平台）",
        },
    )

    out = {
        "factor_id": fid,
        "source": source,
        "source_label": label,
        "ok": True,
        "mean_rank_ic": mean_ic,
        "display_rank_ic": abs(mean_ic) if mean_ic == mean_ic else float("nan"),
        "rank_icir": float(analysis.get("rank_icir", float("nan"))),
        "display_rank_icir": abs(float(analysis.get("rank_icir") or 0.0)),
        "mean_daily_coverage": float(analysis.get("mean_daily_coverage", float("nan"))),
        "ls_mean_one_way_turnover": float(analysis.get("ls_mean_one_way_turnover", float("nan"))),
        "long_short_sharpe": float(analysis.get("long_short_sharpe", float("nan"))),
        "long_short_return_sum": float(analysis.get("long_short_return_sum", float("nan"))),
        "sign_flipped": sign_flip,
        "report_href": f"/reports_weekly_dug/{fid}.html",
        "platform_submit": "skipped_no_formula",
        "materialize": mat,
        "elapsed_sec": round(time.time() - t0, 2),
        "mem_available_gb": round(read_mem_available_gb(), 2),
    }
    # keep selected-compatible fields
    out["label"] = label
    out["abs_mean_rank_ic"] = out["display_rank_ic"]
    release_memory(analysis)
    return out


def refresh_weekly_payload(work: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    json_path = work / "reports/weekly_dug_neutral_rankic.json"
    payload = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    by_id = {r["factor_id"]: r for r in results if r.get("ok")}
    selected = []
    for r in payload.get("selected") or results:
        fid = r["factor_id"]
        if fid in by_id:
            merged = dict(r)
            merged.update(by_id[fid])
            merged["label"] = SOURCE_LABEL.get(str(merged.get("source")), "外部中性化候选")
            selected.append(merged)
        elif r.get("ok_for_select", True) and r.get("abs_mean_rank_ic", 0) > payload.get("threshold", 0.02):
            rr = dict(r)
            rr["label"] = SOURCE_LABEL.get(str(rr.get("source")), "外部中性化候选")
            selected.append(rr)
    selected.sort(key=lambda x: float(x.get("display_rank_ic") or x.get("abs_mean_rank_ic") or 0), reverse=True)
    payload["selected"] = selected
    payload["n_selected"] = len(selected)
    payload["backtested_at"] = datetime.now().isoformat()
    payload["platform_note"] = "无公式交付，无法提交平台；已全部走本地面板回测。"
    # neutralize pool labels stored in payload
    for p in payload.get("pools") or []:
        src = p.get("source")
        p["label"] = SOURCE_LABEL.get(str(src), "外部中性化候选")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N selected")
    ap.add_argument("--force", action="store_true", help="rewrite lake values")
    ap.add_argument("--no-extended", action="store_true")
    ap.add_argument("--patch-html", action="store_true", default=True)
    ap.add_argument("--no-patch-html", action="store_false", dest="patch_html")
    ap.add_argument("--min-mem-gb", type=float, default=4.0)
    args = ap.parse_args()

    work = args.work_dir
    fwd = work / "lqtp_fwd_vwap_returns_cache.parquet"
    cache = work / "candidate_pool_neutral_cache"
    report_dir = work / "reports_weekly_dug"
    sel_path = work / "reports/weekly_dug_neutral_rankic.json"
    payload = json.loads(sel_path.read_text(encoding="utf-8"))
    selected = list(payload.get("selected") or [])
    if args.limit and args.limit > 0:
        selected = selected[: args.limit]
    print(
        f"backtest n={len(selected)} extended={not args.no_extended} "
        f"mem_avail={read_mem_available_gb():.1f}G",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    for i, row in enumerate(selected, 1):
        try:
            ensure_memory_floor(args.min_mem_gb)
            out = backtest_one(
                work=work,
                row=row,
                fwd=fwd,
                cache=cache,
                report_dir=report_dir,
                extended=not args.no_extended,
                force=args.force,
            )
        except Exception as exc:  # noqa: BLE001
            out = {
                "factor_id": row.get("factor_id"),
                "ok": False,
                "error": str(exc),
                "source": row.get("source"),
                "source_label": SOURCE_LABEL.get(str(row.get("source")), "外部中性化候选"),
            }
            release_memory()
        results.append(out)
        print(
            f"[{i}/{len(selected)}] {out.get('factor_id')} ok={out.get('ok')} "
            f"IC={out.get('display_rank_ic')} sharpe={out.get('long_short_sharpe')} "
            f"sec={out.get('elapsed_sec')} mem={out.get('mem_available_gb', read_mem_available_gb()):.1f}G",
            flush=True,
        )

    summary = {
        "generated_at": datetime.now().isoformat(),
        "n": len(results),
        "n_ok": sum(1 for r in results if r.get("ok")),
        "platform_submit": "skipped_no_formula",
        "results": results,
    }
    (work / "reports/weekly_dug_backtest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    refreshed = refresh_weekly_payload(work, results)
    if args.patch_html:
        # ensure crawl module emits brand-neutral section text
        patch_screening_html(
            argparse.Namespace(work_dir=work, html=None, threshold=refreshed.get("threshold", 0.02)),
            payload=refreshed,
        )
    print(
        f"done ok={summary['n_ok']}/{summary['n']} "
        f"summary=reports/weekly_dug_backtest_summary.json",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
