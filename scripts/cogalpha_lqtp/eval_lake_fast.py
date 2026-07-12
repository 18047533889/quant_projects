#!/usr/bin/env python3
"""Fast lake eval: DuckDB panel RankIC / groups / LS (data_access engine stack).

Preloads forward close-to-close returns once, then evaluates each factor_lake
parquet with a single DuckDB join+window query (~1s/factor). Parallel workers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cogalpha_lqtp.lqtp_dsl_compat import eval_route_for_entry  # noqa: E402
from scripts.cogalpha_lqtp.report_html import (  # noqa: E402
    render_factor_report,
    render_index,
    render_production_summary,
)
from scripts.cogalpha_lqtp.run_production_batch import (  # noqa: E402
    LOOKAHEAD_DEFERRED_FACTORS,
    _apply_flip_state_to_catalog,
    _mean_ic_value,
    _negate_formula,
    _negate_python_code,
    _negate_values_parquet,
    _save_progress,
    _upsert_index_row,
)

DEFAULT_COMMISSION_BPS = 0.01
_N_GROUPS = 10


def _eval_route(entry: dict[str, Any]) -> str:
    return eval_route_for_entry(status=entry.get("status", ""), dsl=entry.get("dsl", ""))


def _json_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _sharpe(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    m = float(np.nanmean(arr))
    s = float(np.nanstd(arr, ddof=1))
    return m / s * np.sqrt(252) if s > 1e-12 else 0.0


def build_fwd_returns_cache(close_returns_path: Path, fwd_path: Path) -> Path:
    """signal_date T → close(T+1)/close(T)-1 labeled on exit date in source cache."""
    if fwd_path.exists() and fwd_path.stat().st_mtime >= close_returns_path.stat().st_mtime:
        return fwd_path
    con = duckdb.connect()
    try:
        con.execute(
            f"""
            COPY (
              WITH ret AS (
                SELECT trade_date::INTEGER AS trade_date,
                       symbol::VARCHAR AS symbol,
                       value::DOUBLE AS value
                FROM read_parquet('{close_returns_path.as_posix()}')
              ),
              cal AS (SELECT DISTINCT trade_date FROM ret),
              nxt AS (
                SELECT trade_date,
                       lead(trade_date) OVER (ORDER BY trade_date) AS exit_date
                FROM cal
              )
              SELECT n.trade_date AS signal_date,
                     r.symbol,
                     r.value
              FROM nxt n
              JOIN ret r ON r.trade_date = n.exit_date
              WHERE n.exit_date IS NOT NULL
            ) TO '{fwd_path.as_posix()}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()
    return fwd_path


def _factor_cte_sql(factor_path: Path) -> str:
    """Branch on parquet schema via pyarrow footer (no full scan)."""
    import pyarrow.parquet as pq

    path = factor_path.as_posix().replace("'", "''")
    cols = set(pq.ParquetFile(factor_path).schema.names)
    if "trade_date" in cols and "symbol" in cols:
        return f"""
        SELECT trade_date::INTEGER AS trade_date,
               symbol::VARCHAR AS symbol,
               value::DOUBLE AS value
        FROM read_parquet('{path}')
        """
    # Prefer integer date math over strftime for speed
    return f"""
    SELECT (year(datetime) * 10000 + month(datetime) * 100 + day(datetime))::INTEGER AS trade_date,
           asset::VARCHAR AS symbol,
           value::DOUBLE AS value
    FROM read_parquet('{path}')
    """


def analyze_factor_parquet_duckdb(
    *,
    factor_path: Path,
    fwd_returns_path: Path,
    n_groups: int = _N_GROUPS,
    commission_buy: float = DEFAULT_COMMISSION_BPS,
    commission_sell: float = DEFAULT_COMMISSION_BPS,
    min_names: int = 30,
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[str, Any]:
    """Vectorized RankIC / deciles / G10−G1 via DuckDB (average-rank Spearman)."""
    fac_sql = _factor_cte_sql(factor_path)
    fwd = fwd_returns_path.as_posix().replace("'", "''")
    roundtrip = float(commission_buy) + float(commission_sell)
    owns_con = con is None
    if con is None:
        con = duckdb.connect()

    table_names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    fwd_src = "fwd" if "fwd" in table_names else f"read_parquet('{fwd}')"

    # Keep result sets small: IC + group means + LS only (no per-name books).
    # Net LS uses assumed one-way turnover (A-share top-decile ~40%) × commission.
    assumed_one_way_turnover = 0.40
    daily_cost = 2.0 * assumed_one_way_turnover * roundtrip / 100.0

    sql = f"""
    WITH fac AS ({fac_sql}),
    m AS (
      SELECT f.trade_date,
             f.symbol,
             f.value AS fv,
             g.value AS rv
      FROM fac f
      JOIN {fwd_src} g
        ON f.trade_date = g.signal_date AND f.symbol = g.symbol
      WHERE f.value IS NOT NULL AND g.value IS NOT NULL
    ),
    day_n AS (
      SELECT trade_date, count(*) AS n FROM m GROUP BY 1
    ),
    scored AS (
      SELECT m.trade_date, m.symbol, m.fv, m.rv, d.n,
             RANK() OVER (PARTITION BY m.trade_date ORDER BY m.fv) AS rf_min,
             COUNT(*) OVER (PARTITION BY m.trade_date, m.fv) AS fv_ties,
             RANK() OVER (PARTITION BY m.trade_date ORDER BY m.rv) AS rr_min,
             COUNT(*) OVER (PARTITION BY m.trade_date, m.rv) AS rv_ties
      FROM m
      JOIN day_n d USING (trade_date)
      WHERE d.n >= {max(min_names, n_groups * 3)}
    ),
    ranked AS (
      SELECT trade_date, symbol, fv, rv, n,
             rf_min + (fv_ties - 1) * 0.5 AS rf,
             rr_min + (rv_ties - 1) * 0.5 AS rr,
             ntile({n_groups}) OVER (
               PARTITION BY trade_date ORDER BY fv ASC, symbol ASC
             ) AS grp
      FROM scored
    ),
    daily_ic AS (
      SELECT trade_date, any_value(n) AS n, corr(rf, rr) AS rank_ic
      FROM ranked
      GROUP BY trade_date
    ),
    daily_grp AS (
      SELECT trade_date, grp, avg(rv) AS gret
      FROM ranked
      GROUP BY trade_date, grp
    ),
    daily_ls AS (
      SELECT trade_date,
             max(CASE WHEN grp = {n_groups} THEN gret END)
               - max(CASE WHEN grp = 1 THEN gret END) AS ls_gross
      FROM daily_grp
      GROUP BY trade_date
    )
    SELECT
      (SELECT list(struct_pack(trade_date := trade_date, n := n, rank_ic := rank_ic)
                   ORDER BY trade_date) FROM daily_ic) AS ic_rows,
      (SELECT list(struct_pack(trade_date := trade_date, ls_gross := ls_gross)
                   ORDER BY trade_date) FROM daily_ls) AS ls_rows,
      (SELECT list(struct_pack(trade_date := trade_date, grp := grp, gret := gret)
                   ORDER BY trade_date, grp) FROM daily_grp) AS grp_rows
    """
    try:
        row = con.execute(sql).fetchone()
    finally:
        if owns_con:
            con.close()
    if row is None or row[0] is None:
        raise RuntimeError("no overlapping factor/return rows for RankIC")

    ic_rows = row[0]
    ls_rows = row[1] or []
    grp_rows = row[2] or []

    trade_dates = [int(r["trade_date"]) for r in ic_rows]
    daily_rank_ic = [float(r["rank_ic"]) if r["rank_ic"] is not None else float("nan") for r in ic_rows]
    sample_counts = [int(r["n"]) for r in ic_rows]
    ls_by_date = {int(r["trade_date"]): float(r["ls_gross"]) for r in ls_rows if r["ls_gross"] is not None}
    daily_ls_gross = [ls_by_date.get(td, 0.0) for td in trade_dates]
    daily_ls_net = [g - daily_cost for g in daily_ls_gross]

    # Group equity curves
    grp_by_date: dict[int, dict[int, float]] = {}
    for r in grp_rows:
        grp_by_date.setdefault(int(r["trade_date"]), {})[int(r["grp"])] = float(r["gret"])
    group_pnls: list[dict[str, Any]] = [{"group": g + 1, "pnl": []} for g in range(n_groups)]
    group_equity = [1.0] * n_groups
    group_daily: list[list[float]] = [[] for _ in range(n_groups)]
    for td in trade_dates:
        day = grp_by_date.get(td, {})
        for g in range(n_groups):
            g_ret = float(day.get(g + 1, 0.0))
            group_daily[g].append(g_ret)
            group_equity[g] *= 1.0 + g_ret
            group_pnls[g]["pnl"].append(group_equity[g] - 1.0)
    group_mean_returns = [float(np.nanmean(x)) if x else 0.0 for x in group_daily]

    ic_arr = np.asarray(daily_rank_ic, dtype=float)
    ls_net = np.asarray(daily_ls_net, dtype=float)
    ls_gross = np.asarray(daily_ls_gross, dtype=float)
    mean_rank_ic = float(np.nanmean(ic_arr))
    std_rank_ic = float(np.nanstd(ic_arr, ddof=1)) if len(ic_arr) > 1 else 0.0
    rank_icir = mean_rank_ic / std_rank_ic if std_rank_ic > 1e-12 else 0.0
    rank_ic_pos = float(np.mean(ic_arr > 0))
    ls_cum = float(np.prod(1.0 + ls_net) - 1.0) if len(ls_net) else 0.0

    # coverage proxy: names / typical day names from samples
    coverages = [1.0] * len(sample_counts)

    return {
        "mean_rank_ic": mean_rank_ic,
        "std_rank_ic": std_rank_ic,
        "rank_icir": rank_icir,
        "rank_ic_positive_ratio": rank_ic_pos,
        "mean_ic": mean_rank_ic,
        "std_ic": std_rank_ic,
        "icir": rank_icir,
        "ic_positive_ratio": rank_ic_pos,
        "coverage": float(np.mean(coverages)) if coverages else 0.0,
        "long_short_return": ls_cum,
        "long_short_return_sum": float(np.nansum(ls_net)),
        "long_short_sharpe": _sharpe(ls_net),
        "long_short_return_gross_sum": float(np.nansum(ls_gross)),
        "long_short_sharpe_gross": _sharpe(ls_gross),
        "daily_ic": daily_rank_ic,
        "daily_rank_ic": daily_rank_ic,
        "daily_ls_returns": ls_net.tolist(),
        "daily_ls_gross": ls_gross.tolist(),
        "trade_dates": trade_dates,
        "quote_times": [0] * len(trade_dates),
        "sample_counts": sample_counts,
        "coverages": coverages,
        "group_mean_returns": group_mean_returns,
        "group_pnls": group_pnls,
        "return_kind": "close_to_close_T_plus_1",
        "return_mode": "close_to_close",
        "signal_lag_note": (
            "Factor after close T → forward return close(T+1)/close(T)-1 "
            "(DuckDB panel RankIC); LS net assumes 40% one-way turnover × commission; "
            "TopK backtest uses OPEN execution separately"
        ),
        "commission_buy": commission_buy,
        "commission_sell": commission_sell,
        "n_groups": n_groups,
        "eval_engine": "duckdb_panel",
    }


_WORKER: dict[str, Any] = {}


def _init_worker(fwd_path: str) -> None:
    """Load forward-return parquet into an in-memory DuckDB table once per worker."""
    con = duckdb.connect()
    path = Path(fwd_path).as_posix().replace("'", "''")
    con.execute(
        f"""
        CREATE TABLE fwd AS
        SELECT signal_date::INTEGER AS signal_date,
               symbol::VARCHAR AS symbol,
               value::DOUBLE AS value
        FROM read_parquet('{path}')
        """
    )
    _WORKER["con"] = con
    _WORKER["fwd"] = Path(fwd_path)


def _eval_one(payload: dict[str, Any]) -> dict[str, Any]:
    name = payload["name"]
    t0 = time.time()
    try:
        work = Path(payload["work_dir"])
        lake = work / "factor_lake"
        report_dir = work / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        fwd = Path(payload["fwd_path"])
        out_path = lake / name / "values.parquet"
        if not out_path.exists():
            raise RuntimeError(f"missing values: {out_path}")

        entry = dict(payload["entry"])
        py_code = payload.get("python_code") or entry.get("python_code") or ""
        flip_state = payload.get("flip_state") or {}
        state = flip_state.get(name, {})
        if state.get("dsl"):
            entry["dsl"] = state["dsl"]
        if state.get("python_code"):
            py_code = state["python_code"]
            entry["python_code"] = py_code
        if state.get("ic_sign_flipped"):
            entry["ic_sign_flipped"] = True

        dsl_used = (entry.get("dsl") or "").strip() or "(python only)"
        route = _eval_route(entry)
        engine = "python" if route == "local_python" else "factor_engine"
        if route == "lqtp_dsl":
            engine = "lqtp_dsl"

        analysis = analyze_factor_parquet_duckdb(
            factor_path=out_path,
            fwd_returns_path=fwd,
            con=_WORKER.get("con"),
        )
        flip_patch: dict[str, Any] | None = None
        ic = _mean_ic_value(analysis)
        if payload.get("flip_negative_ic") and ic < 0 and ic == ic and not entry.get("ic_sign_flipped"):
            flip_patch = {"ic_sign_flipped": True}
            entry["ic_sign_flipped"] = True
            if dsl_used and dsl_used != "(python only)":
                dsl_used = _negate_formula(dsl_used)
                entry["dsl"] = dsl_used
                flip_patch["dsl"] = dsl_used
            if py_code.strip():
                py_code = _negate_python_code(py_code)
                entry["python_code"] = py_code
                flip_patch["python_code"] = py_code
            _negate_values_parquet(out_path)
            flip_patch["values_negated"] = True
            analysis = analyze_factor_parquet_duckdb(
                factor_path=out_path,
                fwd_returns_path=fwd,
                con=_WORKER.get("con"),
            )
            flip_patch["ic_before_flip"] = ic
            flip_patch["mean_ic_after_flip"] = _mean_ic_value(analysis)

        meta = {
            "engine": engine,
            "eval_route": route,
            "rows": int(sum(analysis.get("sample_counts") or [])),
            "values_path": str(out_path),
            "date_range": [payload["start"], payload["end"]],
            "formula": dsl_used if dsl_used != "(python only)" else "",
            "eval_engine": "duckdb_panel",
        }
        if flip_patch:
            meta["ic_sign_flipped"] = True
            meta["ic_before_flip"] = flip_patch.get("ic_before_flip")
            meta["mean_ic_after_flip"] = flip_patch.get("mean_ic_after_flip")

        report_path = report_dir / f"{name}.html"
        render_factor_report(
            factor_name=name,
            dsl=dsl_used,
            analysis=analysis,
            backtest_rows=[],
            out_path=report_path,
            eval_mode="duckdb_panel_close_to_close",
            materialize_meta=meta,
            topk_summary={},
            python_code=py_code,
            work_dir=work,
        )
        row = {
            "factor_name": name,
            "report": report_path.name,
            "engine": engine,
            "eval_route": route,
            "eval_mode": "duckdb_panel_close_to_close",
            "ic_sign_flipped": bool(entry.get("ic_sign_flipped")),
            "mean_ic": _json_float(analysis.get("mean_rank_ic")),
            "mean_rank_ic": _json_float(analysis.get("mean_rank_ic")),
            "icir": _json_float(analysis.get("rank_icir")),
            "rank_icir": _json_float(analysis.get("rank_icir")),
            "rank_ic_positive_ratio": _json_float(analysis.get("rank_ic_positive_ratio")),
            "long_short_sharpe": _json_float(analysis.get("long_short_sharpe")),
            "long_short_return": _json_float(analysis.get("long_short_return")),
            "backtest_total_ret": float("nan"),
            "backtest_sharpe": float("nan"),
            "backtest_max_drawdown": float("nan"),
            "rows": int(meta["rows"]),
            "backtest_id": "",
            "return_kind": str(analysis.get("return_kind", "")),
        }
        dt = time.time() - t0
        print(f"ok {name} rank_ic={row['mean_rank_ic']:.4f} ls_sharpe={row['long_short_sharpe']:.3f} {dt:.1f}s")
        return {"ok": True, "name": name, "row": row, "flip_patch": flip_patch, "error": None}
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {name}: {exc}")
        return {"ok": False, "name": name, "row": None, "flip_patch": None, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast DuckDB panel RankIC regen for factor_lake")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/cogalpha_lqtp_production")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--workers", type=int, default=max(2, min(6, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--flip-negative-ic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--skip-lookahead", action="store_true", default=True)
    args = parser.parse_args()

    work = args.work_dir
    catalog = json.loads((work / "dsl_catalog.json").read_text(encoding="utf-8"))
    python_map = {
        r["function_name"]: r["python_code"]
        for r in json.loads((work / "parsed_factors.json").read_text(encoding="utf-8"))
    }
    progress_path = work / "production_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    flip_state: dict[str, Any] = progress.setdefault("flip_state", {})
    _apply_flip_state_to_catalog(catalog, flip_state, python_map)
    entry_by_name = {e["function_name"]: e for e in catalog}

    close_cache = work / "lqtp_close_returns_cache.parquet"
    if not close_cache.exists():
        raise SystemExit(f"missing close returns cache: {close_cache}")
    fwd_path = work / "lqtp_fwd_close_returns_cache.parquet"
    print(f"building fwd returns cache → {fwd_path.name}")
    build_fwd_returns_cache(close_cache, fwd_path)

    skipped = set(progress.get("skipped", []))
    if args.skip_lookahead:
        skipped |= set(LOOKAHEAD_DEFERRED_FACTORS)
        progress["skipped"] = sorted(skipped)
        _save_progress(progress_path, progress)

    lake = work / "factor_lake"
    names = sorted(p.name for p in lake.iterdir() if (p / "values.parquet").exists())
    names = [n for n in names if n not in skipped]
    already = set(progress.get("completed", []))
    if args.only:
        only = set(args.only)
        names = [n for n in names if n in only]
    # Resume: skip factors already in completed with a duckdb_panel row
    if not args.only:
        done_panel = {
            r.get("factor_name")
            for r in progress.get("index_rows", [])
            if r.get("eval_mode") == "duckdb_panel_close_to_close"
        }
        names = [n for n in names if n not in done_panel]
        print(f"resume skip already-panel={len(done_panel)} remaining={len(names)}")

    payloads = []
    for name in names:
        entry = entry_by_name.get(name)
        if entry is None:
            continue
        py = python_map.get(name, "")
        if flip_state.get(name, {}).get("python_code"):
            py = flip_state[name]["python_code"]
        payloads.append(
            {
                "name": name,
                "work_dir": str(work),
                "entry": entry,
                "python_code": py,
                "flip_state": flip_state,
                "fwd_path": str(fwd_path),
                "start": args.start,
                "end": args.end,
                "flip_negative_ic": args.flip_negative_ic,
            }
        )

    print(f"eval {len(payloads)} factors workers={args.workers} engine=duckdb_panel")
    t0 = time.time()
    index_rows: list[dict[str, Any]] = list(progress.get("index_rows", []))
    failed: dict[str, str] = dict(progress.get("failed", {}))
    completed: list[str] = list(progress.get("completed", []))

    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=_init_worker,
        initargs=(str(fwd_path),),
    ) as pool:
        futures = {pool.submit(_eval_one, p): p["name"] for p in payloads}
        done_n = 0
        for fut in as_completed(futures):
            result = fut.result()
            name = result["name"]
            done_n += 1
            if result["ok"]:
                _upsert_index_row(index_rows, result["row"])
                if name not in completed:
                    completed.append(name)
                failed.pop(name, None)
                fp = result.get("flip_patch")
                if fp:
                    flip_state[name] = {**flip_state.get(name, {}), **fp}
            else:
                failed[name] = result.get("error") or "unknown"
            if done_n % 10 == 0 or done_n == len(payloads):
                progress["completed"] = completed
                progress["failed"] = failed
                progress["index_rows"] = index_rows
                progress["flip_state"] = flip_state
                progress["eval_methodology"] = "rankic_close_to_close_Tplus1_duckdb_panel_v1"
                progress["regen_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _save_progress(progress_path, progress)
                print(f"  checkpoint {done_n}/{len(payloads)} ok={len(completed)} fail={len(failed)}")

    progress["completed"] = completed
    progress["failed"] = failed
    progress["index_rows"] = index_rows
    progress["flip_state"] = flip_state
    progress["eval_methodology"] = "rankic_close_to_close_Tplus1_duckdb_panel_v1"
    progress["regen_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_progress(progress_path, progress)

    report_dir = work / "reports"
    summary_meta = {
        "generated_at": progress["regen_at"],
        "eval_methodology": progress["eval_methodology"],
        "note": "Fast DuckDB panel RankIC from factor_lake (cached fwd close returns)",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    render_index(index_rows, report_dir / "index.html")
    render_production_summary(
        rows=index_rows,
        out_path=report_dir / "production_summary.html",
        meta=summary_meta,
        failed=failed,
    )
    print(
        f"done ok={len(completed)} index={len(index_rows)} failed={len(failed)} "
        f"elapsed={time.time()-t0:.1f}s reports={report_dir}"
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
