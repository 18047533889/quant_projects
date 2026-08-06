#!/usr/bin/env python3
"""Append per-year industry+size double-neutral RankIC / RankICIR to the comparison CSV."""
from __future__ import annotations

import csv
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from cogalpha_lqtp.data_access_panel import resolve_factor_values_parquet  # noqa: E402
from cogalpha_lqtp.eval_extensions import _factor_cte_sql  # noqa: E402

WORK = ROOT / "data/cogalpha_lqtp_production"
CSV_PATH = WORK / "reports/factor_rankic_close_vs_vwap_vs_neutral.csv"
FWD = WORK / "lqtp_fwd_vwap_returns_cache.parquet"
IND = WORK / "eval_cache/industry_sw_l1.parquet"
MCAP = WORK / "eval_cache/market_cap.parquet"
MIN_NAMES = 30
# screening window
YEARS = list(range(2019, 2027))


def year_cols(year: int) -> tuple[str, str]:
    return (
        f"rank_ic_{year}_行业市值双中性",
        f"rank_icir_{year}_行业市值双中性",
    )


NEW_COLS = [c for y in YEARS for c in year_cols(y)]


def _fmt(x: float) -> str:
    if x != x or math.isinf(x):
        return ""
    return f"{x:.6f}"


def _mean_icir(xs: list[float]) -> tuple[float, float]:
    vals = [x for x in xs if x == x]
    if not vals:
        return float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    std = math.sqrt(var)
    icir = mean / std if std > 1e-12 else float("nan")
    return mean, icir


def _double_neutral_daily(con, fac_sql: str) -> tuple[list[int], list[float]]:
    ind = IND.as_posix().replace("'", "''")
    mcap = MCAP.as_posix().replace("'", "''")
    fwd = FWD.as_posix().replace("'", "''")
    sql = f"""
    WITH fac AS ({fac_sql}),
    joined AS (
      SELECT f.trade_date, f.symbol, f.value AS fv, g.value AS rv,
             ind.industry_code, cap.log_market_cap AS lm
      FROM fac f
      JOIN read_parquet('{fwd}') g ON f.trade_date = g.signal_date AND f.symbol = g.symbol
      JOIN read_parquet('{ind}') ind ON f.trade_date = ind.trade_date AND f.symbol = ind.symbol
      JOIN read_parquet('{mcap}') cap ON f.trade_date = cap.trade_date AND f.symbol = cap.symbol
      WHERE f.value IS NOT NULL AND g.value IS NOT NULL
    ),
    ind_neutral AS (
      SELECT trade_date, symbol, rv, lm,
             fv - avg(fv) OVER (PARTITION BY trade_date, industry_code) AS fv
      FROM joined
    ),
    stats AS (
      SELECT trade_date, regr_slope(fv, lm) AS beta, avg(fv) AS mu_f, avg(lm) AS mu_m
      FROM ind_neutral GROUP BY trade_date
    ),
    neutral AS (
      SELECT j.trade_date, j.fv - (s.mu_f + s.beta * (j.lm - s.mu_m)) AS fv, j.rv
      FROM ind_neutral j JOIN stats s USING (trade_date)
    ),
    day_n AS (SELECT trade_date, count(*) AS n FROM neutral GROUP BY 1),
    scored AS (
      SELECT m.trade_date, m.fv, m.rv,
             RANK() OVER (PARTITION BY m.trade_date ORDER BY m.fv) AS rf_min,
             COUNT(*) OVER (PARTITION BY m.trade_date, m.fv) AS fv_ties,
             RANK() OVER (PARTITION BY m.trade_date ORDER BY m.rv) AS rr_min,
             COUNT(*) OVER (PARTITION BY m.trade_date, m.rv) AS rv_ties
      FROM neutral m JOIN day_n d USING (trade_date)
      WHERE d.n >= {MIN_NAMES}
    ),
    ranked AS (
      SELECT trade_date,
             rf_min + (fv_ties - 1) * 0.5 AS rf,
             rr_min + (rv_ties - 1) * 0.5 AS rr
      FROM scored
    ),
    daily_ic AS (
      SELECT trade_date, corr(rf, rr) AS rank_ic FROM ranked GROUP BY trade_date
    )
    SELECT list(struct_pack(trade_date := trade_date, rank_ic := rank_ic) ORDER BY trade_date)
    FROM daily_ic
    """
    row = con.execute(sql).fetchone()
    if not row or not row[0]:
        return [], []
    dates = [int(r["trade_date"]) for r in row[0]]
    ics = [float(r["rank_ic"]) if r["rank_ic"] is not None else float("nan") for r in row[0]]
    return dates, ics


def compute_one(name: str) -> dict[str, float]:
    import duckdb

    out = {c: float("nan") for c in NEW_COLS}
    fp = resolve_factor_values_parquet(WORK, name)
    if fp is None or not fp.exists():
        return out
    con = duckdb.connect()
    try:
        fac = _factor_cte_sql(fp)
        dates, ics = _double_neutral_daily(con, fac)
        by_year: dict[int, list[float]] = defaultdict(list)
        for d, ic in zip(dates, ics):
            by_year[d // 10000].append(ic)
        for y in YEARS:
            ic_col, ir_col = year_cols(y)
            mean, icir = _mean_icir(by_year.get(y, []))
            out[ic_col] = mean
            out[ir_col] = icir
        return out
    finally:
        con.close()


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8-sig")))
    names = [r["factor_name"] for r in rows]
    print(f"computing yearly double-neutral RankIC/ICIR for {len(names)} factors "
          f"({YEARS[0]}–{YEARS[-1]})...", flush=True)
    t0 = time.time()
    results: dict[str, dict] = {}
    workers = min(8, max(1, len(names)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(compute_one, n): n for n in names}
        done = 0
        for fut in as_completed(futs):
            n = futs[fut]
            results[n] = fut.result()
            done += 1
            if done % 20 == 0 or done == len(names):
                print(f"  {done}/{len(names)}  elapsed={time.time()-t0:.1f}s", flush=True)

    old_fields = [c for c in rows[0].keys() if c not in NEW_COLS]
    if "备注" in old_fields:
        i = old_fields.index("备注")
        fieldnames = old_fields[:i] + NEW_COLS + old_fields[i:]
    else:
        fieldnames = old_fields + NEW_COLS

    for r in rows:
        met = results.get(r["factor_name"], {})
        for c in NEW_COLS:
            r[c] = _fmt(float(met.get(c, float("nan"))))

    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"done → {CSV_PATH}")
    for y in YEARS:
        ic_col, ir_col = year_cols(y)
        ics = [float(r[ic_col]) for r in rows if r.get(ic_col)]
        irs = [float(r[ir_col]) for r in rows if r.get(ir_col)]
        if not ics:
            print(f"  {y}: (no data)")
            continue
        print(
            f"  {y}: mean double RankIC={sum(ics)/len(ics):.4f}  "
            f"RankICIR={sum(irs)/len(irs):.4f}  n={len(ics)}"
        )


if __name__ == "__main__":
    main()
