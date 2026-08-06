#!/usr/bin/env python3
"""Add 2026 RankIC (raw + industry/size/double neutral) columns to the comparison CSV."""
from __future__ import annotations

import csv
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from cogalpha_lqtp.data_access_panel import resolve_factor_values_parquet  # noqa: E402
from cogalpha_lqtp.eval_extensions import (  # noqa: E402
    _compute_daily_rank_ic_series,
    _factor_cte_sql,
    _neutral_rank_ic_sql,
)

WORK = ROOT / "data/cogalpha_lqtp_production"
CSV_PATH = WORK / "reports/factor_rankic_close_vs_vwap_vs_neutral.csv"
FWD = WORK / "lqtp_fwd_vwap_returns_cache.parquet"
IND = WORK / "eval_cache/industry_sw_l1.parquet"
MCAP = WORK / "eval_cache/market_cap.parquet"
YEAR_START = 20260101
YEAR_END = 20270101
MIN_NAMES = 30

NEW_COLS = [
    "rank_ic_2026_中性化前",
    "rank_ic_2026_行业中性",
    "rank_ic_2026_市值中性",
    "rank_ic_2026_行业市值双中性",
    "变化_2026_原始到行业中性",
    "变化_2026_原始到市值中性",
    "变化_2026_原始到双中性",
    "rank_icir_2026_中性化前",
    "rank_icir_2026_行业中性",
    "rank_icir_2026_市值中性",
    "rank_icir_2026_行业市值双中性",
    "变化_rankicir_2026_原始到行业中性",
    "变化_rankicir_2026_原始到市值中性",
    "变化_rankicir_2026_原始到双中性",
    "rank_ic_2026_交易日数",
]


def _mean(xs: list[float]) -> float:
    vals = [x for x in xs if x == x]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _icir(xs: list[float]) -> float:
    vals = [x for x in xs if x == x]
    if not vals:
        return float("nan")
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    std = math.sqrt(var)
    return mean / std if std > 1e-12 else float("nan")


def _fmt(x: float) -> str:
    if x != x or math.isinf(x):
        return ""
    return f"{x:.6f}"


def _size_neutral_ics(con, fac_sql: str, fwd_src: str, mcap_path: Path) -> list[float]:
    mcap = mcap_path.as_posix().replace("'", "''")
    sql = f"""
    WITH fac AS ({fac_sql}),
    joined AS (
      SELECT f.trade_date, f.symbol, f.value AS fv, g.value AS rv, cap.log_market_cap AS lm
      FROM fac f
      JOIN {fwd_src} g ON f.trade_date = g.signal_date AND f.symbol = g.symbol
      JOIN read_parquet('{mcap}') cap ON f.trade_date = cap.trade_date AND f.symbol = cap.symbol
      WHERE f.value IS NOT NULL AND g.value IS NOT NULL AND cap.log_market_cap IS NOT NULL
    ),
    stats AS (
      SELECT trade_date, regr_slope(fv, lm) AS beta, avg(fv) AS mu_f, avg(lm) AS mu_m
      FROM joined GROUP BY trade_date
    ),
    neutral AS (
      SELECT j.trade_date, j.fv - (s.mu_f + s.beta * (j.lm - s.mu_m)) AS fv, j.rv
      FROM joined j JOIN stats s USING (trade_date)
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
        return []
    return [float(r["rank_ic"]) if r["rank_ic"] is not None else float("nan") for r in row[0]]


def _double_neutral_ics(con, fac_sql: str, fwd_src: str, ind_path: Path, mcap_path: Path) -> list[float]:
    ind = ind_path.as_posix().replace("'", "''")
    mcap = mcap_path.as_posix().replace("'", "''")
    sql = f"""
    WITH fac AS ({fac_sql}),
    joined AS (
      SELECT f.trade_date, f.symbol, f.value AS fv, g.value AS rv,
             ind.industry_code, cap.log_market_cap AS lm
      FROM fac f
      JOIN {fwd_src} g ON f.trade_date = g.signal_date AND f.symbol = g.symbol
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
        return []
    return [float(r["rank_ic"]) if r["rank_ic"] is not None else float("nan") for r in row[0]]


def compute_one(name: str) -> dict[str, float | int]:
    import duckdb

    out = {c: float("nan") for c in NEW_COLS}
    out["rank_ic_2026_交易日数"] = 0
    fp = resolve_factor_values_parquet(WORK, name)
    if fp is None or not fp.exists():
        return out
    if not FWD.exists() or not IND.exists() or not MCAP.exists():
        raise FileNotFoundError("missing fwd/industry/mcap cache")

    con = duckdb.connect()
    try:
        base = _factor_cte_sql(fp)
        fac = (
            f"SELECT * FROM ({base}) "
            f"WHERE trade_date >= {YEAR_START} AND trade_date < {YEAR_END}"
        )
        fwd_src = f"read_parquet('{FWD.as_posix()}')"
        raw_sql = _neutral_rank_ic_sql(
            fac_sql=fac, fwd_src=fwd_src, aux_join="", fv_expr="f.value", min_names=MIN_NAMES
        )
        dates, raw_ic = _compute_daily_rank_ic_series(con, raw_sql)
        if not dates:
            return out

        ind_join = (
            f"JOIN read_parquet('{IND.as_posix()}') ind "
            "ON f.trade_date = ind.trade_date AND f.symbol = ind.symbol"
        )
        ind_sql = _neutral_rank_ic_sql(
            fac_sql=fac,
            fwd_src=fwd_src,
            aux_join=ind_join,
            fv_expr="f.value - avg(f.value) OVER (PARTITION BY f.trade_date, ind.industry_code)",
            min_names=MIN_NAMES,
        )
        _, ind_ic = _compute_daily_rank_ic_series(con, ind_sql)
        size_ic = _size_neutral_ics(con, fac, fwd_src, MCAP)
        both_ic = _double_neutral_ics(con, fac, fwd_src, IND, MCAP)

        raw_m = _mean(raw_ic)
        ind_m = _mean(ind_ic)
        size_m = _mean(size_ic)
        both_m = _mean(both_ic)
        raw_ir = _icir(raw_ic)
        ind_ir = _icir(ind_ic)
        size_ir = _icir(size_ic)
        both_ir = _icir(both_ic)
        out["rank_ic_2026_中性化前"] = raw_m
        out["rank_ic_2026_行业中性"] = ind_m
        out["rank_ic_2026_市值中性"] = size_m
        out["rank_ic_2026_行业市值双中性"] = both_m
        out["变化_2026_原始到行业中性"] = ind_m - raw_m
        out["变化_2026_原始到市值中性"] = size_m - raw_m
        out["变化_2026_原始到双中性"] = both_m - raw_m
        out["rank_icir_2026_中性化前"] = raw_ir
        out["rank_icir_2026_行业中性"] = ind_ir
        out["rank_icir_2026_市值中性"] = size_ir
        out["rank_icir_2026_行业市值双中性"] = both_ir
        out["变化_rankicir_2026_原始到行业中性"] = ind_ir - raw_ir
        out["变化_rankicir_2026_原始到市值中性"] = size_ir - raw_ir
        out["变化_rankicir_2026_原始到双中性"] = both_ir - raw_ir
        out["rank_ic_2026_交易日数"] = int(len(dates))
        return out
    finally:
        con.close()


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8-sig")))
    names = [r["factor_name"] for r in rows]
    print(f"computing 2026 RankIC for {len(names)} factors...", flush=True)
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

    # rebuild fieldnames: keep old order, append new cols (replace if already present)
    old_fields = [c for c in rows[0].keys() if c not in NEW_COLS]
    # insert before 备注 if present
    if "备注" in old_fields:
        i = old_fields.index("备注")
        fieldnames = old_fields[:i] + NEW_COLS + old_fields[i:]
    else:
        fieldnames = old_fields + NEW_COLS

    for r in rows:
        met = results.get(r["factor_name"], {})
        for c in NEW_COLS:
            val = met.get(c, float("nan"))
            r[c] = str(int(val)) if c == "rank_ic_2026_交易日数" else _fmt(float(val))

    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # quick summary
    raw = [float(r["rank_ic_2026_中性化前"]) for r in rows if r["rank_ic_2026_中性化前"]]
    both = [float(r["rank_ic_2026_行业市值双中性"]) for r in rows if r["rank_ic_2026_行业市值双中性"]]
    dlt = [float(r["变化_2026_原始到双中性"]) for r in rows if r["变化_2026_原始到双中性"]]
    raw_ir = [float(r["rank_icir_2026_中性化前"]) for r in rows if r["rank_icir_2026_中性化前"]]
    both_ir = [float(r["rank_icir_2026_行业市值双中性"]) for r in rows if r["rank_icir_2026_行业市值双中性"]]
    dlt_ir = [float(r["变化_rankicir_2026_原始到双中性"]) for r in rows if r["变化_rankicir_2026_原始到双中性"]]
    print(
        f"done → {CSV_PATH}\n"
        f"  mean 2026 RankIC  raw={sum(raw)/len(raw):.4f}  double={sum(both)/len(both):.4f}  "
        f"Δ={sum(dlt)/len(dlt):.4f}  pctΔ>0={sum(1 for x in dlt if x>0)/len(dlt):.1%}\n"
        f"  mean 2026 RankICIR raw={sum(raw_ir)/len(raw_ir):.4f}  double={sum(both_ir)/len(both_ir):.4f}  "
        f"Δ={sum(dlt_ir)/len(dlt_ir):.4f}  pctΔ>0={sum(1 for x in dlt_ir if x>0)/len(dlt_ir):.1%}"
    )


if __name__ == "__main__":
    main()
