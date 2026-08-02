#!/usr/bin/env python3
"""Extended factor evaluation: neutral IC, IC heatmap, half-life, autocorr, turnover.

Auxiliary panels (industry / market cap) load via ``data_access`` (ashare datasets);
falls back to direct COS CLI sync when the store is unavailable.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COS_INDUSTRY_PREFIX = "cos://qs-cold/clean_data/ashare/lqtp_data/StockIndustry"
COS_MCAP_PREFIX = "cos://qs-cold/clean_data/ashare/lqtp_data/StockValuationDaily"
from scripts.cogalpha_lqtp.data_access_panel import (  # noqa: E402
    DEFAULT_INDUSTRY_SOURCE,
    build_eval_aux_parquets_data_access,
    factor_values_cte_sql,
)
COS_CLI = "clean-cos-ro"


def _date_int_to_iso(d: int) -> str:
    s = str(int(d))
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def _iso_to_date_int(iso: str) -> int:
    dt = datetime.strptime(iso[:10], "%Y-%m-%d").date()
    return dt.year * 10000 + dt.month * 100 + dt.day


def _iter_calendar_dates(start: str, end: str) -> list[str]:
    cur = datetime.strptime(start[:10], "%Y-%m-%d").date()
    stop = datetime.strptime(end[:10], "%Y-%m-%d").date()
    out: list[str] = []
    while cur <= stop:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def _run_cos_cli(args: list[str]) -> None:
    proc = subprocess.run(
        [COS_CLI, *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{COS_CLI} failed ({proc.returncode}): {' '.join(args)}\n"
            f"{proc.stderr or proc.stdout}"
        )


def _sync_daily_parquets(
    *,
    cos_prefix: str,
    local_dir: Path,
    start: str,
    end: str,
) -> int:
    """Sync missing daily parquet files from COS for [start, end]. Returns count synced."""
    local_dir.mkdir(parents=True, exist_ok=True)
    synced = 0
    for iso in _iter_calendar_dates(start, end):
        fname = f"{iso}.parquet"
        dst = local_dir / fname
        if dst.exists() and dst.stat().st_size > 0:
            continue
        cos_path = f"{cos_prefix}/{fname}"
        try:
            _run_cos_cli(["cp", cos_path, str(dst)])
            synced += 1
        except RuntimeError:
            # Missing calendar day on COS is normal (weekends/holidays for some tables).
            continue
    return synced


def _build_eval_aux_via_cos_cli(
    *,
    cache_dir: Path,
    industry_out: Path,
    mcap_out: Path,
    start: str,
    end: str,
    industry_source: str,
) -> None:
    """Legacy fallback: sync daily files with clean-cos-ro then consolidate."""
    raw_ind = cache_dir / "raw" / "StockIndustry"
    raw_mcap = cache_dir / "raw" / "StockValuationDaily"
    print(f"[eval_cache] syncing industry from COS ({start} ~ {end})...")
    n_ind = _sync_daily_parquets(cos_prefix=COS_INDUSTRY_PREFIX, local_dir=raw_ind, start=start, end=end)
    print(f"[eval_cache] synced {n_ind} industry files; syncing market cap...")
    n_mcap = _sync_daily_parquets(cos_prefix=COS_MCAP_PREFIX, local_dir=raw_mcap, start=start, end=end)
    print(f"[eval_cache] synced {n_mcap} market cap files; consolidating...")

    con = duckdb.connect()
    try:
        ind_glob = (raw_ind / "*.parquet").as_posix()
        mcap_glob = (raw_mcap / "*.parquet").as_posix()
        src = industry_source.replace("'", "''")
        con.execute(
            f"""
            COPY (
              SELECT
                (year(TradeDate) * 10000 + month(TradeDate) * 100 + day(TradeDate))::INTEGER AS trade_date,
                Symbol::VARCHAR AS symbol,
                IndustryCode::VARCHAR AS industry_code
              FROM read_parquet('{ind_glob}', union_by_name=true)
              WHERE IndustrySource = '{src}'
                AND TradeDate >= DATE '{start[:10]}'
                AND TradeDate <= DATE '{end[:10]}'
            ) TO '{industry_out.as_posix()}' (FORMAT PARQUET)
            """
        )
        con.execute(
            f"""
            COPY (
              SELECT
                (year(TradeDate) * 10000 + month(TradeDate) * 100 + day(TradeDate))::INTEGER AS trade_date,
                Symbol::VARCHAR AS symbol,
                ln(greatest(MarketCap, 1.0))::DOUBLE AS log_market_cap
              FROM read_parquet('{mcap_glob}', union_by_name=true)
              WHERE TradeDate >= DATE '{start[:10]}'
                AND TradeDate <= DATE '{end[:10]}'
                AND MarketCap IS NOT NULL
                AND MarketCap > 0
            ) TO '{mcap_out.as_posix()}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()


def ensure_eval_aux_cache(
    work_dir: Path,
    *,
    start: str = "2019-01-01",
    end: str = "2026-06-30",
    industry_source: str = DEFAULT_INDUSTRY_SOURCE,
    force: bool = False,
) -> dict[str, Path]:
    """Build normalized industry + market-cap panels under work_dir/eval_cache/."""
    cache_dir = work_dir / "eval_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "aux_meta.json"
    industry_out = cache_dir / f"industry_{industry_source}.parquet"
    mcap_out = cache_dir / "market_cap.parquet"

    meta = {
        "start": start,
        "end": end,
        "industry_source": industry_source,
        "industry_out": str(industry_out),
        "mcap_out": str(mcap_out),
    }
    if (
        not force
        and industry_out.exists()
        and mcap_out.exists()
        and meta_path.exists()
    ):
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            if old.get("start") == start and old.get("end") == end:
                return {"industry": industry_out, "market_cap": mcap_out}
        except Exception:  # noqa: BLE001
            pass

    try:
        print(f"[eval_cache] loading industry/mcap via data_access ({start} ~ {end})...")
        build_eval_aux_parquets_data_access(
            industry_out=industry_out,
            mcap_out=mcap_out,
            start=start,
            end=end,
            industry_source=industry_source,
        )
        meta["source"] = "data_access"
    except Exception as exc:  # noqa: BLE001
        print(f"[eval_cache] data_access failed ({exc}); falling back to COS CLI")
        _build_eval_aux_via_cos_cli(
            cache_dir=cache_dir,
            industry_out=industry_out,
            mcap_out=mcap_out,
            start=start,
            end=end,
            industry_source=industry_source,
        )
        meta["source"] = "cos_cli"

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"industry": industry_out, "market_cap": mcap_out}


def _factor_cte_sql(factor_path: Path) -> str:
    return factor_values_cte_sql(factor_path)


def _neutral_rank_ic_sql(
    *,
    fac_sql: str,
    fwd_src: str,
    aux_join: str,
    fv_expr: str,
    min_names: int,
) -> str:
    return f"""
    WITH fac AS ({fac_sql}),
    m AS (
      SELECT f.trade_date, f.symbol, {fv_expr} AS fv, g.value AS rv
      FROM fac f
      JOIN {fwd_src} g
        ON f.trade_date = g.signal_date AND f.symbol = g.symbol
      {aux_join}
      WHERE f.value IS NOT NULL AND g.value IS NOT NULL
    ),
    day_n AS (SELECT trade_date, count(*) AS n FROM m GROUP BY 1),
    scored AS (
      SELECT m.trade_date, m.fv, m.rv, d.n,
             RANK() OVER (PARTITION BY m.trade_date ORDER BY m.fv) AS rf_min,
             COUNT(*) OVER (PARTITION BY m.trade_date, m.fv) AS fv_ties,
             RANK() OVER (PARTITION BY m.trade_date ORDER BY m.rv) AS rr_min,
             COUNT(*) OVER (PARTITION BY m.trade_date, m.rv) AS rv_ties
      FROM m JOIN day_n d USING (trade_date)
      WHERE d.n >= {min_names}
    ),
    ranked AS (
      SELECT trade_date, n,
             rf_min + (fv_ties - 1) * 0.5 AS rf,
             rr_min + (rv_ties - 1) * 0.5 AS rr
      FROM scored
    ),
    daily_ic AS (
      SELECT trade_date, any_value(n) AS n, corr(rf, rr) AS rank_ic
      FROM ranked GROUP BY trade_date
    )
    SELECT list(struct_pack(trade_date := trade_date, rank_ic := rank_ic) ORDER BY trade_date)
    FROM daily_ic
    """


def _compute_daily_rank_ic_series(
    con: duckdb.DuckDBPyConnection,
    sql: str,
) -> tuple[list[int], list[float]]:
    row = con.execute(sql).fetchone()
    if not row or row[0] is None:
        return [], []
    ic_rows = row[0]
    dates = [int(r["trade_date"]) for r in ic_rows]
    ics = [float(r["rank_ic"]) if r["rank_ic"] is not None else float("nan") for r in ic_rows]
    return dates, ics


def _ic_summary(daily_ic: list[float]) -> dict[str, float]:
    arr = np.asarray(daily_ic, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "mean_rank_ic": float("nan"),
            "std_rank_ic": float("nan"),
            "rank_icir": float("nan"),
            "rank_ic_positive_ratio": float("nan"),
        }
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {
        "mean_rank_ic": mean,
        "std_rank_ic": std,
        "rank_icir": mean / std if std > 1e-12 else 0.0,
        "rank_ic_positive_ratio": float(np.mean(arr > 0)),
    }


def ic_monthly_heatmap(
    trade_dates: list[int],
    daily_ic: list[float],
) -> dict[str, Any]:
    """Return year x month matrix for RankIC heatmap."""
    if not trade_dates or not daily_ic:
        return {"years": [], "months": list(range(1, 13)), "matrix": [], "monthly_mean_ic": {}}
    frame = pd.DataFrame({"trade_date": trade_dates, "ic": daily_ic})
    frame["dt"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d")
    frame["year"] = frame["dt"].dt.year
    frame["month"] = frame["dt"].dt.month
    monthly = frame.groupby(["year", "month"], as_index=False)["ic"].mean()
    years = sorted(monthly["year"].unique().tolist())
    months = list(range(1, 13))
    matrix: list[list[float | None]] = []
    monthly_mean: dict[str, float] = {}
    for y in years:
        row: list[float | None] = []
        for m in months:
            sub = monthly[(monthly["year"] == y) & (monthly["month"] == m)]
            if sub.empty:
                row.append(None)
            else:
                v = float(sub["ic"].iloc[0])
                row.append(v)
                monthly_mean[f"{y}-{m:02d}"] = v
        matrix.append(row)
    return {
        "years": years,
        "months": months,
        "matrix": matrix,
        "monthly_mean_ic": monthly_mean,
    }


def ic_autocorrelation(daily_ic: list[float], lags: tuple[int, ...] = (1, 5, 10, 20)) -> dict[str, float]:
    arr = np.asarray([x for x in daily_ic if np.isfinite(x)], dtype=float)
    out: dict[str, float] = {}
    if len(arr) < max(lags) + 2:
        for lag in lags:
            out[f"ic_autocorr_lag{lag}"] = float("nan")
        return out
    centered = arr - np.mean(arr)
    var = float(np.dot(centered, centered))
    if var < 1e-18:
        for lag in lags:
            out[f"ic_autocorr_lag{lag}"] = float("nan")
        return out
    for lag in lags:
        ac = float(np.dot(centered[:-lag], centered[lag:]) / var)
        out[f"ic_autocorr_lag{lag}"] = ac
    return out


def ic_autocorr_curve(daily_ic: list[float], max_lag: int = 20) -> dict[str, Any]:
    """Full autocorrelation curve for bar chart."""
    lags = tuple(range(1, max_lag + 1))
    ac = ic_autocorrelation(daily_ic, lags=lags)
    return {
        "lags": list(lags),
        "values": [ac.get(f"ic_autocorr_lag{lag}", float("nan")) for lag in lags],
    }


def _t_cdf(x: float, df: int) -> float:
    """Regularized incomplete beta approximation for Student-t CDF."""
    if df <= 0:
        return float("nan")
    t = df / (df + x * x)
    a, b = df / 2.0, 0.5
    # Lentz continued fraction for incomplete beta (symmetric two-tail uses |x|)
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(t) + b * math.log(1.0 - t)
    )
    if x >= 0:
        return 1.0 - 0.5 * bt * _betacf(t, a, b)
    return 0.5 * bt * _betacf(t, a, b)


def _betacf(x: float, a: float, b: float) -> float:
    """Continued fraction for incomplete beta function."""
    fpmin = 1e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-7:
            break
    return h


def ic_significance(daily_ic: list[float]) -> dict[str, float]:
    """Two-sided t-test on mean daily RankIC (i.i.d. assumption)."""
    arr = np.asarray([x for x in daily_ic if np.isfinite(x)], dtype=float)
    n = len(arr)
    if n < 3:
        return {"ic_t_stat": float("nan"), "ic_p_value": float("nan"), "ic_n_days": float(n)}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std < 1e-18:
        return {"ic_t_stat": float("nan"), "ic_p_value": float("nan"), "ic_n_days": float(n)}
    t_stat = mean / (std / math.sqrt(n))
    p_val = float(2.0 * (1.0 - _t_cdf(abs(t_stat), n - 1)))
    return {"ic_t_stat": t_stat, "ic_p_value": p_val, "ic_n_days": float(n)}


def rolling_ic_series(
    trade_dates: list[int],
    daily_ic: list[float],
    window: int = 20,
) -> dict[str, Any]:
    if not trade_dates or not daily_ic:
        return {"trade_dates": [], "rolling_ic": [], "window": window}
    frame = pd.DataFrame({"trade_date": trade_dates, "ic": daily_ic})
    frame["rolling_ic"] = frame["ic"].rolling(window=window, min_periods=max(5, window // 2)).mean()
    valid = frame.dropna(subset=["rolling_ic"])
    return {
        "trade_dates": valid["trade_date"].astype(int).tolist(),
        "rolling_ic": valid["rolling_ic"].astype(float).tolist(),
        "window": window,
    }


def decile_monotonicity(group_mean_returns: list[float] | None) -> dict[str, float]:
    """Spearman-like monotonicity: correlation between group index and mean return."""
    if not group_mean_returns or len(group_mean_returns) < 3:
        return {"decile_monotonicity": float("nan"), "decile_spread": float("nan")}
    g = np.arange(1, len(group_mean_returns) + 1, dtype=float)
    r = np.asarray(group_mean_returns, dtype=float)
    if np.std(r) < 1e-18:
        return {"decile_monotonicity": 0.0, "decile_spread": float(r[-1] - r[0])}
    mono = float(np.corrcoef(g, r)[0, 1])
    return {"decile_monotonicity": mono, "decile_spread": float(r[-1] - r[0])}


def ls_diagnostics(daily_ls: list[float] | None) -> dict[str, float]:
    if not daily_ls:
        return {"ls_max_drawdown": float("nan"), "ls_win_rate": float("nan")}
    arr = np.asarray(daily_ls, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"ls_max_drawdown": float("nan"), "ls_win_rate": float("nan")}
    equity = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return {
        "ls_max_drawdown": float(np.min(dd)),
        "ls_win_rate": float(np.mean(arr > 0)),
    }


def _compute_industry_ic_breakdown(
    con: duckdb.DuckDBPyConnection,
    *,
    fac_sql: str,
    fwd_src: str,
    industry_path: str,
    min_names_per_industry: int = 8,
    top_n: int = 15,
) -> list[dict[str, Any]]:
    sql = f"""
    WITH fac AS ({fac_sql}),
    m AS (
      SELECT f.trade_date, f.symbol, f.value AS fv, g.value AS rv, ind.industry_code
      FROM fac f
      JOIN {fwd_src} g ON f.trade_date = g.signal_date AND f.symbol = g.symbol
      JOIN read_parquet('{industry_path}') ind
        ON f.trade_date = ind.trade_date AND f.symbol = ind.symbol
      WHERE f.value IS NOT NULL AND g.value IS NOT NULL
    ),
    scored AS (
      SELECT trade_date, industry_code, fv, rv,
             RANK() OVER (PARTITION BY trade_date, industry_code ORDER BY fv) AS rf_min,
             COUNT(*) OVER (PARTITION BY trade_date, industry_code, fv) AS fv_ties,
             RANK() OVER (PARTITION BY trade_date, industry_code ORDER BY rv) AS rr_min,
             COUNT(*) OVER (PARTITION BY trade_date, industry_code, rv) AS rv_ties
      FROM m
    ),
    ranked AS (
      SELECT trade_date, industry_code,
             rf_min + (fv_ties - 1) * 0.5 AS rf,
             rr_min + (rv_ties - 1) * 0.5 AS rr
      FROM scored
    ),
    daily AS (
      SELECT trade_date, industry_code, corr(rf, rr) AS rank_ic, count(*) AS n
      FROM ranked
      GROUP BY trade_date, industry_code
      HAVING count(*) >= {min_names_per_industry}
    ),
    ind_avg AS (
      SELECT industry_code,
             avg(rank_ic) AS mean_ic,
             count(*) AS n_days,
             avg(n) AS avg_names
      FROM scored
      GROUP BY industry_code
      ORDER BY abs(mean_ic) DESC
      LIMIT {top_n}
    )
    SELECT list(struct_pack(
      industry_code := industry_code,
      mean_ic := mean_ic,
      n_days := n_days,
      avg_names := avg_names
    ) ORDER BY abs(mean_ic) DESC) FROM ind_avg
    """
    row = con.execute(sql).fetchone()
    if not row or not row[0]:
        return []
    return [
        {
            "industry_code": str(r["industry_code"]),
            "mean_ic": float(r["mean_ic"]) if r["mean_ic"] is not None else float("nan"),
            "n_days": int(r["n_days"]),
            "avg_names": float(r["avg_names"]),
        }
        for r in row[0]
    ]


def _compute_size_exposure(
    con: duckdb.DuckDBPyConnection,
    *,
    fac_sql: str,
    market_cap_path: str,
) -> dict[str, float]:
    sql = f"""
    WITH fac AS ({fac_sql}),
    j AS (
      SELECT f.trade_date, f.value AS fv, cap.log_market_cap AS lm
      FROM fac f
      JOIN read_parquet('{market_cap_path}') cap
        ON f.trade_date = cap.trade_date AND f.symbol = cap.symbol
      WHERE f.value IS NOT NULL AND cap.log_market_cap IS NOT NULL
    ),
    daily AS (
      SELECT trade_date, corr(fv, lm) AS size_corr, count(*) AS n
      FROM j GROUP BY trade_date HAVING count(*) >= 30
    )
    SELECT avg(size_corr) AS mean_size_corr,
           stddev_samp(size_corr) AS std_size_corr,
           count(*) AS n_days
    FROM daily
    """
    row = con.execute(sql).fetchone()
    if not row:
        return {"size_exposure_corr": float("nan"), "size_exposure_std": float("nan")}
    return {
        "size_exposure_corr": float(row[0]) if row[0] is not None else float("nan"),
        "size_exposure_std": float(row[1]) if row[1] is not None else float("nan"),
        "size_exposure_n_days": float(row[2] or 0),
    }


def ic_yearly_breakdown(trade_dates: list[int], daily_ic: list[float]) -> list[dict[str, Any]]:
    """Per-year Mean RankIC / ICIR / positive ratio."""
    if not trade_dates or not daily_ic:
        return []
    frame = pd.DataFrame({"trade_date": trade_dates, "ic": daily_ic})
    frame = frame[np.isfinite(frame["ic"])]
    if frame.empty:
        return []
    frame["year"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d").dt.year
    out: list[dict[str, Any]] = []
    for year, sub in frame.groupby("year"):
        arr = sub["ic"].to_numpy(dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        out.append(
            {
                "year": int(year),
                "n_days": int(len(arr)),
                "mean_rank_ic": mean,
                "rank_icir": (mean / std) if std > 1e-12 else float("nan"),
                "rank_ic_positive_ratio": float(np.mean(arr > 0)),
            }
        )
    return out


def enrich_post_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    """Derive diagnostics from base analysis without extra DuckDB joins."""
    out = dict(analysis)
    dates = analysis.get("trade_dates") or []
    daily_ic = analysis.get("daily_rank_ic") or analysis.get("daily_ic") or []
    out.update(ic_significance(daily_ic))
    out["rolling_ic_20d"] = rolling_ic_series(dates, daily_ic, window=20)
    out["ic_autocorr_curve"] = ic_autocorr_curve(daily_ic, max_lag=20)
    out["ic_yearly_breakdown"] = ic_yearly_breakdown(dates, daily_ic)
    out.update(decile_monotonicity(analysis.get("group_mean_returns")))
    out.update(ls_diagnostics(analysis.get("daily_ls_returns")))
    ext = dict(out.get("extended_eval") or {})
    ext.update({k: out[k] for k in (
        "ic_t_stat", "ic_p_value", "ic_n_days",
        "rolling_ic_20d", "ic_autocorr_curve", "ic_yearly_breakdown",
        "decile_monotonicity", "decile_spread",
        "ls_max_drawdown", "ls_win_rate",
    ) if k in out})
    out["extended_eval"] = ext
    return out


def ic_half_life(daily_ic: list[float], max_lag: int = 60) -> dict[str, float]:
    """Estimate IC half-life (days) from autocorrelation decay."""
    arr = np.asarray([x for x in daily_ic if np.isfinite(x)], dtype=float)
    if len(arr) < 10:
        return {"ic_half_life_days": float("nan"), "ic_decay_lambda": float("nan")}
    centered = arr - np.mean(arr)
    var = float(np.dot(centered, centered))
    if var < 1e-18:
        return {"ic_half_life_days": float("nan"), "ic_decay_lambda": float("nan")}
    lags: list[int] = []
    acfs: list[float] = []
    for lag in range(1, min(max_lag, len(arr) - 1)):
        ac = float(np.dot(centered[:-lag], centered[lag:]) / var)
        if not np.isfinite(ac):
            continue
        lags.append(lag)
        acfs.append(max(ac, 1e-6))
    if len(lags) < 3:
        return {"ic_half_life_days": float("nan"), "ic_decay_lambda": float("nan")}
    # Fit ln(acf) ~ -lambda * lag
    x = np.asarray(lags, dtype=float)
    y = np.log(np.asarray(acfs, dtype=float))
    slope, _ = np.polyfit(x, y, 1)
    lam = -float(slope)
    if lam <= 1e-9:
        half_life = float("nan")  # avoid JSON/HTML inf
    else:
        half_life = float(math.log(2.0) / lam)
        if not np.isfinite(half_life) or half_life > 1e6:
            half_life = float("nan")
    return {"ic_half_life_days": half_life, "ic_decay_lambda": lam}


def factor_rank_turnover(factor_path: Path, con: duckdb.DuckDBPyConnection | None = None) -> dict[str, float]:
    """Mean daily Spearman autocorrelation of cross-sectional factor ranks (T vs T-1)."""
    owns = con is None
    if con is None:
        con = duckdb.connect()
    try:
        fac_sql = _factor_cte_sql(factor_path)
        sql = f"""
        WITH fac AS ({fac_sql}),
        ranked AS (
          SELECT trade_date, symbol,
                 RANK() OVER (PARTITION BY trade_date ORDER BY value) AS rk
          FROM fac WHERE value IS NOT NULL
        ),
        cal AS (
          SELECT trade_date,
                 lag(trade_date) OVER (ORDER BY trade_date) AS prev_date
          FROM (SELECT DISTINCT trade_date FROM ranked)
        ),
        paired AS (
          SELECT c.prev_date AS d0, c.trade_date AS d1,
                 a.symbol, a.rk AS rk0, b.rk AS rk1
          FROM cal c
          JOIN ranked a ON a.trade_date = c.prev_date
          JOIN ranked b ON b.trade_date = c.trade_date AND b.symbol = a.symbol
          WHERE c.prev_date IS NOT NULL
        ),
        daily AS (
          SELECT d0 AS trade_date, corr(rk0, rk1) AS rank_autocorr, count(*) AS n
          FROM paired GROUP BY d0 HAVING count(*) >= 30
        )
        SELECT avg(rank_autocorr) AS mean_rank_autocorr,
               median(rank_autocorr) AS median_rank_autocorr,
               avg(1.0 - rank_autocorr) AS mean_rank_turnover
        FROM daily
        """
        row = con.execute(sql).fetchone()
        if row is None:
            return {
                "factor_rank_autocorr": float("nan"),
                "factor_rank_turnover": float("nan"),
                "factor_rank_autocorr_median": float("nan"),
            }
        return {
            "factor_rank_autocorr": float(row[0]) if row[0] is not None else float("nan"),
            "factor_rank_autocorr_median": float(row[1]) if row[1] is not None else float("nan"),
            "factor_rank_turnover": float(row[2]) if row[2] is not None else float("nan"),
        }
    finally:
        if owns:
            con.close()


def compute_extended_eval(
    *,
    factor_path: Path,
    fwd_returns_path: Path,
    industry_path: Path | None = None,
    market_cap_path: Path | None = None,
    min_names: int = 30,
    con: duckdb.DuckDBPyConnection | None = None,
    factor_sql: str | None = None,
) -> dict[str, Any]:
    """Compute neutral RankIC + IC diagnostics on top of existing factor values."""
    fac_sql = factor_sql or _factor_cte_sql(factor_path)
    fwd = fwd_returns_path.as_posix().replace("'", "''")
    owns = con is None
    if con is None:
        con = duckdb.connect()

    table_names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    fwd_src = "fwd" if "fwd" in table_names else f"read_parquet('{fwd}')"

    out: dict[str, Any] = {}

    # Raw IC series (reuse if caller already has it — recompute here for independence)
    raw_sql = _neutral_rank_ic_sql(
        fac_sql=fac_sql,
        fwd_src=fwd_src,
        aux_join="",
        fv_expr="f.value",
        min_names=min_names,
    )
    dates, raw_ic = _compute_daily_rank_ic_series(con, raw_sql)
    out.update(_ic_summary(raw_ic))
    out["ic_monthly_heatmap"] = ic_monthly_heatmap(dates, raw_ic)
    out.update(ic_autocorrelation(raw_ic))
    out.update(ic_half_life(raw_ic))
    out.update(factor_rank_turnover(factor_path, con=con))

    if industry_path and industry_path.exists():
        ind = industry_path.as_posix().replace("'", "''")
        ind_join = f"""
      JOIN read_parquet('{ind}') ind
        ON f.trade_date = ind.trade_date AND f.symbol = ind.symbol
        """
        ind_sql = _neutral_rank_ic_sql(
            fac_sql=fac_sql,
            fwd_src=fwd_src,
            aux_join=ind_join,
            fv_expr="f.value - avg(f.value) OVER (PARTITION BY f.trade_date, ind.industry_code)",
            min_names=min_names,
        )
        _, ind_ic = _compute_daily_rank_ic_series(con, ind_sql)
        ind_sum = _ic_summary(ind_ic)
        out["industry_neutral"] = {
            **ind_sum,
            "industry_source": DEFAULT_INDUSTRY_SOURCE,
            "label": "申万一级行业中性 RankIC",
        }

    if market_cap_path and market_cap_path.exists():
        mcap = market_cap_path.as_posix().replace("'", "''")
        mcap_join = f"""
      JOIN read_parquet('{mcap}') cap
        ON f.trade_date = cap.trade_date AND f.symbol = cap.symbol
        """
        # Per-day OLS residual: fv ~ log(market_cap)
        mcap_sql = f"""
        WITH fac AS ({fac_sql}),
        joined AS (
          SELECT f.trade_date, f.symbol, f.value AS fv, g.value AS rv, cap.log_market_cap AS lm
          FROM fac f
          JOIN {fwd_src} g ON f.trade_date = g.signal_date AND f.symbol = g.symbol
          JOIN read_parquet('{mcap}') cap ON f.trade_date = cap.trade_date AND f.symbol = cap.symbol
          WHERE f.value IS NOT NULL AND g.value IS NOT NULL AND cap.log_market_cap IS NOT NULL
        ),
        stats AS (
          SELECT trade_date,
                 regr_slope(fv, lm) AS beta,
                 avg(fv) AS mu_f,
                 avg(lm) AS mu_m
          FROM joined GROUP BY trade_date
        ),
        neutral AS (
          SELECT j.trade_date, j.symbol,
                 j.fv - (s.mu_f + s.beta * (j.lm - s.mu_m)) AS fv,
                 j.rv
          FROM joined j JOIN stats s USING (trade_date)
        ),
        day_n AS (SELECT trade_date, count(*) AS n FROM neutral GROUP BY 1),
        scored AS (
          SELECT m.trade_date, m.fv, m.rv, d.n,
                 RANK() OVER (PARTITION BY m.trade_date ORDER BY m.fv) AS rf_min,
                 COUNT(*) OVER (PARTITION BY m.trade_date, m.fv) AS fv_ties,
                 RANK() OVER (PARTITION BY m.trade_date ORDER BY m.rv) AS rr_min,
                 COUNT(*) OVER (PARTITION BY m.trade_date, m.rv) AS rv_ties
          FROM neutral m JOIN day_n d USING (trade_date)
          WHERE d.n >= {min_names}
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
        row = con.execute(mcap_sql).fetchone()
        mcap_ic = []
        if row and row[0]:
            mcap_ic = [float(r["rank_ic"]) if r["rank_ic"] is not None else float("nan") for r in row[0]]
        mcap_sum = _ic_summary(mcap_ic)
        out["size_neutral"] = {**mcap_sum, "label": "市值中性 RankIC (OLS vs log MarketCap)"}

    if industry_path and industry_path.exists() and market_cap_path and market_cap_path.exists():
        ind = industry_path.as_posix().replace("'", "''")
        mcap = market_cap_path.as_posix().replace("'", "''")
        combo_sql = f"""
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
          SELECT m.trade_date, m.fv, m.rv, d.n,
                 RANK() OVER (PARTITION BY m.trade_date ORDER BY m.fv) AS rf_min,
                 COUNT(*) OVER (PARTITION BY m.trade_date, m.fv) AS fv_ties,
                 RANK() OVER (PARTITION BY m.trade_date ORDER BY m.rv) AS rr_min,
                 COUNT(*) OVER (PARTITION BY m.trade_date, m.rv) AS rv_ties
          FROM neutral m JOIN day_n d USING (trade_date)
          WHERE d.n >= {min_names}
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
        row = con.execute(combo_sql).fetchone()
        combo_ic = []
        if row and row[0]:
            combo_ic = [float(r["rank_ic"]) if r["rank_ic"] is not None else float("nan") for r in row[0]]
        combo_sum = _ic_summary(combo_ic)
        out["industry_size_neutral"] = {
            **combo_sum,
            "label": "行业+市值双中性 RankIC",
        }

    if industry_path and industry_path.exists():
        try:
            out["industry_ic_breakdown"] = _compute_industry_ic_breakdown(
                con,
                fac_sql=fac_sql,
                fwd_src=fwd_src,
                industry_path=industry_path.as_posix().replace("'", "''"),
            )
        except Exception:  # noqa: BLE001
            out["industry_ic_breakdown"] = []

    if market_cap_path and market_cap_path.exists():
        try:
            out.update(
                _compute_size_exposure(
                    con,
                    fac_sql=fac_sql,
                    market_cap_path=market_cap_path.as_posix().replace("'", "''"),
                )
            )
        except Exception:  # noqa: BLE001
            pass

    if owns:
        con.close()
    return out


def merge_extended_into_analysis(analysis: dict[str, Any], extended: dict[str, Any]) -> dict[str, Any]:
    """Attach extended metrics to base analysis dict (in-place friendly copy)."""
    merged = dict(analysis)
    merged["extended_eval"] = extended
    if "industry_neutral" in extended:
        merged["industry_neutral_mean_rank_ic"] = extended["industry_neutral"].get("mean_rank_ic")
        merged["industry_neutral_rank_icir"] = extended["industry_neutral"].get("rank_icir")
    if "size_neutral" in extended:
        merged["size_neutral_mean_rank_ic"] = extended["size_neutral"].get("mean_rank_ic")
        merged["size_neutral_rank_icir"] = extended["size_neutral"].get("rank_icir")
    if "industry_size_neutral" in extended:
        merged["industry_size_neutral_mean_rank_ic"] = extended["industry_size_neutral"].get("mean_rank_ic")
    for k in (
        "ic_half_life_days",
        "ic_decay_lambda",
        "factor_rank_turnover",
        "factor_rank_autocorr",
        "ic_autocorr_lag1",
        "ic_autocorr_lag5",
        "ic_autocorr_lag20",
        "size_exposure_corr",
        "size_exposure_std",
        "size_exposure_n_days",
        "industry_ic_breakdown",
    ):
        if k in extended:
            merged[k] = extended[k]
    if "ic_monthly_heatmap" in extended:
        merged["ic_monthly_heatmap"] = extended["ic_monthly_heatmap"]
    return enrich_post_analysis(merged)
