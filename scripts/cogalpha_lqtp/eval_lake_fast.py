#!/usr/bin/env python3
"""Fast lake eval: DuckDB panel RankIC / groups / LS (data_access engine stack).

Preloads forward VWAP→VWAP (default) or close-to-close returns once, then
evaluates each factor_lake parquet with a single DuckDB join+window query.
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

from scripts.cogalpha_lqtp.data_access_panel import (  # noqa: E402
    ashare_materialize_data_source_config,
    build_close_returns_cache_data_access,
    build_vwap_returns_cache_data_access,
    factor_values_cte_sql,
    resolve_factor_values_parquet,
)
from scripts.cogalpha_lqtp.eval_extensions import (  # noqa: E402
    compute_extended_eval,
    ensure_eval_aux_cache,
    merge_extended_into_analysis,
)
from scripts.cogalpha_lqtp.lqtp_dsl_compat import eval_route_for_entry  # noqa: E402
from scripts.cogalpha_lqtp.report_html import (  # noqa: E402
    render_factor_report,
    render_index,
    render_production_summary,
)
from scripts.cogalpha_lqtp.reconcile_ic_signs import (  # noqa: E402
    _unwrap_formula,
    _unwrap_python_code,
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


def build_fwd_returns_cache(
    returns_path: Path,
    fwd_path: Path,
    *,
    lead_days: int = 1,
    force: bool = False,
) -> Path:
    """Map signal_date T → return labeled on the *lead_days*-th next session.

    - close panel (lead_days=1): close(T+1)/close(T)-1
    - VWAP panel (lead_days=2): vwap(T+2)/vwap(T+1)-1

    VWAP must use lead_days=2: factor is known after close T, so same-day Vwap_T
    cannot enter the return (Vwap_{T+1}/Vwap_T leaks intraday structure into RankIC).
    """
    if lead_days < 1:
        raise ValueError("lead_days must be >= 1")
    marker = fwd_path.with_suffix(fwd_path.suffix + f".lead{lead_days}")
    if (
        not force
        and fwd_path.exists()
        and marker.exists()
        and fwd_path.stat().st_mtime >= returns_path.stat().st_mtime
        and marker.stat().st_mtime >= fwd_path.stat().st_mtime
    ):
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
                FROM read_parquet('{returns_path.as_posix()}')
              ),
              cal AS (SELECT DISTINCT trade_date FROM ret),
              nxt AS (
                SELECT trade_date,
                       lead(trade_date, {int(lead_days)}) OVER (ORDER BY trade_date) AS exit_date
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
    marker.write_text(f"lead_days={lead_days}\n", encoding="utf-8")
    return fwd_path


def _factor_cte_sql(factor_path: Path) -> str:
    """Branch on parquet schema via pyarrow footer (no full scan)."""
    return factor_values_cte_sql(factor_path)


def analyze_factor_long_df_duckdb(
    long_df: pd.DataFrame,
    *,
    fwd_returns_path: Path,
    n_groups: int = _N_GROUPS,
    commission_buy: float = DEFAULT_COMMISSION_BPS,
    commission_sell: float = DEFAULT_COMMISSION_BPS,
    min_names: int = 30,
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[str, Any]:
    """RankIC / deciles / LS from an in-memory long factor frame (trade_date, symbol, value)."""
    work = long_df.copy()
    if "trade_date" not in work.columns or "symbol" not in work.columns:
        raise ValueError("long_df needs trade_date, symbol, value")
    if "value" not in work.columns:
        raise ValueError("long_df needs value column")
    work = work[["trade_date", "symbol", "value"]].dropna(subset=["value"])
    work["trade_date"] = work["trade_date"].astype(int)
    work["symbol"] = work["symbol"].astype(str)
    # DuckDB can scan a registered DataFrame without a temp parquet.
    owns_con = con is None
    if con is None:
        con = duckdb.connect()
    try:
        con.register("_factor_long_df", work)
        return analyze_factor_parquet_duckdb(
            factor_path=Path("__dataframe__"),
            fwd_returns_path=fwd_returns_path,
            n_groups=n_groups,
            commission_buy=commission_buy,
            commission_sell=commission_sell,
            min_names=min_names,
            con=con,
            factor_sql=(
                "SELECT trade_date::INTEGER AS trade_date, "
                "symbol::VARCHAR AS symbol, value::DOUBLE AS value "
                "FROM _factor_long_df"
            ),
        )
    finally:
        try:
            con.unregister("_factor_long_df")
        except Exception:
            pass
        if owns_con:
            con.close()


def analyze_factor_parquet_duckdb(
    *,
    factor_path: Path,
    fwd_returns_path: Path,
    n_groups: int = _N_GROUPS,
    commission_buy: float = DEFAULT_COMMISSION_BPS,
    commission_sell: float = DEFAULT_COMMISSION_BPS,
    min_names: int = 30,
    con: duckdb.DuckDBPyConnection | None = None,
    factor_sql: str | None = None,
    return_kind: str = "vwap_to_vwap",
) -> dict[str, Any]:
    """Vectorized RankIC / deciles / G10−G1 via DuckDB (average-rank Spearman)."""
    fac_sql = factor_sql if factor_sql else _factor_cte_sql(factor_path)
    fwd = fwd_returns_path.as_posix().replace("'", "''")
    roundtrip = float(commission_buy) + float(commission_sell)
    owns_con = con is None
    if con is None:
        con = duckdb.connect()

    table_names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    fwd_src = "fwd" if "fwd" in table_names else f"read_parquet('{fwd}')"

    # Net LS cost uses *realized* one-way turnover of equal-weight G10 / G1 books
    # (0.5 * L1 weight change per leg), not a fixed 40% assumption.
    sql = f"""
    WITH fac AS ({fac_sql}),
    univ AS (
      SELECT signal_date AS trade_date, count(*)::INTEGER AS n_univ
      FROM {fwd_src}
      WHERE value IS NOT NULL
      GROUP BY 1
    ),
    fac_day AS (
      SELECT trade_date, count(*)::INTEGER AS n_fac
      FROM fac
      WHERE value IS NOT NULL
      GROUP BY 1
    ),
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
      SELECT trade_date, count(*)::INTEGER AS n FROM m GROUP BY 1
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
    ),
    -- Equal-weight G10/G1 one-way turnover computed in-DB (avoid shipping memberships).
    weights AS (
      SELECT trade_date, symbol, grp,
             1.0 / count(*) OVER (PARTITION BY trade_date, grp) AS w
      FROM ranked
      WHERE grp IN (1, {n_groups})
    ),
    cal AS (
      SELECT trade_date,
             lag(trade_date) OVER (ORDER BY trade_date) AS prev_date
      FROM (SELECT DISTINCT trade_date FROM ranked)
    ),
    w_union AS (
      SELECT c.trade_date, w.grp, w.symbol, w.w AS w_t, 0.0 AS w_p
      FROM cal c
      JOIN weights w ON w.trade_date = c.trade_date
      UNION ALL
      SELECT c.trade_date, w.grp, w.symbol, 0.0 AS w_t, w.w AS w_p
      FROM cal c
      JOIN weights w ON w.trade_date = c.prev_date
      WHERE c.prev_date IS NOT NULL
    ),
    w_merged AS (
      SELECT trade_date, grp, symbol, sum(w_t) AS w_t, sum(w_p) AS w_p
      FROM w_union
      GROUP BY 1, 2, 3
    ),
    to_leg AS (
      SELECT trade_date, grp, 0.5 * sum(abs(w_t - w_p)) AS one_way
      FROM w_merged
      GROUP BY 1, 2
    ),
    daily_to AS (
      SELECT trade_date,
             coalesce(max(CASE WHEN grp = {n_groups} THEN one_way END), 0.0) AS to_long,
             coalesce(max(CASE WHEN grp = 1 THEN one_way END), 0.0) AS to_short
      FROM to_leg
      GROUP BY trade_date
    ),
    cov AS (
      SELECT d.trade_date,
             d.n AS n_overlap,
             coalesce(f.n_fac, 0) AS n_fac,
             coalesce(u.n_univ, 0) AS n_univ
      FROM day_n d
      LEFT JOIN fac_day f USING (trade_date)
      LEFT JOIN univ u USING (trade_date)
      WHERE d.n >= {max(min_names, n_groups * 3)}
    )
    SELECT
      (SELECT list(struct_pack(trade_date := trade_date, n := n, rank_ic := rank_ic)
                   ORDER BY trade_date) FROM daily_ic) AS ic_rows,
      (SELECT list(struct_pack(trade_date := trade_date, ls_gross := ls_gross)
                   ORDER BY trade_date) FROM daily_ls) AS ls_rows,
      (SELECT list(struct_pack(trade_date := trade_date, grp := grp, gret := gret)
                   ORDER BY trade_date, grp) FROM daily_grp) AS grp_rows,
      (SELECT list(struct_pack(trade_date := trade_date, to_long := to_long, to_short := to_short)
                   ORDER BY trade_date) FROM daily_to) AS to_rows,
      (SELECT list(struct_pack(
                   trade_date := trade_date, n_overlap := n_overlap,
                   n_fac := n_fac, n_univ := n_univ)
                   ORDER BY trade_date) FROM cov) AS cov_rows
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
    to_rows = row[3] or []
    cov_rows = row[4] or []

    trade_dates = [int(r["trade_date"]) for r in ic_rows]
    daily_rank_ic = [float(r["rank_ic"]) if r["rank_ic"] is not None else float("nan") for r in ic_rows]
    sample_counts = [int(r["n"]) for r in ic_rows]
    ls_by_date = {int(r["trade_date"]): float(r["ls_gross"]) for r in ls_rows if r["ls_gross"] is not None}
    daily_ls_gross = [ls_by_date.get(td, 0.0) for td in trade_dates]

    to_by_date = {
        int(r["trade_date"]): (float(r["to_long"] or 0.0), float(r["to_short"] or 0.0))
        for r in to_rows
    }
    daily_to_long = [to_by_date.get(td, (0.0, 0.0))[0] for td in trade_dates]
    daily_to_short = [to_by_date.get(td, (0.0, 0.0))[1] for td in trade_dates]
    # roundtrip is percent points (e.g. 0.02); convert to decimal return units.
    daily_cost = [(tl + ts) * roundtrip / 100.0 for tl, ts in zip(daily_to_long, daily_to_short)]
    daily_ls_net = [g - c for g, c in zip(daily_ls_gross, daily_cost)]

    cov_by_date = {int(r["trade_date"]): r for r in cov_rows}
    coverages: list[float] = []
    n_fac_list: list[int] = []
    n_univ_list: list[int] = []
    for td, n_ov in zip(trade_dates, sample_counts):
        cr = cov_by_date.get(td) or {}
        n_univ = int(cr.get("n_univ") or 0)
        n_fac = int(cr.get("n_fac") or 0)
        n_univ_list.append(n_univ)
        n_fac_list.append(n_fac)
        coverages.append(float(n_ov) / float(n_univ) if n_univ > 0 else float("nan"))

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
    cov_arr = np.asarray(coverages, dtype=float)
    to_l = np.asarray(daily_to_long, dtype=float)
    to_s = np.asarray(daily_to_short, dtype=float)
    to_sum = to_l + to_s

    return {
        "mean_rank_ic": mean_rank_ic,
        "std_rank_ic": std_rank_ic,
        "rank_icir": rank_icir,
        "rank_ic_positive_ratio": rank_ic_pos,
        "mean_ic": mean_rank_ic,
        "std_ic": std_rank_ic,
        "icir": rank_icir,
        "ic_positive_ratio": rank_ic_pos,
        "coverage": float(np.nanmean(cov_arr)) if len(cov_arr) else float("nan"),
        "mean_daily_coverage": float(np.nanmean(cov_arr)) if len(cov_arr) else float("nan"),
        "median_daily_coverage": float(np.nanmedian(cov_arr)) if len(cov_arr) else float("nan"),
        "mean_overlap_names": float(np.nanmean(sample_counts)) if sample_counts else float("nan"),
        "mean_universe_names": float(np.nanmean(n_univ_list)) if n_univ_list else float("nan"),
        "mean_factor_names": float(np.nanmean(n_fac_list)) if n_fac_list else float("nan"),
        "long_short_return": ls_cum,
        "long_short_return_sum": float(np.nansum(ls_net)),
        "long_short_sharpe": _sharpe(ls_net),
        "long_short_return_gross_sum": float(np.nansum(ls_gross)),
        "long_short_sharpe_gross": _sharpe(ls_gross),
        "ls_mean_one_way_turnover": float(np.nanmean(to_sum)) if len(to_sum) else float("nan"),
        "ls_mean_long_turnover": float(np.nanmean(to_l)) if len(to_l) else float("nan"),
        "ls_mean_short_turnover": float(np.nanmean(to_s)) if len(to_s) else float("nan"),
        "ls_mean_daily_cost": float(np.nanmean(daily_cost)) if daily_cost else float("nan"),
        "daily_ic": daily_rank_ic,
        "daily_rank_ic": daily_rank_ic,
        "daily_ls_returns": ls_net.tolist(),
        "daily_ls_gross": ls_gross.tolist(),
        "daily_ls_turnover": to_sum.tolist(),
        "daily_ls_cost": list(daily_cost),
        "trade_dates": trade_dates,
        "quote_times": [0] * len(trade_dates),
        "sample_counts": sample_counts,
        "coverages": coverages,
        "group_mean_returns": group_mean_returns,
        "group_pnls": group_pnls,
        "return_kind": (
            "vwap_to_vwap_T1_T2"
            if return_kind == "vwap_to_vwap"
            else f"{return_kind}_T_plus_1"
        ),
        "return_mode": return_kind,
        "signal_lag_note": (
            (
                "Factor after close T → trade VWAP(T+1)→VWAP(T+2) "
                "(forward return vwap(T+2)/vwap(T+1)-1; not vwap(T+1)/vwap(T) which leaks) "
                if return_kind == "vwap_to_vwap"
                else "Factor after close T → forward return close(T+1)/close(T)-1 "
            )
            + "(DuckDB panel RankIC); LS net costs use realized G10/G1 one-way turnover × commission; "
            "TopK backtest uses OPEN execution separately"
        ),
        "commission_buy": commission_buy,
        "commission_sell": commission_sell,
        "n_groups": n_groups,
        "eval_engine": "duckdb_panel",
        "ls_cost_model": "realized_decile_turnover",
    }


_WORKER: dict[str, Any] = {}


def _init_worker(fwd_path: str, industry_path: str = "", market_cap_path: str = "") -> None:
    """Load forward-return parquet into an in-memory DuckDB table once per worker."""
    con = duckdb.connect()
    # Soft cap per worker (host has ~30Gi; keep headroom, not ultra-conservative).
    con.execute("SET threads TO 4")
    con.execute("SET memory_limit = '4GB'")
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
    _WORKER["industry_path"] = industry_path
    _WORKER["market_cap_path"] = market_cap_path


def _apply_extended_eval(
    analysis: dict[str, Any],
    *,
    factor_path: Path,
    fwd: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not payload.get("extended_eval", True):
        return analysis
    ind_p = (_WORKER.get("industry_path") or "").strip()
    mcap_p = (_WORKER.get("market_cap_path") or "").strip()
    try:
        extended = compute_extended_eval(
            factor_path=factor_path,
            fwd_returns_path=fwd,
            industry_path=Path(ind_p) if ind_p else None,
            market_cap_path=Path(mcap_p) if mcap_p else None,
            con=_WORKER.get("con"),
        )
        merged = merge_extended_into_analysis(analysis, extended)
        return merged
    except Exception as ext_exc:  # noqa: BLE001
        out = dict(analysis)
        out["extended_eval_error"] = str(ext_exc)
        return out


def _eval_one(payload: dict[str, Any]) -> dict[str, Any]:
    name = payload["name"]
    t0 = time.time()
    try:
        work = Path(payload["work_dir"])
        lake = work / "factor_lake"
        report_subdir = payload.get("report_subdir") or "reports"
        report_dir = work / report_subdir
        report_dir.mkdir(parents=True, exist_ok=True)
        fwd = Path(payload["fwd_path"])
        out_path = resolve_factor_values_parquet(work, name)
        if out_path is None:
            raise RuntimeError(f"missing values for {name} under factor_lake/")

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

        return_kind = str(payload.get("return_kind") or "vwap_to_vwap")
        # t1t2 = enter next-session VWAP, exit following VWAP (no same-day Vwap_T leak)
        eval_mode = (
            "duckdb_panel_vwap_to_vwap_t1t2_realto"
            if return_kind == "vwap_to_vwap"
            else f"duckdb_panel_{return_kind}"
        )
        analysis = analyze_factor_parquet_duckdb(
            factor_path=out_path,
            fwd_returns_path=fwd,
            con=_WORKER.get("con"),
            return_kind=return_kind,
        )
        analysis = _apply_extended_eval(analysis, factor_path=out_path, fwd=fwd, payload=payload)
        flip_patch: dict[str, Any] | None = None
        ic = _mean_ic_value(analysis)
        # Heal negative RankIC: toggle lake values + formula, then re-run the full
        # panel (RankIC + deciles + LS) so report charts match the signed definition.
        if payload.get("flip_negative_ic") and ic < 0 and ic == ic:
            already = bool(entry.get("ic_sign_flipped"))
            new_flipped = not already
            flip_patch = {
                "ic_sign_flipped": new_flipped,
                "values_negated": True,
                "sign_action": "unflip" if already else "flip",
            }
            entry["ic_sign_flipped"] = new_flipped
            if dsl_used and dsl_used != "(python only)":
                dsl_used = _unwrap_formula(dsl_used) if already else _negate_formula(dsl_used)
                entry["dsl"] = dsl_used
                flip_patch["dsl"] = dsl_used
            if py_code.strip():
                py_code = _unwrap_python_code(py_code) if already else _negate_python_code(py_code)
                entry["python_code"] = py_code
                flip_patch["python_code"] = py_code
            _negate_values_parquet(out_path)
            analysis = analyze_factor_parquet_duckdb(
                factor_path=out_path,
                fwd_returns_path=fwd,
                con=_WORKER.get("con"),
                return_kind=return_kind,
            )
            analysis = _apply_extended_eval(analysis, factor_path=out_path, fwd=fwd, payload=payload)
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
            "return_kind": return_kind,
        }
        meta["ic_sign_flipped"] = bool(entry.get("ic_sign_flipped"))
        if flip_patch:
            meta["ic_before_flip"] = flip_patch.get("ic_before_flip")
            meta["mean_ic_after_flip"] = flip_patch.get("mean_ic_after_flip")
            meta["sign_action"] = flip_patch.get("sign_action")

        report_path = report_dir / f"{name}.html"
        render_factor_report(
            factor_name=name,
            dsl=dsl_used,
            analysis=analysis,
            backtest_rows=[],
            out_path=report_path,
            eval_mode=eval_mode,
            materialize_meta=meta,
            topk_summary={},
            python_code=py_code,
            work_dir=work,
            engine=engine,
            eval_route=route,
        )
        row = {
            "factor_name": name,
            "report": report_path.name,
            "engine": engine,
            "eval_route": route,
            "eval_mode": eval_mode,
            "ic_sign_flipped": bool(entry.get("ic_sign_flipped")),
            "mean_ic": _json_float(analysis.get("mean_rank_ic")),
            "mean_rank_ic": _json_float(analysis.get("mean_rank_ic")),
            "icir": _json_float(analysis.get("rank_icir")),
            "rank_icir": _json_float(analysis.get("rank_icir")),
            "rank_ic_positive_ratio": _json_float(analysis.get("rank_ic_positive_ratio")),
            "long_short_sharpe": _json_float(analysis.get("long_short_sharpe")),
            "long_short_return": _json_float(analysis.get("long_short_return")),
            "long_short_sharpe_gross": _json_float(analysis.get("long_short_sharpe_gross")),
            "ls_mean_one_way_turnover": _json_float(analysis.get("ls_mean_one_way_turnover")),
            "ls_mean_daily_cost": _json_float(analysis.get("ls_mean_daily_cost")),
            "mean_daily_coverage": _json_float(analysis.get("mean_daily_coverage")),
            "median_daily_coverage": _json_float(analysis.get("median_daily_coverage")),
            "mean_overlap_names": _json_float(analysis.get("mean_overlap_names")),
            "industry_neutral_mean_rank_ic": _json_float(analysis.get("industry_neutral_mean_rank_ic")),
            "size_neutral_mean_rank_ic": _json_float(analysis.get("size_neutral_mean_rank_ic")),
            "ic_half_life_days": _json_float(analysis.get("ic_half_life_days")),
            "factor_rank_turnover": _json_float(analysis.get("factor_rank_turnover")),
            "size_exposure_corr": _json_float(analysis.get("size_exposure_corr")),
            "decile_monotonicity": _json_float(analysis.get("decile_monotonicity")),
            "ls_max_drawdown": _json_float(analysis.get("ls_max_drawdown")),
            "ic_t_stat": _json_float(analysis.get("ic_t_stat")),
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
    parser.add_argument(
        "--skip-lookahead",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip LOOKAHEAD_DEFERRED_FACTORS (default on). Use --no-skip-lookahead to evaluate them.",
    )
    parser.add_argument(
        "--returns-source",
        choices=("lqtp", "data_access", "auto"),
        default="auto",
        help="Close returns cache: LQTP API, data_access ashare_stock_daily, or auto",
    )
    parser.add_argument("--skip-extended", action="store_true", help="Skip neutral IC / heatmap / half-life")
    parser.add_argument("--force-aux-cache", action="store_true", help="Re-sync industry/mcap panels from COS")
    parser.add_argument("--catalog", type=Path, default=None, help="DSL catalog JSON (default: work-dir/dsl_catalog.json)")
    parser.add_argument("--parsed-json", type=Path, default=None, help="Parsed python factors JSON")
    parser.add_argument("--progress-file", type=Path, default=None, help="Progress checkpoint JSON")
    parser.add_argument("--report-subdir", default="reports", help="Reports subdirectory under work-dir")
    parser.add_argument(
        "--return-kind",
        choices=("vwap", "close"),
        default="vwap",
        help="Forward return for RankIC/LS: vwap→vwap (default) or close→close",
    )
    args = parser.parse_args()

    work = args.work_dir
    catalog_path = args.catalog or (work / "dsl_catalog.json")
    parsed_path = args.parsed_json or (work / "parsed_factors.json")
    progress_path = args.progress_file or (work / "production_progress.json")
    return_kind = "vwap_to_vwap" if args.return_kind == "vwap" else "close_to_close"
    eval_mode = (
        "duckdb_panel_vwap_to_vwap_t1t2_realto"
        if return_kind == "vwap_to_vwap"
        else f"duckdb_panel_{return_kind}"
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    python_map = {
        r["function_name"]: r["python_code"]
        for r in json.loads(parsed_path.read_text(encoding="utf-8"))
    } if parsed_path.exists() else {}
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    flip_state: dict[str, Any] = progress.setdefault("flip_state", {})
    _apply_flip_state_to_catalog(catalog, flip_state, python_map)
    entry_by_name = {e["function_name"]: e for e in catalog}

    if return_kind == "vwap_to_vwap":
        ret_cache = work / "lqtp_vwap_returns_cache.parquet"
        fwd_path = work / "lqtp_fwd_vwap_returns_cache.parquet"
        # Signal after close T → earn vwap(T+2)/vwap(T+1)-1 (lead 2 on daily VWAP returns).
        fwd_lead_days = 2
        if args.returns_source == "data_access" or (
            args.returns_source == "auto" and not ret_cache.exists()
        ):
            print(f"building VWAP returns from data_access → {ret_cache.name}")
            build_vwap_returns_cache_data_access(
                ret_cache,
                start=args.start,
                end=args.end,
            )
        elif not ret_cache.exists():
            raise SystemExit(
                f"missing VWAP returns cache: {ret_cache} "
                "(pass --returns-source data_access)"
            )
    else:
        ret_cache = work / "lqtp_close_returns_cache.parquet"
        fwd_path = work / "lqtp_fwd_close_returns_cache.parquet"
        fwd_lead_days = 1
        if args.returns_source == "data_access" or (
            args.returns_source == "auto" and not ret_cache.exists()
        ):
            print(f"building close returns from data_access → {ret_cache.name}")
            build_close_returns_cache_data_access(
                ret_cache,
                start=args.start,
                end=args.end,
            )
        elif not ret_cache.exists():
            raise SystemExit(
                f"missing close returns cache: {ret_cache} "
                "(run production batch or pass --returns-source data_access)"
            )
    print(
        f"building fwd returns cache ({return_kind}, lead_days={fwd_lead_days}) → {fwd_path.name}"
    )
    build_fwd_returns_cache(ret_cache, fwd_path, lead_days=fwd_lead_days, force=True)

    aux_paths = {"industry": Path(), "market_cap": Path()}
    if not args.skip_extended:
        print("building eval aux cache (industry + market cap via data_access)...")
        aux_paths = ensure_eval_aux_cache(
            work,
            start=args.start,
            end=args.end,
            force=args.force_aux_cache,
        )

    skipped = set(progress.get("skipped", []))
    if args.skip_lookahead:
        skipped |= set(LOOKAHEAD_DEFERRED_FACTORS)
        progress["skipped"] = sorted(skipped)
        _save_progress(progress_path, progress)

    lake = work / "factor_lake"
    names = sorted(
        p.name
        for p in lake.iterdir()
        if resolve_factor_values_parquet(work, p.name) is not None
    )
    names = [n for n in names if n not in skipped]
    already = set(progress.get("completed", []))
    if args.only:
        only = set(args.only)
        names = [n for n in names if n in only]
    # Resume: skip factors already evaluated under the same return kind / eval_mode
    if not args.only:
        done_panel = {
            r.get("factor_name")
            for r in progress.get("index_rows", [])
            if r.get("eval_mode") == eval_mode
        }
        names = [n for n in names if n not in done_panel]
        print(f"resume skip already-panel={len(done_panel)} remaining={len(names)} mode={eval_mode}")

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
                "extended_eval": not args.skip_extended,
                "report_subdir": args.report_subdir,
                "return_kind": return_kind,
            }
        )

    print(
        f"eval {len(payloads)} factors workers={args.workers} "
        f"engine=duckdb_panel return_kind={return_kind} extended={not args.skip_extended}"
    )
    t0 = time.time()
    index_rows: list[dict[str, Any]] = list(progress.get("index_rows", []))
    failed: dict[str, str] = dict(progress.get("failed", {}))
    completed: list[str] = list(progress.get("completed", []))

    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=_init_worker,
        initargs=(
            str(fwd_path),
            str(aux_paths.get("industry", "")),
            str(aux_paths.get("market_cap", "")),
        ),
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
            # Checkpoint every factor so a worker death loses at most one result.
            if True:
                progress["completed"] = completed
                progress["failed"] = failed
                progress["index_rows"] = index_rows
                progress["flip_state"] = flip_state
                progress["eval_methodology"] = (
                    "rankic_vwap_to_vwap_T1T2_duckdb_panel_v1"
                    if return_kind == "vwap_to_vwap"
                    else f"rankic_{return_kind}_Tplus1_duckdb_panel_v1"
                )
                progress["return_kind"] = return_kind
                progress["regen_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                progress["note"] = (
                    f"realto in progress {done_n}/{len(payloads)} "
                    f"(workers capped, duckdb mem limited)"
                )
                _save_progress(progress_path, progress)
                if done_n % 5 == 0 or done_n == len(payloads):
                    print(f"  checkpoint {done_n}/{len(payloads)} ok={len(completed)} fail={len(failed)}")

    progress["completed"] = completed
    progress["failed"] = failed
    progress["index_rows"] = index_rows
    progress["flip_state"] = flip_state
    progress["eval_methodology"] = (
        "rankic_vwap_to_vwap_T1T2_duckdb_panel_v1"
        if return_kind == "vwap_to_vwap"
        else f"rankic_{return_kind}_Tplus1_duckdb_panel_v1"
    )
    progress["return_kind"] = return_kind
    progress["regen_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_progress(progress_path, progress)

    report_dir = work / args.report_subdir
    summary_meta = {
        "generated_at": progress["regen_at"],
        "eval_methodology": progress["eval_methodology"],
        "return_kind": return_kind,
        "note": f"Fast DuckDB panel RankIC from factor_lake (cached fwd {return_kind} returns)",
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
