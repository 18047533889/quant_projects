#!/usr/bin/env python3
"""Re-run LQTP platform eval + TopK for one factor (no DuckDB, no symbol-retry hell)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cogalpha_lqtp.factor_eval import evaluate_lqtp_formula, fetch_lqtp_open_returns  # noqa: E402
from scripts.cogalpha_lqtp.lqtp_client import (  # noqa: E402
    DEFAULT_SERVER,
    LqtpTokenManager,
    factor_values_to_long_df,
    run_topk_backtest_for_long_df,
    summarize_backtest,
)
from scripts.cogalpha_lqtp.report_html import (  # noqa: E402
    render_factor_report,
    render_index,
    render_production_summary,
)
from scripts.cogalpha_lqtp.run_light_test import _yyyymmdd  # noqa: E402
from scripts.cogalpha_lqtp.run_production_batch import (  # noqa: E402
    _apply_flip_state_to_catalog,
    _json_float,
    _normalize_lqtp_symbol,
    _save_progress,
    _upsert_index_row,
)


def _backtest_safe_symbols(
    *,
    token: str,
    begin_date: int,
    end_date: int,
    server: str,
    cache_path: Path,
    universe: set[str],
) -> set[str]:
    returns_long = fetch_lqtp_open_returns(
        token=token,
        begin_date=begin_date,
        end_date=end_date,
        server=server,
        cache_path=cache_path,
    )
    symbols = set(returns_long["symbol"].astype(str).map(_normalize_lqtp_symbol))
    if universe:
        symbols &= universe
    return symbols


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/cogalpha_lqtp_production")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--username", default=os.getenv("LQTP_USERNAME", "james.gd.luo@gmail.com"))
    parser.add_argument("--password", default=os.getenv("LQTP_PASSWORD", "3213709208"))
    args = parser.parse_args()

    work = args.work_dir
    progress_path = work / "production_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    catalog = json.loads((work / "dsl_catalog.json").read_text(encoding="utf-8"))
    parsed = {r["function_name"]: r.get("python_code", "") for r in json.loads((work / "parsed_factors.json").read_text())}
    flip = progress.get("flip_state", {})
    _apply_flip_state_to_catalog(catalog, flip, parsed)
    entry = next(e for e in catalog if e["function_name"] == args.name)
    dsl = (entry.get("dsl") or "").strip()
    if flip.get(args.name, {}).get("dsl"):
        dsl = flip[args.name]["dsl"]
    py = parsed.get(args.name, "")

    begin_i = _yyyymmdd(args.start)
    end_i = _yyyymmdd(args.end)
    auth = LqtpTokenManager.login(args.server, args.username, args.password)

    bad_cache = work / "backtest_missing_quote_symbols.json"
    pre_excluded: set[str] = set()
    if bad_cache.exists():
        pre_excluded = set(json.loads(bad_cache.read_text(encoding="utf-8")))
    else:
        log_bad = work / "production.log"
        if log_bad.exists():
            import re

            pre_excluded = set(
                re.findall(r"missing quote symbol ([0-9]{6}\.[A-Z]{2})", log_bad.read_text(encoding="utf-8", errors="ignore"))
            )

    print(f"{args.name}: LQTP RunFactor analyze=True (platform only)")
    analysis, eval_mode, daily_values = evaluate_lqtp_formula(
        token=auth.token,
        formula=dsl,
        begin_date=begin_i,
        end_date=end_i,
        server=args.server,
        fwd_returns_path=None,
    )
    rows = sum(len(p.values) for p in daily_values)
    lqtp_long = factor_values_to_long_df(daily_values)

    bt_cache = work / "lqtp_open_returns_cache.parquet"
    bt_symbols = _backtest_safe_symbols(
        token=auth.token,
        begin_date=begin_i,
        end_date=end_i,
        server=args.server,
        cache_path=bt_cache,
        universe=set(),
    )
    backtest_rows, backtest_id, topk_summary, excluded = run_topk_backtest_for_long_df(
        auth,
        lqtp_long,
        begin_date=begin_i,
        end_date=end_i,
        server=args.server,
        allowed_symbols=bt_symbols,
        open_returns_long=pd.read_parquet(bt_cache) if bt_cache.exists() else None,
        pre_excluded=set(),
    )
    if backtest_rows:
        eval_mode = f"{eval_mode}+topk_open"

    report_path = work / "reports" / f"{args.name}.html"
    render_factor_report(
        factor_name=args.name,
        dsl=dsl,
        analysis=analysis,
        backtest_rows=backtest_rows,
        out_path=report_path,
        eval_mode=eval_mode,
        materialize_meta={
            "engine": "lqtp_dsl",
            "eval_route": "lqtp_dsl",
            "rows": rows,
            "date_range": [args.start, args.end],
            "formula": dsl,
        },
        topk_summary=topk_summary,
        python_code=py,
        work_dir=work,
        engine="lqtp_dsl",
        eval_route="lqtp_dsl",
    )

    row: dict[str, Any] = {
        "factor_name": args.name,
        "report": report_path.name,
        "engine": "lqtp_dsl",
        "eval_route": "lqtp_dsl",
        "eval_mode": eval_mode,
        "ic_sign_flipped": bool(entry.get("ic_sign_flipped")),
        "mean_ic": _json_float(analysis.get("mean_rank_ic", analysis.get("mean_ic"))),
        "mean_rank_ic": _json_float(analysis.get("mean_rank_ic", analysis.get("mean_ic"))),
        "icir": _json_float(analysis.get("rank_icir", analysis.get("icir"))),
        "rank_icir": _json_float(analysis.get("rank_icir", analysis.get("icir"))),
        "ic_positive_ratio": _json_float(
            analysis.get("rank_ic_positive_ratio", analysis.get("ic_positive_ratio"))
        ),
        "rank_ic_positive_ratio": _json_float(
            analysis.get("rank_ic_positive_ratio", analysis.get("ic_positive_ratio"))
        ),
        "coverage": _json_float(analysis.get("coverage")),
        "long_short_sharpe": _json_float(analysis.get("long_short_sharpe")),
        "long_short_return": _json_float(analysis.get("long_short_return")),
        "backtest_total_ret": _json_float(topk_summary.get("total_return")),
        "backtest_ann_ret": _json_float(topk_summary.get("annualized_return")),
        "backtest_sharpe": _json_float(topk_summary.get("sharpe")),
        "backtest_max_drawdown": _json_float(topk_summary.get("max_drawdown")),
        "backtest_win_rate": _json_float(topk_summary.get("win_rate")),
        "backtest_avg_turnover": _json_float(topk_summary.get("avg_turnover")),
        "backtest_volatility": _json_float(topk_summary.get("volatility")),
        "backtest_calmar": _json_float(topk_summary.get("calmar")),
        "backtest_trading_days": int(topk_summary.get("trading_days") or 0),
        "backtest_final_nav": _json_float(topk_summary.get("final_nav")),
        "rows": rows,
        "backtest_id": backtest_id,
        "return_kind": str(analysis.get("return_kind", "")),
    }
    index_rows = list(progress.get("index_rows", []))
    _upsert_index_row(index_rows, row)
    progress["index_rows"] = index_rows
    _save_progress(progress_path, progress)
    render_index(index_rows, work / "reports" / "index.html")
    render_production_summary(
        rows=index_rows,
        out_path=work / "reports" / "production_summary.html",
        meta={"catalog_total": 176, "date_range": [args.start, args.end]},
        failed=progress.get("failed", {}),
    )
    sh = row["backtest_sharpe"]
    sh_s = f"{sh:.3f}" if sh == sh else "nan"
    print(
        f"ok rank_ic={row['mean_rank_ic']:.4f} ls_sharpe={row['long_short_sharpe']:.3f} "
        f"topk_sharpe={sh_s} bt_days={row['backtest_trading_days']}"
    )
    return 0 if backtest_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
