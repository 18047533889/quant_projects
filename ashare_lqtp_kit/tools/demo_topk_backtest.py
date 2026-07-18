#!/usr/bin/env python3
"""Minimal demo: RunFactor → TopK equal-weight OPEN backtest → print summary."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

KIT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT_ROOT))

from ashare_lqtp import (  # noqa: E402
    DEFAULT_SERVER,
    LqtpTokenManager,
    factor_values_to_long_df,
    run_factor_formula,
    run_topk_backtest_for_long_df,
    summarize_backtest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="LQTP TopK OPEN backtest demo")
    parser.add_argument("--server", default=os.getenv("LQTP_SERVER", DEFAULT_SERVER))
    parser.add_argument("--username", default=os.getenv("LQTP_USERNAME", ""))
    parser.add_argument("--password", default=os.getenv("LQTP_PASSWORD", ""))
    parser.add_argument(
        "--formula",
        default="-abs(close / delay(close, 1) - 1)",
        help="Higher factor value → more likely in TopK long book",
    )
    parser.add_argument("--begin", type=int, default=20240102)
    parser.add_argument("--end", type=int, default=20240329)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if not args.username or not args.password:
        print("Set --username/--password or LQTP_USERNAME / LQTP_PASSWORD", file=sys.stderr)
        return 2

    mgr = LqtpTokenManager.login(args.server, args.username, args.password)
    resp = run_factor_formula(
        token=mgr.token,
        formula=args.formula,
        begin_date=args.begin,
        end_date=args.end,
        warmup=args.warmup,
        analyze=False,
        server=args.server,
    )
    if resp.error:
        raise RuntimeError(resp.error)
    long_df = factor_values_to_long_df(resp.values)
    print(f"factor panel rows={len(long_df)} days={long_df['trade_date'].nunique()}")

    rows, bt_id, summary, excluded = run_topk_backtest_for_long_df(
        mgr,
        long_df,
        begin_date=args.begin,
        end_date=args.end,
        server=args.server,
    )
    print(f"backtest_id={bt_id!r} days={summary.get('trading_days')} excluded={len(excluded)}")
    for k in ("total_return", "annualized_return", "sharpe", "max_drawdown", "avg_turnover"):
        print(f"  {k}: {summary.get(k)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
