#!/usr/bin/env python3
"""Minimal demo: login → RunFactor → print RankIC-ish summary + sample values."""
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
    analysis_to_dict,
    factor_values_to_long_df,
    run_factor_formula,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="LQTP RunFactor demo")
    parser.add_argument("--server", default=os.getenv("LQTP_SERVER", DEFAULT_SERVER))
    parser.add_argument("--username", default=os.getenv("LQTP_USERNAME", ""))
    parser.add_argument("--password", default=os.getenv("LQTP_PASSWORD", ""))
    parser.add_argument("--formula", default="close / delay(close, 1) - 1")
    parser.add_argument("--begin", type=int, default=20240102)
    parser.add_argument("--end", type=int, default=20240131)
    parser.add_argument("--warmup", type=int, default=1)
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
        analyze=True,
        server=args.server,
    )
    if resp.error:
        raise RuntimeError(resp.error)

    long_df = factor_values_to_long_df(resp.values)
    print(f"days={long_df['trade_date'].nunique()} rows={len(long_df)} symbols={long_df['symbol'].nunique()}")
    if resp.analysis:
        print("analysis:", {k: analysis_to_dict(resp.analysis).get(k) for k in ("mean_ic", "icir", "long_short_sharpe")})
    print(long_df.head(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
