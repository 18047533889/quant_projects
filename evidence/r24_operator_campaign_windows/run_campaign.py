"""Thin R24 launcher for the unchanged, reviewed R23 execution harness."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--rows", type=int, default=96)
    args = parser.parse_args()
    if not 0 <= args.offset < 40:
        raise ValueError("offset must select within the 40 reviewed R24 recipes")
    if not 1 <= args.limit <= 30 or args.offset + args.limit > 40:
        raise ValueError("limit must be 1..30 and remain within the R24 catalog")
    root = Path(__file__).resolve().parents[2]
    harness = root / "evidence/r23_operator_campaign/campaign.py"
    recipes = root / "evidence/r24_operator_campaign_windows/recipes.json"
    ledger = root / "evidence/r24_operator_campaign_windows/ledger.jsonl"
    sys.argv = [
        str(harness), "--recipes", str(recipes), "--ledger", str(ledger),
        "--selection", "recipes", "--offset", str(args.offset),
        "--limit", str(args.limit), "--rows", str(args.rows),
    ]
    runpy.run_path(str(harness), run_name="__main__")


if __name__ == "__main__":
    main()
