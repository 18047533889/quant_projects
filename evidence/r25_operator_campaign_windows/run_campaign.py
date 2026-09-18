"""Thin R25 launcher for the unchanged R23 execution harness."""

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
    if not 0 <= args.offset < 20:
        raise ValueError("offset must select within the 20 reviewed R25 recipes")
    if not 1 <= args.limit <= 20 or args.offset + args.limit > 20:
        raise ValueError("limit must remain within the R25 catalog")
    root = Path(__file__).resolve().parents[2]
    harness = root / "evidence/r23_operator_campaign/campaign.py"
    sys.argv = [str(harness), "--recipes", str(Path(__file__).with_name("recipes.json")),
                "--ledger", str(Path(__file__).with_name("ledger.jsonl")),
                "--selection", "recipes", "--offset", str(args.offset),
                "--limit", str(args.limit), "--rows", str(args.rows)]
    runpy.run_path(str(harness), run_name="__main__")


if __name__ == "__main__":
    main()
