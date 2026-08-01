#!/usr/bin/env python3
"""Sample a reproducible cold-start batch as JSON."""
from __future__ import annotations

import argparse
import json

from _bootstrap import REPO_ROOT  # noqa: F401
from factor_cold_start.sampler import sample_factors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True, choices=("ashare", "us"))
    parser.add_argument("--surface", default="daily", choices=("daily", "extended", "research"))
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--seed", default="0")
    parser.add_argument("--tier", action="append", dest="tiers")
    parser.add_argument("--family", action="append", dest="families")
    parser.add_argument("--max-per-family", type=int)
    args = parser.parse_args()
    rows = sample_factors(
        market=args.market,
        surface=args.surface,
        size=args.size,
        seed=args.seed,
        availability_tiers=args.tiers,
        families=args.families,
        max_per_family=args.max_per_family,
    )
    print(json.dumps([row.to_dict() for row in rows], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
