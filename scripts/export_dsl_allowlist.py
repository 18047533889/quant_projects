#!/usr/bin/env python3
"""导出 factor_engine DSL 算子白名单（AFV ``afv_us_pv_daily`` 对齐）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

FE = Path(__file__).resolve().parents[1]
if str(FE) not in sys.path:
    sys.path.insert(0, str(FE))

from factor_engine.api.mining_integration import write_dsl_allowlist  # noqa: E402


def main() -> int:
    """导出 factor_engine DSL 算子白名单 JSON。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=FE / "factor_engine" / "docs" / "dsl_allowlist.json",
        help="输出 JSON 路径",
    )
    args = parser.parse_args()
    out = write_dsl_allowlist(args.output)
    print(f"Wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
