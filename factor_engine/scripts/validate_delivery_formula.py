#!/usr/bin/env python3
"""投递前公式校验 CLI（挖掘框架 / CI 调用）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FE = Path(__file__).resolve().parents[1]
ROOT = FE.parent
if str(FE) not in sys.path:
    sys.path.insert(0, str(FE))

from api.mining_integration import validate_manifest_for_execution  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("formula", nargs="?", help="单行 DSL 公式")
    parser.add_argument("--market", default="us_stock")
    parser.add_argument("--expression-type", default="dsl")
    parser.add_argument("--manifest", type=Path, help="manifest.json 路径")
    args = parser.parse_args()

    if args.manifest:
        data = json.loads(args.manifest.read_text(encoding="utf-8"))
        formula = data.get("formula", "")
        market = data.get("market") or args.market
        expr_type = data.get("expression_type") or args.expression_type
    else:
        formula = args.formula or ""
        market = args.market
        expr_type = args.expression_type

    ok, msg = validate_manifest_for_execution(
        market=market,
        expression_type=expr_type,
        formula=formula,
    )
    if ok:
        print(f"OK: {msg}")
        return 0
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
