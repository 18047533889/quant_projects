#!/usr/bin/env python3
"""从 factor_engine 导出 DSL 算子白名单到 dsl/fe_dsl_allowlist.json。

GTJA185 为 compat 包：默认导出 ``surface=compat``（含 daily + extended 兼容名）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lib.paths import resolve_factor_engine_root  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Export factor_engine DSL allowlist snapshot")
    parser.add_argument(
        "--surface",
        default="compat",
        choices=("daily", "compat", "extended"),
        help="算子面（GTJA185 默认 compat）",
    )
    parser.add_argument("--market", default="ashare")
    args = parser.parse_args()

    fe_root = resolve_factor_engine_root()
    if fe_root is None:
        print("factor_engine not found; set FACTOR_ENGINE_ROOT or place sibling ../factor_engine")
        return 1
    if str(fe_root) not in sys.path:
        sys.path.insert(0, str(fe_root))

    from api.mining_integration import export_dsl_allowlist_json  # noqa: WPS433

    payload = export_dsl_allowlist_json(market=args.market, surface=args.surface)
    out = PACKAGE_ROOT / "dsl" / "fe_dsl_allowlist.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"exported {payload['count']} operators (surface={args.surface}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
