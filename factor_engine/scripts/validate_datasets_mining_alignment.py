#!/usr/bin/env python3
"""校验 mining preset / prod profile 与 data_access datasets.yaml 契约一致。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """校验 mining preset / prod profile 与 data_access datasets.yaml 契约一致。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default="prod",
        help="要审计的 runtime profile 名（默认 prod）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出 JSON（默认人类可读摘要）",
    )
    args = parser.parse_args()

    root = str(FE_ROOT.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    fe = str(FE_ROOT)
    if fe not in sys.path:
        sys.path.insert(0, fe)

    from factor_engine.api.datasets_contract import audit_full_datasets_contract

    report = audit_full_datasets_contract(profile_name=args.profile)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        mining = report["mining"]
        prod = report["prod_profile"]
        print(f"mining datasets: {', '.join(mining['datasets_checked'])}")
        print(f"prod profile={prod['profile']} target={prod['materialization_target']}")
        if report["ok"]:
            print("OK: datasets.yaml 与 mining/prod 契约一致")
        else:
            print("FAIL:")
            for v in report["violations"]:
                print(f"  - {v}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
