#!/usr/bin/env python3
"""校验全部 Week2 DSL 公式。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lib.dsl_validate import validate_formula  # noqa: E402


def main() -> int:
    catalog_path = PACKAGE_ROOT / "source" / "week2_factors_catalog.json"
    if not catalog_path.exists():
        print("catalog missing; run build_catalog.py first")
        return 1

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    fail = 0
    for item in catalog["factors"]:
        ok, msg = validate_formula(item["formula"])
        if not ok:
            fail += 1
            print("FAIL", item["id"], msg)
    total = len(catalog["factors"])
    print(f"validated {total} formulas, failures={fail}")
    return fail


if __name__ == "__main__":
    raise SystemExit(main())
