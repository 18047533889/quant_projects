#!/usr/bin/env python3
"""校验 candidate_pool 下所有 manifest 公式。"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lib.dsl_validate import validate_formula  # noqa: E402


def main() -> int:
    campaign = sys.argv[1] if len(sys.argv) > 1 else "manual_ashare_pv_202607041600"
    pattern = str(PACKAGE_ROOT / "candidate_pool" / campaign / "manual_*/manifest.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"no manifests under {pattern}")
        return 1

    fail = 0
    for path in paths:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        ok, msg = validate_formula(manifest["formula"])
        if not ok:
            fail += 1
            print("FAIL", path, msg)
    print(f"validated {len(paths)} manifests, failures={fail}")
    return fail


if __name__ == "__main__":
    raise SystemExit(main())
