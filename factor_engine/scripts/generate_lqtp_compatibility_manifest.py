#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.lqtp_capabilities import build_lqtp_capability_manifest

DEFAULT_OUTPUT = ROOT / "docs" / "lqtp_compatibility_manifest.json"


def render() -> str:
    return json.dumps(build_lqtp_capability_manifest(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != expected:
            print(f"stale LQTP compatibility manifest: {args.output}", file=sys.stderr)
            return 1
        print(f"LQTP compatibility manifest fresh: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
