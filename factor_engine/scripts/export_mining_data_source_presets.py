#!/usr/bin/env python3
"""导出 mining 默认 data_source preset（契约审计 / 文档生成）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = FE_ROOT / "docs" / "mining_data_source_presets.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"输出 JSON 路径（默认 {DEFAULT_OUT.relative_to(FE_ROOT)}）",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="仅校验已提交 JSON 与当前 preset 一致（CI 门禁）",
    )
    args = parser.parse_args()

    root = str(FE_ROOT.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    fe = str(FE_ROOT)
    if fe not in sys.path:
        sys.path.insert(0, fe)

    from api.mining_integration import default_mining_data_source_presets

    presets = default_mining_data_source_presets()
    payload = json.dumps(presets, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not args.output.is_file():
            print(f"FAIL: missing {args.output}", file=sys.stderr)
            return 1
        if args.output.read_text(encoding="utf-8") != payload:
            print(
                f"FAIL: {args.output} out of date; run export_mining_data_source_presets.py",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {args.output.name} synced ({len(presets)} presets)")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(f"Wrote {len(presets)} presets → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
