#!/usr/bin/env python3
"""单算子 production 认证：parity → evidence → manifest。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]


def _bootstrap() -> None:
    root = str(FE_ROOT.parent)
    fe = str(FE_ROOT)
    for p in (root, fe):
        if p not in sys.path:
            sys.path.insert(0, p)
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical", help="算子 canonical，如 ts_mean / MOM")
    parser.add_argument(
        "--write",
        action="store_true",
        help="通过后写入 evidence JSON（须已有对应 parity test case）",
    )
    parser.add_argument("--no-manifest", action="store_true", help="跳过 manifest 导出")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    args = parser.parse_args()
    _bootstrap()

    from backend.operator_certification import run_certification

    report = run_certification(
        args.canonical,
        write_evidence=args.write,
        refresh_manifest=not args.no_manifest,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"certify {report.canonical}: ok={report.ok}")
        for s in report.stages:
            mark = "✓" if s.passed else "✗"
            print(f"  {mark} {s.stage.value}: {s.detail[:120]}")
        if report.wrote_evidence:
            print("  evidence: updated")
        print(f"  manifest: {'ok' if report.manifest_ok else 'failed'}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
