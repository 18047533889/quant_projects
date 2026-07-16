#!/usr/bin/env python3
"""导出 operator_manifest.json（AI / DSL / production gate 统一使用）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]


def _bootstrap() -> None:
    """初始化 sys.path 并加载算子注册表与 SQL backend。"""
    root = str(FE_ROOT.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    fe = str(FE_ROOT)
    if fe not in sys.path:
        sys.path.insert(0, fe)
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def main() -> int:
    """生成或校验 operator_manifest.json（AI / DSL / production gate 统一清单）。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=FE_ROOT / "benchmarks" / "operator_manifest.json",
    )
    parser.add_argument("--production-only", action="store_true")
    parser.add_argument("--fastpath-only", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="校验 --out 与当前 build 一致（manifest freshness CI）",
    )
    args = parser.parse_args()
    _bootstrap()
    from backend.operator_manifest import build_operator_manifest, write_operator_manifest

    if args.check:
        import json

        ref_path = args.out
        if not ref_path.is_file():
            print(f"缺少参考 manifest: {ref_path}", file=sys.stderr)
            return 1
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        got_ops = build_operator_manifest(
            production_only=args.production_only,
            fastpath_only=args.fastpath_only,
        )
        ref_ops = ref.get("operators", [])
        if ref.get("schema_version") != 2:
            print(
                f"operator_manifest.json 过期: schema_version={ref.get('schema_version')!r} (expected 2)",
                file=sys.stderr,
            )
            return 1
        if ref_ops != got_ops:
            ref_names = {e.get("canonical") for e in ref_ops}
            got_names = {e.get("canonical") for e in got_ops}
            print(
                f"operator_manifest.json 过期: "
                f"missing={sorted(ref_names - got_names)[:5]} "
                f"extra={sorted(got_names - ref_names)[:5]}",
                file=sys.stderr,
            )
            return 1
        print(f"manifest fresh: {ref_path}")
        return 0

    write_operator_manifest(
        args.out,
        production_only=args.production_only,
        fastpath_only=args.fastpath_only,
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
