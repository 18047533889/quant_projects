#!/usr/bin/env python3
"""导出 operator_manifest.json（AI / DSL / production gate 统一使用）。

必须运行：本 manifest 是从 live registry 派生的生成产物（1737 canonicals），
每次算子注册表变化（cleaned_operators 变更 / evidence 重算 / production
policy 变更）后都必须重新运行本脚本刷新 ``--out`` 指向的 JSON，严禁手工
编辑 manifest。CI 用 ``--check`` 校验 freshness（见
``tests/backend/test_operator_manifest_freshness.py``）；当 working tree 的
evidence 与已提交 manifest 不一致时，重新生成本 manifest 是正确的刷新路径
（manifest 是当前 registry 真相的序列化，证据重算后必须随之刷新）。

运行示例：
    PYTHONPATH=/home/sunhaiwei/quant_projects:/home/sunhaiwei/quant_projects/factor_engine \\
        .venv/bin/python scripts/export_operator_manifest.py --out benchmarks/operator_manifest.json
"""
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
    from factor_engine.cleaned_operators import load_all
    from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends

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
    from factor_engine.backend.operator_manifest import build_operator_manifest, write_operator_manifest

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
