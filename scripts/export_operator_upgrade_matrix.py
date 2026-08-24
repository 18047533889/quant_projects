#!/usr/bin/env python3
"""导出 evidence/operator_upgrade_matrix.yaml。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = FE_ROOT / "evidence" / "operator_upgrade_matrix.yaml"
DEFAULT_JSON = FE_ROOT / "evidence" / "operator_upgrade_matrix.json"


def _bootstrap() -> None:
    root = str(FE_ROOT.parent)
    fe = str(FE_ROOT)
    for p in (root, fe):
        if p not in sys.path:
            sys.path.insert(0, p)
    from factor_engine.cleaned_operators import load_all
    from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    _bootstrap()

    from factor_engine.backend.operator_upgrade_matrix import build_upgrade_matrix

    doc = build_upgrade_matrix()
    if args.check:
        import yaml

        if not args.out.is_file():
            print(f"missing {args.out}", file=sys.stderr)
            return 1
        ref = yaml.safe_load(args.out.read_text(encoding="utf-8"))
        if ref.get("counts") != doc.get("counts") or ref.get("operators") != doc.get("operators"):
            print("operator_upgrade_matrix.yaml 过期", file=sys.stderr)
            return 1
        print(f"upgrade matrix fresh ({doc['counts']})")
        return 0

    try:
        import yaml
    except ImportError:
        print("需要 PyYAML", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    args.json.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({doc['counts']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
