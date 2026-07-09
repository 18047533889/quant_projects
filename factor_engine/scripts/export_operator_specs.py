#!/usr/bin/env python3
"""导出算子 OperatorSpec manifest（JSON / YAML）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap() -> None:
    root = Path(__file__).resolve().parents[1]
    project = root.parent
    for p in (str(root), str(project)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from cleaned_operators import load_all
    from runtime.env_bootstrap import bootstrap_runtime_env

    bootstrap_runtime_env()
    load_all()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出算子 manifest")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "operator_specs_manifest.json",
        help="输出路径（.json 或 .yaml）",
    )
    parser.add_argument("--production-only", action="store_true", help="仅 production 允许算子")
    parser.add_argument("--core-only", action="store_true", help="仅 PRODUCTION_CORE_CANONICALS")
    args = parser.parse_args(argv)

    _bootstrap()
    from cleaned_operators.operator_spec import export_operator_manifest

    entries = export_operator_manifest(
        production_only=args.production_only,
        core_only=args.core_only,
    )
    out: Path = args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() in (".yaml", ".yml"):
        import yaml

        out.write_text(
            yaml.safe_dump(entries, allow_unicode=True, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
    else:
        out.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {out} ({len(entries)} operators)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
