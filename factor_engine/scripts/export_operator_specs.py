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
    parser.add_argument(
        "--check",
        action="store_true",
        help="与 --core-only 联用：校验 docs/operator_core_specs.yaml 未过期",
    )
    args = parser.parse_args(argv)

    _bootstrap()
    from cleaned_operators.operator_spec import export_operator_manifest

    entries = export_operator_manifest(
        production_only=args.production_only,
        core_only=args.core_only,
    )
    if args.check:
        if not args.core_only:
            print("--check 须与 --core-only 联用", file=sys.stderr)
            return 2
        ref_path = Path(__file__).resolve().parents[1] / "docs" / "operator_core_specs.yaml"
        if not ref_path.is_file():
            print(f"缺少参考 manifest: {ref_path}", file=sys.stderr)
            return 1
        import yaml

        ref = yaml.safe_load(ref_path.read_text(encoding="utf-8")) or []
        ref_names = {e["name"] for e in ref if isinstance(e, dict) and e.get("name")}
        got_names = {e["name"] for e in entries}
        if ref_names != got_names:
            missing = sorted(ref_names - got_names)
            extra = sorted(got_names - ref_names)
            print(
                f"operator_core_specs.yaml 过期: missing={missing[:5]} extra={extra[:5]}",
                file=sys.stderr,
            )
            return 1
        print(f"operator_core_specs.yaml 与 export 一致 ({len(got_names)} ops)")
        return 0

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
