#!/usr/bin/env python3
"""为 185 条 GTJA-191 因子生成 factor_engine 落值 YAML（examples/materialize/）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML required: pip install pyyaml") from exc

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lib.catalog import deliverable_catalog, deliverable_names  # noqa: E402
from lib.materialize_config import (  # noqa: E402
    CONFIG_DIR,
    build_materialize_config,
    materialize_config_path,
)


def generate_configs(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    lake_root: str | None = None,
    write_target: str = "local",
    factor: str | None = None,
) -> list[Path]:
    catalog = deliverable_catalog()
    names = [factor] if factor else deliverable_names(catalog)
    if factor and factor not in catalog:
        raise SystemExit(f"unknown deliverable factor: {factor}")

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in names:
        item = catalog[name]
        cfg = build_materialize_config(
            name,
            item["dsl_formula"],
            start_date=start_date,
            end_date=end_date,
            lake_root=lake_root,
            write_target=write_target,
        )
        path = materialize_config_path(name)
        path.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate GTJA-191 materialize YAML configs")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--lake-root", default=None)
    parser.add_argument("--write-target", default="local")
    parser.add_argument("--factor", default=None, help="single factor name")
    args = parser.parse_args()

    paths = generate_configs(
        start_date=args.start_date,
        end_date=args.end_date,
        lake_root=args.lake_root,
        write_target=args.write_target,
        factor=args.factor,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "count": len(paths),
                "config_dir": str(CONFIG_DIR),
                "sample": str(paths[0]) if paths else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
