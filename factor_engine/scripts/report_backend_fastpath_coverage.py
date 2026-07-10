#!/usr/bin/env python3
"""Backend fast path coverage 报表（CI 强制检查 production core 全覆盖）。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]


def _bootstrap() -> None:
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


def _validate_strict(rows, *, require_block_reason: bool) -> list[str]:
    from backend.operator_capability import resolve_canonical
    from cleaned_operators.operator_spec import PRODUCTION_CORE_CANONICALS

    errors: list[str] = []
    by_canon = {r.canonical: r for r in rows}
    skip = {"column", "literal", "materialized_series", "plan_ref"}

    missing = sorted(
        c for c in PRODUCTION_CORE_CANONICALS if c not in skip and c not in by_canon
    )
    if missing:
        errors.append(f"PRODUCTION_CORE 未出现在 coverage report: {missing[:20]}")

    for r in rows:
        if r.sql_production_safe and not r.sql_emitter_ok:
            errors.append(f"{r.canonical}: SQL_PRODUCTION_SAFE 但 emitter 编译失败")

    if require_block_reason:
        for r in rows:
            if r.allow_in_production and not r.production_fast_path and not r.fastpath_block_reason:
                errors.append(
                    f"{r.canonical}: production_allowed 但 production_fast_path=False 且无 block_reason"
                )

    for c in PRODUCTION_CORE_CANONICALS:
        if c in skip:
            continue
        rc = resolve_canonical(c)
        if rc not in by_canon:
            continue
        row = by_canon[rc]
        if row.allow_in_production and not row.production_fast_path and not row.fastpath_block_reason:
            errors.append(f"PRODUCTION_CORE {rc}: 缺少 fastpath_block_reason")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="JSON 摘要")
    parser.add_argument("--csv", type=Path, default=None, help="写出 CSV 明细")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="CI：production core 全覆盖 + SQL_PRODUCTION_SAFE emitter_ok",
    )
    parser.add_argument(
        "--production-core-only",
        action="store_true",
        help="仅输出 PRODUCTION_CORE_CANONICALS",
    )
    args = parser.parse_args()

    _bootstrap()

    from backend.fastpath_coverage import (
        build_fastpath_coverage_matrix,
        summarize_fastpath_coverage,
    )
    from backend.operator_capability import resolve_canonical
    from cleaned_operators.operator_spec import PRODUCTION_CORE_CANONICALS

    if args.production_core_only:
        rows = build_fastpath_coverage_matrix(sorted(PRODUCTION_CORE_CANONICALS))
    else:
        rows = build_fastpath_coverage_matrix()

    summary = summarize_fastpath_coverage(rows)

    if args.csv:
        fieldnames = list(rows[0].to_csv_row().keys()) if rows else []
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row.to_csv_row())

    if args.json:
        payload = {
            "summary": summary,
            "rows": [r.to_csv_row() for r in rows],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            f"production_fast_path={summary['production_fast_path_count']} "
            f"production_allowed={summary['production_allowed_count']} "
            f"polars_native_prod={summary['polars_long_native_production_safe_count']} "
            f"sql_prod_emitter_ok={summary['sql_production_safe_emitter_ok']}"
        )
        if args.production_core_only or args.strict:
            core_rows = [
                r
                for r in rows
                if resolve_canonical(r.canonical) in PRODUCTION_CORE_CANONICALS
            ]
            print("canonical,production_allowed,polars_long_tier,duckdb_sql,parity,benchmark,production_fast_path,reason")
            for r in core_rows:
                print(
                    f"{r.canonical},{r.allow_in_production},{r.polars_long_tier},"
                    f"{r.duckdb_sql_status},{r.pandas_polars_long_parity or r.pandas_duckdb_parity},"
                    f"{r.benchmark_available},{r.production_fast_path},{r.fastpath_block_reason}"
                )

    if args.strict:
        errors = _validate_strict(rows, require_block_reason=True)
        if errors:
            print("\nSTRICT FAILURES:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
