#!/usr/bin/env python3
"""输出 backend 覆盖摘要（CI / 本地审计）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--min-polars", type=int, default=320)
    parser.add_argument("--min-sql", type=int, default=95)
    parser.add_argument(
        "--write-doc",
        type=Path,
        default=None,
        help="写入 SQL 下推 canonical 清单 Markdown",
    )
    args = parser.parse_args()

    root = str(FE_ROOT.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    fe = str(FE_ROOT)
    if fe not in sys.path:
        sys.path.insert(0, fe)

    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry
    from backend.sql_pushdown.sql_registry import register_sql_backends, SQL_CAPABLE_CANONICALS

    load_all()
    register_sql_backends()
    canon = [c for c in OperatorRegistry.list_canonical() if OperatorRegistry.backends_for(c)]
    polars_n = sum(1 for c in canon if "polars" in OperatorRegistry.backends_for(c))
    sql_n = sum(1 for c in canon if "sql" in OperatorRegistry.backends_for(c))
    pandas_only = sorted(
        c for c in canon if "polars" not in OperatorRegistry.backends_for(c) and "pandas_numpy" in OperatorRegistry.backends_for(c)
    )
    report = {
        "canonical_implemented": len(canon),
        "polars": polars_n,
        "sql_registry": len(SQL_CAPABLE_CANONICALS),
        "sql_backends": sql_n,
        "pandas_only": len(pandas_only),
        "sql_canonicals": sorted(SQL_CAPABLE_CANONICALS - {"column", "literal"}),
        "ok": polars_n >= args.min_polars and sql_n >= args.min_sql,
    }
    if args.write_doc:
        lines = [
            "# SQL 下推覆盖清单",
            "",
            f"> 自动生成：`python scripts/report_backend_coverage.py --write-doc`",
            "",
            f"- canonical implemented: **{len(canon)}**",
            f"- Polars backend: **{polars_n}**（CI 门禁 ≥ {args.min_polars}）",
            f"- SQL 可下推 canonical: **{len(report['sql_canonicals'])}**（含 `column`/`literal` 共 **{len(SQL_CAPABLE_CANONICALS)}**）",
            f"- Registry sql backend: **{sql_n}**（CI 门禁 ≥ {args.min_sql}）",
            "",
            "## Canonical 列表",
            "",
        ]
        for name in report["sql_canonicals"]:
            lines.append(f"- `{name}`")
        lines.append("")
        args.write_doc.parent.mkdir(parents=True, exist_ok=True)
        args.write_doc.write_text("\n".join(lines), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            f"implemented={report['canonical_implemented']} polars={polars_n} "
            f"sql={sql_n} pandas_only={report['pandas_only']}"
        )
        if not report["ok"]:
            print(f"FAIL: polars>={args.min_polars} sql>={args.min_sql}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
