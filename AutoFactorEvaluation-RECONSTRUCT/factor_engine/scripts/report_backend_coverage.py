#!/usr/bin/env python3
"""输出 backend 覆盖摘要（CI / 本地审计）。

路由原则：SQL 子树下推 → Polars parity 已验证 → Pandas fallback（可观测）。

能力矩阵见 ``backend.operator_capability``。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]


def _bootstrap():
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


def _build_rows() -> list[dict[str, object]]:
    """构建每个 canonical 的后端能力扁平行（pandas / polars / SQL 等）。"""
    from backend.operator_capability import build_capability_matrix, polars_expr_capable, resolve_canonical

    rows: list[dict[str, object]] = []
    for summary in build_capability_matrix():
        rows.append(
            {
                "canonical": summary.canonical,
                "allow_in_production": summary.allow_in_production,
                "pandas": summary.pandas_numpy,
                "polars": summary.polars,
                "duckdb_sql": summary.duckdb_sql,
                "clickhouse_sql": summary.clickhouse_sql,
                "parity_verified": summary.parity_verified,
                "production_polars_safe": summary.polars == "production_safe",
                "polars_expr_capable": polars_expr_capable(summary.canonical),
                "sql_capable": summary.duckdb_sql != "unsupported",
                "sql_emitter_ok": summary.duckdb_sql
                in {"parity_verified", "production_safe"},
            }
        )
    return rows


def main() -> int:
    """输出 backend 三层覆盖摘要，可选写入 Markdown 文档或 CSV 明细。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--detail",
        action="store_true",
        help="输出每个 canonical 的 backend status CSV",
    )
    parser.add_argument(
        "--capability-csv",
        action="store_true",
        help="输出 canonical×backend 扁平行（status/speedup）",
    )
    parser.add_argument("--min-polars", type=int, default=315)
    parser.add_argument("--min-sql", type=int, default=95)
    parser.add_argument(
        "--write-doc",
        type=Path,
        default=None,
        help="写入 SQL 下推 canonical 清单 Markdown",
    )
    parser.add_argument(
        "--write-backend-doc",
        type=Path,
        default=None,
        help="写入 backend 三层覆盖 Markdown",
    )
    parser.add_argument(
        "--production-fast-path",
        action="store_true",
        help="输出 production fast path（三后端 parity）摘要",
    )
    args = parser.parse_args()

    _bootstrap()

    from cleaned_operators.operator_policy import (
        POLARS_PARITY_VERIFIED,
        POLARS_PRODUCTION_SAFE,
        polars_implemented_canonicals,
    )
    from cleaned_operators.operator_spec import PRODUCTION_CORE_CANONICALS
    from cleaned_operators.registry import OperatorRegistry
    from backend.sql_pushdown.sql_registry import SQL_CAPABLE_CANONICALS
    from backend.polars_long_policy import get_polars_long_capable, POLARS_LONG_NATIVE
    from backend.production_fast_path import summarize_production_fast_path
    from backend.operator_capability import resolve_canonical

    rows = _build_rows()
    canon = [r["canonical"] for r in rows]
    sql_ok_canons = {
        r["canonical"] for r in rows if r["sql_emitter_ok"]
    } - {"column", "literal"}
    polars_long_canons = get_polars_long_capable() - {"column", "literal"}
    polars_long_sql_both = sorted(polars_long_canons & sql_ok_canons)
    polars_long_only = sorted(polars_long_canons - sql_ok_canons)
    sql_only_vs_polars_long = sorted(sql_ok_canons - polars_long_canons)
    polars_n = sum(1 for r in rows if r["polars"] != "unsupported")
    sql_n = sum(1 for r in rows if r["sql_emitter_ok"])
    prod_polars_safe_n = sum(1 for r in rows if r["production_polars_safe"])
    polars_expr_n = sum(1 for r in rows if r["polars_expr_capable"])
    pandas_only = sorted(
        r["canonical"]
        for r in rows
        if r["pandas"] != "unsupported" and r["polars"] == "unsupported"
    )
    prod_allowed_pandas_only = sorted(
        resolve_canonical(r["canonical"])
        for r in rows
        if r["allow_in_production"] and not r["production_polars_safe"]
    )
    prod_allowed_pandas_only = sorted(set(prod_allowed_pandas_only))
    fast_path = summarize_production_fast_path()
    polars_long_native_n = len(
        POLARS_LONG_NATIVE - {"column", "literal", "materialized_series", "plan_ref"}
    )
    has_polars_not_safe = sorted(
        r["canonical"]
        for r in rows
        if r["polars"] not in {"unsupported", "production_safe"}
    )
    report = {
        "canonical_implemented": len(canon),
        "polars_implemented": len(polars_implemented_canonicals()),
        "polars_production_safe": prod_polars_safe_n,
        "polars_expr_capable": polars_expr_n,
        "polars_long_capable": len(polars_long_canons),
        "polars_long_sql_both": polars_long_sql_both,
        "polars_long_only": polars_long_only,
        "sql_only_vs_polars_long": sql_only_vs_polars_long[:30],
        "sql_only_vs_polars_long_count": len(sql_only_vs_polars_long),
        "parity_verified": len(POLARS_PARITY_VERIFIED),
        "production_core": len(PRODUCTION_CORE_CANONICALS),
        "production_core_polars_gap": sorted(
            PRODUCTION_CORE_CANONICALS - POLARS_PRODUCTION_SAFE
        ),
        "polars_long_native": polars_long_native_n,
        "production_fast_path": fast_path["production_fast_path_count"],
        "production_fast_path_ops": fast_path["production_fast_path"],
        "triple_parity_duckdb": fast_path["triple_parity_duckdb_count"],
        "production_core_fast_path_gap_count": fast_path["production_core_fast_path_gap_count"],
        "polars": polars_n,
        "sql_registry": len(SQL_CAPABLE_CANONICALS),
        "sql_emitter_ok": sql_n,
        "pandas_only": len(pandas_only),
        "production_allowed_pandas_only": prod_allowed_pandas_only,
        "has_polars_not_production_safe": has_polars_not_safe[:20],
        "has_polars_not_production_safe_count": len(has_polars_not_safe),
        "sql_canonicals": sorted(SQL_CAPABLE_CANONICALS - {"column", "literal"}),
        "ok": polars_n >= args.min_polars and sql_n >= args.min_sql,
    }

    if args.detail:
        writer = csv.DictWriter(
            sys.stdout,
            fieldnames=[
                "canonical",
                "allow_in_production",
                "pandas",
                "polars",
                "duckdb_sql",
                "clickhouse_sql",
                "parity_verified",
                "production_polars_safe",
                "polars_expr_capable",
                "sql_capable",
                "sql_emitter_ok",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    if args.capability_csv:
        from backend.operator_capability import export_flat_capabilities

        flat = export_flat_capabilities()
        writer = csv.DictWriter(
            sys.stdout,
            fieldnames=[
                "canonical",
                "backend",
                "status",
                "estimated_speedup",
                "supports_nulls",
                "supports_min_periods",
                "supports_group",
                "supports_window",
                "notes",
            ],
        )
        writer.writeheader()
        for cap in flat:
            writer.writerow(cap.to_csv_row())

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

    if args.write_backend_doc:
        lines = [
            "# Backend 三层覆盖清单",
            "",
            "> 自动生成：`python scripts/report_backend_coverage.py --write-backend-doc docs/backend_coverage.md`",
            "",
            "## 摘要",
            "",
            f"- runtime 已实现: **{report['canonical_implemented']}**",
            f"- Polars 注册: **{report['polars_implemented']}**",
            f"- parity verified: **{report['parity_verified']}**",
            f"- production Polars safe: **{report['polars_production_safe']}**",
            f"- Polars expr long-table: **{report['polars_expr_capable']}**",
            f"- Polars long ∩ SQL emitter: **{len(report['polars_long_sql_both'])}**",
            f"- Polars long only (无 SQL): **{len(report['polars_long_only'])}**",
            f"- SQL only (无 Polars long): **{report['sql_only_vs_polars_long_count']}**",
            f"- PRODUCTION_CORE: **{report['production_core']}**（Polars gap: {report['production_core_polars_gap'] or '无'}）",
            f"- PolarsLong native: **{report['polars_long_native']}**",
            f"- Production fast path（三后端 parity）: **{report['production_fast_path']}**",
            f"- DuckDB triple parity: **{report['triple_parity_duckdb']}**",
            f"- PRODUCTION_CORE fast path gap: **{report['production_core_fast_path_gap_count']}**",
            f"- production 允许但仅 pandas: **{len(prod_allowed_pandas_only)}**",
            f"- 有 Polars 未进 production safe: **{report['has_polars_not_production_safe_count']}**",
            "",
            "## 晋级路径",
            "",
            "```text",
            "runtime polars 注册 → parity CI → POLARS_PARITY_VERIFIED → POLARS_PRODUCTION_SAFE",
            "```",
            "",
        ]
        if prod_allowed_pandas_only:
            lines.extend(["## production 允许但 production auto 仍走 pandas", ""])
            for name in prod_allowed_pandas_only[:30]:
                lines.append(f"- `{name}`")
            if len(prod_allowed_pandas_only) > 30:
                lines.append(f"- … 共 {len(prod_allowed_pandas_only)} 个")
            lines.append("")
        args.write_backend_doc.parent.mkdir(parents=True, exist_ok=True)
        args.write_backend_doc.write_text("\n".join(lines), encoding="utf-8")

    if args.production_fast_path:
        fp = fast_path
        print(
            f"production_fast_path={fp['production_fast_path_count']} "
            f"triple_parity={fp['triple_parity_count']} "
            f"duckdb_triple={fp['triple_parity_duckdb_count']} "
            f"core_gap={fp['production_core_fast_path_gap_count']}"
        )
        for name in fp["production_fast_path"]:
            print(f"  {name}")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif not args.detail and not args.capability_csv:
        print(
            f"implemented={report['canonical_implemented']} "
            f"polars={polars_n} parity={report['parity_verified']} "
            f"prod_safe={report['polars_production_safe']} "
            f"polars_long={report['polars_long_capable']} "
            f"polars_long_native={report['polars_long_native']} "
            f"fast_path={report['production_fast_path']} "
            f"sql_emitter_ok={sql_n} pandas_only={report['pandas_only']}"
        )
        if report["production_core_polars_gap"]:
            print(f"WARN core_polars_gap={report['production_core_polars_gap']}", file=sys.stderr)
        if not report["ok"]:
            print(f"FAIL: polars>={args.min_polars} sql>={args.min_sql}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
