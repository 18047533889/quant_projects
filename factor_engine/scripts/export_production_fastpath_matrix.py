#!/usr/bin/env python3
# -*- coding: utf-8
"""导出 production fast path 验收矩阵（Markdown / JSON）。"""
from __future__ import annotations

import argparse
import json
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


def _matrix_rows(canonicals):
    """为给定 canonical 列表构建 fast path 覆盖行。"""
    from factor_engine.backend.fastpath_coverage import build_fastpath_coverage_row

    return [build_fastpath_coverage_row(c) for c in sorted(canonicals)]


def _parity_cell(row) -> str:
    """将 pandas↔polars / pandas↔duckdb parity 状态渲染为 yes / partial / no。"""
    if row.pandas_polars_long_parity and row.pandas_duckdb_parity:
        return "yes"
    if row.pandas_polars_long_parity or row.pandas_duckdb_parity:
        return "partial"
    return "no"


def _markdown_table(rows) -> str:
    """将覆盖行渲染为 Markdown 表格。"""
    lines = [
        "| 算子 | PolarsLong native | DuckDB SQL | parity | production safe |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        prod = r.polars_long_native_production_safe or (r.sql_production_safe and r.sql_emitter_ok)
        lines.append(
            f"| `{r.canonical}` | "
            f"{'yes' if r.polars_long_native else 'no'} | "
            f"{'yes' if r.sql_production_safe and r.sql_emitter_ok else 'pending'} | "
            f"{_parity_cell(r)} | "
            f"{'yes' if prod else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """导出 production fast path 验收矩阵（stdout 或 --markdown / --json）。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()
    _bootstrap()

    from factor_engine.backend.production_fastpath_tiers import (
        P0_PRODUCTION_FASTPATH_CANONICALS,
        P1_BINARY_TS_CANONICALS,
        P1_GROUP_CANONICALS,
        P1_ROBUST_CANONICALS,
        P2_MAP_GROUPS_CANONICALS,
        P2_TECHNICAL_RESEARCH_ONLY,
    )

    batches = {
        "P0": sorted(P0_PRODUCTION_FASTPATH_CANONICALS),
        "P1_group": sorted(P1_GROUP_CANONICALS),
        "P1_robust": sorted(P1_ROBUST_CANONICALS),
        "P1_binary_ts": sorted(P1_BINARY_TS_CANONICALS),
        "P2_rolling": sorted(P2_MAP_GROUPS_CANONICALS),
        "P2_technical": sorted(P2_TECHNICAL_RESEARCH_ONLY),
    }
    all_canons = sorted({c for batch in batches.values() for c in batch})
    rows = _matrix_rows(all_canons)
    by_canon = {r.canonical: r for r in rows}

    if args.json:
        payload = {
            "batches": batches,
            "rows": [r.to_csv_row() for r in rows],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for label, canons in batches.items():
            print(f"\n## {label}\n")
            print(_markdown_table([by_canon[c] for c in canons if c in by_canon]))

    if args.markdown:
        parts = ["# Production Fast Path 验收矩阵\n"]
        for label, canons in batches.items():
            parts.append(f"\n## {label}\n")
            parts.append(_markdown_table([by_canon[c] for c in canons if c in by_canon]))
        args.markdown.write_text("".join(parts), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
