#!/usr/bin/env python3
"""Compile-only audit for the sole 1288 single-default fundamental library."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]
if str(FE_ROOT) not in sys.path:
    sys.path.insert(0, str(FE_ROOT))

from fundamental_cold_start import EXPECTED_FILENAME, load_cold_start


def audit_row(row: dict[str, str]) -> dict[str, object]:
    from api.dsl_parser import parse_expr
    from ir.analyzer import Analyzer
    from fundamental_rewrites import rewrite_for
    from cleaned_operators.registry import OperatorRegistry

    factor_id = row["factor_id"]
    rewrite = rewrite_for(factor_id)
    formula = rewrite.expression if rewrite and rewrite.expression else row["dsl"]
    result: dict[str, object] = {
        "factor_id": factor_id,
        "formula": formula,
        "original_formula": row["dsl"],
        "csv_implementation_status": row.get("implementation_status", ""),
        "csv_pit_class": row.get("pit_class", ""),
        "status": "failed",
    }
    if rewrite is not None:
        result.update({"rewrite_status": rewrite.status, "rewrite_reason": rewrite.reason})
        if rewrite.expression is None:
            result["status"] = rewrite.status
            return result
    try:
        expr = parse_expr(formula, surface="compat_research", dialect="lqtp", dialect_version="2026-07-19")
        analysis = Analyzer().lower(expr)
        from cleaned_operators.registry import OperatorRegistry
        from backend.sql_tiers import is_sql_implemented
        operator_backends = {}
        for operator in sorted(set(str(value) for value in str(row.get("operators", "")).split("|") if value)):
            canonical = OperatorRegistry.resolve_canonical_optional(operator)
            operator_backends[operator] = {
                "canonical": canonical,
                "registered_backends": sorted(OperatorRegistry._operators.get(canonical, {})),
                "sql_implemented": bool(is_sql_implemented(canonical)),
            }
        result.update({
            "status": "parseable",
            "referenced_columns": sorted(analysis.referenced_columns),
            "lookback": int(analysis.lookback),
            "requires_full_history": bool(analysis.requires_full_history),
            "operator_backends": operator_backends,
            "field_resolution_status": "requires_runtime_schema" if analysis.referenced_columns else "no_fields",
            "runtime_status": "not_executed_compile_audit",
        })
    except Exception as exc:
        error = str(exc)
        csv_status = str(row.get("implementation_status", ""))
        if isinstance(exc, Exception) and "Unsupported function:" in error and csv_status in {
            "requires_operator_extension", "requires_source_preprocessing", "requires_data_field",
        }:
            result.update({
                "status": "unavailable" if csv_status == "requires_operator_extension" else csv_status,
                "error_type": type(exc).__name__,
                "error": error,
                "unavailable_reason": "runtime_or_preprocessing_not_certified",
            })
        else:
            result.update({"error_type": type(exc).__name__, "error": error})
    return result


def main() -> int:
    from cleaned_operators import load_all
    load_all()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=FE_ROOT.parent / EXPECTED_FILENAME)
    parser.add_argument("--output", type=Path, default=FE_ROOT / "evidence" / "fundamental_1288_compile_report.json")
    args = parser.parse_args()
    rows, source_report = load_cold_start(args.source)
    if source_report.status != "available":
        payload = {"schema_version": "factor_engine.fundamental_1288_audit.v1", "source": source_report.as_dict(), "entries": []}
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload["source"], ensure_ascii=False))
        return 1
    entries = [audit_row(row) for row in rows]
    payload = {
        "schema_version": "factor_engine.fundamental_1288_audit.v2",
        "source": source_report.as_dict(),
        "entry_count": len(entries),
        "status_counts": dict(Counter(str(entry["status"]) for entry in entries)),
        "error_counts": dict(Counter(str(entry.get("error_type", "")) for entry in entries if entry["status"] == "failed")),
        "entries": entries,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("entry_count", "status_counts", "error_counts")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
