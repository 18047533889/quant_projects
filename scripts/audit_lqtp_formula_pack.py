#!/usr/bin/env python3
"""Offline compatibility audit for a directory of FactorEngine/LQTP YAML configs."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

FE = Path(__file__).resolve().parents[1]
if str(FE) not in sys.path:
    sys.path.insert(0, str(FE))

from factor_engine.api.dsl_parser import parse_expr
from factor_engine.api.source_ref import decode_source_ref
from factor_engine.ir.analyzer import Analyzer


def _factor_payload(payload: dict[str, Any]) -> tuple[str, str]:
    factor = payload.get("factor") if isinstance(payload.get("factor"), dict) else payload
    name = str(factor.get("name") or payload.get("factor_id") or "factor")
    expr = factor.get("expr") or factor.get("expression") or factor.get("formula")
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("factor expression missing")
    return name, expr


def audit_config(path: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {"config": str(path), "status": "failed"}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        name, expr = _factor_payload(payload)
        entry.update({"factor": name, "expression": expr})
        parsed = parse_expr(expr, surface="compat_research", dialect="lqtp", dialect_version="2026-07-19")
        analysis = Analyzer().lower(parsed)
        sources=[]; ordinary=[]
        for field in sorted(analysis.referenced_columns):
            spec=decode_source_ref(field)
            if spec is None:
                ordinary.append(field)
            else:
                sources.append(spec.to_payload())
        entry.update({"status":"parseable","ordinary_fields":ordinary,"source_refs":sources})
    except Exception as exc:
        entry["error_type"] = type(exc).__name__
        entry["error"] = str(exc)
    return entry


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("config_dir", type=Path)
    ap.add_argument("--output", type=Path, default=Path("lqtp_compatibility_audit.json"))
    args=ap.parse_args()
    files=sorted({*args.config_dir.rglob("*.yaml"), *args.config_dir.rglob("*.yml")})
    rows=[audit_config(path) for path in files]
    counts=Counter(row["status"] for row in rows)
    errors=Counter(row.get("error_type","") for row in rows if row["status"]=="failed")
    report={"schema_version":"factor_engine.lqtp_pack_audit.v1","dialect":"lqtp","dialect_version":"2026-07-19",
            "config_count":len(rows),"status_counts":dict(counts),"error_counts":dict(errors),"entries":rows}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("config_count","status_counts","error_counts")},ensure_ascii=False))
    return 1 if counts.get("failed",0) else 0

if __name__ == "__main__":
    raise SystemExit(main())
