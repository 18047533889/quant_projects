#!/usr/bin/env python3
"""用 factor_engine（read_auto + pandas）跑 GTJA-191 smoke。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lib.engine_config import build_fastest_engine, enable_fastest_read, fastest_data_source  # noqa: E402
from lib.paths import resolve_factor_engine_root  # noqa: E402


def _ensure_factor_engine() -> Path:
    fe_root = resolve_factor_engine_root()
    if fe_root is None:
        raise SystemExit(
            "factor_engine not found; set FACTOR_ENGINE_ROOT or place sibling ../factor_engine"
        )
    if str(fe_root) not in sys.path:
        sys.path.insert(0, str(fe_root))
    return fe_root


def _load_formula(name: str) -> str:
    catalog_path = PACKAGE_ROOT / "dsl" / "gtja191_dsl_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if name not in catalog:
        raise SystemExit(f"unknown factor: {name}")
    item = catalog[name]
    if not item["valid"]:
        raise SystemExit(f"{name} is invalid: {item['validation']}")
    return item["dsl_formula"]


def run_smoke(
    *,
    factor_name: str = "gtja191_alpha_001",
    start_date: str = "2016-01-04",
    end_date: str = "2016-01-10",
) -> dict:
    _ensure_factor_engine()

    from api.factor import Factor  # noqa: WPS433
    from api.dsl_parser import parse_expr  # noqa: WPS433
    from storage.factory import build_data_source  # noqa: WPS433

    formula = _load_formula(factor_name)
    expr = parse_expr(formula)
    factor = Factor(name=factor_name, expr=expr, freq="1d")
    source_cfg = fastest_data_source(start_date=start_date, end_date=end_date)
    source = build_data_source(source_cfg)
    enable_fastest_read(source)
    engine = build_fastest_engine(source)

    t0 = time.perf_counter()
    out = engine.run(factor)
    elapsed = time.perf_counter() - t0
    series = out["result"]
    backend_name = type(engine.backend).__name__
    return {
        "factor": factor_name,
        "formula": formula,
        "backend": backend_name,
        "elapsed_seconds": round(elapsed, 3),
        "rows": int(series.shape[0]),
        "non_nan_ratio": float(series.notna().mean()) if len(series) else 0.0,
        "lookback": int(out["analysis"].lookback),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run GTJA-191 factor smoke (read_auto + pandas backend)"
    )
    parser.add_argument("--factor", default="gtja191_alpha_001")
    parser.add_argument("--start-date", default="2016-01-04")
    parser.add_argument("--end-date", default="2016-01-10")
    args = parser.parse_args()

    try:
        result = run_smoke(
            factor_name=args.factor,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1

    print(json.dumps({"ok": True, **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
