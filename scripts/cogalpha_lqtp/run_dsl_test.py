#!/usr/bin/env python3
"""Smoke-test DSL catalog: only materialize success counts as pass."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FE_ROOT = ROOT / "factor_engine"
SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FE_ROOT) not in sys.path:
    sys.path.insert(0, str(FE_ROOT))

from api.mining_integration import validate_factor_engine_dsl  # noqa: E402

from scripts.cogalpha_lqtp.materialize import _ashare_smoke_data_source, materialize_factor  # noqa: E402
from scripts.cogalpha_lqtp.run_light_test import _run_py  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke materialize test for DSL catalog")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/cogalpha_lqtp_batch")
    parser.add_argument("--start", default="2024-01-02")
    parser.add_argument("--end", default="2024-03-29")
    parser.add_argument("--n-symbols", type=int, default=300)
    parser.add_argument(
        "--regen-catalog",
        action="store_true",
        help="re-parse factors(1).md and rebuild dsl_catalog.json before testing",
    )
    args = parser.parse_args()

    work = args.work_dir
    parsed = work / "parsed_factors.json"
    catalog_path = work / "dsl_catalog.json"
    lake = work / "factor_lake"
    smoke_dir = work / "smoke_StockDailyBar"
    report_path = work / "dsl_smoke_test_report.json"

    if args.regen_catalog:
        _run_py(SCRIPTS / "parse_factors_md.py", str(ROOT / "factors(1).md"), "--out", str(parsed))
        _run_py(SCRIPTS / "python_to_dsl.py", str(parsed), "--out", str(catalog_path))

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog_stats = Counter(x.get("status", "") for x in catalog)

    if not smoke_dir.exists():
        _run_py(
            SCRIPTS / "smoke_data.py",
            "--out-dir",
            str(smoke_dir),
            "--start",
            args.start,
            "--end",
            args.end,
            "--n-symbols",
            str(args.n_symbols),
        )
    data_cfg = _ashare_smoke_data_source(
        parquet_root=smoke_dir,
        start_date=args.start,
        end_date=args.end,
    )

    results: list[dict[str, str]] = []
    passed: list[str] = []
    failed_ready: list[str] = []

    for item in catalog:
        name = item["function_name"]
        st = item.get("status", "")
        dsl = item.get("dsl", "")
        row: dict[str, str] = {
            "function_name": name,
            "catalog_status": st,
            "source": item.get("source", ""),
            "smoke_test": "skipped",
        }
        if st != "ready" or not dsl:
            row["reason"] = "no_valid_dsl" if st != "ready" else "empty_dsl"
            results.append(row)
            continue

        valid, msg = validate_factor_engine_dsl(dsl)
        if not valid:
            row["smoke_test"] = "fail"
            row["stage"] = "validate"
            row["error"] = msg
            failed_ready.append(name)
            results.append(row)
            continue

        try:
            out = materialize_factor(
                factor_id=name,
                dsl=dsl,
                data_source_cfg=data_cfg,
                lake_root=lake,
            )
            row["smoke_test"] = "pass"
            row["values_path"] = str(out)
            passed.append(name)
        except Exception as exc:  # noqa: BLE001
            row["smoke_test"] = "fail"
            row["stage"] = "materialize"
            row["error"] = str(exc)
            failed_ready.append(name)
        results.append(row)

    summary = {
        "data_root": str(smoke_dir),
        "date_range": [args.start, args.end],
        "catalog_stats": dict(catalog_stats),
        "ready_count": catalog_stats.get("ready", 0),
        "smoke_pass": len(passed),
        "smoke_fail": len(failed_ready),
        "not_tested": len(catalog) - catalog_stats.get("ready", 0),
        "passed_factors": passed,
        "failed_factors": failed_ready,
        "results": results,
    }
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "catalog": dict(catalog_stats),
                "ready": catalog_stats.get("ready", 0),
                "smoke_pass": len(passed),
                "smoke_fail": len(failed_ready),
                "hard_or_unconverted": catalog_stats.get("hard", 0)
                + catalog_stats.get("needs_review", 0)
                + catalog_stats.get("converted", 0),
            },
            ensure_ascii=False,
        )
    )
    print(f"report -> {report_path}")
    return 0 if not failed_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
