#!/usr/bin/env python3
"""Test + submit CogAlpha report LQTP-native factors via RunFactor(factor_name=...).

Platform registers a definition when RunFactor is called with a unique formula +
factor_name. Duplicate formulas are rejected by the server. PublishFactor needs
admin and is not used here; successful registration lands as status=validated.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path

from lqtp_client import (
    DEFAULT_SERVER,
    LqtpTokenManager,
    make_channel,
    require_credentials,
)
import Factor_pb2
import Factor_pb2_grpc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shw/reports/lqtp_submission_formulas.json",
    )
    parser.add_argument(
        "--out",
        default="/home/shw/reports/lqtp_submit_results.json",
    )
    parser.add_argument("--begin", type=int, default=20240102)
    parser.add_argument("--end", type=int, default=20241231)
    parser.add_argument("--warmup", type=int, default=120)
    parser.add_argument(
        "--include-review",
        action="store_true",
        help="Also try status=review factors (LQTP-native with approximations)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only classify already_registered / duplicate; do not RunFactor",
    )
    args = parser.parse_args()

    server = os.getenv("LQTP_SERVER") or DEFAULT_SERVER
    user, pw = require_credentials()
    mgr = LqtpTokenManager.login(server, user, pw)
    ch = make_channel(server)
    stub = Factor_pb2_grpc.FactorServiceStub(ch)

    def md():
        return (("authorization", f"Bearer {mgr.token}"),)

    wanted = json.loads(Path(args.input).read_text(encoding="utf-8"))["factors"]
    statuses = {"ready"}
    if args.include_review:
        statuses.add("review")
    eligible = [
        f
        for f in wanted
        if f.get("lqtp_native")
        and f.get("status") in statuses
        and not f.get("unknown_ops")
        and not f.get("fe_only_ops")
    ]

    resp = stub.ListFactors(Factor_pb2.ListFactorsRequest(), metadata=md(), timeout=60)
    existing: dict[str, dict] = {}
    formula_owners: dict[str, str] = {}
    for f in resp.factors:
        if f.status == "deprecated":
            continue
        existing[f.factor_name] = {
            "definition_id": f.definition_id,
            "status": f.status,
            "formula": f.formula,
            "version": f.version,
        }
        formula_owners.setdefault(f.formula.strip(), f.factor_name)

    results: list[dict] = []
    seen_formulas: set[str] = set()

    for f in eligible:
        name = f["factor_name"]
        formula = f["lqtp_formula"].strip()
        row: dict = {
            "factor_name": name,
            "factor_key": f.get("factor_key"),
            "converter_status": f.get("status"),
            "formula": formula,
        }
        if name in existing:
            row["action"] = "already_registered"
            row["definition_id"] = existing[name]["definition_id"]
            row["platform_status"] = existing[name]["status"]
            if existing[name]["formula"].strip() != formula:
                row["formula_mismatch"] = True
                row["platform_formula"] = existing[name]["formula"]
            results.append(row)
            print(f"SKIP already {name}")
            continue

        owner = formula_owners.get(formula)
        if owner:
            row["action"] = "duplicate_formula_skipped"
            row["duplicate_of"] = owner
            row["platform_error"] = f"公式完全重复，已存在因子 {owner}"
            results.append(row)
            print(f"SKIP dup of {owner}: {name}")
            continue

        if formula in seen_formulas:
            row["action"] = "duplicate_in_batch_skipped"
            results.append(row)
            print(f"SKIP batch-dup {name}")
            continue
        seen_formulas.add(formula)

        if args.dry_run:
            row["action"] = "would_submit"
            results.append(row)
            print(f"DRY would_submit {name}")
            continue

        print(f"SUBMIT {name} ...", flush=True)
        t0 = time.time()
        try:
            req = Factor_pb2.RunFactorRequest(
                formula=formula,
                begin_date=args.begin,
                end_date=args.end,
                warmup=args.warmup,
                analyze=True,
                value_return_mode=Factor_pb2.FACTOR_VALUE_RETURN_MODE_NONE,
                factor_name=name,
                description=(
                    f"CogAlpha report submit | {f.get('factor_key', '')} "
                    f"| status={f.get('status')}"
                ),
                tags={
                    "source": "cogalpha_report",
                    "factor_key": str(f.get("factor_key") or ""),
                    "converter_status": str(f.get("status") or ""),
                },
            )
            r = stub.RunFactor(req, metadata=md(), timeout=900)
            row["elapsed_sec"] = round(time.time() - t0, 1)
            row["error"] = r.error or ""
            row["total_value_rows"] = r.total_value_rows
            row["total_time_points"] = r.total_time_points
            if r.analysis:
                a = r.analysis
                row["mean_ic"] = a.mean_ic
                row["icir"] = a.icir
                row["coverage"] = a.coverage
                row["long_short_sharpe"] = a.long_short_sharpe
            if r.error and ("重复" in r.error or "已存在" in r.error):
                row["action"] = "rejected_duplicate"
                results.append(row)
                print(f"  DUP rejected: {r.error}")
                continue
            if r.error:
                row["action"] = "run_error"
                results.append(row)
                print(f"  ERROR: {r.error}")
                continue

            resp2 = stub.ListFactors(
                Factor_pb2.ListFactorsRequest(), metadata=md(), timeout=60
            )
            hit = next(
                (
                    x
                    for x in resp2.factors
                    if x.factor_name == name and x.status != "deprecated"
                ),
                None,
            )
            if hit:
                row["action"] = "submitted"
                row["definition_id"] = hit.definition_id
                row["platform_status"] = hit.status
                formula_owners[formula] = name
                existing[name] = {
                    "definition_id": hit.definition_id,
                    "status": hit.status,
                    "formula": hit.formula,
                    "version": hit.version,
                }
                print(
                    f"  OK {hit.status} {hit.definition_id} "
                    f"mean_ic={row.get('mean_ic')}"
                )
            else:
                row["action"] = "ran_but_not_registered"
                print(f"  WARN ran but not in ListFactors; error={r.error!r}")
            results.append(row)
        except Exception as e:  # noqa: BLE001
            row["action"] = "exception"
            row["error"] = f"{type(e).__name__}: {e}"
            row["elapsed_sec"] = round(time.time() - t0, 1)
            results.append(row)
            print(f"  EXC {e}")
            traceback.print_exc()

    summary = {
        "begin": args.begin,
        "end": args.end,
        "warmup": args.warmup,
        "counts": dict(Counter(r["action"] for r in results)),
        "n_results": len(results),
    }
    out = Path(args.out)
    out.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n=== SUMMARY ===")
    print(summary)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
