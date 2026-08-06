#!/usr/bin/env python3
"""Batch-submit LQTP-native factors from screening_reeval_catalog (or eligible JSON)."""
from __future__ import annotations

import argparse
import json
import os
import re
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
from lqtp_dsl_compat import is_lqtp_native_dsl, unknown_lqtp_calls
import Factor_pb2
import Factor_pb2_grpc

# ClickHouse rejects cap(..., 1e-12, 1e12) as Float64 vs UInt64.
_SCI_FLOAT = re.compile(r"(?<![A-Za-z0-9_])(\d+\.?\d*)[eE]([+-]?\d+)")


def sanitize_formula(formula: str) -> str:
    """Rewrite scientific literals to plain decimals for LQTP/CH."""

    def repl(m: re.Match[str]) -> str:
        try:
            v = float(m.group(0))
        except ValueError:
            return m.group(0)
        # Keep moderate magnitude; avoid huge ints that become UInt64.
        if abs(v) >= 1e9:
            return "1000000000.0" if v > 0 else "-1000000000.0"
        if 0 < abs(v) < 1e-6:
            return "0.000001" if v > 0 else "-0.000001"
        s = f"{v:.12f}".rstrip("0").rstrip(".")
        return s if "." in s else s + ".0"

    return _SCI_FLOAT.sub(repl, formula)


def load_eligible(path: Path, *, rebuild_from_catalog: Path | None) -> list[dict]:
    if rebuild_from_catalog is None:
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("eligible") or data.get("factors") or data)

    cat = json.loads(rebuild_from_catalog.read_text(encoding="utf-8"))
    out: list[dict] = []
    for f in cat:
        name = f.get("function_name") or f.get("factor_id") or f.get("factor_name")
        form = (f.get("lqtp_formula") or "").strip()
        if not name or not form:
            continue
        if not is_lqtp_native_dsl(form):
            continue
        out.append(
            {
                "name": name,
                "formula": form,
                "status": f.get("status"),
                "source": f.get("source"),
                "factor_key": f.get("factor_id") or f.get("factor_key"),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eligible",
        default="/home/shw/reports/lqtp_batch_eligible.json",
    )
    ap.add_argument(
        "--catalog",
        default="",
        help="If set, rebuild eligible from this screening catalog",
    )
    ap.add_argument("--out", default="/home/shw/reports/lqtp_batch_submit_results.json")
    ap.add_argument("--begin", type=int, default=20240102)
    ap.add_argument("--end", type=int, default=20240628)
    ap.add_argument("--warmup", type=int, default=80)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    catalog = Path(args.catalog) if args.catalog else None
    items = load_eligible(Path(args.eligible), rebuild_from_catalog=catalog)
    if args.limit > 0:
        items = items[: args.limit]

    server = os.getenv("LQTP_SERVER") or DEFAULT_SERVER
    user, pw = require_credentials()
    mgr = LqtpTokenManager.login(server, user, pw)
    stub = Factor_pb2_grpc.FactorServiceStub(make_channel(server))

    def md():
        return (("authorization", f"Bearer {mgr.token}"),)

    resp = stub.ListFactors(Factor_pb2.ListFactorsRequest(), metadata=md(), timeout=120)
    existing: dict[str, dict] = {}
    formula_owners: dict[str, str] = {}
    for f in resp.factors:
        if f.status == "deprecated":
            continue
        existing[f.factor_name] = {
            "definition_id": f.definition_id,
            "status": f.status,
            "formula": f.formula,
        }
        formula_owners.setdefault(f.formula.strip(), f.factor_name)

    results: list[dict] = []
    seen_formulas: set[str] = set()
    n = len(items)
    for i, item in enumerate(items, 1):
        name = item["name"]
        raw = item["formula"].strip()
        formula = sanitize_formula(raw)
        row: dict = {
            "factor_name": name,
            "factor_key": item.get("factor_key"),
            "source": item.get("source"),
            "formula": formula,
            "formula_sanitized": formula != raw,
        }
        print(f"[{i}/{n}] {name}", flush=True)

        if name in existing:
            row["action"] = "already_registered"
            row["definition_id"] = existing[name]["definition_id"]
            row["platform_status"] = existing[name]["status"]
            results.append(row)
            print("  SKIP already")
            continue

        owner = formula_owners.get(formula) or formula_owners.get(raw)
        if owner:
            row["action"] = "duplicate_formula_skipped"
            row["duplicate_of"] = owner
            results.append(row)
            print(f"  SKIP dup of {owner}")
            continue

        if formula in seen_formulas:
            row["action"] = "duplicate_in_batch_skipped"
            results.append(row)
            print("  SKIP batch-dup")
            continue
        seen_formulas.add(formula)

        if not is_lqtp_native_dsl(formula):
            row["action"] = "not_lqtp_native"
            row["unknown_ops"] = unknown_lqtp_calls(formula)
            results.append(row)
            print(f"  SKIP not native {row['unknown_ops']}")
            continue

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
                description=f"CogAlpha screening batch | {item.get('factor_key') or ''}",
                tags={
                    "source": str(item.get("source") or "screening_reeval"),
                    "batch": "cogalpha_full_pool",
                },
            )
            r = stub.RunFactor(req, metadata=md(), timeout=900)
            row["elapsed_sec"] = round(time.time() - t0, 1)
            row["error"] = r.error or ""
            row["total_value_rows"] = r.total_value_rows
            if r.analysis:
                row["mean_ic"] = r.analysis.mean_ic
                row["icir"] = r.analysis.icir
                row["coverage"] = r.analysis.coverage
            if r.error and ("重复" in r.error or "已存在" in r.error):
                row["action"] = "rejected_duplicate"
                results.append(row)
                print(f"  DUP {r.error[:80]}")
                continue
            if r.error:
                row["action"] = "run_error"
                results.append(row)
                print(f"  ERR {(r.error or '')[:160]}")
                continue

            # Empty error + named RunFactor => registered as validated.
            # Skip per-item ListFactors (expensive); refresh on checkpoints.
            row["action"] = "submitted"
            row["platform_status"] = "validated"
            formula_owners[formula] = name
            existing[name] = {
                "definition_id": "",
                "status": "validated",
                "formula": formula,
            }
            print(
                f"  OK mean_ic={row.get('mean_ic')} "
                f"({row['elapsed_sec']}s)"
            )
            results.append(row)
        except Exception as e:  # noqa: BLE001
            row["action"] = "exception"
            row["error"] = f"{type(e).__name__}: {e}"
            row["elapsed_sec"] = round(time.time() - t0, 1)
            results.append(row)
            print(f"  EXC {e}")
            traceback.print_exc()
            # refresh token on auth errors
            if "UNAUTHENTICATED" in str(e) or "token" in str(e).lower():
                mgr.maybe_refresh(force=True)

        if args.sleep > 0:
            time.sleep(args.sleep)

        # checkpoint every 10
        if i % 10 == 0 or i == n:
            # refresh definition_ids for submitted rows missing id
            try:
                resp2 = stub.ListFactors(
                    Factor_pb2.ListFactorsRequest(), metadata=md(), timeout=120
                )
                by_name = {
                    x.factor_name: x
                    for x in resp2.factors
                    if x.status != "deprecated"
                }
                for x in resp2.factors:
                    if x.status == "deprecated":
                        continue
                    existing[x.factor_name] = {
                        "definition_id": x.definition_id,
                        "status": x.status,
                        "formula": x.formula,
                    }
                    formula_owners.setdefault(x.formula.strip(), x.factor_name)
                for rr in results:
                    if rr.get("action") == "submitted" and not rr.get("definition_id"):
                        hit = by_name.get(rr["factor_name"])
                        if hit:
                            rr["definition_id"] = hit.definition_id
            except Exception as e:  # noqa: BLE001
                print(f"  list refresh warn: {e}")

            summary = {
                "begin": args.begin,
                "end": args.end,
                "warmup": args.warmup,
                "counts": dict(Counter(r["action"] for r in results)),
                "n_done": len(results),
                "n_total": n,
            }
            Path(args.out).write_text(
                json.dumps(
                    {"summary": summary, "results": results},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"  checkpoint {summary['counts']}", flush=True)

    summary = {
        "begin": args.begin,
        "end": args.end,
        "warmup": args.warmup,
        "counts": dict(Counter(r["action"] for r in results)),
        "n_done": len(results),
        "n_total": n,
    }
    Path(args.out).write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("=== DONE ===", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
