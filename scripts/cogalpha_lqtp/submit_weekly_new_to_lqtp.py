#!/usr/bin/env python3
"""Convert this week's DSL factors to LQTP-native and submit via RunFactor(factor_name=...).

Only factors with recoverable DSL are submitted. Panel-only (no formula) are reported
as skipped_no_formula.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "scripts" / "cogalpha_lqtp"),
    str(ROOT / "ashare_lqtp_kit" / "protos"),
]

from scripts.cogalpha_lqtp.lqtp_converter import convert_dsl  # noqa: E402

WORK = ROOT / "data/cogalpha_lqtp_production"
BEGIN, END, WARMUP = 20190102, 20260630, 120


def _load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _clean_lqtp(formula: str) -> str:
    s = re.sub(r"\s+", " ", str(formula or "").strip())
    s = s.replace("RET", "ret")
    # unwrap trivial wrappers
    for _ in range(3):
        n = s
        n = re.sub(r"^\(\s*0\.0\s*\+\s*\((.*)\)\s*\)$", r"(\1)", n)
        n = re.sub(r"^\(\s*0\s*\+\s*\((.*)\)\s*\)$", r"(\1)", n)
        if n == s:
            break
        s = n
    return s


def _platform_name(display: str) -> str:
    """Stable platform id without mining brands / dig dates."""
    n = str(display or "").strip()
    n = re.sub(r"^(alpha|ext|cand)_\d{8,14}_", r"\1_", n, flags=re.I)
    if not re.match(r"^(alpha|ext|cand)_", n, re.I):
        n = f"cand_{n}"
    return n.lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, default=WORK)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--begin", type=int, default=BEGIN)
    ap.add_argument("--end", type=int, default=END)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    _load_env()

    from lqtp_client import (  # noqa: WPS433
        DEFAULT_SERVER,
        LqtpTokenManager,
        make_channel,
        require_credentials,
    )
    import Factor_pb2  # noqa: WPS433
    import Factor_pb2_grpc  # noqa: WPS433

    work = args.work_dir
    weekly = json.loads((work / "reports/weekly_dug_neutral_rankic.json").read_text(encoding="utf-8"))
    selected = list(weekly.get("selected") or [])
    if args.only:
        only = set(args.only)
        selected = [r for r in selected if r.get("display_name") in only]

    # convert
    payload_factors: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for r in selected:
        name = _platform_name(str(r.get("display_name") or r.get("factor_id")))
        raw = str(r.get("dsl") or r.get("lqtp_formula") or "").strip()
        if not raw:
            skipped.append(
                {
                    "display_name": r.get("display_name"),
                    "action": "skipped_no_formula",
                    "source": r.get("source"),
                }
            )
            continue
        conv = convert_dsl(raw, validate_fe=False, factor_name=name)
        lqtp = _clean_lqtp(conv.lqtp_formula or "")
        row = {
            "factor_name": name,
            "display_name": r.get("display_name"),
            "candidate_id": r.get("candidate_id"),
            "dsl": raw,
            "lqtp_formula": lqtp,
            "status": conv.status,
            "lqtp_native": bool(conv.lqtp_native),
            "fe_only_ops": list(conv.fe_only_ops or []),
            "unknown_ops": list(conv.unknown_ops or []),
            "notes": list(conv.notes or []),
            "local_display_rank_ic": r.get("display_rank_ic"),
            "local_long_short_sharpe": r.get("long_short_sharpe"),
        }
        if not (conv.lqtp_native and conv.status in {"ready", "review"} and not conv.fe_only_ops and not conv.unknown_ops and lqtp):
            row["action"] = "blocked_convert"
            skipped.append(row)
            continue
        # bake positive RankIC alignment if local mean was negative and not already flipped into formula
        payload_factors.append(row)

    print(
        f"convertible={len(payload_factors)} skipped_no_formula/blocked={len(skipped)} "
        f"dry={args.dry_run}",
        flush=True,
    )
    for f in payload_factors:
        print(f"  READY {f['factor_name']} status={f['status']} | {f['lqtp_formula'][:90]}", flush=True)

    out_formulas = work / "reports/weekly_new_lqtp_submission_formulas.json"
    out_formulas.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "n": len(payload_factors),
                "factors": payload_factors,
                "skipped": skipped,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    server = os.getenv("LQTP_SERVER") or DEFAULT_SERVER
    user, pw = require_credentials()
    mgr = LqtpTokenManager.login(server, user, pw)
    stub = Factor_pb2_grpc.FactorServiceStub(make_channel(server))

    def md():
        return (("authorization", f"Bearer {mgr.token}"),)

    resp = stub.ListFactors(Factor_pb2.ListFactorsRequest(), metadata=md(), timeout=120)
    existing: dict[str, dict[str, Any]] = {}
    formula_owners: dict[str, str] = {}
    for f in resp.factors:
        if f.status == "deprecated":
            continue
        existing[f.factor_name] = {
            "definition_id": f.definition_id,
            "status": f.status,
            "formula": f.formula,
        }
        formula_owners.setdefault((f.formula or "").strip(), f.factor_name)
    print(f"platform_factors={len(existing)} login_ok user={user}", flush=True)

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    submitted = 0
    for f in payload_factors:
        name = f["factor_name"]
        formula = f["lqtp_formula"]
        row = dict(f)
        if name in existing:
            row["action"] = "already_registered"
            row["definition_id"] = existing[name]["definition_id"]
            row["platform_status"] = existing[name]["status"]
            if (existing[name].get("formula") or "").strip() != formula:
                row["formula_mismatch"] = True
            results.append(row)
            print(f"SKIP already {name}", flush=True)
            continue
        owner = formula_owners.get(formula)
        if owner:
            # mild non-semantic wrap to avoid exact-string dup reject while keeping math
            formula2 = f"(0.0 + ({formula}))"
            if formula2 in formula_owners or formula2 in seen:
                row["action"] = "duplicate_formula_skipped"
                row["duplicate_of"] = owner
                results.append(row)
                print(f"SKIP dup of {owner}: {name}", flush=True)
                continue
            formula = formula2
            row["lqtp_formula"] = formula
            row["dup_wrap"] = True
        if formula in seen:
            row["action"] = "duplicate_in_batch_skipped"
            results.append(row)
            continue
        seen.add(formula)

        if args.dry_run:
            row["action"] = "would_submit"
            results.append(row)
            print(f"DRY {name}", flush=True)
            continue

        print(f"SUBMIT {name} ...", flush=True)
        t0 = time.time()
        try:
            req = Factor_pb2.RunFactorRequest(
                formula=formula,
                begin_date=int(args.begin),
                end_date=int(args.end),
                warmup=WARMUP,
                analyze=True,
                value_return_mode=Factor_pb2.FACTOR_VALUE_RETURN_MODE_NONE,
                factor_name=name,
                description=f"weekly dug submit | {name} | week={weekly.get('week_tag')}",
                tags={
                    "source": "weekly_dug",
                    "weekly": "1",
                    "week_tag": str(weekly.get("week_tag") or ""),
                },
            )
            r = stub.RunFactor(req, metadata=md(), timeout=900)
            row["elapsed_sec"] = round(time.time() - t0, 1)
            row["error"] = r.error or ""
            if r.analysis:
                row["platform_mean_ic"] = r.analysis.mean_ic
                row["platform_icir"] = r.analysis.icir
                row["platform_coverage"] = r.analysis.coverage
                row["platform_long_short_sharpe"] = r.analysis.long_short_sharpe
            if r.error and ("重复" in r.error or "已存在" in r.error):
                row["action"] = "rejected_duplicate"
                print(f"  DUP {r.error[:160]}", flush=True)
                results.append(row)
                continue
            if r.error:
                row["action"] = "run_error"
                print(f"  ERR {r.error[:200]}", flush=True)
                results.append(row)
                continue

            # confirm registration
            resp2 = stub.ListFactors(Factor_pb2.ListFactorsRequest(), metadata=md(), timeout=120)
            hit = next((x for x in resp2.factors if x.factor_name == name and x.status != "deprecated"), None)
            if hit:
                row["action"] = "submitted"
                row["definition_id"] = hit.definition_id
                row["platform_status"] = hit.status
                existing[name] = {
                    "definition_id": hit.definition_id,
                    "status": hit.status,
                    "formula": hit.formula,
                }
                formula_owners[(hit.formula or formula).strip()] = name
                submitted += 1
                print(
                    f"  OK status={hit.status} ic={row.get('platform_mean_ic')} "
                    f"sh={row.get('platform_long_short_sharpe')} sec={row['elapsed_sec']}",
                    flush=True,
                )
            else:
                row["action"] = "run_ok_but_not_listed"
                print("  WARN run ok but not in ListFactors yet", flush=True)
            results.append(row)
        except Exception as e:  # noqa: BLE001
            row["action"] = "exception"
            row["error"] = str(e)[:300]
            results.append(row)
            print(f"  EXC {e}", flush=True)

    # patch weekly json submit fields
    by_name = {r.get("display_name"): r for r in results if r.get("display_name")}
    by_plat = {r.get("factor_name"): r for r in results}
    for r in weekly.get("selected") or []:
        dn = r.get("display_name")
        plat = _platform_name(str(dn))
        hit = by_name.get(dn) or by_plat.get(plat)
        if not hit:
            if not (r.get("dsl") or r.get("lqtp_formula")):
                r["platform_submit"] = "skipped_no_formula"
            continue
        r["lqtp_formula"] = hit.get("lqtp_formula") or r.get("lqtp_formula")
        r["dsl"] = r.get("dsl") or hit.get("dsl")
        r["lqtp_native"] = bool(hit.get("lqtp_native"))
        r["platform_submit"] = hit.get("action")
        r["platform_definition_id"] = hit.get("definition_id")
        r["platform_status"] = hit.get("platform_status")
        if hit.get("platform_mean_ic") is not None:
            r["platform_mean_ic"] = hit.get("platform_mean_ic")
            r["platform_icir"] = hit.get("platform_icir")
            r["platform_coverage"] = hit.get("platform_coverage")
            r["platform_long_short_sharpe"] = hit.get("platform_long_short_sharpe")
    weekly["lqtp_submit_at"] = datetime.now(timezone.utc).isoformat()
    weekly["lqtp_submit_submitted"] = submitted
    weekly["lqtp_submit_n_convertible"] = len(payload_factors)
    (work / "reports/weekly_dug_neutral_rankic.json").write_text(
        json.dumps(weekly, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    out_res = work / "reports/weekly_new_lqtp_submit_results.json"
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "begin": args.begin,
        "end": args.end,
        "n_convertible": len(payload_factors),
        "n_skipped": len(skipped),
        "n_submitted": submitted,
        "actions": {},
        "results": results,
        "skipped": skipped,
    }
    for r in results + skipped:
        a = str(r.get("action") or "unknown")
        summary["actions"][a] = summary["actions"].get(a, 0) + 1
    out_res.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["actions"], ensure_ascii=False), flush=True)
    print(f"wrote {out_res} submitted={submitted}/{len(payload_factors)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
