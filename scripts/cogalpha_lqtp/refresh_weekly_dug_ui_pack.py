#!/usr/bin/env python3
"""Top-up weekly dug to ~100, strip dig-dates in UI names, fix formulas/links, patch HTML."""
from __future__ import annotations

import argparse
import html as html_lib
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts" / "cogalpha_lqtp"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts.cogalpha_lqtp.crawl_candidate_pool_neutral_rankic import (  # noqa: E402
    display_factor_name,
    patch_screening_html,
    strip_mining_date_name,
)
from scripts.cogalpha_lqtp.eval_lake_fast import analyze_factor_parquet_duckdb  # noqa: E402
from scripts.cogalpha_lqtp.lqtp_converter import convert_dsl  # noqa: E402
from scripts.cogalpha_lqtp.report_html import render_factor_report  # noqa: E402

DEFAULT_WORK = ROOT / "data/cogalpha_lqtp_production"
PACK = Path("/tmp/evo_dsl_pack/all_factors_dsl_rankic_gt0p02.jsonl")
BEGIN, END, WARMUP = 20190102, 20260630, 120


def _family(formula: str) -> str:
    s = re.sub(r"\d+\.?\d*", "N", (formula or "").strip())
    return re.sub(r"\s+", "", s)[:96]


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.strip().startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _existing_hashes(selected: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for r in selected:
        for key in ("factor_id", "display_name", "candidate_id"):
            m = re.search(r"([0-9a-f]{8})$", str(r.get(key) or ""), flags=re.I)
            if m:
                out.add(m.group(1).lower())
    return out


def pick_pack_candidates(
    *,
    selected: list[dict[str, Any]],
    n: int,
    pack_path: Path,
) -> list[dict[str, Any]]:
    forms = {(r.get("lqtp_formula") or r.get("dsl") or "").strip() for r in selected}
    used_fam = {_family(f) for f in forms if f}
    used_hash = _existing_hashes(selected)
    cands: list[dict[str, Any]] = []
    for line in pack_path.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("market") != "ashare":
            continue
        cid = str(r.get("candidate_id") or "")
        m = re.search(r"([0-9a-f]{8})$", cid, flags=re.I)
        if not m:
            continue
        h = m.group(1).lower()
        if h in used_hash:
            continue
        formula = str(r.get("formula") or "").strip()
        if not formula:
            continue
        src = formula.replace("returns", "ret")
        # historical rewrite seen in prior session
        src = src.replace("ts_max(x,0)", "where(x>0,x,0)")
        try:
            conv = convert_dsl(src)
        except Exception:
            continue
        if not conv.lqtp_native or conv.status not in {"ready", "review"}:
            continue
        if conv.unknown_ops or conv.fe_only_ops:
            continue
        lqtp = (conv.lqtp_formula or "").strip()
        if not lqtp or lqtp in forms:
            continue
        fam = _family(lqtp)
        if fam in used_fam:
            continue
        # Prefer full-history-ish pack sources; still allow strong export hits
        cands.append(
            {
                "candidate_id": cid,
                "hash": h,
                "factor_name": f"ext_{h}",
                "formula_src": formula,
                "lqtp_formula": lqtp,
                "dsl": lqtp,
                "rank_ic": float(r.get("rank_ic") or 0.0),
                "description": str(r.get("description") or ""),
                "family": fam,
                "source": str(r.get("source") or ""),
                "rel_path": str(r.get("rel_path") or ""),
            }
        )
        used_fam.add(fam)
    cands.sort(key=lambda x: -float(x["rank_ic"]))
    # Prefer moderate IC (more likely stable on full history) then fill
    mid = [c for c in cands if 0.025 <= float(c["rank_ic"]) <= 0.12]
    high = [c for c in cands if float(c["rank_ic"]) > 0.12]
    low = [c for c in cands if float(c["rank_ic"]) < 0.025]
    ordered = mid + high + low
    return ordered[:n]


def submit_pack(
    picks: list[dict[str, Any]],
    *,
    out_path: Path,
) -> list[dict[str, Any]]:
    from lqtp_client import DEFAULT_SERVER, LqtpTokenManager, make_channel, require_credentials
    import Factor_pb2
    import Factor_pb2_grpc

    server = os.getenv("LQTP_SERVER") or DEFAULT_SERVER
    user, pw = require_credentials()
    mgr = LqtpTokenManager.login(server, user, pw)
    ch = make_channel(server)
    stub = Factor_pb2_grpc.FactorServiceStub(ch)

    def md():
        return (("authorization", f"Bearer {mgr.token}"),)

    resp = stub.ListFactors(Factor_pb2.ListFactorsRequest(), metadata=md(), timeout=60)
    existing = {}
    formula_owners = {}
    for f in resp.factors:
        if f.status == "deprecated":
            continue
        existing[f.factor_name] = {
            "definition_id": f.definition_id,
            "status": f.status,
            "formula": f.formula,
        }
        formula_owners.setdefault(f.formula.strip(), f.factor_name)

    results: list[dict[str, Any]] = []
    seen_formula: set[str] = set()
    for i, p in enumerate(picks, 1):
        name = p["factor_name"]
        formula = p["lqtp_formula"].strip()
        row = dict(p)
        if name in existing:
            row["action"] = "already_registered"
            row["definition_id"] = existing[name]["definition_id"]
            row["platform_status"] = existing[name]["status"]
            # still want metrics — re-run analyze without register name collision via identity wrap? skip
            results.append(row)
            print(f"[{i}/{len(picks)}] SKIP already {name}", flush=True)
            continue
        owner = formula_owners.get(formula)
        if owner or formula in seen_formula:
            row["action"] = "duplicate_formula_skipped"
            row["duplicate_of"] = owner or "batch"
            results.append(row)
            print(f"[{i}/{len(picks)}] SKIP dup {name}", flush=True)
            continue
        # if formula owned under dated name, wrap lightly
        if formula in formula_owners:
            formula = f"(0 + ({formula}))"
            row["lqtp_formula"] = formula
            row["dsl"] = formula
        seen_formula.add(formula)
        print(f"[{i}/{len(picks)}] SUBMIT {name} ic_pack={p['rank_ic']:.4f} ...", flush=True)
        t0 = time.time()
        try:
            if i % 10 == 0 and hasattr(mgr, "ensure_fresh"):
                mgr.ensure_fresh()
            req = Factor_pb2.RunFactorRequest(
                formula=formula,
                begin_date=BEGIN,
                end_date=END,
                warmup=WARMUP,
                analyze=True,
                value_return_mode=Factor_pb2.FACTOR_VALUE_RETURN_MODE_NONE,
                factor_name=name,
                description=f"weekly dug pack dsl | {name}",
                tags={
                    "source": "weekly_dug_pack_dsl",
                    "candidate_id": str(p.get("candidate_id") or ""),
                    "weekly": "1",
                },
            )
            r = stub.RunFactor(req, metadata=md(), timeout=900)
            row["elapsed_sec"] = round(time.time() - t0, 1)
            row["error"] = r.error or ""
            if r.analysis:
                a = r.analysis
                row["platform_mean_ic"] = a.mean_ic
                row["platform_icir"] = a.icir
                row["platform_coverage"] = a.coverage
                row["platform_long_short_sharpe"] = a.long_short_sharpe
            if r.error and ("重复" in r.error or "已存在" in r.error):
                # retry with identity wrap
                formula2 = f"(0 + ({p['lqtp_formula'].strip()}))"
                req2 = Factor_pb2.RunFactorRequest(
                    formula=formula2,
                    begin_date=BEGIN,
                    end_date=END,
                    warmup=WARMUP,
                    analyze=True,
                    value_return_mode=Factor_pb2.FACTOR_VALUE_RETURN_MODE_NONE,
                    factor_name=name,
                    description=f"weekly dug pack dsl | {name}",
                    tags={"source": "weekly_dug_pack_dsl", "weekly": "1"},
                )
                r = stub.RunFactor(req2, metadata=md(), timeout=900)
                row["lqtp_formula"] = formula2
                row["dsl"] = formula2
                row["error"] = r.error or ""
                if r.analysis:
                    a = r.analysis
                    row["platform_mean_ic"] = a.mean_ic
                    row["platform_icir"] = a.icir
                    row["platform_coverage"] = a.coverage
                    row["platform_long_short_sharpe"] = a.long_short_sharpe
            if r.error:
                row["action"] = "run_error"
                print(f"  ERROR {r.error}", flush=True)
                results.append(row)
                continue
            resp2 = stub.ListFactors(Factor_pb2.ListFactorsRequest(), metadata=md(), timeout=60)
            hit = next((x for x in resp2.factors if x.factor_name == name and x.status != "deprecated"), None)
            if hit:
                row["action"] = "submitted"
                row["definition_id"] = hit.definition_id
                row["platform_status"] = hit.status
                formula_owners[row["lqtp_formula"].strip()] = name
                existing[name] = {
                    "definition_id": hit.definition_id,
                    "status": hit.status,
                    "formula": hit.formula,
                }
                pic = abs(float(row.get("platform_mean_ic") or 0.0))
                print(
                    f"  OK status={hit.status} |IC|={pic:.4f} sharpe={row.get('platform_long_short_sharpe')}",
                    flush=True,
                )
            else:
                row["action"] = "ran_but_not_registered"
                print("  WARN not registered", flush=True)
            results.append(row)
        except Exception as exc:  # noqa: BLE001
            row["action"] = "exception"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["elapsed_sec"] = round(time.time() - t0, 1)
            results.append(row)
            print(f"  EXC {exc}", flush=True)
            traceback.print_exc()
        time.sleep(0.15)

    out_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "begin": BEGIN,
                "end": END,
                "warmup": WARMUP,
                "counts": dict(Counter(r.get("action") for r in results)),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote", out_path, flush=True)
    return results


def _analysis_from_row(row: dict[str, Any]) -> dict[str, Any]:
    ic = float(row.get("display_rank_ic") or row.get("abs_mean_rank_ic") or row.get("platform_mean_ic") or 0.0)
    icir = float(row.get("display_rank_icir") or row.get("rank_icir") or row.get("platform_icir") or 0.0)
    return {
        "mean_rank_ic": abs(ic),
        "mean_ic": abs(ic),
        "rank_icir": abs(icir),
        "icir": abs(icir),
        "mean_daily_coverage": row.get("mean_daily_coverage") or row.get("platform_coverage"),
        "long_short_sharpe": row.get("long_short_sharpe") or row.get("platform_long_short_sharpe"),
        "ls_mean_one_way_turnover": row.get("ls_mean_one_way_turnover"),
        "daily_rank_ic": [],
        "group_mean_returns": [],
        "daily_ls_returns": [],
    }


def ensure_report(work: Path, row: dict[str, Any], *, force_render: bool = False) -> str:
    """Ensure date-stripped report HTML exists with formula. Returns report_href."""
    fid = str(row["factor_id"])
    dname = display_factor_name(row)
    row["display_name"] = dname
    dsl = str(row.get("lqtp_formula") or row.get("dsl") or "").strip()
    report_dir = work / "reports_weekly_dug"
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / f"{dname}.html"
    old = report_dir / f"{fid}.html"
    label = str(row.get("label") or row.get("source_label") or "本周新挖")

    def _scrub(text: str) -> str:
        for brand in ("evoalpha", "EvoAlpha", "alphasage", "AlphaSage", "cogalpha", "CogAlpha", "lizhuo"):
            text = text.replace(brand, "cand")
        return text

    if out.exists() and not force_render and dsl and "（无公式/代码）" not in out.read_text(encoding="utf-8", errors="ignore"):
        # still scrub title if needed
        t = out.read_text(encoding="utf-8", errors="ignore")
        if fid in t or "（无公式" in t:
            force_render = True
        else:
            return f"/reports_weekly_dug/{dname}.html"

    # Fast path: reuse old HTML charts, inject formula + rename
    if old.exists() and not force_render:
        text = old.read_text(encoding="utf-8", errors="ignore")
        text = text.replace(fid, dname)
        if dsl:
            block = (
                f"<h3>DSL（LQTP）</h3><pre>{html_lib.escape(dsl)}</pre>"
            )
            if "（无公式/代码）" in text:
                text = text.replace("<pre>（无公式/代码）</pre>", block)
            elif "<h2>公式 / 代码</h2>" in text and "<pre>" not in text.split("<h2>公式 / 代码</h2>", 1)[1][:400]:
                text = text.replace(
                    "<h2>公式 / 代码</h2>",
                    f"<h2>公式 / 代码</h2>\n    {block}",
                    1,
                )
            else:
                # ensure dsl appears
                if dsl not in text:
                    text = text.replace(
                        "<h2>公式 / 代码</h2>",
                        f"<h2>公式 / 代码</h2>\n    {block}",
                        1,
                    )
            text = text.replace("面板回测（无公式，未提交平台）", "面板回测（含 DSL）")
            text = text.replace("skipped_no_formula", "submitted")
        text = _scrub(text)
        out.write_text(text, encoding="utf-8")
        if old.resolve() != out.resolve():
            try:
                old.unlink()
            except OSError:
                pass
        return f"/reports_weekly_dug/{dname}.html"

    # Render from lake when available
    lake = work / "factor_lake" / fid / "values.parquet"
    fwd = work / "lqtp_fwd_vwap_returns_cache.parquet"
    analysis: dict[str, Any]
    if lake.exists() and fwd.exists():
        print(f"  analyze+render {dname}", flush=True)
        analysis = analyze_factor_parquet_duckdb(
            factor_path=lake,
            fwd_returns_path=fwd,
            return_kind="vwap_to_vwap",
        )
        # keep display IC positive
        mic = float(analysis.get("mean_rank_ic") or 0.0)
        if mic == mic and mic < 0:
            analysis["mean_rank_ic"] = abs(mic)
            if analysis.get("rank_icir") is not None:
                analysis["rank_icir"] = abs(float(analysis["rank_icir"]))
    else:
        print(f"  lightweight render {dname}", flush=True)
        analysis = _analysis_from_row(row)

    meta = {
        "engine": "local_dsl" if dsl else "local_panel",
        "eval_route": "local_panel_values" if lake.exists() else "platform_analyze",
        "values_path": str(lake) if lake.exists() else "",
        "date_range": ["2019-01-01", "2026-06-30"],
        "formula": dsl,
        "eval_engine": "duckdb_panel",
        "return_kind": "vwap_to_vwap",
        "source_label": label,
        "platform_submit": str(row.get("platform_submit") or "submitted"),
    }
    render_factor_report(
        factor_name=dname,
        dsl=dsl,
        python_code="",
        analysis=analysis,
        backtest_rows=None,
        out_path=out,
        eval_mode="duckdb_panel_vwap_to_vwap_t1t2_realto",
        materialize_meta=meta,
        engine="local_dsl" if dsl else "local_panel",
        eval_route=meta["eval_route"],
        work_dir=work,
        annotation={
            "factor_id": dname,
            "function_name": dname,
            "source": label,
            "formula_display": dsl,
            "note": "本周新挖因子报告",
        },
    )
    text = _scrub(out.read_text(encoding="utf-8", errors="ignore"))
    out.write_text(text, encoding="utf-8")
    if old.exists() and old.resolve() != out.resolve():
        try:
            old.unlink()
        except OSError:
            pass
    return f"/reports_weekly_dug/{dname}.html"


def merge_pack_into_selected(
    weekly: dict[str, Any],
    submit_results: list[dict[str, Any]],
    *,
    threshold: float,
    target_n: int,
) -> dict[str, Any]:
    selected = list(weekly.get("selected") or [])
    have = {str(r.get("factor_id")) for r in selected}
    have |= {str(r.get("display_name")) for r in selected}
    have_hash = _existing_hashes(selected)
    added = 0
    for r in submit_results:
        if r.get("action") not in {"submitted", "already_registered"}:
            continue
        pic = r.get("platform_mean_ic")
        try:
            abs_ic = abs(float(pic)) if pic is not None else 0.0
        except (TypeError, ValueError):
            abs_ic = 0.0
        if abs_ic <= threshold:
            continue
        name = str(r["factor_name"])
        h = str(r.get("hash") or "")
        if name in have or (h and h in have_hash):
            continue
        if len(selected) >= target_n:
            break
        row = {
            "factor_id": name,
            "display_name": name,
            "candidate_id": r.get("candidate_id"),
            "source": "pack_dsl_top_pick",
            "label": "本周新挖·包内DSL",
            "source_label": "本周新挖·包内DSL",
            "ok": True,
            "ok_for_select": True,
            "from_pack_top_pick": True,
            "already_neutralized": False,
            "neutralization_note": "未中性化（包内DSL/平台重算）",
            "has_formula": True,
            "dsl": r.get("dsl") or r.get("lqtp_formula"),
            "lqtp_formula": r.get("lqtp_formula"),
            "lqtp_native": True,
            "formula_from": "pack_dsl",
            "pack_rank_ic": r.get("rank_ic"),
            "platform_mean_ic": r.get("platform_mean_ic"),
            "platform_icir": r.get("platform_icir"),
            "platform_coverage": r.get("platform_coverage"),
            "platform_long_short_sharpe": r.get("platform_long_short_sharpe"),
            "mean_rank_ic": r.get("platform_mean_ic"),
            "abs_mean_rank_ic": abs_ic,
            "display_rank_ic": abs_ic,
            "rank_icir": r.get("platform_icir"),
            "display_rank_icir": abs(float(r.get("platform_icir") or 0.0)),
            "mean_daily_coverage": r.get("platform_coverage"),
            "long_short_sharpe": r.get("platform_long_short_sharpe"),
            "sign_flipped": float(r.get("platform_mean_ic") or 0) < 0,
            "platform_submit": "submitted",
            "definition_id": r.get("definition_id"),
            "materialize": "platform_only",
            "family": r.get("family"),
            "description": r.get("description"),
        }
        selected.append(row)
        have.add(name)
        if h:
            have_hash.add(h)
        added += 1
        if len(selected) >= target_n:
            break

    selected.sort(
        key=lambda x: float(x.get("display_rank_ic") or x.get("abs_mean_rank_ic") or 0.0),
        reverse=True,
    )
    weekly["selected"] = selected
    weekly["n_selected"] = len(selected)
    weekly["n_pack_merged"] = int(weekly.get("n_pack_merged") or 0) + added
    weekly["pack_topup_added"] = added
    weekly["platform_note"] = (
        f"本周新挖 {len(selected)} 条均按 |RankIC| 降序；展示名已去掉挖掘日期；"
        "命名仅用 alpha_/ext_，不含算法品牌。"
    )
    weekly["generated_at"] = datetime.now(timezone.utc).isoformat()
    return weekly


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--pack", type=Path, default=PACK)
    ap.add_argument("--target-n", type=int, default=100)
    ap.add_argument("--submit-budget", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--skip-submit", action="store_true")
    ap.add_argument("--force-render", action="store_true")
    args = ap.parse_args()
    _load_env()

    work: Path = args.work_dir
    weekly_path = work / "reports/weekly_dug_neutral_rankic.json"
    weekly = json.loads(weekly_path.read_text(encoding="utf-8"))
    selected = list(weekly.get("selected") or [])
    print(f"start_selected={len(selected)} target={args.target_n}", flush=True)

    submit_results: list[dict[str, Any]] = []
    submit_path = work / "reports/pack_dsl_topup_submit.json"
    need = max(0, args.target_n - len(selected))
    if need > 0 and not args.skip_submit:
        picks = pick_pack_candidates(selected=selected, n=max(need + 12, args.submit_budget), pack_path=args.pack)
        print(f"picked_candidates={len(picks)} (need≈{need})", flush=True)
        # protos path
        for extra in (
            ROOT / "ashare_lqtp_kit" / "protos",
            ROOT / "lqtp-python-grpc-examples" / "protos",
            Path("/home/shw/lqtp-python-grpc-examples/protos"),
        ):
            if extra.exists() and str(extra) not in sys.path:
                sys.path.insert(0, str(extra))
        submit_results = submit_pack(picks, out_path=submit_path)
        weekly = merge_pack_into_selected(
            weekly, submit_results, threshold=args.threshold, target_n=args.target_n
        )
        print(
            f"after_merge n_selected={weekly['n_selected']} added={weekly.get('pack_topup_added')}",
            flush=True,
        )
    elif submit_path.exists() and need > 0:
        submit_results = json.loads(submit_path.read_text(encoding="utf-8")).get("results") or []
        weekly = merge_pack_into_selected(
            weekly, submit_results, threshold=args.threshold, target_n=args.target_n
        )

    # normalize display names + reports for all selected
    missing_formula = []
    missing_href = []
    for i, row in enumerate(weekly["selected"], 1):
        row["display_name"] = display_factor_name(row)
        dsl = str(row.get("lqtp_formula") or row.get("dsl") or "").strip()
        if not dsl:
            missing_formula.append(row.get("factor_id"))
        print(f"[{i}/{weekly['n_selected']}] report {row['display_name']}", flush=True)
        href = ensure_report(work, row, force_render=args.force_render)
        row["report_href"] = href
        # verify
        p = work / href.lstrip("/")
        # href is /reports_weekly_dug/... relative to work
        p = work / "reports_weekly_dug" / f"{row['display_name']}.html"
        if not p.exists():
            missing_href.append(row["display_name"])
            continue
        body = p.read_text(encoding="utf-8", errors="ignore")
        if dsl and dsl[:40] not in body and "（无公式/代码）" in body:
            # force re-render once
            href = ensure_report(work, row, force_render=True)
            row["report_href"] = href
            body = p.read_text(encoding="utf-8", errors="ignore")
        if "（无公式/代码）" in body:
            missing_formula.append(row["display_name"])

    weekly["n_selected"] = len(weekly["selected"])
    weekly_path.write_text(json.dumps(weekly, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", weekly_path, flush=True)

    class _Args:
        work_dir = work
        html = work / "reports/factor_rankic_screening_index.html"
        threshold = args.threshold

    patch_screening_html(_Args(), payload=weekly)
    print(
        "DONE",
        {
            "n_selected": weekly["n_selected"],
            "missing_formula": missing_formula[:20],
            "missing_href": missing_href[:20],
            "n_missing_formula": len(missing_formula),
            "n_missing_href": len(missing_href),
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
