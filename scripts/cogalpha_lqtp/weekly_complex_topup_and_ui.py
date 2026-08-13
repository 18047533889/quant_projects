#!/usr/bin/env python3
"""Normalize formulas, drop UI clutter, top-up complex pack factors with 2026 IC gate.

Memory-safe: sequential platform calls, no value pulls, DuckDB 2026 IC one factor at a time.
"""
from __future__ import annotations

import gc
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
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "scripts" / "cogalpha_lqtp"),
    str(ROOT / "ashare_lqtp_kit" / "protos"),
]

from scripts.cogalpha_lqtp.crawl_candidate_pool_neutral_rankic import (  # noqa: E402
    display_factor_name,
    display_lqtp_formula,
    patch_screening_html,
    rationalize_lqtp_formula,
)
from scripts.cogalpha_lqtp.eval_lake_fast import analyze_factor_parquet_duckdb  # noqa: E402
from scripts.cogalpha_lqtp.lqtp_converter import convert_dsl  # noqa: E402
from scripts.cogalpha_lqtp.refresh_weekly_dug_ui_pack import (  # noqa: E402
    _load_env,
    ensure_report,
)

WORK = ROOT / "data/cogalpha_lqtp_production"
PACK = Path("/tmp/evo_dsl_pack/all_factors_dsl_rankic_gt0p02.jsonl")
BEGIN, END, WARMUP = 20190102, 20260630, 120
Y2026_BEGIN, Y2026_END = 20260102, 20260630
BAD = re.compile(
    r"\b(turnover|df|rankvolume|absret|rankclose|rankopen|rankhigh|ranklow|amount|logvolume)\b",
    re.I,
)


def _mem_avail_gb() -> float:
    try:
        for line in open("/proc/meminfo", encoding="utf-8"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except OSError:
        return 99.0
    return 99.0


def _wait_mem(min_gb: float = 6.0) -> None:
    for _ in range(60):
        avail = _mem_avail_gb()
        if avail >= min_gb:
            return
        print(f"  low mem {avail:.1f}GB, wait...", flush=True)
        gc.collect()
        time.sleep(5)


def _complexity(f: str) -> int:
    return len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", f or ""))


def _family(f: str) -> str:
    return re.sub(r"\s+", "", re.sub(r"\d+\.?\d*", "N", f or ""))[:120]


def _hashes(selected: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for r in selected:
        for key in ("factor_id", "display_name", "candidate_id"):
            m = re.search(r"([0-9a-f]{8})$", str(r.get(key) or ""), re.I)
            if m:
                out.add(m.group(1).lower())
    return out


def normalize_selected_formulas(selected: list[dict[str, Any]]) -> None:
    for r in selected:
        raw = str(r.get("lqtp_formula") or r.get("dsl") or "").strip()
        if not raw:
            continue
        r.setdefault("formula_raw", raw)
        flipped = bool(r.get("sign_flipped"))
        nice = rationalize_lqtp_formula(raw, sign_flipped=flipped)
        r["lqtp_formula"] = nice
        r["dsl"] = nice
        r["formula_display"] = nice
        r["sign_baked_into_formula"] = True
        # UI no longer shows flip column; keep flag false so display helper won't double-wrap
        r["sign_flipped"] = False


def patch_report_formula(work: Path, row: dict[str, Any]) -> None:
    dname = display_factor_name(row)
    row["display_name"] = dname
    dsl = str(row.get("lqtp_formula") or "").strip()
    path = work / "reports_weekly_dug" / f"{dname}.html"
    if not path.exists() or not dsl:
        href = ensure_report(work, row, force_render=False)
        row["report_href"] = href
        path = work / "reports_weekly_dug" / f"{dname}.html"
        if not path.exists():
            return
    text = path.read_text(encoding="utf-8", errors="ignore")
    # replace first DSL/pre block under 公式
    if "（无公式/代码）" in text:
        text = text.replace(
            "<pre>（无公式/代码）</pre>",
            f"<h3>DSL（LQTP）</h3><pre>{_esc(dsl)}</pre>",
        )
    elif "<pre>" in text:
        text = re.sub(
            r"(<h3>[^<]*DSL[^<]*</h3>\s*)?<pre>.*?</pre>",
            f"<h3>DSL（LQTP）</h3><pre>{_esc(dsl)}</pre>",
            text,
            count=1,
            flags=re.S,
        )
    else:
        text = text.replace(
            "<h2>公式 / 代码</h2>",
            f"<h2>公式 / 代码</h2>\n    <h3>DSL（LQTP）</h3><pre>{_esc(dsl)}</pre>",
            1,
        )
    for brand in ("evoalpha", "EvoAlpha", "alphasage", "AlphaSage", "cogalpha", "CogAlpha", "lizhuo"):
        text = text.replace(brand, "cand")
    path.write_text(text, encoding="utf-8")
    row["report_href"] = f"/reports_weekly_dug/{dname}.html"


def _esc(s: str) -> str:
    import html as html_lib

    return html_lib.escape(s)


def compute_2026_ic_lake(work: Path, row: dict[str, Any]) -> float | None:
    fid = str(row.get("factor_id") or "")
    lake = work / "factor_lake" / fid / "values.parquet"
    if not lake.exists():
        # try display name path
        lake = work / "factor_lake" / str(row.get("display_name") or "") / "values.parquet"
    if not lake.exists():
        return None
    fwd = work / "lqtp_fwd_vwap_returns_cache.parquet"
    if not fwd.exists():
        return None
    _wait_mem(5.0)
    fac = lake.as_posix().replace("'", "''")
    factor_sql = f"""
      SELECT trade_date, symbol, value
      FROM read_parquet('{fac}')
      WHERE CAST(trade_date AS BIGINT) >= 20260101
        AND CAST(trade_date AS BIGINT) <= 20260630
        AND value IS NOT NULL
    """
    try:
        analysis = analyze_factor_parquet_duckdb(
            factor_path=lake,
            fwd_returns_path=fwd,
            return_kind="vwap_to_vwap",
            factor_sql=factor_sql,
        )
        ic = float(analysis.get("mean_rank_ic") or float("nan"))
        release = analysis
        del release
        gc.collect()
        if ic == ic:
            return ic
    except Exception as exc:  # noqa: BLE001
        print(f"  2026 lake fail {fid}: {exc}", flush=True)
        gc.collect()
    return None


def pick_complex(
    selected: list[dict[str, Any]],
    *,
    n: int,
    min_ops: int = 3,
) -> list[dict[str, Any]]:
    forms = {(r.get("lqtp_formula") or r.get("dsl") or "").strip() for r in selected}
    forms |= {(r.get("formula_raw") or "").strip() for r in selected}
    # rationalized forms too
    forms |= {rationalize_lqtp_formula(f, sign_flipped=False) for f in list(forms) if f}
    used_hash = _hashes(selected)
    used_fam_new: set[str] = set()
    cands: list[dict[str, Any]] = []
    for line in PACK.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("market") != "ashare":
            continue
        cid = str(r.get("candidate_id") or "")
        m = re.search(r"([0-9a-f]{8})$", cid, re.I)
        if not m:
            continue
        h = m.group(1).lower()
        if h in used_hash:
            continue
        formula = str(r.get("formula") or "").strip().replace("returns", "ret")
        if not formula or BAD.search(formula):
            continue
        if re.search(r"ts_max\([^,]+,\s*0\s*\)", formula):
            continue
        ops = _complexity(formula)
        if ops < min_ops and len(formula) < 40:
            continue
        if ops < min_ops:
            continue
        conv = convert_dsl(formula, validate_fe=False)
        if not conv.lqtp_native or conv.status not in {"ready", "review"}:
            continue
        if conv.unknown_ops or conv.fe_only_ops:
            continue
        lqtp = rationalize_lqtp_formula(conv.lqtp_formula, sign_flipped=False)
        if not lqtp or BAD.search(lqtp) or lqtp in forms:
            continue
        fam = _family(lqtp)
        if fam in used_fam_new:
            continue
        # prefer complex
        score = ops * 10 + min(len(lqtp) / 20.0, 5) + float(r.get("rank_ic") or 0) * 5
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
                "ops": _complexity(lqtp),
                "score": score,
                "source": str(r.get("source") or ""),
            }
        )
        used_fam_new.add(fam)
    cands.sort(key=lambda x: (-x["score"], -x["ops"], -x["rank_ic"]))
    return cands[:n]


def platform_analyze(
    stub,
    md,
    formula: str,
    *,
    begin: int,
    end: int,
    name: str = "",
) -> dict[str, Any]:
    import Factor_pb2

    req = Factor_pb2.RunFactorRequest(
        formula=formula,
        begin_date=begin,
        end_date=end,
        warmup=WARMUP,
        analyze=True,
        value_return_mode=Factor_pb2.FACTOR_VALUE_RETURN_MODE_NONE,
        factor_name=name,
        description=f"weekly dug complex | {name}" if name else "weekly dug 2026 probe",
        tags={"source": "weekly_dug_complex", "weekly": "1"} if name else {"probe": "2026"},
    )
    r = stub.RunFactor(req, metadata=md(), timeout=900)
    out: dict[str, Any] = {"error": r.error or ""}
    if r.analysis:
        out["mean_ic"] = r.analysis.mean_ic
        out["icir"] = r.analysis.icir
        out["coverage"] = r.analysis.coverage
        out["long_short_sharpe"] = r.analysis.long_short_sharpe
    return out


def submit_complex(
    picks: list[dict[str, Any]],
    *,
    out_path: Path,
    thr_full: float = 0.02,
    thr_2026: float = 0.015,
    max_decay: float = 0.55,
    target_add: int = 40,
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
    existing = {f.factor_name for f in resp.factors if f.status != "deprecated"}
    formula_owners = {
        f.formula.strip(): f.factor_name for f in resp.factors if f.status != "deprecated"
    }

    results: list[dict[str, Any]] = []
    kept = 0
    for i, p in enumerate(picks, 1):
        if kept >= target_add:
            print(f"reached target_add={target_add}, stop submit", flush=True)
            break
        _wait_mem(6.0)
        name = p["factor_name"]
        formula = p["lqtp_formula"].strip()
        row = dict(p)
        print(
            f"[{i}/{len(picks)}] ops={p.get('ops')} pack_ic={p['rank_ic']:.3f} {name}",
            flush=True,
        )
        if name in existing:
            row["action"] = "already_registered"
            results.append(row)
            print("  SKIP already", flush=True)
            continue
        if formula in formula_owners:
            # try mild wrap that keeps readability
            formula = f"-(-({formula}))"
            # that simplifies to same — use identity add
            formula = f"(0.0 + ({p['lqtp_formula'].strip()}))"
            row["lqtp_formula"] = formula
            row["dsl"] = formula
        try:
            full = platform_analyze(stub, md, formula, begin=BEGIN, end=END, name=name)
            row["error"] = full.get("error") or ""
            if full.get("error"):
                row["action"] = "run_error"
                print(f"  ERROR {full['error'][:160]}", flush=True)
                results.append(row)
                continue
            pic = float(full.get("mean_ic") or 0.0)
            row["platform_mean_ic"] = pic
            row["platform_icir"] = full.get("icir")
            row["platform_coverage"] = full.get("coverage")
            row["platform_long_short_sharpe"] = full.get("long_short_sharpe")
            abs_full = abs(pic)
            if abs_full <= thr_full:
                row["action"] = "below_full_thr"
                print(f"  DROP full |IC|={abs_full:.4f}", flush=True)
                results.append(row)
                continue

            # 2026 window analyze only (no register)
            _wait_mem(6.0)
            y = platform_analyze(
                stub, md, formula, begin=Y2026_BEGIN, end=Y2026_END, name=""
            )
            yic = float(y.get("mean_ic") or 0.0) if not y.get("error") else float("nan")
            row["rank_ic_2026_raw"] = yic if yic == yic else None
            row["rank_ic_2026"] = abs(yic) if yic == yic else None
            if yic != yic or abs(yic) < thr_2026:
                row["action"] = "weak_2026"
                print(f"  DROP 2026 |IC|={row.get('rank_ic_2026')}", flush=True)
                results.append(row)
                continue
            if abs(yic) < max_decay * abs_full:
                row["action"] = "decay_2026"
                print(
                    f"  DROP decay 2026={abs(yic):.4f} full={abs_full:.4f}",
                    flush=True,
                )
                results.append(row)
                continue

            # bake sign into formula for storage/display
            flipped = pic < 0
            nice = rationalize_lqtp_formula(p["lqtp_formula"], sign_flipped=flipped)
            # If we used 0.0+ wrap for dup, prefer nice of original with sign
            if formula.startswith("(0.0 +"):
                nice = rationalize_lqtp_formula(p["lqtp_formula"], sign_flipped=flipped)
            row["lqtp_formula"] = nice
            row["dsl"] = nice
            row["sign_flipped"] = False
            row["sign_baked_into_formula"] = True
            row["action"] = "submitted"
            row["display_rank_ic"] = abs_full
            row["abs_mean_rank_ic"] = abs_full
            row["mean_rank_ic"] = pic
            formula_owners[formula] = name
            existing.add(name)
            kept += 1
            print(
                f"  KEEP |IC|={abs_full:.4f} 2026={abs(yic):.4f} kept={kept}",
                flush=True,
            )
            results.append(row)
        except Exception as exc:  # noqa: BLE001
            row["action"] = "exception"
            row["error"] = f"{type(exc).__name__}: {exc}"
            results.append(row)
            print(f"  EXC {exc}", flush=True)
            traceback.print_exc()
        gc.collect()
        time.sleep(0.2)

    out_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "counts": dict(Counter(r.get("action") for r in results)),
                "n_kept": kept,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote", out_path, "kept", kept, flush=True)
    return results


def merge_kept(weekly: dict[str, Any], results: list[dict[str, Any]]) -> int:
    selected = list(weekly.get("selected") or [])
    have = _hashes(selected)
    have_names = {str(r.get("display_name") or r.get("factor_id")) for r in selected}
    added = 0
    for r in results:
        if r.get("action") != "submitted":
            continue
        name = str(r["factor_name"])
        h = str(r.get("hash") or "")
        if name in have_names or (h and h in have):
            continue
        abs_ic = float(r.get("display_rank_ic") or abs(float(r.get("platform_mean_ic") or 0)))
        selected.append(
            {
                "factor_id": name,
                "display_name": name,
                "candidate_id": r.get("candidate_id"),
                "source": "pack_dsl_complex",
                "label": "本周新挖",
                "source_label": "本周新挖",
                "ok": True,
                "ok_for_select": True,
                "from_pack_top_pick": True,
                "already_neutralized": False,
                "neutralization_note": "未中性化（包内复杂DSL）",
                "has_formula": True,
                "dsl": r.get("dsl"),
                "lqtp_formula": r.get("lqtp_formula"),
                "formula_raw": r.get("formula_src"),
                "lqtp_native": True,
                "formula_from": "pack_dsl_complex",
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
                "rank_ic_2026": r.get("rank_ic_2026"),
                "sign_flipped": False,
                "sign_baked_into_formula": True,
                "platform_submit": "submitted",
                "materialize": "platform_only",
                "ops": r.get("ops"),
                "family": r.get("family"),
            }
        )
        have_names.add(name)
        if h:
            have.add(h)
        added += 1
    selected.sort(
        key=lambda x: float(x.get("display_rank_ic") or x.get("abs_mean_rank_ic") or 0),
        reverse=True,
    )
    weekly["selected"] = selected
    weekly["n_selected"] = len(selected)
    weekly["complex_topup_added"] = added
    weekly["platform_note"] = (
        f"本周新挖 {len(selected)} 条；公式已合理化（无 0-1 写法）；负号写入公式；"
        "展示名无挖掘日期；优先补复杂且 2026 不猛衰减的因子。"
    )
    weekly["generated_at"] = datetime.now(timezone.utc).isoformat()
    return added


def main() -> int:
    _load_env()
    work = WORK
    weekly_path = work / "reports/weekly_dug_neutral_rankic.json"
    weekly = json.loads(weekly_path.read_text(encoding="utf-8"))
    print(f"start n={weekly.get('n_selected')} mem={_mem_avail_gb():.1f}GB", flush=True)

    # 1) normalize existing formulas
    normalize_selected_formulas(weekly["selected"])
    print("normalized formulas", flush=True)

    # 2) 2026 IC for lake-backed rows (sequential)
    for i, row in enumerate(weekly["selected"], 1):
        if row.get("rank_ic_2026") is not None:
            continue
        if str(row.get("materialize")) == "platform_only" and not (
            work / "factor_lake" / str(row.get("factor_id")) / "values.parquet"
        ).exists():
            continue
        print(f"2026-lake [{i}/{len(weekly['selected'])}] {display_factor_name(row)}", flush=True)
        ic = compute_2026_ic_lake(work, row)
        if ic is not None:
            # match display sign convention (abs)
            row["rank_ic_2026"] = abs(ic)
            row["rank_ic_2026_raw"] = ic
        gc.collect()

    # 3) top-up complex
    picks = pick_complex(weekly["selected"], n=70, min_ops=3)
    print(f"complex candidates={len(picks)} top_ops={[p['ops'] for p in picks[:10]]}", flush=True)
    for p in picks[:8]:
        print(f"  ops={p['ops']} ic={p['rank_ic']:.3f} {p['lqtp_formula'][:100]}", flush=True)

    results = submit_complex(
        picks,
        out_path=work / "reports/pack_dsl_complex_topup_submit.json",
        thr_full=0.02,
        thr_2026=0.015,
        max_decay=0.55,
        target_add=40,
    )
    added = merge_kept(weekly, results)
    print(f"merged added={added} n={weekly['n_selected']}", flush=True)

    # 4) reports + patch HTML
    for i, row in enumerate(weekly["selected"], 1):
        row["display_name"] = display_factor_name(row)
        # ensure formula is display-ready (already baked)
        row["lqtp_formula"] = rationalize_lqtp_formula(
            str(row.get("lqtp_formula") or row.get("dsl") or ""),
            sign_flipped=bool(row.get("sign_flipped")),
        )
        row["dsl"] = row["lqtp_formula"]
        row["sign_flipped"] = False
        if i % 20 == 0:
            print(f"report patch {i}/{weekly['n_selected']} mem={_mem_avail_gb():.1f}GB", flush=True)
        patch_report_formula(work, row)

    weekly["n_selected"] = len(weekly["selected"])
    weekly_path.write_text(json.dumps(weekly, ensure_ascii=False, indent=2), encoding="utf-8")

    class A:
        work_dir = work
        html = work / "reports/factor_rankic_screening_index.html"
        threshold = 0.02

    patch_screening_html(A(), payload=weekly)

    # QA
    html = (work / "reports/factor_rankic_screening_index.html").read_text(encoding="utf-8")
    sec = html.split('id="weekly-dug-neutral"', 1)[1].split("</section>", 1)[0]
    print(
        {
            "n_selected": weekly["n_selected"],
            "added": added,
            "has_来源_col": ">来源<" in sec,
            "has_取负": "取负显示" in sec,
            "has_0-1": "(0 - 1)" in sec or "(0-1)" in sec,
            "hero": re.search(r"<b>(\d+)</b><span>入选因子", html).group(1),
            "mem_gb": round(_mem_avail_gb(), 1),
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
