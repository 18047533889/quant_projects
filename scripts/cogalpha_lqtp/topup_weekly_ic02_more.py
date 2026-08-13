#!/usr/bin/env python3
"""Top-up more weekly-dug factors with hard |platform RankIC| > 0.02.

Platform ListFactors/RunFactor analyze only — no FactorEngine materialize
(so it does not compete with the slow no-cs-rank chart-fill queue).

Among passers: prefer high |LS Sharpe|, then prefer formulas WITHOUT
cross-sectional rank( (easier later chart fill).
"""
from __future__ import annotations

import csv
import gc
import json
import os
import re
import sys
import time
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
    patch_screening_html,
    rationalize_lqtp_formula,
)
from scripts.cogalpha_lqtp.lqtp_converter import convert_dsl  # noqa: E402
from scripts.cogalpha_lqtp.weekly_complex_topup_and_ui import (  # noqa: E402
    BAD,
    BEGIN,
    END,
    PACK,
    WORK,
    _complexity,
    _hashes,
    _load_env,
    _mem_avail_gb,
    _wait_mem,
    patch_report_formula,
    platform_analyze,
)
from lqtp_client import (  # noqa: E402
    DEFAULT_SERVER,
    LqtpTokenManager,
    make_channel,
    require_credentials,
)
import Factor_pb2  # noqa: E402
import Factor_pb2_grpc  # noqa: E402

TARGET = 15
PROBE_CAP = 220
HARD_IC = 0.02
CSV_PATH = ROOT / "formulas.csv"
OUT_SUBMIT = WORK / "reports/pack_dsl_topup_ic02_more_submit.json"


def _norm(f: str) -> str:
    return re.sub(r"\s+", "", str(f or ""))


def _has_cs_rank(dsl: str) -> bool:
    d = re.sub(r"ts_rank\s*\(", "", str(dsl).lower())
    return bool(re.search(r"(?<![a-z_])rank\s*\(", d))


def _clean(f: str, pic: float) -> str:
    nice = rationalize_lqtp_formula(str(f or ""), sign_flipped=(float(pic) < 0)).replace("RET", "ret")
    nice = re.sub(r"\(\s*0\.0\s*\+\s*\((.*)\)\s*\)", r"(\1)", nice)
    nice = re.sub(r"-\(\s*0\.0\s*\+\s*\((.*)\)\s*\)", r"-(\1)", nice)
    return nice


def _blocked(forms: set[str], f: str) -> bool:
    key = _norm(f)
    if not key:
        return True
    if key in forms:
        return True
    flip = "-" + key if not key.startswith("-") else key[1:]
    return flip in forms


def _name(h: str, raw: str = "") -> str:
    n = str(raw or "")
    m = re.match(r"^(alpha|ext|cand)_(?:\d{8,14}_)?([0-9a-f]{8})$", n, re.I)
    if m:
        return f"{m.group(1).lower()}_{m.group(2).lower()}"
    if n.startswith("alpha_"):
        return f"alpha_{h}"
    return f"ext_{h}"


def _try_convert(formula: str) -> str | None:
    formula = str(formula or "").strip().replace("returns", "ret")
    if not formula or BAD.search(formula):
        return None
    if re.search(r"ts_max\([^,]+,\s*0\s*\)", formula):
        return None
    try:
        conv = convert_dsl(formula, validate_fe=False)
    except Exception:
        return None
    if not conv.lqtp_native or conv.status not in {"ready", "review"}:
        return None
    if conv.unknown_ops or conv.fe_only_ops:
        return None
    lqtp = rationalize_lqtp_formula(conv.lqtp_formula, sign_flipped=False).replace("RET", "ret")
    if BAD.search(lqtp) or _complexity(lqtp) < 1:
        return None
    return lqtp


def main() -> None:
    _load_env()
    work = WORK
    wp = work / "reports/weekly_dug_neutral_rankic.json"
    weekly = json.loads(wp.read_text())
    selected = list(weekly["selected"])
    baseline = len(selected)
    have = _hashes(selected)
    forms = {_norm(r.get("lqtp_formula") or r.get("dsl")) for r in selected}
    forms |= {("-" + f if not f.startswith("-") else f[1:]) for f in list(forms) if f}
    print("start", baseline, "mem", round(_mem_avail_gb(), 1), "pack", PACK.exists(), flush=True)

    server = os.getenv("LQTP_SERVER") or DEFAULT_SERVER
    user, pw = require_credentials()
    mgr = LqtpTokenManager.login(server, user, pw)
    stub = Factor_pb2_grpc.FactorServiceStub(make_channel(server))

    def md():
        return (("authorization", f"Bearer {mgr.token}"),)

    resp = stub.ListFactors(Factor_pb2.ListFactorsRequest(), metadata=md(), timeout=120)
    existing = {f.factor_name for f in resp.factors if f.status != "deprecated"}
    owners = {
        f.formula.strip(): f.factor_name
        for f in resp.factors
        if f.status != "deprecated" and f.formula
    }

    probes: list[dict[str, Any]] = []
    seen_h: set[str] = set()
    seen_form: set[str] = set(forms)

    def push(row: dict[str, Any]) -> None:
        h = row["hash"]
        if h in have or h in seen_h:
            return
        key = _norm(row["lqtp_formula"])
        if _blocked(seen_form, key):
            return
        seen_h.add(h)
        seen_form.add(key)
        seen_form.add("-" + key if not key.startswith("-") else key[1:])
        probes.append(row)

    # Prefer pack first (pack already filtered |IC|>0.02 short window), then platform, then csv
    pack_n = 0
    if PACK.exists():
        for line in PACK.open():
            r = json.loads(line)
            if r.get("market") != "ashare":
                continue
            cid = str(r.get("candidate_id") or "")
            m = re.search(r"([0-9a-f]{8})$", cid, re.I)
            if not m:
                continue
            h = m.group(1).lower()
            lqtp = _try_convert(r.get("formula") or "")
            if not lqtp:
                continue
            push(
                {
                    "factor_name": f"ext_{h}",
                    "hash": h,
                    "candidate_id": cid,
                    "lqtp_formula": lqtp,
                    "formula_src": r.get("formula"),
                    "ops": _complexity(lqtp),
                    "rank_ic": float(r.get("rank_ic") or 0),
                    "from": "pack",
                    "need_reg": f"ext_{h}" not in existing,
                    "already_neutralized": False,
                    "neutralization_note": "未中性化（包内DSL）",
                    "cs_rank": _has_cs_rank(lqtp),
                }
            )
            pack_n += 1
    print("pack push", pack_n, "probes", len(probes), flush=True)

    plat_n = 0
    for f in resp.factors:
        if f.status == "deprecated":
            continue
        name = f.factor_name or ""
        if not (name.startswith("ext_") or name.startswith("alpha_")):
            continue
        m = re.search(r"([0-9a-f]{8})$", name, re.I)
        if not m:
            continue
        h = m.group(1).lower()
        formula = (f.formula or "").strip().replace("RET", "ret")
        if not formula or BAD.search(formula):
            continue
        push(
            {
                "factor_name": _name(h, name),
                "hash": h,
                "candidate_id": name,
                "lqtp_formula": formula,
                "formula_src": formula,
                "ops": _complexity(formula),
                "rank_ic": 0.0,
                "from": "platform",
                "need_reg": False,
                "already_neutralized": False,
                "neutralization_note": "未中性化（平台已有）",
                "cs_rank": _has_cs_rank(formula),
            }
        )
        plat_n += 1
    print("platform push", plat_n, "probes", len(probes), flush=True)

    csv_n = 0
    if CSV_PATH.exists():
        for row in csv.DictReader(CSV_PATH.open()):
            fid = str(row.get("factor_id") or "")
            m = re.search(r"([0-9a-f]{8})$", fid, re.I)
            if not m:
                continue
            h = m.group(1).lower()
            lqtp = _try_convert(row.get("formula") or "")
            if not lqtp:
                continue
            try:
                csv_ic = abs(float(row.get("abs_rank_ic_vwap_2024_2026") or 0))
            except Exception:
                csv_ic = 0.0
            push(
                {
                    "factor_name": _name(h, fid),
                    "hash": h,
                    "candidate_id": fid,
                    "lqtp_formula": lqtp,
                    "formula_src": row.get("formula"),
                    "ops": _complexity(lqtp),
                    "rank_ic": csv_ic,
                    "from": "formulas_csv",
                    "need_reg": _name(h, fid) not in existing,
                    "already_neutralized": True,
                    "neutralization_note": "已中性化（COS池）",
                    "cs_rank": _has_cs_rank(lqtp),
                }
            )
            csv_n += 1
    print("csv push", csv_n, "probes", len(probes), flush=True)

    # Prefer: higher pack/short IC, no cs-rank, moderate complexity
    probes.sort(
        key=lambda x: (
            0 if not x.get("cs_rank") else 1,
            -float(x.get("rank_ic") or 0),
            -min(int(x["ops"]), 6),
        )
    )
    skel: Counter[str] = Counter()
    picked: list[dict[str, Any]] = []
    for c in probes:
        sk = re.sub(r"\d+\.?\d*", "N", _norm(c["lqtp_formula"]))[:120]
        if skel[sk] >= 4:
            continue
        skel[sk] += 1
        picked.append(c)
        if len(picked) >= PROBE_CAP:
            break
    probes = picked
    print(f"probe list {len(probes)}", flush=True)

    passers: list[dict[str, Any]] = []
    used_forms = set(forms)

    for i, row in enumerate(probes, 1):
        if len(passers) >= max(TARGET * 3, 45):
            break
        # leave headroom for concurrent chart-fill child (~1–2G)
        _wait_mem(10)
        name = row["factor_name"]
        formula = row["lqtp_formula"]
        print(
            f"[{i}/{len(probes)}] {row['from']} cs={int(bool(row.get('cs_rank')))} "
            f"ops={row['ops']} ric={row.get('rank_ic', 0):.4f} {name} passers={len(passers)}",
            flush=True,
        )
        try:
            full = platform_analyze(stub, md, formula, begin=BEGIN, end=END, name="")
            if full.get("error"):
                full = platform_analyze(
                    stub, md, f"(0.0 + ({formula}))", begin=BEGIN, end=END, name=""
                )
            if full.get("error"):
                print("  ERR", str(full["error"])[:120], flush=True)
                continue
            pic = float(full.get("mean_ic") or 0)
            sh = full.get("long_short_sharpe")
            if abs(pic) <= HARD_IC:
                print(f"  DROP ic={abs(pic):.4f} sh={sh}", flush=True)
                continue
            nice = _clean(formula, pic)
            if _blocked(used_forms, nice):
                print("  SKIP dup formula", flush=True)
                continue
            used_forms.add(_norm(nice))
            used_forms.add(
                "-" + _norm(nice) if not _norm(nice).startswith("-") else _norm(nice)[1:]
            )
            passers.append(
                {
                    **row,
                    "lqtp_formula": nice,
                    "platform_mean_ic": pic,
                    "meta": full,
                    "abs_sharpe": abs(float(sh) if sh is not None else 0.0),
                    "cs_rank": _has_cs_rank(nice),
                }
            )
            print(f"  PASS ic={abs(pic):.4f} sh={sh} cs={_has_cs_rank(nice)}", flush=True)
        except Exception as e:
            print("  EXC", e, flush=True)
        gc.collect()
        time.sleep(0.05)

    # Prefer no-cs-rank, then sharpe, then IC
    passers.sort(
        key=lambda x: (
            0 if not x.get("cs_rank") else 1,
            -x["abs_sharpe"],
            -abs(x["platform_mean_ic"]),
        )
    )
    print("passers", len(passers), flush=True)
    for c in passers[:40]:
        print(
            f"  cand cs={int(bool(c.get('cs_rank')))} sh={c['abs_sharpe']:.3f} "
            f"ic={abs(c['platform_mean_ic']):.4f} {c['factor_name']}",
            flush=True,
        )

    added: list[dict[str, Any]] = []
    for row in passers:
        if len(added) >= TARGET:
            break
        name = row["factor_name"]
        formula = row["lqtp_formula"]
        pic = float(row["platform_mean_ic"])
        meta = row["meta"]
        if row.get("need_reg") and name not in existing:
            submit = formula if formula not in owners else f"(0.0 + ({formula}))"
            try:
                reg = platform_analyze(stub, md, submit, begin=BEGIN, end=END, name=name)
                if not reg.get("error"):
                    existing.add(name)
                    meta = reg
                    pic = float(reg.get("mean_ic") or pic)
                    formula = _clean(formula, pic)
                else:
                    print("  REG_ERR", name, str(reg["error"])[:100], flush=True)
            except Exception as e:
                print("  REG_EXC", name, e, flush=True)

        abs_full = abs(pic)
        if abs_full <= HARD_IC:
            continue
        item = {
            "factor_id": name,
            "display_name": name,
            "candidate_id": row.get("candidate_id"),
            "source": "formulas_csv" if row["from"] == "formulas_csv" else "pack_dsl",
            "label": "本周新挖",
            "source_label": "本周新挖",
            "ok": True,
            "ok_for_select": True,
            "from_pack_top_pick": row["from"] != "formulas_csv",
            "already_neutralized": bool(row.get("already_neutralized")),
            "neutralization_note": row.get("neutralization_note") or "未中性化（包内DSL）",
            "has_formula": True,
            "dsl": formula,
            "lqtp_formula": formula,
            "formula_raw": row.get("formula_src"),
            "lqtp_native": True,
            "formula_from": "topup_ic02_more",
            "pack_rank_ic": row.get("rank_ic"),
            "platform_mean_ic": pic,
            "platform_icir": meta.get("icir"),
            "platform_coverage": meta.get("coverage"),
            "platform_long_short_sharpe": meta.get("long_short_sharpe"),
            "mean_rank_ic": pic,
            "abs_mean_rank_ic": abs_full,
            "display_rank_ic": abs_full,
            "rank_icir": meta.get("icir"),
            "display_rank_icir": abs(float(meta.get("icir") or 0)),
            "mean_daily_coverage": meta.get("coverage"),
            "long_short_sharpe": meta.get("long_short_sharpe"),
            "sign_flipped": False,
            "sign_baked_into_formula": True,
            "platform_submit": "submitted",
            "materialize": "platform_only",
            "ops": row.get("ops"),
            "cs_rank": bool(row.get("cs_rank")),
        }
        dpath = work / "reports_weekly_dug" / f"{name}.html"
        if dpath.exists():
            item["report_href"] = f"reports_weekly_dug/{name}.html"
        added.append(item)
        print(
            f"ADD {name} ic={abs_full:.4f} sh={meta.get('long_short_sharpe')} cs={item['cs_rank']}",
            flush=True,
        )

    print("ADD_N", len(added), flush=True)
    selected.extend(added)
    for r in selected:
        f = rationalize_lqtp_formula(str(r.get("lqtp_formula") or ""), sign_flipped=False).replace(
            "RET", "ret"
        )
        f = re.sub(r"\(\s*0\.0\s*\+\s*\((.*)\)\s*\)", r"(\1)", f)
        r["lqtp_formula"] = f
        r["dsl"] = f
        r["sign_flipped"] = False
        r["display_name"] = display_factor_name(r)
    selected.sort(key=lambda x: float(x.get("display_rank_ic") or 0), reverse=True)
    weekly["selected"] = selected
    weekly["n_selected"] = len(selected)
    weekly["topup_ic02_more_added"] = len(added)
    weekly["platform_note"] = (
        f"本周新挖 {len(selected)} 条；硬门槛 |RankIC|>2%；本轮新补 {len(added)} 条（优先无截面rank、多空Sharpe）。"
    )
    weekly["generated_at"] = datetime.now(timezone.utc).isoformat()
    for i, row in enumerate(selected, 1):
        if i % 40 == 0:
            print("patch", i, flush=True)
        patch_report_formula(work, row)
    wp.write_text(json.dumps(weekly, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_SUBMIT.write_text(
        json.dumps(
            {
                "n_added": len(added),
                "baseline": baseline,
                "n_selected": len(selected),
                "added": added,
                "n_passers": len(passers),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    class A:
        work_dir = work
        html = work / "reports/factor_rankic_screening_index.html"
        threshold = 0.02

    patch_screening_html(A(), payload=weekly)
    html = (work / "reports/factor_rankic_screening_index.html").read_text()
    hero = re.search(r"<b>(\d+)</b><span>入选因子", html)
    print(
        {
            "baseline": baseline,
            "n": weekly["n_selected"],
            "added": len(added),
            "passers": len(passers),
            "hero": hero.group(1) if hero else None,
            "new": [
                (
                    x["display_name"],
                    round(float(x.get("display_rank_ic") or 0), 4),
                    round(float(x.get("long_short_sharpe") or 0), 3),
                    bool(x.get("cs_rank")),
                )
                for x in added
            ],
            "mem": round(_mem_avail_gb(), 1),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
