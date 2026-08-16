#!/usr/bin/env python3
"""FE + data_access yearly materialize for this week's DSL-capable factors.

- Resolves DSL from screening catalog / EXTRA_PYTHON_FE_DSL
- Year-chunk FE via materialize_halfyear_chunk (FACTOR_ENGINE_MAX_MEMORY_MB<=12000)
- DuckDB RankIC on FE lakes; refresh reports under reports_weekly_dug_wnew/
- Factors without recoverable formulas stay on COS panel (noted in JSON)
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "cogalpha_lqtp")]

from scripts.cogalpha_lqtp.crawl_candidate_pool_neutral_rankic import (  # noqa: E402
    _weekly_section_html,
)
from scripts.cogalpha_lqtp.eval_lake_fast import analyze_factor_parquet_duckdb  # noqa: E402
from scripts.cogalpha_lqtp.fill_one_weekly_chart import (  # noqa: E402
    display_factor_name,
    materialize_dsl_by_halfyear,
    _flip_lake,
    _has_cs_rank,
)
from scripts.cogalpha_lqtp.memory_utils import read_mem_available_gb, release_memory  # noqa: E402
from scripts.cogalpha_lqtp.report_html import render_factor_report  # noqa: E402

WORK = ROOT / "data/cogalpha_lqtp_production"


def _load_dsl_maps() -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    spec = importlib.util.spec_from_file_location(
        "py_fe_map", ROOT / "scripts/cogalpha_lqtp/_python_fe_dsl_map.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    py_map: dict[str, str] = dict(getattr(mod, "EXTRA_PYTHON_FE_DSL") or {})

    cat = json.loads((WORK / "screening_reeval_catalog.json").read_text(encoding="utf-8"))
    items = cat if isinstance(cat, list) else cat.get("factors") or []
    by = {str(x.get("function_name") or x.get("factor_id")): x for x in items}
    return py_map, by


def resolve_dsl(row: dict[str, Any], py_map: dict[str, str], by: dict[str, dict[str, Any]]) -> tuple[str | None, str]:
    cid = str(row.get("candidate_id") or "")
    name = display_factor_name(row)
    base = re.sub(r"_v\d+$", "", cid)
    cands = [
        base,
        name.replace("cand_", "factor_"),
        "factor_" + name.replace("cand_", ""),
        base.replace("factor_", ""),
    ]
    for c in cands:
        if c in py_map and str(py_map[c]).strip():
            return str(py_map[c]).strip(), f"python_map:{c}"
        if c in by and str(by[c].get("dsl") or "").strip():
            return str(by[c]["dsl"]).strip(), f"catalog:{c}"
    return None, ""


def _render(work: Path, name: str, dsl: str, lake: Path, analysis: dict[str, Any], row: dict[str, Any]) -> str:
    out_dir = work / "reports_weekly_dug_wnew"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.html"
    meta = {
        "engine": "factor_engine",
        "eval_route": "fe_data_access_yearly",
        "values_path": str(lake),
        "date_range": ["2019-01-01", "2026-06-30"],
        "formula": dsl,
        "eval_engine": "duckdb_panel",
        "return_kind": "vwap_to_vwap",
        "source_label": row.get("label") or "本周新挖",
        "ic_sign_flipped": bool(row.get("sign_flipped")),
        "platform_submit": "skipped_pending",
        "data_source": "data_access",
    }
    render_factor_report(
        factor_name=name,
        dsl=dsl,
        python_code="",
        analysis=analysis,
        backtest_rows=None,
        out_path=out,
        eval_mode="duckdb_panel_vwap_to_vwap_t1t2_realto",
        materialize_meta=meta,
        engine="factor_engine",
        eval_route="fe_data_access_yearly",
        work_dir=work,
        annotation={
            "factor_id": name,
            "function_name": name,
            "source": "本周新挖",
            "note": "FactorEngine + data_access 按年物化回测",
        },
    )
    text = out.read_text(encoding="utf-8", errors="ignore")
    for brand in ("factorminer", "FactorMiner", "cogalpha", "CogAlpha", "alphasage", "AlphaSage", "evoalpha", "EvoAlpha"):
        text = text.replace(brand, "cand")
    out.write_text(text, encoding="utf-8")
    return f"reports_weekly_dug_wnew/{name}.html"


def _patch_index(work: Path, payload: dict[str, Any]) -> None:
    html_path = work / "reports/factor_rankic_screening_index.html"
    text = html_path.read_text(encoding="utf-8")
    section = _weekly_section_html(payload, threshold=float(payload.get("threshold", 0.02)))
    week_tag = payload.get("week_tag") or datetime.now().strftime("%Y-%m-%d")
    section = section.replace("本周新挖 · 中性化候选", f"本周新挖 · {week_tag}（含 FE/data_access）", 1)
    # keep prev section if present
    prev_m = re.search(r'<section id="weekly-dug-prev"[\s\S]*?</section>', text)
    prev_html = prev_m.group(0) if prev_m else ""
    text = re.sub(r'<section id="weekly-dug-(?:neutral|prev)"[\s\S]*?</section>\s*', "", text, count=4)
    combined = section + ("\n" + prev_html if prev_html else "")
    m = re.search(
        r"(<section class=['\"]week2-section['\"]>|<section class=['\"]corr-embed['\"]>|</main>)",
        text,
    )
    if m:
        text = text[: m.start()] + combined + "\n" + text[m.start() :]
    else:
        text = text.replace("</main>", combined + "\n</main>")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = html_path.with_name(f"{html_path.name}.bak_fe_{stamp}")
    bak.write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")
    html_path.write_text(text, encoding="utf-8")
    print(f"patched {html_path} backup={bak.name}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, default=WORK)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-avail-gb", type=float, default=12.0)
    ap.add_argument("--dry-resolve", action="store_true")
    args = ap.parse_args()
    work = args.work_dir

    os.environ.setdefault("FACTOR_ENGINE_MAX_MEMORY_MB", "12000")
    os.environ.setdefault("FACTOR_ENGINE_RESERVE_GB", "8")
    os.environ.setdefault("FACTOR_ENGINE_DISABLE_PANEL_NATIVE", "1")
    os.environ.setdefault("FACTOR_ENGINE_DISABLE_CSE", "1")
    os.environ.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    os.environ.setdefault("FACTOR_ENGINE_OPERATOR_BACKEND", "polars")
    os.environ.setdefault("FACTOR_ENGINE_SPILL_DIR", str(work / "_fe_spill"))
    (work / "_fe_spill").mkdir(parents=True, exist_ok=True)

    wp = work / "reports/weekly_dug_neutral_rankic.json"
    weekly = json.loads(wp.read_text(encoding="utf-8"))
    py_map, by = _load_dsl_maps()

    targets: list[dict[str, Any]] = []
    for r in weekly.get("selected") or []:
        dsl, src = resolve_dsl(r, py_map, by)
        if dsl:
            r["dsl"] = dsl
            r["lqtp_formula"] = dsl
            r["has_formula"] = True
            r["formula_from"] = src
            targets.append(r)
        else:
            r.setdefault("fe_note", "no_recoverable_dsl_keep_cos_panel")
    if args.only:
        only = set(args.only)
        targets = [r for r in targets if display_factor_name(r) in only or r.get("factor_id") in only]
    # prefer no cs-rank first
    targets.sort(key=lambda r: (1 if _has_cs_rank(str(r.get("dsl") or "")) else 0, -float(r.get("display_rank_ic") or 0)))
    if args.limit:
        targets = targets[: args.limit]

    print(
        f"dsl_targets={len(targets)} avail={read_mem_available_gb():.1f}G "
        f"fe_max_mb={os.environ.get('FACTOR_ENGINE_MAX_MEMORY_MB')}",
        flush=True,
    )
    for r in targets:
        print(
            f"  {display_factor_name(r)} cs={int(_has_cs_rank(str(r.get('dsl'))))} "
            f"from={r.get('formula_from')} dsl={str(r.get('dsl'))[:70]}",
            flush=True,
        )
    wp.write_text(json.dumps(weekly, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_resolve:
        return 0

    fwd = work / "lqtp_fwd_vwap_returns_cache.parquet"
    log = work / "reports/new_week_fe_data_access.log"
    done = 0
    for i, row in enumerate(targets, 1):
        name = display_factor_name(row)
        dsl = str(row.get("dsl") or "")
        if _has_cs_rank(dsl):
            msg = f"[{i}/{len(targets)}] SKIP_CS_RANK {name}"
            print(msg, flush=True)
            with log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
            row["fe_status"] = "skipped_cs_rank"
            continue
        t0 = time.time()
        print(f"[{i}/{len(targets)}] FE {name} avail={read_mem_available_gb():.1f}G", flush=True)
        try:
            lake = materialize_dsl_by_halfyear(
                work, name, dsl, min_avail_gb=float(args.min_avail_gb)
            )
            analysis = analyze_factor_parquet_duckdb(
                factor_path=lake, fwd_returns_path=fwd, return_kind="vwap_to_vwap"
            )
            mean_ic = float(analysis.get("mean_rank_ic", float("nan")))
            if mean_ic == mean_ic and mean_ic < 0:
                _flip_lake(lake)
                release_memory(analysis)
                analysis = analyze_factor_parquet_duckdb(
                    factor_path=lake, fwd_returns_path=fwd, return_kind="vwap_to_vwap"
                )
                mean_ic = float(analysis.get("mean_rank_ic", float("nan")))
                row["sign_flipped"] = True
            href = _render(work, name, dsl, lake, analysis, row)
            row.update(
                {
                    "mean_rank_ic": mean_ic,
                    "abs_mean_rank_ic": abs(mean_ic) if mean_ic == mean_ic else float("nan"),
                    "display_rank_ic": abs(mean_ic) if mean_ic == mean_ic else float("nan"),
                    "rank_icir": float(analysis.get("rank_icir", float("nan"))),
                    "display_rank_icir": abs(float(analysis.get("rank_icir") or 0.0)),
                    "mean_daily_coverage": float(analysis.get("mean_daily_coverage", float("nan"))),
                    "long_short_sharpe": float(analysis.get("long_short_sharpe", float("nan"))),
                    "ls_mean_one_way_turnover": float(analysis.get("ls_mean_one_way_turnover", float("nan"))),
                    "materialize": "fe_data_access_yearly",
                    "lake_path": str(lake),
                    "report_href": href,
                    "fe_status": "ok",
                    "fe_elapsed_sec": round(time.time() - t0, 2),
                }
            )
            done += 1
            msg = (
                f"  OK {name} |IC|={row['display_rank_ic']:.4f} sh={row['long_short_sharpe']:.3f} "
                f"sec={row['fe_elapsed_sec']} avail={read_mem_available_gb():.1f}G"
            )
            print(msg, flush=True)
            with log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
            release_memory(analysis)
            gc.collect()
        except Exception as e:  # noqa: BLE001
            row["fe_status"] = f"error:{e}"[:300]
            msg = f"  ERR {name} {e}"
            print(msg, flush=True)
            with log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
            gc.collect()

    weekly["selected"].sort(key=lambda r: float(r.get("display_rank_ic") or 0), reverse=True)
    weekly["fe_data_access_at"] = datetime.now().isoformat()
    weekly["fe_data_access_ok"] = done
    weekly["platform_note"] = (
        f"本周 {weekly.get('n_selected')} 条；其中 {done} 条已用 FactorEngine+data_access 按年物化回测 "
        f"(MAX_MEMORY_MB={os.environ.get('FACTOR_ENGINE_MAX_MEMORY_MB')})；"
        "无公式的仍用 COS 中性化面板。"
    )
    wp.write_text(json.dumps(weekly, ensure_ascii=False, indent=2), encoding="utf-8")
    _patch_index(work, weekly)
    print(f"DONE fe_ok={done}/{len(targets)} avail={read_mem_available_gb():.1f}G", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
