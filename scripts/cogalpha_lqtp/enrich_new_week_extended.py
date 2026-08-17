#!/usr/bin/env python3
"""Re-analyze all new-week lakes with extended metrics and re-render HTML reports.

Memory: one factor at a time, DuckDB only (no FE). Abort if MemAvailable < 12G
or process RSS > 15G.
"""
from __future__ import annotations

import argparse
import gc
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

os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from scripts.cogalpha_lqtp.crawl_candidate_pool_neutral_rankic import (  # noqa: E402
    _weekly_section_html,
)
from scripts.cogalpha_lqtp.eval_extensions import (  # noqa: E402
    compute_extended_eval,
    merge_extended_into_analysis,
)
from scripts.cogalpha_lqtp.eval_lake_fast import analyze_factor_parquet_duckdb  # noqa: E402
from scripts.cogalpha_lqtp.memory_utils import read_mem_available_gb, release_memory  # noqa: E402
from scripts.cogalpha_lqtp.report_html import render_factor_report  # noqa: E402

WORK = ROOT / "data/cogalpha_lqtp_production"
MIN_AVAIL = 12.0
MAX_RSS = 15.0


def _rss_gb() -> float:
    with open("/proc/self/status", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def _guard(tag: str) -> None:
    avail = read_mem_available_gb()
    rss = _rss_gb()
    if avail < MIN_AVAIL or rss > MAX_RSS:
        raise MemoryError(f"{tag}: avail={avail:.1f}G rss={rss:.1f}G")


def _resolve_lake(work: Path, row: dict[str, Any]) -> Path | None:
    name = str(row.get("display_name") or row.get("factor_id"))
    cands = [
        Path(str(row.get("lake_path") or "")),
        work / "factor_lake" / name / "values.parquet",
        work / "candidate_pool_neutral_cache" / "week_new_20260816" / "lakes" / f"{name}.parquet",
    ]
    # prefer FE lake when marked
    if row.get("materialize") == "fe_data_access_yearly":
        cands = [cands[1], cands[0], cands[2]]
    for p in cands:
        if p and p.exists() and p.stat().st_size > 1000:
            return p
    return None


def _patch_index(work: Path, payload: dict[str, Any]) -> None:
    html_path = work / "reports/factor_rankic_screening_index.html"
    text = html_path.read_text(encoding="utf-8")
    section = _weekly_section_html(payload, threshold=float(payload.get("threshold", 0.02)))
    week_tag = payload.get("week_tag") or "本周"
    section = section.replace(
        "本周新挖 · 中性化候选",
        f"本周新挖 · {week_tag}（FE/面板 + 扩展评估）",
        1,
    )
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
    for brand in (
        "factorminer",
        "FactorMiner",
        "cogalpha",
        "CogAlpha",
        "alphasage",
        "AlphaSage",
        "evoalpha",
        "EvoAlpha",
    ):
        text = text.replace(brand, "cand")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = html_path.with_name(f"{html_path.name}.bak_ext_{stamp}")
    bak.write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")
    html_path.write_text(text, encoding="utf-8")
    print(f"patched {html_path.name} backup={bak.name}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, default=WORK)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-index", action="store_true")
    args = ap.parse_args()
    work = args.work_dir
    wp = work / "reports/weekly_dug_neutral_rankic.json"
    weekly = json.loads(wp.read_text(encoding="utf-8"))
    fwd = work / "lqtp_fwd_vwap_returns_cache.parquet"
    ind = work / "eval_cache/industry_sw_l1.parquet"
    mcap = work / "eval_cache/market_cap.parquet"
    out_dir = work / "reports_weekly_dug_wnew"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = work / "reports/new_week_extended_enrich.log"

    rows = list(weekly.get("selected") or [])
    if args.only:
        only = set(args.only)
        rows = [r for r in rows if r.get("display_name") in only or r.get("factor_id") in only]
    if args.limit:
        rows = rows[: args.limit]

    print(
        f"enrich n={len(rows)} avail={read_mem_available_gb():.1f}G ind={ind.exists()} mcap={mcap.exists()}",
        flush=True,
    )
    ok = 0
    for i, row in enumerate(rows, 1):
        name = str(row.get("display_name") or row.get("factor_id"))
        try:
            _guard(f"pre:{name}")
        except MemoryError as e:
            print(f"STOP {e}", flush=True)
            break
        lake = _resolve_lake(work, row)
        if lake is None:
            msg = f"[{i}/{len(rows)}] MISS_LAKE {name}"
            print(msg, flush=True)
            with log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
            continue
        t0 = time.time()
        print(f"[{i}/{len(rows)}] EXT {name} lake={lake.name} avail={read_mem_available_gb():.1f}G", flush=True)
        try:
            analysis = analyze_factor_parquet_duckdb(
                factor_path=lake, fwd_returns_path=fwd, return_kind="vwap_to_vwap"
            )
            _guard(f"mid:{name}")
            extended = compute_extended_eval(
                factor_path=lake,
                fwd_returns_path=fwd,
                industry_path=ind if ind.exists() else None,
                market_cap_path=mcap if mcap.exists() else None,
            )
            analysis = merge_extended_into_analysis(analysis, extended)
            mean_ic = float(analysis.get("mean_rank_ic", float("nan")))
            # keep display convention: nonnegative IC in table (sign already baked/flipped in lakes)
            if mean_ic == mean_ic and mean_ic < 0 and not row.get("sign_flipped"):
                # don't flip here — lakes already aligned for FE/COS paths
                pass
            dsl = str(row.get("dsl") or row.get("lqtp_formula") or "").strip()
            is_panel = not dsl
            meta = {
                "engine": (
                    "factor_engine"
                    if row.get("materialize") == "fe_data_access_yearly" and dsl
                    else "local_panel"
                ),
                "eval_route": (
                    row.get("materialize")
                    if row.get("materialize") == "fe_data_access_yearly" and dsl
                    else "cos_panel"
                ),
                "values_path": str(lake),
                "date_range": ["2019-01-01", "2026-06-30"],
                "formula": dsl,
                "eval_engine": "duckdb_panel",
                "return_kind": "vwap_to_vwap",
                "source_label": row.get("label") or "本周新挖",
                "ic_sign_flipped": bool(row.get("sign_flipped")),
                "platform_submit": row.get("platform_submit") or (
                    "skipped_no_formula" if is_panel else "submitted"
                ),
                "data_source": "data_access"
                if row.get("materialize") == "fe_data_access_yearly" and dsl
                else "cos_panel",
                "extended_eval": True,
            }
            out = out_dir / f"{name}.html"
            render_factor_report(
                factor_name=name,
                dsl=dsl,
                python_code="",
                analysis=analysis,
                backtest_rows=None,
                out_path=out,
                eval_mode="duckdb_panel_vwap_to_vwap_t1t2_realto",
                materialize_meta=meta,
                engine=meta["engine"],
                eval_route=str(meta["eval_route"]),
                work_dir=work,
                annotation={
                    "factor_id": name,
                    "function_name": name,
                    "source": "本周新挖",
                    "note": (
                        "扩展评估：行业/市值中性、滚动IC、半衰期、分层/多空诊断"
                        if dsl
                        else "扩展评估基于中性化面板；源侧未提供可回收公式"
                    ),
                    "theme": "量价因子",
                },
            )
            text = out.read_text(encoding="utf-8", errors="ignore")
            for brand in (
                "factorminer",
                "FactorMiner",
                "cogalpha",
                "CogAlpha",
                "alphasage",
                "AlphaSage",
                "evoalpha",
                "EvoAlpha",
            ):
                text = text.replace(brand, "cand")
            out.write_text(text, encoding="utf-8")

            ext = analysis.get("extended_eval") or {}
            ind_neu = (ext.get("industry_neutral") or {}).get("mean_rank_ic")
            size_neu = (ext.get("size_neutral") or {}).get("mean_rank_ic")
            both_neu = (ext.get("industry_size_neutral") or {}).get("mean_rank_ic")
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
                    "long_short_return_sum": float(analysis.get("long_short_return_sum", float("nan"))),
                    "decile_monotonicity": float(analysis.get("decile_monotonicity", float("nan"))),
                    "decile_spread": float(analysis.get("decile_spread", float("nan"))),
                    "ls_max_drawdown": float(analysis.get("ls_max_drawdown", float("nan"))),
                    "ls_win_rate": float(analysis.get("ls_win_rate", float("nan"))),
                    "ic_half_life_days": float(ext.get("ic_half_life_days", float("nan"))),
                    "industry_neutral_rank_ic": float(ind_neu) if ind_neu is not None else None,
                    "size_neutral_rank_ic": float(size_neu) if size_neu is not None else None,
                    "industry_size_neutral_rank_ic": float(both_neu) if both_neu is not None else None,
                    "report_href": f"reports_weekly_dug_wnew/{name}.html",
                    "lake_path": str(lake),
                    "extended_eval_ok": True,
                    "extended_eval_sec": round(time.time() - t0, 2),
                }
            )
            ok += 1
            n_png = text.count("data:image/png;base64,")
            msg = (
                f"  OK {name} png={n_png} |IC|={row['display_rank_ic']:.4f} sh={row['long_short_sharpe']:.3f} "
                f"mono={row.get('decile_monotonicity')} ind={row.get('industry_neutral_rank_ic')} "
                f"sec={row['extended_eval_sec']} avail={read_mem_available_gb():.1f}G rss={_rss_gb():.2f}G"
            )
            print(msg, flush=True)
            with log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
            release_memory(analysis, extended)
            gc.collect()
        except MemoryError as e:
            print(f"  MEM {e}", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            msg = f"  ERR {name} {e}"
            print(msg, flush=True)
            with log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
            row["extended_eval_ok"] = False
            row["extended_eval_error"] = str(e)[:300]
            gc.collect()

    weekly["selected"].sort(key=lambda r: float(r.get("display_rank_ic") or 0), reverse=True)
    weekly["extended_enriched_at"] = datetime.now().isoformat()
    weekly["extended_enriched_ok"] = ok
    weekly["platform_note"] = (
        f"本周 {weekly.get('n_selected')} 条；扩展评估完成 {ok} 条"
        f"（RankIC/ICIR/覆盖/多空Sharpe/换手/分层单调/回撤/胜率/行业·市值中性/滚动IC/半衰期等）；"
        "有 DSL 的走 FE+data_access 物化，其余 COS 中性化面板。"
    )
    wp.write_text(json.dumps(weekly, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.skip_index:
        _patch_index(work, weekly)
    print(f"DONE ok={ok}/{len(rows)} avail={read_mem_available_gb():.1f}G", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
