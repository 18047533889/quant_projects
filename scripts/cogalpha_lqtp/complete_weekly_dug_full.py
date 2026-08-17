#!/usr/bin/env python3
"""Full weekly dug pack: RankIC + 分层 + 多空 + 扩展评估 + TopK OPEN charts.

Memory-safe: one factor at a time; no FE; abort/wait when MemAvailable is low.
Prefers FE-retest lakes when present.
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

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "cogalpha_lqtp")]

os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from scripts.cogalpha_lqtp.crawl_candidate_pool_neutral_rankic import (  # noqa: E402
    _weekly_section_html,
    refresh_index_hero_totals,
)
from scripts.cogalpha_lqtp.eval_extensions import (  # noqa: E402
    compute_extended_eval,
    merge_extended_into_analysis,
)
from scripts.cogalpha_lqtp.eval_lake_fast import analyze_factor_parquet_duckdb  # noqa: E402
from scripts.cogalpha_lqtp.lqtp_client import (  # noqa: E402
    DEFAULT_SERVER,
    LqtpTokenManager,
    require_credentials,
    run_topk_backtest_for_long_df,
)
from scripts.cogalpha_lqtp.memory_utils import read_mem_available_gb, release_memory  # noqa: E402
from scripts.cogalpha_lqtp.report_html import render_factor_report  # noqa: E402
from scripts.cogalpha_lqtp.run_production_batch import _json_float  # noqa: E402
from scripts.cogalpha_lqtp.factor_eval import fetch_lqtp_open_returns  # noqa: E402

WORK = ROOT / "data/cogalpha_lqtp_production"
MIN_AVAIL = 10.0
BEGIN, END = 20190102, 20260630


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


def _rss_gb() -> float:
    with open("/proc/self/status", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def _wait_mem(min_gb: float = MIN_AVAIL, tag: str = "") -> None:
    while True:
        avail = read_mem_available_gb()
        if avail >= min_gb and _rss_gb() < 14.0:
            return
        print(f"WAIT_MEM {tag} avail={avail:.1f}G rss={_rss_gb():.1f}G", flush=True)
        gc.collect()
        time.sleep(20)


def _resolve_lake(work: Path, row: dict[str, Any]) -> Path | None:
    name = str(row.get("display_name") or row.get("factor_id"))
    cands = [
        Path(str(row.get("fe_retest_lake") or "")),
        Path(str(row.get("lake_path") or "")),
        work / "factor_lake" / name / "values.parquet",
        work / "candidate_pool_neutral_cache" / "week_new_20260816" / "lakes" / f"{name}.parquet",
    ]
    mat = str(row.get("materialize") or "")
    if "fe_data_access" in mat:
        cands = [cands[2], cands[0], cands[1], cands[3]]
    for p in cands:
        if p and str(p) not in {".", ""} and p.exists() and p.stat().st_size > 1000:
            return p
    return None


def _load_long(lake: Path) -> pd.DataFrame:
    df = pd.read_parquet(lake)
    cols = {c.lower(): c for c in df.columns}
    td = cols.get("trade_date") or cols.get("date") or cols.get("tradedate")
    sym = cols.get("symbol") or cols.get("code") or cols.get("ticker")
    val = cols.get("value") or cols.get("factor") or cols.get("factor_value")
    if not (td and sym and val):
        raise RuntimeError(f"lake columns missing: {list(df.columns)}")
    out = df[[td, sym, val]].copy()
    out.columns = ["trade_date", "symbol", "value"]
    out["trade_date"] = (
        pd.to_datetime(out["trade_date"]).dt.strftime("%Y%m%d").astype(int)
        if not str(out["trade_date"].dtype).startswith("int")
        else out["trade_date"].astype(int)
    )
    out["symbol"] = out["symbol"].astype(str)
    out["value"] = out["value"].astype(float)
    return out.dropna(subset=["value"])


def _scrub(text: str) -> str:
    for brand in (
        "factorminer",
        "FactorMiner",
        "cogalpha",
        "CogAlpha",
        "alphasage",
        "AlphaSage",
        "evoalpha",
        "EvoAlpha",
        "hsunbj",
        "quantaalpha",
    ):
        text = text.replace(brand, "cand")
    return text


def _audit_html(path: Path) -> dict[str, bool]:
    if not path.exists():
        return {"html": False}
    t = path.read_text(encoding="utf-8", errors="ignore")
    topk_m = re.search(r"<h2>[^<]*TopK[\s\S]*?(?=<h2|$)", t, re.I)
    topk_chunk = topk_m.group(0) if topk_m else ""
    return {
        "html": True,
        "rankic": "RankIC" in t and "Daily RankIC" in t,
        "decile": "分层回测" in t and "Decile cumulative return" in t,
        "long_short": "多空" in t and "Cumulative long-short" in t,
        "extended": "扩展评估" in t and "分年 RankIC" in t,
        "topk": bool(topk_chunk) and ("data:image" in topk_chunk or "Sharpe" in topk_chunk),
        "imgs_ge3": t.count("data:image") >= 3,
        "imgs_ge6": t.count("data:image") >= 6,
    }


def _patch_index(work: Path, weekly: dict[str, Any]) -> None:
    html_path = work / "reports/factor_rankic_screening_index.html"
    text = html_path.read_text(encoding="utf-8")
    section = _weekly_section_html(weekly, threshold=float(weekly.get("threshold", 0.02)))
    week_tag = weekly.get("week_tag") or "本周"
    section = section.replace(
        "本周新挖 · 中性化候选",
        f"本周新挖 · {week_tag}（全量：分层/多空/扩展/TopK）",
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
    text = _scrub(text)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = html_path.with_name(f"{html_path.name}.bak_full_{stamp}")
    bak.write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        text = refresh_index_hero_totals(text)
    except Exception as exc:  # noqa: BLE001
        print(f"hero refresh skip: {exc}", flush=True)
    html_path.write_text(text, encoding="utf-8")
    print(f"patched index backup={bak.name}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, default=WORK)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-topk", action="store_true")
    ap.add_argument("--skip-index", action="store_true")
    ap.add_argument("--min-avail-gb", type=float, default=MIN_AVAIL)
    args = ap.parse_args()
    _load_env()

    work = args.work_dir
    wp = work / "reports/weekly_dug_neutral_rankic.json"
    weekly = json.loads(wp.read_text(encoding="utf-8"))
    rows = list(weekly.get("selected") or [])
    if args.only:
        only = set(args.only)
        rows = [r for r in rows if r.get("display_name") in only]
    if args.limit:
        rows = rows[: args.limit]

    fwd = work / "lqtp_fwd_vwap_returns_cache.parquet"
    ind = work / "eval_cache/industry_sw_l1.parquet"
    mcap = work / "eval_cache/market_cap.parquet"
    out_dir = work / "reports_weekly_dug_wnew"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = work / "reports/weekly_full_pack.log"
    open_cache = work / "lqtp_open_returns_cache.parquet"

    token_mgr = None
    open_returns = None
    if not args.skip_topk:
        try:
            user, pw = require_credentials()
            server = os.getenv("LQTP_SERVER") or DEFAULT_SERVER
            token_mgr = LqtpTokenManager.login(server, user, pw)
            print("LQTP login ok for TopK", flush=True)
            if open_cache.exists():
                _wait_mem(args.min_avail_gb, "load_open")
                open_returns = pd.read_parquet(open_cache)
                print(f"open_returns cache rows={len(open_returns)}", flush=True)
            else:
                print("fetching open returns cache (one-time)...", flush=True)
                _wait_mem(max(args.min_avail_gb, 12.0), "fetch_open")
                open_returns = fetch_lqtp_open_returns(
                    token=token_mgr.token,
                    begin_date=BEGIN,
                    end_date=END,
                    server=server,
                    cache_path=open_cache,
                )
                print(f"open_returns fetched rows={len(open_returns)}", flush=True)
                gc.collect()
        except Exception as exc:  # noqa: BLE001
            print(f"WARN TopK disabled: {exc}", flush=True)
            args.skip_topk = True
            token_mgr = None
            open_returns = None

    print(
        f"full_pack n={len(rows)} topk={not args.skip_topk} avail={read_mem_available_gb():.1f}G",
        flush=True,
    )
    ok = 0
    audits: list[dict[str, Any]] = []
    for i, row in enumerate(rows, 1):
        name = str(row.get("display_name") or row.get("factor_id"))
        _wait_mem(args.min_avail_gb, name)
        lake = _resolve_lake(work, row)
        if lake is None:
            msg = f"[{i}/{len(rows)}] MISS_LAKE {name}"
            print(msg, flush=True)
            with log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
            audits.append({"name": name, "ok": False, "error": "miss_lake"})
            continue
        t0 = time.time()
        print(f"[{i}/{len(rows)}] FULL {name} lake={lake} avail={read_mem_available_gb():.1f}G", flush=True)
        try:
            analysis = analyze_factor_parquet_duckdb(
                factor_path=lake, fwd_returns_path=fwd, return_kind="vwap_to_vwap"
            )
            try:
                extended = compute_extended_eval(
                    factor_path=lake,
                    fwd_returns_path=fwd,
                    industry_path=ind if ind.exists() else None,
                    market_cap_path=mcap if mcap.exists() else None,
                )
                analysis = merge_extended_into_analysis(analysis, extended)
            except Exception as ext_exc:  # noqa: BLE001
                analysis = dict(analysis)
                analysis["extended_eval_error"] = str(ext_exc)[:300]

            backtest_rows: list[dict[str, Any]] | None = None
            topk_summary: dict[str, Any] = {}
            if not args.skip_topk and token_mgr is not None and open_returns is not None:
                try:
                    long_df = _load_long(lake)
                    # Align display sign: if mean IC negative, flip for TopK so long book is positive side
                    mean_ic0 = float(analysis.get("mean_rank_ic") or 0.0)
                    if mean_ic0 == mean_ic0 and mean_ic0 < 0:
                        long_df = long_df.copy()
                        long_df["value"] = -long_df["value"]
                    bt_rows, _bt_id, topk_summary, _excl = run_topk_backtest_for_long_df(
                        token_mgr,
                        long_df,
                        begin_date=BEGIN,
                        end_date=END,
                        server=os.getenv("LQTP_SERVER") or DEFAULT_SERVER,
                        open_returns_long=open_returns,
                    )
                    backtest_rows = bt_rows or None
                    release_memory(long_df)
                except Exception as bt_exc:  # noqa: BLE001
                    print(f"  TopK ERR {name}: {bt_exc}", flush=True)
                    topk_summary = {}
                    backtest_rows = None

            mean_ic = float(analysis.get("mean_rank_ic", float("nan")))
            # Formula already permanently negated → keep signed IC (should be ≥0)
            if row.get("formula_sign_flipped"):
                display_ic = mean_ic if mean_ic == mean_ic else float("nan")
            else:
                display_ic = abs(mean_ic) if mean_ic == mean_ic else float("nan")
            dsl = str(row.get("lqtp_formula") or row.get("fe_dsl") or row.get("dsl") or "")
            meta = {
                "engine": "factor_engine" if "fe_data_access" in str(row.get("materialize") or "") else "local_panel",
                "eval_route": row.get("materialize") or "cos_panel",
                "values_path": str(lake),
                "date_range": ["2019-01-01", "2026-06-30"],
                "formula": dsl,
                "eval_engine": "duckdb_panel",
                "return_kind": "vwap_to_vwap",
                "source_label": row.get("label") or "本周新挖",
                "ic_sign_flipped": bool(row.get("formula_sign_flipped") or row.get("sign_flipped")),
                "platform_submit": row.get("platform_submit"),
                "extended_eval": True,
                "note": "全量报告：RankIC/分层/多空/扩展评估（与既有周报口径一致，不含 TopK）",
            }
            out = out_dir / f"{name}.html"
            eval_mode = "duckdb_panel_vwap_to_vwap_t1t2_realto"
            if backtest_rows:
                eval_mode = f"{eval_mode}+topk_open"
            render_factor_report(
                factor_name=name,
                dsl=dsl,
                python_code=str(row.get("python_code") or ""),
                analysis=analysis,
                backtest_rows=backtest_rows,
                out_path=out,
                eval_mode=eval_mode,
                materialize_meta=meta,
                engine=str(meta["engine"]),
                eval_route=str(meta["eval_route"]),
                work_dir=work,
                topk_summary=topk_summary or None,
                annotation={
                    "factor_id": name,
                    "function_name": name,
                    "source": "本周新挖",
                    "note": "全量落图：分层/多空/扩展评估（同既有周报，不含 TopK）",
                    "theme": "量价因子",
                },
            )
            out.write_text(_scrub(out.read_text(encoding="utf-8", errors="ignore")), encoding="utf-8")

            row.update(
                {
                    "display_rank_ic": display_ic,
                    "display_rank_icir": abs(float(analysis.get("rank_icir") or 0.0)),
                    "mean_rank_ic": mean_ic,
                    "abs_mean_rank_ic": display_ic,
                    "rank_icir": float(analysis.get("rank_icir", float("nan"))),
                    "long_short_sharpe": float(analysis.get("long_short_sharpe", float("nan"))),
                    "long_short_return": _json_float(analysis.get("long_short_return")),
                    "mean_daily_coverage": float(analysis.get("mean_daily_coverage", float("nan"))),
                    "lake_path": str(lake),
                    "full_pack_at": datetime.now().isoformat(),
                    "full_pack_sec": round(time.time() - t0, 1),
                    "report_href": f"/reports_weekly_dug_wnew/{name}.html",
                }
            )
            if topk_summary:
                row.update(
                    {
                        "backtest_total_ret": _json_float(topk_summary.get("total_return")),
                        "backtest_ann_ret": _json_float(topk_summary.get("annualized_return")),
                        "backtest_sharpe": _json_float(topk_summary.get("sharpe")),
                        "backtest_max_drawdown": _json_float(topk_summary.get("max_drawdown")),
                        "backtest_volatility": _json_float(topk_summary.get("volatility")),
                        "backtest_calmar": _json_float(topk_summary.get("calmar")),
                        "backtest_win_rate": _json_float(topk_summary.get("win_rate")),
                        "backtest_avg_turnover": _json_float(topk_summary.get("avg_turnover")),
                        "backtest_total_commission": _json_float(topk_summary.get("total_commission")),
                        "backtest_trading_days": int(topk_summary.get("trading_days") or 0),
                        "backtest_final_nav": _json_float(topk_summary.get("final_nav")),
                    }
                )

            audit = _audit_html(out)
            audit.update(
                {
                    "name": name,
                    "ok": True,
                    "ls_sharpe": row.get("long_short_sharpe"),
                    "ic": display_ic,
                    "topk_sharpe": row.get("backtest_sharpe"),
                    "sec": row.get("full_pack_sec"),
                }
            )
            audits.append(audit)
            ok += 1
            msg = (
                f"  OK {name} ic={display_ic:.4f} sh={row.get('long_short_sharpe'):.3f} "
                f"topk={row.get('backtest_sharpe')} "
                f"charts={ {k: audit.get(k) for k in ('rankic','decile','long_short','extended','topk')} } "
                f"sec={row.get('full_pack_sec')} avail={read_mem_available_gb():.1f}G"
            )
            print(msg, flush=True)
            with log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
            release_memory(analysis, backtest_rows, topk_summary)
            gc.collect()
            weekly["selected"].sort(key=lambda r: float(r.get("display_rank_ic") or 0), reverse=True)
            weekly["full_pack_at"] = datetime.now().isoformat()
            weekly["full_pack_ok"] = ok
            wp.write_text(json.dumps(weekly, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            msg = f"  ERR {name} {exc}"
            print(msg, flush=True)
            with log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
            audits.append({"name": name, "ok": False, "error": str(exc)[:300]})
            gc.collect()

    weekly["selected"].sort(key=lambda r: float(r.get("display_rank_ic") or 0), reverse=True)
    weekly["full_pack_at"] = datetime.now().isoformat()
    weekly["full_pack_ok"] = ok
    weekly["full_pack_audits"] = audits
    weekly["platform_note"] = (
        f"本周 {weekly.get('n_selected')} 条全量落图完成 {ok} 条"
        f"（RankIC/分层/多空/扩展评估；不含 TopK）。"
    )
    wp.write_text(json.dumps(weekly, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_path = work / "reports/weekly_full_pack_audit.json"
    audit_path.write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.skip_index:
        try:
            _patch_index(work, weekly)
        except Exception as exc:  # noqa: BLE001
            print(f"index patch ERR {exc}", flush=True)

    bad = [a for a in audits if not a.get("ok") or not a.get("rankic") or not a.get("decile") or not a.get("long_short")]
    print(
        f"DONE ok={ok}/{len(rows)} bad_core={len(bad)} avail={read_mem_available_gb():.1f}G "
        f"audit={audit_path}",
        flush=True,
    )
    return 0 if not bad else 2


if __name__ == "__main__":
    raise SystemExit(main())
