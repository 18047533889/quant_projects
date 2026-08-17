#!/usr/bin/env python3
"""FE+data_access retest for weekly factors after DSL recovery.

COS panels are already CS-neutral (mean~0). This rematerializes from formula
(without relying on the neut panel) and refreshes RankIC charts.
Outer rank() is stripped for FE (RankIC-equivalent, avoids CS-rank RAM).
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

from scripts.cogalpha_lqtp.eval_extensions import (  # noqa: E402
    compute_extended_eval,
    merge_extended_into_analysis,
)
from scripts.cogalpha_lqtp.eval_lake_fast import analyze_factor_parquet_duckdb  # noqa: E402
from scripts.cogalpha_lqtp.fill_one_weekly_chart import (  # noqa: E402
    materialize_dsl_by_halfyear,
    _flip_lake,
    _has_cs_rank,
)
from scripts.cogalpha_lqtp.memory_utils import read_mem_available_gb, release_memory  # noqa: E402
from scripts.cogalpha_lqtp.permanent_flip_weekly_signs import (  # noqa: E402
    already_outer_negated,
    prep_fe_dsl,
    wipe_lake,
)
from scripts.cogalpha_lqtp.report_html import render_factor_report  # noqa: E402
from scripts.cogalpha_lqtp.run_production_batch import (  # noqa: E402
    _negate_formula,
    _negate_python_code,
)

WORK = ROOT / "data/cogalpha_lqtp_production"

# Fields that FE retest must not clobber when another process patched DSL on disk.
_PRESERVE_FROM_DISK = (
    "dsl",
    "fe_dsl",
    "lqtp_formula",
    "lqtp_submit_eligible",
    "formula_source",
    "formula_notes",
    "hand_dsl_at",
    "python_code",
    "convert_status",
    "platform_submit",
    "has_formula",
    "formula_sign_flipped",
    "ic_sign_flipped",
    "dsl_before_flip",
    "fe_dsl_before_flip",
    "formula_flipped_at",
)


def _merge_preserve_formulas(weekly: dict[str, Any], wp: Path) -> None:
    """Re-load disk rows and keep hand/DSL patches for names we haven't finished."""
    try:
        disk = json.loads(wp.read_text(encoding="utf-8"))
    except Exception:
        return
    disk_by = {r.get("display_name"): r for r in (disk.get("selected") or []) if isinstance(r, dict)}
    for row in weekly.get("selected") or []:
        name = row.get("display_name")
        d = disk_by.get(name)
        if not d:
            continue
        if d.get("formula_source") == "hand_dsl_python9" or (
            str(d.get("hand_dsl_at") or "") and row.get("fe_retest_status") != "ok"
        ):
            for k in _PRESERVE_FROM_DISK:
                if k in d and d.get(k) not in (None, ""):
                    row[k] = d[k]
    if disk.get("python9_hand_dsl_at"):
        weekly["python9_hand_dsl_at"] = disk["python9_hand_dsl_at"]
        weekly["python9_hand_dsl_n"] = disk.get("python9_hand_dsl_n")


def _strip_outer_rank(expr: str) -> str:
    """Backward-compatible alias; prefer prep_fe_dsl (keeps leading minus)."""
    return prep_fe_dsl(expr)


def _bake_formula_negation(row: dict[str, Any]) -> str:
    """Permanently wrap show/FE formulas with -(...); return FE dsl to rematerialize."""
    show = str(row.get("lqtp_formula") or row.get("dsl") or "").strip()
    if show and not show.startswith("(python)") and not already_outer_negated(show):
        row["dsl_before_flip"] = show
        negated = _negate_formula(show)
        row["dsl"] = negated
        row["lqtp_formula"] = negated
    fe = str(row.get("fe_dsl") or prep_fe_dsl(str(row.get("dsl") or ""))).strip()
    if fe and not already_outer_negated(fe) and not fe.startswith("-"):
        row["fe_dsl_before_flip"] = fe
        fe = _negate_formula(fe)
        row["fe_dsl"] = fe
    elif row.get("lqtp_formula"):
        fe = prep_fe_dsl(str(row["lqtp_formula"]))
        row["fe_dsl"] = fe
    py = str(row.get("python_code") or "")
    if py.strip() and "IC_SIGN_FLIPPED" not in py:
        row["python_code_before_flip"] = py
        row["python_code"] = _negate_python_code(py)
    row["formula_sign_flipped"] = True
    row["ic_sign_flipped"] = True
    row["sign_flipped"] = False
    row["formula_flipped_at"] = datetime.now().isoformat()
    return prep_fe_dsl(str(row.get("fe_dsl") or row.get("dsl") or ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, default=WORK)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-avail-gb", type=float, default=10.0)
    args = ap.parse_args()

    os.environ.setdefault("FACTOR_ENGINE_MAX_MEMORY_MB", "12000")
    os.environ.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    os.environ.setdefault("FACTOR_ENGINE_OPERATOR_BACKEND", "polars")

    work = args.work_dir
    wp = work / "reports/weekly_dug_neutral_rankic.json"
    weekly = json.loads(wp.read_text(encoding="utf-8"))
    fwd = work / "lqtp_fwd_vwap_returns_cache.parquet"
    out_dir = work / "reports_weekly_dug_wnew"
    log = work / "reports/weekly_fe_retest_recovered.log"

    rows = [r for r in weekly.get("selected") or [] if (r.get("fe_dsl") or r.get("dsl"))]
    if not args.only:
        rows = [r for r in rows if r.get("lqtp_submit_eligible")]
    if args.only:
        only = set(args.only)
        rows = [r for r in rows if r.get("display_name") in only]
    if args.limit:
        rows = rows[: args.limit]

    print(f"retest n={len(rows)} avail={read_mem_available_gb():.1f}G", flush=True)
    ok = 0
    for i, row in enumerate(rows, 1):
        name = str(row.get("display_name") or row.get("factor_id"))
        dsl = prep_fe_dsl(str(row.get("fe_dsl") or row.get("dsl") or ""))
        dsl = re.sub(r"\blogclose\b", "log(close)", dsl, flags=re.I)
        dsl = re.sub(r"\bpre_close\b", "delay(close,1)", dsl, flags=re.I)
        dsl = re.sub(r"(?<![\w.])ret(?![\w.])", "(close/delay(close,1)-1)", dsl)
        if not dsl or dsl.startswith("(python)") or "encoding" in dsl:
            print(f"[{i}/{len(rows)}] SKIP_BAD {name}", flush=True)
            continue
        if _has_cs_rank(dsl):
            print(f"[{i}/{len(rows)}] SKIP_CS_RANK {name}", flush=True)
            row["fe_retest_status"] = "skipped_cs_rank"
            continue
        avail = read_mem_available_gb()
        if avail < args.min_avail_gb:
            print(f"[{i}/{len(rows)}] STOP_MEM avail={avail:.1f}G", flush=True)
            break
        print(f"[{i}/{len(rows)}] FE {name} dsl={dsl[:80]} avail={avail:.1f}G", flush=True)
        t0 = time.time()
        try:
            # prefer pandas for sigmoid/tanh-heavy; polars default otherwise
            if any(op in dsl for op in ("sigmoid", "tanh")):
                os.environ["FACTOR_ENGINE_OPERATOR_BACKEND"] = "pandas"
            else:
                os.environ["FACTOR_ENGINE_OPERATOR_BACKEND"] = "polars"
            lake = materialize_dsl_by_halfyear(work, name, dsl, min_avail_gb=args.min_avail_gb)
            analysis = analyze_factor_parquet_duckdb(
                factor_path=lake, fwd_returns_path=fwd, return_kind="vwap_to_vwap"
            )
            mean_ic = float(analysis.get("mean_rank_ic", float("nan")))
            # Auto direction fix: bake '-' into formula (not display-only lake flip)
            if mean_ic == mean_ic and mean_ic < 0:
                print(f"  AUTO_FLIP_FORMULA mean_ic={mean_ic:.4f}<0", flush=True)
                dsl = _bake_formula_negation(row)
                dsl = re.sub(r"\blogclose\b", "log(close)", dsl, flags=re.I)
                wipe_lake(work, name)
                lake = materialize_dsl_by_halfyear(work, name, dsl, min_avail_gb=args.min_avail_gb)
                release_memory(analysis)
                analysis = analyze_factor_parquet_duckdb(
                    factor_path=lake, fwd_returns_path=fwd, return_kind="vwap_to_vwap"
                )
                mean_ic = float(analysis.get("mean_rank_ic", float("nan")))
            # Full report pack: RankIC + 分层 + 多空 + 扩展评估（滚动IC/中性/半衰期等）
            try:
                ind = work / "eval_cache/industry_sw_l1.parquet"
                mcap = work / "eval_cache/market_cap.parquet"
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
            mean_ic = float(analysis.get("mean_rank_ic", float("nan")))
            # After formula bake-in, RankIC should already be ≥0; do not abs-mask sign.
            display_ic = mean_ic if mean_ic == mean_ic else float("nan")
            if display_ic == display_ic and display_ic < 0:
                # last-resort lake flip + formula bake if rematerialize still negative
                print(f"  FALLBACK_FLIP_LAKE mean_ic={display_ic:.4f}", flush=True)
                _flip_lake(lake)
                if not row.get("formula_sign_flipped"):
                    _bake_formula_negation(row)
                release_memory(analysis)
                analysis = analyze_factor_parquet_duckdb(
                    factor_path=lake, fwd_returns_path=fwd, return_kind="vwap_to_vwap"
                )
                mean_ic = float(analysis.get("mean_rank_ic", float("nan")))
                display_ic = mean_ic if mean_ic == mean_ic else float("nan")
            meta = {
                "engine": "factor_engine",
                "eval_route": "fe_data_access_yearly_retest",
                "values_path": str(lake),
                "date_range": ["2019-01-01", "2026-06-30"],
                "formula": str(row.get("lqtp_formula") or row.get("dsl") or dsl),
                "eval_engine": "duckdb_panel",
                "return_kind": "vwap_to_vwap",
                "source_label": "本周新挖·公式重测",
                "ic_sign_flipped": bool(row.get("formula_sign_flipped") or row.get("ic_sign_flipped")),
                "platform_submit": row.get("platform_submit"),
                "data_source": "data_access",
                "note": "非 COS 中性化面板；公式 FE 重算；负向已写入公式取负",
                "extended_eval": True,
            }
            out = out_dir / f"{name}.html"
            render_factor_report(
                factor_name=name,
                dsl=str(row.get("lqtp_formula") or row.get("dsl") or dsl),
                python_code=str(row.get("python_code") or ""),
                analysis=analysis,
                backtest_rows=None,
                out_path=out,
                eval_mode="duckdb_panel_vwap_to_vwap_t1t2_realto",
                materialize_meta=meta,
                engine="factor_engine",
                eval_route="fe_data_access_yearly_retest",
                work_dir=work,
                annotation={
                    "factor_id": name,
                    "function_name": name,
                    "source": "本周新挖",
                    "note": "公式重测（FE+data_access）；负向因子已在公式内取负并调正",
                    "theme": "量价因子",
                },
            )
            text = out.read_text(encoding="utf-8", errors="ignore")
            for brand in ("factorminer", "FactorMiner", "cogalpha", "CogAlpha", "hsunbj", "quantaalpha"):
                text = text.replace(brand, "cand")
            out.write_text(text, encoding="utf-8")

            old_ic = float(row.get("display_rank_ic") or 0)
            row.update(
                {
                    "fe_retest_status": "ok",
                    "fe_retest_rank_ic": mean_ic,
                    "fe_retest_display_rank_ic": display_ic,
                    "fe_retest_rank_icir": float(analysis.get("rank_icir", float("nan"))),
                    "fe_retest_long_short_sharpe": float(analysis.get("long_short_sharpe", float("nan"))),
                    "fe_retest_coverage": float(analysis.get("mean_daily_coverage", float("nan"))),
                    "fe_retest_sec": round(time.time() - t0, 1),
                    "fe_retest_lake": str(lake),
                    "materialize": "fe_data_access_yearly_retest",
                    "display_rank_ic": display_ic,
                    "display_rank_icir": abs(float(analysis.get("rank_icir") or 0.0)),
                    "mean_rank_ic": mean_ic,
                    "abs_mean_rank_ic": abs(display_ic) if display_ic == display_ic else float("nan"),
                    "rank_icir": float(analysis.get("rank_icir", float("nan"))),
                    "long_short_sharpe": float(analysis.get("long_short_sharpe", float("nan"))),
                    "mean_daily_coverage": float(analysis.get("mean_daily_coverage", float("nan"))),
                    "panel_rank_ic_before_retest": old_ic,
                    "lake_path": str(lake),
                    "report_href": f"/reports_weekly_dug_wnew/{name}.html",
                    "sign_flipped": False,
                }
            )
            ok += 1
            msg = (
                f"  OK {name} panelIC={old_ic:.4f} -> feIC={display_ic:.4f} "
                f"sh={row['long_short_sharpe']:.3f} sec={row['fe_retest_sec']} avail={read_mem_available_gb():.1f}G"
            )
            print(msg, flush=True)
            with log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
            release_memory(analysis)
            gc.collect()
            # persist incrementally with flock (safe under parallel workers)
            _persist_row_locked(wp, name, row, weekly_meta={"fe_retest_at": datetime.now().isoformat()})
            # keep in-memory weekly roughly in sync for end summary
            for rr in weekly.get("selected") or []:
                if rr.get("display_name") == name:
                    rr.update(row)
                    break
        except Exception as e:  # noqa: BLE001
            msg = f"  ERR {name} {e}"
            print(msg, flush=True)
            with log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
            row["fe_retest_status"] = "error"
            row["fe_retest_error"] = str(e)[:400]
            _persist_row_locked(wp, name, {"fe_retest_status": "error", "fe_retest_error": str(e)[:400]})
            gc.collect()

    # final merge/sort under lock
    _finalize_weekly_locked(wp, ok_note=ok)
    print(f"DONE ok={ok}/{len(rows)} avail={read_mem_available_gb():.1f}G", flush=True)
    return 0


def _persist_row_locked(wp: Path, name: str, patch: dict[str, Any], weekly_meta: dict[str, Any] | None = None) -> None:
    import fcntl

    wp.parent.mkdir(parents=True, exist_ok=True)
    lock_path = wp.with_suffix(wp.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            weekly = json.loads(wp.read_text(encoding="utf-8"))
            found = False
            for r in weekly.get("selected") or []:
                if r.get("display_name") == name:
                    r.update(patch)
                    found = True
                    break
            if not found:
                weekly.setdefault("selected", []).append({"display_name": name, **patch})
            if weekly_meta:
                weekly.update(weekly_meta)
            weekly["selected"].sort(key=lambda r: float(r.get("display_rank_ic") or 0), reverse=True)
            wp.write_text(json.dumps(weekly, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def _finalize_weekly_locked(wp: Path, ok_note: int) -> None:
    import fcntl

    lock_path = wp.with_suffix(wp.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            weekly = json.loads(wp.read_text(encoding="utf-8"))
            weekly["selected"].sort(key=lambda r: float(r.get("display_rank_ic") or 0), reverse=True)
            weekly["fe_retest_at"] = datetime.now().isoformat()
            weekly["fe_retest_ok"] = int(ok_note)
            weekly["platform_note"] = (
                f"本周 {weekly.get('n_selected')} 条已回补公式；FE 公式重测完成（含公式取负调正）。"
            )
            wp.write_text(json.dumps(weekly, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    raise SystemExit(main())
