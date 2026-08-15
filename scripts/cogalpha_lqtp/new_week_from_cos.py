#!/usr/bin/env python3
"""Pull a fresh weekly factor set from COS, backtest locally, patch screening HTML.

Sources (brand-scrubbed in UI as 本周新挖 / 外部中性化候选):
  - cos://qs-cold/candidate_pool/factorminer_qa_rankic_0.02_neu/*.parquet  (long)
  - cos://qs-cold/candidate_pool/cogalpha_factors_neutral/*.parquet         (wide)
  - unused alphasage_* under alphasage_factors_neu/ (yearly wide)

Hard constraints:
  - exclude any hash/name already in current HTML / previous weekly JSON
  - sequential eval only; abort if MemAvailable < MIN_AVAIL_GB
  - no FactorEngine materialize (panel-only DuckDB RankIC)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "cogalpha_lqtp")]

from scripts.cogalpha_lqtp.crawl_candidate_pool_neutral_rankic import (  # noqa: E402
    _download_year,
    _list_years,
    _panel_qc,
    _weekly_section_html,
)
from scripts.cogalpha_lqtp.eval_lake_fast import analyze_factor_parquet_duckdb  # noqa: E402
from scripts.cogalpha_lqtp.report_html import render_factor_report  # noqa: E402

WORK = ROOT / "data/cogalpha_lqtp_production"
COS_CLI = os.environ.get("CANDIDATE_COS_CLI", "candidate-cos-ro")
THRESHOLD = 0.02
MIN_AVAIL_GB = 12.0  # keep headroom under 15G process budget on 30G host
MAX_RSS_GB = 15.0

FM_PREFIX = "cos://qs-cold/candidate_pool/factorminer_qa_rankic_0.02_neu/"
CG_PREFIX = "cos://qs-cold/candidate_pool/cogalpha_factors_neutral/"
AS_PREFIX = "cos://qs-cold/candidate_pool/alphasage_factors_neu/"


def _avail_gb() -> float:
    for line in open("/proc/meminfo", encoding="utf-8"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


def _rss_gb() -> float:
    # self RSS
    with open("/proc/self/status", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def _guard(tag: str = "") -> None:
    avail = _avail_gb()
    rss = _rss_gb()
    if avail < MIN_AVAIL_GB or rss > MAX_RSS_GB:
        raise MemoryError(f"mem guard {tag}: avail={avail:.1f}G rss={rss:.1f}G")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, text=True, capture_output=True)


def _cos_ls(prefix: str) -> str:
    return _run([COS_CLI, "ls", prefix]).stdout


def _cos_cp(uri: str, dest: Path) -> Path | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()
    proc = _run([COS_CLI, "cp", uri, str(tmp)])
    if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size <= 0:
        if tmp.exists():
            tmp.unlink()
        return None
    tmp.replace(dest)
    return dest


def _existing_hashes(work: Path) -> set[str]:
    have: set[str] = set()
    paths = [
        work / "reports/weekly_dug_neutral_rankic.json",
        work / "reports/weekly_dug_neutral_rankic_prev.json",
    ]
    for p in paths:
        if not p.exists():
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for r in payload.get("selected") or []:
            for k in ("display_name", "factor_id", "candidate_id"):
                for m in re.finditer(r"([0-9a-f]{8})", str(r.get(k) or ""), re.I):
                    have.add(m.group(1).lower())
    idx = work / "reports/factor_rankic_screening_index.html"
    if idx.exists():
        text = idx.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"(?:alpha|ext|alphasage|cand)_([0-9a-f]{8})", text, re.I):
            have.add(m.group(1).lower())
        for m in re.finditer(r"([0-9a-f]{16})", text, re.I):
            have.add(m.group(1)[:8].lower())
            have.add(m.group(1)[-8:].lower())
    return have


def _normalize_long(src: Path, dest: Path) -> Path:
    """datetime/asset/factor_value → trade_date/symbol/value."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{src.as_posix()}')").fetchall()}
        if {"trade_date", "symbol", "value"} <= cols:
            con.execute(
                f"COPY (SELECT trade_date::INTEGER AS trade_date, symbol::VARCHAR AS symbol, "
                f"value::DOUBLE AS value FROM read_parquet('{src.as_posix()}') WHERE value IS NOT NULL) "
                f"TO '{dest.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
        elif {"datetime", "asset"} <= cols:
            val = "factor_value" if "factor_value" in cols else "value"
            con.execute(
                f"""
                COPY (
                  SELECT (year(datetime)*10000+month(datetime)*100+day(datetime))::INTEGER AS trade_date,
                         asset::VARCHAR AS symbol,
                         {val}::DOUBLE AS value
                  FROM read_parquet('{src.as_posix()}')
                  WHERE {val} IS NOT NULL
                ) TO '{dest.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        else:
            raise ValueError(f"unknown long schema {sorted(cols)[:20]}")
    finally:
        con.close()
    return dest


def _wide_to_long_lake(src: Path, dest: Path, *, chunk: int = 400) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        cols = [
            r[0]
            for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{src.as_posix()}')").fetchall()
            if r[0] != "date"
        ]
        parts: list[str] = []
        tmpdir = dest.parent / f"_parts_{dest.stem}"
        if tmpdir.exists():
            shutil.rmtree(tmpdir)
        tmpdir.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(cols), chunk):
            _guard(f"unpivot:{dest.stem}:{i}")
            inns = ",".join(f'"{c}"' for c in cols[i : i + chunk])
            part = tmpdir / f"p{i:05d}.parquet"
            con.execute(
                f"""
                COPY (
                  SELECT (year(CAST(date AS DATE))*10000+month(CAST(date AS DATE))*100
                          +day(CAST(date AS DATE)))::INTEGER AS trade_date,
                         symbol::VARCHAR AS symbol,
                         value::DOUBLE AS value
                  FROM read_parquet('{src.as_posix()}')
                  UNPIVOT (value FOR symbol IN ({inns}))
                  WHERE value IS NOT NULL
                ) TO '{part.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            parts.append(part.as_posix())
        con.execute(
            f"COPY (SELECT * FROM read_parquet({parts})) TO '{dest.as_posix()}' "
            f"(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        shutil.rmtree(tmpdir, ignore_errors=True)
    finally:
        con.close()
    return dest


def _years_wide_to_long(paths: list[Path], dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    tmpdir = dest.parent / f"_yparts_{dest.stem}"
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        _guard(f"year:{path}")
        part = tmpdir / f"{path.stem}.parquet"
        _wide_to_long_lake(path, part)
        parts.append(part.as_posix())
    con = duckdb.connect()
    try:
        con.execute(
            f"COPY (SELECT * FROM read_parquet({parts})) TO '{dest.as_posix()}' "
            f"(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        con.close()
    shutil.rmtree(tmpdir, ignore_errors=True)
    return dest


def _eval_lake(lake: Path, fwd: Path) -> dict[str, Any]:
    _guard("pre-eval")
    analysis = analyze_factor_parquet_duckdb(factor_path=lake, fwd_returns_path=fwd)
    mean_ic = float(analysis.get("mean_rank_ic", float("nan")))
    out = {
        "mean_rank_ic": mean_ic,
        "abs_mean_rank_ic": abs(mean_ic) if mean_ic == mean_ic else float("nan"),
        "rank_icir": float(analysis.get("rank_icir", float("nan"))),
        "mean_daily_coverage": float(analysis.get("mean_daily_coverage", float("nan"))),
        "long_short_sharpe": float(analysis.get("long_short_sharpe", float("nan"))),
        "ls_mean_one_way_turnover": float(analysis.get("ls_mean_one_way_turnover", float("nan"))),
        "analysis": analysis,
    }
    return out


def _flip_if_needed(lake: Path, fwd: Path, row: dict[str, Any]) -> dict[str, Any]:
    mean_ic = float(row["mean_rank_ic"])
    if not (mean_ic == mean_ic and mean_ic < 0):
        row["sign_flipped"] = False
        return row
    tmp = lake.with_suffix(".flip.parquet")
    con = duckdb.connect()
    try:
        con.execute(
            f"""
            COPY (
              SELECT trade_date, symbol, -value AS value
              FROM read_parquet('{lake.as_posix()}')
            ) TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        con.close()
    tmp.replace(lake)
    analysis = analyze_factor_parquet_duckdb(factor_path=lake, fwd_returns_path=fwd)
    mean_ic = float(analysis.get("mean_rank_ic", float("nan")))
    row.update(
        {
            "mean_rank_ic": mean_ic,
            "abs_mean_rank_ic": abs(mean_ic) if mean_ic == mean_ic else float("nan"),
            "rank_icir": float(analysis.get("rank_icir", float("nan"))),
            "mean_daily_coverage": float(analysis.get("mean_daily_coverage", float("nan"))),
            "long_short_sharpe": float(analysis.get("long_short_sharpe", float("nan"))),
            "ls_mean_one_way_turnover": float(analysis.get("ls_mean_one_way_turnover", float("nan"))),
            "analysis": analysis,
            "sign_flipped": True,
        }
    )
    return row


def _render(work: Path, name: str, label: str, lake: Path, row: dict[str, Any]) -> str:
    report_dir = work / "reports_weekly_dug_wnew"
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / f"{name}.html"
    meta = {
        "engine": "local_panel",
        "eval_route": "local_panel_values",
        "values_path": str(lake),
        "date_range": ["2016-01-01", "2026-06-30"],
        "formula": "",
        "eval_engine": "duckdb_panel",
        "return_kind": "vwap_to_vwap",
        "source_label": label,
        "ic_sign_flipped": bool(row.get("sign_flipped")),
        "platform_submit": "skipped_no_formula",
    }
    render_factor_report(
        factor_name=name,
        dsl="",
        python_code="",
        analysis=row["analysis"],
        backtest_rows=None,
        out_path=out,
        eval_mode="duckdb_panel_vwap_to_vwap_t1t2_realto",
        materialize_meta=meta,
        engine="local_panel",
        eval_route="local_panel_values",
        work_dir=work,
        annotation={
            "factor_id": name,
            "function_name": name,
            "source": label,
            "note": "本周新挖·COS中性化面板回测（无公式，未提交平台）",
        },
    )
    # light brand scrub on display text only
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
        "lizhuo",
    ):
        text = text.replace(brand, "cand")
    out.write_text(text, encoding="utf-8")
    return f"reports_weekly_dug_wnew/{name}.html"


def _list_candidates(have: set[str]) -> list[dict[str, Any]]:
    cands: list[dict[str, Any]] = []

    # factorminer qa long
    for line in _cos_ls(FM_PREFIX).splitlines():
        m = re.search(r"([0-9a-f]{16})_neu\.parquet", line)
        if not m:
            continue
        fid16 = m.group(1).lower()
        if fid16[:8] in have or fid16[-8:] in have:
            continue
        cands.append(
            {
                "kind": "fm_long",
                "cid": fid16,
                "name": f"ext_{fid16[:8]}",
                "uri": f"{FM_PREFIX}{fid16}_neu.parquet",
                "label": "本周新挖",
                "source": "pool_fm_qa",
            }
        )

    # cogalpha named wide
    for line in _cos_ls(CG_PREFIX).splitlines():
        m = re.search(r"(factor_[a-z0-9_]+)_neu\.parquet", line)
        if not m:
            continue
        raw = m.group(1)
        # display without mining brand / factor_ prefix noise
        slug = re.sub(r"^factor_", "", raw)
        slug = re.sub(r"_v\d+$", "", slug)
        name = "cand_" + slug[:48]
        cands.append(
            {
                "kind": "cg_wide",
                "cid": raw,
                "name": name,
                "uri": f"{CG_PREFIX}{raw}_neu.parquet",
                "label": "本周新挖",
                "source": "pool_cg_neu",
            }
        )

    # unused alphasage yearly
    for line in _cos_ls(AS_PREFIX).splitlines():
        m = re.search(r"(alphasage_[0-9a-fA-F_]+)/", line)
        if not m:
            continue
        fid = m.group(1)
        h = re.search(r"([0-9a-f]{8})", fid, re.I)
        if h and h.group(1).lower() in have:
            continue
        name = f"ext_{h.group(1).lower()}" if h else f"ext_{fid[-8:].lower()}"
        cands.append(
            {
                "kind": "as_years",
                "cid": fid,
                "name": name,
                "uri": AS_PREFIX,
                "label": "本周新挖",
                "source": "pool_b",
            }
        )

    # stable order: prefer named / qa first
    order = {"cg_wide": 0, "fm_long": 1, "as_years": 2}
    cands.sort(key=lambda x: (order.get(x["kind"], 9), x["name"]))
    return cands


def _prepare_lake(cand: dict[str, Any], cache: Path) -> Path | None:
    kind = cand["kind"]
    name = cand["name"]
    lake = cache / "lakes" / f"{name}.parquet"
    if lake.exists() and lake.stat().st_size > 0:
        return lake
    if kind == "fm_long":
        raw = cache / "fm_raw" / f"{cand['cid']}_neu.parquet"
        if _cos_cp(cand["uri"], raw) is None:
            return None
        return _normalize_long(raw, lake)
    if kind == "cg_wide":
        raw = cache / "cg_raw" / f"{cand['cid']}_neu.parquet"
        if _cos_cp(cand["uri"], raw) is None:
            return None
        return _wide_to_long_lake(raw, lake)
    if kind == "as_years":
        years = _list_years(AS_PREFIX, cand["cid"])
        years = [y for y in years if y >= 2019] or years
        paths: list[Path] = []
        for y in years:
            dest = cache / "as_raw" / cand["cid"] / f"{y}.parquet"
            p = _download_year(cos_prefix=AS_PREFIX, factor_id=cand["cid"], year=y, dest=dest)
            if p is not None:
                paths.append(p)
        if not paths:
            return None
        qc = _panel_qc(paths)
        if qc.get("degenerate"):
            return None
        return _years_wide_to_long(paths, lake)
    return None


def _archive_prev(work: Path) -> None:
    cur = work / "reports/weekly_dug_neutral_rankic.json"
    prev = work / "reports/weekly_dug_neutral_rankic_prev.json"
    if cur.exists() and not prev.exists():
        shutil.copy2(cur, prev)
        print(f"archived previous weekly -> {prev}", flush=True)
    elif cur.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = work / f"reports/weekly_dug_neutral_rankic_archive_{stamp}.json"
        shutil.copy2(cur, bak)
        print(f"backup current weekly -> {bak}", flush=True)


def _patch_html(work: Path, payload: dict[str, Any], prev: dict[str, Any] | None) -> None:
    html_path = work / "reports/factor_rankic_screening_index.html"
    text = html_path.read_text(encoding="utf-8")
    section = _weekly_section_html(payload, threshold=float(payload.get("threshold", THRESHOLD)))
    # retitle current week
    week_tag = payload.get("week_tag") or datetime.now().strftime("%Y-%m-%d")
    section = section.replace(
        "本周新挖 · 中性化候选",
        f"本周新挖 · {week_tag}",
        1,
    )
    prev_html = ""
    if prev and prev.get("selected"):
        prev_section = _weekly_section_html(prev, threshold=float(prev.get("threshold", THRESHOLD)))
        prev_section = prev_section.replace('id="weekly-dug-neutral"', 'id="weekly-dug-prev"', 1)
        prev_section = prev_section.replace("本周新挖 · 中性化候选", "上周新挖（存档）", 1)
        prev_html = prev_section
    combined = section + "\n" + prev_html
    if re.search(r'<section id="weekly-dug-neutral"[\s\S]*?</section>', text):
        # drop old weekly + any previous prev section, reinsert
        text = re.sub(
            r'<section id="weekly-dug-(?:neutral|prev)"[\s\S]*?</section>\s*',
            "",
            text,
            count=4,
        )
        m = re.search(
            r"(<section class=['\"]week2-section['\"]>|<section class=['\"]corr-embed['\"]>|</main>)",
            text,
        )
        if m:
            text = text[: m.start()] + combined + "\n" + text[m.start() :]
        else:
            text = text.replace("</main>", combined + "\n</main>")
    else:
        text = text.replace("</main>", combined + "\n</main>")
    if 'href="#weekly-dug-neutral"' not in text:
        text = text.replace(
            "<h2>快速目录</h2>",
            '<h2>快速目录</h2>\n    <p class="muted"><a href="#weekly-dug-neutral">↓ 跳到本周新挖</a>'
            ' · <a href="#weekly-dug-prev">上周存档</a></p>',
            1,
        )
    # scrub brands in weekly blocks only is hard; light global display scrub of known brands in visible labels
    for brand in ("factorminer", "FactorMiner", "CogAlpha", "cogalpha", "AlphaSage", "alphasage", "EvoAlpha", "evoalpha"):
        text = text.replace(brand, "cand")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = html_path.with_name(f"{html_path.name}.bak_newweek_{stamp}")
    bak.write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")
    html_path.write_text(text, encoding="utf-8")
    print(f"patched {html_path} (backup {bak.name})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, default=WORK)
    ap.add_argument("--limit", type=int, default=0, help="debug cap on candidates screened")
    ap.add_argument("--max-select", type=int, default=60)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--skip-render", action="store_true")
    args = ap.parse_args()
    work: Path = args.work_dir
    fwd = work / "lqtp_fwd_vwap_returns_cache.parquet"
    if not fwd.is_file():
        raise SystemExit(f"missing fwd cache: {fwd}")

    cache = work / "candidate_pool_neutral_cache" / "week_new_20260816"
    cache.mkdir(parents=True, exist_ok=True)
    reports = work / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    log_path = reports / "new_week_from_cos.log"

    def log(msg: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    _archive_prev(work)
    prev_payload = None
    prev_path = work / "reports/weekly_dug_neutral_rankic_prev.json"
    if prev_path.exists():
        prev_payload = json.loads(prev_path.read_text(encoding="utf-8"))

    have = _existing_hashes(work)
    cands = _list_candidates(have)
    if args.limit:
        cands = cands[: args.limit]
    log(f"candidates={len(cands)} exclude_hashes={len(have)} avail={_avail_gb():.1f}G")

    results: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []

    for i, cand in enumerate(cands, 1):
        try:
            _guard(f"loop:{cand['name']}")
        except MemoryError as e:
            log(f"STOP {e}")
            break
        t0 = time.time()
        log(f"[{i}/{len(cands)}] prepare {cand['name']} kind={cand['kind']} avail={_avail_gb():.1f}G")
        try:
            lake = _prepare_lake(cand, cache)
            if lake is None:
                results.append({**cand, "ok": False, "error": "prepare_failed"})
                continue
            row = _eval_lake(lake, fwd)
            row = _flip_if_needed(lake, fwd, row)
            analysis = row.pop("analysis")
            item = {
                "factor_id": cand["name"],
                "display_name": cand["name"],
                "candidate_id": cand["cid"],
                "source": cand["source"],
                "label": cand["label"],
                "source_label": cand["label"],
                "ok": True,
                "ok_for_select": True,
                "already_neutralized": True,
                "neutralization_note": "已中性化（COS池）",
                "has_formula": False,
                "platform_submit": "skipped_no_formula",
                "materialize": "cos_panel",
                "mean_rank_ic": row["mean_rank_ic"],
                "abs_mean_rank_ic": row["abs_mean_rank_ic"],
                "display_rank_ic": row["abs_mean_rank_ic"],
                "rank_icir": row["rank_icir"],
                "display_rank_icir": abs(float(row["rank_icir"] or 0.0)),
                "mean_daily_coverage": row["mean_daily_coverage"],
                "long_short_sharpe": row["long_short_sharpe"],
                "ls_mean_one_way_turnover": row["ls_mean_one_way_turnover"],
                "sign_flipped": bool(row.get("sign_flipped")),
                "lake_path": str(lake),
                "elapsed_sec": round(time.time() - t0, 2),
            }
            pass_ic = (
                item["abs_mean_rank_ic"] == item["abs_mean_rank_ic"]
                and float(item["abs_mean_rank_ic"]) > float(args.threshold)
            )
            item["ok_for_select"] = bool(pass_ic)
            if pass_ic and not args.skip_render:
                item["analysis_tmp"] = analysis  # hold briefly
                href = _render(work, cand["name"], cand["label"], lake, {**row, "analysis": analysis})
                item["report_href"] = href
                del item["analysis_tmp"]
            results.append(item)
            if pass_ic:
                selected.append(item)
            log(
                f"  ok |IC|={item['abs_mean_rank_ic']:.4f} sh={item['long_short_sharpe']:.3f} "
                f"pass={pass_ic} sec={item['elapsed_sec']} avail={_avail_gb():.1f}G rss={_rss_gb():.2f}G"
            )
            del analysis
            gc.collect()
        except MemoryError as e:
            log(f"  MEM {e}")
            results.append({**cand, "ok": False, "error": str(e)})
            gc.collect()
            break
        except Exception as e:  # noqa: BLE001
            log(f"  ERR {e}")
            results.append({**cand, "ok": False, "error": str(e)[:300]})
            gc.collect()

    selected.sort(
        key=lambda r: (
            float(r.get("display_rank_ic") or 0),
            float(r.get("long_short_sharpe") or 0),
        ),
        reverse=True,
    )
    if args.max_select and len(selected) > args.max_select:
        selected = selected[: args.max_select]

    # drop heavy keys
    for r in selected:
        r.pop("analysis", None)
        r.pop("analysis_tmp", None)

    week_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    payload = {
        "generated_at": datetime.now().isoformat(),
        "week_tag": week_tag,
        "threshold": float(args.threshold),
        "fwd_returns": str(fwd),
        "n_crawled": len(results),
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_selected": len(selected),
        "platform_note": "COS 中性化面板本地 DuckDB 回测（VWAP→VWAP）；无公式不提交平台。内存硬顶约 15G。",
        "pools": [
            {"source": "pool_fm_qa", "label": "本周新挖", "prefix": FM_PREFIX},
            {"source": "pool_cg_neu", "label": "本周新挖", "prefix": CG_PREFIX},
            {"source": "pool_b", "label": "本周新挖", "prefix": AS_PREFIX},
        ],
        "neutralization_summary": {
            "n_probed": 0,
            "n_looks_cs_neutral": 0,
            "n_differs_from_raw": 0,
            "note": "面板取自 COS *_neu / factors_neutral 路径",
        },
        "selected": selected,
        "all_results": [
            {k: v for k, v in r.items() if k not in {"analysis", "analysis_tmp"}} for r in results
        ],
    }
    out_json = reports / "weekly_dug_neutral_rankic.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports / "new_week_from_cos_results.json").write_text(
        json.dumps(
            {
                "n_crawled": payload["n_crawled"],
                "n_ok": payload["n_ok"],
                "n_selected": payload["n_selected"],
                "selected_names": [r["display_name"] for r in selected],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"wrote {out_json} selected={len(selected)}/{payload['n_ok']}")
    _patch_html(work, payload, prev_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
