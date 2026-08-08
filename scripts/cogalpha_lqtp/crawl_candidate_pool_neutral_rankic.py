#!/usr/bin/env python3
"""Crawl COS neutral candidate_pool dumps, verify neutralization, compute RankIC.

Sources:
  cos://qs-cold/candidate_pool/lizhuo_factors_neutral/   (Evoalpha / lizhuo)
  cos://qs-cold/candidate_pool/alphasage_factors_neu/    (AlphaSage)

Writes:
  work-dir/reports/weekly_dug_neutral_rankic.json
  work-dir/reports/weekly_dug_neutral_rankic.csv
and can patch the RankIC screening HTML with a dedicated weekly section.
"""
from __future__ import annotations

import argparse
import csv
import html as html_lib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cogalpha_lqtp.eval_lake_fast import analyze_factor_long_df_duckdb  # noqa: E402

DEFAULT_WORK = ROOT / "data/cogalpha_lqtp_production"
COS_CLI = os.environ.get("CANDIDATE_COS_CLI", "candidate-cos-ro")
POOLS = (
    {
        "source": "pool_a",
        "label": "外部中性化候选 A",
        "prefix": "cos://qs-cold/candidate_pool/lizhuo_factors_neutral/",
        "raw_prefix": "cos://qs-cold/candidate_pool/lizhuo_factors/",
        "id_prefix": "alpha_",
        # legacy source key kept for existing local cache dirs
        "cache_source": "evoalpha",
    },
    {
        "source": "pool_b",
        "label": "外部中性化候选 B",
        "prefix": "cos://qs-cold/candidate_pool/alphasage_factors_neu/",
        "raw_prefix": "cos://qs-cold/candidate_pool/alphasage_factors/",
        "id_prefix": "alphasage_",
        "cache_source": "alphasage",
    },
)
YEARS_DEFAULT = list(range(2019, 2027))
THRESHOLD_DEFAULT = 0.02


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def _list_factor_ids(prefix: str) -> list[str]:
    proc = _run([COS_CLI, "ls", prefix])
    ids: list[str] = []
    for line in proc.stdout.splitlines():
        m = re.search(r"(?:^|\s)((?:alpha_|alphasage_)[0-9a-fA-F_]+)/?\s*\|", line)
        if not m:
            # fallback: path tail
            m2 = re.search(r"candidate_pool/[^/]+/((?:alpha_|alphasage_)[^/\s|]+)/", line)
            if not m2:
                continue
            fid = m2.group(1).rstrip("/")
        else:
            fid = m.group(1).rstrip("/")
        if fid not in ids:
            ids.append(fid)
    return sorted(ids)


def _list_years(prefix: str, factor_id: str) -> list[int]:
    proc = _run([COS_CLI, "ls", f"{prefix}{factor_id}/"], check=False)
    years: list[int] = []
    for line in proc.stdout.splitlines():
        m = re.search(rf"{re.escape(factor_id)}/((?:19|20)\d{{2}})\.parquet", line)
        if m:
            years.append(int(m.group(1)))
    return sorted(set(years))


def _download_year(
    *,
    cos_prefix: str,
    factor_id: str,
    year: int,
    dest: Path,
) -> Path | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    uri = f"{cos_prefix}{factor_id}/{year}.parquet"
    tmp = dest.with_suffix(".parquet.partial")
    if tmp.exists():
        tmp.unlink()
    proc = _run([COS_CLI, "cp", uri, str(tmp)], check=False)
    if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size <= 0:
        if tmp.exists():
            tmp.unlink()
        return None
    tmp.replace(dest)
    return dest


def _wide_to_long(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pq.read_table(path).to_pandas(date_as_object=True, ignore_metadata=True)
        if "date" not in df.columns:
            # pandas may have used date as index in some converters; re-read via arrow
            table = pq.read_table(path)
            names = table.column_names
            if "date" not in names:
                raise ValueError(f"no date column in {path}")
            df = table.to_pandas(date_as_object=True, ignore_metadata=True)
        long = df.melt(id_vars=["date"], var_name="symbol", value_name="value")
        long = long.dropna(subset=["value"])
        long["trade_date"] = pd.to_datetime(long["date"]).dt.strftime("%Y%m%d").astype(int)
        frames.append(long[["trade_date", "symbol", "value"]])
    if not frames:
        return pd.DataFrame(columns=["trade_date", "symbol", "value"])
    out = pd.concat(frames, ignore_index=True)
    out["symbol"] = out["symbol"].astype(str)
    out["value"] = out["value"].astype(np.float64)
    return out


def _panel_qc(paths: list[Path]) -> dict[str, Any]:
    """Reject all-zero / near-constant panels that can fabricate RankIC via ties."""
    if not paths:
        return {"degenerate": True, "reason": "no_files"}
    path = paths[-1]
    df = pq.read_table(path).to_pandas(date_as_object=True, ignore_metadata=True)
    if "date" in df.columns:
        df = df.set_index("date")
    arr = df.to_numpy(dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        std = float(np.nanstd(arr))
        absmean = float(np.nanmean(np.abs(arr)))
    uniq_ratios: list[float] = []
    for i in range(min(arr.shape[0], 40)):
        row = arr[i]
        m = np.isfinite(row)
        if int(m.sum()) < 30:
            continue
        uniq_ratios.append(len(np.unique(np.round(row[m], 12))) / float(m.sum()))
    med_uniq = float(np.median(uniq_ratios)) if uniq_ratios else 0.0
    degenerate = (not np.isfinite(std)) or std < 1e-12 or absmean < 1e-12 or med_uniq < 0.01
    return {
        "year": int(path.stem),
        "std": std if np.isfinite(std) else None,
        "absmean": absmean if np.isfinite(absmean) else None,
        "median_unique_ratio": med_uniq,
        "degenerate": bool(degenerate),
        "reason": "constant_or_zero" if degenerate else "ok",
    }


def _neutralization_probe(neu_path: Path, raw_path: Path | None) -> dict[str, Any]:
    neu = pq.read_table(neu_path).to_pandas(date_as_object=True, ignore_metadata=True)
    neu = neu.set_index("date")
    cs_mean_abs = float(np.nanmedian(np.abs(neu.mean(axis=1))))
    cs_scale = float(np.nanmedian(np.nanmean(np.abs(neu.to_numpy(dtype=float)), axis=1)))
    rel = cs_mean_abs / max(cs_scale, 1e-12)
    out: dict[str, Any] = {
        "neu_cs_mean_abs_median": cs_mean_abs,
        "neu_cs_abs_scale_median": cs_scale,
        "neu_cs_mean_rel": rel,
        "looks_cs_neutral": bool(rel < 0.05),
    }
    if raw_path is None or not raw_path.exists():
        out["raw_compared"] = False
        return out
    raw = pq.read_table(raw_path).to_pandas(date_as_object=True, ignore_metadata=True).set_index("date")
    syms = sorted(set(neu.columns) & set(raw.columns))
    if not syms:
        out["raw_compared"] = False
        return out
    common = neu.index.intersection(raw.index)
    if len(common) == 0:
        out["raw_compared"] = False
        return out
    a = neu.loc[common, syms]
    b = raw.loc[common, syms]
    out["raw_compared"] = True
    out["raw_cs_mean_abs_median"] = float(np.nanmedian(np.abs(b.mean(axis=1))))
    out["raw_neu_median_daily_corr"] = float(a.corrwith(b, axis=1).median())
    out["identical_cell_frac"] = float(
        ((((a - b).abs() < 1e-9) | (a.isna() & b.isna())).to_numpy().mean())
    )
    out["differs_from_raw"] = bool(out["identical_cell_frac"] < 0.99)
    return out


def _eval_factor(
    *,
    factor_id: str,
    source: str,
    label: str,
    paths: list[Path],
    fwd: Path,
) -> dict[str, Any]:
    long = _wide_to_long(paths)
    if long.empty:
        return {
            "factor_id": factor_id,
            "source": source,
            "label": label,
            "ok": False,
            "error": "empty_panel",
        }
    qc = _panel_qc(paths)
    if qc.get("degenerate"):
        return {
            "factor_id": factor_id,
            "source": source,
            "label": label,
            "ok": True,
            "ok_for_select": False,
            "panel_qc": qc,
            "mean_rank_ic": float("nan"),
            "abs_mean_rank_ic": float("nan"),
            "error": "degenerate_panel",
            "n_rows": int(len(long)),
            "n_days": int(long["trade_date"].nunique()),
        }
    analysis = analyze_factor_long_df_duckdb(long, fwd_returns_path=fwd)
    mean_ic = float(analysis.get("mean_rank_ic", float("nan")))
    return {
        "factor_id": factor_id,
        "source": source,
        "label": label,
        "ok": True,
        "ok_for_select": True,
        "panel_qc": qc,
        "mean_rank_ic": mean_ic,
        "abs_mean_rank_ic": abs(mean_ic) if mean_ic == mean_ic else float("nan"),
        "rank_icir": float(analysis.get("rank_icir", float("nan"))),
        "mean_daily_coverage": float(analysis.get("mean_daily_coverage", float("nan"))),
        "ls_mean_one_way_turnover": float(analysis.get("ls_mean_one_way_turnover", float("nan"))),
        "long_short_sharpe": float(analysis.get("long_short_sharpe", float("nan"))),
        "n_rows": int(len(long)),
        "n_days": int(long["trade_date"].nunique()),
        "years": sorted({int(str(d)[:4]) for d in long["trade_date"].unique()}),
    }


def _eval_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Process-pool entrypoint."""
    t0 = time.time()
    try:
        row = _eval_factor(
            factor_id=payload["factor_id"],
            source=payload["source"],
            label=payload["label"],
            paths=[Path(p) for p in payload["paths"]],
            fwd=Path(payload["fwd"]),
        )
    except Exception as exc:  # noqa: BLE001
        row = {
            "factor_id": payload["factor_id"],
            "source": payload["source"],
            "label": payload["label"],
            "ok": False,
            "error": str(exc),
        }
    row["elapsed_sec"] = round(time.time() - t0, 2)
    if payload.get("neutral_probe") is not None:
        row["neutral_probe"] = payload["neutral_probe"]
    return row


def crawl_and_eval(args: argparse.Namespace) -> dict[str, Any]:
    work: Path = args.work_dir
    cache = work / "candidate_pool_neutral_cache"
    cache.mkdir(parents=True, exist_ok=True)
    fwd = work / "lqtp_fwd_vwap_returns_cache.parquet"
    if not fwd.is_file():
        raise FileNotFoundError(fwd)

    years = list(args.years)
    jobs: list[dict[str, Any]] = []
    for pool in POOLS:
        if args.source and pool["source"] not in args.source and pool.get("cache_source") not in args.source:
            continue
        ids = _list_factor_ids(pool["prefix"])
        print(f"[{pool['source']}] factors={len(ids)}", flush=True)
        for fid in ids:
            jobs.append({**pool, "factor_id": fid})

    # download
    dl_tasks: list[tuple[str, str, str, int, Path]] = []
    for job in jobs:
        fid = job["factor_id"]
        available = _list_years(job["prefix"], fid)
        use_years = [y for y in years if y in available] or available
        job["years"] = use_years
        cache_key = str(job.get("cache_source") or job["source"])
        job["cache_source"] = cache_key
        for y in use_years:
            dest = cache / cache_key / fid / f"{y}.parquet"
            dl_tasks.append((job["prefix"], fid, cache_key, y, dest))
            # optional raw year for neutralization probe (one year later)
        if args.probe_neutral and use_years:
            y = use_years[-2] if len(use_years) >= 2 else use_years[-1]
            dest = cache / f"{cache_key}_raw" / fid / f"{y}.parquet"
            dl_tasks.append((job["raw_prefix"], fid, f"{cache_key}_raw", y, dest))
            job["probe_year"] = y

    print(f"download tasks={len(dl_tasks)} workers={args.download_workers}", flush=True)
    ok_dl = 0
    with ThreadPoolExecutor(max_workers=args.download_workers) as ex:
        futs = {
            ex.submit(
                _download_year,
                cos_prefix=prefix,
                factor_id=fid,
                year=year,
                dest=dest,
            ): (fid, year, dest)
            for prefix, fid, _src, year, dest in dl_tasks
        }
        for i, fut in enumerate(as_completed(futs), 1):
            fid, year, dest = futs[fut]
            path = fut.result()
            if path is not None:
                ok_dl += 1
            if i % 50 == 0 or i == len(futs):
                print(f"  dl {i}/{len(futs)} ok={ok_dl}", flush=True)

    results: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    eval_payloads: list[dict[str, Any]] = []
    for job in jobs:
        fid = job["factor_id"]
        cache_key = str(job.get("cache_source") or job["source"])
        paths = [
            cache / cache_key / fid / f"{y}.parquet"
            for y in job.get("years", [])
            if (cache / cache_key / fid / f"{y}.parquet").exists()
        ]
        if not paths:
            results.append(
                {
                    "factor_id": fid,
                    "source": job["source"],
                    "label": job["label"],
                    "ok": False,
                    "error": "no_local_years",
                }
            )
            continue
        probe = None
        if args.probe_neutral:
            y = job.get("probe_year") or int(paths[-1].stem)
            neu_p = cache / cache_key / fid / f"{y}.parquet"
            raw_p = cache / f"{cache_key}_raw" / fid / f"{y}.parquet"
            if neu_p.exists():
                probe = _neutralization_probe(neu_p, raw_p if raw_p.exists() else None)
                probe.update({"factor_id": fid, "source": job["source"], "year": y})
                probes.append(probe)
        eval_payloads.append(
            {
                "factor_id": fid,
                "source": job["source"],
                "label": job["label"],
                "paths": [str(p) for p in paths],
                "fwd": str(fwd),
                "neutral_probe": probe,
            }
        )

    print(f"eval factors={len(eval_payloads)} workers={args.eval_workers}", flush=True)
    if args.eval_workers <= 1:
        for i, payload in enumerate(eval_payloads, 1):
            row = _eval_job(payload)
            results.append(row)
            print(
                f"[{i}/{len(eval_payloads)}] {row.get('factor_id')} ok={row.get('ok')} "
                f"|IC|={row.get('abs_mean_rank_ic')} sec={row.get('elapsed_sec')}",
                flush=True,
            )
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=args.eval_workers) as ex:
            futs = [ex.submit(_eval_job, payload) for payload in eval_payloads]
            for fut in as_completed(futs):
                row = fut.result()
                results.append(row)
                done += 1
                print(
                    f"[{done}/{len(eval_payloads)}] {row.get('factor_id')} ok={row.get('ok')} "
                    f"|IC|={row.get('abs_mean_rank_ic')} sec={row.get('elapsed_sec')}",
                    flush=True,
                )

    selected = [
        r
        for r in results
        if r.get("ok")
        and r.get("ok_for_select", True)
        and isinstance(r.get("abs_mean_rank_ic"), (int, float))
        and r["abs_mean_rank_ic"] == r["abs_mean_rank_ic"]
        and r["abs_mean_rank_ic"] > float(args.threshold)
    ]
    selected.sort(key=lambda r: r["abs_mean_rank_ic"], reverse=True)

    # display with nonnegative IC (sign-flip convention matches homepage)
    for r in selected:
        raw_ic = float(r["mean_rank_ic"])
        r["display_rank_ic"] = abs(raw_ic)
        r["sign_flipped"] = bool(raw_ic < 0)
        r["display_rank_icir"] = abs(float(r.get("rank_icir") or 0.0))

    payload = {
        "generated_at": datetime.now().isoformat(),
        "threshold": float(args.threshold),
        "years": years,
        "fwd_returns": str(fwd),
        "pools": [
            {
                "source": p["source"],
                "label": p["label"],
                "prefix": p["prefix"],
                "raw_prefix": p["raw_prefix"],
            }
            for p in POOLS
        ],
        "n_crawled": len(results),
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_selected": len(selected),
        "neutralization_summary": {
            "n_probed": len(probes),
            "n_looks_cs_neutral": sum(1 for p in probes if p.get("looks_cs_neutral")),
            "n_differs_from_raw": sum(1 for p in probes if p.get("differs_from_raw")),
            "probes": probes,
        },
        "selected": selected,
        "all_results": results,
    }

    reports = work / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    json_path = reports / "weekly_dug_neutral_rankic.json"
    csv_path = reports / "weekly_dug_neutral_rankic.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "factor_id",
                "source",
                "label",
                "mean_rank_ic",
                "display_rank_ic",
                "rank_icir",
                "display_rank_icir",
                "sign_flipped",
                "mean_daily_coverage",
                "ls_mean_one_way_turnover",
                "long_short_sharpe",
                "n_days",
            ],
        )
        w.writeheader()
        for r in selected:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print(f"wrote {json_path} selected={len(selected)}/{len(results)}", flush=True)
    return payload


def _weekly_section_html(payload: dict[str, Any], *, threshold: float) -> str:
    selected = payload.get("selected") or []
    neu = payload.get("neutralization_summary") or {}
    rows = []
    toc = []
    for i, r in enumerate(selected, 1):
        ic = float(r.get("display_rank_ic") or r.get("abs_mean_rank_ic") or 0.0)
        icir = r.get("display_rank_icir")
        icir_s = f"{float(icir):.4f}" if icir is not None else "—"
        flip = "是" if r.get("sign_flipped") else "否"
        cov = r.get("mean_daily_coverage")
        cov_s = f"{float(cov):.1%}" if isinstance(cov, (int, float)) and cov == cov else "—"
        sh = r.get("long_short_sharpe")
        sh_s = f"{float(sh):.3f}" if isinstance(sh, (int, float)) and sh == sh else "—"
        fid_raw = str(r.get("factor_id", ""))
        fid = html_lib.escape(fid_raw)
        src = html_lib.escape(str(r.get("label") or r.get("source_label") or "外部中性化候选"))
        href = str(r.get("report_href") or "")
        name_cell = (
            f'<a href="{html_lib.escape(href)}"><code>{fid}</code></a>'
            if href
            else f"<code>{fid}</code>"
        )
        plat = "本地" if str(r.get("platform_submit") or "").startswith("skipped") else (
            "平台" if r.get("platform_submit") else "—"
        )
        rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{name_cell}</td>"
            f"<td>{src}</td>"
            f"<td>{ic:.4f}</td>"
            f"<td>{icir_s}</td>"
            f"<td>{cov_s}</td>"
            f"<td>{sh_s}</td>"
            f"<td>{flip}</td>"
            f"<td>{plat}</td>"
            "</tr>"
        )
        toc_name = (
            f'<a href="{html_lib.escape(href)}"><code>{fid}</code></a>'
            if href
            else f"<code>{fid}</code>"
        )
        toc.append(f"<li>{toc_name} <span class='muted'>({ic:.2%})</span></li>")

    gen = html_lib.escape(str(payload.get("generated_at", "")))
    plat = html_lib.escape(str(payload.get("platform_note") or "无公式则本地回测；有平台公式才提交平台。"))
    return f"""
<section id="weekly-dug-neutral" class="week2-section">
  <h2>本周新挖 · 中性化候选</h2>
  <div class="notice">
    <p><b>来源</b>：外部中性化候选池（面板已中性化落盘）。抽样核对截面均值≈0，且与未中性化副本数值不同。</p>
    <p><b>筛选</b>：VWAP→VWAP Mean |RankIC| &gt; {threshold:.0%}；表中 RankIC 取绝对值（负向已标注「取负」）。
    与上方历史精选表<strong>分开列出</strong>。</p>
    <p><b>回测</b>：{plat}</p>
    <p class="muted">生成 {gen} · 爬取 {payload.get('n_crawled')} · 成功评估 {payload.get('n_ok')} ·
    入选 {payload.get('n_selected')} · 中性抽样 {neu.get('n_looks_cs_neutral')}/{neu.get('n_probed')} ·
    异于未中性化 {neu.get('n_differs_from_raw')}/{neu.get('n_probed')}</p>
  </div>
  <h3>快速目录</h3>
  <ul class="toc">{''.join(toc) if toc else '<li class="muted">（本周无过线因子）</li>'}</ul>
  <h3>因子列表</h3>
  <table>
    <thead>
      <tr>
        <th>#</th><th>因子</th><th>来源</th><th>Mean RankIC</th><th>RankICIR</th>
        <th>日覆盖率</th><th>多空Sharpe</th><th>取负显示</th><th>回测</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows) if rows else '<tr><td colspan="9" class="muted">无 |RankIC|&gt;阈值 的因子</td></tr>'}
    </tbody>
  </table>
</section>
"""


def patch_screening_html(args: argparse.Namespace, payload: dict[str, Any] | None = None) -> Path:
    work: Path = args.work_dir
    html_path = args.html or (work / "reports/factor_rankic_screening_index.html")
    json_path = work / "reports/weekly_dug_neutral_rankic.json"
    if payload is None:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    section = _weekly_section_html(payload, threshold=float(payload.get("threshold", args.threshold)))
    text = html_path.read_text(encoding="utf-8")
    # remove prior weekly section if present
    text = re.sub(
        r'<section id="weekly-dug-neutral"[\s\S]*?</section>\s*',
        "",
        text,
        count=1,
    )
    # insert before Week2 / corr / </main>
    anchor = re.search(
        r"(<section class=['\"]week2-section['\"]>|<section class=['\"]corr-embed['\"]>|</main>)",
        text,
    )
    if not anchor:
        raise RuntimeError(f"cannot find insertion point in {html_path}")
    text = text[: anchor.start()] + section + "\n" + text[anchor.start() :]
    # toc link in 快速目录 header area — add notice line near hero if missing
    if "本周新挖 · 中性化候选" not in text.split("本周新挖 · 中性化候选")[0][-2000:]:
        pass
    if 'href="#weekly-dug-neutral"' not in text:
        text = text.replace(
            "<h2>快速目录</h2>",
            '<h2>快速目录</h2>\n    <p class="muted"><a href="#weekly-dug-neutral">↓ 跳到本周新挖（中性化候选）</a></p>',
            1,
        )
    bak = html_path.with_suffix(html_path.suffix + f".bak_weekly_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    bak.write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")
    html_path.write_text(text, encoding="utf-8")
    print(f"patched {html_path} (backup {bak.name})", flush=True)
    return html_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    p.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT)
    p.add_argument("--years", type=int, nargs="+", default=YEARS_DEFAULT)
    p.add_argument("--source", nargs="+", choices=["pool_a", "pool_b", "evoalpha", "alphasage"], default=None)
    p.add_argument("--download-workers", type=int, default=12)
    p.add_argument("--eval-workers", type=int, default=4)
    p.add_argument("--probe-neutral", action="store_true", default=True)
    p.add_argument("--no-probe-neutral", action="store_false", dest="probe_neutral")
    p.add_argument("--skip-eval", action="store_true", help="Only patch HTML from existing JSON")
    p.add_argument("--patch-html", action="store_true", default=True)
    p.add_argument("--no-patch-html", action="store_false", dest="patch_html")
    p.add_argument("--html", type=Path, default=None)
    args = p.parse_args()

    payload = None
    if not args.skip_eval:
        payload = crawl_and_eval(args)
    if args.patch_html:
        patch_screening_html(args, payload=payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
