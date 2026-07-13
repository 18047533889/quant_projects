#!/usr/bin/env python3
"""Re-render Week3 factor detail HTML with Chinese interpretation blocks."""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cogalpha_lqtp.eval_lake_fast import analyze_factor_parquet_duckdb, build_fwd_returns_cache
from scripts.cogalpha_lqtp.factor_report_zh import WEEK3_FACTOR_GUIDES
from scripts.cogalpha_lqtp.report_html import _enrich_topk_summary, render_factor_report

WEEK3_FACTORS = list(WEEK3_FACTOR_GUIDES.keys())

_TOPK_IMG_RE = re.compile(
    r'<img alt="(?:TopK NAV|TopK 多头组合净值[^"]*)" src="(data:image/png;base64,[^"]+)"/>'
)


def _extract_topk_img(html: str) -> str:
    m = _TOPK_IMG_RE.search(html)
    if not m:
        return ""
    return f'<img alt="TopK 多头组合净值" src="{m.group(1)}"/>'


def _inject_topk_img(html: str, img_tag: str) -> str:
    if not img_tag:
        return html
    if _TOPK_IMG_RE.search(html):
        return html
    needle = "有 TopK 指标摘要，但净值曲线未缓存"
    if needle in html:
        return html.replace(
            '<p class="note">有 TopK 指标摘要，但净值曲线未缓存（需重新跑 LQTP TopK 回测才会出现 NAV 图）。</p>',
            img_tag,
        )
    # fallback: insert before TopK kv table
    marker = "<h3>TopK 回测指标明细</h3>"
    if marker in html:
        return html.replace(marker, img_tag + "\n    " + marker, 1)
    return html


def _topk_summary_from_row(row: dict) -> dict:
    return _enrich_topk_summary(
        {
            "total_return": row.get("backtest_total_ret"),
            "annualized_return": row.get("backtest_ann_ret"),
            "max_drawdown": row.get("backtest_max_drawdown"),
            "sharpe": row.get("backtest_sharpe"),
            "volatility": row.get("backtest_volatility"),
            "calmar": row.get("backtest_calmar"),
            "win_rate": row.get("backtest_win_rate"),
            "avg_turnover": row.get("backtest_avg_turnover"),
            "total_commission": row.get("backtest_total_commission"),
            "trading_days": row.get("backtest_trading_days"),
            "final_nav": row.get("backtest_final_nav"),
        }
    )


def regen_week3_reports(work_dir: Path | None = None) -> int:
    work_dir = work_dir or ROOT / "data/cogalpha_lqtp_production"
    progress = json.loads((work_dir / "production_progress.json").read_text(encoding="utf-8"))
    catalog = {e["function_name"]: e for e in json.loads((work_dir / "dsl_catalog.json").read_text())}
    parsed = {r["function_name"]: r.get("python_code", "") for r in json.loads((work_dir / "parsed_factors.json").read_text())}
    flip = progress.get("flip_state", {})
    by_name = {r["factor_name"]: r for r in progress.get("index_rows", [])}

    fwd = build_fwd_returns_cache(
        work_dir / "lqtp_close_returns_cache.parquet",
        work_dir / "lqtp_fwd_close_returns_cache.parquet",
    )
    report_dir = work_dir / "reports"
    n = 0

    for name in WEEK3_FACTORS:
        row = by_name.get(name)
        if not row:
            print(f"skip (not in progress): {name}")
            continue
        pq = work_dir / "factor_lake" / name / "values.parquet"
        if not pq.exists():
            print(f"skip (no parquet): {name}")
            continue
        entry = catalog.get(name, {})
        py = parsed.get(name, "")
        if flip.get(name, {}).get("python_code"):
            py = flip[name]["python_code"]
        dsl = (entry.get("dsl") or "").strip() or "(python only)"
        report_path = report_dir / row.get("report", f"{name}.html")
        bt_path = work_dir / "factor_lake" / name / "backtest_topk.json"
        backtest_rows: list = []
        if bt_path.exists():
            try:
                backtest_rows = json.loads(bt_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                backtest_rows = []
        print(f"regen: {name} (topk_rows={len(backtest_rows)})")
        analysis = analyze_factor_parquet_duckdb(factor_path=pq, fwd_returns_path=fwd)
        bt = row.get("backtest_sharpe")
        topk_summary = _topk_summary_from_row(row) if bt is not None and not (
            isinstance(bt, float) and math.isnan(bt)
        ) else None
        render_factor_report(
            factor_name=name,
            dsl=dsl,
            analysis=analysis,
            backtest_rows=backtest_rows,
            out_path=report_path,
            eval_mode=row.get("eval_mode", ""),
            materialize_meta={
                "engine": row.get("engine", ""),
                "eval_route": row.get("eval_route", ""),
                "rows": row.get("rows"),
            },
            topk_summary=topk_summary,
            python_code=py,
            work_dir=work_dir,
            engine=str(row.get("engine") or ""),
            eval_route=str(row.get("eval_route") or ""),
        )
        n += 1
    return n


if __name__ == "__main__":
    count = regen_week3_reports()
    print(f"done: {count} factor detail pages")
