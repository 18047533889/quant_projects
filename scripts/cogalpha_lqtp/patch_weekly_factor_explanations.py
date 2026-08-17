#!/usr/bin/env python3
"""Write Chinese explanations for this week's 53 dug factors into detail HTML + JSON cache."""
from __future__ import annotations

import datetime as dt
import html as html_lib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "cogalpha_lqtp")]

from scripts.cogalpha_lqtp.factor_report_zh import render_operator_list_html  # noqa: E402
from scripts.cogalpha_lqtp.weekly_dug_factor_guides import (  # noqa: E402
    WEEKLY_DUG_FACTOR_GUIDES,
    assert_complete,
)

WORK = ROOT / "data/cogalpha_lqtp_production"


def _block(guide: dict[str, Any]) -> str:
    steps_html = "".join(f"<li>{html_lib.escape(s)}</li>" for s in guide.get("steps", []))
    ops_html = render_operator_list_html(list(guide.get("operators") or []))
    ops_section = f"<h3>公式里用到的算子</h3>{ops_html}" if ops_html else ""
    return f"""  <div class="card">
    <h2>因子说明</h2>
    <p class="theme-line"><b>{html_lib.escape(guide.get('theme',''))}</b></p>
    <p>{html_lib.escape(guide.get('summary',''))}</p>
    <h3>计算步骤</h3>
    <ol class="steps">{steps_html}</ol>
    <h3>因子值怎么理解</h3>
    <p>{html_lib.escape(guide.get('direction',''))}</p>
    {ops_section}
  </div>"""


def _patch_html(text: str, block: str) -> tuple[str, int]:
    new_text, n = re.subn(
        r'<div class="card">\s*<h2>因子说明</h2>[\s\S]*?</div>\s*(?=<div class="card">)',
        block.strip() + "\n  \n  ",
        text,
        count=1,
    )
    return new_text, n


def _summary_html(rows: list[dict[str, Any]]) -> str:
    items = []
    for i, r in enumerate(rows, 1):
        steps = "".join(f"<li>{html_lib.escape(s)}</li>" for s in r.get("steps") or [])
        ic = r.get("ic")
        ls = r.get("ls_sharpe")
        meta = []
        if ic is not None:
            meta.append(f"RankIC={ic}")
        if ls is not None:
            meta.append(f"多空Sharpe={ls}")
        meta_s = " · ".join(meta)
        items.append(
            f"""<article class="f">
  <h2>{i}. <code>{html_lib.escape(r['name'])}</code></h2>
  <p class="meta">{html_lib.escape(meta_s)}</p>
  <p><b>{html_lib.escape(r.get('theme',''))}</b></p>
  <p>{html_lib.escape(r.get('summary',''))}</p>
  <ol>{steps}</ol>
  <p class="dir">{html_lib.escape(r.get('direction',''))}</p>
</article>"""
        )
    body = "\n".join(items)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>本周因子释义 · 53</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;max-width:920px;margin:24px auto;padding:0 16px;line-height:1.55;color:#111}}
h1{{font-size:1.4rem}} h2{{font-size:1.05rem;margin:0 0 6px}}
.f{{border-top:1px solid #e5e7eb;padding:18px 0}}
.meta{{color:#6b7280;font-size:.9rem;margin:0 0 8px}}
.dir{{background:#f8fafc;padding:10px 12px;border-radius:6px}}
code{{font-size:.92em}}
ol{{margin:8px 0 8px 1.2em}}
</style></head><body>
<h1>本周新挖因子 · 怎么算 / 啥意思（53）</h1>
<p>每条详情页「因子说明」已同步；也可在索引点进单因子页查看公式与图表。</p>
{body}
</body></html>
"""


def main() -> int:
    wp = WORK / "reports/weekly_dug_neutral_rankic.json"
    weekly = json.loads(wp.read_text(encoding="utf-8"))
    names = [str(r.get("display_name") or r.get("factor_id")) for r in (weekly.get("selected") or [])]
    missing = assert_complete(names)
    if missing:
        print("MISSING_GUIDES", missing, flush=True)
        return 2

    out_dir = WORK / "reports_weekly_dug_wnew"
    cache = WORK / "reports/weekly_factor_explanations.json"
    rows: list[dict[str, Any]] = []
    ok = 0
    for r in weekly.get("selected") or []:
        name = str(r.get("display_name") or r.get("factor_id"))
        g = dict(WEEKLY_DUG_FACTOR_GUIDES[name])
        block = _block(g)
        hp = out_dir / f"{name}.html"
        if not hp.exists():
            print(f"MISS_HTML {name}", flush=True)
            continue
        text = hp.read_text(encoding="utf-8")
        new_text, n = _patch_html(text, block)
        if n == 0:
            print(f"WARN_NO_REPLACE {name}", flush=True)
            continue
        hp.write_text(new_text, encoding="utf-8")
        ok += 1
        rows.append(
            {
                "name": name,
                "theme": g.get("theme"),
                "summary": g.get("summary"),
                "steps": g.get("steps"),
                "direction": g.get("direction"),
                "ic": r.get("display_rank_ic"),
                "ls_sharpe": r.get("long_short_sharpe"),
            }
        )
        print(f"OK {name}", flush=True)

    cache.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    by = {x["name"]: x for x in rows}
    for r in weekly.get("selected") or []:
        name = r.get("display_name")
        if name in by:
            r["explanation_zh"] = {
                "theme": by[name]["theme"],
                "summary": by[name]["summary"],
                "steps": by[name]["steps"],
                "direction": by[name]["direction"],
            }
    weekly["explanations_patched_at"] = dt.datetime.now().isoformat()
    weekly["explanations_n"] = ok
    wp.write_text(json.dumps(weekly, ensure_ascii=False, indent=2), encoding="utf-8")

    sum_path = WORK / "reports/weekly_dug_factor_explanations.html"
    sum_path.write_text(_summary_html(rows), encoding="utf-8")
    print(f"DONE ok={ok}/{len(names)} cache={cache} summary={sum_path}", flush=True)
    return 0 if ok == len(names) else 1


if __name__ == "__main__":
    raise SystemExit(main())
