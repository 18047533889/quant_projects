#!/usr/bin/env python3
"""Live preview of the 本周新增因子 section, for use while the repair chain runs.

Why this exists
---------------
The published homepage (``index.html``) only gets its new-mining section during
``finalize_newmining.sh`` step [5/7], which cannot start until the serial
re-landing finishes (FactorEngine may not run the heavy jobs in parallel).
Until then the report is stuck on the previous run's numbers.

This script renders the *current* manifest state into a standalone page inside
the report directory, so it is reachable over the same 9123 tunnel as the main
report.  It never touches ``index.html``, and it is safe to re-run at any time.

What it adds over the published section
---------------------------------------
* the readability column (``source_formula``) next to every factor,
* a full roster of all candidates with status / metrics / failure reason,
* a banner stating the landing progress, so a partial number is never mistaken
  for the final one.

Usage
-----
    .venv/bin/python jobs/nm_preview.py [--out nm_preview.html]
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import new_mining_publish as P  # noqa: E402

REPORT_DIR = Path(
    "/home/sunhaiwei/quant_project_archives/factor_engine-docs/reports/2026-08-23"
)

STATUS_LABEL = {
    "evaluated": "达门槛",
    "below_gate": "未达门槛",
    "unavailable": "不可用",
    "failed": "失败",
}
STATUS_COLOR = {
    "evaluated": ("#dcfce7", "#166534"),
    "below_gate": ("#fef3c7", "#92400e"),
    "unavailable": ("#e2e8f0", "#475569"),
    "failed": ("#fee2e2", "#991b1b"),
}


def report_styles(report_dir: Path) -> str:
    """The report's own inline CSS, so the preview matches the main page."""
    index = report_dir / "index.html"
    if not index.exists():
        return ""
    text = index.read_text(encoding="utf-8", errors="replace")
    cut = text.lower().find("</head>")
    head = text[:cut] if cut > 0 else text[:20000]
    return "\n".join(re.findall(r"<style[^>]*>(.*?)</style>", head, re.S))


def row_formula(row) -> str:
    entry = row.get("entry") or {}
    for key in ("source_formula", "fe_formula", "lqtp_formula"):
        value = entry.get(key) or row.get(key)
        if value:
            return str(value)
    return ""


def row_note(row) -> str:
    entry = row.get("entry") or {}
    bits = []
    if entry.get("dsl_repair"):
        bits.append("公式已修：" + str(entry["dsl_repair"])[:120])
    if entry.get("landing_backend_fallbacks"):
        bits.append("后端回退：" + ", ".join(map(str, entry["landing_backend_fallbacks"]))[:120])
    reason = row.get("reason") or entry.get("reason") or ""
    if reason:
        bits.append(P.scrub_names(str(reason))[:220])
    return "；".join(bits)


def _ic_key(row) -> float:
    value = (row.get("metrics") or {}).get("rank_ic")
    return -value if isinstance(value, (int, float)) else 9.0


def build_appendix(rows) -> str:
    order = {"evaluated": 0, "below_gate": 1, "unavailable": 2, "failed": 3}
    rows = sorted(
        rows,
        key=lambda r: (order.get(str(r.get("status")), 9), _ic_key(r)),
    )
    body = []
    for i, r in enumerate(rows, 1):
        status = str(r.get("status"))
        bg, fg = STATUS_COLOR.get(status, ("#e2e8f0", "#334155"))
        m = r.get("metrics") or {}
        body.append(
            "<tr>"
            f'<td class="rank">{i}</td>'
            f'<td><code>{P.esc(P.display_name(r["name"]))}</code></td>'
            f'<td><code class="fe-formula">{P.esc(row_formula(r))}</code></td>'
            f'<td><span class="tag" style="background:{bg};color:{fg}">'
            f'{status}·{STATUS_LABEL.get(status, "?")}</span></td>'
            f'<td>{P._fmt(m.get("rank_ic"))}</td>'
            f'<td>{P._fmt(m.get("ic_ir"), "+.3f")}</td>'
            f'<td>{P._fmt(m.get("ls_sharpe"), "+.2f")}</td>'
            f'<td style="font-size:0.7rem;color:#64748b">{P.esc(str(r.get("evaluated_at") or "—")[:19])}</td>'
            f'<td style="font-size:0.72rem;color:#64748b">{P.esc(row_note(r))}</td>'
            "</tr>"
        )
    return (
        '<h2 id="roster">全部候选台账（含未达标/未落地）</h2>'
        '<p style="font-size:0.82rem;color:#64748b">按状态排序（达门槛 → 未达门槛 → 不可用 → 失败），'
        '同状态内按 RankIC 降序。备注列保留原始失败原因与修复痕迹，不做美化。</p>'
        '<table><thead><tr><th>#</th><th>因子</th><th>公式</th><th>状态</th><th>RankIC</th>'
        "<th>IR</th><th>LS Sharpe</th><th>本轮评估时间</th><th>备注</th></tr></thead>"
        f'<tbody>{"".join(body)}</tbody></table>'
    )


def inject_formulas(section: str, rows) -> str:
    """Add the readable expression under each headline factor name.

    Each headline display name occurs once in the table; the collapsed blocks
    reuse the same ``<code>`` markup, so only the first hit is replaced.
    """
    for r in rows:
        dn = P.display_name(r["name"])
        formula = row_formula(r)
        if not formula:
            continue
        needle = f"<code>{dn}</code>"
        if needle in section:
            section = section.replace(
                needle,
                f'<code>{dn}</code><div class="fe-formula">{P.esc(formula)}</div>',
                1,
            )
    return section


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="nm_preview.html")
    ap.add_argument("--report-dir", default=str(REPORT_DIR))
    args = ap.parse_args()

    report_dir = Path(args.report_dir)

    rows = list(P.load_results().values())
    status = collections.Counter(str(r.get("status")) for r in rows)
    landed = status["evaluated"] + status["below_gate"]

    section = P.build_new_mining_section(rows)
    section = inject_formulas(section, rows)
    # A detail page exists only after ``--pages`` has run (finalize step [4/7],
    # or a manual ``--pages`` while the repair is still going).  Keep the link
    # for factors that already have one, and render the rest as plain text with
    # a hint instead of a link that would 404.
    def _keep_or_hint(match):
        if (report_dir / match.group(1)).exists():
            return match.group(0)
        return ('title="详情页在收尾阶段（finalize [4/7]）生成；当前为实时预览"'
                ' style="cursor:not-allowed;opacity:.8"')

    section = re.sub(r'href="(factors/[^"]+)"', _keep_or_hint, section)
    pages = len(list((report_dir / "factors").glob("factor_nm_*.html")))

    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    banner = f"""
<div style="border:2px solid #f59e0b;background:#fffbeb;border-radius:10px;padding:14px 18px;margin:18px 0">
  <div style="font-size:1.05rem;font-weight:700;color:#92400e">
    实时预览 · 本页不是已发布版本
  </div>
  <p style="margin:8px 0 4px;font-size:0.86rem;color:#78350f">
    生成时间（北京时间）：<b>{now}</b>　|　数据源：<code>work/newmining_20260919</code> 的 manifest 实时台账。
  </p>
  <p style="margin:4px 0;font-size:0.86rem;color:#78350f">
    落地进度：<b>{landed}/{len(rows)}</b> 已完成落值 + 评估
    （达门槛 <b>{status['evaluated']}</b>、未达门槛 <b>{status['below_gate']}</b>）；
    另有 <b>{status['failed'] + status['unavailable']}</b> 个（失败 {status['failed']}、不可用 {status['unavailable']}）
    正在<b>串行重落地</b>，完成后数字还会上升。
  </p>
  <p style="margin:4px 0;font-size:0.86rem;color:#78350f">
    单因子详情页：已生成 <b>{pages}</b> 个（口径：仅达门槛因子生成详情页，即
    <code>factors/factor_nm_*.html</code>）；表内名称可点即表示该详情页已就绪，未就绪的为灰色不可点。
  </p>
  <p style="margin:4px 0 0;font-size:0.86rem;color:#78350f">
    正式首页 <code>index.html</code> 的新增栏由收尾脚本（finalize [4/7]–[5/7]）统一重建，
    重落地跑完后自动接续；本页在收尾前可反复刷新查看进度。
  </p>
</div>
"""

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>本周新增因子 · 实时预览（{landed}/{len(rows)} 已落地）</title>
<style>
{report_styles(report_dir)}
code.fe-formula, .fe-formula {{
  display:block; font-size:0.72rem; color:#475569; margin-top:3px;
  white-space:normal; word-break:break-word; line-height:1.35;
}}
table {{ table-layout:auto; }}
td code {{ font-size:0.76rem; }}
</style>
</head>
<body>
<main>
{banner}
{section}
{build_appendix(rows)}
</main>
</body>
</html>
"""

    out = Path(args.out)
    if not out.is_absolute():
        out = report_dir / out
    out.write_text(page, encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "bytes": out.stat().st_size,
        "generated_at": now,
        "landed": landed,
        "total": len(rows),
        "status": dict(status),
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
