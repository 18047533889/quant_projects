#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""首页统计头部重做 + 详情页「计算步骤」区块注入。

职责：
  1. 首页 index.html：重做统计头（真实数字 470/339/25/267/22/82）+ 整体 CSS 现代化
     + 因子族聚类区块（factor_clusters.json，Agent D 的归因区块保持不动）。
  2. 详情页 factors/factor_*.html：注入「计算步骤」卡片（在 DSL 卡片之后）。

用法：
  python jobs/inject_steps_and_header.py [--pages p1,p2,...] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT = Path("/home/sunhaiwei/quant_projects")
REPORT_DIR = PROJECT / "factor_engine" / "docs" / "reports" / "2026-08-23"
INDEX = REPORT_DIR / "index.html"
FACTORS_DIR = REPORT_DIR / "factors"
META = PROJECT / "weekly_backtest_output" / "optimized_meta.json"
EVAL = PROJECT / "weekly_backtest_output" / "eval_shift2.json"
CLUSTER = PROJECT / "weekly_backtest_output" / "factor_clusters.json"
ROB = PROJECT / "weekly_backtest_output" / "robustness_2026.json"
LQTP_ALL = Path("/home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json")
OLD_META = PROJECT / "weekly_backtest_output" / "optimized_meta_STALE_UNADJ_0828.json"

sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "jobs"))

from dsl_steps.generator import build_steps_block  # noqa: E402


# ---------------------------------------------------------------------------
# 头部数据
# ---------------------------------------------------------------------------
def compute_headline() -> dict:
    meta = json.loads(META.read_text())
    eval2 = json.loads(EVAL.read_text())
    cluster = json.loads(CLUSTER.read_text())
    rob = json.loads(ROB.read_text())
    old = json.loads(OLD_META.read_text()) if OLD_META.exists() else {}

    nonproxy = {k: v for k, v in eval2.items() if not v.get("proxy", False)}
    true_alpha = [k for k in nonproxy if meta.get(k, {}).get("best_rankic_ir", 0) > 0]
    strong = [k for k in nonproxy if meta.get(k, {}).get("best_rankic_ir", 0) > 0.5]
    flipped = [k for k, v in meta.items() if v.get("is_flipped")]
    gate_pass = [k for k, v in (cluster.get("quality_gate") or {}).items()
                 if v.get("quality_gate") == "pass"]
    stable = [r["page"] for r in rob if r.get("status") == "stable"]
    new_only = [k for k in meta if k not in old]

    return {
        "n_total": len(meta),
        "n_new": len(new_only),
        "true_alpha": sorted(true_alpha),
        "strong": sorted(strong),
        "flipped": sorted(flipped),
        "gate_pass": sorted(gate_pass),
        "stable": sorted(stable),
        "n_clusters": len(cluster.get("cluster_members") or {}),
        "n_multi": sum(1 for v in (cluster.get("cluster_members") or {}).values() if len(v) > 1),
        "largest": cluster.get("summary", {}).get("largest_cluster", 0),
        "gen_date": "2026-08-29",
    }


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _link_join(names: list[str], limit: int = 18) -> str:
    """生成逗号分隔的因子链接（可点击跳详情页）。"""
    shown = names[:limit]
    parts = "".join(
        f'<a href="factors/factor_{_esc(n)}.html" style="color:#0d9488;text-decoration:none;border-bottom:1px dashed #99f6e4">{_esc(n)}</a>'
        for n in shown
    )
    if len(names) > limit:
        parts += f'<span style="color:#64748b">…等 {len(names)} 个</span>'
    return parts


# ---------------------------------------------------------------------------
# 首页：头部 + CSS + 聚类区块
# ---------------------------------------------------------------------------
HEADER_HTML = """<header>
  <div class="hd-top">
    <div>
      <h1>量化因子总览 <span class="badge badge-qe">⚡ quant_evaluator</span></h1>
      <div class="sub">factor_engine 落值 + quant_evaluator 评估 · 收益口径 Vwap 后复权 vwap-to-vwap（shift(-2) 企业级）· 生成 {gen_date}</div>
    </div>
    <div class="hd-stats">
      <div class="hd-stat"><b>{n_total}</b><span>因子总数</span></div>
      <div class="hd-stat hd-new"><b>+{n_new}</b><span>本周新挖</span></div>
    </div>
  </div>
  <div class="hd-cards">
    <a class="hd-card hd-blue" href="#all-factors">
      <b>{n_total}</b><span>因子总数（含本周新挖 {n_new}）</span>
      <i>全部已落值并评估</i>
    </a>
    <a class="hd-card hd-teal" href="#opt-summary">
      <b>{true_alpha_n}</b><span>真 Alpha（非代理 + 正 IR）</span>
      <i>排除代理类因子后仍有效的信号</i>
    </a>
    <a class="hd-card hd-violet" href="#opt-summary">
      <b>{strong_n}</b><span>强候选（IR&gt;0.5 且非代理）</span>
      <i>最值得入池的头部因子</i>
    </a>
    <a class="hd-card hd-amber" href="#all-factors">
      <b>{flipped_n}</b><span>自动翻正（负 IC→正）</span>
      <i>原始 IC&lt;0，已整体取反调正</i>
    </a>
    <a class="hd-card hd-green" href="#clusters">
      <b>{gate_n}</b><span>过质量门槛（聚类 gate）</span>
      <i>IR≥0.5 且 RankIC≥0.03</i>
    </a>
    <a class="hd-card hd-rose" href="#robustness-2026">
      <b>{stable_n}</b><span>2026 稳定因子</span>
      <i>全年未失效，存活 82/470</i>
    </a>
  </div>
</header>
"""


def build_header(h: dict) -> str:
    return HEADER_HTML.format(
        gen_date=h["gen_date"],
        n_total=h["n_total"],
        n_new=h["n_new"],
        true_alpha_n=len(h["true_alpha"]),
        strong_n=len(h["strong"]),
        flipped_n=len(h["flipped"]),
        gate_n=len(h["gate_pass"]),
        stable_n=len(h["stable"]),
    )


def build_cluster_section(h: dict) -> str:
    cluster = json.loads(CLUSTER.read_text())
    cm = cluster.get("cluster_members") or {}
    reps = cluster.get("representatives") or {}
    qg = cluster.get("quality_gate") or {}
    rows = []
    for cid, members in cm.items():
        rep = reps.get(cid, {})
        factor = rep.get("factor") or members[0]
        ir = rep.get("best_rankic_ir", 0) or 0
        rows.append((len(members), ir, cid, factor, members))
    rows.sort(key=lambda r: (-r[0], -r[1]))
    trs = []
    for size, ir, cid, factor, members in rows[:25]:
        gate = qg.get(factor, {}).get("quality_gate", "—")
        member_str = ", ".join(f"<code>{_esc(m)}</code>" for m in members[:6])
        if len(members) > 6:
            member_str += f"… 共 {len(members)} 个"
        trs.append(
            f'<tr><td><code>{_esc(cid)}</code></td><td>{size}</td>'
            f'<td><a href="factors/factor_{_esc(factor)}.html"><code>{_esc(factor)}</code></a></td>'
            f'<td>{ir:.3f}</td><td>{gate}</td>'
            f'<td style="font-size:0.75rem;color:#64748b">{member_str}</td></tr>'
        )
    gate_pass_names = ", ".join(f"<code>{_esc(n)}</code>" for n in h["gate_pass"][:12])
    return f"""
<section id="clusters">
<h2>🧩 因子族（相关性聚类去冗余）</h2>
<p style="font-size:0.82rem;color:#64748b">按因子值 Spearman 相关 ≥0.85 聚类（shift(-2) 企业级口径，339 真因子）。
共 {h['n_clusters']} 簇（{h['n_multi']} 个多成员簇，最大簇 {h['largest']} 个）——同簇因子高度相关，建议每簇只保留代表因子（簇内 IR 最高者）入池。</p>
<p style="font-size:0.82rem;color:#0f766e"><b>通过质量门槛（IR≥0.5 &amp; RankIC≥0.03）的代表因子 {len(h['gate_pass'])} 个：</b>{gate_pass_names}</p>
<table>
  <thead><tr><th>簇</th><th>成员数</th><th>代表因子</th><th>代表 IR</th><th>质量门槛</th><th>成员示例</th></tr></thead>
  <tbody>{''.join(trs)}</tbody>
</table>
</section>
"""


MODERN_CSS = """
:root {
  --bg:#0f172a; --bg2:#1e293b; --panel:#ffffff; --fg:#0f172a; --muted:#64748b;
  --line:#e2e8f0; --primary:#0f4c81; --primary2:#0d9488; --pos:#16a34a; --neg:#dc2626;
}
* { box-sizing:border-box }
html { scroll-behavior:smooth }
body { margin:0; font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif; color:var(--fg); background:#f1f5f9 }
header { background:linear-gradient(135deg,#0c1830,#0f4c81 55%,#0d9488); color:#fff; padding:32px 44px 28px }
header h1 { margin:0 0 6px; font-size:1.7rem; font-weight:700; letter-spacing:0.3px }
header .sub { opacity:0.82; font-size:0.85rem; margin-top:2px }
.hd-top { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; flex-wrap:wrap }
.hd-stats { display:flex; gap:10px }
.hd-stat { background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.22); border-radius:10px; padding:8px 16px; text-align:center; min-width:84px }
.hd-stat b { display:block; font-size:1.35rem; line-height:1.2 }
.hd-stat span { font-size:0.7rem; opacity:0.85 }
.hd-new b { color:#fde68a }
.hd-cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin-top:20px }
.hd-card { background:rgba(255,255,255,0.97); border-radius:14px; padding:14px 16px; color:var(--fg); text-decoration:none; box-shadow:0 10px 30px rgba(2,6,23,0.25); border-top:4px solid var(--primary); transition:transform .15s ease, box-shadow .15s ease; display:block }
.hd-card:hover { transform:translateY(-3px); box-shadow:0 16px 40px rgba(2,6,23,0.35) }
.hd-card b { display:block; font-size:1.7rem; line-height:1.25 }
.hd-card span { font-size:0.8rem; font-weight:600; color:#334155 }
.hd-card i { display:block; font-style:normal; font-size:0.72rem; color:#94a3b8; margin-top:3px }
.hd-blue { border-top-color:#2563eb } .hd-blue b { color:#2563eb }
.hd-teal { border-top-color:#0d9488 } .hd-teal b { color:#0d9488 }
.hd-violet { border-top-color:#7c3aed } .hd-violet b { color:#7c3aed }
.hd-amber { border-top-color:#d97706 } .hd-amber b { color:#d97706 }
.hd-green { border-top-color:#16a34a } .hd-green b { color:#16a34a }
.hd-rose { border-top-color:#e11d48 } .hd-rose b { color:#e11d48 }
main { max-width:1400px; margin:0 auto; padding:24px }
.cards { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:14px; margin:20px 0 }
.metric { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px; box-shadow:0 4px 24px rgba(15,23,42,0.06); text-align:center }
.metric b { display:block; font-size:1.6rem; color:var(--primary) }
.metric span { color:var(--muted); font-size:0.78rem }
.pos { color:var(--pos) } .neg { color:var(--neg) }
.chart-grid { display:grid; grid-template-columns:1fr; gap:20px; margin:20px 0 }
img { width:100%; border:1px solid var(--line); border-radius:10px; background:#fff }
table { width:100%; border-collapse:collapse; font-size:0.83rem; margin:10px 0 20px }
th { background:#f1f5f9; color:#475569; padding:8px 10px; text-align:left; font-weight:600; border-bottom:2px solid #cbd5e1; position:sticky; top:0 }
td { padding:7px 10px; border-bottom:1px solid var(--line) }
tr:hover td { background:#f8fafc }
.rank { color:#94a3b8; font-weight:600; width:36px }
code { background:#f1f5f9; padding:2px 6px; border-radius:4px; font-size:0.78rem; color:#1e293b }
.tag { display:inline-block; padding:1px 6px; border-radius:10px; font-size:0.68rem; margin-left:4px; vertical-align:middle }
.tag-flip { background:#fef3c7; color:#92400e }
h2 { font-size:1.05rem; color:var(--primary); margin:24px 0 10px; border-bottom:1px solid var(--line); padding-bottom:6px }
.badge { display:inline-block; background:linear-gradient(90deg,#ede9fe,#dbeafe); color:#5b21b6; padding:2px 10px; border-radius:12px; font-size:0.72rem; font-weight:600; margin-left:8px }
.badge-qe { background:rgba(255,255,255,0.18); color:#fff; border:1px solid rgba(255,255,255,0.35) }
section { margin:28px 0 }
"""


def rebuild_index():
    h = compute_headline()
    html = INDEX.read_text(encoding="utf-8")

    # ---- 1) 替换 CSS（整段 <style> 到 </style>）----
    html = re.sub(r"<style>.*?</style>", f"<style>{MODERN_CSS}</style>", html, count=1, flags=re.S)

    # ---- 2) 替换 <title> ----
    html = re.sub(r"<title>.*?</title>", f"<title>量化因子总览 · 470 因子（2026-08-29）</title>", html, count=1, flags=re.S)

    # ---- 3) 替换 header（<header> ... </header>）----
    new_header = build_header(h)
    html = re.sub(r"<header>.*?</header>", new_header, html, count=1, flags=re.S)

    # ---- 4) 替换旧统计卡（<div class="cards"> ... </div> 的第一处）----
    old_cards = re.search(r'<div class="cards">.*?</div>\s*</div>\s*<div class="chart-grid">', html, flags=re.S)
    if old_cards:
        html = html.replace(old_cards.group(0), '<div class="chart-grid">', 1)

    # ---- 5) 给「全部因子」表加锚点 id ----
    html = html.replace(
        '<h2>全部 456 个因子</h2>',
        '<h2 id="all-factors">全部 470 个因子</h2>', 1
    )
    html = html.replace(
        '<h2>🧬 优化因子汇总（预处理 + 择优）</h2>',
        '<h2 id="opt-summary">🧬 优化因子汇总（预处理 + 择优）</h2>', 1
    )

    # ---- 6) 重建「因子族聚类」区块（插在 </main> 前）----
    cluster_section = build_cluster_section(h)
    # 若已存在 id="clusters" 区块则替换，否则插入
    if re.search(r'<section id="clusters">.*?</section>', html, flags=re.S):
        html = re.sub(
            r'<section id="clusters">.*?</section>',
            cluster_section,
            html, count=1, flags=re.S
        )
    else:
        html = html.replace("</main>", cluster_section + "\n</main>", 1)

    INDEX.write_text(html, encoding="utf-8")
    print(f"[index] 头部 + CSS + 聚类区块 已更新: {len(html)} bytes")
    return True


# ---------------------------------------------------------------------------
# 详情页：计算步骤注入
# ---------------------------------------------------------------------------
def inject_steps(page: str) -> bool:
    """向单个详情页注入「计算步骤」卡片（在 DSL 卡片之后）。"""
    path = FACTORS_DIR / f"factor_{page}.html"
    if not path.exists():
        return False
    html = path.read_text(encoding="utf-8")
    if "📝 计算步骤" in html:
        return False  # 已注入

    # 从 LQTP 记录取 dsl / code / is_flipped
    rec = None
    for e in json.loads(LQTP_ALL.read_text()):
        if e.get("page_name") == page:
            rec = e
            break
    dsl = (rec or {}).get("dsl", "") or ""
    code = (rec or {}).get("code", "") or ""
    is_flipped = bool((rec or {}).get("is_flipped", False))

    block = build_steps_block(page, dsl, code, is_flipped)
    if not block:
        return False

    # 注入位置：DSL 卡片（<h2>📐 因子表达式 DSL</h2> 所在 .card 的结束）之后。
    # 找 DSL 卡片结尾：该 card 到下一个 '<!-- 输入字段释义 -->' 或下一个 '<div class="card">'
    dsl_marker = "📐 因子表达式 DSL"
    if dsl_marker not in html:
        # 没有 DSL 卡片则插到「公式」卡片后
        dsl_marker = "因子路由 / 公式"
    idx = html.find(dsl_marker)
    if idx == -1:
        return False
    # 找到所在 card 的结束 </div>
    card_end = html.find("</div>", idx)
    if card_end == -1:
        return False
    # 找到该 card 的下一段（可能是另一个 <div class="card"> 或注释或 <h2>）
    next_seg = html.find("\n<!--", card_end)
    if next_seg == -1:
        next_seg = html.find("<div class=\"card\">", card_end)
    if next_seg == -1:
        next_seg = html.find("</main>", card_end)
    if next_seg == -1:
        next_seg = card_end + 6
    # 计算卡片自身的 end：从 marker 所在 card 的开头算
    # 更稳妥：找该 card 的 <div class="card"> 起点（向前找最近的）
    card_start = html.rfind('<div class="card">', 0, idx)
    if card_start == -1:
        return False
    # card 的结束 = 下一个 '</div>\n' 之后的 '\n<!--' 或下一个 card
    seg = html[card_start:next_seg]
    # 该 seg 里找第一个 '</div>' 后跟 '</div>'（card 两层：formula-wrap 内层 div + card 外层 div）
    # 简单方式：从 idx 向后找 '</div>\n</div>' 或 '</div>\n\n'
    m = re.search(r"</div>\s*</div>\s*(?=\n|$)", seg)
    if m:
        inject_at = card_start + m.end()
    else:
        inject_at = card_start + seg.find("</div>", idx) + 6
    html = html[:inject_at] + "\n" + block + html[inject_at:]
    path.write_text(html, encoding="utf-8")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-only", action="store_true", help="只更新首页")
    ap.add_argument("--pages", type=str, default="", help="逗号分隔的页面（只注入这些）")
    ap.add_argument("--limit", type=int, default=0, help="注入前 N 页（按字母序，0=全部）")
    args = ap.parse_args()

    if not args.index_only:
        # 收集页面
        pages = sorted(f.stem.replace("factor_", "") for f in FACTORS_DIR.glob("factor_*.html"))
        if args.pages:
            want = [p.strip() for p in args.pages.split(",") if p.strip()]
            pages = [p for p in pages if p in want]
        elif args.limit and args.limit > 0:
            pages = pages[:args.limit]
        done = 0
        for i, page in enumerate(pages):
            try:
                if inject_steps(page):
                    done += 1
            except Exception as exc:
                print(f"  [ERR] {page}: {str(exc)[:120]}", flush=True)
            if (i + 1) % 100 == 0 or (i + 1) == len(pages):
                print(f"  {i+1}/{len(pages)} 注入{done}", flush=True)
        print(f"[steps] 详情页注入完成: {done}/{len(pages)}")

    rebuild_index()
    print("[index] 首页重建完成")


if __name__ == "__main__":
    main()
