#!/usr/bin/env python3
"""用 summary_stats.json 整体重写 docs/index.html 的因子表 (61 因子)"""
import json, re
from pathlib import Path
DOCS_INDEX = Path("/home/sunhaiwei/quant_projects/docs/index.html")
SUMMARY = Path("/home/sunhaiwei/quant_projects/docs/reports/2026-08-23/summary_stats.json")

with open(SUMMARY) as f:
    stats = json.load(f)
print(f"[init] stats loaded: {len(stats)} factors")

positive = [s for s in stats if s.get("ls_sharpe", 0) > 0]
negative = [s for s in stats if s.get("ls_sharpe", 0) <= 0]
positive.sort(key=lambda s: s.get("ls_sharpe", 0), reverse=True)
negative.sort(key=lambda s: s.get("ls_sharpe", 0))
print(f"  positive: {len(positive)}, negative: {len(negative)}")

def render_row(idx, s):
    name = s["name"]
    is_flipped = s.get("is_flipped", False)
    flip_tag = ' <span class="tag tag-flip">FLIP</span>' if is_flipped else ''
    sharpe = s.get("ls_sharpe", 0)
    annual = s.get("ls_annual", 0)
    mean_ic = s.get("mean_ic", 0)
    ic_ir = s.get("ic_ir", 0)
    win_rate = s.get("win_rate", 0)
    sharpe_str = f"{sharpe:+.2f}"
    cls = "pos" if sharpe >= 0 else "neg"
    annual_str = f"{annual*100:+.1f}%"
    annual_cls = "pos" if annual >= 0 else "neg"
    ic_str = f"{mean_ic:+.4f}"
    ic_cls = "pos" if mean_ic >= 0 else "neg"
    ir_str = f"{ic_ir:+.2f}"
    ir_cls = "pos" if ic_ir >= 0 else "neg"
    win_str = f"{win_rate*100:.1f}%"
    win_cls = "pos" if win_rate >= 0.5 else "neg"
    return f'<tr><td>{idx}</td><td><a href="reports/2026-08-23/factors/factor_{name}.html"><code>factor_{name}</code></a>{flip_tag}</td><td class="{ic_cls}">{ic_str}</td><td class="{ir_cls}">{ir_str}</td><td class="{win_cls}">{win_str}</td><td class="{cls}">{sharpe_str}</td><td class="{annual_cls}">{annual_str}</td></tr>'

pos_rows = "\n".join(render_row(i+1, s) for i, s in enumerate(positive))
neg_rows = "\n".join(render_row(i+1, s) for i, s in enumerate(negative))

html = DOCS_INDEX.read_text()

# 替换全部 61 个因子 表的 tbody
# 找 <h2>🏆 全部 61 个因子 (按 LS Sharpe 排序)</h2> ... <tbody>...</tbody>
m = re.search(r'(<h2>🏆 全部 61 个因子.*?</thead>\s*<tbody>)(.*?)(</tbody>)', html, re.DOTALL)
if not m:
    print("❌ 找不到 61 因子表头")
else:
    new_block = m.group(1) + "\n" + pos_rows + "\n" + neg_rows + "\n" + m.group(3)
    html = html[:m.start()] + new_block + html[m.end():]
    DOCS_INDEX.write_text(html)
    print(f"✅ 已更新 {DOCS_INDEX} (正 {len(positive)} + 负 {len(negative)})")
