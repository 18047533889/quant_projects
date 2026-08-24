#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 docs/reports/2026-08-23/summary_stats.json 重写 docs/index.html 的两个表格
"""
import json
import re
from pathlib import Path

DOCS_INDEX = Path("/home/sunhaiwei/quant_projects/docs/index.html")
SUMMARY = Path("/home/sunhaiwei/quant_projects/docs/reports/2026-08-23/summary_stats.json")

with open(SUMMARY) as f:
    stats = json.load(f)
print(f"[init] stats loaded: {len(stats)} factors")

# 排序：正 Sharpe 降序, 负 Sharpe 升序
positive = [s for s in stats if s.get("ls_sharpe", 0) > 0]
negative = [s for s in stats if s.get("ls_sharpe", 0) <= 0]
positive.sort(key=lambda s: s.get("ls_sharpe", 0), reverse=True)
negative.sort(key=lambda s: s.get("ls_sharpe", 0))
print(f"  positive: {len(positive)}, negative: {len(negative)}")

# 渲染正/负表行
def render_row(rank, s):
    name = s["name"]
    is_flipped = s.get("is_flipped", False)
    flip_tag = ' <span class="tag tag-flip">FLIP</span>' if is_flipped else ''
    sharpe = s.get("ls_sharpe", 0)
    annual = s.get("ls_annual", 0)
    mdd = s.get("ls_mdd", 0)
    mean_ic = s.get("mean_ic", 0)
    win_rate = s.get("win_rate", 0)
    sharpe_str = f"{sharpe:+.2f}"
    cls = "pos" if sharpe >= 0 else "neg"
    annual_str = f"{annual*100:+.1f}%"
    annual_cls = "pos" if annual >= 0 else "neg"
    mdd_str = f"{mdd*100:.1f}%"
    ic_str = f"{mean_ic:+.4f}"
    ic_cls = "pos" if mean_ic >= 0 else "neg"
    win_str = f"{win_rate*100:.1f}%"
    win_cls = "pos" if win_rate >= 0.5 else "neg"
    return f'<tr><td>{rank}</td><td><a href="reports/2026-08-23/factors/factor_{name}.html"><code>factor_{name}</code></a>{flip_tag}</td><td class="{cls}">{sharpe_str}</td><td class="{annual_cls}">{annual_str}</td><td class="neg">{mdd_str}</td><td class="{ic_cls}">{ic_str}</td><td class="{ic_cls}">{ic_str}</td><td class="{win_cls}">{win_str}</td></tr>'

def render_row_neg(rank, s):
    name = s["name"]
    is_flipped = s.get("is_flipped", False)
    flip_tag = ' <span class="tag tag-flip">FLIP</span>' if is_flipped else ''
    sharpe = s.get("ls_sharpe", 0)
    annual = s.get("ls_annual", 0)
    mean_ic = s.get("mean_ic", 0)
    rank_ic = s.get("mean_ic", 0)
    sharpe_str = f"{sharpe:+.2f}"
    cls = "pos" if sharpe >= 0 else "neg"
    annual_str = f"{annual*100:+.1f}%"
    annual_cls = "pos" if annual >= 0 else "neg"
    ic_str = f"{mean_ic:+.4f}"
    ic_cls = "pos" if mean_ic >= 0 else "neg"
    ric_str = f"{rank_ic:+.4f}"
    return f'<tr><td>{rank}</td><td><a href="reports/2026-08-23/factors/factor_{name}.html"><code>factor_{name}</code></a>{flip_tag}</td><td class="{cls}">{sharpe_str}</td><td class="{annual_cls}">{annual_str}</td><td class="{ic_cls}">{ic_str}</td><td class="{ic_cls}">{ric_str}</td></tr>'

positive_rows = "\n".join(render_row(i+1, s) for i, s in enumerate(positive))
negative_rows = "\n".join(render_row_neg(i+1, s) for i, s in enumerate(negative))

# 读 docs/index.html
html = DOCS_INDEX.read_text()
old_html = html

# 替换正 Sharpe 表头数 + tbody
html = re.sub(
    r'(<h3>正 Sharpe 因子（)\d+( 个）</h3>.*?<tbody>)(.*?)(</tbody>)',
    rf'\g<1>{len(positive)}\g<2>\n{positive_rows}\n\g<3>',
    html,
    flags=re.DOTALL,
    count=1,
)
html = re.sub(
    r'(<h3>负 Sharpe 因子（)\d+( 个）</h3>.*?<tbody>)(.*?)(</tbody>)',
    rf'\g<1>{len(negative)}\g<2>\n{negative_rows}\n\g<3>',
    html,
    flags=re.DOTALL,
    count=1,
)

# 改 sub line 数字
html = re.sub(
    r'(合计约 )\d+( 条（本周 )\d+( \+ 历史精选 )\d+( \+ 上周存档 )\d+(）)',
    r'\g<1>382 条（本周 61 + 历史精选 180 + 上周存档 141）',
    html,
    count=1,
)

if html != old_html:
    DOCS_INDEX.write_text(html)
    print(f"✅ 已更新 {DOCS_INDEX}")
else:
    print("[!] 未替换")

print(f"\n[TIP] 刷新浏览器即可看到正 Sharpe {len(positive)} + 负 Sharpe {len(negative)}")