#!/usr/bin/env python3
"""Remove duplicate nested <details> in ACL/MM entry blocks."""
import re
from pathlib import Path

HTML = Path(__file__).with_name("Massive数据治理与改进行动清单.html")
html = HTML.read_text(encoding="utf-8")

# 1) Empty outer shell: (ACL|MM) 条目 with no body
html = re.sub(
    r'<details class="fold">\s*'
    r'<summary><strong>([^<]+)</strong>（(?:ACL|MM) 条目）</summary>\s*'
    r'<div class="fold-body">\s*'
    r'</div>\s*'
    r'</details>\s*',
    "",
    html,
    flags=re.DOTALL,
)

# 2) Double fold: outer label + inner … summary + redundant h3
html = re.sub(
    r'<details class="fold">\s*'
    r'<summary><strong>([^<]+)</strong>（(?:ACL|MM) 条目）</summary>\s*'
    r'<div class="fold-body">\s*'
    r'<details class="fold">\s*'
    r'<summary>[^<]*…</summary>\s*'
    r'<div class="fold-body">\s*'
    r'<h3>[^<]*</h3>\s*',
    r'<details class="fold">\n        <summary><strong>\1</strong></summary>\n        <div class="fold-body">\n        ',
    html,
    flags=re.DOTALL,
)

# 3) Inner-only duplicate: … summary + redundant h3 → single strong summary (h3 may contain inline tags)
html = re.sub(
    r'<summary>[^<]+…</summary>\s*'
    r'<div class="fold-body">\s*'
    r'<h3>((?:[^<]|<(?!/h3>))+)</h3>\s*',
    r'<summary><strong>\1</strong></summary>\n        <div class="fold-body">\n        ',
    html,
    flags=re.DOTALL,
)

# 4) Stray extra closers left from removed outer wrappers (at most 2 per section tail)
for _ in range(4):
    html = re.sub(
        r'(</details>)\s*</div>\s*</details>(\s*</section>)',
        r'\1\2',
        html,
        count=1,
    )

# 4) Normalize summary suffix text
html = html.replace("（ACL 条目）", "")
html = html.replace("（MM 条目）", "")

HTML.write_text(html, encoding="utf-8")
remaining = len(re.findall(r"（(?:ACL|MM) 条目）", html))
double = len(re.findall(r'…</summary>\s*<div class="fold-body">\s*<h3>', html, re.DOTALL))
print(f"done · remaining entry labels: {remaining} · inner h3 duplicates: {double}")
