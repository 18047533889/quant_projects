#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修正 27 个 _flipped 变体详情页的"已翻转"标记。

这些因子旧版(1699天) rankic<0 被标 flipped，但新重测(1854天到2026-08) rankic>0，
方向转正。详情页的"已翻转"标记基于文件名误标"是"，改为实际判定：
  - Mean RankIC < 0 → 已翻转（是）
  - Mean RankIC > 0 → 否（重测后方向转正，无需翻转）
"""
import re, glob, os, json
from pathlib import Path

FACTORS_DIR = Path("/home/sunhaiwei/quant_projects/factor_engine/docs/reports/2026-08-23/factors")
LQTP = json.load(open("/home/sunhaiwei/factor_delivery_converted/formula_lqtp.json"))


def fix_flip_label(name: str) -> bool:
    html_path = FACTORS_DIR / f"factor_{name}.html"
    if not html_path.exists():
        return False
    html = html_path.read_text(encoding="utf-8")
    # 从 aacd 记录取 flipped（实际判定）
    rec = LQTP.get(name)
    actually_flipped = bool(rec and rec.get("flipped")) if rec else ("_flipped" in name)
    if actually_flipped:
        return False  # 本来就标翻转，无需改
    # 替换"已翻转"字段 → 否（重测后转正）
    old = '<tr><td>已翻转</td><td>是 <span class="badge badge-yellow">IC&lt;0 时取负调正</span></td></tr>'
    new = '<tr><td>已翻转</td><td>否 <span class="badge badge-green" style="background:#dcfce7;color:#166534">重测后方向转正，无需翻转</span></td></tr>'
    if old in html:
        html = html.replace(old, new)
        html_path.write_text(html, encoding="utf-8")
        return True
    # 备用匹配
    m = re.search(r'<tr><td>已翻转</td><td>(.*?)</td></tr>', html)
    if m and '否' not in m.group(1):
        html = re.sub(
            r'<tr><td>已翻转</td><td>.*?</td></tr>',
            '<tr><td>已翻转</td><td>否 <span class="badge badge-green" style="background:#dcfce7;color:#166534">重测后方向转正，无需翻转</span></td></tr>',
            html, count=1)
        html_path.write_text(html, encoding="utf-8")
        return True
    return False


def main():
    flipped_pages = sorted([p.stem.replace("factor_", "") for p in FACTORS_DIR.glob("factor_*_flipped.html")])
    fixed = 0
    for name in flipped_pages:
        rec = LQTP.get(name)
        if rec and not rec.get("flipped"):
            if fix_flip_label(name):
                fixed += 1
                print(f"  [FIX] {name}: 标为未翻转（重测后转正）")
    print(f"[fliplabel] 修正 {fixed} 个页面")


if __name__ == "__main__":
    main()
