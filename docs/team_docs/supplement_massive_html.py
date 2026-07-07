#!/usr/bin/env python3
"""Supplement HTML checklist with Massive原始数据报告 §4.0.x + 附录 H/I."""
from __future__ import annotations

import re
from pathlib import Path

from expand_massive_html import (
    HTML,
    extract_section,
    inline,
    md_to_html_block,
    wrap_section,
)

BASE = Path(__file__).resolve().parent
REPORT = BASE / "Massive原始数据报告.md"


def strip_leading_h(md_block: str, level: str = "2", also_h3: bool = False) -> str:
    html = md_to_html_block(md_block)
    html = re.sub(rf"^<h{level}>[^<]*</h{level}>\s*", "", html, count=1)
    if also_h3:
        html = re.sub(r"^<h3>[^<]*</h3>\s*", "", html, count=1)
    return html


def build_section(sec_id: str, title: str, md_block: str, intro: str = "") -> str:
    body = strip_leading_h(md_block, also_h3=True)
    intro_html = f'<p class="section-desc">{intro}</p>\n' if intro else ""
    return wrap_section(
        sec_id,
        title,
        intro_html + body,
        skip_first_h2=False,
    )


def insert_after_section(html: str, after_id: str, new_block: str) -> str:
    pat = rf'(    <section id="{after_id}">.*?</section>\s*)'
    m = re.search(pat, html, flags=re.DOTALL)
    if not m:
        raise SystemExit(f"section not found: {after_id}")
    end = m.end()
    return html[:end] + "\n" + new_block + "\n" + html[end:]


def remove_section(html: str, sec_id: str) -> str:
    return re.sub(
        rf'    <section id="{sec_id}">.*?</section>\s*\n*',
        "",
        html,
        count=1,
        flags=re.DOTALL,
    )


def update_toc(html: str) -> str:
    old = """        <li><a href="#sec-ok">1. 当前可用数据</a></li>
        <li><a href="#sec-p0">2. P0 问题与补数</a></li>"""
    new = """        <li><a href="#sec-ok">1. 当前可用数据</a></li>
        <li><a href="#sec-empirical">1b. 复权实证（§4.0）</a></li>
        <li><a href="#sec-p0">2. P0 问题与补数</a></li>
        <li><a href="#sec-quality">2b. 缺失·截断·日分钟（§4.0.1–3 + H）</a></li>"""
    html = html.replace(old, new)

    old2 = """        <li><a href="#sec-accept">15. 验收标准</a></li>
      </ul>"""
    new2 = """        <li><a href="#sec-accept">15. 验收标准</a></li>
        <li><a href="#sec-cheatsheet">16. 一页纸速查（附录 I）</a></li>
      </ul>"""
    return html.replace(old2, new2)


def main():
    report = REPORT.read_text(encoding="utf-8")
    html = HTML.read_text(encoding="utf-8")

    s40 = extract_section(report, "### 4.0 量价", "### 4.0.1")
    s401_41 = extract_section(report, "### 4.0.1 缺失", "### 4.1 行情")
    h = extract_section(report, "## 附录 H", "## 附录 I")
    i = extract_section(report, "## 附录 I", "## 文档修订记录")

    empirical = build_section(
        "sec-empirical",
        "1b. 量价复权实证（数据报告 §4.0）",
        s40,
        "本地两套日线复权状态不同，<strong>禁止混用</strong>。下文 NVDA/TSLA/KO 为磁盘实测。",
    )

    quality_body = strip_leading_h(s401_41, also_h3=True)
    h_body = strip_leading_h(h, "2", also_h3=True)
    quality = wrap_section(
        "sec-quality",
        "2b. 缺失值·分页截断·分钟/日 K（§4.0.1–4.0.3 + 附录 H）",
        '<p class="section-desc">来自 <a href="Massive原始数据报告.md">Massive原始数据报告.md</a> 实证章节。</p>\n'
        + quality_body
        + '<h3>附录 H：REST 分页截断完整清单（需重下）</h3>\n'
        + h_body,
        skip_first_h2=False,
    )

    cheatsheet = build_section(
        "sec-cheatsheet",
        "16. 一页纸速查卡（附录 I）",
        i,
        "路径、下载命令、因子 checklist、数据集可用性一览——可打印贴墙。",
    )

    for sid in ("sec-empirical", "sec-quality", "sec-cheatsheet"):
        html = remove_section(html, sid)

    html = insert_after_section(html, "sec-ok", empirical)
    html = insert_after_section(html, "sec-p0", quality)

    if '<footer>' in html:
        html = html.replace("<footer>", cheatsheet + "\n\n    <footer>", 1)
    else:
        html += cheatsheet

    html = update_toc(html)
    HTML.write_text(html, encoding="utf-8")
    print("supplemented", HTML, "lines:", len(html.splitlines()))


if __name__ == "__main__":
    main()
