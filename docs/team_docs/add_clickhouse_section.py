#!/usr/bin/env python3
"""Insert ClickHouse section into Massive HTML checklist."""
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
HTML = BASE / "Massive数据治理与改进行动清单.html"
CH_SECTION = BASE / "_clickhouse_section.html"


def main():
    ch = CH_SECTION.read_text(encoding="utf-8")
    html = HTML.read_text(encoding="utf-8")

    # remove old section if re-run
    html = re.sub(
        r'\s*<!-- 11 ClickHouse -->.*?<section id="sec-clickhouse">.*?</section>\s*',
        "\n",
        html,
        count=1,
        flags=re.DOTALL,
    )

    marker = '    <section id="sec-gemini">'
    if marker not in html:
        marker = '    <!-- 11 Gemini -->'
    if marker not in html:
        raise SystemExit("sec-gemini marker not found")
    html = html.replace(marker, ch + "\n" + marker, 1)

    # TOC
    toc_old = '        <li><a href="#sec-preproc">10. Layer 2 预处理索引</a></li>\n        <li><a href="#sec-gemini">11. 外部建议勘误</a></li>'
    toc_new = """        <li><a href="#sec-preproc">10. Layer 2 预处理索引</a></li>
        <li><a href="#sec-clickhouse">11. ClickHouse 存储与读取</a></li>
        <li><a href="#sec-gemini">12. 外部建议勘误</a></li>"""
    html = html.replace(toc_old, toc_new)

    # Renumber TOC 12-16 -> 13-17 if full toc exists
    renums = [
        ("12. cleaned", "13. cleaned"),
        ("13. 任务总表", "14. 任务总表"),
        ("14. 路线图", "15. 路线图"),
        ("15. 验收标准", "16. 验收标准"),
        ("16. 一页纸速查", "17. 一页纸速查"),
    ]
    for a, b in renums:
        html = html.replace(f">{a}</a>", f">{b}</a>")

    # Section headers renumber
    html = html.replace(
        '<section id="sec-gemini">\n      <h2>11. 外部建议',
        '<section id="sec-gemini">\n      <h2>12. 外部建议',
    )
    html = html.replace(
        '<section id="sec-cleaned">\n      <h2>12. cleaned',
        '<section id="sec-cleaned">\n      <h2>13. cleaned',
    )
    html = html.replace(
        '<section id="sec-tasks">\n      <h2>13. 任务总表',
        '<section id="sec-tasks">\n      <h2>14. 任务总表',
    )
    html = html.replace(
        '<section id="sec-roadmap">\n      <h2>14. 四阶段',
        '<section id="sec-roadmap">\n      <h2>15. 四阶段',
    )
    html = html.replace(
        '<section id="sec-accept">\n      <h2>15. 验收标准',
        '<section id="sec-accept">\n      <h2>16. 验收标准',
    )
    html = html.replace(
        '<section id="sec-cheatsheet">\n      <h2>16. 一页纸',
        '<section id="sec-cheatsheet">\n      <h2>17. 一页纸',
    )
    for a, b in [
        ("15.1 数据补全", "16.1 数据补全"),
        ("15.2 工程改造", "16.2 工程改造"),
        ("15.3 因子可用性", "16.3 因子可用性"),
        ("15.4 一键截断", "16.4 一键截断"),
    ]:
        html = html.replace(f"<h3>{a}", f"<h3>{b}")

    # arch box
    html = html.replace(
        "Parquet 物化管道（首选）→ 可选导入 ClickHouse",
        "Parquet 物化（真相源）→ <strong>ClickHouse 热读</strong>（详见 §11）",
    )

    # Add CH tasks to task table if missing
    if "CH-001" not in html:
        row = """          <tr><td>CH-001</td><td><span class="badge badge-p0">P0</span></td><td>qs_massive DDL + 建表</td><td class="owner-cell"></td><td class="status-cell">待办</td><td></td></tr>
          <tr><td>CH-002</td><td><span class="badge badge-p0">P0</span></td><td>Parquet → CH 批量导入</td><td class="owner-cell"></td><td class="status-cell">待办</td><td></td></tr>
          <tr><td>CH-003</td><td><span class="badge badge-p1">P1</span></td><td>CH T+1 增量 + meta_load_batch</td><td class="owner-cell"></td><td class="status-cell">待办</td><td></td></tr>
          <tr><td>CH-004</td><td><span class="badge badge-p1">P1</span></td><td>data_access CH 只读适配器</td><td class="owner-cell"></td><td class="status-cell">待办</td><td></td></tr>
"""
        html = html.replace(
            "<tr><td>PREPROC-007</td>",
            row + "          <tr><td>PREPROC-007</td>",
            1,
        )

    # FEAT table
    if "FEAT-009" not in html and "FEAT-008</code>" in html:
        feat = """          <tr>
            <td><code>FEAT-009</code></td>
            <td><strong>ClickHouse 导入与查询层</strong></td>
            <td><span class="badge badge-p0">P0</span></td>
            <td>CH-001~004；panel_daily 与 Parquet 双写一致</td>
            <td>PREPROC-007</td>
          </tr>
"""
        html = html.replace(
            "<td><code>FEAT-008</code></td>",
            "<td><code>FEAT-008</code></td>",
        )
        # insert after FEAT-008 row - find closing tr after FEAT-008
        idx = html.find("FEAT-008")
        if idx > 0:
            end_tr = html.find("</tr>", idx) + 5
            html = html[:end_tr] + feat + html[end_tr:]

    # Roadmap phase 3
    if "CH-002" not in html.split("Phase 3")[1][:500] if "Phase 3" in html else "":
        html = html.replace(
            "PREPROC-002~007 · FEAT-001~004",
            "PREPROC-002~007 · FEAT-001~004 · CH-001~003",
        )

    if "qs_massive.panel_daily" not in html:
        html = html.replace(
            "<li>评审会通过：P0 任务全部关闭或已接受风险</li>",
            "<li>CH <code>qs_massive.panel_daily</code> 与物化 Parquet 同分区行数一致，可跑通截面 SQL</li>\n        <li>评审会通过：P0 任务全部关闭或已接受风险</li>",
        )

    HTML.write_text(html, encoding="utf-8")
    print("OK", HTML, "lines:", len(html.splitlines()))


if __name__ == "__main__":
    main()
