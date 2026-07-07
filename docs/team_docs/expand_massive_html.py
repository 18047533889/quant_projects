#!/usr/bin/env python3
"""Inject full ATR/附录 C/D sections into Massive HTML checklist."""
from pathlib import Path

BASE = Path(__file__).resolve().parent
HTML = BASE / "Massive数据治理与改进行动清单.html"
MD = BASE / "美股原始数据预处理标准方案_因子入模前.md"


def md_table_to_html(lines: list[str]) -> str:
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return ""
    head = rows[0]
    body = rows[1:]
    out = ["<table>", "<thead><tr>"]
    for c in head:
        out.append(f"<th>{inline(c)}</th>")
    out.append("</tr></thead><tbody>")
    for r in body:
        out.append("<tr>")
        for c in r:
            out.append(f"<td>{inline(c)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def inline(s: str) -> str:
    import re

    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)

    def esc(t: str) -> str:
        return t.replace("&", "&amp;").replace("<", "&lt;")

    parts = re.split(r"(\*\*.+?\*\*)", s)
    out = []
    for p in parts:
        if p.startswith("**") and p.endswith("**"):
            out.append("<strong>" + esc(p[2:-2]) + "</strong>")
        else:
            out.append(esc(p))
    s = "".join(out)
    parts = []
    i = 0
    while "`" in s[i:]:
        a, s = s[: s.index("`", i)], s[s.index("`", i) + 1 :]
        if a:
            parts.append(a)
        if "`" not in s:
            parts.append(s)
            return "".join(parts)
        b, s = s[: s.index("`")], s[s.index("`") + 1 :]
        parts.append(f"<code>{b}</code>")
        i = 0
    parts.append(s)
    return "".join(parts)


def extract_section(md: str, start_h: str, end_h: str | None) -> str:
    i = md.find(start_h)
    if i < 0:
        raise SystemExit(f"missing {start_h}")
    j = len(md) if end_h is None else md.find(end_h, i + 1)
    return md[i:j]


def md_to_html_block(text: str) -> str:
    """Simple converter: headings, tables, lists, fenced code."""
    lines = text.splitlines()
    html = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            lang = line[3:].strip()
            buf = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i].replace("<", "&lt;"))
                i += 1
            i += 1
            if lang in ("mermaid", "text") and buf and buf[0].startswith("flowchart"):
                html.append('<div class="box-info"><em>流程图见标准方案 MD（mermaid）</em></div>')
                continue
            cls = "json-block" if lang == "json" else ""
            tag = f'<div class="{cls}">' if cls else "<pre>"
            end = "</div>" if cls else "</pre>"
            html.append(tag + "\n".join(buf) + end)
            continue
        if line.startswith("#### "):
            html.append(f"<h4>{inline(line[5:])}</h4>")
        elif line.startswith("### "):
            html.append(f"<h3>{inline(line[4:])}</h3>")
        elif line.startswith("## "):
            html.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("|"):
            tbl = []
            while i < len(lines) and lines[i].startswith("|"):
                tbl.append(lines[i])
                i += 1
            html.append(md_table_to_html(tbl))
            continue
        elif line.startswith("- "):
            html.append("<ul>")
            while i < len(lines) and lines[i].startswith("- "):
                html.append(f"<li>{inline(lines[i][2:])}</li>")
                i += 1
            html.append("</ul>")
            continue
        elif line.startswith("> "):
            html.append(f'<p class="section-desc">{inline(line[2:])}</p>')
        elif line.strip() == "---":
            pass
        elif line.strip():
            html.append(f"<p>{inline(line)}</p>")
        i += 1
    return "\n".join(html)


def wrap_section(sec_id: str, title: str, inner: str, skip_first_h2=True) -> str:
    body = inner
    while skip_first_h2 and body.lstrip().startswith("<h2>"):
        body = body.lstrip()
        body = body[body.index("</h2>") + 5 :].lstrip()
    return f"""    <section id="{sec_id}">
      <h2>{title}</h2>
{body}
    </section>
"""


def main():
    md = MD.read_text(encoding="utf-8")
    s2 = extract_section(md, "## 2. ATR", "## 3. Stage")
    c = extract_section(md, "## 附录 C", "## 附录 D")
    d = extract_section(md, "## 附录 D", "## 附录 A")

    import re

    atr_body = md_to_html_block(s2)
    atr_body = re.sub(r"^<h2>2\. ATR[^<]*</h2>\s*", "", atr_body, count=1)
    atr = wrap_section(
        "sec-atr",
        "6. ATR 工程规范（Align-time · Ticker · Resampling）",
        '<p class="section-desc">平台 <strong>ATR</strong> ≠ 技术指标 Average True Range；落地 Layer 1.5～2；<code>cleaned</code> 不可直读 QuantaAlpha。</p>\n'
        + atr_body,
        skip_first_h2=False,
    )
    fields_body = md_to_html_block(c)
    fields_body = re.sub(r"^<h2>附录 C[^<]*</h2>\s*", "", fields_body, count=1)
    fields = wrap_section(
        "sec-fields",
        "7. 基本面·新闻·申报 — 字段形式（附录 C 全文）",
        fields_body,
        skip_first_h2=False,
    )
    # C.6 as dedicated time section
    c6 = extract_section(md, "### C.6", "### C.7")
    time_sec = f"""    <section id="sec-time">
      <h2>8. 时间语义与 PiT（专章）</h2>
      <p class="section-desc">本库无 EDGAR acceptance-datetime；逻辑键优先 <code>trade_date</code>。</p>
{md_to_html_block(c6)}
      <div class="box-info"><strong>PiT：</strong>按 <code>(period_end, timeframe)</code> 分组取 filing_date ≤ cutoff 的最新版；禁 keep_latest 全历史。</div>
    </section>
"""
    align_body = md_to_html_block(d)
    align_body = re.sub(r"^<h2>附录 D[^<]*</h2>\s*", "", align_body, count=1)
    align = wrap_section(
        "sec-align",
        "9. 异构多源对齐实操（附录 D 全文）",
        align_body,
        skip_first_h2=False,
    )

    gemini_md = extract_section(md, "## 附录 A", "## 19. 修订")
    gemini_body = md_to_html_block(gemini_md)
    gemini_body = re.sub(r"^<h2>附录 A[^<]*</h2>\s*", "", gemini_body, count=1)
    gemini = wrap_section(
        "sec-gemini",
        "11. 外部建议（Gemini）逐项勘误（附录 A）",
        gemini_body,
        skip_first_h2=False,
    )

    frag = "\n\n".join([atr, fields, time_sec, align])

    html = HTML.read_text(encoding="utf-8")
    if "    <!-- 5b ATR -->" not in html:
        start = html.index('    <section id="sec-atr">')
        end = html.index('    <section id="sec-preproc">')
        html = html[:start] + frag + "\n\n" + html[end:]
    else:
        start = html.index("    <!-- 5b ATR -->")
        end = html.index("    <!-- 6. Preproc -->")
        html = html[:start] + frag + "\n\n" + html[end:]

    # preproc renumber
    html = html.replace(
        '<section id="sec-preproc">\n      <h2>6. 因子入模前预处理',
        '<section id="sec-preproc">\n      <h2>10. 因子入模前预处理（Layer 2 索引）',
    )
    html = html.replace("完整规范见 <strong>v2.2</strong>", "完整规范见 <strong>v2.4</strong>")
    for a, b in [
        ("<h3>6.1 常见反模式", "<h3>10.1 常见反模式"),
        ("<h3>6.2 按场景最小路径", "<h3>10.2 按场景最小路径"),
        ("<h3>6.3 Layer 2 上线检查", "<h3>10.5 Layer 2 上线检查"),
    ]:
        html = html.replace(a, b)

    marker = "    <!-- 7. cleaned -->"
    if marker in html:
        # replace existing gemini block if re-running
        import re as _re

        html = _re.sub(
            r"    <section id=\"sec-gemini\">.*?</section>\s*\n*",
            "",
            html,
            count=1,
            flags=_re.DOTALL,
        )
        html = html.replace(marker, gemini + "\n\n" + marker, 1)

    html = html.replace(
        '<section id="sec-cleaned">\n      <h2>7. cleaned',
        '<section id="sec-cleaned">\n      <h2>12. cleaned',
    )
    html = html.replace(
        '<section id="sec-tasks">\n      <h2>8. 任务总表',
        '<section id="sec-tasks">\n      <h2>13. 任务总表',
    )
    html = html.replace(
        '<section id="sec-roadmap">\n      <h2>9. 四阶段路线图',
        '<section id="sec-roadmap">\n      <h2>14. 四阶段路线图',
    )
    html = html.replace(
        '<section id="sec-accept">\n      <h2>10. 验收标准',
        '<section id="sec-accept">\n      <h2>15. 验收标准',
    )
    for a, b in [
        ("<h3>9.1 数据补全", "<h3>15.1 数据补全"),
        ("<h3>9.2 工程改造", "<h3>15.2 工程改造"),
        ("<h3>9.3 因子可用性", "<h3>15.3 因子可用性"),
        ("<h3>9.4 一键截断", "<h3>15.4 一键截断"),
    ]:
        html = html.replace(a, b)

    html = html.replace(
        "本清单：Massive数据治理与改进行动清单.html",
        "本清单：Massive数据治理与改进行动清单.html v2.0 综合版",
    )

    HTML.write_text(html, encoding="utf-8")
    print("Wrote", HTML, "lines:", len(html.splitlines()))


if __name__ == "__main__":
    main()
