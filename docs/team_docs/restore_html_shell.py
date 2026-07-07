#!/usr/bin/env python3
"""Restore DOCTYPE/head/header/sec-arch when catalog build stripped document shell."""
import re
from pathlib import Path

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"
BEAUTIFY = Path(__file__).resolve().parent / "beautify_massive_html.py"

ARCH = """
    <section id="sec-arch">
      <h2><span class="sec-badge">0</span> 三层架构与文档边界</h2>
      <p class="section-desc">
        <strong>团队唯一正式交付：</strong>本 HTML（v3.6）。姊妹 MD 仅作历史对照；新规范请只改本文件。
      </p>
      <table class="table-schema">
        <thead><tr><th>层</th><th>路径</th><th>体量</th><th>说明</th></tr></thead>
        <tbody>
          <tr><td>Layer 0</td><td><code>raw_massive_data/</code></td><td>~10 TB</td><td>供应商原始；SIP 不复权；REST 部分已复权</td></tr>
          <tr><td>Layer 1</td><td><code>cleaned_massive_data/</code></td><td>~243 GB</td><td>ticker + align_time；<strong>不改 OHLCV</strong></td></tr>
          <tr><td>Layer 1.5～2</td><td><code>materialized_panel/</code></td><td>见 <a href="#sec-engineering">§10c</a></td><td>复权、Universe、PiT、宽表、CH</td></tr>
        </tbody>
      </table>
      <h3>0.3 因子入模前 / 入模后边界</h3>
      <table class="table-schema">
        <thead><tr><th>阶段</th><th>包含</th><th>不包含</th></tr></thead>
        <tbody>
          <tr><td>Layer 1 清洗</td><td>ticker、align_time、explode</td><td>复权、PiT、Panel</td></tr>
          <tr><td>Layer 2</td><td>复权、Universe、PiT、panel、CH</td><td>MAD、Barra、IC/IR</td></tr>
          <tr><td>因子评估层</td><td>去极值、中性化、IC、入库</td><td>改 Massive 原始 OHLCV</td></tr>
        </tbody>
      </table>
      <div class="box-info">
        <strong>执行载体：</strong>清洗 <code>massive_cleaning_framework</code>；物化 <code>build_panel_daily</code> / CH；
        算因子 <code>factor_engine</code>；评估 <code>factor_evaluation</code>。
      </div>
    </section>
"""

READMAP = """
    <section id="sec-readmap">
      <h2><span class="sec-badge">导读</span> 读者导读（按角色）</h2>
      <table class="table-schema">
        <thead><tr><th>角色</th><th>必读章节</th><th>目标</th></tr></thead>
        <tbody>
          <tr><td>量化研究员</td><td>
            <a href="#sec-data-catalog"><strong>§数据手册</strong></a>
            <a href="#sec-operator-adj"><strong>§3.2 分钟复权</strong></a>
            <a href="#sec-before-factor"><strong>★入模前</strong></a>
            <a href="#sec-engine-checklist"><strong>§17 闸门</strong></a>
            <a href="#sec-empirical">§1b</a> <a href="#sec-p1">§3</a> <a href="#sec-scenarios">§3b</a>
          </td><td>价源、PiT、能否做因子</td></tr>
          <tr><td>数据工程</td><td><a href="#sec-p0">§2</a> <a href="#sec-sre">§2c</a> <a href="#sec-download">§4</a> <a href="#sec-quality">§2b</a></td><td>P0 补数、三层门禁</td></tr>
          <tr><td>平台 / ETL</td><td><a href="#sec-pipeline">§10b</a> <a href="#sec-clickhouse">§11</a> <a href="#sec-engineering">§10c</a></td><td>物化 Panel、CH</td></tr>
          <tr><td>评审 / PM</td><td><a href="#sec-tasks">§14</a> <a href="#sec-roadmap">§15</a> <a href="#sec-accept">§16</a></td><td>分工与验收</td></tr>
        </tbody>
      </table>
      <div class="box-warn">
        <strong>P0 前全局禁令：</strong>含 2024 截断基本面/做空/分红 — 训练 <code>end_date ≤ 2023-12-31</code>；
        禁止 SIP+REST 混用；分钟算子须 <code>fact_bars_adjusted_minute</code>。
      </div>
    </section>
"""

HEADER = """
    <header class="header">
      <h1>Massive 数据治理与改进行动清单</h1>
      <p class="meta">版本 v3.6 单文件交付（MD v2.6 已并入）· 2026-06-03 · 扫描截止 2026-03-05 · <strong>仅以本 HTML 为准</strong></p>
      <div class="purpose">
        综合手册：26 源数据形态、P0/P1 红线、ATR 入模前流程、Deep Research 采纳项、任务分工与可折叠代码。
        打开左侧导航或文末 cheatsheet 跳转。
      </div>
      <div class="summary-grid">
        <div class="summary-card red">
          <h3>P0 阻断</h3>
          <div class="num">7+ 类</div>
          <div style="font-size:13px;color:var(--muted)">2024 财报/做空/dividends 截断</div>
        </div>
        <div class="summary-card amber">
          <h3>P1 误用风险</h3>
          <div class="num">5 项</div>
          <div style="font-size:13px;color:var(--muted)">复权混用、PiT、分钟≠日K、算子须复权</div>
        </div>
        <div class="summary-card green">
          <h3>生产锚点</h3>
          <div class="num">day_aggs</div>
          <div style="font-size:13px;color:var(--muted)">cleaned + Layer2 复权物化</div>
        </div>
      </div>
    </header>
"""


def load_style() -> str:
    import beautify_massive_html as b

    return b.NEW_STYLE.strip()


def main():
    html = HTML.read_text(encoding="utf-8")
    if html.lstrip().startswith("<!DOCTYPE"):
        print("shell OK, skip restore")
        return

    # strip accidental closing tags at start
    body = re.sub(r"^\s*(</(?:main|div|body|html)>)\s*", "", html, flags=re.I)
    # extract body: from first fold/catalog to before final closers
    m = re.search(r"(<details class=\"fold\" id=\"sec-head-samples\">.*)", body, re.DOTALL)
    if not m:
        m = re.search(r"(<section id=\"sec-data-catalog\">.*)", body, re.DOTALL)
    if not m:
        raise SystemExit("cannot find body start")
    body = m.group(1)
    body = re.sub(r"\s*</main>\s*</div>\s*</body>\s*</html>\s*$", "", body, flags=re.DOTALL)

    style = load_style()
    full = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Massive 数据治理与改进行动清单（综合手册）</title>
  <style>{style}</style>
</head>
<body>
  <div class="app-layout">
  <main class="main-content">
  <div class="wrap">
{HEADER.strip()}
{ARCH.strip()}
{READMAP.strip()}

{body.strip()}
  </div>
  </main>
  </div>
</body>
</html>
"""
    HTML.write_text(full, encoding="utf-8")
    print("restored shell, lines:", full.count("\n") + 1)

    import subprocess
    subprocess.run(["python3", str(BEAUTIFY)], check=True)


if __name__ == "__main__":
    main()
