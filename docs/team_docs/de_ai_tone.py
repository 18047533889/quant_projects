#!/usr/bin/env python3
"""Strip tutorial / AI-style phrasing from the handbook HTML."""
from pathlib import Path

PATH = Path(__file__).with_name("Massive数据治理与改进行动清单.html")

# Order matters: longer / more specific first.
REPLACEMENTS = [
    ("0.2 白话术语表（看不懂缩写先扫这里）", "0.2 常用术语表"),
    ("展开表：0.2 白话术语表（看不懂缩写先扫这里）", "展开表：0.2 常用术语表"),
    ("<span class=\"nav-label\">白话术语表</span>", "<span class=\"nav-label\">常用术语表</span>"),
    ("<span class=\"nav-label\">任务总表（先看这里）</span>", "<span class=\"nav-label\">任务总表</span>"),
    ("<th>白话意思</th>", "<th>含义</th>"),
    ("<strong>表格怎么读（三类标注）：</strong>", "<strong>表格标注：</strong>"),
    ("<strong>怎么读「落地状态」：</strong>", "<strong>状态列：</strong>"),
    ("<th>落地状态</th>", "<th>当前状态</th>"),
    ("同款的<strong>短释义</strong>", "同款的<strong>括号说明</strong>"),
    ("概念词见", "术语见"),
    ("；概念词见", "；术语见"),
    ("（概念词释义）", ""),
    ("但<strong>概念</strong>尽量用下面说法理解。", "代号含义对照下表。"),
    ("下文<strong>全部用表格</strong>展示路径、分区、频率、列与样例行，无需打开 Parquet 文件。",
     "路径、分区、频率、列与样例行均列于下表，不用单独打开 Parquet。"),
    ("术语不懂先看", "术语见"),
    ("§0.2 白话术语表", "§0.2 常用术语表"),
    ("<strong>每条规则怎么读：</strong>", "<strong>规则表列说明：</strong>"),
    ("<strong>本章小结：</strong>", "<strong>小结：</strong>"),
    ("<strong>怎么读本节：</strong>", "<strong>列说明：</strong>"),
    ("<a href=\"#sec-acl\">§ACL（本章）</a>", "<a href=\"#sec-acl\">§ACL</a>"),
    (" 本章讲怎么", " 说明怎么"),
    ("§ACL 管", "§ACL 处理"),
    ("；本章管", "；本节处理"),
    ("，本章不重复。", "，本节不重复。"),
    ("本章是", "本节是"),
    ("任务总表（评审会分工 · 建议先看本章）", "任务总表（评审会分工）"),
    ("本章 = 全手册的<strong>待办清单</strong>。你可以<strong>只读 §14</strong> 知道要做什么，再点「详见」跳正文。",
     "全手册待办汇总于此；任务说明见下表，实现细节点「详见」进正文。"),
    ("<strong>怎么读（3 步）：</strong>", "<strong>阅读顺序：</strong>"),
    (" — 知道大方向（先补数 → 再建宽表 → 再极端场景）", "（补数 → 宽表 → 入模检查）"),
    (" — 每条有白话「要做什么」；P0 的 A/B 组默认展开", "；P0 的 A/B 组默认展开"),
    ("14.0 分组速览（一张表看懂全貌）", "14.0 分组概览"),
    ("要做什么（白话）", "任务说明"),
    (" · 点击展开/收起）", "）"),
    (" · 点击展开）", "）"),
    ("（速查 · 点击展开）", "（速查）"),
    ("<th>我想…</th><th>去看</th>", "<th>问题</th><th>章节</th>"),
    ("<tr><td>看不懂 PiT、物化、ACL 等词</td>", "<tr><td>PiT、物化、ACL 等术语</td>"),
    ("<strong>不必再建「最终版 Task Matrix」：</strong><a href=\"#sec-tasks\">§14 任务总表</a> + §12a–12e 登记表即为评审分发源；ENG-011 / CH-007 已在册，无需重复追加 ID。",
     "评审分发以 <a href=\"#sec-tasks\">§14 任务总表</a> 与 §12a–12e 登记表为准；ENG-011、CH-007 已登记。"),
    ("高度重复 — <strong>无需再写长文</strong>，评审会直接查下表索引。",
     "与 v3.18 重复部分不展开，见下表索引。"),
    ("一并构成入模前定稿清单。", "合并为入模前定稿清单。"),
    ("一并构成<strong>入模前定稿终审清单</strong>。", "合并为<strong>入模前定稿终审清单</strong>。"),
    ("十五、高阶工程索引", "十五、工程规范索引"),
    ("十六、v3.25 工程补全索引（复权 · 跑批 · 因子生命周期）", "十六、v3.25 增量索引（复权 · 跑批 · 因子生命周期）"),
    ("v3.31", "v3.32"),
]

def main():
    text = PATH.read_text(encoding="utf-8")
    before = text
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    if text == before:
        print("no changes")
    else:
        PATH.write_text(text, encoding="utf-8")
        print("de-ai tone applied → v3.32")

if __name__ == "__main__":
    main()
