#!/usr/bin/env python3
"""从 cleaned_operators 注册表生成《算子全览.md》。"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = FE_ROOT / "cleaned_operators" / "算子全览.md"

CATEGORY_ZH = {
    "elementwise_math": "元素级数学",
    "time_series": "时序滚动",
    "shift_diff_cum": "滞后 / 差分 / 累计",
    "cross_sectional": "截面变换",
    "group_neutralization": "分组中性化",
    "data_cleaning": "数据清洗",
    "statistics_regression": "统计与回归",
    "price_volume": "价量衍生",
    "technical_signal": "技术信号",
    "fundamental": "基本面",
    "intraday_microstructure": "日内微观结构",
    "other": "其他",
}

CATEGORY_INTRO = {
    "elementwise_math": "逐元素四则运算、比较、三角函数、矩阵与数值工具；两个序列按位置一一运算。",
    "time_series": "沿时间轴对每个标的单独滚动；窗口参数 `d` / `window` / `span` 均为 bar 根数。",
    "shift_diff_cum": "滞后、差分、累计统计；与 `ts_*` 滚动不同，部分为全样本累计。",
    "cross_sectional": "每个交易日对全市场横截面计算（rank、zscore 等）；与时序 rolling 勿混淆。",
    "group_neutralization": "按行业、市值等分组后在组内做 rank / 中性化 / 标准化。",
    "data_cleaning": "缺失值填充、缩尾、保护除法/对数等数值安全算子。",
    "statistics_regression": "相关、协方差、回归、分布统计量与假设检验。",
    "price_volume": "收益、波动、Beta、夏普、回撤等价量衍生指标。",
    "technical_signal": "经典技术分析指标（MACD/RSI/ADX 等）与条件信号（if_else/trade_when）。",
    "fundamental": "财报衍生：TTM、同比、季度化等；需基本面数据列。",
    "intraday_microstructure": "换手率、微观结构类；多数 catalog 条目仍为 stub。",
    "other": "未归入上述分类的注册名。",
}


def _load_registry():
    import sys

    if str(FE_ROOT) not in sys.path:
        sys.path.insert(0, str(FE_ROOT))
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry
    from api.operator_registry import build_dsl_allowlist

    load_all()
    allowlist = build_dsl_allowlist()
    return OperatorRegistry, allowlist


def _category_by_canonical() -> dict[str, str]:
    cat_by_canon: dict[str, str] = {}
    for py in (FE_ROOT / "cleaned_operators").glob("*.py"):
        if py.name.startswith("_"):
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        for block in re.findall(r"@register_operator\(([^)]*)\)", text):
            cm = re.search(r'canonical="([^"]+)"', block)
            bm = re.search(r'business_category="([^"]+)"', block)
            if cm and bm:
                cat_by_canon[cm.group(1)] = bm.group(1)
    return cat_by_canon


def _collect_canonical_entries(OperatorRegistry, allowlist: dict) -> dict[str, dict]:
    catalog = OperatorRegistry._catalog
    alias_to_canon = dict(OperatorRegistry._aliases)

    # dsl 名 -> canonical
    dsl_names_by_canon: dict[str, set[str]] = defaultdict(set)
    for dsl_name in allowlist:
        if dsl_name == "col":
            continue
        canon = alias_to_canon.get(dsl_name, dsl_name)
        dsl_names_by_canon[canon].add(dsl_name)

    cat_by_canon = _category_by_canonical()
    entries: dict[str, dict] = {}

    for canon, dsl_names in dsl_names_by_canon.items():
        info = catalog.get(canon, {})
        op = OperatorRegistry.get(canon)
        description = info.get("description", "")
        params = info.get("param_names", [])
        examples: list[str] = []
        if op is not None and hasattr(op, "metadata"):
            description = description or getattr(op.metadata, "description", "")
            params = params or list(getattr(op.metadata, "param_names", []) or [])
            examples = list(getattr(op.metadata, "examples", []) or [])

        entries[canon] = {
            "canonical": canon,
            "dsl_names": sorted(dsl_names, key=str.lower),
            "category": cat_by_canon.get(canon, info.get("business_category", "other")),
            "description": description or "（暂无描述）",
            "params": params,
            "examples": examples,
            "status": info.get("status", "implemented"),
        }
    return entries


def _render(entries: dict[str, dict], allowlist_count: int) -> str:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for e in entries.values():
        by_cat[e["category"]].append(e)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda x: x["canonical"].lower())

    lines: list[str] = [
        "# factor_engine 算子全览（cleaned_operators）",
        "",
        f"> 自动生成日期：{date.today().isoformat()}  ",
        f"> DSL 白名单：**{allowlist_count}** 个名字（含别名）；下文按 **{len(entries)}** 个规范算子分组。  ",
        "> 重新生成：`cd factor_engine && PYTHONPATH=. python3 scripts/generate_operators_guide.py`",
        "",
        "## 怎么用这份文档",
        "",
        "| 你想… | 做法 |",
        "|--------|------|",
        "| 写 manifest 公式 | 用下文 **DSL 可用名** 之一；须能通过 `parse_expr` |",
        "| 查能不能投递 | `PYTHONPATH=. python3 scripts/validate_delivery_formula.py \"你的公式\"` |",
        "| 查别名 | 每个算子下的 **DSL 可用名** 列表 |",
        "| 字段引用 | `close` / `open` / `high` / `low` / `volume` 等，或 `col(\"close\")`（Python API） |",
        "",
        "### 特殊：字段引用 `col`",
        "",
        "- **说明**：引用行情或特征列；manifest 字符串里通常直接写 `close`，不必写 `col(\"close\")`。",
        "- **示例**：`rank(ts_mean(close, 20))`",
        "",
        "### 时序 vs 截面（最易混）",
        "",
        "| 类型 | 代表算子 | 语义 |",
        "|------|----------|------|",
        "| 时序 | `ts_mean(x, 20)` | 每只股票自己的时间轴上滚 20 根 bar |",
        "| 截面 | `rank(x)` | 每个交易日全市场横截面排名 |",
        "| 分组 | `group_rank(x, sector)` | 每个交易日、每个行业组内排名 |",
        "",
        "---",
        "",
        "## 分类索引",
        "",
        "| 分类 | 规范算子数 | 说明 |",
        "|------|------------|------|",
    ]

    for cat in sorted(by_cat.keys(), key=lambda c: CATEGORY_ZH.get(c, c)):
        zh = CATEGORY_ZH.get(cat, cat)
        lines.append(f"| [{zh}](#{cat}) | {len(by_cat[cat])} | {CATEGORY_INTRO.get(cat, '')[:50]} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    for cat in sorted(by_cat.keys(), key=lambda c: CATEGORY_ZH.get(c, c)):
        zh = CATEGORY_ZH.get(cat, cat)
        lines.append(f"## {zh} {{#{cat}}}")
        lines.append("")
        lines.append(CATEGORY_INTRO.get(cat, ""))
        lines.append("")

        for e in by_cat[cat]:
            canon = e["canonical"]
            lines.append(f"### `{canon}`")
            lines.append("")
            lines.append(f"- **DSL 可用名**：`{'` · `'.join(e['dsl_names'])}`")
            if e["params"]:
                lines.append(f"- **参数**：`{', '.join(e['params'])}`")
            lines.append(f"- **说明**：{e['description']}")
            if e["examples"]:
                lines.append(f"- **示例**：`{e['examples'][0]}`")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 延伸阅读")
    lines.append("")
    lines.append("- [`operators_catalog.md`](../cleaned_operators/operators_catalog.md) — 含 stub / 未实现条目")
    lines.append("- [`dsl_operators_reference.md`](dsl_operators_reference.md) — 投递白名单速查")
    lines.append("- [`算子与导入教程.md`](算子与导入教程.md) — 写公式与 import 教程")
    lines.append("- [`operators_semantics.md`](operators_semantics.md) — 语义细节")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    OperatorRegistry, allowlist = _load_registry()
    entries = _collect_canonical_entries(OperatorRegistry, allowlist)
    content = _render(entries, len(allowlist))
    OUT_PATH.write_text(content, encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(entries)} canonical ops, {len(allowlist)} dsl names)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
