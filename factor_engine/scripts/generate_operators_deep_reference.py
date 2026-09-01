# -*- coding: utf-8 -*-
"""生成 docs/OPERATORS_DEEP_REFERENCE.md — 算子深度参考手册.

合并四个真相源:
1. cleaned_operators/docs/operators_catalog.json  (1737 canonical, 元数据标签)
2. cleaned_operators/docs/算子全览.md             (259 节, LaTeX 公式/含义/示例)
3. docs/operator_core_specs.yaml                  (86 条 24 字段核心 spec)
4. cleaned_operators/docs/operator_doc_semantics.py (115 条显式 OpDoc 精确语义)

每算子四要素: (a)标签 (b)公式/构造 (c)可用条件 (d)功能讲解
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cleaned_operators" / "docs"))
from operator_doc_semantics import _EXPLICIT as OPDOC_EXPLICIT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent  # factor_engine/
CATALOG = ROOT / "cleaned_operators" / "docs" / "operators_catalog.json"
OVERVIEW = ROOT / "cleaned_operators" / "docs" / "算子全览.md"
SPECS = ROOT / "docs" / "operator_core_specs.yaml"
ALLOWLIST = ROOT / "docs" / "dsl_allowlist.json"
OUT = ROOT / "docs" / "OPERATORS_DEEP_REFERENCE.md"

# ---------------------------------------------------------------- 源 1: catalog
cat = json.loads(CATALOG.read_text(encoding="utf-8"))
ops = {o["canonical"]: o for o in cat["operators"]}

# ---------------------------------------------------------------- 源 2: 算子全览
ov_text = OVERVIEW.read_text(encoding="utf-8")
ov = {}  # name -> dict(dsl_names, params, meaning, compute, latex, note, example)
_sec_re = re.compile(r"^### `([A-Za-z_][A-Za-z0-9_]*)`\s*$", re.M)
Bullet = re.compile(r"^- \*\*(.+?)\*\*[：:]\s*(.*)$")
positions = [(m.group(1), m.start()) for m in _sec_re.finditer(ov_text)]
for idx, (name, start) in enumerate(positions):
    end = positions[idx + 1][1] if idx + 1 < len(positions) else len(ov_text)
    body = ov_text[start:end]
    entry = {}
    for line in body.splitlines():
        m = Bullet.match(line)
        if m:
            entry[m.group(1)] = m.group(2).strip()
    latex_m = re.search(r"\$\$\s*(.*?)\s*\$\$", body, re.S)
    entry["_latex_block"] = latex_m.group(1).strip() if latex_m else ""
    ov[name] = entry

# ---------------------------------------------------------------- 源 3: specs
specs = {s["name"]: s for s in yaml.safe_load(SPECS.read_text(encoding="utf-8"))}

# ---------------------------------------------------------------- 源 4: allowlist
# 真相 = 运行时 build_dsl_allowlist()（1483 名）；docs/dsl_allowlist.json 是快照
# （1421 名，略滞后 —— 运行时多出的名字标「在」不会误导，快照独有的标「不在」会误导，
# 故以运行时为准，快照仅作 fallback）。
try:
    import sys as _sys
    _sys.path.insert(0, "/home/sunhaiwei/quant_projects")
    from factor_engine.api.operator_registry import build_dsl_allowlist as _bda
    allow = set(_bda().keys())
except Exception:
    try:
        _al = json.loads(ALLOWLIST.read_text(encoding="utf-8"))
        allow = set(_al["operators"] if isinstance(_al, dict) and "operators" in _al
                    else _al)
    except Exception:
        allow = None

# ---------------------------------------------------------------- 分族
FAMILY_PATTERNS = [
    ("截面/排名/中性化", lambda n: n.startswith(("cs_", "rank", "zscore", "normalize",
        "winsorize", "blom")) or n in ("rank", "zscore", "normalize", "winsorize")),
    ("分组 group_", lambda n: n.startswith(("group_",))),
    ("时序 ts_", lambda n: n.startswith("ts_")),
    ("均线/MA 族", lambda n: n in ("SMA", "WMA", "DEMA", "TEMA", "HMA", "KAMA", "ALMA",
        "VWMA", "ts_ema", "ts_mean", "ts_wma")),
    ("技术指标 (MACD/RSI/ATR/布林等)", lambda n: re.match(
        r"^(MACD|RSI|ATR|ADX|ADXR|AROON|CCI|STOCH|KDJ|WR|MFI|OBV|PSAR|PPO|TSI|CMF|CMO|"
        r"Bollinger|Keltner|Donchian|Supertrend|UltimateOscillator|DMI_|ForceIndex|"
        r"TRIX|ROC|WPR| Williams)", n, re.I) is not None),
    ("事件/条件/门控", lambda n: n.startswith(("event_", "trade_when", "where",
        "cluster_", "fano_", "hawkes_")) or n in ("and_", "or_", "not_", "if_else")),
    ("量价/流动性", lambda n: any(k in n for k in ("volume", "turnover", "amihud",
        "adv", "vwap", "float", "free_float", "liquidity"))),
    ("波动/风险估计", lambda n: any(k in n for k in ("vol", "drawdown", "var",
        "skew", "kurt", "sharpe", "sortino", "garman", "parkinson", "yang_zhang",
        "medrv", "semivariance"))),
    ("分钟级 intraday", lambda n: n.startswith(("intraday_", "session_"))),
    ("基本面/财务/PIT", lambda n: n.startswith(("fin_", "yoy_", "ttm_", "qoq_",
        "period_", "quarter_", "holder", "shareholder", "book_", "altman", "earnings",
        "pe_", "pb_", "ps_", "dividend", "split")) or "fundamental" in n),
    ("A 股特有", lambda n: n.startswith(("ashare_", "limit_"))),
    ("数学/安全运算", lambda n: n in ("abs", "log", "log_abs", "signed_log", "sqrt",
        "signed_sqrt", "power", "exp", "sign", "clip", "floor", "ceil", "round",
        "add", "subtract", "multiply", "divide", "protected_div", "safe_div_null",
        "maximum", "minimum", "inverse", "neg", "mod", "tanh", "cbrt")),
    ("比较/逻辑/判断", lambda n: n in ("gt", "ge", "lt", "le", "eq", "ne", "is_null",
        "is_not_null", "is_finite", "is_infinite", "coalesce", "fillna_const")),
]
_FAMILIES_ORDER = [f for f, _ in FAMILY_PATTERNS] + ["其他"]


def family_of(name: str) -> str:
    for fam, pred in FAMILY_PATTERNS:
        if pred(name):
            return fam
    return "其他"


SCOPE_CN = {"ts": "时序", "cross_section": "截面", "cs": "截面", "elementwise": "逐元素",
            "group": "分组", "pairwise": "成对", "stateful": "状态", "panel": "面板"}
SURFACE_CN = {"daily": "日线", "extended": "扩展", "research": "研究专用",
              "unsafe": "不安全", "legacy": "遗留", "internal": "内部"}


def fmt_label(name: str, o: dict) -> str:
    parts = []
    s = o.get("surface")
    if s:
        parts.append(f"`{SURFACE_CN.get(s, s)}`")
    sc = o.get("scope")
    if sc:
        parts.append(f"`{SCOPE_CN.get(sc, sc)}`")
    if o.get("pit_safe"):
        parts.append("`PIT 安全`")
    else:
        parts.append("`⚠️ PIT 需核查`")
    if o.get("aliases"):
        parts.append("别名: " + ", ".join(f"`{a}`" for a in o["aliases"]))
    return " · ".join(parts)


def fmt_cond(name: str, o: dict, sp: dict | None) -> str:
    lines = []
    lb, mp = o.get("lookback"), o.get("min_periods")
    if lb is not None:
        lines.append(f"- 窗口 lookback = {lb} 根 bar")
    if mp is not None:
        lines.append(f"- 最少样本 min_periods = {mp}")
    if o.get("lag"):
        lines.append(f"- 内含滞后 lag = {o['lag']}")
    if allow is not None:
        lines.append("- DSL 白名单: **在**" if name in allow else
                     "- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）")
    if sp:
        lines.append(f"- 空值策略: {sp.get('null_policy', '—')}"
                     f" ｜ 值域: {sp.get('domain_policy', '—')}"
                     f" ｜ 溢出: {sp.get('overflow_policy', '—')}")
        lines.append(f"- 状态: {sp.get('status', '—')}"
                     f" ｜ 生产准入: {sp.get('production_policy', sp.get('allow_in_production', '—'))}")
        bk = sp.get("backends") or o.get("backends") or []
        if bk:
            lines.append(f"- 后端: {', '.join(f'`{b}`' for b in bk)}")
    else:
        bk = o.get("backends") or []
        if bk:
            lines.append(f"- 后端: {', '.join(f'`{b}`' for b in bk)}")
    return "\n".join(lines) if lines else "- （无额外限制）"


def fmt_explain(name: str, ov_e: dict | None, od) -> str:
    """讲解: 优先 OpDoc(显式语义), 其次 算子全览, 最后 catalog 生成的骨架句."""
    chunks = []
    if ov_e:
        if ov_e.get("含义"):
            chunks.append(ov_e["含义"])
        if ov_e.get("计算方式") and ov_e["计算方式"] != ov_e.get("含义"):
            chunks.append("计算: " + ov_e["计算方式"])
        if ov_e.get("备注") and ov_e["备注"] not in ("—",):
            chunks.append("备注: " + ov_e["备注"])
        if ov_e.get("示例"):
            chunks.append("示例: `" + ov_e["示例"] + "`")
    if od is not None:
        chunks.append("精讲: " + od.compute)
    if not chunks:
        o = ops.get(name, {})
        sc = o.get("scope", "?")
        chunks.append(f"{SCOPE_CN.get(sc, sc)}算子（基础骨架句，暂无深挖文档 —— "
                      f"公式与语义待补；参数与标签以下方元数据为准）。")
    return "\n\n".join(chunks)


def fmt_formula(name: str, ov_e: dict | None, od) -> str:
    latex = None
    if od is not None and od.latex:
        latex = od.latex
    elif ov_e and ov_e.get("_latex_block"):
        latex = ov_e["_latex_block"]
    if latex:
        return f"$${latex}$$"
    # 无 LaTeX: 给构造式说明
    if ov_e and ov_e.get("计算方式"):
        return "（构造）" + ov_e["计算方式"]
    return "（暂无公式 —— 见讲解中的构造说明）"


# ---------------------------------------------------------------- 生成
by_family = defaultdict(list)
for name in ops:
    by_family[family_of(name)].append(name)

have_doc = [n for n in ops if n in ov or n in OPDOC_EXPLICIT or n in specs]

lines = []
A = lines.append
A("# factor_engine 算子深度参考手册（OPERATORS_DEEP_REFERENCE）")
A("")
A("> **生成方式**: 本手册由 `scripts/generate_operators_deep_reference.py` 从四个真相源机器合并生成：")
A("> ① `cleaned_operators/docs/operators_catalog.json`（1737 canonical 元数据）")
A("> ② `cleaned_operators/docs/算子全览.md`（人工编写公式节）")
A("> ③ `docs/operator_core_specs.yaml`（86 条核心 spec：空值/值域/溢出/后端策略）")
A("> ④ `cleaned_operators/docs/operator_doc_semantics.py`（115 条显式精讲 OpDoc）")
A("> **勿手改本文件** —— 改对应真相源后重跑生成器（文档与代码同步硬性规则）。")
A(f"> 生成覆盖: canonical {len(ops)} 个；其中带公式/讲解文档的 {len(have_doc)} 个，"
  f"其余为元数据骨架条目（已如实标注「暂无公式」）。")
A("")
A("## 四要素说明")
A("")
A("每个算子四段：")
A("")
A("1. **标签** — surface（日线/扩展/研究）、scope（时序/截面/分组/逐元素）、PIT 安全性、别名")
A("2. **公式/构造** — LaTeX 公式（无公式时给构造式说明）")
A("3. **可用条件** — lookback/min_periods/lag、DSL 白名单、空值/值域/溢出策略、生产准入、后端覆盖")
A("4. **功能讲解** — 含义、计算方式、精讲、示例")
A("")
A("## 快速导航（按族）")
A("")
for fam in _FAMILIES_ORDER:
    if by_family.get(fam):
        A(f"- **{fam}**（{len(by_family[fam])}）")
A("")
A("---")
A("")

for fam in _FAMILIES_ORDER:
    names = sorted(by_family.get(fam, []), key=str.lower)
    if not names:
        continue
    A(f"## {fam}（{len(names)} 个）")
    A("")
    for name in names:
        o = ops[name]
        ov_e = ov.get(name)
        od = OPDOC_EXPLICIT.get(name)
        sp = specs.get(name)
        A(f"### `{name}`")
        A("")
        A(f"**标签**: {fmt_label(name, o)}")
        A("")
        A("**公式/构造**:")
        A("")
        A(fmt_formula(name, ov_e, od))
        A("")
        A("**可用条件**:")
        A("")
        A(fmt_cond(name, o, sp))
        A("")
        A("**功能讲解**:")
        A("")
        A(fmt_explain(name, ov_e, od))
        A("")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"written {OUT}  lines={len(lines)}  ops={len(ops)}  with_doc={len(have_doc)}")
