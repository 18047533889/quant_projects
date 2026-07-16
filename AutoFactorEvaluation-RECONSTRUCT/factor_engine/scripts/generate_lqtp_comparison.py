#!/usr/bin/env python3
"""从 cleaned_operators 白名单生成 LQTP vs factor_engine 算子对照 Markdown。"""
from __future__ import annotations

import re
from pathlib import Path

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry
from api.operator_registry import build_dsl_allowlist

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "cleaned_operators" / "docs" / "算子全览.md"
OUT = ROOT / "docs" / "lqtp_vs_factor_engine_operators.md"

LQTP_ROWS: list[tuple[str, str, set[str]]] = [
    ("截面", "rank / cs_rank", {"rank", "cs_rank"}),
    ("截面", "zscore / cs_zscore", {"zscore", "cs_zscore"}),
    ("截面", "cs_demean", {"cs_demean"}),
    ("截面", "scale", {"scale"}),
    ("截面", "winsorize", {"winsorize"}),
    ("截面", "cs_resid", {"cs_resid"}),
    ("截面", "cs_regression(y,x[,mode])", {"cs_regression"}),
    ("标量", "abs", {"abs"}),
    ("标量", "log", {"log"}),
    ("标量", "sqrt", {"sqrt"}),
    ("标量", "sign", {"sign"}),
    ("标量", "round", {"round"}),
    ("标量", "signed_sqrt", {"signed_sqrt"}),
    ("标量", "sigmoid", {"sigmoid"}),
    ("标量", "power", {"power"}),
    ("标量", "cap", {"cap", "clip", "clamp"}),
    ("标量", "where / iif", {"where", "if_else", "trade_when"}),
    ("标量", "is_null / is_nan", {"is_null", "is_nan", "is_finite"}),
    ("标量", "nan_to_num", {"nan_to_num"}),
    ("标量", "coalesce", {"coalesce"}),
    ("时序", "ts_mean", {"ts_mean", "sma", "mean"}),
    ("时序", "ts_sum", {"ts_sum"}),
    ("时序", "ts_std", {"ts_std", "std"}),
    ("时序", "ts_max", {"ts_max", "max"}),
    ("时序", "ts_min", {"ts_min", "min"}),
    ("时序", "ts_rank", {"ts_rank"}),
    ("时序", "ts_delta", {"ts_delta", "delta"}),
    ("时序", "ts_pct", {"ts_pct", "returns", "m_pct_change"}),
    ("时序", "delay", {"delay", "ts_delay", "m_delay"}),
    ("时序", "decay_linear / ts_decay_linear", {"decay_linear", "ts_decay_linear"}),
    ("时序", "ema / EMA", {"ema"}),
    ("时序", "price_spread_deviation", {"price_spread_deviation"}),
    ("时序", "ts_corr", {"ts_corr", "corr"}),
    ("时序", "ts_cov", {"ts_cov", "cov"}),
    ("时序", "ts_regression_slope", {"ts_regression_slope", "ts_regression"}),
    ("时序", "ts_quantile", {"ts_quantile", "quantile", "m_percentile"}),
    ("时序", "ts_skew", {"ts_skew", "skew"}),
    ("时序", "ts_kurt", {"ts_kurt", "kurt"}),
    ("时序", "ts_moment", {"ts_moment"}),
    ("时序", "ts_topk_sum", {"ts_topk_sum", "m_top_n_sum", "tm_top_n_sum"}),
    ("时序", "ts_rank_corr / rankcorr", {"rank_corr", "rankcorr"}),
    ("时序", "ts_poly2_coeff", {"ts_poly2_coeff"}),
    ("时序", "ts_poly2_resid", {"ts_poly2_resid"}),
    ("时序", "digital_count", {"digital_count"}),
    ("时序", "ts_max_buildup", {"ts_max_buildup"}),
    ("时序", "ts_argmax", {"ts_argmax", "m_argmax", "at_imax"}),
    ("时序", "ts_argmin", {"ts_argmin", "m_argmin", "at_imin"}),
    ("市场", "benchmark_index(index)", set()),
    ("市场", "rolling_beta_to_market", {"rolling_beta_to_market"}),
    ("市场", "fp_beta", {"rolling_beta_to_market"}),
    ("市场", "downside_beta", {"downside_beta"}),
    ("市场", "tail_beta", {"tail_beta"}),
    ("市场", "residual_momentum_capm", {"residual_momentum_capm"}),
    ("市场", "coskewness_to_market", {"coskewness_to_market"}),
    ("市场", "idio_vol", {"idio_vol"}),
    ("市场", "idio_skew", {"idio_skew"}),
    ("财报", "ttm", {"ttm"}),
    ("财报", "quarter", {"quarter"}),
    ("财报", "yoy", {"yoy"}),
    ("财报", "avg2", {"avg2"}),
    ("中性化", "industry_neutralize 及别名", {"industry_neutralize"}),
    ("中性化", "size_neutralize 及别名", {"size_neutralize"}),
    ("中性化", "neutralize 组合", {"neutralize"}),
    ("分钟/L2", "minute_bar", set()),
    ("分钟/L2", "l2_sum / l2_sum_if", set()),
    ("分钟/L2", "l2_count / l2_count_if", set()),
    ("分钟/L2", "real_turnover_rate", {"real_turnover_rate"}),
    ("YAML模板", "safe_div", {"safe_div", "protected_div"}),
    ("YAML模板", "nullif_zero", set()),
    ("YAML模板", "safe_log", {"safe_log", "protected_log"}),
    ("YAML模板", "clean", {"clean", "fillna"}),
    ("YAML模板", "ma / sum_n / std_n / delta / pct_change", {"ts_mean", "ts_sum", "ts_std", "ts_delta", "ts_pct"}),
    ("YAML模板", "daily_return 等收益模板", {"returns", "cumulative_returns"}),
    ("YAML模板", "momentum / reversal 等", {"ts_mom", "mom"}),
    ("YAML模板", "atr / realized_vol / downside_vol", {"atr", "volatility"}),
    ("YAML模板", "volume_ma / volume_ratio 等", {"ts_mean", "window_mean"}),
    ("YAML模板", "vwap_gap / illiquidity / 量价相关模板", set()),
    ("YAML模板", "return_rank / volume_rank 等", {"rank"}),
    ("YAML模板", "quality_mask / neutral_return 等", set()),
    ("YAML模板", "L2 成交模板", set()),
]


def _parse_categories(guide_text: str) -> dict[str, str]:
    """从算子全览 Markdown 解析 canonical → 中文分类名映射。"""
    cat_map: dict[str, str] = {}
    current = "未分类"
    for line in guide_text.splitlines():
        m = re.match(r"^## (.+?) \{#", line)
        if m:
            current = m.group(1).split("（")[0].strip()
            continue
        m = re.match(r"^### `([^`]+)`", line)
        if m:
            cat_map[m.group(1).lower()] = current
    return cat_map


def _fe_has(names: set[str], allow_lower: set[str]) -> bool:
    """判断 LQTP 手册列出的算子名是否在 factor_engine DSL 白名单中。"""
    return any(n.lower() in allow_lower for n in names)


def main() -> None:
    """生成 ``docs/lqtp_vs_factor_engine_operators.md`` 对照表。"""
    load_all()
    allow = build_dsl_allowlist()
    allow_lower = {k.lower() for k in allow}
    cat_map = _parse_categories(GUIDE.read_text(encoding="utf-8"))

    lqtp_covered: set[str] = set()
    for _, _, names in LQTP_ROWS:
        lqtp_covered |= {n.lower() for n in names}

    seen_canon: set[str] = set()
    fe_only_by_cat: dict[str, list[str]] = {}
    for dsl in sorted(allow.keys(), key=str.lower):
        canon = OperatorRegistry._aliases.get(dsl, OperatorRegistry._aliases.get(dsl.lower(), dsl))
        key = canon.lower() if isinstance(canon, str) else dsl.lower()
        if key in seen_canon or key in lqtp_covered:
            continue
        seen_canon.add(key)
        cat = cat_map.get(key, cat_map.get(dsl.lower(), "未分类"))
        fe_only_by_cat.setdefault(cat, []).append(dsl)

    lines: list[str] = []
    lines += [
        "# LQTP-backtest vs factor_engine 算子对照表",
        "",
        "> 对照基准：LQTP-backtest 因子服务用户手册（2026-06-18 更新）  ",
        "> factor_engine：`cleaned_operators` DSL 白名单 **512** 名 / **378** 规范算子  ",
        "> 重新生成：`cd factor_engine && PYTHONPATH=. python3 scripts/generate_lqtp_comparison.py`",
        "",
        "## 图例",
        "",
        "| 符号 | 含义 |",
        "|------|------|",
        "| ✓ | 该侧手册/白名单明确列出或可直接使用 |",
        "| — | 该侧未列出 / 无等价实现 |",
        "| ✅ | **factor_engine 有，LQTP 手册未列** |",
        "",
        "## 一、LQTP 手册算子 → factor_engine 覆盖情况",
        "",
        "| 分类 | LQTP 手册算子 | LQTP | FE | 备注 |",
        "|------|---------------|:----:|:--:|------|",
    ]

    fe_ok = fe_miss = 0
    for cat, op, names in LQTP_ROWS:
        fe = "✓" if _fe_has(names, allow_lower) else "—"
        note = ""
        if fe == "—":
            fe_miss += 1
            if "benchmark" in op or "minute_bar" in op or "l2_" in op:
                note = "FE 无内置数据源/聚合入口"
            elif "模板" in cat:
                note = "LQTP functions.yaml 预置；FE 需手写"
            elif op == "nullif_zero":
                note = "可用 where/条件写法等价"
        else:
            fe_ok += 1
        lines.append(f"| {cat} | {op} | ✓ | {fe} | {note} |")

    lines += [
        "",
        f"**小结**：LQTP 手册 **{len(LQTP_ROWS)}** 条；FE 覆盖 **{fe_ok}** 条，缺口 **{fe_miss}** 条。",
        "",
        "## 二、factor_engine 有而 LQTP 手册未列的算子（✅）",
        "",
    ]

    total = sum(len(v) for v in fe_only_by_cat.values())
    lines.append(f"**合计 {total} 个 DSL 名**（含别名）。")
    lines.append("")
    lines.append("| factor_engine 分类 | FE独有算子（节选） | 数量 |")
    lines.append("|-------------------|-------------------|:----:|")

    cat_order = [
        "技术信号", "时序滚动", "统计与回归", "截面变换", "分组中性化",
        "价量衍生", "数据清洗", "滞后 / 差分 / 累计", "元素级数学", "日内微观结构",
    ]
    for cat in cat_order:
        ops = fe_only_by_cat.get(cat, [])
        if not ops:
            continue
        sample = ", ".join(f"`{o}`" for o in ops[:6])
        if len(ops) > 6:
            sample += f" …（+{len(ops) - 6}）"
        lines.append(f"| {cat} | ✅ {sample} | {len(ops)} |")

    lines.append("")
    for cat in cat_order:
        ops = fe_only_by_cat.get(cat, [])
        if not ops:
            continue
        lines += [f"### {cat}（{len(ops)}）", "", "| 算子 | FE独有 |", "|------|:------:|"]
        lines += [f"| `{op}` | ✅ |" for op in ops]
        lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
