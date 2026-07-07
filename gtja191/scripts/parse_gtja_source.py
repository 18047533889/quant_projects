#!/usr/bin/env python3
"""从 BigQuant / GTJA 原始公式文本解析 191 条因子表达式。"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIGQUANT_CACHE = ROOT / "source" / "bigquant_formula_cache.txt"

# BigQuant 页面损坏或 vendor 注释更可靠的补全
MANUAL_OVERRIDES: dict[int, str] = {
    183: "0",
    184: "rank(ts_corr(delay(open - close, 1), close, 200)) + rank(open - close)",
    185: "rank(-1 * power(1 - open / close, 2))",
    186: (
        "(ts_mean(abs((ts_sum(if_else(and_(ld_gt_0, ld_gt_hd), ld, 0), 14) * 100 / ts_sum(tr, 14) "
        "- ts_sum(if_else(and_(hd_gt_0, hd_gt_ld), hd, 0), 14) * 100 / ts_sum(tr, 14)) "
        "/ (ts_sum(if_else(and_(ld_gt_0, ld_gt_hd), ld, 0), 14) * 100 / ts_sum(tr, 14) "
        "+ ts_sum(if_else(and_(hd_gt_0, hd_gt_ld), hd, 0), 14) * 100 / ts_sum(tr, 14)) * 100), 6) "
        "+ delay(ts_mean(abs((ts_sum(if_else(and_(ld_gt_0, ld_gt_hd), ld, 0), 14) * 100 / ts_sum(tr, 14) "
        "- ts_sum(if_else(and_(hd_gt_0, hd_gt_ld), hd, 0), 14) * 100 / ts_sum(tr, 14)) "
        "/ (ts_sum(if_else(and_(ld_gt_0, ld_gt_hd), ld, 0), 14) * 100 / ts_sum(tr, 14) "
        "+ ts_sum(if_else(and_(hd_gt_0, hd_gt_ld), hd, 0), 14) * 100 / ts_sum(tr, 14)) * 100), 6), 6)) / 2"
    ),
}


def _parse_bigquant_block(text: str) -> dict[int, str]:
    start = text.find("* Alpha1 ")
    end = text.find("## 示例")
    if start < 0 or end < 0:
        raise ValueError("BigQuant formula block not found")
    block = text[start:end]
    parts = re.split(r"\* Alpha(\d+)\s+", block)
    out: dict[int, str] = {}
    for i in range(1, len(parts), 2):
        num = int(parts[i])
        expr = parts[i + 1].strip()
        if num == 191:
            expr = expr.split("##")[0].strip()
        out[num] = expr
    return out


def load_source_formulas() -> dict[str, str]:
    if not BIGQUANT_CACHE.exists():
        existing = ROOT / "source" / "gtja191_formulas.json"
        if existing.exists():
            return json.loads(existing.read_text(encoding="utf-8"))
        raise FileNotFoundError(
            f"missing {BIGQUANT_CACHE}; keep source/gtja191_formulas.json or add cache file"
        )
    text = BIGQUANT_CACHE.read_text(encoding="utf-8")
    parsed = _parse_bigquant_block(text)
    for num, expr in MANUAL_OVERRIDES.items():
        parsed[num] = expr
    missing = [n for n in range(1, 192) if n not in parsed]
    if missing:
        raise RuntimeError(f"missing GTJA formulas: {missing}")
    return {f"gtja191_alpha_{n:03d}": parsed[n] for n in range(1, 192)}


def main() -> None:
    formulas = load_source_formulas()
    out = ROOT / "source" / "gtja191_formulas.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(formulas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(formulas)} formulas -> {out}")


if __name__ == "__main__":
    main()
