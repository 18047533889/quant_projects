#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""456 因子公式 LQTP 转换（全量版）。

三级转换 + 自动翻转，输出 factor_delivery_converted/formula_lqtp_all.json。
输入：/home/sunhaiwei/factor_delivery_converted/factors_combined/*.json（456 唯一）。
"""
import json, re, glob, os, sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

CONV_DIR = Path("/home/sunhaiwei/factor_delivery_converted/factors_combined")
OUT_PATH = Path("/home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json")

# ---- 别名/规范化表（与 61 版一致）----
_TOKEN_MAP = {
    "ema": "ewm_mean", "EMA": "ewm_mean", "EWMA": "ewm_mean", "ewm": "ewm_mean",
    "z_score": "ts_zscore", "zscore": "ts_zscore", "rank": "cs_rank",
    "R": "ts_corr", "ATR": "true_range", "delay": "ts_delay",
    "shift": "ts_delay", "pct_change": "ts_pct", "var": "ts_var",
    "SMA": "ts_mean", "MA": "ts_mean", "VWAP": "rolling_vwap",
    "median": "ts_median", "Mean": "ts_mean",
    "flex_min": "flex_min", "flex_max": "flex_max",
    "amount_weighted_mean": "group_weighted_mean",
    "volume_weighted_mean": "cs_weighted_mean",
    "mean": "ts_mean", "sum_30": "ts_sum",
}

def _normalize_multi_arg_ema(f):
    def emaN(m): return f"ewm_mean({m.group(2)}, {m.group(1)})"
    f = re.sub(r"\bEMA_(\d+)\s*\(([^)]*)\)", emaN, f)
    f = re.sub(r"\bEWM_(\d+)\s*\(([^)]*)\)", emaN, f)
    f = re.sub(r"\bEMA(\d+)\s*\(([^)]*)\)", emaN, f)
    f = re.sub(r"\bMedian(\d+)\s*\(([^)]*)\)", lambda m: f"ts_median({m.group(2)}, {m.group(1)})", f)
    f = re.sub(r"\bMean_(\d+)\s*\(([^)]*)\)", lambda m: f"ts_mean({m.group(2)}, {m.group(1)})", f)
    return f

def _token_replace_formula(f):
    f = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.pct_change\(\)", lambda m: f"ts_pct({m.group(1)}, 1)", f)
    f = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.ts_pct\(\)", lambda m: f"ts_pct({m.group(1)}, 1)", f)
    for tok, repl in _TOKEN_MAP.items():
        f = re.sub(rf"\b{re.escape(tok)}\s*\(", f"{repl}(", f)
    return f

def _normalize_rolling(f):
    f = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.rolling\s*\(\s*(\d+)\s*\)\s*\.mean\s*\(\s*\)",
               lambda m: f"ts_mean({m.group(1)}, {m.group(2)})", f)
    f = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.rolling\s*\(\s*(\d+)\s*\)\s*\.std\s*\(\s*\)",
               lambda m: f"ts_std({m.group(1)}, {m.group(2)})", f)
    f = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.rolling\s*\(\s*(\d+)\s*\)\s*\.sum\s*\(\s*\)",
               lambda m: f"ts_sum({m.group(1)}, {m.group(2)})", f)
    return f

def normalize_to_fe(f):
    if not f: return ""
    f = _normalize_multi_arg_ema(f)
    f = _token_replace_formula(f)
    f = _normalize_rolling(f)
    return f

def is_fe_parseable(f):
    try:
        from factor_engine.api.dsl_parser import parse_factor
        parse_factor(f, name="_probe")
        return True
    except Exception:
        return False

def load_all_defs():
    best = {}
    for fp in sorted(glob.glob(str(CONV_DIR / "*.json"))):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        fn = d.get("factor_name", "")
        if not fn: continue
        is_primary = os.path.basename(fp) == f"factor_{fn}.json"
        if fn not in best or is_primary:
            best[fn] = fp
    out = []
    for fn, fp in sorted(best.items()):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        code = d.get("code", "")
        if not code: continue
        page = fn.replace("factor_", "")
        out.append({"page_name": page, "factor_name": fn, "code": code, "d": d})
    return out

def main():
    defs = load_all_defs()
    print(f"[456lqtp] 因子数: {len(defs)}")
    records = []
    stats = {"ok": 0, "fallback": 0, "custom": 0}

    for it in defs:
        page = it["page_name"]
        d = it["d"]
        code = it["code"]
        raw_formula = (d.get("formula") or "").strip()
        lqtp = (d.get("lqtp_formula") or "").strip()
        fe_formula = (d.get("fe_formula") or "").strip()

        raw_metrics = d.get("raw_metrics") or {}
        raw_rank_ic = raw_metrics.get("rank_ic")
        is_flipped = (raw_rank_ic is not None and float(raw_rank_ic) < 0)

        candidate_raw = lqtp or fe_formula or raw_formula
        candidate = normalize_to_fe(candidate_raw)
        status = "custom"
        can_fe = False
        note = ""

        if candidate and is_fe_parseable(candidate):
            status = "ok"; can_fe = True
            dsl_text = candidate
            note = "已按 LQTP 转换，factor_engine DSL 可直接执行"
        elif candidate_raw and is_fe_parseable(candidate_raw):
            status = "ok"; can_fe = True
            dsl_text = candidate_raw
            note = "LQTP/因子引擎算子可直接执行"
        elif candidate:
            status = "fallback"
            dsl_text = candidate
            note = "已转 FactorEngine（DSL 语法近似，落值用真实 Python code）"
        else:
            status = "custom"
            dsl_text = f"MYDSL({page})"
            note = "自命名 DSL（用不了 factor_engine），无可用公式文本"

        if is_flipped and dsl_text and not dsl_text.startswith("-"):
            dsl_text = f"-({dsl_text})"

        stats[status] += 1
        records.append({
            "page_name": page,
            "factor_name": it["factor_name"],
            "status": status,
            "dsl": dsl_text,
            "lqtp_formula": lqtp,
            "fe_formula": fe_formula,
            "code": code,
            "is_flipped": is_flipped,
            "can_use_factor_engine": can_fe,
            "note": note,
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"[456lqtp] 写入 {OUT_PATH}")
    print(f"[456lqtp] 统计: ok={stats['ok']} fallback={stats['fallback']} custom={stats['custom']} 共 {len(records)}")

if __name__ == "__main__":
    main()
