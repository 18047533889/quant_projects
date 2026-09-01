# -*- coding: utf-8 -*-
"""#36 系统性重做 2026 稳健性分析的分类学。

修复三个真问题:
  1. logic_tags 多标签失真 -> 改为互斥主分类(按证据优先级), 字段名永远不算算子
  2. 算子表混入字段/语法碎片 -> 用 factor_engine canonical 算子白名单过滤, 表里只留真算子
  3. figure4 logic_stacked 图空白 -> 修复 string logic_tags 被逐字符迭代的 bug

产物:
  - weekly_backtest_output/robustness_2026.json   (logic_tags 改为主分类字符串 + 新增 logic_summary)
  - weekly_backtest_output/robustness_logic_stats.json / robustness_survivors.json (按新分类重算)
  - factor_engine/docs/reports/2026-08-23/robustness_2026/robustness_2026.html (算子表/逻辑表/图重画)
  - factor_engine/docs/reports/2026-08-23/index.html (2026 区块归因文字同步)
  - /tmp/reclass36_done.json + /tmp/reclass_progress.log

用法: .venv/bin/python jobs/robustness_reclassify.py
"""
import os, sys, json, re, time
from collections import Counter

for _p in (".", "vectorbt_qs", "jobs"):
    _x = os.path.abspath(_p)
    if _x not in sys.path:
        sys.path.insert(0, _x)

import analyze_2026_robustness as A

ROOT = os.path.abspath(".")
MB = os.path.join(ROOT, "weekly_backtest_output")
FORMULA = "/home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json"
OUT_JSON = os.path.join(MB, "robustness_2026.json")
LOGIC_STATS = os.path.join(MB, "robustness_logic_stats.json")
SURVIVORS = os.path.join(MB, "robustness_survivors.json")
PROG = "/tmp/reclass_progress.log"
DONE = "/tmp/reclass36_done.json"

# factor_engine canonical 算子白名单 (468 个)
import factor_engine.cleaned_operators as co
CANON = set(co.OperatorRegistry().list_canonical())
CANON_LOWER = {c.lower() for c in CANON}

# 字段名(永远不算算子)
FIELD_TOKENS = {
    "close", "high", "low", "open", "pre_close", "volume", "amount", "turnover",
    "vwap", "adjclose", "adjhigh", "adjlow", "adjopen", "adjvolume", "adjamount",
    "ret", "return", "chg", "pb_lf", "debttoassets", "eps", "dy", "bp", "cf",
    "roa", "roe", "leverage", "yoyrev", "npr", "ttm", "value", "liquid", "turn",
    "span", "alpha", "beta", "residual", "jump", "frac", "micro", "breadth",
    "spread", "acd", "depth", "wapimg", "wvapw", "wap", "wvol", "wspread",
    "lo", "hi", "_t", "then", "else", "if", "nan", "isnan", "where", "left",
    "right", "tail", "extreme", "distortion", "up_vol", "down_vol", "up_avg",
    "down_avg", "up_ret", "down_ret", "max_ret", "min_ret", "poss_ret",
    "minute", "intraday", "vol_ratio", "adv", "delta", "delay", "std", "corr",
    "mean", "sum", "max", "min", "quantile", "skew", "kurt", "ema", "ewma",
    "atr", "momentum", "reversal", "mean_reversion", "breakout", "volatility",
}


def plog(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(PROG, "a") as f:
        f.write(line + "\n")


def classify_primary(page, dsl, code):
    """互斥主分类, 按证据优先级. 字段名永远不算算子."""
    low_name = page.lower()
    low = (dsl + " " + (code or "")).lower()
    # 1. 名称含 reversal/reversion/mean_reversion
    if re.search(r"revers|reversion", low_name):
        return "mean_reversion"
    # 2. 主导算子为滚动相关
    if any(k in low for k in ("ts_corr", "ewm_corr", "rank_corr")):
        return "correlation"
    # 3. 趋势跟随(无反转语义)
    if any(k in low for k in ("ts_delta", "ts_pct", "pct_change", "ts_log_return", "mom", "roc")):
        return "momentum"
    # 4. 突破结构
    if "breakout" in low_name or ("ts_max" in low and "high" in low) or ("ts_min" in low and "low" in low):
        return "breakout"
    # 5. vwap 偏离类 (放在 volume 前, 语义更强)
    if any(k in low for k in ("vwap_deviation", "intraday_vwap_deviation", "vwap_to_close_return", "open_to_vwap_return")):
        return "vwap_deviation"
    # 6. 量/流动性主导
    if any(k in low for k in ("volume", "amount", "turnover", "vol_ratio", "turn", "adv")):
        return "volume_liquidity"
    # 7. 波动率族
    if any(k in low for k in ("ts_std", "ewm_std", "atr", "ts_var", "ewm_var", "volatility")):
        return "volatility"
    return "other"


def canonical_ops(dsl, code):
    """只返回 canonical 算子(字段/语法碎片剔除)."""
    low = (dsl + " " + (code or "")).lower()
    toks = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", low)
    return {t for t in toks if t in CANON_LOWER}


def compute_stats(rob, fm):
    op_counts = Counter(); op_stable = Counter(); op_total = Counter()
    logic_total = Counter(); logic_stable = Counter(); logic_failed = Counter()
    for r in rob:
        st = r.get("status")
        if st not in ("stable", "decay", "failed"):
            continue
        e = fm.get(r["page"], {})
        dsl = e.get("dsl") or ""; code = e.get("code") or ""
        for o in canonical_ops(dsl, code):
            op_counts[o] += 1; op_total[o] += 1
            if st == "stable":
                op_stable[o] += 1
        tag = r.get("logic_tags") or "other"
        logic_total[tag] += 1
        if st == "stable":
            logic_stable[tag] += 1
        if st == "failed":
            logic_failed[tag] += 1
    op_stable_ratio = {k: (op_stable[k] / v) if v else 0.0 for k, v in op_total.items()}
    logic_stats = {}
    for t in logic_total:
        tot = logic_total[t]
        logic_stats[t] = dict(total=int(tot), stable=int(logic_stable[t]),
                              failed=int(logic_failed[t]),
                              stable_ratio=round(logic_stable[t] / tot, 4) if tot else 0.0,
                              failed_ratio=round(logic_failed[t] / tot, 4) if tot else 0.0)
    return op_counts, op_stable_ratio, logic_stats


def write_logic_stats(rob):
    """robustness_logic_stats.json 按新主分类重算."""
    out = {"single_operator": {}}
    for r in rob:
        st = r.get("status")
        if st not in ("stable", "decay", "failed"):
            continue
        tag = r.get("logic_tags") or "other"
        d = out["single_operator"].setdefault(tag, {"total": 0, "failed": 0, "decay": 0})
        d["total"] += 1
        if st == "failed":
            d["failed"] += 1
        elif st == "decay":
            d["decay"] += 1
    for d in out["single_operator"].values():
        d["failed_rate"] = round(d["failed"] / d["total"], 4) if d["total"] else 0.0
        d["decay_rate"] = round(d["decay"] / d["total"], 4) if d["total"] else 0.0
    json.dump(out, open(LOGIC_STATS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    plog("[write_logic_stats]", LOGIC_STATS)


def write_survivors(rob):
    """robustness_survivors.json 按新主分类重算."""
    op_surv = {}
    stable_list = []
    for r in rob:
        st = r.get("status")
        if st not in ("stable", "decay", "failed"):
            continue
        tag = r.get("logic_tags") or "other"
        d = op_surv.setdefault(tag, {"total": 0, "stable": 0, "decay": 0, "not_failed": 0})
        d["total"] += 1
        if st == "stable":
            d["stable"] += 1
        if st == "decay":
            d["decay"] += 1
        if st != "failed":
            d["not_failed"] += 1
        if st == "stable":
            stable_list.append({
                "page": r["page"],
                "2026_ic": (r.get("rankIC_2026") or {}).get("ic", 0),
                "full_ic": (r.get("rankIC_full") or {}).get("ic", 0),
                "tags": [r.get("logic_tags") or "other"],
            })
    for d in op_surv.values():
        d["stable_rate"] = round(d["stable"] / d["total"], 4) if d["total"] else 0.0
        d["not_failed_rate"] = round(d["not_failed"] / d["total"], 4) if d["total"] else 0.0
    mr = op_surv.get("mean_reversion", {})
    out = {
        "operator_survival": op_surv,
        "stable_list": stable_list,
        "mean_reversion": {
            "total": mr.get("total", 0),
            "stable": mr.get("stable", 0),
            "list": [r["page"] for r in stable_list if r["tags"] == ["mean_reversion"]],
        },
        "operator_not_failed": {k: {"total": v["total"], "not_failed": v["not_failed"],
                                    "not_failed_rate": v["not_failed_rate"]}
                                for k, v in op_surv.items()},
    }
    json.dump(out, open(SURVIVORS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    plog("[write_survivors]", SURVIVORS)


def main():
    t0 = time.time()
    plog("=" * 60)
    plog(f"[reclass] start {time.strftime('%H:%M:%S')}")

    rob = json.load(open(OUT_JSON))
    fm = {e["page_name"]: e for e in json.load(open(FORMULA))}
    plog(f"[reclass] rob={len(rob)} fm={len(fm)}")

    # 1. 重分类: logic_tags -> 互斥主分类字符串
    new_dist = Counter()
    for r in rob:
        e = fm.get(r["page"], {})
        tag = classify_primary(r["page"], e.get("dsl") or "", e.get("code") or "")
        r["logic_tags"] = tag
        new_dist[tag] += 1
    plog("[reclass] 新主分类分布:")
    for k, v in new_dist.most_common():
        plog(f"  {k}: {v}")

    # 2. 算子/逻辑统计 (canonical 白名单)
    op_counts, op_stable_ratio, logic_stats = compute_stats(rob, fm)
    plog(f"[reclass] canonical 算子数: {len(op_counts)}")
    plog("[reclass] 算子表 top10 (canonical):")
    for k, v in op_counts.most_common(10):
        plog(f"  {k:12s} n={v:4d} stable%={op_stable_ratio[k]:.0%}")

    # 3. 写回 robustness_2026.json (logic_tags + logic_summary)
    rob_out = []
    for r in rob:
        o = {k: v for k, v in r.items() if k != "r26"}
        o["logic_summary"] = {k: dict(v) for k, v in logic_stats.items()}
        rob_out.append(o)
    json.dump(rob_out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    plog(f"[reclass] robustness_2026.json 已写 (logic_tags 主分类 + logic_summary)")

    # 4. 重算 logic_stats / survivors 辅助文件
    write_logic_stats(rob)
    write_survivors(rob)

    # 5. 重画图 + 重建页面
    A.load_GV()
    stable_ranked = [r for r in rob if r.get("status") == "stable"]
    stable_ranked.sort(key=lambda r: (r.get("ls2026") or {}).get("cumret", -1e9), reverse=True)
    figure_list = [
        A.figure1_top20_nav(stable_ranked, rob),
        A.figure2_stable_vs_failed(rob),
        A.figure3_op_stability_ratio(op_counts, op_stable_ratio),
        A.figure4_logic_stacked(rob),
    ]
    A.build_pages(rob, op_counts, op_stable_ratio, logic_stats, figure_list)
    plog("[reclass] robustness_2026.html 已重建")

    # 6. 首页注入 (归因文字用新分类)
    A.inject_index(rob, figure_list[0])
    plog("[reclass] index.html 2026 区块已更新")

    # 7. done.json
    done = {
        "new_class_distribution": dict(new_dist),
        "operator_table_top20": [{"op": k, "count": op_counts[k], "stable_ratio": round(op_stable_ratio[k], 4)}
                                 for k, v in op_counts.most_common(20)],
        "logic_summary": {k: dict(v) for k, v in logic_stats.items()},
        "momentum_count": new_dist.get("momentum", 0),
        "mean_reversion_count": new_dist.get("mean_reversion", 0),
        "before": {"momentum": 474, "mean_reversion": 12},
        "status": Counter(r.get("status") for r in rob),
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    json.dump(done, open(DONE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    plog(f"[reclass] done -> {DONE}  {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
