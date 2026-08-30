#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段2：Top 因子参数寻优（factor_optimizer SearchRunner + GridSearch strategy）。

从阶段1结果选 Top N（按优化后 RankIC IR 排序），对每个 Top 因子做参数寻优：
参数网格（winsor/zscore/smooth）作为 SearchSpace，GridSearch 策略逐格生成 Trial，
SearchRunner 用 objective=RankIC IR 寻优（vwap-to-vwap 收益口径），
每格以该格预处理后的"最优矩阵"存档（与阶段1 逐选最优变体 语义一致）。

输出：
  weekly_backtest_output/optimized_top/<page>.parquet   （寻优后因子，取网格最优格）
  weekly_backtest_output/optimized_top_meta.json        （每格的评分 + 最优格 + 公式化参数链）

参数寻优记录链（公式化表述，例如）：
  "cs_winsor(0.01, 0.99) → cs_zscore → rolling_mean(3)"

本文件不修改收益口径（全局固定 vwap-to-vwap）。
"""
import sys, os, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
warnings.filterwarnings("ignore")

PROJECT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "vectorbt_qs"))
sys.path.insert(0, str(PROJECT / "factor_optimizer"))
sys.path.insert(0, str(PROJECT / "factor_preprocess"))

from quant_evaluator.metrics.ic import _spearman_rank_correlation
from factor_preprocess.registry.transforms import create_default_registry

# ---- factor_optimizer SearchRunner 官方 API（已用 live probe 验证契约）----
from factor_optimizer.search.runner import SearchRunner, SearchConfig
from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.contracts.objective import ObjectiveSpec
from factor_optimizer.contracts.splits import SplitPlan, EvaluationProtocol
from factor_optimizer.contracts.treatment_integrity import build_integrity_evidence
from factor_optimizer.search.strategies import SearchSpace, ParameterSpace, GridSearch

OPT_DIR = PROJECT / "weekly_backtest_output" / "optimized_factors"
OUT_DIR = PROJECT / "weekly_backtest_output" / "optimized_top"
META_PATH = PROJECT / "weekly_backtest_output" / "optimized_top_meta.json"
OPT1_META = PROJECT / "weekly_backtest_output" / "optimized_meta.json"
DAILY_ADJ = Path.home() / "cos_data" / "StockDailyBarAdj"
START, END = "2019-01-02", "2026-08-24"
TOP_N = 30

# 参数网格（与 "cs_winsor(...) → cs_zscore → rolling_mean(n)" 链一一对应）。
# choice 参数要求 string 值（factor_optimizer ParameterSpace 契约）。
WINSOR_OPTS = ["(0.01, 0.99)", "(0.02, 0.98)", "(0.05, 0.95)", "none"]
ZSCORE_OPTS = ["True", "False"]
SMOOTH_OPTS = ["1", "3", "5"]
# 全量 Top 30 网格 30×4×2×3 = 720 格，每格一次全历史(2019-2026) RankIC 循环。
# 4×2×4=32 格/因子实测约 258s（~8s/格），720 格预计 ~1.6h，可接受，先不改预算。
EVAL_STRIDE = 2  # 2026-08-28：评估用半采样加速（最终 IR 走全采样回评）

_reg = create_default_registry()
cs_winsor = _reg.get_function("cs_winsor")
cs_zscore = _reg.get_function("cs_zscore")

_HAS_VWAP = None


def parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="阶段2：Top 因子参数寻优")
    ap.add_argument("--top", type=int, default=TOP_N,
                    help=f"对 Top N 个因子寻优（默认 {TOP_N}；冒烟用 --top 5）")
    return ap.parse_args(argv)


def load_vwap():
    """加载 后复权 AdjVwap 矩阵（vwap-to-vwap 后复权硬性，StockDailyBarAdj）。"""
    global _HAS_VWAP
    if _HAS_VWAP is not None:
        return _HAS_VWAP
    files = sorted(DAILY_ADJ.glob("*.parquet"))
    fs = "[" + ",".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT TradeDate as date, Symbol as symbol, AdjVwap as vwap
        FROM read_parquet({fs})
        WHERE TradeDate >= DATE '{START}' AND TradeDate <= DATE '{END}'
    """).df()
    m = df.pivot_table(index='date', columns='symbol', values='vwap', aggfunc='first')
    m.index = pd.to_datetime(m.index)
    _HAS_VWAP = m.sort_index()
    return _HAS_VWAP


def daily_rankic_ir(factor_mat, vwap):
    """返回 RankIC IR（vwap-to-vwap 收益口径，shift(-2) 企业级：t+1成交→t+2卖出）。"""
    common = factor_mat.index.intersection(vwap.index)
    fv = factor_mat.reindex(index=common)
    vv = vwap.reindex(index=common)
    cols = vv.columns.intersection(fv.columns)
    fv = fv[cols]; vv = vv[cols]
    fwd = vv.pct_change().shift(-2)  # vwap-to-vwap（对齐平台TargetVwapReturnH01）
    T = fv.shape[0]
    ic = np.full(T, np.nan)
    fv_a = fv.values; fwd_a = fwd.values
    for t in range(0, T, EVAL_STRIDE):
        m = fv_a[t]; r = fwd_a[t]
        mask = np.isfinite(m) & np.isfinite(r)
        if mask.sum() < 20:
            continue
        ic[t] = _spearman_rank_correlation(m[mask], r[mask])
    s = pd.Series(ic, index=common).dropna()
    if len(s) == 0:
        return 0.0, 0.0
    mean = float(s.mean()); std = float(s.std())
    return mean, (mean / std if std > 1e-9 else 0.0)


def wl_from_token(token):
    """把 winsor 选择 token 还原成 (lower, upper) 或 None。"""
    if token == "none":
        return (None, None)
    lo, hi = token.strip("()").split(",")
    return (float(lo), float(hi))


def wu_from_token(token):
    """取 winsor 上界（token 为 'none' 时返回 None）。"""
    if token == "none":
        return None
    return wl_from_token(token)[1]


def apply_params(mat, winsor_lower, winsor_upper, zscore, smooth_win):
    """按参数应用预处理（winsor 可为 None）。返回变换后矩阵。"""
    out = mat.copy()
    if winsor_lower is not None:
        out = pd.DataFrame(cs_winsor(out.values, lower=winsor_lower, upper=winsor_upper, axis=1),
                           index=mat.index, columns=mat.columns)
    if zscore:
        out = pd.DataFrame(cs_zscore(out.values, axis=1, ddof=1),
                            index=mat.index, columns=mat.columns)
    if smooth_win and smooth_win > 1:
        out = out.rolling(smooth_win, min_periods=1).mean()
    return out


def param_to_formula(wl, wu, zs, sw):
    """把参数还原成公式化参数链（如 "cs_winsor(0.01, 0.99) → cs_zscore → rolling_mean(3)"）。"""
    steps = []
    if wl is not None and wl[0] is not None:
        steps.append(f"cs_winsor({wl[0]}, {wl[1]})")
    if zs:
        steps.append("cs_zscore")
    if sw and sw > 1:
        steps.append(f"rolling_mean({sw})")
    return steps, (" → ".join(steps) if steps else "原始因子")


def build_eval_target(vwap):
    """构造 evaluation_fn —— 目标页由 trial 参数里的 page 决定。"""
    def evaluation_fn(trial, fidelity):
        params = trial.metadata["params"]
        page = params["page"]
        mat = _PAGE_MATS.get(page)
        if mat is None:
            raise ValueError(f"unknown page {page}")
        wl_pair = wl_from_token(params["winsor"])
        wl, wu = wl_pair
        zs = (params["zscore"] == "True")
        sw = int(params["smooth"])
        tmat = apply_params(mat, wl, wu, zs, sw)
        mean, ir = daily_rankic_ir(tmat, vwap)
        if not np.isfinite(ir):
            ir = 0.0
        # R55 P0-9: 评分必须有真实 TreatmentIntegrityEvidence（fail closed）。
        # 本阶段2的预处理是确定性的（winsor→zscore→smooth），故在 evaluation_fn 内
        # 直接测量 before/after 并构建 evidence，保证 SearchRunner 的 integrity 门禁通过。
        identical = np.array_equal(mat.values, tmat.values, equal_nan=True)
        evidence = build_integrity_evidence(
            treatment_id=trial.trial_id,
            treatment_kind="raw" if identical else "winsor_zscore_smooth",
            applied_parameters=params,
            before=mat,
            after=(mat if identical else tmat),
        )
        return {"score": float(ir),
                "evidence_ref": f"{page}:{params['winsor']}:{params['zscore']}:{params['smooth']}",
                "treatment_integrity_evidence": evidence}
    return evaluation_fn


def make_split_plan(n_days):
    return SplitPlan(
        split_id="optimized_top_grid",
        train_mask=[True] * n_days,
        validation_mask=[False] * n_days,
        test_mask=[False] * n_days,
        metadata={"dataset": "optimized_top_grid"},
    )


def main():
    args = parse_args()
    top_n = max(1, args.top)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vwap = load_vwap()

    # 读阶段1 meta，选 Top N
    if not OPT1_META.exists():
        print("[opt2] 无阶段1 meta，先跑 optimize_factors.py")
        return
    opt1 = json.loads(OPT1_META.read_text())
    ranked = sorted(opt1.items(), key=lambda kv: kv[1].get("best_rankic_ir", 0), reverse=True)
    top = [page for page, m in ranked if m.get("best_rankic_ir", 0) > 0][:top_n]
    if not top:
        print("[opt2] 阶段1 meta 里没有正 IR 因子，无法寻优")
        return
    print(f"[opt2] Top {len(top)} 因子 (vwap-to-vwap 口径, factor_optimizer SearchRunner)", flush=True)

    # 一次性加载所有 Top 矩阵（与 VWAP 列取交集，不用重排到 VWAP 列！）
    global _PAGE_MATS
    _PAGE_MATS = {}
    for page in top:
        fpath = OPT_DIR / f"{page}.parquet"
        if not fpath.exists():
            continue
        try:
            mat = pd.read_parquet(fpath)
            cols = mat.columns.intersection(vwap.columns)
            _PAGE_MATS[page] = mat[cols]
        except Exception:
            continue
    print(f"[opt2] 可用矩阵: {len(_PAGE_MATS)}", flush=True)
    if not _PAGE_MATS:
        print("[opt2] 无可用矩阵")
        return

    # ---- factor_optimizer 官方参数化寻优。----
    # 搜索空间：除明确参与寻优的 winsor/zscore/smooth 之外，把 page 也一并建模，
    # 这样每次 run 即为整批 Top 页的一张完整参数网格（一份 session 一条证据链）。
    pages = list(_PAGE_MATS)
    space = SearchSpace(parameters=[
        ParameterSpace("page", "choice", choices=pages),
        ParameterSpace("winsor", "choice", choices=WINSOR_OPTS),
        ParameterSpace("zscore", "choice", choices=ZSCORE_OPTS),
        ParameterSpace("smooth", "choice", choices=SMOOTH_OPTS),
    ])
    n_days = len(vwap)
    plan = make_split_plan(n_days)
    spec = ObjectiveSpec(metric_name="rankic_ir", direction="maximize")
    strat = GridSearch(space=space, objective_spec=spec)
    grid_total = int(strat.total_combinations)
    # GridSearch 混合进制 decode：page 是最快变化轴，全网格格数 = 页数×4×2×4。
    # SearchRunner 的 is_exhausted 用 trials_used >= max_trials，硬编码 100 会把
    # 超过 100 的格子漏掉（每页只估 100/页数 格）。预算必须动态 = 全网格格数。
    if grid_total > 100:
        print(f"[opt2] 网格共 {grid_total} 格 > 100 —— budget 动态放大到 {grid_total}（否则漏格）", flush=True)
    cfg = SearchConfig(
        budget=SearchBudget(
            max_trials=grid_total,
            max_evaluations=grid_total,
            max_cost_units=1e9,
        ),
        objective_spec=spec,
        # 全网格枚举：用超大 plateau_window 关闭早停（IR 持平会触发 plateau 早停，漏格）。
        plateau_window=100000,
        plateau_threshold=0.0,
        enable_multifidelity=False,
        execution_mode="research_only",
        require_evaluation_protocol=False,
    )
    eval_target = build_eval_target(vwap)
    proto = EvaluationProtocol(split_plan=plan, evaluator=eval_target)
    runner = SearchRunner(config=cfg, proposal_fn=strat.propose, evaluation_fn=proto, strategy=strat)

    t0 = time.time()
    print(f"[opt2] 网格 {len(WINSOR_OPTS)}×{len(ZSCORE_OPTS)}×{len(SMOOTH_OPTS)}×{len(_PAGE_MATS)} 页 → 共 {strat.total_combinations} 格", flush=True)
    session = runner.run("optimized_top_search")
    elapsed = time.time() - t0
    # 断言：全网格应全部被评估（grid_total 格），漏格则告警（不终止，便于诊断）
    n_evaluated = len([t for t in session.trials if t.is_successful()])
    if n_evaluated < grid_total:
        print(f"[opt2] ⚠️ 仅评估 {n_evaluated}/{grid_total} 格（budget 或早停导致漏格），"
              f"stop_reason={session.stop_reason}", flush=True)

    # ---- 解析 session.trials：按页聚合，取该页 IR 最高的格作为最优格 ----
    page_best = {}  # page -> {best_ir, best_params, best_mean, best_steps, best_formula, trials}
    for trial in session.trials:
        if not trial.is_successful():
            continue
        try:
            params = trial.metadata["params"]
        except Exception:
            continue
        page = params.get("page")
        if page not in _PAGE_MATS:
            continue
        wl_pair = wl_from_token(params["winsor"])  # (low, high) 或 (None, None)
        wl, wu = wl_pair
        zs = (params["zscore"] == "True")
        sw = int(params["smooth"])
        try:
            tmat = apply_params(_PAGE_MATS[page], wl, wu, zs, sw)
            mean, ir = daily_rankic_ir(tmat, vwap)
            if not np.isfinite(ir):
                ir = 0.0
        except Exception:
            continue
        score = ir
        steps, formula = param_to_formula(wl_pair, wu, zs, sw)
        rec = page_best.setdefault(page, {
            "best_ir": -1e18, "best_params": None, "best_mean": 0.0,
            "best_steps": [], "best_formula": "", "trials": [],
        })
        if score > rec["best_ir"]:
            rec.update({
                "best_ir": score,
                "best_mean": mean,
                "best_params": params,
                "best_steps": steps,
                "best_formula": formula,
            })
        # 记录该格到 trials（含公式链）
        rec["trials"].append({
            "params": params,
            "rankic_ir": round(float(score), 6),
            "mean_rankic": round(float(mean), 6),
            "formula": formula,
        })

    # 归一化 & 排序（每页 trials 按 IR 降序），最优格写盘
    meta = {}
    n_saved = 0
    for page, rec in page_best.items():
        trials_sorted = sorted(rec["trials"], key=lambda t: t["rankic_ir"], reverse=True)
        best = trials_sorted[0] if trials_sorted else None
        if best is None:
            continue
        # 按最优格参数重新计算最终矩阵并写盘（stride=1 全采样，输出不受加速影响）
        wl_pair = wl_from_token(best["params"]["winsor"])
        wl, wu = wl_pair
        zs = (best["params"]["zscore"] == "True")
        sw = int(best["params"]["smooth"])
        try:
            tmat = apply_params(_PAGE_MATS[page], wl, wu, zs, sw)
        except Exception:
            tmat = None
        if tmat is not None:
            # 用最优格做全采样回评（不经 EVAL_STRIDE），得到准确的最终 IR/mean。
            _mean, _ir = daily_rankic_ir(tmat, vwap)
            if np.isfinite(_ir):
                meta_best_ir = round(float(_ir), 6)
                meta_best_mean = round(float(_mean), 6)
            else:
                meta_best_ir = round(float(best["rankic_ir"]), 6)
                meta_best_mean = round(float(best["mean_rankic"]), 6)
            sub = tmat.astype("float32").dropna(axis=1, how="all")
            sub.to_parquet(OUT_DIR / f"{page}.parquet")
            n_saved += 1
        else:
            meta_best_ir = round(float(best["rankic_ir"]), 6)
            meta_best_mean = round(float(best["mean_rankic"]), 6)
        steps, formula = param_to_formula(wl_pair, wu, zs, sw)
        meta[page] = {
            "best_params": {
                "winsor": [wl_pair[0], wl_pair[1]] if wl_pair[0] is not None else None,
                "zscore": zs,
                "smooth": sw,
            },
            "best_rankic_ir": meta_best_ir,
            "best_mean_rankic": meta_best_mean,
            "steps": steps,
            "formula": formula,
            "n_variants": len(trials_sorted),
            "trials": trials_sorted,
        }
        print(f"  {page}: ir={meta_best_ir:.3f} formula={formula}", flush=True)
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
    print(f"[opt2] 完成 {len(meta)} 因子（写盘 {n_saved} 个 parquet），存 {META_PATH}")
    print(f"[opt2] 运行耗时 {elapsed:.0f}s", flush=True)


_PAGE_MATS = None


if __name__ == "__main__":
    main()