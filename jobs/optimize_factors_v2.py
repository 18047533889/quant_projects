#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段1v2：456 因子预处理自动化 + 择优 —— factor_optimizer SearchRunner 统一版。

对每个因子生成 4 个预处理变体（raw / winsor / winsor_zscore / winsor_zscore_neutral），
用 factor_optimizer 的 SearchSpace + GridSearch 把「变体择优」建模为离散参数搜索：
  - SearchSpace: page(choice) × treatment(choice, 4 token)   → 每页 4 格，全网格 = 456×4
  - SearchBudget(max_trials = max_evaluations = strat.total_combinations) 全网格枚举
    （吸取阶段2 max_trials=100 漏格教训：预算必须 ≥ 组合数）
  - plateau 关闭（window=1e6），multifidelity 关闭，execution_mode=research_only
  - evaluation_fn = 预处理 + 逐日 RankIC IR（vwap-to-vwap 口径，全局固定）
    neutral 格成本高 → 搜索阶段 stride=4 降采样打分，胜出格再 stride=1 全周期重打。

输出（与旧版目录/格式对齐，便于下游阶段2/4 无缝衔接）：
  weekly_backtest_output/optimized_factors/<page>.parquet   （最优变体矩阵）
  weekly_backtest_output/optimized_meta.json                 （含 SearchSession 审计字段）

多进程：按 page 分块（每 chunk 32 页），每 worker 只载自己 chunk 的矩阵，
共享 FWD 收益矩阵与 neutralize 设计矩阵（fork COW 继承），防 8×80MB 爆内存。

用法：
  PYTHONUNBUFFERED=1 nohup .venv/bin/python jobs/optimize_factors_v2.py > /tmp/opt1v2.log 2>&1 &
"""
import sys, os, json, time, gc, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
warnings.filterwarnings("ignore")

PROJECT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "factor_optimizer"))
sys.path.insert(0, str(PROJECT / "factor_preprocess"))

from scipy.stats import rankdata

from factor_optimizer.search.runner import SearchRunner, SearchConfig
from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.contracts.objective import ObjectiveSpec
from factor_optimizer.contracts.splits import SplitPlan, EvaluationProtocol
from factor_optimizer.search.strategies import SearchSpace, ParameterSpace, GridSearch

FV_DIR = PROJECT / "weekly_backtest_output" / "factor_matrices_all"
OUT_DIR = PROJECT / "weekly_backtest_output" / "optimized_factors"
META_PATH = PROJECT / "weekly_backtest_output" / "optimized_meta.json"
LQTP_ALL = json.load(open("/home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json"))
DAILY_ADJ = Path.home() / "cos_data" / "StockDailyBarAdj"
INDUSTRY = Path.home() / "cos_data" / "StockIndustry"
VALUATION = Path.home() / "cos_data" / "StockValuationDaily"
START, END = "2019-01-02", "2026-08-24"
MIN_UNIVERSE = 20
MIN_VALID_ROWS = 30  # neutralize 行覆盖阈值
CHUNK_SIZE = 32
N_WORKERS = 8
NEUTRAL_STRIDE = 4   # neutral 格搜索阶段降采样

# treatment token -> (winsor_pair, zscore, neutralize, steps)
TREATMENTS = {
    "raw":                   (None,             False, False, ["原始因子"]),
    "winsor_001":            ((0.01, 0.99),     False, False, ["cs_winsor(0.01, 0.99)"]),
    "winsor_zscore_001":     ((0.01, 0.99),     True,  False, ["cs_winsor(0.01, 0.99)", "cs_zscore"]),
    "winsor_zscore_neutral_001": ((0.01, 0.99), True,  True,  ["cs_winsor(0.01, 0.99)", "cs_zscore", "ols_neutralize(行业+log市值)"]),
}
TOKEN2NAME = {k: {"raw": "raw", "winsor_001": "winsor",
                  "winsor_zscore_001": "winsor_zscore",
                  "winsor_zscore_neutral_001": "winsor_zscore_neutral"}[k] for k in TREATMENTS}

# 全局共享数据（fork 后 worker 继承）
_GV = {"FWD": None, "row_index": None, "n_days": 0, "universe": None,
       "neutral_bundle": None}


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_vwap_matrix():
    """全周期 后复权 AdjVwap 矩阵 + 前向收益 FWD（vwap-to-vwap 后复权硬性）。
    数据源：StockDailyBarAdj.AdjVwap（后复权），禁止未复权 StockDailyBar.Vwap。"""
    files = sorted(DAILY_ADJ.glob("*.parquet"))
    fs = "[" + ",".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT TradeDate as date, Symbol as symbol, AdjVwap as vwap
        FROM read_parquet({fs})
        WHERE TradeDate >= DATE '{START}' AND TradeDate <= DATE '{END}'
    """).df()
    vwap = df.pivot_table(index='date', columns='symbol', values='vwap', aggfunc='first')
    vwap.index = pd.to_datetime(vwap.index)
    vwap = vwap.sort_index().astype(np.float32)
    va = vwap.values.astype(np.float64)
    FWD = np.full_like(va, np.nan)
    FWD[:-1] = va[1:] / va[:-1] - 1.0
    return vwap, FWD


def load_industry():
    files = sorted(INDUSTRY.glob("*.parquet"))
    fs = "[" + ",".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT TradeDate as date, Symbol as symbol, IndustryName
        FROM read_parquet({fs})
        WHERE IndustrySource = 'sw_l1' AND TradeDate >= DATE '{START}' AND TradeDate <= DATE '{END}'
    """).df()
    df = df.drop_duplicates(subset=['date', 'symbol'], keep='first')
    m = df.pivot_table(index='date', columns='symbol', values='IndustryName', aggfunc='first')
    m.index = pd.to_datetime(m.index)
    return m.sort_index()


def load_mktcap():
    files = sorted(VALUATION.glob("*.parquet"))
    fs = "[" + ",".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT TradeDate as date, Symbol as symbol, CirculatingMarketCap
        FROM read_parquet({fs})
        WHERE TradeDate >= DATE '{START}' AND TradeDate <= DATE '{END}'
    """).df()
    df["log_mktcap"] = np.log(df["CirculatingMarketCap"].replace(0, np.nan))
    m = df.pivot_table(index='date', columns='symbol', values='log_mktcap', aggfunc='first')
    m.index = pd.to_datetime(m.index)
    return m.sort_index()


def build_neutral_bundle(vwap):
    """全量构建一次 neutralize 静态设计矩阵（行业哑变量 + log市值）。
    所有 worker 共享（fork COW 只读），只构建一次。"""
    universe = list(vwap.columns)
    ind = load_industry().reindex(index=vwap.index, columns=universe)
    mc = load_mktcap().reindex(index=vwap.index, columns=universe)
    ind_mat = ind.fillna("").values
    uniq = sorted({x for row in ind_mat for x in row if x})
    T, N = ind_mat.shape
    K = len(uniq) + 2
    X = np.zeros((T, N, K), dtype=np.float64)
    X[:, :, 0] = 1.0
    for i, name in enumerate(uniq):
        X[:, :, 1 + i] = (ind_mat == name).astype(np.float64)
    X[:, :, 1 + len(uniq)] = mc.values
    return {"X": X, "uniq": uniq, "K": K, "T": T, "N": N}


def neutralize_array_bundle(arr, bundle):
    """对已对齐全宇宙 float64 数组做逐日 OLS 残差（行业哑变量 + log市值）。"""
    X = bundle["X"]
    T, N = arr.shape
    resid = np.full((T, N), np.nan)
    for t in range(T):
        y = arr[t]; Xt = X[t]
        ok = np.isfinite(y) & np.isfinite(Xt).all(axis=1)
        if ok.sum() < MIN_VALID_ROWS:
            continue
        try:
            coef, _, _, _ = np.linalg.lstsq(Xt[ok], y[ok], rcond=None)
            resid[t, ok] = y[ok] - Xt[ok] @ coef
        except Exception:
            pass
    return resid


def detect_dsl_preproc(page):
    """检测 DSL 是否已含预处理算子（meta 审计字段）。"""
    FE_PREPROC_OPS = [
        "cs_zscore", "cs_rank", "cs_demean", "cs_winsorize", "cs_winsor",
        "neutralize", "industry_neutral", "market_cap_neutralize", "size_neutralize",
        "standardize", "zscore", "rank", "cs_scale", "winsorize", "normalize",
        "c_zscore", "c_rank", "c_demean", "c_winsorize", "c_scale",
    ]
    rec = next((r for r in LQTP_ALL if r.get("page_name") == page), None)
    if not rec:
        return []
    dsl = (rec.get("dsl") or "") + " " + (rec.get("lqtp_formula") or "") + " " + (rec.get("fe_formula") or "")
    dsl_l = dsl.lower()
    return [op for op in FE_PREPROC_OPS if op.lower() in dsl_l]


# ---------------------------------------------------------------------------
# treatment 应用 + 逐日 RankIC IR
# ---------------------------------------------------------------------------

def apply_treatment(arr, token):
    """在已对齐全宇宙 float32 arr 上应用 treatment。返回 float64 结果数组。"""
    wl_pair, zs, neu, _ = TREATMENTS[token]
    out = arr
    if wl_pair is not None:
        lo_q = np.nanpercentile(out, wl_pair[0] * 100, axis=1, keepdims=True)
        hi_q = np.nanpercentile(out, wl_pair[1] * 100, axis=1, keepdims=True)
        out = np.clip(out, lo_q, hi_q)
    if zs:
        mu = np.nanmean(out, axis=1, keepdims=True)
        sd = np.nanstd(out, axis=1, keepdims=True)
        out = (out - mu) / sd
    out = np.asarray(out, dtype=np.float64)
    if neu:
        out = neutralize_array_bundle(out, _GV["neutral_bundle"])
    return out


def rankic_ir_fast(arr, n_days, stride=1):
    """逐日 spearman rankic（vwap-to-vwap），返回 (mean_rankic, rankic_ir, n_used)。
    stride>1 为半成本降采样（只取每 stride 天的样本）。数组已按全宇宙对齐到 FWD。"""
    fwd = _GV["FWD"]
    ics = []
    for t in range(0, n_days, stride):
        m = arr[t]; r = fwd[t]
        mask = np.isfinite(m) & np.isfinite(r)
        if mask.sum() < MIN_UNIVERSE:
            continue
        ra = rankdata(m[mask]); rb = rankdata(r[mask])
        am = ra - ra.mean(); bm = rb - rb.mean()
        d = np.sqrt((am * am).sum() * (bm * bm).sum())
        ics.append((am * bm).sum() / d if d > 1e-18 else 0.0)
    if not ics:
        return 0.0, 0.0, 0
    s = np.asarray(ics)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return 0.0, 0.0, 0
    mean = float(s.mean()); std = float(s.std())
    return mean, (mean / std if std > 1e-9 else 0.0), int(len(s))


# ---------------------------------------------------------------------------
# worker：chunk 网格
# ---------------------------------------------------------------------------

def process_chunk(pages):
    """worker：载 chunk 的矩阵，跑该 chunk 的 SearchRunner 网格，返回 {page: meta}。"""
    g = _GV
    FWD = g["FWD"]; n_days = g["n_days"]; universe = g["universe"]
    row_index = g["row_index"]
    n_universe = len(universe)
    col_pos = {c: i for i, c in enumerate(universe)}

    # 载 chunk 矩阵 → 对齐全宇宙 float32 数组
    page_arr = {}
    for page in pages:
        fpath = FV_DIR / f"{page}.parquet"
        try:
            mat = pd.read_parquet(fpath)
        except Exception:
            continue
        if mat.shape[1] == 0 or mat.isna().all().all():
            continue
        cols = [c for c in universe if c in mat.columns]
        if not cols:
            continue
        common = mat.index.intersection(row_index)
        if len(common) == 0:
            continue
        sub = mat.reindex(index=row_index)[cols].astype(np.float32)
        arr = np.full((n_days, n_universe), np.nan, dtype=np.float32)
        arr[:, [col_pos[c] for c in cols]] = sub.values
        page_arr[page] = arr
    if not page_arr:
        return {}
    del mat, sub
    gc.collect()

    pages_list = list(page_arr)
    tokens = list(TREATMENTS)
    space = SearchSpace(parameters=[
        ParameterSpace("page", "choice", choices=pages_list),
        ParameterSpace("treatment", "choice", choices=tokens),
    ])
    spec = ObjectiveSpec(metric_name="rankic_ir", direction="maximize")
    strat = GridSearch(space=space, objective_spec=spec)
    grid_total = int(strat.total_combinations)  # pages × 4
    cfg = SearchConfig(
        budget=SearchBudget(max_trials=grid_total, max_evaluations=grid_total, max_cost_units=1e9),
        objective_spec=spec,
        plateau_window=1000000,
        plateau_threshold=0.0,
        enable_multifidelity=False,
        execution_mode="research_only",
        require_evaluation_protocol=False,
    )
    plan = SplitPlan(
        split_id=f"op1v2_{abs(hash(tuple(pages))) % 100000}",
        train_mask=[True] * n_days,
        validation_mask=[False] * n_days,
        test_mask=[False] * n_days,
        metadata={"dataset": "optimize_factors_v2"},
    )

    def evaluation_fn(trial, fidelity):
        params = trial.metadata["params"]
        page = params["page"]; token = params["treatment"]
        arr = page_arr[page]
        out = apply_treatment(arr, token)
        stride = NEUTRAL_STRIDE if token.endswith("neutral") else 1
        mean, ir, n = rankic_ir_fast(out, n_days, stride=stride)
        if not np.isfinite(ir):
            ir = 0.0
        return {"score": float(ir), "evidence_ref": f"{page}:{token}"}

    proto = EvaluationProtocol(split_plan=plan, evaluator=evaluation_fn)
    runner = SearchRunner(config=cfg, proposal_fn=strat.propose, evaluation_fn=proto, strategy=strat)
    session = runner.run(f"op1v2_{abs(hash(tuple(pages))) % 100000}")

    # 按页聚合：每格全周期重打分（neutral 也 stride=1），取最高 IR 格为最优。
    # 若某页全 4 格都是负 IR（或无有效 trials），SearchRunner 的主循环用时过大；
    # 当 chunk 仅含全负页时，我们仍要产出 {page: {best_tok, best_steps, variants, ...}}
    # 使 meta 完整（best_tok 可能为 raw 兜底）。这里沿用 session.trials 原始打分聚合。
    page_best = {}
    seen_success = 0
    for trial in session.trials:
        if not trial.is_successful():
            continue
        seen_success += 1
        params = trial.metadata.get("params", {})
        page = params.get("page"); token = params.get("treatment")
        if page not in page_arr or token not in TREATMENTS:
            continue
        out = apply_treatment(page_arr[page], token)
        mean, ir, n = rankic_ir_fast(out, n_days, stride=1)
        if not np.isfinite(ir):
            ir = 0.0
        rec = page_best.setdefault(page, {"best_ir": -1e18, "variants": {}, "trials": []})
        rec["variants"][token] = {"mean_rankic": round(mean, 6), "rankic_ir": round(ir, 6), "n_days": n}
        rec["trials"].append({"treatment": token, "rankic_ir": round(ir, 6),
                              "mean_rankic": round(mean, 6), "steps": TREATMENTS[token][3]})
        if ir > rec["best_ir"]:
            rec.update({"best_ir": ir, "best_mean": mean, "best_tok": token,
                        "best_steps": TREATMENTS[token][3]})
    if not seen_success:
        # chunk 内全部页无成功 trials（不应发生，保护性兜底）——直接按 raw 给占位
        for page in pages_list:
            page_best.setdefault(page, {"best_ir": -1e18, "variants": {}, "trials": []})
            rec = page_best[page]
            mean, ir, n = rankic_ir_fast(page_arr[page], n_days, stride=1)
            rec.update({"best_ir": ir if np.isfinite(ir) else 0.0,
                        "best_mean": mean, "best_tok": "raw", "best_steps": TREATMENTS["raw"][3]})
            rec["variants"]["raw"] = {"mean_rankic": round(mean, 6), "rankic_ir": round(float(ir if np.isfinite(ir) else 0.0), 6), "n_days": n}
            rec["trials"].append({"treatment": "raw", "rankic_ir": round(float(ir if np.isfinite(ir) else 0.0), 6),
                                  "mean_rankic": round(mean, 6), "steps": TREATMENTS["raw"][3]})

    # 写盘最优格矩阵 + 组装 meta
    out_meta = {}
    for page, rec in page_best.items():
        if rec.get("best_tok") is None:
            # 保护性兜底：仍写 raw（原始矩阵），保证 456 页 parquet 全非空
            rec.update({"best_ir": 0.0, "best_mean": 0.0, "best_tok": "raw",
                        "best_steps": TREATMENTS["raw"][3]})
        best_mat = apply_treatment(page_arr[page], rec["best_tok"])
        df_out = pd.DataFrame(best_mat, index=row_index, columns=universe).astype("float32")
        df_out = df_out.dropna(axis=1, how="all")
        df_out.to_parquet(OUT_DIR / f"{page}.parquet")
        variants_meta = {TOKEN2NAME[k]: v for k, v in rec["variants"].items()}
        out_meta[page] = {
            "best": TOKEN2NAME[rec["best_tok"]],
            "best_mean_rankic": round(rec["best_mean"], 6),
            "best_rankic_ir": round(rec["best_ir"], 6),
            "steps": rec["best_steps"],
            "best_treatment_token": rec["best_tok"],
            "dsl_preproc_ops": detect_dsl_preproc(page),
            "variants": variants_meta,
            "search": {
                "engine": "factor_optimizer.SearchRunner",
                "strategy": "grid",
                "session_id": session.session_id,
                "objective": {"metric_name": "rankic_ir", "direction": "maximize"},
                "budget": {"max_trials": grid_total, "max_evaluations": grid_total,
                           "n_cells_per_page": len(tokens)},
                "n_trials": len(rec["trials"]),
                "stop_reason": session.stop_reason,
                "trials": rec["trials"],
            },
        }
    return out_meta


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vwap, FWD = load_vwap_matrix()
    n_days, n_universe = vwap.shape
    universe = list(vwap.columns)
    row_index = vwap.index
    print(f"[opt1v2] vwap {vwap.shape} ({START}~{END})", flush=True)

    # neutralize 设计矩阵（所有页共享，构建一次 ≤15s）
    t0 = time.time()
    neutral_bundle = build_neutral_bundle(vwap)
    print(f"[opt1v2] neutralize 设计矩阵 K={neutral_bundle['K']} 构建 {time.time()-t0:.0f}s", flush=True)

    global _GV
    _GV = {"FWD": FWD, "row_index": row_index, "n_days": n_days,
           "universe": universe, "neutral_bundle": neutral_bundle}

    all_names = sorted([f.stem for f in FV_DIR.glob("*.parquet")]) if FV_DIR.exists() else []
    already = {p.stem for p in OUT_DIR.glob("*.parquet")} if OUT_DIR.exists() else set()
    old_meta = {}
    if META_PATH.exists():
        try:
            old_meta = json.loads(META_PATH.read_text())
        except Exception:
            old_meta = {}
    todo = [p for p in all_names if p not in already]
    print(f"[opt1v2] 因子数 {len(all_names)}, 待算 {len(todo)}, 已写 {len(already)}", flush=True)
    if not todo:
        print("[opt1v2] 无可算因子", flush=True)
        return
    print(f"[opt1v2] 网格: {len(todo)} 页 × {len(TREATMENTS)} treatment = {len(todo)*len(TREATMENTS)} 格", flush=True)

    n_workers = min(N_WORKERS, (os.cpu_count() or 4) - 2)
    chunks = [todo[i:i + CHUNK_SIZE] for i in range(0, len(todo), CHUNK_SIZE)]
    meta = dict(old_meta)
    t0 = time.time()
    done = 0
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(process_chunk, pages): i for i, pages in enumerate(chunks)}
        for fut in as_completed(futures):
            try:
                chunk_meta = fut.result()
            except Exception as exc:
                print(f"[opt1v2] chunk {futures[fut]} 失败: {str(exc)[:150]}", flush=True)
                chunk_meta = {}
            meta.update(chunk_meta)
            done += len(chunk_meta)
            if done % 64 == 0 or done >= len(todo):
                print(f"[opt1v2] {done}/{len(todo)} 完成, 耗时{time.time()-t0:.0f}s", flush=True)

    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
    ok = [p for p, m in meta.items() if not m.get("error") and m.get("best_rankic_ir", 0) > 0]
    errs = [p for p, m in meta.items() if m.get("error")]
    print(f"[opt1v2] 完成 {len(ok)} 非零 IR 因子 (err {len(errs)}), meta 存 {META_PATH}, 总耗时{time.time()-t0:.0f}s", flush=True)
    import collections
    print("[opt1v2] best 分布:", dict(collections.Counter(m.get("best") for m in meta.values() if not m.get("error"))), flush=True)


if __name__ == "__main__":
    main()