#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段3 shift(-2) 企业级口径：339 真 alpha 冗余聚类 + 质量门槛标注。

筛选：非代理（eval_shift2.json proxy=False）且 best_rankic_ir > 0 的因子（339 个）。
相关：每日横截面 Spearman（公共列 457，公共窗口 485 天，按列采样 800 股折衷）。
      缺值行用该日该因子有限值行内均值填充（补均值相当于秩平均，不引入负向偏误），
      保证每日满秩相关可用，避免"整对丢弃"造成的相关样本异质性。
聚类：|corr|>=0.85 建边 → SparseCorrelationGraph → ConnectedComponents(RESEARCH)。
门槛：best_rankic_ir > 0.5 且 best_mean_rankic > 0.03 → pass（仅标注，不删）。

输出：weekly_backtest_output/factor_clusters.json（覆盖旧 shift-1 时代产物）。
"""
import sys, os, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROJECT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "factor_assets"))

from factor_assets.graph.sparse import SparseCorrelationGraph, CorrelationEdge
from factor_assets.clustering.families import ConnectedComponents
from factor_assets.clustering.certification import ExecutionMode
from factor_assets.selection.gates import ThresholdGate, GateResult

FV_DIR = PROJECT / "weekly_backtest_output" / "optimized_factors"
META_P = PROJECT / "weekly_backtest_output" / "optimized_meta.json"
EVAL_P = PROJECT / "weekly_backtest_output" / "eval_shift2.json"
OUT = PROJECT / "weekly_backtest_output" / "factor_clusters.json"
LOG = Path("/tmp/agent_cluster3.log")

CORR_THRESHOLD = 0.85
IR_GATE = 0.5
RANKIC_GATE = 0.03
MIN_COLS = 30
SAMPLE_DAYS = 3      # 日期步长 3
SAMPLE_COLS = 800    # 列采样 800 股
RESUME = True


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] [stage3-shift2-339] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def compute_corr_blocks(names, common_cols, sample_idx, all_idx_full):
    """分块算 339×339 平均横截面秩相关。

    每日：每因子取当日行 → (n, cols)；逐行 rankdata（有限值）→ 按均值填�� NaN 后中心化。
    Cday = 中心化后向量两两余弦（即 Spearman 秩相关）。
    返回 (C, pages) 平均相关矩阵。
    """
    n = len(names)
    C = np.zeros((n, n), dtype=np.float64)
    cnt = np.zeros((n, n), dtype=np.float64)
    t0 = time.time()
    for di, t in enumerate(sample_idx):
        rows = np.empty((n, len(common_cols)), dtype=np.float64)
        ok = np.zeros(n, dtype=bool)
        for i, nm in enumerate(names):
            # 每因子仅当日一行：reindex 开销小
            s = mats[nm].loc[t].reindex(common_cols)
            v = s.to_numpy(dtype=np.float64, na_value=np.nan)
            m = np.isfinite(v)
            if m.sum() >= MIN_COLS:
                r = np.full(len(v), np.nan)
                r[m] = rankdata(v[m])
                # 均值填充缺失（补均值 = 秩平均）
                r[~m] = np.mean(r[m])
                rows[i] = r
                ok[i] = True
            # else 整行保持 NaN
        idx = np.where(ok)[0]
        if len(idx) < 2:
            continue
        sub = rows[idx]
        sub = np.where(np.isnan(sub), 0.0, sub)
        subc = sub - sub.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(subc, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        subn = subc / norms
        Cday = subn @ subn.T
        C[np.ix_(idx, idx)] += Cday
        cnt[np.ix_(idx, idx)] += 1.0
        if (di + 1) % 20 == 0:
            log(f"[corr] day {di + 1}/{len(sample_idx)} 耗时{time.time() - t0:.0f}s")
    C = np.divide(C, np.maximum(cnt, 1.0), out=np.zeros_like(C), where=cnt > 0)
    np.fill_diagonal(C, 1.0)
    np.clip(C, -1.0, 1.0, out=C)
    return C


if __name__ == "__main__":
    log("=== 阶段3 shift(-2) 339 真 alpha 聚类 + 门槛 ===")
    meta = json.loads(META_P.read_text())
    ev = json.loads(EVAL_P.read_text())
    names_all = sorted(os.path.basename(p)[: -len(".parquet")] for p in (FV_DIR.glob("*.parquet")))
    sel = sorted(k for k in meta if ev.get(k, {}).get("proxy") is False
                 and (meta[k].get("best_rankic_ir") or 0.0) > 0.0)
    sel = [k for k in sel if (FV_DIR / f"{k}.parquet").exists()]
    log(f"[sel] meta={len(meta)} files={len(names_all)} 非代理正IR={len(sel)}")

    import collections
    import pyarrow.parquet as pq
    colfreq = collections.Counter()
    for n in sel:
        pf = pq.ParquetFile(FV_DIR / f"{n}.parquet")
        colfreq.update(pf.schema.names)
    common_cols = sorted([c for c, k in colfreq.items() if k == len(sel)])
    log(f"[cols] 公共列(全{len(sel)}): {len(common_cols)}")
    if len(common_cols) < MIN_COLS:
        log(f"[cols] 公共列太少，退出"); sys.exit(1)

    # 公共日期轴
    all_idx = None
    for n in sel:
        ix = pd.read_parquet(FV_DIR / f"{n}.parquet", columns=[]).index
        all_idx = ix if all_idx is None else all_idx.intersection(ix)
    log(f"[dates] 公共日期 {len(all_idx)} ({all_idx.min()} ~ {all_idx.max()})")
    window = all_idx[all_idx >= pd.Timestamp("2019-01-02")]
    sample_idx = window[::SAMPLE_DAYS]
    log(f"[dates] 采样(步长{SAMPLE_DAYS}) {len(sample_idx)} 天")

    # 列采样
    if len(common_cols) > SAMPLE_COLS:
        rng = np.random.default_rng(42)
        cols_s = sorted(rng.choice(common_cols, size=SAMPLE_COLS, replace=False))
        log(f"[cols] 列采样 {len(common_cols)} -> {SAMPLE_COLS}")
    else:
        cols_s = common_cols

    # 读入内存（只存采样日行）
    t0 = time.time()
    mats = {}
    for i, n in enumerate(sel):
        try:
            m = pd.read_parquet(FV_DIR / f"{n}.parquet", columns=cols_s)
            mats[n] = m.astype(np.float32)
        except Exception:
            log(f"[load] FAIL {n}: {sys.exc_info()[1]}"); continue
        if (i + 1) % 100 == 0:
            log(f"[load] {i + 1}/{len(sel)}")
    sel = [n for n in sel if n in mats]
    log(f"[load] 读入 {len(sel)}/{len(names_all)}, 耗时{time.time() - t0:.0f}s")

    C = compute_corr_blocks(sel, cols_s, sample_idx, all_idx)
    off = C[np.triu_indices(len(sel), k=1)]
    log(f"[corr] {len(sel)}×{len(sel)} 完成 耗时{time.time() - t0:.0f}s, "
        f"平均|offdiag|={np.abs(off).mean():.4f}, 均值={off.mean():.4f}, max|off|={np.abs(off).max():.4f}")

    # 聚类
    pairs = np.argwhere(np.abs(C) >= CORR_THRESHOLD)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    edges = [CorrelationEdge(sel[a], sel[b], float(C[a, b])) for a, b in pairs]
    log(f"[graph] 边数(|corr|>={CORR_THRESHOLD}): {len(edges)}")
    graph = SparseCorrelationGraph(edges, nodes=set(sel))
    log(f"[graph] {graph.node_count} nodes, {graph.edge_count} edges, density {graph.density():.4f}")

    cc = ConnectedComponents(graph, execution_mode=ExecutionMode.RESEARCH)
    res = cc.find_components()
    assignments = res.assignments
    members = {}
    for f, cid in assignments.items():
        members.setdefault(cid, []).append(f)
    order = sorted(members.keys(), key=lambda c: (-len(members[c]), c))
    cluster_members, page_to_cluster, representatives = {}, {}, {}
    for new_id, old_id in enumerate(order):
        mem = sorted(members[old_id])
        cid = f"cluster_{new_id}"
        cluster_members[cid] = mem
        for m in mem:
            page_to_cluster[m] = cid
        best = max(mem, key=lambda x: (meta.get(x, {}).get("best_rankic_ir") or 0.0,
                                       meta.get(x, {}).get("best_mean_rankic") or 0.0))
        representatives[cid] = {
            "factor": best,
            "best_rankic_ir": meta.get(best, {}).get("best_rankic_ir"),
            "best_mean_rankic": meta.get(best, {}).get("best_mean_rankic"),
            "cluster_size": len(mem),
        }
    log(f"[cluster] {len(cluster_members)} 簇")

    # 质量门槛
    ir_gate = ThresholdGate("rankic_ir_min", IR_GATE, higher_is_better=True, gate_version="2026-08-29")
    rankic_gate = ThresholdGate("best_mean_rankic_min", RANKIC_GATE, higher_is_better=True, gate_version="2026-08-29")
    quality_gate = {}
    n_pass = 0
    for name in sel:
        ir = meta.get(name, {}).get("best_rankic_ir")
        rc = meta.get(name, {}).get("best_mean_rankic")
        if ir is None or rc is None:
            quality_gate[name] = {"best_mean_rankic": rc, "best_rankic_ir": ir,
                                  "gate_ir": "SKIP", "gate_rankic": "SKIP",
                                  "quality_gate": "fail", "reason": "meta 无有效指标"}
            continue
        e_ir = ir_gate.evaluate(name, evidence_id="optimized_meta.json", metric_name="best_rankic_ir", metric_value=ir)
        e_rc = rankic_gate.evaluate(name, evidence_id="optimized_meta.json", metric_name="best_mean_rankic", metric_value=rc)
        passed = (e_ir.result == GateResult.PASS) and (e_rc.result == GateResult.PASS)
        if passed:
            n_pass += 1
        quality_gate[name] = {
            "best_mean_rankic": rc, "best_rankic_ir": ir,
            "mean_rankic": rc, "rankic_ir": ir,
            "gate_ir": e_ir.result.value, "gate_rankic": e_rc.result.value,
            "quality_gate": "pass" if passed else "fail",
        }

    sizes = sorted([len(v) for v in cluster_members.values()], reverse=True)
    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "shift(-2) 企业级口径 339 真 alpha（非代理 + 正IR）",
        "source_dir": str(FV_DIR),
        "meta_file": str(META_P),
        "eval_file": str(EVAL_P),
        "n_factors_total": len(names_all),
        "n_factors_clustered": len(sel),
        "corr_method": f"cross-sectional spearman (daily, {len(cols_s)} sampled cols of {len(common_cols)}, "
                       f"{len(sample_idx)} sampled days step {SAMPLE_DAYS}, mean-filled ranks)",
        "corr_threshold": CORR_THRESHOLD,
        "clustering": "factor_assets.clustering.families.ConnectedComponents (RESEARCH)",
        "graph": {"nodes": graph.node_count, "edges": graph.edge_count, "density": graph.density()},
        "num_clusters": len(cluster_members),
        "cluster_sizes_top20": sizes[:20],
        "cluster_members": cluster_members,
        "page_to_cluster": page_to_cluster,
        "representatives": representatives,
        "quality_gate_config": {"rankic_ir_min": IR_GATE, "best_mean_rankic_min": RANKIC_GATE, "basis_date": "2026-08-29"},
        "quality_gate": quality_gate,
        "summary": {
            "n_factors_total": len(names_all),
            "n_factors_clustered": len(sel),
            "n_clusters": len(cluster_members),
            "largest_cluster": sizes[0] if sizes else 0,
            "n_singletons": sum(1 for v in cluster_members.values() if len(v) == 1),
            "gate_pass": n_pass,
            "gate_fail": len(sel) - n_pass,
            "gate_pass_rate": round(n_pass / max(len(sel), 1), 4),
        },
    }
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    tmp.replace(OUT)
    log(f"[done] 写 {OUT}, 总耗时{time.time() - t0:.0f}s")

    log("===== 聚类统计 =====")
    log(f"簇数量: {len(cluster_members)}")
    log(f"最大簇: {sizes[0] if sizes else 0} (Top10: {sizes[:10]})")
    log(f"单例簇: {out['summary']['n_singletons']}")
    log("===== 代表因子 Top20 (按IR降序) =====")
    reps = sorted([(cid, r["factor"], r["best_rankic_ir"] or 0, r["best_mean_rankic"] or 0, r["cluster_size"])
                   for cid, r in representatives.items()], key=lambda x: -x[2])
    for cid, rep, ir, ic, size in reps[:20]:
        log(f"  {cid:10s} {rep:55s} ir={ir:.4f} mean_rankic={ic:.4f} 簇大小={size}")
    log("===== 质量门槛 =====")
    log(f"gate pass: {n_pass}/{len(sel)} ({n_pass / len(sel) * 100:.1f}%)")
    passed_list = sorted([(p, v["best_rankic_ir"] or 0, v["best_mean_rankic"] or 0)
                          for p, v in quality_gate.items() if v.get("quality_gate") == "pass"], key=lambda x: -x[1])
    for p, ir, ic in passed_list[:40]:
        log(f"  {p:55s} ir={ir:.4f} mean_rankic={ic:.4f}")
