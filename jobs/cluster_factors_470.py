#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段3 重跑 v3：470 因子冗余聚类 + 质量门槛标注（含 61 个新补齐因子）。
横截面 Spearman 相关（每日横截面秩相关，对齐公共列 457，逐日向量化 matmul）。

方法：
  1. 读全部 470 个 optimized_factors/*.parquet，对齐公共列（457 只），
     公共窗口 2019-01-02 起最近 500 天。
  2. 每日横截面 Spearman 相关 → 平均相关矩阵 470×470。
  3. |corr|>=0.85 建边 → SparseCorrelationGraph → ConnectedComponents(RESEARCH)。
  4. ThresholdGate 质量门槛（rankic_ir>0.5 且 best_mean_rankic>0.015），全 470 标注。
  5. 写 factor_clusters.json。

修复 v2：61 个新因子列数不同（499/500/全量），改为横截面相关（按日对齐公共列），
不再展平；公共列 457 保证 470 因子全部可算。
"""
import sys, os, json, time, glob
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
OUT = PROJECT / "weekly_backtest_output" / "factor_clusters.json"
LOG = Path("/tmp/agent_cluster2.log")

CORR_THRESHOLD = 0.85
LAST_N_DAYS = 500
IR_GATE = 0.5
RANKIC_GATE = 0.015
MIN_COLS = 30


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] [stage3-470] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def main():
    log("=== 阶段3 重跑 v3: 470 横截面聚类 + 门槛 ===")
    meta = json.loads(META_P.read_text())
    names = sorted(os.path.basename(f)[: -len(".parquet")] for f in glob.glob(str(FV_DIR / "*.parquet")))
    log(f"[meta] {len(meta)}, [files] {len(names)}")

    # 1. 公共列（所有 470 都有）
    import pyarrow.parquet as pq
    import collections
    colfreq = collections.Counter()
    for n in names:
        pf = pq.ParquetFile(FV_DIR / f"{n}.parquet")
        colfreq.update(pf.schema.names)
    common_cols = sorted([c for c, k in colfreq.items() if k == len(names)])
    log(f"[cols] 公共列(全470): {len(common_cols)}")
    if len(common_cols) < MIN_COLS:
        log(f"[cols] 公共列太少 {len(common_cols)}，退出")
        return

    # 2. 读矩阵，对齐公共列，取最近 LAST_N_DAYS
    t0 = time.time()
    mats = {}
    for i, n in enumerate(names):
        try:
            m = pd.read_parquet(FV_DIR / f"{n}.parquet", columns=common_cols)
        except Exception:
            log(f"[load] FAIL {n}: {sys.exc_info()[1]}")
            continue
        mats[n] = m.astype(np.float32)
        if (i + 1) % 100 == 0:
            log(f"[load] {i + 1}/{len(names)}")
    log(f"[load] 读入 {len(mats)}/{len(names)}, 耗时{time.time()-t0:.0f}s")

    # 公共窗口日期
    all_idx = None
    for n in mats:
        ix = mats[n].index
        all_idx = ix if all_idx is None else all_idx.intersection(ix)
    window = all_idx[all_idx >= pd.Timestamp("2019-01-02")]
    sample_idx = window[-LAST_N_DAYS:] if len(window) > LAST_N_DAYS else window
    log(f"[window] 公共日期 {len(window)}, 采样 {len(sample_idx)} 天 ({sample_idx.min()} ~ {sample_idx.max()})")

    # 3. 组装 3D 数组 (n, days, cols)，横截面秩相关
    n = len(names)
    days = len(sample_idx)
    cols = len(common_cols)
    X = np.full((n, days, cols), np.nan, dtype=np.float32)
    for i, nm in enumerate(names):
        X[i] = mats[nm].reindex(sample_idx).values.astype(np.float32)
    log(f"[X] {X.shape}, NaN占比 {np.isnan(X).mean():.4f}")

    # 横截面 rank（沿 cols 轴）
    R = np.empty_like(X, dtype=np.float64)
    for i in range(n):
        for d in range(days):
            row = X[i, d]
            mask = np.isfinite(row)
            if mask.sum() >= MIN_COLS:
                R[i, d, mask] = rankdata(row[mask])
            else:
                R[i, d] = np.nan
    log(f"[rank] 完成 耗时{time.time()-t0:.0f}s")

    # 每日 (n, cols) → 平均相关 (n,n)
    Rt = np.transpose(R, (1, 0, 2))  # (days, n, cols)
    Rt = np.where(np.isnan(Rt), 0.0, Rt)
    Rc = Rt - Rt.mean(axis=2, keepdims=True)
    norms = np.linalg.norm(Rc, axis=2, keepdims=True)
    norms[norms == 0] = 1.0
    Rn = Rc / norms
    Cday = Rn @ Rn.transpose(0, 2, 1)  # (days, n, n)
    C = Cday.mean(axis=0)
    np.fill_diagonal(C, 1.0)
    np.clip(C, -1.0, 1.0, out=C)
    log(f"[corr] 470×470 完成 耗时{time.time()-t0:.0f}s, 平均|offdiag|={np.abs(C - np.eye(n)).mean():.4f}")

    # 4. 聚类
    pairs = np.argwhere(np.abs(C) >= CORR_THRESHOLD)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    edges = [CorrelationEdge(names[a], names[b], float(C[a, b])) for a, b in pairs]
    log(f"[graph] 边数(|corr|>={CORR_THRESHOLD}): {len(edges)}")
    graph = SparseCorrelationGraph(edges, nodes=set(names))
    log(f"[graph] {graph.node_count} nodes, {graph.edge_count} edges, density {graph.density():.4f}")

    cc = ConnectedComponents(graph, execution_mode=ExecutionMode.RESEARCH)
    res = cc.find_components()
    assignments = res.assignments

    members = {}
    for f, cid in assignments.items():
        members.setdefault(cid, []).append(f)
    order = sorted(members.keys(), key=lambda c: (-len(members[c]), c))
    cluster_members = {}
    page_to_cluster = {}
    representatives = {}
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

    # 5. 质量门槛
    ir_gate = ThresholdGate("rankic_ir_min", IR_GATE, higher_is_better=True, gate_version="2026-08-28")
    rankic_gate = ThresholdGate("best_mean_rankic_min", RANKIC_GATE, higher_is_better=True, gate_version="2026-08-28")
    quality_gate = {}
    n_pass = 0
    for name in names:
        ir = meta.get(name, {}).get("best_rankic_ir")
        rc = meta.get(name, {}).get("best_mean_rankic")
        if ir is None or rc is None:
            quality_gate[name] = {
                "best_mean_rankic": rc, "best_rankic_ir": ir,
                "gate_ir": "SKIP", "gate_rankic": "SKIP",
                "quality_gate": "fail", "reason": "meta 无有效指标",
            }
            continue
        e_ir = ir_gate.evaluate(name, evidence_id="optimized_meta.json", metric_name="best_rankic_ir", metric_value=ir)
        e_rc = rankic_gate.evaluate(name, evidence_id="optimized_meta.json", metric_name="best_mean_rankic", metric_value=rc)
        passed = (e_ir.result == GateResult.PASS) and (e_rc.result == GateResult.PASS)
        if passed:
            n_pass += 1
        quality_gate[name] = {
            "best_mean_rankic": rc, "best_rankic_ir": ir,
            "gate_ir": e_ir.result.value, "gate_rankic": e_rc.result.value,
            "quality_gate": "pass" if passed else "fail",
        }

    sizes = sorted([len(v) for v in cluster_members.values()], reverse=True)
    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_dir": str(FV_DIR),
        "meta_file": str(META_P),
        "n_factors_total": len(names),
        "n_factors_clustered": len(names),
        "corr_method": f"cross-sectional spearman (daily, {len(common_cols)} common cols, last {len(sample_idx)} days)",
        "corr_threshold": CORR_THRESHOLD,
        "clustering": "factor_assets.clustering.families.ConnectedComponents (RESEARCH)",
        "graph": {"nodes": graph.node_count, "edges": graph.edge_count, "density": graph.density()},
        "num_clusters": len(cluster_members),
        "cluster_sizes_top20": sizes[:20],
        "cluster_members": cluster_members,
        "page_to_cluster": page_to_cluster,
        "representatives": representatives,
        "quality_gate_config": {"rankic_ir_min": IR_GATE, "best_mean_rankic_min": RANKIC_GATE, "basis_date": "2026-08-28"},
        "quality_gate": quality_gate,
        "summary": {
            "n_factors_total": len(names),
            "n_factors_clustered": len(names),
            "n_clusters": len(cluster_members),
            "largest_cluster": sizes[0] if sizes else 0,
            "n_singletons": sum(1 for v in cluster_members.values() if len(v) == 1),
            "gate_pass": n_pass,
            "gate_fail": len(names) - n_pass,
            "gate_pass_rate": round(n_pass / max(len(names), 1), 4),
        },
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    log(f"[done] 写 {OUT}, 总耗时{time.time()-t0:.0f}s")

    # 报告
    log("===== 聚类统计 =====")
    log(f"簇数量: {len(cluster_members)}")
    log(f"最大簇: {sizes[0] if sizes else 0} (Top10: {sizes[:10]})")
    log(f"单例簇: {out['summary']['n_singletons']}")
    log("===== 代表因子 Top20 =====")
    reps = sorted(
        [(cid, r["factor"], r["best_rankic_ir"] or 0, r["best_mean_rankic"] or 0, r["cluster_size"])
         for cid, r in representatives.items()],
        key=lambda x: -x[2],
    )
    for cid, rep, ir, ic, size in reps[:20]:
        log(f"  {cid:10s} {rep:55s} ir={ir:.4f} mean_rankic={ic:.4f} 簇大小={size}")
    log("===== 质量门槛 =====")
    log(f"gate pass: {n_pass}/{len(names)} ({n_pass / len(names) * 100:.1f}%)")
    passed_list = sorted(
        [(p, v["best_rankic_ir"] or 0, v["best_mean_rankic"] or 0) for p, v in quality_gate.items() if v.get("quality_gate") == "pass"],
        key=lambda x: -x[1],
    )
    for p, ir, ic in passed_list[:40]:
        log(f"  {p:55s} ir={ir:.4f} mean_rankic={ic:.4f}")


if __name__ == "__main__":
    main()
