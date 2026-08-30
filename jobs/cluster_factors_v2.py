#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段3：470 优化后因子冗余聚类 + 质量门槛标注（factor_assets 库）。

输入（只读，严禁修改）：
  weekly_backtest_output/optimized_factors/*.parquet   470 个 date×symbol 因子矩阵
  weekly_backtest_output/optimized_meta.json           每因子 best_mean_rankic / best_rankic_ir / steps

库调用（硬性）：
  factor_assets.graph.sparse.SparseCorrelationGraph + CorrelationEdge   -> 稀疏相关图
  factor_assets.clustering.families.ConnectedComponents                 -> RESEARCH 模式连通分量
  factor_assets.selection.gates.ThresholdGate                           -> 质量门槛标注

流程：
  1. 读 meta（等待 470 条，由外层轮询保证）。
  2. 读 470 个矩阵，对齐共同 index/columns，取最近 500 天 × 800 只股票抽样，
     展平成 470 × N 长向量，一次 matmul 求两两 Spearman 相关（秩变换后标准化）。
  3. |corr| >= 0.85 的边入稀疏图 -> ConnectedComponents(RESEARCH) 聚类。
  4. ThresholdGate(rankic_ir>0.5) + ThresholdGate(best_mean_rankic>0.015) 标 quality_gate。
     不删任何因子，只标注。
  5. 输出 weekly_backtest_output/factor_clusters.json。
"""
import sys, json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(PROJECT))

from factor_assets.graph.sparse import SparseCorrelationGraph, CorrelationEdge
from factor_assets.clustering.families import ConnectedComponents, ExecutionMode
from factor_assets.selection.gates import (
    ThresholdGate,
    MetricDirection,
    GateResult,
)

FV_DIR = PROJECT / "weekly_backtest_output" / "optimized_factors"
META = PROJECT / "weekly_backtest_output" / "optimized_meta.json"
OUT = PROJECT / "weekly_backtest_output" / "factor_clusters.json"
PROGRESS = Path("/tmp/agent_cluster.log")

CORR_THRESHOLD = 0.85
LAST_N_DAYS = 500
N_SYMBOL_SAMPLE = 800
IR_GATE = 0.5
RANKIC_GATE = 0.015


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] [stage3-cluster] {msg}"
    print(line, flush=True)
    try:
        with open(PROGRESS, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_sample_matrix():
    """读所有可用矩阵，取最近 500 天 × 800 列抽样，返回 (names, X)。

    X: (n_factors, LAST_N_DAYS * n_sel_cols) float64。
    注意：meta 有 470 条但 parquet 只有 409 个（写入进程已于 17:59 停止），
    缺失矩阵的因子不进相关图（无数据不可算相关），但仍会从 meta 做门槛标注。
    """
    import pandas as pd

    all_names = sorted(p.stem for p in FV_DIR.glob("*.parquet"))
    log(f"parquet 因子数: {len(all_names)}")

    # 列并集（用第一个矩阵的 index 做参考；所有矩阵都来自同一全市场重算）
    col_counts = {}
    idx_ref = None
    for p in FV_DIR.glob("*.parquet"):
        m = pd.read_parquet(p)
        if idx_ref is None:
            idx_ref = m.index
        for c in m.columns:
            col_counts[c] = col_counts.get(c, 0) + 1
        del m
    common_cols = sorted([c for c, k in col_counts.items() if k >= len(all_names)])
    log(f"共同列数: {len(common_cols)}")

    sample_idx = idx_ref[-LAST_N_DAYS:]
    # 先抽样列（用第一遍读时的覆盖率），再第二次读只留需要的行列
    rng = np.random.default_rng(42)
    if len(common_cols) > N_SYMBOL_SAMPLE:
        sel = rng.choice(len(common_cols), size=N_SYMBOL_SAMPLE, replace=False)
        sel = np.sort(sel)
        sel_cols = [common_cols[i] for i in sel]
    else:
        sel_cols = common_cols
    log(f"列抽样: {len(sel_cols)} 只")

    X = np.empty((len(all_names), LAST_N_DAYS * len(sel_cols)), dtype=np.float64)
    for i, p in enumerate(sorted(FV_DIR.glob("*.parquet"))):
        m = pd.read_parquet(p)
        sub = m.reindex(index=sample_idx, columns=sel_cols)
        X[i] = sub.values.astype(np.float64).reshape(-1)
        del m, sub
    log(f"展平矩阵 X: {X.shape}, NaN占比 {np.isnan(X).mean():.4f}")
    return all_names, X, sample_idx, sel_cols


def spearman_corr_matrix(X, names):
    """秩变换 → 行标准化 → 一次 matmul 得 Pearson(=Spearman) 相关矩阵。"""
    from scipy.stats import rankdata

    n, d = X.shape
    log(f"秩变换中 ({n}×{d}) ...")
    R = rankdata(X, axis=1, nan_policy="omit")
    R = np.where(np.isnan(R), 0.0, R)
    # 行中心化（NaN→0 后均值即非NaN秩均值近似，可接受）
    R = R - R.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(R, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Rn = R / norms
    C = Rn @ Rn.T
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    log(f"相关矩阵 {C.shape} 完成, max|offdiag|={np.abs(C - np.eye(len(names))).max():.4f}")
    return C


def build_graph(C, names, threshold):
    """|corr|>=threshold 的边 → SparseCorrelationGraph。"""
    n = len(names)
    absC = np.abs(C)
    iu = np.triu_indices(n, k=1)
    mask = absC[iu] >= threshold
    pairs = list(zip(iu[0][mask], iu[1][mask]))
    edges = [
        CorrelationEdge(factor_a=names[i], factor_b=names[j], correlation=float(C[i, j]))
        for i, j in pairs
    ]
    log(f"边数(|corr|>={threshold}): {len(edges)} / {len(pairs)}对")
    g = SparseCorrelationGraph(edges=edges, nodes=list(names))
    log(f"图: {g.node_count} nodes, {g.edge_count} edges, density={g.density():.4f}")
    return g


def main():
    t0 = time.time()
    meta = json.loads(META.read_text())
    log(f"meta 条数: {len(meta)}")

    names, X, sample_idx, sel_cols = load_sample_matrix()

    # meta 里可能没有的因子 → 用 0 填充（gate 会 FAIL）
    def get_meta(name, key):
        rec = meta.get(name) or {}
        v = rec.get(key)
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    C = spearman_corr_matrix(X, names)
    np.save("/tmp/stage3_corr.npy", C)
    with open("/tmp/stage3_names.json", "w") as f:
        json.dump(names, f)
    log(f"相关性缓存已写 /tmp/stage3_corr.npy, 耗时{time.time()-t0:.0f}s")

    g = build_graph(C, names, CORR_THRESHOLD)
    cc = ConnectedComponents(graph=g, execution_mode=ExecutionMode.RESEARCH)
    res = cc.find_components()
    assignments = res.assignments
    log(f"聚类数: {res.cluster_sizes and len(res.cluster_sizes)}")

    # 组装 cluster -> members
    members = {}
    for f, cid in assignments.items():
        members.setdefault(cid, []).append(f)
    # 簇编号按大小降序重排 cluster_0 = 最大簇
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
        # 代表 = 簇内 best_rankic_ir 最高（缺失则 best_mean_rankic，再缺失取首成员）
        def ir_of(x):
            v = get_meta(x, "best_rankic_ir")
            if v is None:
                v = get_meta(x, "best_mean_rankic")
            return v if v is not None else -1e18

        best = max(mem, key=ir_of)
        representatives[cid] = {
            "factor": best,
            "best_rankic_ir": get_meta(best, "best_rankic_ir"),
            "best_mean_rankic": get_meta(best, "best_mean_rankic"),
            "cluster_size": len(mem),
        }

    # ---- 质量门槛（ThresholdGate，硬口径 2026-08-28）----
    ir_gate = ThresholdGate(
        gate_name="rankic_ir_min",
        threshold=IR_GATE,
        higher_is_better=True,
        gate_version="2026-08-28",
    )
    rankic_gate = ThresholdGate(
        gate_name="best_mean_rankic_min",
        threshold=RANKIC_GATE,
        higher_is_better=True,
        gate_version="2026-08-28",
    )

    quality_gate = {}
    n_pass = 0
    # meta 里有但 parquet 缺失的因子（61 个）：不进相关图，但同样做门槛标注
    meta_only = sorted([k for k in meta if k not in set(names)])
    log(f"meta 有但 parquet 缺失（不进图，仅标注）: {len(meta_only)}")
    all_names_for_gate = list(names) + meta_only
    for name in all_names_for_gate:
        ir = get_meta(name, "best_rankic_ir")
        rc = get_meta(name, "best_mean_rankic")
        if ir is None or rc is None:
            quality_gate[name] = {
                "best_mean_rankic": rc,
                "best_rankic_ir": ir,
                "gate_ir": "SKIP",
                "gate_rankic": "SKIP",
                "quality_gate": "fail",
                "has_matrix": name in set(names),
                "reason": "meta 无有效指标（error=empty 或字段缺失）",
            }
            continue
        in_graph = name in set(names)
        e_ir = ir_gate.evaluate(name, evidence_id="optimized_meta.json",
                                metric_name="best_rankic_ir", metric_value=ir)
        e_rc = rankic_gate.evaluate(name, evidence_id="optimized_meta.json",
                                    metric_name="best_mean_rankic", metric_value=rc)
        passed = (e_ir.result == GateResult.PASS) and (e_rc.result == GateResult.PASS)
        if passed:
            n_pass += 1
        quality_gate[name] = {
            "best_mean_rankic": rc,
            "best_rankic_ir": ir,
            "gate_ir": e_ir.result.value,
            "gate_rankic": e_rc.result.value,
            "quality_gate": "pass" if passed else "fail",
            "has_matrix": in_graph,
            "message": f"{e_ir.message}; {e_rc.message}",
        }

    sizes = sorted([len(v) for v in cluster_members.values()], reverse=True)
    n_total = len(all_names_for_gate)
    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_dir": str(FV_DIR),
        "meta_file": str(META),
        "n_factors_meta": len(meta),
        "n_factors_with_matrix": len(names),
        "n_factors_meta_only": len(meta_only),
        "corr_method": "spearman (rank+matmul), last 500 days x 800 sampled symbols",
        "corr_threshold": CORR_THRESHOLD,
        "clustering": "factor_assets.clustering.families.ConnectedComponents (RESEARCH mode)",
        "graph": {"nodes": g.node_count, "edges": g.edge_count,
                  "density": g.density(), "graph_identity": g.graph_identity},
        "num_clusters": len(cluster_members),
        "cluster_sizes_top20": sizes[:20],
        "cluster_members": cluster_members,
        "page_to_cluster": page_to_cluster,
        "representatives": representatives,
        "meta_only_factors": meta_only,
        "quality_gate_config": {
            "rankic_ir_min": IR_GATE,
            "best_mean_rankic_min": RANKIC_GATE,
            "basis_date": "2026-08-28",
            "engine": "factor_assets.selection.gates.ThresholdGate",
        },
        "quality_gate": quality_gate,
        "summary": {
            "n_factors_total": n_total,
            "n_factors_clustered": len(names),
            "n_factors_meta_only": len(meta_only),
            "n_clusters": len(cluster_members),
            "largest_cluster": sizes[0] if sizes else 0,
            "n_singletons": sum(1 for v in cluster_members.values() if len(v) == 1),
            "gate_pass": n_pass,
            "gate_fail": n_total - n_pass,
            "gate_pass_rate": round(n_pass / max(n_total, 1), 4),
        },
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    log(f"完成: {len(cluster_members)} 簇, 最大簇 {sizes[0] if sizes else 0}, "
        f"gate pass {n_pass}/{n_total}, 写 {OUT}, 总耗时 {time.time()-t0:.0f}s")
    log(f"簇大小 Top10: {sizes[:10]}")
    log(f"单例簇: {out['summary']['n_singletons']}")


if __name__ == "__main__":
    main()