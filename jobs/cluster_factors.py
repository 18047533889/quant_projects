#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段3：456 因子冗余聚类 + 质量门槛标注（factor_assets）。

1. 用 456 因子值算相关性矩阵 → 相关性图 → 聚类（ConnectedComponents / Modularity）。
2. 每个因子标注所属"因子族"（cluster_id）+ 代表因子。
3. 用 selection/gates 的 ThresholdGate 按 RankIC/IR/Sharpe 阈值标注"通过/未通过质量门槛"。
4. 不删除任何因子，仅标注分组 + 门槛状态。

输出：weekly_backtest_output/factor_clusters.json
"""
import sys, os, json, glob, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

PROJECT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "factor_assets"))

FV_DIR = PROJECT / "weekly_backtest_output" / "factor_matrices_all"
OUT = PROJECT / "weekly_backtest_output" / "factor_clusters.json"
SUMMARY = PROJECT / "factor_engine" / "docs" / "reports" / "2026-08-23" / "summary_stats.json"

# 相关性阈值（|corr| > 阈值视为同族）
CORR_THRESHOLD = 0.7


def load_all_matrices(names):
    """加载全部因子矩阵（抽样日期控制内存）。返回 {page: DataFrame}。"""
    mats = {}
    for page in names:
        p = FV_DIR / f"{page}.parquet"
        if not p.exists():
            continue
        try:
            m = pd.read_parquet(p)
            if m.shape[1] == 0:
                continue
            mats[page] = m
        except Exception:
            continue
    return mats


def compute_corr_matrix(mats, common_idx):
    """逐日横截面相关性 → 平均相关性矩阵。"""
    pages = list(mats.keys())
    n = len(pages)
    corr_sum = np.zeros((n, n))
    cnt = np.zeros((n, n))
    # 公共 symbol 列（所有因子都有的）
    common_cols = None
    for m in mats.values():
        if common_cols is None:
            common_cols = set(m.columns)
        else:
            common_cols = common_cols.intersection(m.columns)
    common_cols = sorted(common_cols)
    print(f"[cluster] 公共 symbol 列: {len(common_cols)}", flush=True)
    # 抽样日期（每 5 天取 1 天，控制计算量）
    idx = common_idx[::5]
    for t in idx:
        vecs = {}
        for i, page in enumerate(pages):
            m = mats[page]
            if t in m.index:
                row = m.loc[t].reindex(common_cols).astype(float)
                vecs[i] = row
        # 两两相关性
        keys = list(vecs.keys())
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                i, j = keys[a], keys[b]
                va = vecs[i].values; vb = vecs[j].values
                mask = np.isfinite(va) & np.isfinite(vb)
                if mask.sum() < 30:
                    continue
                c = np.corrcoef(va[mask], vb[mask])[0, 1]
                if np.isfinite(c):
                    corr_sum[i, j] += c
                    corr_sum[j, i] += c
                    cnt[i, j] += 1
                    cnt[j, i] += 1
    corr = np.where(cnt > 0, corr_sum / np.maximum(cnt, 1), 0.0)
    return corr, pages


def connected_components(corr, pages, threshold):
    """基于相关性图做连通分量聚类。"""
    n = len(pages)
    adj = np.abs(corr) > threshold
    np.fill_diagonal(adj, False)
    visited = [False] * n
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        # BFS
        comp = []
        stack = [i]
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in range(n):
                if adj[u, v] and not visited[v]:
                    visited[v] = True
                    stack.append(v)
        clusters.append(comp)
    return clusters


def main():
    # 1. 因子列表
    names = sorted([f.stem for f in FV_DIR.glob("*.parquet")]) if FV_DIR.exists() else []
    print(f"[cluster] 因子数: {len(names)}", flush=True)
    if not names:
        return

    # 2. 加载矩阵（抽样日期）
    mats = load_all_matrices(names)
    print(f"[cluster] 加载 {len(mats)} 个因子矩阵", flush=True)

    # 公共日期轴
    all_idx = None
    for m in mats.values():
        if all_idx is None:
            all_idx = m.index
        else:
            all_idx = all_idx.intersection(m.index)
    common_idx = all_idx
    print(f"[cluster] 公共日期 {len(common_idx)} 天", flush=True)

    # 3. 相关性矩阵
    t0 = time.time()
    corr, pages = compute_corr_matrix(mats, common_idx)
    print(f"[cluster] 相关性矩阵 {corr.shape} 耗时{time.time()-t0:.0f}s", flush=True)

    # 4. 聚类
    clusters = connected_components(corr, pages, CORR_THRESHOLD)
    print(f"[cluster] 聚类数: {len(clusters)}", flush=True)

    # 5. 质量门槛（从 summary_stats 读指标）
    stats = {}
    if SUMMARY.exists():
        try:
            for s in json.loads(SUMMARY.read_text()):
                stats[s.get("name")] = s
        except Exception:
            pass

    # 6. 组装输出
    page_to_cluster = {}
    cluster_members = {}
    for ci, comp in enumerate(clusters):
        members = [pages[i] for i in comp]
        cluster_members[f"cluster_{ci}"] = members
        for m in members:
            page_to_cluster[m] = f"cluster_{ci}"

    # 代表因子：族内 RankIC 最高
    representatives = {}
    for cid, members in cluster_members.items():
        best = None; best_ir = -1e9
        for m in members:
            s = stats.get(m, {})
            ir = s.get("ic_ir", 0)
            if ir > best_ir:
                best_ir = ir; best = m
        representatives[cid] = best

    # 质量门槛标注
    gate_meta = {}
    for page in names:
        s = stats.get(page, {})
        mean_ic = s.get("mean_ic", 0)
        ir = s.get("ic_ir", 0)
        sharpe = s.get("ls_sharpe", 0)
        passed = (abs(mean_ic) >= 0.02) and (abs(ir) >= 0.2) and (sharpe > 0)
        gate_meta[page] = {
            "mean_rankic": mean_ic,
            "rankic_ir": ir,
            "ls_sharpe": sharpe,
            "quality_gate": "通过" if passed else "未通过",
        }

    out = {
        "corr_threshold": CORR_THRESHOLD,
        "num_clusters": len(clusters),
        "page_to_cluster": page_to_cluster,
        "cluster_members": cluster_members,
        "representatives": representatives,
        "quality_gate": gate_meta,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    print(f"[cluster] 完成: {len(clusters)} 族, 存 {OUT}", flush=True)
    # 打印族大小分布
    sizes = sorted([len(v) for v in cluster_members.values()], reverse=True)
    print(f"[cluster] 族大小分布(前10): {sizes[:10]}", flush=True)


if __name__ == "__main__":
    main()
