#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段3：409 优化后因子冗余聚类 + 质量门槛标注（factor_assets 库）。

输入:
  weekly_backtest_output/optimized_factors/*.parquet  (409 个因子矩阵, date×symbol, 后复权口径)
  weekly_backtest_output/optimized_meta.json          (409 条有效, 61 条 error 跳过)

方法:
  1. 每因子采样 (日期步长3, 列800) → 展平 (rank 变换) → 逐块 Spearman 相关 → 409×409。
  2. factor_assets SparseCorrelationGraph (|corr| >= 0.85 建边) → ConnectedComponents 聚类。
  3. 质量门槛 (rankic_ir>0.5 且 best_mean_rankic>0.015 → pass) 用 ThresholdGate/CompositeGate 标注。
  4. 代表因子 = 簇内 rankic_ir 最高。

输出: weekly_backtest_output/factor_clusters.json + 统计报告 (stdout / progress log)。
不修改 optimized_factors/、optimized_meta.json。
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
from factor_assets.clustering.families import ConnectedComponents, LeidenClustering
from factor_assets.selection.gates import ThresholdGate, CompositeGate, MetricEvidence, MetricBinding, MetricDirection

FV_DIR = PROJECT / "weekly_backtest_output" / "optimized_factors"
META_P = PROJECT / "weekly_backtest_output" / "optimized_meta.json"
OUT = PROJECT / "weekly_backtest_output" / "factor_clusters.json"
LOG = Path("/tmp/agent_cluster2.log")

# 超参数
CORR_THRESHOLD = 0.85          # |corr| 建边阈值
DATE_STEP = 3                  # 日期步长(天)
COL_SAMPLE = 800               # 每因子列采样数
RANKIC_IR_THRESHOLD = 0.5      # 质量门槛: rankic_ir
MEAN_RANKIC_THRESHOLD = 0.015  # 质量门槛: best_mean_rankic
MIN_ROWS = 800                 # 每次外积相关最小有效行数

# 模块级公共日期 (load 阶段填充)
self_common_dates = None
# 日期窗口(天)内公共日期集合; 若某因子窗口覆盖该天数比例不足则跳过
MIN_COVER_FRAC = 0.25
MIN_DATE_COVER = 300


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_meta() -> dict:
    meta = json.loads(META_P.read_text())
    valid = {}
    errors = {}
    for k, v in meta.items():
        if "error" in v:
            errors[k] = v
        else:
            valid[k] = v
    return meta, valid, errors


def load_factor_names() -> list:
    names = sorted(os.path.basename(f)[: -len(".parquet")]
                   for f in glob.glob(str(FV_DIR / "*.parquet")))
    return names


def load_flat_matrices(names: list) -> dict:
    """加载全部因子矩阵, 对齐公共日期, 采样后返回 {page: ndarray(n_flat,)} + 元数据。"""
    global self_common_dates
    # 先读所有 index, 求公共日期 (使用 2019-01-02 起的稳定轴, 不要求全部重叠)
    # 方案: 用出现次数最多的起始日 2019-01-02 作为窗口起点, 2026-08-24 为终点
    all_idx = None
    for page in names:
        try:
            ix = pd.read_parquet(FV_DIR / f"{page}.parquet", columns=[]).index
        except Exception:
            continue
        all_idx = ix if all_idx is None else all_idx.union(ix)
    # 窗口起点: 多数因子 2019-01-02 开始
    window_start = pd.Timestamp("2019-01-02")
    window_end = all_idx.max() if all_idx is not None else pd.Timestamp("2026-08-24")
    window_dates = all_idx[(all_idx >= window_start) & (all_idx <= window_end)]
    self_common_dates = window_dates
    log(f"[load] 公共窗口 {len(self_common_dates)} 天 ({self_common_dates.min()} ~ {self_common_dates.max()})")

    mats = {}
    for i, page in enumerate(names):
        try:
            m = pd.read_parquet(FV_DIR / f"{page}.parquet")
        except Exception:
            log(f"[load] FAIL {page}: {sys.exc_info()[1]}")
            continue
        if m.shape[0] < 20 or m.shape[1] < 30:
            continue
        # 对齐到窗口日期 (只保留覆盖 >= MIN_DATE_COVER 天的因子, 保证相关样本量)
        if self_common_dates is not None:
            m = m.reindex(self_common_dates)
            n_cover = int(m.index.isin(m.index).sum())
        if m.shape[0] < MIN_DATE_COVER:
            log(f"[load] 跳过 (窗口下样本不足): {page} rows={m.shape[0]}")
            continue
        cols = m.columns.to_numpy()
        if len(cols) > COL_SAMPLE:
            rng = np.random.RandomState(20260828)
            cols = np.sort(rng.choice(cols, size=COL_SAMPLE, replace=False))
        sub = m[cols]
        flat = sub.to_numpy(dtype=np.float32, copy=False)
        mats[page] = {
            "data": flat,
            "date_stride": DATE_STEP,
            "sample_rows": sub.shape[0],
            "sample_cols": len(cols),
        }
        if (i + 1) % 100 == 0:
            log(f"[load] {i + 1}/{len(names)} 因子已加载")
    log(f"[load] 成功加载 {len(mats)}/{len(names)} 个因子矩阵")
    return mats


def compute_block_spearman(mats: dict, names: list, block: int = 45):
    """逐块外积计算 Spearman 相关矩阵 (带进度续跑, 部分结果落盘)。"""
    n = len(names)
    # 进度检查: 已有部分结果
    partial = PROJECT / "weekly_backtest_output" / "_stage3_corr_partial.npz"
    corr = np.full((n, n), np.nan)
    done = 0
    if partial.exists():
        try:
            old = np.load(partial)
            corr = old["corr"]
            done = int(np.sum(np.isfinite(corr)) // 2)
            log(f"[corr] 发现部分结果, 已算 {done} 对, 续跑")
        except Exception:
            log("[corr] 部分结果损坏, 从头计算")

    def rows_vec(page):
        s = mats[page]
        data = s["data"]
        rng = np.random.RandomState(20260828)
        nrows = data.shape[0]
        ridx = np.arange(0, nrows, s["date_stride"])
        if len(ridx) < 1000:
            ridx = np.arange(nrows)
        ridx = rng.choice(ridx, size=min(len(ridx), 6000), replace=False)
        ridx = np.sort(ridx)
        return data[ridx]

    # 预取所有向量
    t0 = time.time()
    vecs = {i: rows_vec(names[i]) for i in range(n)}
    total_pairs = n * (n - 1) // 2
    done_pairs = 0
    for i in range(0, n, block):
        for j in range(i, n, block):
            ia, ib = i, min(i + block, n)
            ja, jb = j, min(j + block, n)
            for a in range(ia, ib):
                va = vecs[a]
                for b in range(max(j, a + 1), jb):
                    if np.isfinite(corr[a, b]):
                        continue
                    vb = vecs[b]
                    mask = np.isfinite(va) & np.isfinite(vb)
                    if mask.sum() < MIN_ROWS:
                        continue
                    # Spearman = Pearson on ranks; 必须按 (a,b) 对的联合掩码分别 rank
                    va_rank = rankdata(va[mask])
                    vb_rank = rankdata(vb[mask])
                    c = np.corrcoef(va_rank, vb_rank)[0, 1]
                    if np.isfinite(c):
                        corr[a, b] = c
                        corr[b, a] = c
                    done_pairs += 1
            if (i * n + j) % (n * 3) == 0 or (j - i == block):
                pass
        if (i // block + 1) % 3 == 0:
            np.savez_compressed(partial, corr=corr)
            log(f"[corr] 进度 {i + block}/{n} 行块完成, 存盘续跑点")
            # 打印进度
            done_pairs = int(np.sum(np.isfinite(corr)) // 2)
            log(f"[corr] 已算 {done_pairs}/{total_pairs} 对 ({done_pairs / total_pairs * 100:.1f}%), 耗时 {time.time() - t0:.0f}s")

    # 最终落盘
    np.savez_compressed(partial, corr=corr)
    log(f"[corr] 相关矩阵完成, 有效对 {int(np.sum(np.isfinite(corr)) // 2)}/{total_pairs}, 耗时 {time.time() - t0:.0f}s")
    return corr


def build_graph_and_cluster(corr: np.ndarray, names: list) -> dict:
    n = len(names)
    idx = np.arange(n)
    valid_pairs = np.argwhere(np.abs(corr) >= CORR_THRESHOLD)
    valid_pairs = valid_pairs[valid_pairs[:, 0] < valid_pairs[:, 1]]
    log(f"[graph] |corr|>=0.85 边数: {len(valid_pairs)}")
    edges = []
    for a, b in valid_pairs:
        edges.append(CorrelationEdge(names[a], names[b], float(corr[a, b])))
    graph = SparseCorrelationGraph(edges, nodes=set(names))
    log(f"[graph] 节点 {graph.node_count}, 边 {graph.edge_count}, 密度 {graph.density():.4f}")

    cc = ConnectedComponents(graph, execution_mode="RESEARCH")
    result = cc.find_components()
    log(f"[cluster] ConnectedComponents 聚类数: {result.num_clusters}")
    return result.assignments, result.cluster_sizes


def leiden_cluster(corr: np.ndarray, names: list) -> dict:
    """Leiden 参考聚类 (作为补充信息记录, 不作为主输出)。"""
    n = len(names)
    idx = np.arange(n)
    valid_pairs = np.argwhere(np.abs(corr) >= 0.50)
    valid_pairs = valid_pairs[valid_pairs[:, 0] < valid_pairs[:, 1]]
    edges = []
    for a, b in valid_pairs:
        edges.append(CorrelationEdge(names[a], names[b], float(corr[a, b])))
    graph = SparseCorrelationGraph(edges, nodes=set(names))
    try:
        lc = LeidenClustering(graph, resolution=1.0, seed=42)
        result = lc.cluster()
        log(f"[cluster] Leiden 参考聚类数: {result.num_clusters}")
        return result.assignments
    except Exception as e:
        log(f"[cluster] Leiden 失败: {e}")
        return {}


def build_quality_gate():
    """用 factor_assets selection/gates 构造质量门槛。"""
    gate_ir = ThresholdGate("rankic_ir_gate", RANKIC_IR_THRESHOLD, higher_is_better=True, gate_version="1.0")
    gate_ic = ThresholdGate("mean_rankic_gate", MEAN_RANKIC_THRESHOLD, higher_is_better=True, gate_version="1.0")
    composite = CompositeGate(
        "quality_gate",
        [gate_ir, gate_ic],
        require_all=True,
        gate_version="1.0",
        metric_bindings={
            "rankic_ir_gate": MetricBinding("rankic_ir", "ratio", MetricDirection.HIGHER_IS_BETTER),
            "mean_rankic_gate": MetricBinding("mean_rankic", "ratio", MetricDirection.HIGHER_IS_BETTER),
        },
    )
    return composite


def evaluate_gate(composite, page: str, meta: dict):
    """评估单个因子的质量门槛, 返回 dict 标注。"""
    ir = meta.get("best_rankic_ir", 0.0) or 0.0
    ic = meta.get("best_mean_rankic", 0.0) or 0.0
    composite_eval, subs = composite.evaluate_all(
        page,
        evidence_id=f"optimized_meta::{page}",
        metrics={
            "rankic_ir": MetricEvidence(ir, "ratio", MetricDirection.HIGHER_IS_BETTER),
            "mean_rankic": MetricEvidence(ic, "ratio", MetricDirection.HIGHER_IS_BETTER),
        },
    )
    passed = composite_eval.result.value == "PASS"
    # 同时用独立��算确认 (Task 口径: rankic_ir>0.5 且 best_mean_rankic>0.015)
    confirm = (ir > RANKIC_IR_THRESHOLD) and (ic > MEAN_RANKIC_THRESHOLD)
    if passed != confirm:
        log(f"[gate] 不一致 {page}: composite={composite_eval.result.value} confirm={confirm} ir={ir:.4f} ic={ic:.4f}")
        passed = confirm
    return {
        "quality_gate": "pass" if passed else "fail",
        "rankic_ir": ir,
        "mean_rankic": ic,
    }


def main():
    log("=== 阶段3 冗余聚类 + 质量门槛标注 (重跑) ===")
    meta, valid, errors = load_meta()
    names = load_factor_names()
    log(f"[meta] 总 {len(meta)}, error {len(errors)}, 有效 {len(valid)}")
    log(f"[files] 优化后因子 parquet: {len(names)}")

    # 只用有效因子 (meta 中无 error 且 parquet 存在)
    valid_names = [nm for nm in names if nm in valid]
    log(f"[run] 参与聚类的因子: {len(valid_names)}")
    error_skipped = sorted(set(names) - set(valid_names))
    if error_skipped:
        log(f"[run] 跳过 error 因子: {len(error_skipped)} {error_skipped[:10]}...")

    mats = load_flat_matrices(valid_names)
    active = [nm for nm in valid_names if nm in mats]
    log(f"[run] 实际进入相关计算的因子: {len(active)}")

    # 相关矩阵 (块状外积)
    corr = compute_block_spearman(mats, active, block=45)

    # 聚类
    assignments, sizes = build_graph_and_cluster(corr, active)
    # 补充: Leiden 参考
    leiden_assign = leiden_cluster(corr, active)

    # 组装 cluster_members / page_to_cluster
    cluster_members = {}
    for fid, cid in assignments.items():
        cluster_members.setdefault(f"cluster_{cid}", []).append(fid)
    for members in cluster_members.values():
        members.sort()
    page_to_cluster = {fid: f"cluster_{cid}" for fid, cid in assignments.items()}

    # 代表因子: 簇内 rankic_ir 最高
    representatives = {}
    for cid, members in cluster_members.items():
        best = None
        best_ir = -1e9
        for m in members:
            ir = valid.get(m, {}).get("best_rankic_ir", 0.0) or 0.0
            if ir > best_ir:
                best_ir = ir
                best = m
        representatives[cid] = best

    # 质量门槛
    composite = build_quality_gate()
    quality_gate = {}
    for page in active:
        quality_gate[page] = evaluate_gate(composite, page, valid.get(page, {}))
    # 也把 error 因子标注上 (fail, 未参与聚类), 保持与旧格式一致
    for page in error_skipped:
        quality_gate[page] = {
            "quality_gate": "fail",
            "rankic_ir": 0.0,
            "mean_rankic": 0.0,
        }

    # 输出
    out = {
        "corr_threshold": CORR_THRESHOLD,
        "num_clusters": len(cluster_members),
        "page_to_cluster": page_to_cluster,
        "cluster_members": cluster_members,
        "representatives": representatives,
        "quality_gate": quality_gate,
        "leiden_reference_assignments": {fid: f"leiden_{cid}" for fid, cid in leiden_assign.items()} if leiden_assign else {},
        "_meta": {
            "input": "weekly_backtest_output/optimized_factors/*.parquet + optimized_meta.json",
            "corr_method": "Spearman (采样: 日期步长3, 列800, 展平)",
            "n_factors": len(active),
            "n_error_skipped": len(error_skipped),
            "gate_criteria": f"rankic_ir>{RANKIC_IR_THRESHOLD} 且 best_mean_rankic>{MEAN_RANKIC_THRESHOLD}",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    log(f"[done] 写 {OUT}")

    # ===== 报告 =====
    sizes_sorted = sorted([len(v) for v in cluster_members.values()], reverse=True)
    log("===== 聚类统计 =====")
    log(f"簇数量: {len(cluster_members)}")
    log(f"最大簇: {sizes_sorted[0] if sizes_sorted else 0} 个因子 (簇大小Top10: {sizes_sorted[:10]})")
    n_cluster1 = sum(1 for v in cluster_members.values() if len(v) == 1)
    log(f"单例簇: {n_cluster1}")

    # 代表因子 Top20 (按代表因子的 rankic_ir 排序)
    rep_rows = []
    for cid, rep in representatives.items():
        if rep is None:
            continue
        ir = valid.get(rep, {}).get("best_rankic_ir", 0.0) or 0.0
        ic = valid.get(rep, {}).get("best_mean_rankic", 0.0) or 0.0
        size = len(cluster_members.get(cid, []))
        rep_rows.append((cid, rep, ir, ic, size))
    rep_rows.sort(key=lambda x: -x[2])
    log("===== 代表因子 Top20 (按簇代表 rankic_ir 降序) =====")
    for cid, rep, ir, ic, size in rep_rows[:20]:
        log(f"  {cid:10s} {rep:55s} ir={ir:.4f} mean_rankic={ic:.4f} 簇大小={size}")

    # gate 通过率
    n_pass = sum(1 for v in quality_gate.values() if v.get("quality_gate") == "pass")
    n_total_gate = len(quality_gate)
    log("===== 质量门槛 =====")
    log(f"gate 通过: {n_pass}/{n_total_gate} ({n_pass / n_total_gate * 100:.1f}%)")
    # 通过名单 Top 由 IR 排序
    passed_list = sorted(
        [(p, v.get("rankic_ir", 0), v.get("mean_rankic", 0)) for p, v in quality_gate.items() if v.get("quality_gate") == "pass"],
        key=lambda x: -x[1],
    )
    log(f"通过门槛的因子 ({len(passed_list)} 个, 按 IR 降序):")
    for p, ir, ic in passed_list[:40]:
        log(f"  {p:55s} ir={ir:.4f} mean_rankic={ic:.4f}")


if __name__ == "__main__":
    main()
