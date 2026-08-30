#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段4：报告集成 — 详情页新增「🧬 优化因子」区块 + 首页新增「优化因子汇总」。

依赖阶段1(optimized_factors/ + optimized_meta.json)、阶段2(optimized_top/ + optimized_top_meta.json)、
阶段3(factor_clusters.json)。

每个因子详情页新增：
  - 预处理步骤公式化表述（阶段1 meta）
  - 原始 vs 优化后 RankIC/IR 柱状对比图
  - 原始 vs 优化后十分层净值并排图
  - 原始 vs 优化后多空 NAV 叠加图
  - 因子族标注 + 质量门槛状态（阶段3）
  - Top 因子额外显示参数寻优记录（阶段2）

首页新增「优化因子汇总」区块：优化后 RankIC/IR 排序表 + 优化前后对比图 + 因子族分布。
全部中文图表（Noto CJK）。
"""
import sys, os, json, glob, time, warnings, io, base64
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

PROJECT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "vectorbt_qs"))
sys.path.insert(0, str(PROJECT / "scripts" / "archive" / "jobs"))
sys.path.insert(0, str(PROJECT / "jobs"))

from quant_evaluator.metrics.ic import _spearman_rank_correlation

# 中文字体
_CN_FONT = "/home/sunhaiwei/.fonts/NotoSansSC-Regular.otf"
if os.path.exists(_CN_FONT):
    try:
        import matplotlib.font_manager as _fm
        _fm.fontManager.addfont(_CN_FONT)
        _CN_NAME = _fm.FontProperties(fname=_CN_FONT).get_name()
        plt.rcParams["font.sans-serif"] = [_CN_NAME, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

REPORT_DIR = PROJECT / "factor_engine" / "docs" / "reports" / "2026-08-23"
FACTORS_DIR = REPORT_DIR / "factors"
RAW_DIR = PROJECT / "weekly_backtest_output" / "factor_matrices_all"
OPT_DIR = PROJECT / "weekly_backtest_output" / "optimized_factors"
OPT_TOP_DIR = PROJECT / "weekly_backtest_output" / "optimized_top"
OPT1_META = PROJECT / "weekly_backtest_output" / "optimized_meta.json"
OPT2_META = PROJECT / "weekly_backtest_output" / "optimized_top_meta.json"
CLUSTER = PROJECT / "weekly_backtest_output" / "factor_clusters.json"
DAILY_ADJ = Path.home() / "cos_data" / "StockDailyBarAdj"
START, END = "2019-01-02", "2026-08-24"

_HAS_VWAP = None


def load_vwap():
    """加载 后复权 AdjVwap 矩阵（全局收益口径：vwap-to-vwap 后复权硬性）。
    数据源：StockDailyBarAdj.AdjVwap，禁止未复权 StockDailyBar.Vwap。"""
    global _HAS_VWAP
    if _HAS_VWAP is not None:
        return _HAS_VWAP
    import duckdb
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


def daily_rankic_series(factor_mat, vwap):
    """逐日 spearman rankic，收益 = vwap-to-vwap。"""
    common = factor_mat.index.intersection(vwap.index)
    fv = factor_mat.reindex(index=common)
    vv = vwap.reindex(index=common)
    cols = vv.columns.intersection(fv.columns)
    fv = fv[cols]; vv = vv[cols]
    fwd = vv.pct_change().shift(-2)  # vwap-to-vwap（t+1成交→t+2卖出，企业级）
    T = fv.shape[0]
    ic = np.full(T, np.nan)
    fv_a = fv.values; fwd_a = fwd.values
    for t in range(T):
        m = fv_a[t]; r = fwd_a[t]
        mask = np.isfinite(m) & np.isfinite(r)
        if mask.sum() < 20:
            continue
        ic[t] = _spearman_rank_correlation(m[mask], r[mask])
    return pd.Series(ic, index=common)


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#fff")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def plot_ic_compare(raw_ic, opt_ic, name):
    """原始 vs 优化后 RankIC 时序 + IR 柱状对比。"""
    common = raw_ic.index.intersection(opt_ic.index)
    raw = raw_ic.reindex(common).rolling(20).mean()
    opt = opt_ic.reindex(common).rolling(20).mean()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [3, 1]})
    ax1.plot(common, raw, color="#dc2626", lw=1.0, label="原始 (MA20)")
    ax1.plot(common, opt, color="#0d9488", lw=1.2, label="优化后 (MA20)")
    ax1.axhline(0, color="#94a3b8", lw=0.6)
    ax1.set_title(f"{name} — RankIC 时序: 原始 vs 优化后", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(ls="--", alpha=0.3)
    raw_ir = raw_ic.dropna().mean() / raw_ic.dropna().std() if raw_ic.dropna().std() > 1e-9 else 0
    opt_ir = opt_ic.dropna().mean() / opt_ic.dropna().std() if opt_ic.dropna().std() > 1e-9 else 0
    ax2.bar(["原始", "优化后"], [raw_ir, opt_ir], color=["#dc2626", "#0d9488"])
    ax2.set_title("RankIC IR", fontsize=9)
    ax2.grid(axis="y", ls="--", alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_decile_compare(raw_mat, opt_mat, vwap, name):
    """原始 vs 优化后十分层净值并排。收益口径 vwap-to-vwap（shift(-2)）。
    2026-08-29 企业级修复：扣双边成本 0.1%（按换手计）+ 停牌股不参与（fwd NaN 剔除）。"""
    COST = 0.001  # 双边 0.1%
    def decile_nav(mat):
        common = mat.index.intersection(vwap.index)
        fv = mat.reindex(index=common)
        vv = vwap.reindex(index=common)
        cols = vv.columns.intersection(fv.columns)
        fv = fv[cols]; vv = vv[cols]
        fwd = vv.pct_change().shift(-2)  # vwap-to-vwap（t+1成交→t+2卖出，企业级）
        ranks = fv.rank(axis=1, method='first', pct=True).values
        valid = np.isfinite(fv.values) & np.isfinite(fwd.values)
        gids = np.floor(ranks * 10).clip(0, 9).astype(int)
        gids[~valid] = -1
        T = fv.shape[0]
        gr = np.zeros((T, 10))
        prev_gids = np.full(fv.shape[1], -1)
        for t in range(T):
            # 换手成本：每组内与昨日组员不同的比例 × 双边成本
            turnover = np.zeros(10)
            for k in range(10):
                mk = (gids[t] == k)
                if mk.any():
                    # 组内成员相对昨日的变化比例（近似换手）
                    if t > 0:
                        same = prev_gids[mk] == k
                        turnover[k] = 1.0 - same.mean()
                    gr[t, k] = np.nanmean(fwd.values[t, mk]) - turnover[k] * COST
            prev_gids = gids[t].copy()
        return common, np.cumprod(1 + gr, axis=0)
    raw_common, raw_nav = decile_nav(raw_mat)
    opt_common, opt_nav = decile_nav(opt_mat)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.4))
    for k in range(10):
        ax1.plot(raw_common, raw_nav[:, k], lw=0.8, alpha=0.7)
    # 2026 分段标注（decay 因子提示）
    y26 = pd.Timestamp("2026-01-01")
    for ax, common in ((ax1, raw_common), (ax2, opt_common)):
        if common.min() < pd.Timestamp("2026-01-01") <= common.max():
            ax.axvline(pd.Timestamp("2026-01-01"), color="#f59e0b", lw=1.0, ls="--", alpha=0.7)
            ax.text(pd.Timestamp("2026-01-01"), ax.get_ylim()[1]*0.95, " 2026→", fontsize=7, color="#b45309")
    ax1.set_title(f"{name} — 原始十分层净值(扣费0.1%)", fontsize=9)
    ax1.grid(ls="--", alpha=0.3)
    for k in range(10):
        ax2.plot(opt_common, opt_nav[:, k], lw=0.8, alpha=0.7)
    ax2.set_title(f"{name} — 优化后十分层净值(扣费0.1%)", fontsize=9)
    ax2.grid(ls="--", alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_ls_compare(raw_mat, opt_mat, vwap, name):
    """原始 vs 优化后多空 NAV 叠加。收益口径 vwap-to-vwap（shift(-2)）。
    2026-08-29 企业级修复：扣双边成本 0.1%（多空双边）+ 2026 分段线。"""
    COST = 0.001
    def ls_nav(mat):
        common = mat.index.intersection(vwap.index)
        fv = mat.reindex(index=common)
        vv = vwap.reindex(index=common)
        cols = vv.columns.intersection(fv.columns)
        fv = fv[cols]; vv = vv[cols]
        fwd = vv.pct_change().shift(-2)  # vwap-to-vwap（t+1成交→t+2卖出，企业级）
        ranks = fv.rank(axis=1, method='first', pct=True).values
        valid = np.isfinite(fv.values) & np.isfinite(fwd.values)
        gids = np.floor(ranks * 10).clip(0, 9).astype(int)
        gids[~valid] = -1
        T = fv.shape[0]
        gr = np.zeros((T, 10))
        prev_gids = np.full(fv.shape[1], -1)
        for t in range(T):
            for k in range(10):
                mk = (gids[t] == k)
                if mk.any():
                    if t > 0:
                        same = prev_gids[mk] == k
                        to_cost = (1.0 - same.mean()) * COST  # 单腿换手成本
                    else:
                        to_cost = 0.0
                    gr[t, k] = np.nanmean(fwd.values[t, mk]) - to_cost
            prev_gids = gids[t].copy()
        ls = gr[:, 9] - gr[:, 0]
        return common, np.cumprod(1 + ls, axis=0)
    raw_common, raw = ls_nav(raw_mat)
    opt_common, opt = ls_nav(opt_mat)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(raw_common, raw, color="#dc2626", lw=1.2, label="原始多空(扣费)")
    ax.plot(opt_common, opt, color="#0d9488", lw=1.4, label="优化后多空(扣费)")
    ax.axhline(1, color="#94a3b8", lw=0.6)
    # 2026 分段线
    for common in (raw_common, opt_common):
        if common.min() < pd.Timestamp("2026-01-01") <= common.max():
            ax.axvline(pd.Timestamp("2026-01-01"), color="#f59e0b", lw=1.0, ls="--", alpha=0.7)
    ax.set_title(f"{name} — 多空净值: 原始 vs 优化后（扣双边0.1%）", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(ls="--", alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


def build_opt_block(page, opt1_meta, opt2_meta, cluster_meta):
    """构建优化因子区块 HTML。"""
    m1 = opt1_meta.get(page, {})
    steps = m1.get("steps", [])
    best_ir = m1.get("best_rankic_ir", 0)
    best_mean = m1.get("best_mean_rankic", 0)
    dsl_ops = m1.get("dsl_preproc_ops", [])

    # 因子族 + 质量门槛
    cluster_id = (cluster_meta.get("page_to_cluster") or {}).get(page, "—")
    gate = (cluster_meta.get("quality_gate") or {}).get(page, {})
    gate_status = gate.get("quality_gate", "—")
    rep = (cluster_meta.get("representatives") or {}).get(cluster_id, "")
    is_rep = "是" if rep == page else "否"

    # 参数寻优记录（Top 因子）
    opt2 = opt2_meta.get(page, {})
    opt2_html = ""
    if opt2:
        opt2_html = f"""
        <tr><td>参数寻优</td><td>{opt2.get('formula', '—')}</td></tr>
        <tr><td>寻优后 RankIC IR</td><td>{opt2.get('best_rankic_ir', 0):.3f}</td></tr>
        """

    steps_html = " → ".join(steps) if steps else "（无预处理）"
    dsl_ops_html = ", ".join(dsl_ops) if dsl_ops else "（无）"

    return f"""
<div class="card">
  <h2>🧬 优化因子（预处理 + 择优）</h2>
  <p style="font-size:0.82rem;color:#64748b;margin:0 0 10px">
    对原始因子做按需预处理（检测 DSL 已含算子避免重复），多方案择优选 RankIC IR 最高者。
  </p>
  <table class="meta-table" style="margin-top:6px">
    <tbody>
      <tr><td>预处理步骤</td><td><code>{steps_html}</code></td></tr>
      <tr><td>DSL 已含预处理算子</td><td><code>{dsl_ops_html}</code></td></tr>
      <tr><td>最优变体</td><td>{m1.get('best', '—')}</td></tr>
      <tr><td>优化后 RankIC</td><td>{best_mean:.4f}</td></tr>
      <tr><td>优化后 RankIC IR</td><td>{best_ir:.3f}</td></tr>
      <tr><td>因子族</td><td>{cluster_id}（代表因子: {rep}，本因子是代表: {is_rep}）</td></tr>
      <tr><td>质量门槛</td><td>{gate_status}</td></tr>
      {opt2_html}
    </tbody>
  </table>
  <div id="opt-charts-{page}" style="margin-top:10px">
    <p style="color:#94a3b8;font-size:0.8rem">（优化对比图将在下方渲染）</p>
  </div>
</div>
"""


def inject_detail_page(page, opt1_meta, opt2_meta, cluster_meta, vwap):
    """向详情页注入优化因子区块 + 对比图（vwap-to-vwap 口径）。"""
    html_path = FACTORS_DIR / f"factor_{page}.html"
    if not html_path.exists():
        return False
    html = html_path.read_text(encoding="utf-8")
    if "🧬 优化因子" in html:
        return False  # 已注入

    # 加载原始 + 优化后矩阵
    raw_path = RAW_DIR / f"{page}.parquet"
    opt_path = OPT_DIR / f"{page}.parquet"
    if not raw_path.exists() or not opt_path.exists():
        return False
    raw_mat = pd.read_parquet(raw_path)
    opt_mat = pd.read_parquet(opt_path)
    if raw_mat.shape[1] == 0 or opt_mat.shape[1] == 0:
        return False

    # 生成对比图（单页任一图失败不拖垮全流程）
    ic_chart = decile_chart = ls_chart = None
    try:
        raw_ic = daily_rankic_series(raw_mat, vwap)
        opt_ic = daily_rankic_series(opt_mat, vwap)
        ic_chart = plot_ic_compare(raw_ic, opt_ic, page)
    except Exception as exc:
        print(f"  [ERR] {page} RankIC 对比图: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
    try:
        decile_chart = plot_decile_compare(raw_mat, opt_mat, vwap, page)
    except Exception as exc:
        print(f"  [ERR] {page} 十分层图: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
    try:
        ls_chart = plot_ls_compare(raw_mat, opt_mat, vwap, page)
    except Exception as exc:
        print(f"  [ERR] {page} 多空图: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
    if ic_chart is None and decile_chart is None and ls_chart is None:
        print(f"  [ERR] {page} 三张对比图全部失败，跳过注入", flush=True)
        return False

    block = build_opt_block(page, opt1_meta, opt2_meta, cluster_meta)
    # 替换占位 div 为真实图（失败的图留占位说明）
    charts_html = ""
    if ic_chart:
        charts_html += f'  <img src="data:image/png;base64,{ic_chart}" style="width:100%;border-radius:8px;margin-bottom:8px" alt="RankIC对比"/>\n'
    if decile_chart:
        charts_html += f'  <img src="data:image/png;base64,{decile_chart}" style="width:100%;border-radius:8px;margin-bottom:8px" alt="十分层对比"/>\n'
    if ls_chart:
        charts_html += f'  <img src="data:image/png;base64,{ls_chart}" style="width:100%;border-radius:8px" alt="多空对比"/>\n'
    if not charts_html:
        charts_html = '  <p style="color:#94a3b8;font-size:0.8rem">（对比图生成失败）</p>\n'
    block = block.replace(
        f'<div id="opt-charts-{page}" style="margin-top:10px">\n    <p style="color:#94a3b8;font-size:0.8rem">（优化对比图将在下方渲染）</p>\n  </div>',
        f'<div style="margin-top:10px">\n{charts_html}  </div>'
    )

    # 在 </main> 前插入
    if "</main>" in html:
        html = html.replace("</main>", block + "\n</main>", 1)
    html_path.write_text(html, encoding="utf-8")
    return True


def main():
    # 参数：--limit N（只注入前 N 页，冒烟用）、--pages p1,p2,...（指定页面）
    import argparse
    ap = argparse.ArgumentParser(description="阶段4：详情页注入优化因子区块 + 首页汇总")
    ap.add_argument("--limit", type=int, default=0, help="只注入前 N 页（0=全部）")
    ap.add_argument("--pages", type=str, default="", help="逗号分隔的指定页面（优先于 --limit）")
    args = ap.parse_args()

    # 加载 meta
    opt1_meta = json.loads(OPT1_META.read_text()) if OPT1_META.exists() else {}
    opt2_meta = json.loads(OPT2_META.read_text()) if OPT2_META.exists() else {}
    cluster_meta = json.loads(CLUSTER.read_text()) if CLUSTER.exists() else {}
    print(f"[opt4] 阶段1 meta: {len(opt1_meta)}, 阶段2 meta: {len(opt2_meta)}, 聚类: {len(cluster_meta.get('cluster_members', {}))}", flush=True)

    close = load_vwap()
    pages = sorted([f.stem for f in OPT_DIR.glob("*.parquet")]) if OPT_DIR.exists() else []
    if args.pages:
        want = [p.strip() for p in args.pages.split(",") if p.strip()]
        pages = [p for p in pages if p in want]
        print(f"[opt4] 指定页面: {len(pages)} 页", flush=True)
    elif args.limit and args.limit > 0:
        pages = pages[:args.limit]
        print(f"[opt4] 限制前 {len(pages)} 页", flush=True)
    print(f"[opt4] 优化因子数: {len(pages)} (vwap-to-vwap 口径)", flush=True)

    t0 = time.time()
    injected = 0
    for i, page in enumerate(pages):
        try:
            ok = inject_detail_page(page, opt1_meta, opt2_meta, cluster_meta, close)
            if ok:
                injected += 1
        except Exception as exc:
            print(f"  [ERR] {page}: {type(exc).__name__}: {str(exc)[:150]}", flush=True)
            continue
        if (i + 1) % 50 == 0 or (i + 1) == len(pages):
            print(f"  {i+1}/{len(pages)} 注入{injected} 耗时{time.time()-t0:.0f}s", flush=True)
    print(f"[opt4] 详情页注入完成: {injected} 页, 耗时{time.time()-t0:.0f}s", flush=True)

    # 首页优化汇总（追加到 index.html）
    index_path = REPORT_DIR / "index.html"
    if index_path.exists():
        try:
            html = index_path.read_text(encoding="utf-8")
            if "优化因子汇总" in html:
                print(f"[opt4] 首页已含优化因子汇总，跳过", flush=True)
            else:
                # 生成优化因子排序表
                rows = []
                for page in sorted(opt1_meta.items(), key=lambda kv: kv[1].get("best_rankic_ir", 0), reverse=True):
                    p, m = page
                    if m.get("best_rankic_ir", 0) <= 0:
                        continue
                    cid = (cluster_meta.get("page_to_cluster") or {}).get(p, "—")
                    gate = (cluster_meta.get("quality_gate") or {}).get(p, {}).get("quality_gate", "—")
                    rows.append(
                        f'<tr><td class="rank">{len(rows)+1}</td>'
                        f'<td><a href="factors/factor_{p}.html"><code>{p}</code></a></td>'
                        f'<td>{m.get("best_mean_rankic", 0):.4f}</td>'
                        f'<td>{m.get("best_rankic_ir", 0):.3f}</td>'
                        f'<td>{m.get("best", "—")}</td>'
                        f'<td>{cid}</td><td>{gate}</td></tr>'
                    )
                opt_table = f"""
<h2>🧬 优化因子汇总（预处理 + 择优）</h2>
<table>
  <thead><tr><th>#</th><th>因子</th><th>优化后 RankIC</th><th>优化后 IR</th><th>最优变体</th><th>因子族</th><th>质量门槛</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
"""
                if "</main>" in html:
                    html = html.replace("</main>", opt_table + "\n</main>", 1)
                    index_path.write_text(html, encoding="utf-8")
                    print(f"[opt4] 首页优化汇总已追加", flush=True)
                else:
                    # index.html 无 </main> 标记（旧格式），追加到 </body> 前
                    html = html.replace("</body>", opt_table + "\n</body>", 1)
                    index_path.write_text(html, encoding="utf-8")
                    print(f"[opt4] 首页优化汇总已追加（无 </main>，追加到 </body> 前）", flush=True)
        except Exception as exc:
            print(f"[opt4] 首页追加失败: {type(exc).__name__}: {str(exc)[:150]}", flush=True)

    print(f"[opt4] 全部完成", flush=True)


if __name__ == "__main__":
    main()
