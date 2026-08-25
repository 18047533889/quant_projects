#!/usr/bin/env python3
"""补齐61因子的"中性化前后 RankIC/RankICIR"评估，并注入详情页。

中性化口径：每日横截面，行业(sw_l1)哑变量 + log(流通市值) 对因子值做线性回归，
取残差作为中性化后的因子值，再对残差算每日 rankic，聚合 rankic_ir / 胜率 / 月度IC。
输出 raw vs neutralized 对比，写入 all_eval_neutral.json，并把对比图(base64 PNG)
注入每个详情页的 "中性化前后 RankIC/RankICIR" 区块。
"""
from __future__ import annotations
import json, io, base64, time, os, math
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PROJECT = Path("/home/sunhaiwei/quant_projects")
FV_PATH = PROJECT / "weekly_backtest_output" / "factor_values.parquet"
REPORT_DIR = PROJECT / "factor_engine" / "docs" / "reports" / "2026-08-25"
FACTORS_DIR = REPORT_DIR / "factors"
COS_DATA = Path.home() / "cos_data"
DAILY = COS_DATA / "StockDailyBar"
INDUSTRY = COS_DATA / "StockIndustry"
VALUATION = COS_DATA / "StockValuationDaily"

OUT_JSON = REPORT_DIR / "all_eval_neutral.json"

POS = "#16a34a"
NEG = "#dc2626"
PRIMARY = "#1e4d8c"
MUTED = "#64748b"


def load_close() -> pd.DataFrame:
    files = sorted(DAILY.glob("*.parquet"))
    files_str = "[" + ",".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    df = con.execute("""
        SELECT TradeDate as date, Symbol as symbol, Close as close
        FROM read_parquet({fs})
        WHERE TradeDate >= DATE '2019-01-02' AND TradeDate <= DATE '2026-08-24'
    """.format(fs=files_str)).df()
    mat = df.pivot_table(index='date', columns='symbol', values='close', aggfunc='first')
    mat.index = pd.to_datetime(mat.index)
    return mat.sort_index()


def load_industry() -> pd.DataFrame:
    """每日 (date, symbol) -> sw_l1 行业名。多行取第一个 sw_l1。"""
    files = sorted(INDUSTRY.glob("*.parquet"))
    files_str = "[" + ",".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    df = con.execute("""
        SELECT TradeDate as date, Symbol as symbol, IndustryName
        FROM read_parquet({fs})
        WHERE IndustrySource = 'sw_l1'
          AND TradeDate >= DATE '2019-01-02' AND TradeDate <= DATE '2026-08-24'
    """.format(fs=files_str)).df()
    df = df.drop_duplicates(subset=['date', 'symbol'], keep='first')
    df = df.pivot_table(index='date', columns='symbol', values='IndustryName', aggfunc='first')
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_mktcap() -> pd.DataFrame:
    """每日 (date, symbol) -> log(流通市值) 用于市值中性化。"""
    files = sorted(VALUATION.glob("*.parquet"))
    files_str = "[" + ",".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    df = con.execute("""
        SELECT TradeDate as date, Symbol as symbol, CirculatingMarketCap
        FROM read_parquet({fs})
        WHERE TradeDate >= DATE '2019-01-02' AND TradeDate <= DATE '2026-08-24'
    """.format(fs=files_str)).df()
    df["log_mktcap"] = np.log(df["CirculatingMarketCap"].replace(0, np.nan))
    df = df.pivot_table(index='date', columns='symbol', values='log_mktcap', aggfunc='first')
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def neutralize_matrix(fv: np.ndarray, X: np.ndarray) -> np.ndarray:
    """横截面回归取残差：fv (T,N), X (T,N,K)。每日期最小二乘。
    返回 (T,N) 残差矩阵（含 NaN）。
    有效行 = y 与 X 的截距/市值列非 NaN 且市值有值；
    每日期动态剔除"整列全0/无成员行业"列（用 keep_cols）。
    """
    T, N = fv.shape
    K = X.shape[2]
    resid = np.full_like(fv, np.nan)
    for t in range(T):
        Xt = X[t]      # (N,K)
        yt = fv[t]     # (N,)
        # 有效：y非NaN、市值列(最后)有限、且至少一个行业非NaN
        valid = np.isfinite(yt) & np.isfinite(Xt[:, -1])
        nv = np.where(valid)[0]
        if len(nv) < 30:
            continue
        Xv = Xt[nv]          # (M,K)
        yv = yt[nv]          # (M,)
        # 剔除整列全0 或含NaN 的列（行业哑变量/市值列）
        finite_cols = np.isfinite(Xv).all(axis=0)
        keep_cols = finite_cols & (np.abs(Xv).sum(axis=0) > 1e-12)
        Xv = Xv[:, keep_cols]
        if Xv.shape[1] < 1 or Xv.shape[0] < Xv.shape[1] + 10:
            continue
        try:
            coef, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
            pred = Xv @ coef
            resid[t, nv] = yv - pred
        except np.linalg.LinAlgError:
            continue
    return resid


def compute_daily_rankic(factor_mat: pd.DataFrame, close: pd.DataFrame) -> pd.Series:
    """逐日 Spearman rank ic (T,)；因子列与 close 列对齐。"""
    common = factor_mat.index.intersection(close.index)
    fv = factor_mat.loc[common].reindex(columns=close.columns)
    fwd = close.pct_change().shift(-1)
    T = len(common)
    ic = np.full(T, np.nan)
    fv_v = fv.values
    fr_v = fwd.loc[common].values
    for t in range(T):
        m = fv_v[t]
        r = fr_v[t]
        mask = np.isfinite(m) & np.isfinite(r)
        if mask.sum() < 20:
            continue
        mv = m[mask]; rv = r[mask]
        rm = mv.argsort().argsort()
        rr = rv.argsort().argsort()
        sm, sr = rm.std(), rr.std()
        if sm > 1e-9 and sr > 1e-9:
            ic[t] = float(((rm - rm.mean()) * (rr - rr.mean())).sum() / (len(mv) * sm * sr))
    return pd.Series(ic, index=common)


def agg_ic(ic: pd.Series) -> dict:
    s = ic.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 5:
        return {}
    mean = float(s.mean())
    std = float(s.std())
    return {
        "mean_rankic": mean,
        "rankic_ir": mean / std if std > 0 else 0.0,
        "win_rate": float((s > 0).sum() / len(s)),
        "n_periods": len(s),
        "monthly": s.resample("ME").mean().dropna(),
        "series": s,
    }


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def plot_neutral_compare(raw_ic, neu_ic, factor_name) -> str:
    """raw vs neutral 的 RankIC 时序叠加 + 柱状 RankICIR 对比。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4),
                                   gridspec_kw={"width_ratios": [2.4, 1]})
    fig.patch.set_facecolor("#ffffff")
    common = raw_ic.index
    ax1.plot(common, raw_ic.rolling(20).mean(), color=PRIMARY, lw=1.2, label="Raw RankIC(MA20)")
    ax1.plot(common, neu_ic.reindex(common).rolling(20).mean(), color="#0d9488", lw=1.2, label="中性化后(MA20)")
    ax1.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax1.set_title("RankIC 时序: 原始 vs 中性化", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(6))

    raw_ir = raw_ic.mean() / raw_ic.std() if raw_ic.std() > 0 else 0
    neu_ir = neu_ic.mean() / neu_ic.std() if neu_ic.std() > 0 else 0
    ax2.bar(["Raw", "中性化"], [raw_ir, neu_ir],
            color=[PRIMARY, "#0d9488"], width=0.5)
    ax2.set_title(f"RankIC IR: {raw_ir:.3f} → {neu_ir:.3f}", fontsize=10)
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig_to_b64(fig)


def main():
    t0 = time.time()
    print("加载 close / 行业 / 市值 ...", flush=True)
    close = load_close()
    industry = load_industry()
    mktcap = load_mktcap()
    print(f"close {close.shape}, industry {industry.shape}, mktcap {mktcap.shape}, "
          f"{time.time()-t0:.0f}s", flush=True)

    # 对齐到 close 的日期与股票列
    common_idx = close.index
    common_cols = close.columns
    industry = industry.reindex(index=common_idx, columns=common_cols)
    mktcap = mktcap.reindex(index=common_idx, columns=common_cols)

    # 构造每日 X 矩阵 (T,N,K)
    print("构造行业哑变量 + 市值列 ...", flush=True)
    ind_mat = industry.fillna("").values
    uniq_inds = sorted({x for row in ind_mat for x in row if x})
    K = len(uniq_inds) + 2  # 截距 + 行业 + log市值
    T, N = close.shape
    X = np.zeros((T, N, K), dtype=np.float64)
    X[:, :, 0] = 1.0  # 截距
    for i, name in enumerate(uniq_inds):
        X[:, :, 1 + i] = (ind_mat == name).astype(np.float64)
    X[:, :, 1 + len(uniq_inds)] = mktcap.values  # log市值
    # 行业哑变量列全0（当日该行业无人）→ 置 NaN，中性化时会剔除该列
    for i in range(len(uniq_inds)):
        colsum = X[:, :, 1 + i].sum(axis=1)
        X[colsum == 0, :, 1 + i] = np.nan
    print(f"X shape {X.shape}, 行业数 {len(uniq_inds)}", flush=True)

    # 读因子值
    print("读 factor_values.parquet ...", flush=True)
    fv_full = pd.read_parquet(FV_PATH, engine='pyarrow',
                              thrift_string_size_limit=2**31-1,
                              thrift_container_size_limit=2**31-1)
    factor_names = sorted({c[0] for c in fv_full.columns})
    print(f"共 {len(factor_names)} 个因子", flush=True)

    out = {}
    series_store = {}  # 供注入时画图，不入 json
    # 用共享的 X 回归 61 个因子（X不变，可多因子复用）
    for fi, fn in enumerate(factor_names):
        cols = [c for c in fv_full.columns if c[0] == fn]
        mat = fv_full[cols].copy()
        mat.columns = [c[1] for c in mat.columns]
        mat = mat.reindex(index=common_idx, columns=common_cols)
        if mat.isna().all().all():
            out[fn] = {"raw": {}, "neutral": {}, "error": "empty"}
            continue
        fv = mat.values.astype(np.float64)

        # raw rankic
        raw_ic = compute_daily_rankic(mat, close)

        # 中性化残差
        resid = neutralize_matrix(fv, X)
        resid_df = pd.DataFrame(resid, index=common_idx, columns=common_cols)
        neu_ic = compute_daily_rankic(resid_df, close)

        raw_a = agg_ic(raw_ic)
        neu_a = agg_ic(neu_ic)

        out[fn] = {
            "raw": {k: v for k, v in raw_a.items() if k in ("mean_rankic", "rankic_ir", "win_rate", "n_periods")},
            "neutral": {k: v for k, v in neu_a.items() if k in ("mean_rankic", "rankic_ir", "win_rate", "n_periods")},
        }
        series_store[fn] = {"raw": raw_ic, "neutral": neu_ic}
        if fi % 10 == 0:
            print(f"  {fi}/{len(factor_names)} {fn}: raw_ir={raw_a.get('rankic_ir'):.3f} neu_ir={neu_a.get('rankic_ir'):.3f}", flush=True)

    # 序列也存（供注入时画图）
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"评估已存 {OUT_JSON}, 用时 {time.time()-t0:.0f}s", flush=True)

    # 注入详情页
    print("注入中性化区块到详情页 ...", flush=True)
    injected = missing = 0
    for name in factor_names:
        clean = name.replace("factor_", "")
        page = FACTORS_DIR / f"factor_{clean}.html"
        if not page.exists():
            # 尝试其他命名
            cands = list(FACTORS_DIR.glob(f"factor_{clean}.html")) + list(FACTORS_DIR.glob(f"{clean}.html"))
            page = cands[0] if cands else None
        if page is None:
            missing += 1
            continue
        try:
            entry = out.get(name)
            raw = (entry or {}).get("raw", {})
            neu = (entry or {}).get("neutral", {})
            html = page.read_text(encoding="utf-8")
            # 若已注入则跳过
            if "中性化" in html:
                continue
            # 生成对比图
            raw_s = (series_store.get(name) or {}).get("raw", pd.Series(dtype=float))
            neu_s = (series_store.get(name) or {}).get("neutral", pd.Series(dtype=float))
            chart = ""
            if len(raw_s) > 5:
                chart = plot_neutral_compare(raw_s, neu_s, clean)
            block = f"""
<div class="card">
  <h2>🧪 中性化前后 RankIC / RankICIR</h2>
  <p style="font-size:0.82rem;color:#64748b;margin:0 0 10px">
    口径：每日横截面，对因子值做 行业(sw_l1)哑变量 + log(流通市值) 线性回归取残差，再算残差的 RankIC。
  </p>
  {f'<div><img src="{chart}" style="width:100%;border-radius:8px" alt="中性化对比"></div>' if chart else ''}
  <table class="meta-table" style="margin-top:10px">
    <thead><tr><th>指标</th><th>原始 Raw</th><th>中性化后</th><th>变化</th></tr></thead>
    <tbody>
      <tr><td>Mean RankIC</td><td>{raw.get('mean_rankic', 0):.4f}</td><td>{neu.get('mean_rankic', 0):.4f}</td><td>{neu.get('mean_rankic', 0) - raw.get('mean_rankic', 0):+.4f}</td></tr>
      <tr><td>RankIC IR</td><td>{raw.get('rankic_ir', 0):.3f}</td><td>{neu.get('rankic_ir', 0):.3f}</td><td>{neu.get('rankic_ir', 0) - raw.get('rankic_ir', 0):+.3f}</td></tr>
      <tr><td>胜率</td><td>{raw.get('win_rate', 0):.1%}</td><td>{neu.get('win_rate', 0):.1%}</td><td>{neu.get('win_rate', 0) - raw.get('win_rate', 0):+.1%}</td></tr>
      <tr><td>交易日</td><td>{raw.get('n_periods', 0)}</td><td>{neu.get('n_periods', 0)}</td><td>—</td></tr>
    </tbody>
  </table>
</div>
"""
            # 在 "RankIC 分布" 区块后插入；若找不到就在 </main> 前插入
            marker = "</div>\n\n</main>"
            if marker in html:
                html = html.replace(marker, block + marker, 1)
            else:
                html = html.replace("</main>", block + "\n</main>", 1)
            page.write_text(html, encoding="utf-8")
            injected += 1
        except Exception as e:
            print(f"  [注入失败] {name}: {e}")
    print(f"注入完成: {injected} 成功, {missing} 缺页", flush=True)


if __name__ == "__main__":
    main()
