#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
456 因子中性化前后 RankIC/RankICIR 评估 + 注入详情页。

口径：每日横截面，对因子值做 行业(sw_l1)哑变量 + log(流通市值) 线性回归取残差，
再对残差算每日 rankic，聚合 rankic_ir / 胜率 / 月度IC。
窗口：2019-01-02 ~ 2026-08-24（全历史最新）。
输出：all_eval_neutral_456.json + 注入每个详情页"中性化前后"区块（含对比图）。
"""
import sys, os, json, glob, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64
warnings.filterwarnings("ignore")

PROJECT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "vectorbt_qs"))
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
FV_DIR = PROJECT / "weekly_backtest_output" / "factor_matrices_all"
OUT_JSON = REPORT_DIR / "all_eval_neutral_456.json"
DAILY = Path.home() / "cos_data" / "StockDailyBar"
INDUSTRY = Path.home() / "cos_data" / "StockIndustry"
VALUATION = Path.home() / "cos_data" / "StockValuationDaily"
START, END = "2019-01-02", "2026-08-24"


def load_close():
    files = sorted(DAILY.glob("*.parquet"))
    fs = "[" + ",".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT TradeDate as date, Symbol as symbol, Close as close
        FROM read_parquet({fs})
        WHERE TradeDate >= DATE '{START}' AND TradeDate <= DATE '{END}'
    """).df()
    m = df.pivot_table(index='date', columns='symbol', values='close', aggfunc='first')
    m.index = pd.to_datetime(m.index)
    return m.sort_index()


def load_industry():
    files = sorted(INDUSTRY.glob("*.parquet"))
    fs = "[" + ",".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT TradeDate as date, Symbol as symbol, IndustryName
        FROM read_parquet({fs})
        WHERE IndustrySource = 'sw_l1'
          AND TradeDate >= DATE '{START}' AND TradeDate <= DATE '{END}'
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


def neutralize_matrix(fv, X):
    """逐日横截面回归取残差。fv (T,N), X (T,N,K)。返回 (T,N) 残差。"""
    T, N = fv.shape
    resid = np.full((T, N), np.nan)
    for t in range(T):
        y = fv[t]
        Xt = X[t]
        ok = np.isfinite(y) & np.isfinite(Xt).all(axis=1)
        if ok.sum() < 30:
            continue
        Xo = Xt[ok]
        yo = y[ok]
        try:
            coef, _, _, _ = np.linalg.lstsq(Xo, yo, rcond=None)
            pred = Xo @ coef
            resid[t, ok] = yo - pred
        except Exception:
            pass
    return resid


def compute_daily_rankic(factor_mat, close):
    common = factor_mat.index.intersection(close.index)
    fv = factor_mat.reindex(index=common)
    cc = close.loc[common]
    cols = cc.columns.intersection(fv.columns)
    fv = fv[cols]
    cc = cc[cols]
    fwd = cc.pct_change().shift(-1)
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


def agg_ic(s):
    s = s.dropna()
    if len(s) == 0:
        return {"mean_rankic": 0.0, "rankic_ir": 0.0, "win_rate": 0.0, "n_periods": 0}
    mean = float(s.mean()); std = float(s.std())
    return {
        "mean_rankic": mean,
        "rankic_ir": mean / std if std > 1e-9 else 0.0,
        "win_rate": float((s > 0).sum() / len(s)),
        "n_periods": len(s),
    }


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#fff")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def plot_neutral_compare(raw_ic, neu_ic, name):
    common = raw_ic.index.intersection(neu_ic.index)
    raw = raw_ic.reindex(common).rolling(20).mean()
    neu = neu_ic.reindex(common).rolling(20).mean()
    fig, ax1 = plt.subplots(figsize=(10, 3.2))
    ax1.plot(common, raw, color="#dc2626", lw=1.0, label="原始 (MA20)")
    ax1.plot(common, neu, color="#0d9488", lw=1.2, label="中性化后 (MA20)")
    ax1.axhline(0, color="#94a3b8", lw=0.6)
    ax1.set_title(f"{name} — RankIC 时序: 原始 vs 中性化", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(ls="--", alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


def main():
    print("加载 close/行业/市值 ...", flush=True)
    close = load_close()
    industry = load_industry()
    mktcap = load_mktcap()
    common_idx = close.index
    common_cols = close.columns
    industry = industry.reindex(index=common_idx, columns=common_cols)
    mktcap = mktcap.reindex(index=common_idx, columns=common_cols)

    print("构造行业哑变量 + 市值 X ...", flush=True)
    ind_mat = industry.fillna("").values
    uniq_inds = sorted({x for row in ind_mat for x in row if x})
    K = len(uniq_inds) + 2
    T, N = close.shape
    X = np.zeros((T, N, K), dtype=np.float64)
    X[:, :, 0] = 1.0
    for i, name in enumerate(uniq_inds):
        X[:, :, 1 + i] = (ind_mat == name).astype(np.float64)
    X[:, :, 1 + len(uniq_inds)] = mktcap.values
    for i in range(len(uniq_inds)):
        colsum = X[:, :, 1 + i].sum(axis=1)
        X[colsum == 0, :, 1 + i] = np.nan
    print(f"X {X.shape}, 行业数 {len(uniq_inds)}", flush=True)

    names = sorted([f.stem for f in FV_DIR.glob("*.parquet")]) if FV_DIR.exists() else []
    print(f"共 {len(names)} 个因子", flush=True)

    out = {}
    series_store = {}
    t0 = time.time()
    for fi, page in enumerate(names):
        fpath = FV_DIR / f"{page}.parquet"
        try:
            mat = pd.read_parquet(fpath)
        except Exception:
            continue
        if mat.shape[1] == 0 or mat.isna().all().all():
            out[page] = {"raw": {}, "neutral": {}, "error": "empty"}
            continue
        mat = mat.reindex(index=common_idx, columns=common_cols)
        fv = mat.values.astype(np.float64)
        raw_ic = compute_daily_rankic(mat, close)
        resid = neutralize_matrix(fv, X)
        resid_df = pd.DataFrame(resid, index=common_idx, columns=common_cols)
        neu_ic = compute_daily_rankic(resid_df, close)
        out[page] = {"raw": agg_ic(raw_ic), "neutral": agg_ic(neu_ic)}
        series_store[page] = {"raw": raw_ic, "neutral": neu_ic}
        if fi % 50 == 0:
            print(f"  {fi}/{len(names)} {page} 耗时{time.time()-t0:.0f}s", flush=True)

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    print(f"评估存 {OUT_JSON} 用时{time.time()-t0:.0f}s", flush=True)

    # 注入详情页
    injected = 0
    for page in names:
        html_path = FACTORS_DIR / f"factor_{page}.html"
        if not html_path.exists():
            continue
        entry = out.get(page) or {}
        raw = entry.get("raw", {}) or {}
        neu = entry.get("neutral", {}) or {}
        try:
            html = html_path.read_text(encoding="utf-8")
        except Exception:
            continue
        if "中性化" in html:
            continue
        raw_s = (series_store.get(page) or {}).get("raw", pd.Series(dtype=float))
        neu_s = (series_store.get(page) or {}).get("neutral", pd.Series(dtype=float))
        chart = plot_neutral_compare(raw_s, neu_s, page) if len(raw_s.dropna()) > 5 else ""
        block = f"""
<div class="card">
  <h2>🧪 中性化前后 RankIC / RankICIR</h2>
  <p style="font-size:0.82rem;color:#64748b;margin:0 0 10px">口径：每日横截面，对因子值做 行业(sw_l1)哑变量 + log(流通市值) 线性回归取残差，再算残差的 RankIC。</p>
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
        if "</main>" in html:
            html = html.replace("</main>", block + "\n</main>", 1)
        html_path.write_text(html, encoding="utf-8")
        injected += 1
    print(f"注入 {injected} 页")


if __name__ == "__main__":
    main()
