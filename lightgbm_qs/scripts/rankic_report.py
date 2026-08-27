# -*- coding: utf-8 -*-
"""rankIC / IC 展示报告: 从滚动前向预测(predictions_adj.parquet) + 后复权中性化标签(fwd_adj_neu.parquet)
计算并输出:
  1. 整体/逐月的 预测 rankIC 与 IC (截面 Spearman/Pearson, 预测 pred vs 标签 fwd_neu)
  2. 滚动20日均值曲线 + 月度表格
  3. 专门的报告 markdown + 图(pred_ic_adj.png)
输出: outputs/rankic_report.md, outputs/pred_ic_adj.png
用户强调: 模型预测的 rankIC 和 IC 很重要, 要专门展示。
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
BUILD = os.path.join(ROOT, "data/build")
OUT = os.path.join(ROOT, "outputs")
PRED = os.path.join(BUILD, "predictions_adj.parquet")
LABEL = os.path.join(BUILD, "fwd_adj_neu.parquet")
OOS_START = "2019-01-01"

C_AQUA = "#1baf7a"
C_INK2 = "#52514e"

def log(m):
    print(m, flush=True)

def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    pred = pd.read_parquet(PRED)
    pred["date"] = pd.to_datetime(pred["date"])
    lab = pd.read_parquet(LABEL)
    lab["date"] = pd.to_datetime(lab["date"])
    m = pred.merge(lab, on=["date", "asset"], how="inner")
    m = m[m["date"] >= pd.Timestamp(OOS_START)]
    log(f"pred cells={len(m)} range={m.date.min().date()}..{m.date.max().date()}")

    # per-date IC
    rows = []
    for d, g in m.groupby("date"):
        v = g.dropna(subset=["pred", "fwd_neu"])
        if len(v) < 10:
            continue
        p = v["pred"].to_numpy(); y = v["fwd_neu"].to_numpy()
        if not (np.isfinite(p).all() and np.isfinite(y).all()):
            continue
        ic = np.corrcoef(p, y)[0, 1]
        ric = pd.Series(p).rank().corr(pd.Series(y).rank())
        rows.append((d, float(ic) if np.isfinite(ic) else np.nan,
                     float(ric) if np.isfinite(ric) else np.nan))
    icdf = pd.DataFrame(rows, columns=["date", "IC", "rankIC"]).dropna()
    icdf = icdf.set_index("date").sort_index()
    log(f"valid IC days={len(icdf)}")

    # monthly + overall
    icdf["ym"] = icdf.index.strftime("%Y-%m")
    monthly = icdf.groupby("ym").agg(mean_IC=("IC", "mean"), mean_rankIC=("rankIC", "mean"),
                                     days=("IC", "size"))
    overall_ic = float(icdf["IC"].mean()); overall_ric = float(icdf["rankIC"].mean())

    lines = ["# 模型预测 rankIC / IC 报告", "",
             f"- 预测输入: {os.path.basename(PRED)}",
             f"- 标签: {os.path.basename(LABEL)} (后复权Vwap 10日收益, 截面中性化)",
             f"- 期间: {icdf.index.min().date()} .. {icdf.index.max().date()}  共 {len(icdf)} 个截面日",
             "",
             "## 整体表现",
             f"| 指标 | 值 |",
             f"| --- | --- |",
             f"| 平均 IC (Pearson) | {overall_ic:.5f} |",
             f"| 平均 rankIC (Spearman) | {overall_ric:.5f} |",
             f"| IC>0 比例 | {(icdf['IC']>0).mean()*100:.1f}% |",
             f"| rankIC>0 比例 | {(icdf['rankIC']>0).mean()*100:.1f}% |",
             "",
             "## 月度表现", "",
             "| 月份 | 平均IC | 平均rankIC | 天数 |", "| --- | --- | --- | --- |"]
    for ym, r in monthly.iterrows():
        lines.append(f"| {ym} | {r['mean_IC']:.4f} | {r['mean_rankIC']:.4f} | {int(r['days'])} |")
    with open(os.path.join(OUT, "rankic_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"WROTE {OUT}/rankic_report.md  overall IC={overall_ic:.5f} rankIC={overall_ric:.5f}")

    # chart
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(icdf.index, icdf["rankIC"], lw=0.7, color=C_INK2, alpha=0.55, label="Daily rankIC")
    ax.plot(icdf.index, icdf["rankIC"].rolling(20, min_periods=5).mean(), lw=1.8, color=C_AQUA,
            label="20D mean rankIC")
    ax.axhline(0, color=C_INK2, lw=0.8, ls="--", alpha=0.6)
    ax.axhline(overall_ric, color="#e34948", lw=1.0, ls=":", label="Overall rankIC")
    ax.set_title(f"Model Prediction rankIC vs 10d adj-neutralized Vwap return (mean rankIC {overall_ric:.4f})")
    ax.set_xlabel("Date"); ax.set_ylabel("rankIC")
    ax.grid(True, color="#e1e0d9", lw=0.7, alpha=0.8); ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "pred_ic_adj.png"), dpi=150)
    plt.close(fig)
    log(f"WROTE {OUT}/pred_ic_adj.png  total {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
