"""
批量回测 + 可视化 — 后台运行脚本

遍历 examples/configs/ 下所有 YAML，逐一回测并生成：
  - performance_stats.csv   完整绩效指标
  - equity_curve.png        净值曲线 + 回撤
  - monthly_returns.png     月度收益热力图
  - annual_returns.png      年度收益柱状图
  - portfolio_dashboard.html vectorbt Plotly 交互式报告

用法:
    Set-Location D:\quantsociety\whh_local_workspace
    $env:DATA_ACCESS_SKIP_COS_MIRROR = "1"
    $env:ASHARE_PARQUET_ROOT = "D:\quantsociety\whh_local_workspace\lqtp_data"
    python vectorbt_qs/scripts/batch_backtest_viz.py
"""
import sys, os, time
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ.parent))
sys.path.insert(0, str(_PROJ / "vectorbt"))

import pandas as pd
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from vectorbt_qs.mvp.engine.runner import run_backtest, portfolio_report
from vectorbt_qs.mvp.visualization import build_nav_curves, export_plotly_dashboard

CONFIGS_DIR = _PROJ / "examples" / "configs"
OUTPUT_ROOT = _PROJ / "examples" / "output"
DEFAULT_BT = {"init_cash": 10_000_000, "freq": "1D", "slippage": 0.0}


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_charts(pf, out_dir: Path, label: str, init_cash: float):
    """生成 3 张图表"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 净值曲线 + 回撤 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    nav_curves = build_nav_curves(pf)
    nav_handles = []
    nav_labels = []
    strategy_line, = ax1.plot(
        nav_curves.index,
        nav_curves["strategy_nav"],
        linewidth=1.2,
        color="#1f77b4",
    )
    nav_handles.append(strategy_line)
    nav_labels.append("Strategy NAV")
    if "benchmark_nav" in nav_curves:
        benchmark_symbol = getattr(pf, "_qs_benchmark_symbol", "Benchmark")
        benchmark_line, = ax1.plot(
            nav_curves.index,
            nav_curves["benchmark_nav"],
            linewidth=1.0,
            color="#7f7f7f",
        )
        excess_line, = ax1.plot(
            nav_curves.index,
            nav_curves["excess_nav"],
            linewidth=1.0,
            color="#d62728",
        )
        nav_handles.extend([benchmark_line, excess_line])
        nav_labels.extend([
            f"Benchmark NAV ({benchmark_symbol})",
            "Excess NAV",
        ])
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Net Value")
    ax1.set_title(f"{label} — Equity Curve", fontsize=14, fontweight="bold")
    ax1.legend(nav_handles, nav_labels, loc="best")
    ax1.grid(True, alpha=0.3)

    dd = pf.drawdown()
    ax2.fill_between(dd.index, dd.values * 100, 0, color="#d62728", alpha=0.6)
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "equity_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 月度收益热力图 ---
    returns = pf.returns()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    mdf = monthly.to_frame("return")
    mdf["year"] = mdf.index.year
    mdf["month"] = mdf.index.month
    pivot = mdf.pivot(index="year", columns="month", values="return") * 100

    fig, ax = plt.subplots(figsize=(14, max(len(pivot) * 0.4 + 2, 4)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-15, vmax=15)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot)):
        for j in range(12):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 8 else "black")
    ax.set_title(f"{label} — Monthly Returns (%)", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "monthly_returns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 年度收益 ---
    yearly = mdf.groupby("year")["return"].apply(lambda x: (1 + x).prod() - 1) * 100
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in yearly.values]
    ax.bar(yearly.index.astype(str), yearly.values, color=colors, edgecolor="white")
    ax.axhline(y=0, color="gray", linewidth=0.8)
    for i, v in enumerate(yearly.values):
        ax.text(i, v + (2 if v >= 0 else -4), f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_title(f"{label} — Annual Returns", fontsize=14, fontweight="bold")
    ax.set_ylabel("Return %")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "annual_returns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- vectorbt Plotly 交互式组合报告 ---
    export_plotly_dashboard(
        pf,
        out_dir / "portfolio_dashboard.html",
        title=f"{label} — Portfolio Overview",
        include_plotlyjs="cdn",
    )


def main():
    configs = sorted(CONFIGS_DIR.glob("*.yaml"))
    print(f"找到 {len(configs)} 个配置文件\n")

    total_start = time.time()
    for i, cfg_path in enumerate(configs, 1):
        name = cfg_path.stem
        out_dir = OUTPUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{i}/{len(configs)}] {name} ...", end=" ", flush=True)
        t0 = time.time()

        try:
            cfg = load_config(cfg_path)
            positions_path = Path(cfg["input"]["positions"])
            if not positions_path.is_absolute():
                positions_path = cfg_path.parent / positions_path
            tw = pd.read_parquet(positions_path)

            bt_cfg = {**DEFAULT_BT, **cfg.get("backtest", {})}

            pf = run_backtest(cfg.get("market", "ashare"), tw, config=bt_cfg)
            stats = portfolio_report(pf)
            stats.to_csv(out_dir / "performance_stats.csv")

            # 图表
            init_cash = bt_cfg.get("init_cash", 10_000_000)
            make_charts(pf, out_dir, name, init_cash)

            ret = stats.get("Total Return [%]", 0)
            elapsed = time.time() - t0
            print(f"OK ({elapsed:.0f}s, ret={ret:.1f}%)")

        except Exception as e:
            print(f"FAIL: {e}")

    total = time.time() - total_start
    print(f"\n全部完成，总耗时 {total/60:.1f} 分钟")


if __name__ == "__main__":
    main()
