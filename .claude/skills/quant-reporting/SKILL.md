---
name: quant-reporting
description: 量化研究/回测结果的报告产出 — 用 web-design（HTML 报告）+ huashu-slides（PPTX 汇报）+ vectorbt_qs（回测图/HTML dashboard）+ plotly 产出专业可视化。涵盖：因子评估报告、模型 OOS 报告、组合回测绩效报告。当任务涉及"写报告、做 PPT、汇报、结果展示、生成图表、dashboard、月度小结"时先读此 skill。
version: 1.0.0
---

# 量化报告产出（web-design + huashu-slides + vectorbt_qs）

**前提**：已读 quant-platform-workflow（口径）。本 skill 只管"把量化结果渲染成好看的东西"，**不改任何评估/回测口径**。

## 报告素材从哪来（先算对，再画美）

| 内容 | 真源 |
|---|---|
| 因子 IC / 分层 / 换手 | `quant_evaluator` 51 metric + `rankic_report.py` |
| 特征矩阵 / 去重 | `merge_all_factors.py` / `build_full_features.py` CSV |
| 模型 OOS 预测 | walk-forward 每折 OOS 预测矩阵 |
| 组合优化权重 | `riskfolio_qs` `target_positions.parquet` |
| 回测绩效 | `portfolio_report(pf)`（252 日口径） |
| 净值/超额曲线 | `build_nav_curves(pf)` → strategy/benchmark/excess NAV |
| 回测图/交互报告 | `export_plotly_dashboard(pf, path)` |

## 路线 1：HTML 报告（web-design skill，最常用）

单页 HTML 数据报告/落地页，Tailwind + Chart.js + Font Awesome，一键部署 Cloudflare Pages。

```bash
# 调 web-design skill 的约定（生成精美单页 HTML）
# 素材：从上述真源导出 CSV/JSON，用 Chart.js 画 IC 曲线、分层收益、净值、回撤
```

关键画什么：
- **因子报告**：IC 时序（rank_ic rolling）、分十层累计收益、IC 衰减图、换手成本柱状
- **模型报告**：OOS 逐折 rank_ic 表、IS/OOS 对比、预测分布
- **组合报告**：净值 vs 基准 vs 超额（几何）、回撤 underwater、滚动 Sharpe、月度收益热力图

## 路线 2：PPTX 汇报（huashu-slides skill）

```bash
# 调 huashu-slides（18 种设计风格），从 HTML 报告/图表导出 PNG 插图，组织成汇报
# 结构建议：研究问题 → 数据口径 → 因子/模型 → 组合 → 回测绩效 → 风险 → 结论与下一步
```

## 路线 3：vectorbt_qs 自带产物（回测专用，零额外开发）

```python
from vectorbt_qs.mvp.engine.runner import run_backtest, portfolio_report, compare_reports
from vectorbt_qs.mvp.visualization import build_nav_curves, export_plotly_dashboard

pf = run_backtest("ashare", target_positions, config={...})   # 见 quant-ml-optimization
nav = build_nav_curves(pf)                 # strategy_nav / benchmark_nav / excess_nav
export_plotly_dashboard(pf, "output/portfolio_dashboard.html", title="组合回测",
                        include_plotlyjs="cdn")
# 批量对比多策略
stats_df = compare_reports({"策略A": pf, "策略B": pf2})   # 行=label，列=指标
stats_df.to_csv("output/comparison.csv", encoding="utf-8-sig")
```

CLI 方式：
```bash
cd /home/sunhaiwei/quant_projects
.venv/bin/python -m vectorbt_qs backtest --positions target_positions.parquet --plot \
    --output-dir output --start 2024-01-01
# 产出 equity_curve.png + portfolio_dashboard.html + 控制台指标
```

## 输出规范（量化报告的铁律）

1. **口径标注**：所有收益数字必须标"后复权 vwap-to-vwap / 252 交易日 / 单边费用 Xbps"。
2. **用 portfolio_report 不用 pf.stats()**：252 交易日年化 vs 365 天年化会显著不同，别画错。
3. **基准必须挂**：`benchmark_index`（000300/000905/000852），画净值同时给基准 + 超额。
4. **别画假结论**：rank_ic>0.1 的因子图要标"疑似泄漏待查"，先 debug 再报告。
5. **CSV 用 utf-8-sig**（vectorbt_qs CLI 同款），Excel 打开中文不乱码。
6. **证据齐全**：HTML/PPT 内嵌 git_sha、数据版本、metric 版本；NOT_RUN 项如实标。

## 禁止

- ❌ 报告里出现未复权收益/365 天年化/裸 pf.stats()
- ❌ 用 AI 手画的 matplotlib 替代库内现成产物（export_plotly_dashboard/build_nav_curves 都有）
- ❌ 把可疑泄漏的因子结果当结论汇报
- ❌ 报告与评估口径不一致（改图不改口径，先对齐）
