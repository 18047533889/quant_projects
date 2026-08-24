# RankIC ≥ 0.03 候选因子统一回测报告模板

入口：`index.html`。模板可直接通过本地文件打开，不依赖外部 CDN。

## 当前内容

- 172 个候选因子及三条落值路由已预填：LQTP 130、factor_engine 36、Python 6。
- 所有回测数值、日期、文件路径、序列、相关矩阵均为空；空值显示为 `—`，不代表 0。
- `factor_detail.html?factor=<factor_key>` 是统一详情页，包含 Week3 的 RankIC、十分位、多空、TopK 四段，并增加因子自相关。
- 汇总页增加因子间相关性热力图、三条路径职责、服务器运行方法、结果清单和 HTML 填充映射。

## 文件

- `index.html`：报告入口、方法说明、172 因子清单、热力图和填充说明。
- `factor_detail.html`：动态单因子详情模板。
- `report_data.js`：结构化模板数据；结果字段为 `null` 或空数组。
- `../../tools/build_factor_backtest_report_data.py`：从三份路由 JSON 重新生成空的 `report_data.js`。

## 重要约定

筛选条件固定为：`mean_rank_ic > 0.03 and topk_sharpe > 0`。

第一阶段不做因子组合。批量 RunFactor、config-dir 或 run_many 只用于加速独立落值；每个因子仍需独立的 factor panel、RankIC/分层/多空结果、TopK backtest_id 和自相关结果。通过筛选后才进入相关性去冗余；可选组合必须作为单独的新因子回测。

真实结果应由统一评估程序更新 `report_data.js` 的相同数据结构。不要在已有结果后重新运行空模板生成器，否则会覆盖结果。

## LQTP 提交公式（本批剩余因子）

已按 LQTP 手册 **2026-07-19** 完成命名对齐与公式转换：

- [`lqtp_submission_formulas.md`](lqtp_submission_formulas.md) — 可直接复制到 `factor_client.py run --formula ...`
- [`lqtp_submission_formulas.json`](lqtp_submission_formulas.json) — 结构化清单

**专用转换器**（FE DSL / Python → LQTP）：

```bash
cd /home/shw/quant_projects
python scripts/convert_to_lqtp.py --dsl 'tanh(clip(cs_rank(close), -3, 3))'
python scripts/convert_to_lqtp.py --report-js /home/shw/reports/report_data.js \
  --out-json /home/shw/reports/lqtp_submission_formulas.json \
  --out-md /home/shw/reports/lqtp_submission_formulas.md
```

说明见 [`quant_projects/scripts/cogalpha_lqtp/README.md`](/home/shw/quant_projects/scripts/cogalpha_lqtp/README.md)。
算子对照：`quant_projects/factor_engine/docs/lqtp_vs_factor_engine_operators.md`。
