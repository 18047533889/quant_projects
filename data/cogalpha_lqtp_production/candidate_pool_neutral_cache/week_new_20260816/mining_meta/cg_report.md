# CogAlpha v1 fleet 汇总 · 2026-08-13

> **CogAlpha v1 口径**：无 0–10 Logic Score；质检门禁为 accept / repair / reject；因子分级为 qualified / elite；预测期 horizon_days=1（T+1 open → T+2 open，非 mining_config 字段）。

Campaign stamp: `202608130940`（8 worker 独立 candidate_pool，factor_pool 不合并）

> **本次混合汇报口径**：
> - **A股**：fleet 四路 worker 正式产物（`cogalpha_ashare_pv_202608130940_w01`–`w04`）。
> - **美股 fleet worker**：四路仍在挖掘，本节仅展示池内进度，**不导出** fleet candidate_pool。
> - **美股 interim**：引用 legacy 共享池 `cogalpha_us_stock_pv_202606291711`（2026-06 批次，共 6 个 qualified 因子），**非** fleet 分 worker 产物。

## 1. 口径（miner_delivery_spec §2.7）

| 参数 | 值 |
|------|---|
| train | N/A ~ N/A |
| valid | N/A ~ N/A |
| test | N/A ~ N/A |
| 预测期 | 1 日 (T+1 open → T+2 open) | horizon_days=1 |
| 截面算子库 A/B | w01-w02 **ON** · w03-w04 **OFF**（A股/美股各 2+2） |
| 原始数据 | `~/quant_projects/data/a_share/lqtp_data` · `~/quant_projects/data/us_stock/massive_data` |

## 2. Worker 产物对比

> 截面算子库 ON：agent 可选用 `cs_rank` / `cs_zscore` 等 `cross_sectional_transform`；OFF：仅允许 `none`。

| instance | worker | 状态 | 截面算子库 | qualified | elite | pool | manifest | token 请求 | campaign_id |
|----------|--------|------|:----------:|----------:|------:|-----:|---------:|-----------:|-------------|
| cogalpha_ashare_v1 | w01 | 已完成 | ON | 83 | 14 | 340 | 0 | 0 | `cogalpha_ashare_pv_202608130940_w01` |
| cogalpha_ashare_v1 | w02 | 已完成 | ON | 92 | 10 | 344 | 0 | 0 | `cogalpha_ashare_pv_202608130940_w02` |
| cogalpha_ashare_v1 | w03 | 已完成 | OFF | 67 | 17 | 315 | 0 | 0 | `cogalpha_ashare_pv_202608130940_w03` |
| cogalpha_ashare_v1 | w04 | 已完成 | OFF | 149 | 34 | 565 | 0 | 0 | `cogalpha_ashare_pv_202608130940_w04` |
| cogalpha_usstock_sp500_v1 | w01 | 已完成 | ON | 16 | 0 | 495 | —（未导出） | 0 | `cogalpha_us_sp500_pv_202608130940_w01` |
| cogalpha_usstock_sp500_v1 | w02 | 已完成 | ON | 33 | 0 | 525 | —（未导出） | 0 | `cogalpha_us_sp500_pv_202608130940_w02` |
| cogalpha_usstock_sp500_v1 | w03 | 已完成 | OFF | 33 | 0 | 522 | —（未导出） | 0 | `cogalpha_us_sp500_pv_202608130940_w03` |
| cogalpha_usstock_sp500_v1 | w04 | 已完成 | OFF | 80 | 26 | 624 | —（未导出） | 0 | `cogalpha_us_sp500_pv_202608130940_w04` |

## 2.1 美股 interim（legacy 共享池，非 fleet worker）

> campaign：`cogalpha_us_stock_pv_202606291711` · 来源：fleet 启动前共享 `outputs/factor_pool` 挖掘批次；fleet 四路 worker 仍在跑，正式产物待完成后导出。

| # | factor | train IC | valid IC | test IC | candidate_id |
|--:|--------|--------:|---------:|--------:|--------------|
| 1 | `factor_volatility_expansion_liquidity_reversal_linear` | 0.0165 | 0.0084 | 0.0086 | `cogalpha_20260629092214_777dc1a5` |
| 2 | `factor_volatility_expansion_reversal_smooth` | 0.0170 | 0.0129 | 0.0073 | `cogalpha_20260629094145_af4bfbf5` |
| 3 | `factor_range_expansion_reversal` | 0.0169 | 0.0118 | 0.0079 | `cogalpha_20260629100032_3dc90493` |
| 4 | `factor_volume_rank_trend_42d` | 0.0170 | 0.0076 | 0.0118 | `cogalpha_20260629101928_15200614` |
| 5 | `factor_volume_rank_trend_smooth` | 0.0154 | 0.0059 | 0.0088 | `cogalpha_20260629103521_2645075a` |
| 6 | `factor_log_volume_rank_trend` | 0.0235 | 0.0442 | 0.0293 | `cogalpha_20260629105046_fa907e03` |

## 3. Token 汇总

| 项目 | 值 |
|------|---:|
| 总 API 请求 | 0 |
| 总输入 tokens | 0 |
| 总输出 tokens | 0 |
| 总 tokens | 0 |

## 4. Valid+Test 段 Alphalens 回测（日频累积）

> 窗口：`N/A` ~ `N/A`（valid+test；test 段起点 `N/A`）；池：qualified + elite；质量来自日频 `valid_test/daily_*`（战役结束默认不再统一批回测）。

| instance | worker | 截面算子库 | 状态 | 回测因子 | 失败 | 平均 |IC| | Top factor | Top IC |
|----------|--------|:----------:|------|--------:|-----:|---------:|-----------|-------:|
| cogalpha_ashare_v1 | w01 | ON | missing | 0 | 0 | N/A | N/A | N/A |
| cogalpha_ashare_v1 | w02 | ON | missing | 0 | 0 | N/A | N/A | N/A |
| cogalpha_ashare_v1 | w03 | OFF | missing | 0 | 0 | N/A | N/A | N/A |
| cogalpha_ashare_v1 | w04 | OFF | missing | 0 | 0 | N/A | N/A | N/A |
| cogalpha_usstock_sp500_v1 | w01 | ON | missing | 0 | 0 | N/A | N/A | N/A |
| cogalpha_usstock_sp500_v1 | w02 | ON | missing | 0 | 0 | N/A | N/A | N/A |
| cogalpha_usstock_sp500_v1 | w03 | OFF | missing | 0 | 0 | N/A | N/A | N/A |
| cogalpha_usstock_sp500_v1 | w04 | OFF | missing | 0 | 0 | N/A | N/A | N/A |

## 5. 路径

- candidate_pool：`~/quant_projects/data/factor_pools/candidate_pool/cogalpha_{market}_pv_{stamp}_w{NN}/`
- 内部池：`{dev_root}/outputs/fleet_workers/w{NN}/factor_pool`
- worker 配置：`{dev_root}/outputs/fleet_workers/w{NN}/worker_config.json`（含截面算子库 ON/OFF）
- valid+test 回测（日频）：`{dev_root}/outputs/fleet_workers/w{NN}/backtests/valid_test/daily_{YYYY-MM-DD}/`
- 可选批跑 compact（legacy）：`.../backtests/test_split/compact/overall_test_split.csv`

