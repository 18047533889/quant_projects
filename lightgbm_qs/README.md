# lightgbm_qs — LightGBM 因子训练与回测研究管线

A 股因子研究主链路（脚本集，非库）：**因子矩阵 → LightGBM 训练（GPU/CPU）→
vwap-to-vwap 标签 → 组合优化 → vectorbt_qs 回测 → 报告**。
运行于 `quant_projects` monorepo 内，直接复用各平台库，不做任何重复实现。

**定位:** 企业级 A 股横截面日频多因子量化项目的**端到端研究管线** —— 从因子池出发，
一条命令跑完筛选 → 特征 → 训练 → 回测 → 报告。

**仓库:** https://github.com/HKUST-QUANT-SOCIETY/lightgbm_qs (私有)
**版本:** 不设版本号（研究脚本集，非打包库）

> ⚠️ 本仓库**只同步代码/文档/测试**（`scripts/`、`tests/`、`MISTAKES_AND_LEAKAGE_LESSONS.md`）。
> `data/`（222G parquet 因子/面板）与 `outputs/`、`snapshots/`（回测图/快照）为
> 本地再生产物，**不进入 Git**。

---

## 与平台库的关系（谁提供什么）

| 本仓库脚本使用 | 来自哪个库 | 用途 |
|---|---|---|
| `factor_preprocess.transforms.cross_sectional` | [factor_preprocess](../factor_preprocess/README.md) | `cs_winsor` / `cs_rank` / `cs_zscore` 截面预处理 |
| `vectorbt_qs.mvp.data.adapter.set_data_root` | [vectorbt_qs](../vectorbt_qs/README.md) | 回测数据根配置 |
| `vectorbt_qs.mvp.engine.runner.run_backtest / portfolio_report` | [vectorbt_qs](../vectorbt_qs/README.md) | accurate 回测与绩效报告 |
| `vectorbt_qs.mvp.visualization.build_nav_curves` | [vectorbt_qs](../vectorbt_qs/README.md) | 净值曲线 |
| `import lightgbm as lgb` | 第三方 | GPU/CPU 训练 |
| 因子矩阵 / 行情 | [data_access](../data_access/README.md) 与本地 `data/panel` | 特征与标签 |

## 口径（硬性，改前必读 `MISTAKES_AND_LEAKAGE_LESSONS.md`）

- **标签一律 vwap-to-vwap 后复权**。最新权威口径 = COS 官方 `TargetVwapReturnH10`：
  `AdjVwap[t+11]/AdjVwap[t+1] - 1`（T+1 建仓、T+11 平仓）。
  ⚠️ 手册已校正：本地旧标签 `shift(-10)` / `shift(-1)` 三个版本全错；
  2026-08-30 起的最新跑（2000fac / 818fac）已直接用 COS 官方靶。
- 训练窗口 2016 起，**滚动 3 个月重训**，预测下一个 3 个月（walk-forward）。
- 特征：全部入选因子，NaN 保留（LightGBM 原生处理）。
- GPU：`device='gpu'`、`max_bin=63/127`，异常 fallback CPU。机器 32 核 92G：并行 ≤31 workers。
- **防泄漏**：selection 与 train 尾均 purge 10 交易日（=标签视野）；全样本 rank_ic 只做诊断，严禁入模。

## 研究管线全景（scripts/ 约 50 个脚本，按阶段）

```
data/factor_pools（10 池 parquet）→ merge → feature matrix → factor_preprocess 截面变换
→ walk-forward manifest → LightGBM 滚动训练 → predictions（date×asset）
→ 组合优化（top-K QP）→ vectorbt_qs accurate 回测 → outputs/
```

| 阶段 | 脚本 | 职责 |
|---|---|---|
| **数据/复权** | `build_adj_label.py` | 重建后复权 10 日前向标签（vwap-to-vwap） |
| | `build_label_adj.py` | 后复权中性化标签 `fwd_adj_neu`（截面 demean+zscore） |
| | `build_ohlcv_adj.py` / `build_ohlcv_adj_from_table.py` | 后复权 OHLCV 面板（价×Factor、volume 不复权、amount×Factor） |
| | `audit_adj_consistency.py` | 复权口径审计（`\|corr(因子,F)\|>0.3` ⇒ 水平型需重算） |
| | `recompute_lqtp_adj*.py` / `recompute_chunk.py` / `rebuild_fullmarket_dispatch.py` | LQTP 因子后复权重算（分片/多进程） |
| | `rebuild_all_factors_from_new_cos.py` | 用新 COS 全量重建因子落盘（全市场 5460 列） |
| | `lqtp_dsl_eval.py` | 本地 LQTP DSL 求值器（后复权 OHLCV 面板，Open=Open×Factor、volume=Volume/Factor 的平台口径） |
| **因子合并/筛选** | `merge_all_factors.py` / `merge_all_factors_flip.py` | 合并全部本地因子池（10 池）→ rank_ic>0.015 筛选 |
| | `merge_lqtp_fast.py` / `merge_flip_chunk.py` / `merge_flip_combine.py` / `merge_manifest_parts.py` / `merge_recomputed.py` | 分块合并/并行 rank_ic+flip/去重 |
| | `factor_selection.py` | walk-forward 因子选择（每折只在 purged train 窗算 rank_ic，`RANK_IC_THRESHOLD=0.015`、`PURGE_TRADING_DAYS=10`） |
| | `build_walkforward_manifest_flip.py` | 向量化 per-fold rank-ic → `walkforward_selection_flip.json`（可选 gate：min-coverage / require-history-days） |
| | `dedup_selection_adj.py` / `rebuild_selection_adj.py` / `drop_level_based.py` | 完全去重（corr>0.98 聚类）/ 重筛 / 剔除水平型因子 |
| **特征构建** | `build_features*.py`（含 `_all/_flip/_full`）、`build_factor_matrix.py`、`build_dataset.py`、`build_full_features.py` | 特征矩阵构建（82→158→2166 列演进；flip 取负翻转；fill 补全） |
| | `fill_factors.py` / `fill_features.py` | NaN 补全（ffill+bfill+截面中位数） |
| **预处理** | `preprocess_features*.py`（含 `_flip`） | factor_preprocess 截面变换：池级别策略（fm247/fmqa/cog* 已中性化→仅 cs_winsor；delivery/optfac/factmat/lqtp→cs_winsor+cs_rank+cs_zscore） |
| **训练** | `train_lightgbm_gpu.py` / `train_lgb_cpu.py` / `train_final.py` / `train_v2/v3.py` / `train_parallel.py` / `train_opt*.py`（含 `_adj/_flip`） | 滚动 walk-forward 训练；Optuna 超参重搜（RE_OPT_EVERY=2） |
| | `probe_gpu_fit.py` | GPU 拟合探针 |
| **回测/组合** | `backtest_final_vectorbt.py` | 组合 → vectorbt_qs 回测（top-K max-Sharpe QP、换手≤0.30、tc=5bp、Vwap T+1 执行、`execution_lag=1`、accurate 模式） |
| | `portfolio_and_backtest.py` | 老版仓位生成+回测（cvxpy QP，Charnes-Cooper 形式） |
| | `export_curve_outputs.py` / `rankic_report.py` | 结果导出 / rankIC-IC 报告 |
| **全链路** | `run_528full_chain.sh` / `run_818full_chain.sh` / `run_818full_stage23.sh` / `run_818full_train_bt.sh` / `run_final_chain.sh` | 一键全链（merge→features→preprocess→manifest→train→backtest），并行 8 worker，OMP=31 |
| **增量** | `fac818/`（step1_build_inc_features → step2_redundancy_check → step3_prep_inc35 → step4_818_compare） | 818 增量链：35 新因子增量并入 783，严禁全量重算 |

## 组合参数（backtest_final_vectorbt.py 常量）

`topk=30`、`maxw=5%`、`turnover=30%`、`tc=5bp`、`rebal=10 交易日`、`cov_window=60`、`MIN_HOLDINGS=8`。
基准：最新跑用 `000985.CSI TR`（中证全指全收益）。

## 已知结果（outputs/ 实测）

| 跑次 | 配置 | 策略 | 超额 |
|---|---|---|---|
| **run_20260830_2000fac**（最新权威） | 783 因子、NO_DEDUP、label=COS TargetVwapReturnH10、基准 000985.CSI TR | total=502.41% ann=32.90% **sharpe=1.316** mdd=-26.35% | total=314.04% **excess_sharpe=1.880** mdd=-16.55% |
| run_20260830_818fac_v1 | 783+34 增量=818、RE_OPT_EVERY=2 | total=586.27% **sharpe=1.361** mdd=-26.48% | 基准读取路径不同不可比；OOS rank_ic 0.04038（+3.35% vs 783） |
| run_20260830_allfac | IC_THRESH=0、989 特征 | total=373.78% sharpe=1.102 mdd=-34.47% | sharpe=1.472；OOS ic mean 0.0320 |
| outputs/metrics_20260829_782fac.txt | 782 因子、基准 000300.SH | total=571.89% ann=35.22% sharpe=1.354 mdd=-26.62% | total=486.27% sharpe=1.940 |
| outputs/report.md | 737 因子（候选 3177→rank_ic>0.015→737） | total=458.3% sharpe=1.209 mdd=-31.1% | excess=387.1% sharpe=1.455 |
| outputs/rankic_report.md | predictions_adj vs fwd_adj_neu，2020-01-06..2026-08-10 共 1598 截面日 | 平均 IC=0.1118、平均 rankIC=0.0997、IC>0 比例 76.2% | |
| snapshots/final_20260826_1.758 | 早期快照（修复前） | sharpe=1.758 | ⚠️ MISTAKES 手册指出旧 mu/cov 用未来收益虚高 |

## 安装

```bash
cd /home/sunhaiwei/quant_projects/lightgbm_qs
pip install -r requirements.txt
```

GPU 训练需要另行安装与机器 CUDA/OpenCL 驱动匹配的 LightGBM 构建；基础依赖清单
不强制 GPU wheel，脚本保留 CPU fallback。平台库从 monorepo 根按源码安装，禁止从
PyPI 猜测同名内部包。

## 测试

```bash
cd /home/sunhaiwei/quant_projects
PYTHONPATH=. pytest lightgbm_qs/tests/ -q     # test_purge / test_walkforward_leak
```
- `test_purge.py` — P0-B purge/embargo 切分逻辑（标签窗口不跨 cut、purge 精确等于标签视野、逐折 selection 无泄漏）
- `test_walkforward_leak.py` — 植入"未来窥探因子"，断言 walk-forward selection 拒绝它

## 布局

```
scripts/          # 全部研究脚本（build/train/backtest/audit，约 50 个）
tests/            # 泄漏/切分回归测试（2 个）
data/             # 本地 parquet（222G，不入 git）
outputs/          # 回测图/报告（不入 git）
snapshots/        # 训练快照（不入 git）
MISTAKES_AND_LEAKAGE_LESSONS.md   # ⭐防再犯手册（动回测/训练前必读）
```

## 相关仓库（完整平台）

见 `quant_projects` 根目录 `HANDOVER.md` 交接文档 —— 12+ 个库的依赖与接口全景。
