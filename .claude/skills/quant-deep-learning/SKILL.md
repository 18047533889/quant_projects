---
name: quant-deep-learning
description: 深度学习/强化学习因子挖掘与建模的正确方式 — 用 alphaprobe（runner / continuous / fe_bridge factor_engine 后端 + RL 基线 train_ppo / train_gfn）+ lightgbm_qs（train_lightgbm_gpu）+ modeling 库。label 一律 vwap_to_vwap 后复权。当任务涉及"深度学习、强化学习、RL 因子挖掘、GFlowNet、PPO、AlphaProbe、候选因子池、LSTM、GPU 训练"时先读此 skill。
version: 1.0.0
---

# 深度学习 / RL 因子挖掘（alphaprobe + lightgbm_qs GPU）

**前提**：已读 quant-platform-workflow + quant-ml-modeling。本 skill 覆盖我们自研的 AlphaProbe 深度因子挖掘框架 + LightGBM GPU 训练。

## 我们有什么（先认识，别另起炉灶）

- **alphaprobe/**：自研 A 股深度因子挖掘框架。核心是表达式搜索（AlphaGen 系），数据后端默认 **factor_engine**（`data_backend=factor_engine`），label 是 **vwap_to_vwap 后复权**（config `mining.label_price_pair: vwap_to_vwap`，非 close→close）。
- **fe_bridge**：把 AlphaGen `Expression.evaluate` 路由到 factor_engine DSL 求值（`expression_to_dsl` → `FactorEngineDslExpression` → `FactorEngineStockData.evaluate_dsl`），所以挖掘出的因子**天然是 factor_engine 可落值算子**，不会出现"挖出来落不了盘"。
- **RL 基线**（在 `apps/baselines/`）：PPO（MaskablePPO）、GFlowNet（gflownet）、GAN、GP——都是表达式搜索的变体，不是交易策略 RL。

## 路线 1：AlphaProbe 单轮挖掘（runner）

> **重要**：AlphaProbe 主框架是 **LLM 迭代挖掘**（DeepSeek API 驱动），不是纯 RL。RL 基线是研究用的表达式搜索器，见路线 2。
> 环境必配（`runner.py` 会 `load_dotenv(.env)`）：`OPENAI_API_KEY`、`OPENAI_BASE_URL=https://api.deepseek.com`、`OPENAI_MODEL_NAME=deepseek-v4-flash`。LLM 额度耗尽抛 `LLMQuotaExhaustedError`，continuous 模式会写 `STOP_QUOTA` 旗标退出。
> 冷启动依赖 `~/quant_projects/cold_start_library`（本机可能没有），纯 RL/深度因子场景可跳过冷启动直接建 pool。

真实命令行（`src/alphaprobe/runner.py build_train_parser`，30+ 参数，关键默认值）：

```bash
cd /home/sunhaiwei/quant_projects/alphaprobe/ashare/alphaprobe-dev
# 需 torch+cuda 的 python 环境（deploy 脚本用 rdagent4qlib，可用 ALPHAPROBE_PYTHON 覆盖）
PYTHONPATH=src python apps/alphaprobe/run_continuous.py \
  --experiment_config configs/alphaprobe/experiment_ashare_pv_7x24.yaml \
  --cuda 0
```

常用覆盖参数（runner 单轮或 continuous 通用）：
```
--data_backend factor_engine     # factor_engine | qlib
--instruments csi1000            # 股票池
--label_days 20                  # 前瞻收益天数（vwap-to-vwap，20 日）
--ic_threshold 0.006             # 候选入池 IC 门槛
--threshold_ric 0.015            # 投递门槛 rank_ic
--threshold_ricir 0.15
--pool_capacity 2000             # 候选池容量
--search_time 40                 # 搜索迭代数
--cold_start_library <yaml>      # 冷启动主库（backend_v9_core.yaml）
--resume                         # checkpoint 续跑
--campaign_id <id>               # 覆盖 delivery.campaign_id
--seed 0
```

数据/切分/label 都在 `configs/alphaprobe/experiment_ashare_pv_7x24.yaml`：
- `split.train_start: 2016-01-01 / valid 2022-01-01 / test 2024-01-01`（train 段算 IC，valid/test 免重算）
- `mining.label_price_pair: vwap_to_vwap`（硬口径）
- `mining_scope.primary_tables: [StockDailyBar, StockValuationDaily, StockCapitalDaily]`，`forbidden_tables: [StockBalance, StockIncome, StockCashFlow]`
- `delivery`：投递到 `~/quant_projects/data/factor_pools/candidate_pool`（candidate_pool 落盘，供 merge_all_factors 合并）

**7×24 连续模式**（`deploy/run_continuous.sh` + `apps/alphaprobe/run_continuous.py`）：每轮新 campaign_id → 冷启动 100 条 → 40 轮迭代 → disk.v1 投递 candidate_pool，支持 `--sleep_seconds` / `--max_rounds` / `--max_hours` / `--ignore_stop_quota`，崩溃后同 campaign resume。

## 路线 2：RL 基线（研究用）

### PPO 表达式搜索（`apps/baselines/train_ppo.py`）

> **坑**：baselines 默认走 qlib 数据（`StockData`），`qlib_path` 是占位符 `PATH/TO/.qlib/qlib_data/cn_data`，必须设 `QLIB_PATH_CN` 或用 factor_engine 数据源替换；`--cuda` 相关：train_ppo 硬编码 `cuda:1`。

```bash
cd /home/sunhaiwei/quant_projects/alphaprobe/ashare/alphaprobe-dev
PYTHONPATH=src python apps/baselines/train_ppo.py \
  --seed 0 --instruments csi1000 --pool 50 --steps 100_000
```

- 核心装配（import 正确，��跑）：`MaskablePPO`（sb3_contrib）`'MlpPolicy'` + `LSTMSharedNet` 特征提取器（`n_layers=2, d_model=128, dropout=0.1`）+ `AlphaEnv(pool, device, print_expr)` + `AlphaPool(capacity=50, stock_data=data_train, target=target, ic_lower_bound=None)`。
- label = `Ref(close, -20)/close - 1`（baselines 用 close；**主框架 alphaprobe 用 vwap**——见"数据口径铁律"）。
- checkpoint 存 `{prefix}_{timestamp}/{steps}_steps` + `{steps}_pool.json`；logger 记录 `pool/size`、`test/ic`、`test/rank_ic`。

### GFlowNet 表达式搜索（`apps/baselines/train_gfn.py`）

> **坑：`train_gfn.py` 的 import 是坏的**��照抄会 `ModuleNotFoundError`）。仓库里没有 `src/alphagen`、`src/alpha_gfn` 目录。正确 import 是：
> `from baselines.alpha_gfn.config import *`、`from baselines.alpha_gfn.env.core import GFNEnvCore`、`from baselines.alpha_gfn.modules import SequenceEncoder`、`from baselines.alpha_gfn.alpha_pool import AlphaPoolGFN`、`from baselines.alpha_gfn.gflownet import EntropyTBGFlowNet`、`from shared.alphagen_qlib.stock_data import StockData`。要跑得先修脚本 import，再配 qlib path。

```bash
cd /home/sunhaiwei/quant_projects/alphaprobe/ashare/alphaprobe-dev
PYTHONPATH=src python apps/baselines/train_gfn.py \
  --seed 0 --instrument csi1000 --pool_capacity 50 \
  --encoder_type gnn --entropy_coef 0.01 --ssl_weight 1.0 --nov_weight 0.3 \
  --n_episodes 10000 --update_freq 128 --log_freq 1000
```

- `gfn` 库的 `EntropyTBGFlowNet` / `TBGFlowNet` + `DiscretePolicyEstimator`；编码器 `--encoder_type transformer|lstm|gnn`（`SequenceEncoder`，HIDDEN_DIM=128，LR=1e-4）。
- 奖励 = `ic_reward + ssl_weight*ssl_reward + nov_weight*nov_reward`（SSL 自监督 + novelty，权重 `--ssl_weight`/`--nov_weight` 线性衰减到 `--final_weight_ratio`）。
- 池用 `AlphaPoolGFN`（`try_new_expr_with_ssl(expr, embedding)`）。

### FQF/IQN/QRDQN（分位数 DQN 系，**无现成训练入口**）

`src/baselines/fqf_iqn_qrdqn/` 有 agent/model/memory 但没有训练脚本（configs/baselines/qcm/*.yaml 无 loader）。要跑自己写驱动：

```python
from baselines.fqf_iqn_qrdqn.agent import FQFAgent, IQNAgent, QRDQNAgent
from baselines.fqf_iqn_qrdqn.model.alpha_fqf import FQF   # 用 alpha_ 前缀（LSTM 版）
# 构造 AlphaPool → AlphaEnv(pool, device, print_expr) → FQFAgent(env, data_valid, data_test, target, log_dir, num_steps=...) → agent.run()
```

- **用 `model/alpha_*.py`**（LSTM 版：`LSTMBase(n_actions, embedding_dim=128, n_layers=2, dropout=0.1)`），`model/fqf.py`/`iqn.py`/`qrdqn.py` 是 Atari CNN 遗留，别用。
- `FQFAgent` 分位学习率 `quantile_lr=5e-5`、fraction `2.5e-9`；QCM 变体 `FQCMAgent` 带 `MeanNetwork` 高阶矩。

### 其他基线

- `apps/baselines/train_GP.py`（gplearn 遗传规划）、`train_AFF.py`（GAN）、`combine_AFF.py` / `run_adaptive_combination.py`（自适应组合）。

## 路线 3：LightGBM GPU 训练（深度特征 → 模型）

```bash
cd /home/sunhaiwei/quant_projects
# 1) 候选因子合并（alphaprobe candidate_pool + 库内因子 → 特征矩阵）
.venv/bin/python lightgbm_qs/scripts/merge_all_factors.py
.venv/bin/python lightgbm_qs/scripts/build_full_features.py    # corr>0.98 完全去重
# 2) GPU 训练（自带 walk-forward）
.venv/bin/python lightgbm_qs/scripts/train_lightgbm_gpu.py
# 3) 最终训练 + 调参
.venv/bin/python lightgbm_qs/scripts/train_final.py
.venv/bin/python lightgbm_qs/scripts/train_opt.py
```

机器约束：L20 GPU（45GB 显存），CPU ≤31 核、单实例 ≤56G 内存（.bashrc guardrail）；`probe_gpu_fit.py` 可先探测 batch 能不能塞下。alphaprobe pyproject 固定 `torch 2.4.0+cu121`（Python 3.11，quant_projects/.venv 无 torch，需独立 conda 环境）。

## 数据口径铁律（深度学习最容易踩）

- **label 一律 vwap-to-vwap 后复权**：alphaprobe 用 `mining.label_price_pair: vwap_to_vwap`；lightgbm_qs 用 `vwap.shift(-10)/vwap - 1`（后复权 `vwap_trad_adj.parquet`）。**禁止 close 收益 / 未复权**。
- **⚠ label 日偏移陷阱**：主框架 runner 的 target = `Ref(vwap, -label_days)/vwap - 1` = `AdjVwap[t+20]/AdjVwap[t] - 1`（t..t+20）；而存储 label 列 `TargetVwapReturnH20` = `AdjVwap[t+21]/AdjVwap[t+1] - 1`（**相对 t 偏移一天**，H01/H05/H10 同理）。**别把存储 label 列当 `Ref(vwap,-20)/vwap-1` 直接用**；两者差一天。
- 训练/评估特征矩阵行列 = (date, asset)，MultiIndex 排序一致性。
- 挖掘出的表达式**必须能落盘**：走 factor_engine DSL（fe_bridge 保证），投递 candidate_pool → `merge_all_factors.py` → 正常评估流程。**不许手写 pandas 实现挖掘出来的算子**。
- 数据源（factor_engine 后端）：`default_ashare_pv_data_source_config` 读 `ashare_stock_daily_adj`（COS 镜像 `~/cos_data/StockDailyBarAdj`），字段映射 `close→AdjClose, vwap→AdjVwap, volume→Volume/Factor(复权量), ret→Return(bp)`；裸 `close/open/high/low/vwap` 自动落后复权列。生产走 data_access 路径；本地 smoke 若 `root` 不存在需覆写 `cfg['root']`。

## 产出衔接

```
AlphaProbe 挖掘 → candidate_pool → merge_all_factors → build_full_features
                                                          ↓
quant-ml-modeling（LightGBM / modeling） → quant-ml-optimization（组合） → quant-reporting
```

## 禁止

- ❌ 手写 walk-forward / train_test_split（modeling.split）
- ❌ 用未复权或 close 收益做 label
- ❌ 在深度挖掘里手写 pandas 实现 factor_engine 算子（fe_bridge 已有，走它）
- ❌ 把 RL 基线当"交易策略 RL"——它们是表达式搜索器，产出因子不是直接下单
- ❌ 绕过 alphaprobe 另搭���套因子挖掘（我们已有体系）

## 易踩坑速查（写代码前先看）

1. **`train_gfn.py` import 坏**（`src/alphagen`/`src/alpha_gfn` 不存在）→ 改 `baselines.alpha_gfn.*` / `shared.alphagen.*`，否则 `ModuleNotFoundError`。
2. **fqf_iqn_qrdqn 无训练入口** → 自己写驱动循环（`AlphaPool → AlphaEnv → FQFAgent.run()`）。
3. **baselines 的 qlib_path 是占位符** → 必须设 `QLIB_PATH_CN` 或换 factor_engine 数据。
4. **`enable_factor_engine_evaluation()` 必须先调用**，否则 `Expression.evaluate(FactorEngineStockData)` 走错逻辑。
5. **LLM 挖掘必须配 DeepSeek API**（`.env`：OPENAI_API_KEY/BASE_URL/MODEL_NAME），额度耗尽抛 `LLMQuotaExhaustedError`。
6. **存储 label 列与 runner target 差一天**（见口径铁律），别混用。
7. **数据根路径**：`default_ashare_pv_data_source_config` 默认 root 本机可能不存在，本地 smoke 覆写 `cfg['root']='~/cos_data/StockDailyBarAdj'`。
