# alphaprobe — 深度/RL 因子挖掘框架

基于 torch 的深度/强化学习因子挖掘框架（A 股）。用知识池（knowledge pool）在
冷启动因子库上迭代挖掘，候选表达式经 **factor_engine（DSL）** 评估，投递
`candidate_pool` 供下游入库。

**版本:** 0.1.0 ｜ **仓库:** https://github.com/HKUST-QUANT-SOCIETY/alphaprobe (私有)

> ⚠️ **状态：代码完整，尚未在产线跑通。** 已知阻塞 = 缺少 cold-start 因子库
> （`COLD_START_LIBRARY_SRC`）。LLM 额度需在 `.env` 配置（DeepSeek）。

## 布局

```
ashare/alphaprobe-dev/
├── src/alphaprobe/
│   ├── runner.py            # 单轮挖掘：冷启动 → 训练 → 投递
│   ├── continuous.py        # 7×24 连续挖掘（campaign、resume、额度管控）
│   ├── cold_start/          # 冷启动库加载（ALPHA101/158/191、V9+expand）
│   ├── trainer/             # AlphaKnowledgePool、AlphaKnowledgeTrainer、checkpoint
│   ├── delivery/            # DiskV1DeliveryExporter → candidate_pool（miner_delivery_spec）
│   └── fe_bridge/           # 表达式 → FE DSL、FE 评估、knowledge-graph patch
├── src/baselines/           # GAN、FQF/IQN/QRDQN、AlphaGFN、GPlearn
├── src/shared/              # alphagen / alphagen_qlib / utils / data_collection
├── apps/alphaprobe/run_continuous.py
├── configs/alphaprobe/      # experiment_ashare_pv_7x24.yaml（A 股 7×24）
├── configs/prompts/         # LLM prompts（generation / validity / separation / FE-DSL）
├── deploy/                  # run_continuous.sh + alphaprobe-continuous.service（systemd）
└── scripts/                 # FE prompt 检查、投递导出
```

## 入口

```bash
# 单轮
PYTHONPATH=src python -m alphaprobe.runner --experiment_config configs/alphaprobe/experiment_ashare_pv_7x24.yaml
# 连续（7×24）
bash deploy/run_continuous.sh
```

- `enable_factor_engine_evaluation()` — 把知识图评估切换到 factor_engine（DSL）。
- `evaluate_via_factor_engine(expr, ...)` — 用 FE + `calc_ic_ir` 算指标。
- 投递格式遵循 `factor_engine/docs/miner_delivery_spec.md`
  （campaign_id、formula、universe、domain、operator_policy、data_source local/cos）。

## 数据与标签

- 数据：A 股经 `data_access`（或本地 `lqtp_data`），`fe_profile: ashare_pv_valuation`
  （价量 + pe/pb/turnover/market_cap）。
- 标签：**vwap-to-vwap**（`mining.label_price_pair: vwap_to_vwap`，`label_days: 20`）。
  切分：train 2016–2021、valid 2022–2023、test 2024–2026-07。
- 算子策略：`lqtp_pv_daily`；禁止基本面表（balance/income/cashflow）。

## 依赖与接口（谁 import 谁）

- **依赖**：`factor_engine`（`api.dsl_parser` / `api.mining_integration` / `api.operator_registry` /
  `backend.factory` / `runtime.engine` / `storage.factory`）、`data_access`（行情）、torch（CUDA）。
- **被谁调用**：无（独立挖掘引擎）。投递给 `factor_assets`/`factor_optimizer` 的下游管线。

## 环境

Python 3.11、torch（CUDA）、qlib、gymnasium、stable-baselines3、torchgfn、
transformers、openai（DeepSeek）。见 `pyproject.toml`。LLM key 在 `.env`（禁止提交）。

## 相关仓库

- **factor_engine** — DSL 评估后端（fe_bridge）
- **data_access** — 数据后端
- **factor_assets / factor_optimizer** — 下游入库 / 寻优
