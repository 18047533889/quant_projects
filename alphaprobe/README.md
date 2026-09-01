# alphaprobe — 深度/RL 因子挖掘框架

基于 torch 的深度/强化学习因子挖掘框架（A 股）。用知识池（knowledge pool）在冷启动
因子库上迭代挖掘，候选表达式经 **factor_engine（DSL）** 评估，投递 `candidate_pool`
供下游入库。

**定位:** 企业级 A 股横截面日频多因子量化项目的**可选上游挖掘引擎** —— 7×24 自动
发现新因子，喂给 factor_evaluator → factor_assets 的候选池。

**版本:** 0.1.0 ｜ **仓库:** https://github.com/HKUST-QUANT-SOCIETY/alphaprobe (私有)
**文档:** `docs/`（任务书）· `ashare/alphaprobe-dev/ALPHAPROBE_REFACTOR_REPORT.md`（基础层重构报告）

> ⚠️ **状态：代码完整，尚未在产线跑通。** 阻塞点 = 冷启动因子库的 **yaml 产物缺失**：
> `cold_start_library`（外部包）存在且 V9 JSON 审计库（6137 条 default core）已就绪，
> 但 `data/ashare/backend_v9_core.yaml`（README/配置的 `COLD_START_LIBRARY_SRC`）**未生成**，
> runner 在 sampling 阶段即失败。LLM 额度已在 `.env` 配置（DeepSeek）。

---

## 布局（ashare/alphaprobe-dev/）

```
ashare/alphaprobe-dev/
├── src/alphaprobe/          # 主挖掘链
│   ├── runner.py            # 单轮挖掘入口（run_mining_campaign）
│   ├── continuous.py        # 7×24 连续挖掘（run_continuous / run_continuous_v2）
│   ├── cold_start/          # 冷启动库加载（external cold_start_library 包）
│   ├── trainer/             # pool.py(AlphaKnowledgePool) / trainer.py(AlphaKnowledgeTrainer) / checkpoint.py
│   ├── delivery/            # config.py(ExperimentConfig) / exporter.py(DiskV1DeliveryExporter)
│   ├── fe_bridge/           # bootstrap / dsl_convert / dsl_expression / expr_parse /
│   │                        # knowledge_graph / metrics / stock_data / paths / allowlist
│   ├── identity.py          # FactorIdentity factory（§75.1）
│   ├── dedup.py             # 四层去重（canonical/sign-invariant/param family/256bit SimHash）+ GlobalSeenIndex
│   ├── contracts.py         # §5/§8/§26/§75 共享契约层
│   ├── memory/              # GlobalMemoryStore(SQLite) / retriever / seed_ingestion
│   ├── seen/                # fingerprint(LSH) / dedup_service / nearest / registry / subtree 等
│   ├── search/              # arms(RefineArm/BranchArm) / orchestrator / structured_llm
│   ├── fitness/             # MetricCalibrator / funnel(L0-L4) / D10 cliff / isotonic / HardGates
│   ├── survival/            # dna / profile / regime
│   ├── research_protocol/   # orientation / survival（L0-L4 sealed test 零读取）
│   ├── export/gate.py       # §58 ExportDecision
│   ├── pool/                # admission /（ActivePool+Pareto+QD）
│   ├── integration/         # factor_engine_adapter / evaluator_client / refinement_client
│   ├── evalcache.py / lineage/ / checkpoint/ / observability/
├── src/baselines/           # gan / fqf_iqn_qrdqn / alpha_gfn / gplearn
├── src/shared/              # alphagen / alphagen_qlib / alphagen_generic / data_collection / utils
├── apps/alphaprobe/run_continuous.py + apps/baselines/
├── configs/alphaprobe/      # experiment_ashare_pv_7x24.yaml（主配置）/ factor_engine_smoke.yaml / backtest_ashare.yaml
├── configs/prompts/         # 6 个 LLM prompt txt
├── configs/baselines/qcm/   # fqf/iqn/qrdqn.yaml；configs/cold_start/unique_templates.txt
├── deploy/                  # run_continuous.sh + alphaprobe-continuous.service（systemd）
├── scripts/                 # check_fe_prompts.py + delivery/export.py
└── tests/                   # 9 个测试文件（约 40 个测试）
```

## 单轮挖掘流程（runner.run_mining_campaign）

1. **配置** — `ExperimentConfig.from_yaml` 读 delivery/data/mining 块；`.env` 注入 LLM
   （`OPENAI_MODEL_NAME=deepseek-v4-flash`、`OPENAI_BASE_URL=https://api.deepseek.com`、`OPENAI_PROVIDER=deepseek`）。
2. **切分** — train 2016-01-01..2021-12-31 / valid 2022-01-01..2023-12-31 /
   test 2024-01-01..2026-07-31；**test 封存**（构造 data_test 但不喂 trainer，只写投递 metadata）。
3. **数据** — 默认 `FactorEngineStockData`（factor_engine parquet 数据源，fe_profile=ashare_pv_valuation，
   价量 + pe/pb/turnover/market_cap）；备选 qlib StockData（sp500/csi300）；`--data_backend factor_engine|qlib`。
4. **标签** — `target = Ref(VWAP, -label_days)/VWAP - 1`（**vwap_to_vwap**，label_days=20）。
5. **冷启动** — `resolve_cold_start` → cold_start_library 包读 yaml，multi-library mix
   （pv:1.0、alpha101/191/158:0，默认比例 0.7/0.1/0.1/0.1），sample 100 条，structural diversity。
6. **知识池** — `AlphaKnowledgePool`（capacity 2000、top_k 15、ic_mut_threshold 0.9、
   depth_decay 0.05、times_decay 0.1、语义相似度 SentenceTransformer、res_corr、edit_distance）。
7. **训练** — `AlphaKnowledgeTrainer.train`，search_time=40 轮 LLM 迭代；每轮 pool.search()
   选 top-k 节点 → LLM 生成 generate_num=5 变体 → 知识图谱去重/校验 → `pool.try_new_expr`
   （准入：`|ic|>0.006` 且 `|icir|>父节点 icir`，或低互相关变体）；每轮 checkpoint
   （v1 JSON + v2 轻量，崩溃可 resume）。
8. **评估/投递** — `DiskV1DeliveryExporter.export_from_pool` → `evaluate_all_period_metrics`
   （train/valid/test 三窗重算 IC/ICIR/rank_IC，走 FE calc）→ `candidate_pool/{campaign_id}/`。
   ic_export_threshold=0.006。
9. **记忆** — `GlobalMemoryStore`（SQLite）；冷启动 seed 摄入 + 每轮 pool 快照 upsert
   （signal id 去重，rediscovery 合并）。

## 连续挖掘（continuous.py + round_manager）

- 每轮：新 campaign id（`alphaprobe_ashare_pv_YYYYMMDDHHMM`）→ 新冷启动 → `run_mining_campaign`
  → 投递 → pool 快照 upsert → 下一轮。
- **崩溃衔接**：`data/logs/continuous/active_round.json` 记录 status；检测到未完成轮 →
  同一 campaign + checkpoint resume（`args.resume=True`）。
- **额度管控**：`LLMQuotaExhaustedError` → 写 `STOP_QUOTA` 文件并干净退出（exit 0 不自动重启）；
  充值后删旗标再启。
- 参数：`max_hours=24`、`max_rounds=null`、sleep 60s、error_sleep 300s、`llm_quota_max_consecutive=1`。
- **run_continuous_v2**（§56 RoundManager）：load memory → calibrator freeze → campaign 回调 →
  memory update → lineage patience early stop。
- **部署**：`deploy/run_continuous.sh` + systemd service（`User=hsunbj`、`Restart=on-failure`、
  RestartSec=30；额度耗尽/满 24h 干净退出不重启）。

## FE 桥（fe_bridge/）— 表达式→DSL 唯一真值

| 模块 | 功能 |
|---|---|
| `bootstrap.enable_factor_engine_evaluation()` | monkey-patch `Expression.evaluate` → 对 `FactorEngineStockData` 走 FE 评估 |
| `dsl_convert.expression_to_dsl` | AlphaGen 表达式树 → FE DSL（Feature→open/close/high/low/volume/vwap；一元/滚动/双序列算子映射；Div 加 `+1e-9` 保护） |
| `expr_parse.parse_mining_expression` | 只接受 FE DSL（拒绝 AlphaGen 旧语法 `$close`/`TsMean`/`Div`） |
| `dsl_expression.FactorEngineDslExpression` | DSL 字符串包装为 Expression；`extract_dsl_operator_names` 用 FE `build_dsl_allowlist` 校验算子 |
| `knowledge_graph.patch_knowledge_graph_for_factor_engine_dsl` | 知识图谱接受 FE DSL 冷启动因子 |
| `metrics.evaluate_all_period_metrics` | train/valid/test 三窗 IC/ICIR/rank_IC |
| `stock_data.FactorEngineStockData` | FE parquet 数据源；`evaluate_many` 批执行（`engine.run_many(enable_cse=True)`，共享子树只算一次，失败降级单条）；冷启动 `value_cache` 免重算 |
| `identity.py` | **FactorIdentity factory**：factor_id=canonical_ast_hash[:12]、canonical_formula、signal_equivalence_id（f/-f/0-f/(-1)*f 同 id）、parameter_family_id（窗口参数打码）、orientation |
| `paths` / `allowlist` | sys.path 引导 factor_engine / 与 FE 白名单对齐的算子集 |

另有 `integration/factor_engine_adapter.py`（validate/canonicalize/inspect/run_many，FE 不可用时
优雅降级 dedup）与 `evaluator_client.py`（legacy_compat adapter）。

## 去重（dedup.py — 四层）

1. **canonical** — 语法规范化后的 AST hash
2. **sign-invariant** — f / -f / 0-f / (-1)*f 视为同一信号（signal_equivalence_id）
3. **parameter family** — 窗口参数打码后的族（parameter_family_id）
4. **256-bit SimHash** — 大规模近似近邻去重（`topk_fingerprint_neighbors`，堆化实现，O(k) 内存）

外加 `GlobalSeenIndex`（跨轮持久去重）。

## Delivery（disk.v1）

`DiskV1DeliveryExporter`：输出 `candidate_pool/{campaign_id}/` 下 `config.json`
（schema_version=disk.v1、campaign_id、generator_name/version、market、universe_id=A_SHARE_ALL_A_EX_ST、
domain_root=price_volume、frequency_bucket=daily、**operator_policy=lqtp_pv_daily**、
mining_scope 含 primary/auxiliary/**forbidden 表（StockBalance/StockIncome/StockCashFlow）**、
data_source local+cos、mining_config、mining_run_stats 含 llm_model/token 等）+ 每个候选
`{candidate_id}/manifest.json`（candidate_id=generator_时间戳_hash8、formula、metrics、description、born_timestamp）。
`_candidate_hash = sha256(formula|universe_id|frequency_bucket)[:8]` 同 hash 去重；train_ic < 0.006 不导出。
投递规范遵循 `factor_engine/docs/miner_delivery_spec.md`。

## Baselines（src/baselines/，基于 shared/alphagen + qlib）

| 目录 | 是什么 |
|---|---|
| `gan/` | **AlphaGAN**：NetG 生成器（LSTM/CNN/DCGAN/ResBlock）+ NetM masker + NetP predictor；loss=simi/pred/potential/entropy |
| `fqf_iqn_qrdqn/` | **分位数 Q-learning 三兄弟**：FQF（Fractional Quantile）、IQN（Implicit Quantile）、QRDQN（Quantile Regression DQN）；分布 RL + Atari 风格 env + PER 记忆 |
| `alpha_gfn/` | **GFlowNet 因子挖掘**：EntropyTBGFlowNet + GFNEnvCore + GNN/Transformer 序列编码（torchgfn） |
| `gplearn/` | **遗传编程符号回归**：sklearn 风格 SymbolicRegressor（protected div/sqrt/log/inverse/sigmoid） |

`src/shared/`：alphagen（表达式树/知识图谱/AlphaPool/RL env + LSTM/Transformer policy）、
alphagen_qlib（qlib StockData/Calculator/Strategy）、alphagen_generic（通用算子）、
data_collection（fetch_baostock / qlib_dump_bin）、utils（llm 带限次重试+额度识别）。

## 配置要点（experiment_ashare_pv_7x24.yaml）

- `data`：source=ashare、**fe_profile=ashare_pv_valuation**（价量+pe/pb/turnover/market_cap）
- `split`：train 2016-2021 / valid 2022-2023 / test 2024-2026-07（test 封存）
- `mining`：**label_days=20、label_price_pair=vwap_to_vwap**、search_time=40、pool_capacity=2000、
  ic_threshold=0.006、checkpoint enabled+resume、cold_start_sample_size=100、
  cold_start_library=~/quant_projects/cold_start_library/data/ashare/backend_v9_core.yaml（**当前缺失**）
- `continuous`：enabled、sleep_seconds=30、max_rounds=null、max_hours=24、llm_quota_max_consecutive=1

## 环境

Python 3.11、torch 2.4.0+cu121、gymnasium、stable-baselines3、sb3-contrib、torchgfn、
torch-geometric、pyqlib、transformers 4.56、openai 1.55.1、sentence-transformers 3.3.1。
LLM key 在 `.env`（禁止提交）。

## 测试

```bash
cd ashare/alphaprobe-dev && PYTHONPATH=src pytest tests/ -q    # 9 个文件，约 40 个测试
```
覆盖：dedup_client / fe_adapter / funnel_admission / leakage_gates / memory /
round_checkpoint / search_arms / survival / taskbook_units。

## 相关仓库

- **factor_engine** — DSL 评估后端（fe_bridge）
- **data_access** — 数据后端
- **factor_assets / factor_optimizer** — 下游入库 / 寻优
