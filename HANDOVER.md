# Quant Platform Handover — 量化因子平台交接文档

> **给接手人**：本文档说明这套量化因子研究与生产平台的**全貌**：12 个仓库各自干什么、
> 数据怎么流、依赖与接口怎么连、口径与红线、以及新手 30 分钟上手路径。
>
> **仓库归属**：全部为**私有**仓库，位于 GitHub 组织 **HKUST-QUANT-SOCIETY**；
> 个人镜像主仓 `18047533889/quant_projects` 是 monorepo（所有代码合在一个仓库）。
> 每个独立库都有各自的 `README.md`（英文，含 Quick Start 与 API）。
>
> **更新**：2026-09-01 ｜ **作者**：Sun Haiwei（孙海崴）

---

## 0. TL;DR（60 秒读完）

```
data_access (行情/PIT 数据层)
      │  读 (get_store)
      ▼
factor_engine (DSL 因子计算引擎) ──materialize──► 因子湖 (factor_lake)
      │  读因子值
      ▼
quant_evaluator (评估：IC/分层/回撤等 60+ metric)
      │  EvaluationBundle (证据)
      ▼
factor_assets (因子资产注册表：身份/去重/聚类/生命周期/入库)
      ▲
      │  注册/治理
factor_optimizer (因子寻优：变异语法 + 搜索 + 验收)
      │  SearchRunner
      ▼
factor_preprocess (模型输入预处理：中性化/标准化/清洗)
      ▼
modeling (ML 建模：walk-forward/泄漏防护/artifact)
      ▼
vectorbt_qs (组合回测，accurate/fast)  ◄── riskfolio_qs (组合优化/风控，生成目标权重)
      │
      ▼
quant_platform (平台合同层 DTO + Web)  ◄── platform_web (React 前端)
alphaprobe (深度/RL 因子挖掘，可选上游)
```

一句话各库职责：

| 库 | 职责 | 版本 |
|---|---|---|
| **data_access** | 全团队统一数据读写层（73+ 数据集、PIT、COS/ClickHouse、读因子） | 0.10.x |
| **factor_engine** | DSL 因子计算引擎：编译 → 多后端执行 → 落盘因子湖（1737 个算子） | 0.3.x |
| **quant_evaluator** | 因子评估器：注册 60+ 指标（rank_ic/ic_ir/quantile_spread/max_drawdown/…），产出 EvidenceBundle | 0.0.1a1 |
| **factor_assets** | 因子资产库：身份/注册/去重(seen)/相似度/聚类(Leiden)/生命周期/入库治理 | 0.1.0 |
| **factor_optimizer** | 因子寻优：变异语法、搜索编排、desirability/Pareto 验收、sealed test | 0.1.0 |
| **factor_preprocess** | 模型输入准备：中性化(OLS/岭/套索)、标准化、滚动/平滑、regime 检测、FeatureBundle | 0.1.0 |
| **modeling** | ML 建模：walk-forward 日期权威切分、purge/embargo、泄漏防护、模型 artifact 生命周期 | 0.1.0 |
| **vectorbt_qs** | A 股/美股组合回测（vendor vectorbt 1.1.0 + 公司行动）：accurate（真实股数）/ fast（无摩擦） | 0.4.0 |
| **riskfolio_qs** | cvxpy 组合优化：Barra 预计算/历史 mean-variance、指增 TE 约束、中性约束、目标权重输出 | 0.3.0 |
| **quant_platform** | 平台集成层：纯 stdlib DTO 合同（23 模块）、auth/RBAC/outbox/worker 骨架、候选编排 | 0.1.0 |
| **platform_web** | React 前端（20 页面），数据只读后端 API 不做计算 | — |
| **alphaprobe** | 深度/RL 因子挖掘（torch）：冷启动库 + 7×24 连续挖掘 → FE 评估 → candidate_pool 投递 | 0.1.0 |

---

## 1. 最重要的口径（红线，改前必读）

1. **收益口径 = vwap-to-vwap**。所有因子评估/回测/建模的标签一律是
   `VWAP_{t+H}/VWAP_t - 1`（后复权 vwap，`shift(-1)` 得到下一交易日 vwap 收益）。
   **禁止**用 close-to-close / open-to-open。QE 的 `LabelBundle` 默认
   `price_convention="vwap_to_vwap"`；modeling 的 `LabelContract` 默认
   `return_basis="vwap_to_vwap"`。**这是全局硬性口径，换人不得改。**
2. **决策时钟**：日频 = 收盘(15:00)后信号可用 → **下一交易日 VWAP 成交** →
   下一交易日收盘完成标签。即"今天算因子、明天 vwap 下单、后天 vwap 再调仓"
   的收益就是 `明天收盘→后天收盘的 vwap 收益`。见 `modeling.contracts.DecisionClock`。
3. **禁止手写 RankIC / 手写回测 / 手写 walk-forward**。评估必须走
   `quant_evaluator`，回测必须走 `vectorbt_qs`，时序切分必须走 `modeling.split` +
   `leakage_guard`。
4. **PIT（Point-in-Time）**：读数据一律 `data_access`，禁止裸 `pd.read_parquet`；
   需要历史快照语义时用 PIT/ASOF，禁止未来函数。
5. **数据复权**：因子计算用后复权；`data_access` 的 `adj_factor`/`adjusted_price_backward`
   是唯一复权因子来源。预测目标 = 前复权 VWAP 10 日收益（已在验证脚本中固化）。

---

## 2. 仓库间依赖与接口（怎么连）

### 2.1 data_access → 所有人（数据地基）

- **接口**：`from data_access import get_store`；`store.read_frame/read/read_factors/
  read_joined/sql/compute_and_write/...`。`read_uri` 支持 parquet/csv/feather/arrow。
- **数据集**：`config/datasets.yaml`（73 个），逻辑字段在 `config/semantic_fields.yaml`
  （56 个：close/open/high/low/vwap/return_bp/adj_factor/...）。
- **PIT 契约**：`contract/runtime_contract.py` 的 `PITContract`；事件表读 `read_cos_events_asof`。
- **HTTP 服务**：`data-access-server`（默认 8765），`DataAccessClient`。
- **COS**：mirror / remote / auto 三模式；`COS_SECRET_ID/KEY` 环境变量。
- **谁依赖**：factor_engine（数据源）、quant_evaluator（universe/日历 provider）、
  factor_preprocess（_data_access_impl）、riskfolio_qs（benchmark/tradable/Return）、
  vectorbt_qs（mvp/data/adapter）、alphaprobe（FactorEngineStockData 下层）、quant_platform（storage adapter）。

### 2.2 factor_engine

- **接口**：`from factor_engine.api import col, rank, ts_mean` + `Factor(name, expr)` +
  `FactorEngine(backend, data_source)`；或 `FactorEngine.run_from_config(yaml)`。
  `data_source.type: data_access`（推荐）。后端 `auto`（SQL 下推→Polars→Pandas）。
- **算子**：`cleaned_operators/` 单一实现源；daily surface 1238 / 全量 canonical 1737。
- **落盘**：`engine.materialize(factor)` → 因子湖；profile 见 `examples/profiles/prod.yaml`。
- **HTTP 服务**：`factor-engine-serve`（默认 8088）：`/factor-engine/operators`,
  `/validate-spec`, `/jobs/compute`, `/jobs/materialize`, `/jobs/{id}/artifacts`。
- **被谁调用**：alphaprobe（`fe_bridge` 把表达式转 DSL 评估）、quant_evaluator（FactorBatchProvider）、
  factor_optimizer（canonical hash/validate_mutation/complexity）、factor_assets（身份 hash）。

### 2.3 quant_evaluator

- **接口**：`EvaluationRequest(batch_or_factor_ids, label_bundle, metric_ids=...)` →
  `EvaluationBundle`。`LabelBundle(values, horizon, price_convention="vwap_to_vwap")`。
  适配器：`adapters/factor_engine.py`（把 FE 结果转 FactorBatch）、`adapters/data_access.py`
  （universe/日历 provider）。
- **指标**：注册 60+：`rank_ic` `pearson_ic` `ic_ir` `hac_tstat` `quantile_spread`
  `max_drawdown` `cvar_95/99` `turnover*` `ic_autocorr_lag1` `half_life` `block_bootstrap_ci` ...；
  权威清单 `docs/METRIC_REGISTRY_COVERAGE.csv`。metric 注册表在 `registry/metrics.py`
  （`MetricRegistry`，seal 后不可再注册）。
- **被谁调用**：factor_optimizer（评估证据）、factor_assets（证据 ref）、
  quant_platform（EvidenceStatus）、modeling（评估 IC）。

### 2.4 factor_assets

- **接口**：`create_repository()` → `AssetRepository`（注册/查询/生命周期）；
  `FactorIdentityProvider.get_full_identity`；`create_factor_id(canonical_hash)`。
  去重：`SeenIndex`（exact + persistent sqlite）；聚类：`LeidenClustering`（igraph/leidenalg，强制认证图）；
  相似度：`QEPairwiseSimilarity` / ANN（faiss/annoy）；入库：`PromotionGate`。
- **契约**：`FactorAsset`/`FactorAdmissionArtifact`/`FactorSetArtifact`/
  `TreatmentSelectionArtifact`/`SimilarityArtifact`。生命周期 `LifecycleState`（6 态）。
- **被谁调用**：quant_platform（factor library DTO 映射）、factor_optimizer（入库验收）。

### 2.5 factor_optimizer

- **接口**：`SearchRunner(config, ...)`；`select_winner`/`UncertaintyAwareWinnerSelector`
  （desirability + pareto）。`grammar/` 变异语法（mutation_spec/registry/validation）。
  `contracts/`：trial/trial_ledger/search_budget/treatment_integrity。`llm/` 变异提案 prompt。
- **依赖适配**：`adapters/factor_engine.py`（canonical_hash/validate_mutation/estimate_complexity/
  get_operator_metadata）、`adapters/quant_evaluator.py`（评估证据）。
- **被谁调用**：platform 候选流水线（candidate→treatment→search）。

### 2.6 factor_preprocess

- **接口**：`get_default_policy_registry()` / `get_default_registry()`；
  `PolicyPreset`（如 `production_full`/`research_full`/`cs_only`）；`NeutralizationSpec`
  （OLS/ridge/lasso/elastic_net，行业/市值中性）；`FeatureBundle`/`FittedState`/
  `TreatmentRecipe`/`FactorProfileArtifact`。
- **被谁调用**：modeling（把因子变成模型输入）、platform（feature_set DTO 映射）。

### 2.7 modeling

- **接口**：`PanelDataset`/`FeatureSchema`；`train_model(...)` → `ModelArtifact`；
  `make_walk_forward_splits`/`purge_overlap`/`apply_embargo`/`purge_before_boundary`；
  `TrainOnlyFitGuard`/`assert_frozen_preprocessing`/`run_all_negative_controls`；
  `predict_oos`（禁止泄漏）。`LabelContract(return_basis="vwap_to_vwap")`。
- **契约/常量**：`AFTER_CLOSE_TO_NEXT_VWAP`（默认决策时钟）、`BEFORE_SAME_DAY_VWAP`。
- **被谁调用**：lightgbm_qs 训练脚本（同 repo）、platform model DTO 映射。
- 注意：测试在 `../tests/modeling/`（从 repo 根跑）。

### 2.8 vectorbt_qs

- **接口**：`run_backtest("ashare", target_weights, config={...})` → `pf`（vectorbt Portfolio）；
  `portfolio_report(pf)`。CLI：`python -m vectorbt_qs backtest -b B001` / `run config.yaml` /
  `accurate-batch --positions ... --barra-root ...`。
- **模式**：`fast`（零摩擦因子筛选）vs `accurate`（真实股数/现金/整手/涨跌停/费用/公司行动）。
- **被谁调用**：lightgbm_qs 的 `backtest_final_vectorbt.py`；风险模型 B/f 暴露分析 `mvp/analysis`。

### 2.9 riskfolio_qs

- **接口**：`OptimizationPipeline().run(bundle, scenario="index_enhancement")`；
  `BarraPrecomputedAdapter(risk_root, ...).build_bundle()`；CLI `riskfolio-qs optimize --config ...`。
- **输出**：`target_positions.parquet` + `trades.parquet` + `summary.parquet` + `metadata.json`。
- **依赖**：`data_access`（benchmark/tradable/Return）。**被谁调用**：vectorbt_qs（下游回测）。

### 2.10 quant_platform（平台集成层）

- **现状**：纯 stdlib DTO 合同层（23 个模块，`__all__` 135 项），auth/RBAC/outbox/worker/
  orchestrator 骨架已建；**Control/Metadata 平面未实现**（无 Postgres/无 Temporal/无统一 registry）。
  只做"合同 + 适配"，**禁止重算量化逻辑**（不重算 RankIC，不落第二套因子库）。
- **接口**：`quant_platform.app.contracts`（ArtifactRef/EventEnvelope/JobSpec/WorkflowSpec/
  FactorCandidateManifest/...）；`app.api.create_app()`（FastAPI：/auth/* /health）。
- **被谁调用**：platform_web（前端类型即 contracts DTO）。

### 2.11 platform_web

- React 18 + TS + Vite + TanStack Query；20 页面；`src/api/types.ts` 镜像 contracts DTO。
- 无 node 环境，构建走 CI（`npm run typecheck/test/build`）。

### 2.12 alphaprobe（深度/RL 挖掘）

- **现状**：代码完整、**未在产线跑通**（缺 cold_start_library 是已知阻塞）。
- 单轮：`runner.py`（冷启动库 → AlphaKnowledgePool/Trainer → FE DSL 评估 → candidate_pool 投递）。
  连续：`continuous.py`（7×24，systemd `deploy/alphaprobe-continuous.service`）。
  FE 桥：`fe_bridge/`（表达式→DSL、`enable_factor_engine_evaluation`）。
- **依赖**：torch + factor_engine（DSL 评估）+ data_access（行情）。LLM 额度走 `.env`。

---

## 3. 数据流与典型链路

### 3.1 因子挖掘链路（ML/DL）

```
alphaprobe(候选公式) / 人工 DSL
   │ formula
   ▼
factor_engine 算因子值 ──► factor_lake
   │ 因子值 (date×instrument)
   ▼
quant_evaluator 评估（rank_ic/ic_ir/...） ──► EvaluationBundle
   │ 证据
   ▼
factor_assets 注册/去重/聚类/入库（LifecycleState: REGISTERED→…→PRODUCTION_READY）
```

### 3.2 因子到模型链路

```
factor_lake ─► factor_preprocess（中性化/标准化/FeatureBundle）
   ▼
modeling.train_model（walk-forward 切分 + purge/embargo + leakage guard）
   ▼
predict_oos（vwap-to-vwap 标签）
```

### 3.3 组合与回测链路

```
信号/因子 → riskfolio_qs（组合优化）→ target_positions.parquet
   → vectorbt_qs（accurate 回测）→ 净值/交易/暴露报告
```

### 3.4 平台展示链路

```
各域仓库 ─► quant_platform(contracts DTO + API) ─► platform_web(React)
```

---

## 4. 仓库目录树速查（每库顶层）

| 库 | 顶层要点 |
|---|---|
| factor_engine | `api/`(DSL 入口) `expr/`(AST) `ir/`(分析) `planner/`(编译+CSE) `backend/`(执行) `cleaned_operators/`(算子唯一实现) `storage/`(数据源+落盘) `runtime/`(FactorEngine) `service/`(HTTP) |
| data_access | `store.py`(门面) `core/`(DuckDB 引擎) `read/`(读+PIT+ReadHandle) `write/`(publish/upsert) `cos/`(镜像/直读) `clickhouse/` `service/`(HTTP) `config/datasets.yaml` |
| quant_evaluator | `api/`(EvaluationRequest/Bundle) `contracts/`(LabelBundle/FactorBatch/…) `metrics/`(各指标实现) `registry/`(MetricRegistry) `runtime/`(Evaluator/streaming) `adapters/`(FE/DA) `reporting/` |
| factor_assets | `contracts/` `registry/`(AssetRepository+SQLite) `identity/`(canonical hash) `seen_index/` `similarity/`(ANN) `clustering/`(Leiden) `library/`(PromotionGate) `lifecycle/` `assembly/` `aggregation/` `campaigns/` `adapters/` |
| factor_optimizer | `search/`(SearchRunner/pareto/desirability/winner) `grammar/`(变异) `policy/`(admission) `contracts/`(trial/ledger) `llm/`(prompts) `adapters/`(FE/QE) |
| factor_preprocess | `registry/`(PolicyPreset/TransformRegistry) `transforms/`(rolling/smoothing/cs/volatility) `neutralization/`(OLS/ridge/…) `regime/` `representation/` `eligibility/` `contracts/`(FeatureBundle/FittedState) `adapters/`(data_access/factor_assets) |
| modeling | `contracts.py`(DecisionClock/LabelContract) `dataset.py`(PanelDataset) `split.py`(date_bounded_split) `walk_forward.py`(purge/embargo) `leakage_guard.py`(TrainOnlyFitGuard) `trainer.py` `artifact.py` `predictor.py` `evaluation.py` `learners/` |
| vectorbt_qs | `cli.py` `configs/` `mvp/`(engine/data/constraints/analysis) `contracts/`(BacktestRequest/Artifact) `vectorbt/`(vendored) `docs/` |
| riskfolio_qs | `src/riskfolio_qs/`: `optimizers/`(router/cvxpy) `adapters/`(BarraPrecomputed/mock/real) `runners/pipeline.py` `constraints/` `smoothers/` `analysis/`(仓位分析) `configs/optimizer/`(YAML) |
| quant_platform | `app/contracts/`(23 模块 DTO) `app/db/`(33 表 schema) `app/security/`(auth/RBAC) `app/storage/`(COS adapter) `app/worker/`(jobs/outbox) `app/orchestrator.py` `app/api/app.py` |
| platform_web | `src/pages/`(20) `src/api/`(client+types) `src/hooks/`(Query) `src/components/` |
| alphaprobe | `ashare/alphaprobe-dev/src/alphaprobe/`: `runner.py` `continuous.py` `cold_start/` `trainer/` `delivery/` `fe_bridge/`; `src/baselines/`(GAN/GFN/…) `deploy/`(systemd) `configs/` |

---

## 5. 测试与验证（怎么确认没改坏）

| 库 | 跑法 |
|---|---|
| factor_engine | `cd factor_engine && PYTHONPATH=. pytest tests/ -q`（1144+ tests） |
| data_access | `cd data_access && pytest tests/ -q`（161+ tests） |
| quant_evaluator | `cd quant_evaluator && pytest tests/ -q`（29 文件，全量 332 passed） |
| factor_assets | `cd factor_assets && pytest tests/ -q`（63 tests） |
| factor_optimizer | `cd factor_optimizer && pytest tests/ -q`（28 tests） |
| factor_preprocess | `cd factor_preprocess && pytest tests/ -q`（12 tests） |
| modeling | `cd /repo-root && pytest tests/modeling -q`（33 tests，**从 repo 根跑**） |
| vectorbt_qs | `cd vectorbt_qs && pytest tests/ -q`（6 tests） |
| riskfolio_qs | `cd riskfolio_qs && python -m pytest -q`（5 tests） |
| quant_platform | `cd quant_platform && pytest tests/ -q`（31 tests） |
| alphaprobe | 未跑通（阻塞：缺 cold_start_library） |
| platform_web | `npm run typecheck && npm test`（需 node；CI 里跑） |

内存/核数红线：机器 32 核 92G，子进程并行一律 `OMP_NUM_THREADS<=31`、workers≤31。

---

## 6. 代码同步机制（重要，别乱动）

- **主仓** = `github.com/18047533889/quant_projects`（monorepo，所有代码）。
- **12 个独立库** = HKUST-QUANT-SOCIETY 下镜像，由 `push_both.sh` 同步：
  - 规则：`git push origin main` 后 pre-push hook 自动后台跑 `bash push_both.sh`；
  - 手动：`bash push_both.sh`（全 12 库）或 `bash push_both.sh --only fe`（单库）；
  - 每个库整树 rsync（排除 r26_cache/factors HTML/backup 等可再生文件），独立 commit + push main + dev；
  - 克隆缓存 `~/.cache/qs_sync/`（factor_engine 首次 3.1G，之后增量）。
- **禁止**：往 12 个独立库直接 push（只通过 push_both.sh 同步）；改仓库间接口要先在
  对应独立库 + 主仓同步更新。

---

## 7. 交接清单（接手人逐项确认）

- [ ] 读本文档 §1（口径红线）与 §3（数据流）
- [ ] `git clone` 主仓 + 至少 factor_engine/data_access 两个独立库
- [ ] 跑通 factor_engine 最小示例（`examples/simple_factor.py` 或 `run_from_config`）
- [ ] 跑通 data_access 最小读（`get_store().read_frame("ashare_stock_daily", ...)`）
- [ ] 跑通 quant_evaluator 一个评估（构造 `LabelBundle` + `EvaluationRequest`）
- [ ] 跑通 vectorbt_qs accurate 一个回测（B001 或自备 target_positions）
- [ ] 确认复权/vwap 口径在你们环境一致（对齐 `data_access` 的 adj_factor）
- [ ] 确认 12 个 HKUST 库在 org 下可访问（private，需被加为成员）
- [ ] 知悉 alphaprobe 未跑通（cold_start_library 阻塞），如需启用先补数据

## 8. 已知缺口 / 未来工作

- **quant_platform**：Control Plane（Temporal/工作流）、Metadata Plane（Postgres 统一 registry）未建。
- **alphaprobe**：cold_start_library 缺失，LLM 额度需配置（`.env`）。
- **research_platform**（旧）双 artifact authority 待迁移到新域包。
- 独立库与主仓镜像由 push_both.sh 保证；改共享契约时两边都要同步。

---

*Handover prepared for HKUST Quant Society · Sun Haiwei · 2026-09-01*
