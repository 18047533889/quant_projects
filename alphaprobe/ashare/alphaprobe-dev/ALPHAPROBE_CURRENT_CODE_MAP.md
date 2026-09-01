# AlphaPROBE 全量重构 Phase 0 代码地图

> 生成日期：2026-09-01 · 侦察范围：`ashare/alphaprobe-dev/`（src/alphaprobe、src/baselines、src/shared、apps、configs、scripts、deploy）
> 性质：**只读侦察，未修改任何代码**。所有 file:line 均为侦察时点（2026-09-01）证据。

---

## §1 顶层结构

```
alphaprobe-dev/
├── src/alphaprobe/          # 主挖掘链（runner / continuous / trainer / fe_bridge / delivery / cold_start）
├── src/shared/              # shared.alphagen（表达式/图谱/池/RL）+ shared.alphagen_qlib（StockData）+ shared.alphagen_generic + utils
├── src/baselines/           # gplearn / gan / fqf_iqn_qrdqn / alpha_gfn
├── apps/alphaprobe/run_continuous.py    # 7×24 连续挖掘入口
├── apps/baselines/          # train_ppo / train_gfn / train_AFF / train_GP / run_adaptive_combination / combine_AFF / test_gp
├── configs/alphaprobe/      # experiment_ashare_pv_7x24.yaml（当前主配置）、backtest_ashare.yaml、factor_engine_smoke.yaml
├── configs/prompts/         # LLM 提示词（6 个 txt，FE DSL 版）
├── configs/cold_start/unique_templates.txt  # 784 行冷启动模板（历史产物，当前未被代码 import）
├── scripts/                 # check_fe_prompts.py、delivery/export.py
├── deploy/run_continuous.sh # systemd/手动启动脚本
└── pyproject.toml           # 依赖：torch 2.4.0+cu121 / openai 1.55.1 / sentence-transformers 3.3.1 / pyqlib（baselines 用）
```

**主执行链（调用图）**：

```
deploy/run_continuous.sh
  └─ apps/alphaprobe/run_continuous.py
       └─ src/alphaprobe/continuous.py::run_continuous   (7×24 循环，每轮新 campaign)
            └─ src/alphaprobe/runner.py::run_mining_campaign
                 ├─ cold_start/__init__.py  → cold_start_library（外部包）读 yaml 抽样
                 ├─ trainer/pool.py::AlphaKnowledgePool   (容量 2000，pop 最低 IC)
                 ├─ trainer/trainer.py::AlphaKnowledgeTrainer.train  (search_time 轮 LLM 迭代)
                 │    ├─ fe_bridge/expr_parse.py::parse_mining_expression → FactorEngineDslExpression
                 │    ├─ fe_bridge/knowledge_graph.py::patch_knowledge_graph_for_factor_engine_dsl
                 │    └─ trainer/checkpoint.py::save/load_checkpoint_if_exists
                 └─ delivery/exporter.py::DiskV1DeliveryExporter.export_from_pool
                      ├─ fe_bridge/metrics.py::evaluate_all_period_metrics  (train/valid/test 三窗重算)
                      └─ fe_bridge/dsl_convert.py::expression_to_dsl → disk.v1 manifest
```

**baselines 链（独立）**：`apps/baselines/train_ppo.py / train_gfn.py / train_AFF.py / train_GP.py` 全部基于 `shared.alphagen.*`（qlib StockData），**不接 factor_engine、不走 disk.v1 投递**。

---

## §2.3 历史组件 → 当前映射表

| # | 历史组件 | 当前路径 | 当前符号 | 状态 |
|---|----------|----------|----------|------|
| 1 | runner.py | `src/alphaprobe/runner.py` | `run_mining_campaign` (L229) | **已重构**（FE 版单轮入口，v0.1） |
| 2 | trainer.py | `src/alphaprobe/trainer/trainer.py` | `AlphaKnowledgeTrainer` (L441) | **仍存在**（LLM 迭代核心，未批处理） |
| 3 | pool.py | `src/alphaprobe/trainer/pool.py` | `AlphaKnowledgePool` (L13) | **仍存在**（固定 capacity 2000、pop 最低 IC、search 全池重算 embedding） |
| 4 | continuous.py | `src/alphaprobe/continuous.py` | `run_continuous` (L90) | **仍存在**（每轮新 campaign，无跨轮 global memory） |
| 5 | checkpoint.py | `src/alphaprobe/trainer/checkpoint.py` | `save_checkpoint` (L48) / `load_checkpoint_if_exists` (L210) | **已重构**（JSON v1，含 graph） |
| 6 | exporter.py | `src/alphaprobe/delivery/exporter.py` | `DiskV1DeliveryExporter` (L31) | **已重构**（disk.v1 manifest） |
| 7 | fe_bridge/stock_data.py | `src/alphaprobe/fe_bridge/stock_data.py` | `FactorEngineStockData` (L17) | **仍存在**（FE 后端 StockData，pandas 单后端） |
| 8 | fe_bridge/bootstrap.py | `src/alphaprobe/fe_bridge/bootstrap.py` | `enable_factor_engine_evaluation` (L41) | **仍存在**（monkey-patch Expression.evaluate，L53-65） |
| 9 | fe_bridge/dsl_convert.py | `src/alphaprobe/fe_bridge/dsl_convert.py` | `expression_to_dsl` (L97) | **仍存在**（AlphaGen 树 → FE DSL；冷启动走 FactorEngineDslExpression 直通） |
| 10 | AlphaKnowledgeTrainer | `src/alphaprobe/trainer/trainer.py` | `AlphaKnowledgeTrainer` (L441) | **仍存在** |
| 11 | AlphaKnowledgeLogger | `src/alphaprobe/trainer/trainer.py` | `AlphaKnowledgeLogger` (L268) | **仍存在**（logger 内含 adaptive combination / load_test_res 回测式重算） |
| 12 | AlphaKnowledgePool | `src/alphaprobe/trainer/pool.py` | `AlphaKnowledgePool` (L13) | **仍存在** |
| 13 | ExpressionKnowledgeGraph | `src/shared/alphagen/data/expression_knowledge_graph.py` | `ExpressionKnowledgeGraph` (L140) | **仍存在**（单父树，字符串 identity） |
| 14 | ExpressionNode | `src/shared/alphagen/data/expression_knowledge_graph.py` | `ExpressionNode` (L111) | **仍存在**（dataclass，单 parent） |
| 15 | FactorEngineStockData | `src/alphaprobe/fe_bridge/stock_data.py` | `FactorEngineStockData` (L17) | **仍存在** |
| 16 | enable_factor_engine_evaluation | `src/alphaprobe/fe_bridge/bootstrap.py` | `enable_factor_engine_evaluation` (L41) | **仍存在**（monkey-patch） |
| 17 | get_ic | `src/alphaprobe/trainer/pool.py` | `AlphaKnowledgePool.get_ic` (L76) | **仍存在**（`expr: str → eval()`，L78） |
| 18 | get_ic_icir_mutics | `src/alphaprobe/trainer/pool.py` | `AlphaKnowledgePool.get_ic_icir_mutics` (L83) | **仍存在**（str → eval()，L85） |
| 19 | get_mutl_ic | `src/alphaprobe/trainer/pool.py` | `AlphaKnowledgePool.get_mutl_ic` (L90) | **仍存在**（str → eval()，L92） |
| 20 | try_new_expr | `src/alphaprobe/trainer/pool.py` | `AlphaKnowledgePool.try_new_expr` (L352) | **仍存在**（str → eval()，L354） |
| 21 | search | `src/alphaprobe/trainer/pool.py` | `AlphaKnowledgePool.search` (L100) | **仍存在**（top_k 采样，每次全池重算 embedding L169） |
| 22 | Expression.evaluate | `src/shared/alphagen/data/expression.py` | `Expression.evaluate` (L18 abstract) | **仍存在**（被 bootstrap monkey-patch） |
| 23 | ExpressionParser | `src/shared/alphagen/data/tree.py` | `ExpressionParser` (L89) | **仍存在**（仅 legacy baselines 用；挖掘链改走 parse_mining_expression） |
| 24 | shared.alphagen | `src/shared/alphagen/`（data/models/rl/trade/utils/config.py） | 同上 | **仍存在**（baselines + trainer/pool + checkpoint 依赖） |
| 25 | shared.alphagen_qlib | `src/shared/alphagen_qlib/`（stock_data/calculator/strategy/utils） | `StockData` (stock_data.py L39) | **仍存在**（qlib 后端；alphaprobe 挖掘链仅在 logger.load_test_res L289-291 直接实例化） |

---

## §2.4 现状确认（逐项，附 file:line 证据）

### 1. 当前默认 execution backend
- **factor_engine（pandas 单后端）**。
  - `runner.py:330-332`：`--data_backend` 默认 `os.getenv("ALPHAPROBE_DATA_BACKEND", "factor_engine")`，choices `("factor_engine","qlib")`。
  - `stock_data.py:73-74`：`build_backend("pandas")` 硬编码。
  - `deploy/run_continuous.sh:25`：`export ALPHAPROBE_DATA_BACKEND="...factor_engine"`。
  - FE 侧真正生产路由是 Global Physical Planner（`factor_engine/api/mining_integration.py:9-29` 文档），但 AlphaPROBE 的 `FactorEngineStockData._build_engine` 绕过了它，**直接 build_backend("pandas")**。即：AlphaPROBE 未使用 FE 的 auto/hybrid 批执行能力。

### 2. 是否仍 monkey-patch Expression.evaluate
- **是**。`fe_bridge/bootstrap.py:53-65`：`Expression.evaluate = patched_evaluate`（`_PATCHED` 全局守卫 L15/L42-44）。runner 在 `_make_stock_data`（L200）里调用 `enable_factor_engine_evaluation()`。

### 3. 是否仍有 eval(/exec(
- **有，共 5 处**（全部 `eval(`，无 `exec(`）：
  - `src/alphaprobe/trainer/pool.py:78`（`get_ic`）
  - `src/alphaprobe/trainer/pool.py:85`（`get_ic_icir_mutics`）
  - `src/alphaprobe/trainer/pool.py:92`（`get_mutl_ic`）
  - `src/alphaprobe/trainer/pool.py:354`（`try_new_expr`）
  - `src/shared/alphagen_qlib/utils.py:27`（`load_alpha_pool`，legacy qlib 路径，仅 baselines）
  - 注：生产路径所有表达式已是 `FactorEngineDslExpression`（`dsl_expression.py`），`isinstance(expr, str)` 分支实际不触发；但**仍存在**。`src/baselines/gan/network/*.py` 里的 `net.eval()` 是 PyTorch 模式切换，非 eval()。

### 4. Test 是否仍在 logger/search 中被读取
- **是**。
  - runner 构造 `data_test`（`runner.py:265`）传入 trainer（L292 `test_data=data_test`）。
  - `AlphaKnowledgeLogger` 在每次 log 时用 test_data 计算 test_ic/test_icir（`trainer.py:412-421`），并写 `node.test_ic/test_icir` 到图节点。
  - `load_test_res`（`trainer.py:275-403`）在 test 集上做**逐日 adaptive combination 回测式重算**（回归预测 → valid/test IC），产出 `res.txt`（L437-438）。该函数内部硬编码 2010-01-01 起 qlib StockData（L289-291），**不读 experiment yaml 的 split，也绕开了 factor_engine**。

### 5. Pool 是否仍固定 capacity / pop 最低 IC
- **是**。
  - `pool.py:366`：`if self.size < self.capacity` 固定容量。
  - `pool.py:374-375`：`min_ic_idx = np.argmin(self.single_ics[:self.size])`，`pool.py:407-413` `_pop()` 弹最低 `single_ics`。
  - 与父类 `AlphaPool`（`shared/alphagen/models/alpha_pool.py:259-264`）不同：父类按权重 `np.argmin(np.abs(self.weights))`，子类改为按单因子 IC。

### 6. graph 是否仍单父
- **是**。`ExpressionNode.parent: Optional["ExpressionNode"]`（expression_knowledge_graph.py:113）单引用；`children: Set[str]`（L114）只存字符串 key；`insert()`（L191-202）只设一个 parent。无多父/合并/去重逻辑（插入前 `search` 查重 L192）。

### 7. exporter 是否仍重新回测
- **是**。`delivery/exporter.py:170`：`export_from_pool` 对池内每个因子调 `evaluate_all_period_metrics`（`fe_bridge/metrics.py:50-60`），对 train/valid/test 三窗**各新建一个 FactorEngineStockData 并重算**（metrics.py:34-44）。train 窗的 IC 在挖掘时已算过，但 exporter 不读池缓存，全部重算（`_formula_cache` 仅 per-StockData 实例，三窗是三个实例）。无 vectorbt/组合回测，但 metrics 阶段仍是全量重算。

### 8. continuous 是否已有真正 global memory
- **否**。`continuous.py` 每轮：新 campaign_id（L171）→ 新冷启动（`args.cold_start_seed=None` L175）→ `run_mining_campaign`。唯一跨轮状态是 `active_round.json` 崩溃衔接（L54-72）与 STOP_QUOTA 旗标（L75-87）。**无跨轮因子记忆/去重库/累积先验**；`AlphaKnowledgePool` 生命周期随单轮结束。

### 9. checkpoint 是否按对象值重算
- **是**。`checkpoint.py` 只存字符串（expr/topic/description/ic/icir/parent/children），`restore_pool_from_checkpoint`（L150-207）对每个 expr 重新 `parse_mining_expression` + `expr_obj.evaluate(pool.data)` + `_calc_ics_and_icir`（L179-180）全量重算值/IC/互相关。图节点同样按字符串重建（`restore_graph_from_checkpoint` L93-147）。**无序列化对象值**。

### 10. embedding 是否每次 search 全池重算
- **是**。`pool.py:164-176`：`search()` 内对全部候选节点调 `self.embedding_model.encode(texts)`（L169），每轮迭代、每次 search 全池重算，无缓存。模型在 `AlphaKnowledgePool.__init__`（L46-50）用默认 `Qwen/Qwen3-Embedding-4B`（L22；runner 默认 `paraphrase-multilingual-MiniLM-L12-v2`，runner.py:347）。

### 11. formula identity 是否仍字符串
- **是**。图节点 key、pool 查重、checkpoint、`_candidate_hash`（exporter.py:19-22，`" ".join(str(formula).split())` 归一化后 sha256）全部基于字符串。`FactorEngineDslExpression._dsl` 本身可作 identity，但未做 AST 规范化（`dsl_expression.py:12-21`）。`dsl_payload_tokens`（dsl_expression.py:96-106）提供粗粒度 token 化，仅用于图 payload length，不参与 identity。

### 12. 当前 label / split / universe（configs/alphaprobe/experiment_ashare_pv_7x24.yaml）
- **label**：`mining.label_days: 20`（yaml L44），runner `runner.py:269-271` 构造 `target = Ref(vwap, -label_days) / vwap - 1`（vwap→vwap 远期收益，`label_price_pair: vwap_to_vwap` yaml L46）。注：`Ref(vwap, -20)` 在 AlphaGen 语义是**过去** t-20 的 vwap 除以当前 vwap（方向为「20 日前值 / 今日值 - 1」= 反向 20 日收益），与 `Vwap.pct_change().shift(-1)` 的「未来收益」方向相反 —— 实际以 `Ref(vwap, -label_days)/vwap - 1` 为准（挖掘侧已用此口径）。
- **split**：train 2016-01-01..2021-12-31 / valid 2022-01-01..2023-12-31 / test 2024-01-01..2026-07-31（yaml L12-18，`delivery/config.py:22-26` 默认同值）。
- **universe**：`delivery.universe_id: A_SHARE_ALL_A_EX_ST`（yaml L27）；`data.instruments` 默认 `csi300`（config.py:163）但 runner `_make_stock_data` 用 `args.instruments or experiment.data.instruments`（runner.py:197），默认未设置时用 csi300；**注意 universe_id（全 A 去 ST）与 instruments（csi300）不一致**。
- **fe_profile**：`ashare_pv_valuation`（yaml L9），数据源 `default_ashare_pv_valuation_data_source_config`（config.py:309-315）→ `data_access` 数据集 `ashare_stock_daily_adj`（Adj* 复权权威）+ `ashare_stock_valuation_daily` 复合。
- 阈值：`ic_threshold: 0.006`、`ic_export_threshold: 0.006`（yaml L49-50）、`ic_mut_threshold` 默认 0.9（runner.py:363）。

### 13. Evaluator / Refinement / Consolidation 公共包是否已存在可调用
- **Evaluator：已存在可调用**。`quant_projects/quant_evaluator/`：
  - `runtime/evaluator.py:118` `class Evaluator`，`evaluate(factor_batch, label_bundle, metric_specs, use_chunking=True)`（L178）。
  - `runtime/evaluator.py:895` 公共函数 `evaluate(factors, labels=None, *, context, metrics, where, evaluator, split_ref)`。
  - 输入契约：`contracts/factor_batch.py:24` `FactorBatch`（wide 3D (T,A,F) + 显式 AxisRef）、`contracts/label_bundle.py:11` `LabelBundle`（`price_convention: "vwap_to_vwap"` 默认）。
  - 51 个注册 metric（`registry/metrics.py`，rank_ic L546、pearson_ic、ic_ir、quantile_spread、turnover、hac_tstat 等）。
  - `adapters/factor_engine.py:44` `FactorEngineAdapter`：`compute_batch(factor_exprs, start_date, end_date, universe, market)` → FactorBatch；`get_factor_id` / `get_structural_id`（identity 提取，**结构化 ID 已存在**）。
  - **AlphaPROBE 当前零引用**：`grep quant_evaluator src/alphaprobe` 无命中（fe_bridge/metrics.py 仍手写 QLibStockDataCalculator）。
- **Refinement / Consolidation：无独立公共包**。quant_projects 下没有名为 refinement/consolidation 的包；`factor_preprocess/`（transforms/neutralization/eligibility/grammar）、`factor_optimizer/`（search/policy/complexity/llm）、`factor_assets/`（campaigns/clustering/aggregation）各自成库，**均未被 alphaprobe import**。
- **fe_bridge/metrics.py 现状**：仍手写 `_calc_ic_ir`（L16-22）用 `QLibStockDataCalculator` + 自算 icir，未走 quant_evaluator。

---

## §3 FE/DA 可用 API 摘要

### 3.1 factor_engine 公开入口（import 路径 + 签名）

**核心门面** `factor_engine.runtime.engine.FactorEngine`
```python
FactorEngine(backend, data_source, cache=None, *, run_mode: str|None=None, production_fallback_policy: str|None=None)  # engine.py:1255
engine.compile(factor, *, pit_enforce=False, pit_forbid_forward_fill=False)   # :1302
engine.run(factor, *, plan=None, analysis=None, input_dq_check=False, input_dq_strict=True,
           input_dq_thresholds=None, auto_warmup=False, trim_warmup=True, market=None,
           pit_enforce=False, pit_forbid_forward_fill=False)                  # :3018
engine.run_many(factors: Sequence[Factor], *, perf=None, enable_cse=None, auto_warmup=False,
                trim_warmup=True, market=None, input_dq_check=False, input_dq_strict=True,
                input_dq_thresholds=None, pit_enforce=False, pit_forbid_forward_fill=False,
                result_policy="return", sink=None, warmup_clusters=False) -> dict  # :3599
engine.run_many_parallel(factors, *, n_jobs=None, perf=None, enable_cse=None, auto_warmup=False,
                trim_warmup=True, market=None, input_dq_check=False, input_dq_strict=True,
                input_dq_thresholds=None, pit_enforce=False, pit_forbid_forward_fill=False,
                result_policy="return", sink=None) -> dict                        # :3707
engine.run_from_config(config_path) / materialize_from_config(...) / materialize_incremental(...)  # :1645/:2738/:2787
engine.with_data_source(data_source, *, fresh_cache=False)                         # :3764
```

**执行编排（run_many 实现体）** `factor_engine.runtime.batch_service`
```python
execute_run_many(engine, factors, *, perf=None, enable_cse=None, ...)             # batch_service.py:1222
execute_run_many_parallel(engine, factors, *, n_jobs=None, ...)                   # :1614
materialize_shared_nodes_parallel(dag, backend, ctx, *, max_workers=None, executor=None)  # :51
choose_execution_mode(...)                                                          # :856
```

**批调度** `factor_engine.runtime.adaptive_batch_scheduler.AdaptiveBatchScheduler`
```python
AdaptiveBatchScheduler(...)      # :282   (SchedulerPlan :70, task_execution_certificate :526)
```
（micro-batch / 动态 wave budget / OOM replan 已在库内。）

**DSL 解析 / 校验** `factor_engine.api.dsl_parser`
```python
parse_expr(text, *, surface="daily", dialect="native", dialect_version=None, budget=None) -> Expr  # dsl_parser.py:294
parse_factor(text, *, name="factor", freq="1d", universe=None, description=None, surface="daily", ...) -> Factor  # :295
DSLParseError  # :23
```

**DSL 白名单** `factor_engine.api.operator_registry`
```python
build_dsl_allowlist(*, surface="daily", dialect="native", dialect_version=None) -> dict[str, Callable]  # operator_registry.py:12
build_authoring_allowlist(...) / build_research_mining_allowlist() / build_production_mining_allowlist()
```

**挖掘对接** `factor_engine.api.mining_integration`
```python
validate_factor_engine_dsl(formula, *, surface="daily") -> tuple[bool, str]         # :237
validate_production_dsl(formula, *, market) -> tuple[bool, str]                     # :58
validate_production_fastpath_dsl(formula, *, strict=None, market) -> tuple[bool, str]  # :148
default_mining_operator_allowlist(*, tier="production_fastpath") -> list[str]       # :1172
default_typed_mining_search_space_config(*, tier, fields, max_domains=2, max_cost, field_dq_policy) -> dict  # :1352
default_ashare_pv_data_source_config(*, max_files, start_date, end_date) -> dict    # :352  (data_access 数据集 ashare_stock_daily_adj, Adj* 权威)
default_ashare_pv_valuation_data_source_config(*, start_date, end_date) -> dict     # :392  (pv + valuation composite)
default_ashare_pv_universe_data_source_config(*, index_symbol, start_date, end_date) -> dict  # :548 (status asof + constituent exact)
validate_manifest_for_execution(*, market, expression_type, formula, require_production, require_fastpath, surface)  # :1102
default_mining_label_config(*, horizon_bars=5, feature_lookback_bars=20, gap_bars=1) -> dict  # :1648
validate_mining_label_formula(formula, *, enforce=True) -> tuple[bool, str]          # :1664
```

**CSE** `factor_engine.planner.cse`
```python
apply_cse(roots: list[PlanNode]) -> tuple[list[PlanNode], dict[str, PlanNode]]      # cse.py:317
```

**Backend 工厂** `factor_engine.backend.factory.build_backend(backend_type: str)`
- 支持 `pandas` / `polars` / `polars_long` / `auto_long` / `duckdb_sql` / `clickhouse_sql` / **`auto`/`hybrid`**（SQL+Polars 混合；数据源有 `scan_polars_long` 时自动 hybrid_long）/ `debug` / `q_kdb`。

**数据源工厂** `factor_engine.storage.factory.build_data_source(config, *, build_context=None)`
- 支持 `data_access` 数据集 / parquet / composite / clickhouse 等。AlphaPROBE 现用 `default_ashare_pv_valuation_data_source_config` → composite(data_access)。

**标签契约（PIT）** `factor_engine.api.label_spec`
```python
LabelSpec(horizon)                    # label_spec.py:26 ; mature(anchor, fit_time) :31
label_available_mask(dates, horizon, fit_date) -> np.ndarray   # :45
purged_time_split(dates, *, train_frac=0.7, horizon=5, embargo=0) -> (train_idx, valid_idx)  # :87
```
`factor_engine.api.label_pit`：PIT 标签公式校验（`validate_label_formula_for_pit`），`AdjVwap(t+H)/AdjVwap(t+1)-1` 对齐 `TargetVwapReturnH01/H05/H10/H20`（label_pit.py:317）。

**Universe 契约** `factor_engine.backend.universe_spec`
```python
UniverseSpec(empty_universe="preserve_keys_null", filtered_all_invalid="preserve_keys_null",
             shape_preserving_cross_section=True, strict_axes_mandatory=True,
             preserve_delisted_in_history=True)   # universe_spec.py:27
UNIVERSE_SPEC / empty_universe_preserves_keys() / strict_axes_mandatory()
```

**计划/物化**：`engine.plan_many_fast`（:2277）、`materialize_many_fast`（:2359）、`materialize_matrix`（:2591）、`plan_incremental_from_event`（:2645）、`materialize_incremental_from_event`（:2676）——因子湖增量落盘能力齐全。

### 3.2 data_access 公开入口（import 路径 + 签名）

**统一 Store** `data_access.get_store()`（`__init__.py` 顶层）→ `data_access.store.DataAccessStore`（store.py:234）
```python
store.load_columns(dataset, *, columns: Sequence[str], time_range=None, instrument_filter=None,
                   filters=None, output_names=None, normalize_timestamp=None, timestamp_unit=None,
                   **params) -> dict[str, pd.Series]         # store.py:7058  (每列 (ts,instrument) MultiIndex Series —— 与 FactorEngineStockData._load_columns_dict 直接对接)
store.read_auto(dataset, *, columns=None, time_range=None, instrument_filter=None, filters=None,
                limit=None, query_budget=None, mode="auto", prefer_polars=False, batch_size=100_000) -> pa.Table  # :2981
store.read_arrow(...) :2448 / read_arrow_stream(...) :2502 / read_frame(...) :2950 / read(...) :3162
store.read_joined(...) :3377 / store.read_factors(...) :6558 / store.read_asof(...) :2409 / store.read_cached(...) :4814
```
- 构造：`DataAccessStore(registry, engine, *, principal, authorizer, credential_provider)`（store.py:234）；registry 来自 `data_access.registry.loader.load_registry(config_path=None)`（registry/loader.py:789）→ `DatasetRegistry`。
- **PIT 事件索引**：`data_access.read.pit_event_index.PITEventIndex`（:781 `load_pit_event_index(path)`）。
- **语义目录**：`data_access.read.semantic_catalog.SemanticFieldCatalog` / `get_semantic_catalog()`。
- **交易日历**：`data_access.read.session_calendar.get_market_calendar / get_market_session / MarketCalendar`。
- **COS 契约**：`data_access.cos_contract`（`COS_DATASET_CONTRACTS` / `get_cos_contract` / `validate_panel_request`）——A 股权威口径 `ashare_stock_daily_adj`：`Return` 单位 1/10000、`Factor` 后复权乘子、`TargetVwapReturnH01/H05/H10/H20` 标签列已登记（cos_contract_ashare.py:19）。
- **因子湖批量读**：`data_access.read.factors.build_factor_union_sql`（factors.py:330）+ `FactorMeta`（:32）——一次 UNION ALL 读多因子。
- 未发现 `UniverseSpec`（数据访问层）；universe 契约在 **factor_engine**（backend/universe_spec.py）。股票池/停牌/成分在 FE `default_ashare_pv_universe_data_source_config`（status + constituent composite）中作为数据源表达。

---

## §4 冷启动接入点

### 4.1 `src/alphaprobe/cold_start/__init__.py`（现状）
- 薄封装：`_locate_cold_start_library_src()`（L9）在 `quant_projects/` 下找 `factor_engine/` + `cold_start_library/src/`，插入 `sys.path`（L24-32）。
- 再导出 `cold_start_library` 顶层符号（L35-48）：`ColdStartEntry`、`default_yaml_path`、`load_cold_start_for_training`、`load_cold_start_mixed_for_training`、`load_cold_start_multi_library_for_training`、`load_cold_start_yaml`、`package_root`、`sample_cold_start_entries`、`recommended_max_backtrack_days`（来自 `cold_start_library.runtime.backtrack`）。
- 默认 yaml 常量（L50-53）：`DEFAULT_YAML = default_yaml_path("ashare", "backend_v9_core.yaml")`、`ALPHA101/158/191` 指向 `wq101.yaml` / `qlib158.yaml` / `gtja191.yaml`。
- **注意**：`cold_start_library` 的 `data/ashare/` 目录**当前为空**（无 `backend_v9_core.yaml`，无 value_cache 产物）——冷启动 yaml 文件尚未落盘到该目录（library/ 下是 JSON 版 V9 算子清单，格式不同）。

### 4.2 `configs/cold_start/unique_templates.txt` 格式
- 784 行，每行一个 factor_engine DSL 模板，含 `{P}` 窗口占位符（如 `zscore(ts_mean(volume, {P}))`），末尾已有完整 DSL 形态（`if_else` / `pb_ratio` / `pe_ratio` / `market_cap` / `total_capital` 等估值字段）。
- **未被任何代码 import**（`grep unique_templates src/ apps/ scripts/` 无命中）——属历史产物/归档。

### 4.3 experiment yaml 里 `cold_start_library` 指向
- `configs/alphaprobe/experiment_ashare_pv_7x24.yaml:58`：`cold_start_library: ~/quant_projects/cold_start_library/data/ashare/backend_v9_core.yaml`（CLI `--cold_start_library` 可覆盖，runner.py:108/319）。
- `cold_start_mix.libraries.pv`（yaml L68）同指向；ratios `pv: 1.0`，alpha101/191/158: 0.0（yaml L63-66）。
- 运行入口：`runner.py::resolve_cold_start`（L100-168）→ `_resolve_cold_start_library_paths`（L71-97）→ `load_cold_start_multi_library_for_training`（cold_start_library/runtime/loader.py:188）。

### 4.4 哪些模块 import 它
- `src/alphaprobe/runner.py`（L13-22）：import 全部冷启动符号 + `recommended_max_backtrack_days`。
- `src/alphaprobe/fe_bridge/stock_data.py`（L133-171）：`_try_load_value_cache` 懒加载 `cold_start_library.runtime.value_cache`（读 `COLD_START_VALUE_CACHE` env 或默认 `~/quant_projects/cold_start_library/data/ashare/value_cache`，L136-147）——**value_cache 目录当前不存在**，命中失败静默返回 None（L170）。
- `scripts/check_fe_prompts.py`（L158）：读 `~/quant_projects/cold_start_library/library/production_default_core_v9.json` 做 V9 算子覆盖率校验。

---

## §5 configs/prompts / scripts / apps / deploy 调用关系

```
configs/prompts/*.txt (6 文件)
  system_head.txt → PROMPT_HEAD
  features_operators_fe_dsl.txt → PROMPT_FEATURES_AND_OPERATORS (289 行，算子白名单+section5)
  generation.txt → PROMPT_GENERARTION (51 行，含 {topic}{expressions}{explanations}{traces}{num})
  validity_fe_dsl.txt → PROMPT_DIMENSION_REDUCTION (30 行)
  compare_fe_dsl.txt → PROMPT_COMPARE (16 行)
  separation_fe_dsl.txt → PROMPT_SEPARATION (24 行)
       ↑ 由 shared/utils/prompt_loader.py (load_prompt @lru_cache L56-74, _PROMPT_FILES 映射 L18-25)
       ↑ 再经 shared/utils/prompt.py 导出 PROMPT_* 常量 (prompt.py:11-16)
       ↑ 运行时目录可被 runner._init_prompts 覆盖 (runner.py:57-68, yaml mining.prompts.dir)
       ↓ 消费方
         trainer.py:588 chat_generate(PROMPT_HEAD, build_generation_user_prompt(...))
         knowledge_graph.py (import PROMPT_COMPARE/PROMPT_HEAD，但当前图已 patch 掉 LLM 对比)

scripts/
  check_fe_prompts.py         # 独立校验脚本：EXAMPLES 全部 parse_expr(compat)；legacy 语法拒绝；section5 ops 对照 compat allowlist；V9 JSON 覆盖率。不进入运行链。
  delivery/export.py          # 只导出 candidate_pool（--pool_json 从 AlphaKnowledgeLogger 的 pool_*.json 恢复），复用 DiskV1DeliveryExporter + evaluate_all_period_metrics。

apps/
  alphaprobe/run_continuous.py   # sys.path+=src → alphaprobe.continuous.main()  (7×24)
  baselines/*.py                  # 独立入口（train_ppo/train_gfn/train_AFF/train_GP/...），依赖 shared.alphagen(qlib)，不接 FE/投递。

deploy/
  run_continuous.sh               # 启动 apps/alphaprobe/run_continuous.py --experiment_config configs/alphaprobe/experiment_ashare_pv_7x24.yaml --cuda 0
                                  # PYTHONPATH=src, ALPHAPROBE_DATA_BACKEND=factor_engine, 默认 python rdagent4qlib

数据流终点：DiskV1DeliveryExporter → ~/quant_projects/data/factor_pools/candidate_pool/{campaign_id}/ (config.json + {candidate_id}/manifest.json)
```

---

## P0 风险清单（改造成批执行 / Global Memory / 封存 Test 时最容易踩的 10 个点）

1. **Test 泄漏进训练信号**：`AlphaKnowledgeLogger.log()`（trainer.py:405-438）把 test_ic/test_icir 写回图节点 `node.test_ic/test_icir`（L430-431），`search()` 的 `compute_leaf_quality`/`compute_non_leaf_quality`（pool.py:219-296）读 `node.icir`/children 的 icir——若改造后图节点复用，Test 会污染后续轮次的选择。封存 Test 前必须物理隔离：logger 用独立图副本或只读快照。
2. **eval() 残留**：pool.py:78/85/92/354 四个 `eval()` 分支仍在（str 入参）。冷启动库接入后若恢复"字符串→表达式"路径，等于重开任意代码执行。重构时必须改走 `parse_mining_expression`（已封死 legacy），并删掉 str 分支。
3. **logger.load_test_res 硬编码 qlib 且不读 yaml**：trainer.py:279-291 固定 `QLIB_PATH`、`valid 2021-01-01..2022-06-30 / test 2022-07-01..2025-06-30`、20 日 close 目标——与 yaml split（test 2024..2026-07-31）与 vwap 目标**不一致**。它产出 res.txt 但结果与官方 split 无关，改造/封存时勿引用其数字。
4. **checkpoint 全量重算 + 无值序列化**：checkpoint.py:179-180 恢复时 `evaluate` 全池重算，三窗 StockData 重载 + `_optimize`（L200）500 iter Adam——大池（2000）恢复会非常慢；且值/互相关矩阵不落盘，Global Memory 若复用 checkpoint 结构会重复计算。建议改存因子值 hash 或复用 value_cache。
5. **search() 每次全池 embedding + O(n²) 相关矩阵**：pool.py:169（embedding encode 全池）、L160（`mutual_ics` 全矩阵 abs）、L186-195（edit distance 双循环）。池 2000 时 search 单次成本极高；批执行/Global Memory 需缓存 embedding（增量更新）并稀疏化 mutual_ics。
6. **单父树 ≠ DAG 去重**：graph.insert 只查字符串查重（knowledge_graph.py:192），同形不同字符串（`ts_mean(close,20)` vs 换窗）会成不同节点；`children` 是字符串 Set（L114），多父会丢指针——Global Memory 若想跨轮合并图，需先把 identity 从字符串升级为规范化 DSL/AST hash（`FactorEngineDslExpression._dsl` + `dsl_payload_tokens` 已可作雏形），并重建多父/引用计数。
7. **pool 弹最低 IC ≠ 组合最优**：pool.py:374-382 `try_new_expr` 用 `argmin(single_ics)` 弹出——父类 `AlphaPool._pop`（alpha_pool.py:262）是 `argmin(|weights|)`。Global Memory 若保留旧因子淘汰逻辑，会偏向单因子 IC 而丢掉组合互补性（与 quant_evaluator 51 metric + 组合链的去重口径冲突）。
8. **FactorEngineStockData 绕过 FE 生产路由**：stock_data.py:73-74 硬编码 `build_backend("pandas")`，没用 FE 的 `run_many`/`run_many_parallel`/CSE/hybrid/AdaptiveBatchScheduler。批执行改造时若只替换 `evaluate_dsl`（L117-131 逐个 `engine.run`）而不切 `run_many`，共享子树白算、性能上不去；且 `_formula_cache` 是 per-instance dict，三窗/多 data 实例间不共享。
9. **dsl_convert 字段集 vs 数据源字段集不一致**：`FEATURE_TO_FIELD`（dsl_convert.py:51-58）只含 open/close/high/low/volume/vwap，但 `fe_profile=ashare_pv_valuation`（config.py:165/296-315）数据源含 pe/pb/turnover_ratio/market_cap；`_VALUATION_FIELD_ALIASES`（mining_integration.py:41-47）在 DSL 侧可用这些字段，但 `AlphaKnowledgePool.search()` 的候选表达式里若有估值字段，`dsl_payload_tokens`（dsl_expression.py:100-103）的 field 正则只匹配 6 个价量字段 → token 数少算、图 payload length 失真；`_candidate_hash` 只做空白归一（exporter.py:20）→ `rank(close)` 与 `rank( close )` 同 hash，但 `if_else` 参数顺序不同不会归一。
10. **universe_id 与 instruments 口径分裂**：yaml `universe_id: A_SHARE_ALL_A_EX_ST`（全 A 去 ST）vs `data.instruments` 默认 `csi300`（config.py:163），runner 传 instruments 给数据源（L197-207）但 manifest 写 universe_id——批执行/Global Memory 若要跨轮对齐因子，需先统一「训练 universe = 实际数据源 universe」，否则 checkpoint 恢复、投递去重都会错位；`default_ashare_pv_universe_data_source_config`（mining_integration.py:548，status asof + constituent exact）已是现成模板但 AlphaPROBE 未用。

---

*P0 侦察结束：本文件仅记录现状，未修改任何源代码。*
