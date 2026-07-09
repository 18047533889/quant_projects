# 企业级因子引擎完善路线图

> 对照外部「私募级因子引擎」九类标准，结合 **本仓库现状**（`factor_engine` + `data_access` + cleaned_operators）制定的可执行计划。  
> 原则：**能落地的先做，不照搬与架构不符的建议**。

---

## 现状摘要（2026-07）

| 能力 | 现状 | 评级 |
|------|------|------|
| 算子数量 | **361** canonical implemented · DSL 白名单 **~560** | ✅ |
| Backend 覆盖 | Polars **325** · SQL 下推 **58** 算子（+ column/literal = **60**）· registry **62** · pandas-only **36** | ✅ |
| 算子层因果 | `_causal.py` + Tier-1 自动扫测 + `test_causal_operators.py` | ✅ |
| 数据层 PIT | Composite `asof_backward` + mining preset 审计 + **examples 全量 `data_access`** | ✅ |
| lookback 分析 | `Analyzer` + `OperatorPolicy` + `WindowedDataSource` 日内精度 | ✅ |
| 增量计算 | tick 精确扩窗 + watermark + 交易日历 | ✅ |
| 血缘/版本 | `factor_run` + `RunLineage` + catalog hash | ✅ |
| DQ Gate | profile + strict/input DQ + SLO | ✅ |
| 输出格式 | 长表元数据 + 宽表 pivot | ✅ |
| 测试 | golden parquet + Tier-1 causal + nightly CI | ✅ |

---

## ChatGPT 九类建议 → 本仓库映射

### 1. Point-in-Time（最高优先级）✅

**适用**：全部因子；**分工**：

- **数据层**（主战场）：`publish_time` / `available_time`、universe、行业、成分股、停牌 → `CompositeDataSource` + `data_access` YAML
- **算子层**（已做一轮）：禁止 lead/bfill/全样本广播/双向滤波 → `_causal.py`
- **标签层**（已完成）：`api/label_pit.py` + `mining_integration.default_mining_label_config`

**不适用/需改写**：

- 「所有 join 默认 backward asof」——已在 Composite 支持，但 **A 股估值 preset 仍可能用 exact**，需改配置而非改算子
- 「引擎内置 trade_date/asof_time 六元组」——当前是 `(timestamp, instrument)` MultiIndex，扩展 schema 放在 **因子湖长表** 而非改中间态

**计划**：

| 阶段 | 任务 | 文件 |
|------|------|------|
| P0 | 扩展 causal 测试 + 算子 `pit_safe` 标记 | `operator_policy.py`, `test_causal_operators.py` |
| P0 | 基本面 Composite 默认 `asof_backward` 审计 | `api/mining_integration.py`, examples configs |
| P1 ✅ | `tests/test_composite_pit.py` 构造「未来才发布」的财报 | `tests/` |
| P2 ✅ | 因子湖增加 `calc_time`, `data_snapshot_id`, `factor_version`, `is_valid` | `storage/materializer.py` |

---

### 2. 算子定义标准化 ✅ policy + catalog 自动生成

**适用**：企业级需要统一 window/lag/nan_policy，避免「同名不同义」。

**现状**：`OperatorPolicy` + `infer_operator_policy` + 50+ 核心算子显式 override；`operators_catalog.md` 自动生成。

**计划**：

| 阶段 | 任务 |
|------|------|
| P0 ✅ | 新增 `OperatorPolicy`：`scope`, `lookback`, `min_periods`, `lag`, `pit_safe`, `nan_policy` |
| P0 ✅ | `infer_operator_policy(op)` 按 category/tags 推断 |
| P0 ✅ | `compute_operator_catalog_hash()` 用于 lineage |
| P1 ✅ | 核心 50+ 算子显式声明 policy（ts_mean, rank, neutralize…） |
| P2 ✅ | policy 写入 `operators_catalog.md` 自动生成 |

---

### 3. 时间序列算子窗口边界 ✅

**适用**：rolling 仅历史、min_periods 固定、delay 先于 rolling（文档约定）。

**现状**：pandas rolling/expanding；Analyzer 消费 `OperatorPolicy.lookback_window` / `min_periods`；`ts_std` ddof=1 已文档化。

**计划**：

| 阶段 | 任务 |
|------|------|
| P0 ✅ | 数学正确性基准测试：`ts_mean(3)` → `[nan,nan,2,3,4]` |
| P1 ✅ | `Analyzer` 消费 `OperatorPolicy.lookback` 而不仅是 kwargs |
| P1 ✅ | 文档固定：样本 std vs 总体 std（`ts_std` ddof=1） |

---

### 4. 横截面算子按日独立 ✅

**适用**：rank/zscore/winsorize/neutralize 必须 `axis=1`（按行截面）。

**现状**：`cross_sectional.py`, `group_neutralization.py` 已逐行；前缀截断 rank 不变测试已加。

**计划**：

| 阶段 | 任务 |
|------|------|
| P0 ✅ | 测试：截断未来日期后，历史日截面 rank 不变（`test_cs_rank_prefix_invariant.py`） |
| P2 ✅ | 固定流水线顺序：缺失→winsorize→neutralize→zscore（`examples/profiles/factor_cs_pipeline.yaml`） |

---

### 5. 数据质量校验层 ✅

**适用**：产出后覆盖率、缺失率、inf、截面 std=0、主键唯一。

**现状**：`runtime/dq_gates.py` + profile + strict/input DQ + pipeline `--strict-dq` / `--input-dq`。

**计划**：

| 阶段 | 任务 | 文件 |
|------|------|------|
| P0 ✅ | `runtime/dq_gates.py`：可配置阈值 | |
| P0 ✅ | materialize 前可选 DQ；失败 non-zero | `storage/materializer.py` |
| P1 ✅ | pipeline CLI `--strict-dq` | `pipeline.py`, `run_pipeline.py` |
| P2 ✅ | 输入侧：run 前检查依赖列覆盖率 | `runtime/input_dq.py`, `runtime/engine.py` |

---

### 6. 性能：研究 vs 生产 ✅ 增量 + prefetch

**适用**：生产只算新 bar + lookback 缓冲。

**现状**：`run_incremental` / `materialize_incremental` + watermark + IO 下推 + plan 磁盘缓存。

**计划**：

| 阶段 | 任务 |
|------|------|
| P0 ✅ | `compute_history_buffer(lookback)` 工具函数 |
| P1 ✅ | `run_incremental` / `materialize_incremental` | `runtime/engine.py`, `runtime/incremental.py` |
| P1 ✅ | `narrow_data_source_for_window` IO 下推 | `storage/time_window.py` |
| P1 ✅ | pipeline `--incremental` / `--strict-dq` | `pipeline.py`, `run_pipeline.py` |
| P2 ✅ | DataSource date range prefetch | `KlineParquetSource.prefetch_columns` |
| P2 ✅ | 持久化 plan 子树缓存 | `storage/cache.py`, `storage/data_scope.py` |

---

### 7. 可复现 / 血缘 ✅

**适用**：记录 code/data/operator/parameter hash、run_id。

**现状**：`factor_run` + `RunLineage` + `operator_catalog_hash`；`factor_registry.data_source_json`；`resolve_git_commit_hash()` 写入 run extra。

**计划**：

| 阶段 | 任务 |
|------|------|
| P0 ✅ | `factor_run` 表 + `runtime/lineage.py` |
| P1 ✅ | 注册时写入 `data_source` JSON snapshot | `storage/catalog.py` |
| P2 ✅ | git commit hash 自动采集 | `runtime/lineage.py`, `runtime/engine.py` |

---

### 8. 测试标准 ✅ 五类门禁 + nightly

**五类测试映射**：

| 类型 | 本仓库 |
|------|--------|
| 数学正确性 | `test_operator_math_regression.py` ✅ |
| 边界 | `test_operator_boundary.py` + comprehensive smoke ✅ |
| 截面隔离 | `test_polars_cs_tier1.py` + cs 截断测试 ✅ |
| PIT | `test_causal_operators.py` + `test_mining_pit_audit.py` ✅ |
| 回归 | `test_golden_parquet_regression.py` + nightly ✅ |
| 契约 | `test_datasets_mining_alignment.py` + CI 脚本 ✅ |

---

### 9. 因子输出格式 ✅ 长表元数据 + 宽表 pivot

**适用**：模型/优化器需要 `is_valid`, `universe`, `factor_version`。

**现状**：长表含 `datetime, asset, value` + 可选 `factor_version`, `is_valid`, `data_snapshot_id`, `calc_time`；宽表 pivot API 可用。

---

## 实施优先级（总览）

```
Phase 0（本次）— 设计与基础设施
├── 本文档
├── operator_policy.py + catalog hash
├── dq_gates.py + 测试
├── lineage.py + catalog factor_run 表
└── lookback buffer 工具

Phase 1（已完成）— 生产安全
├── run_incremental + watermark 联动 ✅
├── Composite PIT 配置审计 + test_composite_pit ✅
├── 核心算子 math regression 基准 ✅
└── materialize strict DQ（pipeline --strict-dq）✅

Phase 2（已完成）— 平台化
├── 因子湖元数据列（calc_time/factor_version/data_snapshot_id/is_valid）✅
├── 输入 DQ（runtime/input_dq.py + engine/pipeline --input-dq）✅
├── lineage data_snapshot_id ✅
├── golden 回归 + 截面 rank 隔离测试 ✅
├── A股/美股 data_access 示例 YAML ✅
└── pipeline --max-retries 重试 ✅

Phase 3（已完成）— 深化
├── A 股/美股交易日历（storage/trading_calendar.py + incremental 接入）✅
├── 持久化 plan 子树缓存（PersistentPlanCache + data_scope）✅
└── 核心算子显式 OperatorPolicy（ts_mean/ts_std 等）✅

Phase 4（已完成）— 语义与血缘
├── Analyzer 消费 OperatorPolicy.lag ✅
├── lineage extra 写入 data_source_config snapshot ✅
├── 核心算子 policy 扩展（rank/neutralize/group_*）✅
├── 宽表 pivot API（factor_format + ResultStore.to_wide）✅
└── 生产示例 YAML（plan_cache_dir + materialize）✅

Phase 5 — 生产加固（已完成）
├── Analyzer 消费 OperatorPolicy.lookback_window ✅
├── data_source JSON 写入 factor_registry ✅
├── git commit hash 自动采集（lineage）✅
├── DataSource prefetch_columns 优化 ✅
└── data_access instrument_filter/schema 守卫 ✅

Phase 6 — 双市场深化（已完成）
├── 美股 instrument_column 与真实 parquet 契约测试 ✅
├── factor_engine universe/成分股 composite preset ✅
├── cos_mirror 与 datasets.yaml 双向一致性 ✅
└── scan_polars 路径 telemetry ✅

Phase 7 — 运维与可观测（已完成）
├── factor_run / audit 与 telemetry 指标导出 ✅
├── 生产 YAML 多环境 profile（dev/staging/prod）✅
├── 增量 materialize 断点续跑与失败分区隔离 ✅
└── 双市场 golden 数据集 CI 门禁 ✅

Phase 8 — 平台扩展（已完成）
├── factor_run 指标 Prometheus / OTLP 导出 ✅
├── pipeline 多 config 并行调度 ✅
├── 因子湖跨环境 publish 审批流 ✅
└── 全链路 data_snapshot_id 对账工具 ✅

Phase 9 — 生态集成（已完成）
├── OTLP gRPC 推送与 Grafana 仪表盘模板 ✅
├── pipeline 分布式任务队列（多机）✅
├── 因子湖版本 diff 与自动回滚 ✅
└── data_access audit ↔ factor_run 关联查询 ✅

Phase 10 — 生产规模化（已完成）
├── OTLP gRPC 原生推送（可选 opentelemetry SDK）✅
├── SLO 告警规则 runtime/slo_rules.py ✅
├── Redis / S3 对象存储任务队列 + enqueue/worker CLI ✅
├── 因子湖跨 region 复制 runtime/lake_replication.py ✅
├── Grafana 仪表盘 + datasource provisioning ✅
└── Nightly CI golden + Tier-1 causal + ClickHouse ✅
├── datasets.yaml ↔ mining/prod 契约 CI ✅
├── test_data_access_source 纳入 CI ✅
└── 真实 parquet optional nightly smoke ✅
├── RunWindow + 全量 auto_warmup/trim（`runtime/run_window.py` + `engine.run`）✅
├── 配置 run/dq/pit/materialization（`runtime/config.py` + `examples/profiles/prod.yaml`）✅
├── config_runtime 统一 pipeline / materialize_from_config 语义 ✅
├── DQ Profile YAML（`runtime/dq_profiles.yaml` + research/prod 阈值）✅
├── materializer：invalid_reason + staging 单写（staging 优先 upsert）✅
├── PIT enforce（`runtime/pit_audit.py` + compile 门禁）✅
├── CI 门禁（`.github/workflows/ci.yml` + enterprise_readiness）✅

Phase 12 — 算子语义与 intraday（已完成）
├── SQL 截面/分组/回归：normalize · group_normalize · group_percentile · group_decay_linear · cs_resid · cs_regression（58 算子 SQL 下推）✅
├── SQL 覆盖清单：`docs/sql_pushdown_coverage.md` CI 自动生成 ✅
├── SQL Arrow dtype：Decimal→object 统一 float64；`is_nan`/`is_finite` 三后端 0/1 float64 ✅
├── SQL 条件链路：where/if_else 浮点真值 + is_finite→where→rank 集成测 ✅
├── SQL clip：positional 边界 + NaN 保持 NULL ✅
├── SQL winsorize/group_winsorize：NaN 保持 NULL ✅
├── SQL 截面/分组语义：rank NaN 跳过 · group_rank/group_zscore 对齐 pandas ✅
├── Polars bfill 对齐因果语义（与 pandas/SQL 一致）✅
├── OperatorSpec Tier-1 显式 policy（55+ canonical + CI 门禁）✅
├── IntradayAggregator + intraday_daily 数据源 ✅
├── Analyzer 消费 min_periods → lookback ✅
├── effective_lookback 频率换算修正 + bar_freq 接线 ✅
├── micro 算子 9 个 ✅
├── write_target: staging_clickhouse 多目标物化 ✅
├── materialize_incremental + clickhouse/staging_clickhouse 双写 ✅
├── defer_watermark 两阶段提交 ✅
├── dual_write_reconcile + staging 行级补偿删除 ✅
├── intraday_tick_precise 增量扩窗 ✅
├── quantile/normalize Polars ✅
├── prod.yaml 默认 staging_clickhouse ✅
└── PIT 标签层 api/label_pit.py ✅

Phase 13 — 读端统一（已完成）
├── examples/configs 全量迁移 `data_access`（0 legacy type）✅
├── `us_stocks_sip_quotes/trades` + `massive_ticks` 参数化 `kind` ✅
├── fundamentals composite preset（balance_sheet/cash_flow/income_statement/floats）✅
├── `default_mining_data_source_presets()` 单点注册 ✅
└── CI：`test_example_yaml_migration` + datasets 契约门禁 ✅

Phase 14 — 企业级加固（已完成）
├── factor_schema 单点契约 + factor_lake/staging schema 对齐 ✅
├── production auto_warmup + lineage expression 修复 ✅
├── QueryBudget + read/sql 审计 + delete_rows dry_run ✅
├── composite join_reports → lineage ✅
├── dual_write_service / warmup_service / lineage_service / materialize_service ✅
├── FactorWriteTarget Protocol + LocalParquet / Staging / ClickHouse ✅
├── schema_migration + migrate_factor_lake_schema CLI ✅
├── sql_stream + apply_sql_row_limit（含 min 语义）✅
├── run_many_from_config 按 data_scope 分组 + materialize_many_from_config ✅
├── run_many/run_many_parallel/run_many_from_config 生产开关 ✅
├── CH 写后 verify_factor_write（QUANT_CH_VERIFY_WRITE）✅
└── SessionBarCalendar 分钟频精确 warmup ✅

Phase 15 — 批量编排与写目标收敛（已完成）
├── `config_run_batch_key`：同 data_scope 内按 run kwargs 子分组 ✅
├── `run_many_from_config`：生产开关一致时仍可 run_many（含 auto_warmup）✅
├── `ClickHouseWriteTarget.write_factor_series` + dual_write 统一 ✅
├── `storage` 公开 export write_targets ✅
└── `test_enterprise_p4` + CI `check_enterprise_docs` ✅

Phase 16 — 物化批量与配置映射（已完成）
├── `ResolvedMaterializeKwargs.to_engine_materialize_kwargs` / `to_run_kwargs` ✅
├── `config_materialize_batch_key` + `materialize_many_from_config(batch_run)` ✅
├── 同 scope 多因子共享 `run_many` + `execute_materialize_from_resolved` ✅
├── `materialize_from_config` 补全 resume/isolate 传参 ✅
├── `run_many_from_config(parallel=True)` + `run_many_from_config_parallel` ✅
└── `test_enterprise_p5` ✅

Phase 17 — pipeline 收敛与 allowlist（已完成）
├── `schema_migration.py` allowlist 登记 ✅
├── `ResolvedRunKwargs.to_run_kwargs` / `to_incremental_materialize_kwargs` ✅
├── `pipeline._execute_config` 统一 kwargs 映射 ✅
├── `run_config_directory` 委托 Engine 批量 API（`batched_engine`）✅
├── `from_config(profile=...)` + batch 方法 profile 传参 ✅
└── `test_pipeline_batch` + `INTERFACE.md` 更新 ✅

Phase 18 — Pipeline 覆盖与增量批量（已完成）
├── `PipelineConfigOverrides`：CLI 覆盖统一传入 batch resolve ✅
├── `config_*_batch_key(..., pipeline=...)` 按覆盖分组 ✅
├── `materialize_many_from_config_parallel` + batch 内 `run_many_parallel` ✅
├── `materialize_incremental_many_from_config` + pipeline incremental batch ✅
├── `write_targets.LocalParquet` lazy import `ParquetMaterializer` ✅
└── `test_enterprise_p6` + `test_pipeline_batch` 扩展 ✅

Phase 19 — 读路径优化与 run_many 生产化（已完成）
├── `run_many` / `run_many_parallel`：`input_dq_check` 走 batch 快路径 + CSE ✅
├── `DataSourceReadSession` + `DataAccessSource.column_cache_stats` ✅
├── `data_access/stats.py`：footer 行数估算 + `suggest_read_mode` ✅
├── `store.read_auto` / `dataset_read_stats`：arrow / stream / polars 自动路由 ✅
├── `storage/partition_policy.py`：year / year+month 分区策略 ✅
├── `ParquetMaterializer`：`storage_format=wide` + 可配置 `partition_columns` ✅
├── `factor_lake_wide` 数据集 + registry `partition_columns` / `storage_format` ✅
├── `ResultStore.load_factor_wide` 宽表 panel 直读快路径 ✅
├── `cache/` 分层缓存：`ExecutionCacheSession` + L0–L3 命中统计 ✅
├── `PolarsBackend` 真继承 + 时序 Polars / 截面 pandas 回退 ✅
├── `backend/polars_hot_ops.py` + runtime 计数 ✅
└── 相关测试 ✅

Phase 20 — 编排与性能门禁（已完成）
├── `planner/dependency_graph.py`：列重叠组 + 并行层 + `batch_graph` ✅
├── `FactorEngine.analyze_batch` / `run_many` 返回 `batch_graph` ✅
├── `runtime/shard_materialize.py`：`shard_factor_ids` + 统一 `shard_by_hash` ✅
├── `task_queue.shard_config_paths` 委托分片模块 ✅
├── `tests/perf/test_regression_gate.py`：compile/run_many smoke 门禁 ✅
├── `tests/test_dependency_graph_shard.py` ✅
├── 增量 by data event ⏳
└── 分布式物化 worker 编排 ⏳

Phase 21 — 减少重复 + 宽矩阵 + 分片物化（已完成）
├── `cache/column_cache.py` + `cache/cache_policy.py`：列缓存 scope key ✅
├── `ReadSession.prepare_batch` / `load_columns_once`；`run()` 统一读路径 ✅
├── `runtime/production_policy.py` + `engine.run_mode` 硬门禁 ✅
├── `planner/rolling_cache.py`：`run_many` 返回 rolling 共享摘要 ✅
├── `storage/factor_matrix_materializer.py` + `engine.materialize_matrix` ✅
├── `engine.materialize_sharded`：factor_id 哈希分片 + batch run_many ✅
├── `data_access/layout_policy.py`：bucket 剪枝骨架 ✅
└── `tests/test_phase21_platform.py` ✅

Phase 22 — 增量事件 + 读路径深化（已完成）
├── `factor_dependency` / `factor_column_dep` catalog 表 ✅
├── `runtime/incremental_scheduler.py`：DataEvent → FactorUpdatePlan ✅
├── 物化时自动 `record_factor_dependency` ✅
├── `engine.plan_incremental_from_event` ✅
├── registry `layout_policy` + bucket 路径/SQL 剪枝 ✅
├── `stats.py` sidecar + `scripts/refresh_dataset_stats.py` ✅
├── `read_auto` v2：sidecar + query_budget 超行数 → stream ✅
├── `datasets.yaml`：`factor_matrix` 数据集登记 ✅
└── `tests/test_phase22_platform.py` + `test_bucket_stats_phase22.py` ✅

Phase 23 — 事件驱动执行 + 算子代价（已完成）
├── `engine.materialize_incremental_from_event`（dry_run / 自动增量物化）✅
├── `execute_incremental_updates_from_event`：catalog expression 重建 Factor ✅
├── `backend/operator_cost.py` + `backend/routing.py` Tier 路由 ✅
├── `run_many` 返回 `plan_costs` 摘要 ✅
├── 分钟数据集 `layout_policy.bucket`（ashare / us_sip minute）✅
└── `tests/test_phase23_platform.py` ✅

Phase 11 — 生产地基（已完成）
```

---

## 明确不做 / 低优先级

- **Spark window 语义移植**：已有 pandas/polars，无需 Spark 层
- **每个算子手写 15 字段 YAML**：用推断 + 核心 override
- **引擎内建完整 asof join**：保持在 DataSource/Composite，避免双轨
- **79 个 stub 算子填实现**：与 enterprise 无关，继续禁止投递
- **36 个 intentional pandas-only**：FFT/矩阵分解/随机/CDF-PDF/constant；auto 路径可桥接其余算子

---

## 验收表（研发 Checklist）

复制此表用于 PR / 算子 PR：

- [ ] 算子是否 `pit_safe`？负 lag / bfill / 全样本广播已禁用？
- [ ] 横截面算子是否 `axis=1`？
- [ ] rolling 是否只用 `[t-window+1, t]`？
- [ ] 是否有 prefix-invariant 或手算单元测试？
- [ ] `OperatorPolicy` 是否可推断或显式声明？
- [ ] 落盘是否通过 DQ gate（覆盖率、缺失率、inf）？
- [ ] 是否记录 run lineage（ast_hash + operator_catalog_hash）？
- [ ] 增量生产是否预留 lookback buffer？
