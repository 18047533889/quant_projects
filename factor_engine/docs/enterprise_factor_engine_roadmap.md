# 企业级因子引擎完善路线图

> 对照外部「私募级因子引擎」九类标准，结合 **本仓库现状**（`factor_engine` + `data_access` + cleaned_operators）制定的可执行计划。  
> 原则：**能落地的先做，不照搬与架构不符的建议**。

---

## 现状摘要（2026-07）

| 能力 | 现状 | 评级 |
|------|------|------|
| 算子数量 | 441 implemented + DSL 白名单 ~512 | ✅ |
| 算子层因果 | `_causal.py` + 多轮修复 + `test_causal_operators.py` | 🟡 部分覆盖 |
| 数据层 PIT | `CompositeDataSource.merge_asof(backward)`；引擎**不**代做 PiT | 🟡 配置驱动 |
| lookback 分析 | `Analyzer.lookback` + `effective_lookback` + `WindowedDataSource` 裁剪加载 | 🟡 |
| 增量计算 | `run_incremental` / `materialize_incremental` + watermark + lookback 缓冲 + 交易日历 ✅ | 🟡 |
| 血缘/版本 | `factor_run` + `RunLineage` + operator_catalog_hash ✅ | 🟡 |
| DQ Gate | `dq_gates.py` + materialize/pipeline `--strict-dq` ✅ | 🟡 |
| 输出格式 | 长表 + 元数据列 + 宽表 pivot 读 API | 🟡 |
| 测试 | golden 回归 + 输入/输出 DQ + causal 专项 | 🟡 |

---

## ChatGPT 九类建议 → 本仓库映射

### 1. Point-in-Time（最高优先级）✅ 部分已做，继续加强

**适用**：全部因子；**分工**：

- **数据层**（主战场）：`publish_time` / `available_time`、universe、行业、成分股、停牌 → `CompositeDataSource` + `data_access` YAML
- **算子层**（已做一轮）：禁止 lead/bfill/全样本广播/双向滤波 → `_causal.py`
- **标签层**（未做）：forward return 与特征窗口不重叠 → 需在 mining/backtest 配置中固定

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

### 2. 算子定义标准化 🟡 从 metadata 扩展到 policy

**适用**：企业级需要统一 window/lag/nan_policy，避免「同名不同义」。

**现状**：`OperatorMetadata` 仅有 name/category/description/param_names。

**不适用**：要求每个算子 YAML 手写 15 字段——441 个算子成本过高；采用 **推断 defaults + 关键算子显式 override**。

**计划**：

| 阶段 | 任务 |
|------|------|
| P0 ✅ | 新增 `OperatorPolicy`：`scope`, `lookback`, `min_periods`, `lag`, `pit_safe`, `nan_policy` |
| P0 ✅ | `infer_operator_policy(op)` 按 category/tags 推断 |
| P0 ✅ | `compute_operator_catalog_hash()` 用于 lineage |
| P1 | 核心 50 算子显式声明 policy（ts_mean, rank, neutralize…） |
| P2 | policy 写入 `operators_catalog.md` 自动生成 |

---

### 3. 时间序列算子窗口边界 ✅ 大部分已符合

**适用**：rolling 仅历史、min_periods 固定、delay 先于 rolling（文档约定）。

**现状**：pandas rolling/expanding；已修复 convolve/decimate/FFT 等。

**计划**：

| 阶段 | 任务 |
|------|------|
| P0 | 数学正确性基准测试：`ts_mean(3)` → `[nan,nan,2,3,4]` |
| P1 | `Analyzer` 消费 `OperatorPolicy.lookback` 而不仅是 kwargs |
| P1 | 文档固定：样本 std vs 总体 std（`ts_std` ddof=1） |

---

### 4. 横截面算子按日独立 ✅ 已符合

**适用**：rank/zscore/winsorize/neutralize 必须 `axis=1`（按行截面）。

**现状**：`cross_sectional.py`, `group_neutralization.py` 已逐行；`normalize`/`standardize` 已改为 axis=1。

**计划**：

| 阶段 | 任务 |
|------|------|
| P0 | 测试：截断未来日期后，历史日截面 rank 不变 |
| P2 | 固定流水线顺序：缺失→winsorize→neutralize→zscore（配置模板，非强制引擎） |

---

### 5. 数据质量校验层 🔴 新建

**适用**：产出后覆盖率、缺失率、inf、截面 std=0、主键唯一。

**不适用**：完整 Great Expectations 集成（过重）；先做 **轻量 Python gate**。

**计划**：

| 阶段 | 任务 | 文件 |
|------|------|------|
| P0 ✅ | `runtime/dq_gates.py`：可配置阈值 | |
| P0 ✅ | materialize 前可选 DQ；失败 non-zero | `storage/materializer.py` |
| P1 | pipeline CLI `--strict-dq` | `pipeline.py` |
| P2 ✅ | 输入侧：run 前检查依赖列覆盖率 | `runtime/input_dq.py`, `runtime/engine.py` |

---

### 6. 性能：研究 vs 生产 🔴 增量计算缺失

**适用**：生产只算新 bar + lookback 缓冲。

**现状**：`ParquetMaterializer` upsert；`FactorEngine.run()` 全量。

**计划**：

| 阶段 | 任务 |
|------|------|
| P0 ✅ | `compute_history_buffer(lookback)` 工具函数 |
| P1 ✅ | `run_incremental` / `materialize_incremental`：watermark → lookback 窗口 → upsert | `runtime/engine.py`, `runtime/incremental.py` |
| P1 ✅ | `narrow_data_source_for_window` IO 下推（Kline/DataAccess/Composite/Parquet） | `storage/time_window.py` |
| P1 ✅ | 增量 fresh_cache：避免子计划 cache 与全量混用 | `runtime/engine.py` |
| P1 ✅ | tail 重算默认 = analysis_lookback+1（非整个 load 缓冲） | `runtime/incremental.py` |
| P1 ✅ | pipeline `--incremental` / `--strict-dq` | `pipeline.py`, `run_pipeline.py` |
| P2 | DataSource 原生 date range prefetch（减少 IO） |
| P2 ✅ | 持久化 plan 子树缓存（磁盘 parquet + data_scope 隔离） | `storage/cache.py`, `storage/data_scope.py` |

---

### 7. 可复现 / 血缘 🟡 扩展 catalog

**适用**：记录 code/data/operator/parameter hash、run_id。

**现状**：`ast_hash` 仅含 IR 结构，**不含**算子实现版本。

**计划**：

| 阶段 | 任务 |
|------|------|
| P0 ✅ | `factor_run` 表：run_id, ast_hash, operator_catalog_hash, row_count, dq_passed |
| P0 ✅ | `runtime/lineage.py` 构建 RunLineage |
| P1 | 注册时写入 `data_source` JSON snapshot |
| P2 | git commit hash 自动采集 |

---

### 8. 测试标准 🟡 加强中

**五类测试映射**：

| 类型 | 本仓库 |
|------|--------|
| 数学正确性 | 新增 `test_operator_math_regression.py` |
| 边界 | comprehensive smoke + 待补 inf/短窗 |
| 截面隔离 | 新增 cs 截断测试 |
| PIT | `test_causal_operators.py` ✅ |
| 回归 | 待建 golden parquet 基准 |

---

### 9. 因子输出格式 🟡 长表已有，缺元数据列

**适用**：模型/优化器需要 `is_valid`, `universe`, `factor_version`。

**现状**：`datetime, asset, value`。

**计划**：

| 阶段 | 任务 |
|------|------|
| P2 ✅ | 长表可选列：`factor_version`, `is_valid`, `data_snapshot_id`, `calc_time` | `storage/materializer.py` |
| P2 ✅ | 宽表 pivot API（`storage/factor_format.py`, `ResultStore.to_wide`） | `storage/result_store.py` |

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

Phase 10 — 生产规模化（待定）
├── OTLP gRPC SDK 原生推送与 SLO 告警规则
├── 多队列编排（Redis/对象存储任务中心）
├── 因子湖跨 region 复制与灾备演练
└── 统一观测平面（Grafana + Loki + Tempo）

Phase 11 — 生产地基（已完成）
├── RunWindow + 全量 auto_warmup/trim（`runtime/run_window.py` + `engine.run`）✅
├── 配置 run/dq/pit/materialization（`runtime/config.py` + `examples/profiles/prod.yaml`）✅
├── config_runtime 统一 pipeline / materialize_from_config 语义 ✅
├── DQ Profile YAML（`runtime/dq_profiles.yaml` + research/prod 阈值）✅
├── materializer：invalid_reason + staging 单写（staging 优先 upsert）✅
├── PIT enforce（`runtime/pit_audit.py` + compile 门禁）✅
├── CI 门禁（`.github/workflows/ci.yml` + enterprise_readiness）✅

Phase 12 — 算子语义与 intraday（进行中 → 核心已落地）
├── OperatorSpec Tier-1 显式 policy（55+ canonical + CI 门禁）✅
├── IntradayAggregator + intraday_daily 数据源 ✅
├── Analyzer 消费 min_periods → lookback ✅
├── effective_lookback 频率换算修正 + bar_freq 接线 ✅
├── micro 算子 7 个（realized_vol/spread/amihud/mid_return/bipower/jump）✅
├── write_target: staging_clickhouse 多目标物化 ✅
├── materialize_incremental + clickhouse/staging_clickhouse 双写 ✅
├── materialize_incremental_from_config + incremental CLI/config ✅
├── DualWriteError：CH 失败时携带 partial summary ✅
├── factor_frame 共享 Parquet/CH 清洗路径 ✅
├── incremental 日内频率换算（bars_to_calendar_trading_days）✅
├── run_incremental 全量路径 auto_warmup ✅
├── defer_watermark 两阶段提交（CH 成功后再更新 watermark）✅
├── sync_local_factor_to_staging 原子目录替换 ✅
├── micro_trade_imbalance / micro_vpin 算子（pandas + polars）✅
├── ClickHouse docker-compose 本地集成模板 ✅
├── dual_write_reconcile + catalog.list_dual_write_failures ✅
├── 日内增量 window_mode（intraday_calendar_approx）✅
├── prod_clickhouse profile ✅
├── ClickHouse + config_runtime 对齐 ✅
├── DQ + preserve_invalid_rows 语义一致 ✅
├── pipeline CLI：--write-target / --preserve-invalid-rows ✅
└── 其余 micro_* stub / bar 精确 tick 级扩窗（待做）
```

---

## 明确不做 / 低优先级

- **Spark window 语义移植**：已有 pandas/polars，无需 Spark 层
- **每个算子手写 15 字段 YAML**：用推断 + 核心 override
- **引擎内建完整 asof join**：保持在 DataSource/Composite，避免双轨
- **79 个 stub 算子填实现**：与 enterprise 无关，继续禁止投递

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
