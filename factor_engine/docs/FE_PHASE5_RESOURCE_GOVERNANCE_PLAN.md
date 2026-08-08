# FactorEngine 第五阶段计划：生产级资源治理 + 大规模执行稳定化（Phase 5）

> 依据：AI 对 FactorEngine 的第四轮审查（34 项工程任务 + P0-1~27 + P1-1~17 + P2）。
> 目标：让 FactorEngine 在不同配置服务器（8GB / 16GB / 32GB / 128GB、Docker/K8s/SLURM、
> 本地盘 / COS）上大规模自动因子挖掘时——不 OOM、不跨后端掩盖真实错误、
> 不数据版本缓存污染、不因 SourceRef 阻断原生执行。
> 全部改动只在服务器本地（`/home/shw/quant_projects/factor_engine` 与
> `/home/shw/quant_projects/dataaccess`），不 push GitHub。

---

## 一、现状结论（已确认不重复做的部分）

以下已被 DataAccess 0.8 + 最新 FactorEngine 解决，本次**不再重做**：

- production `run_many` 不再因 PIT 退化为逐因子 `run()`（编译期 `assert_pit_safe`）
- batch warmup 已合并共享加载窗口（`_maybe_prepare_batch_warmup`）
- DataAccess `period_selection=latest_period` / session calendar / A股分钟 UTC→上海 / 午休 bar index
- 一次 scan 多聚合（`aggregate_minute_bundle`）、US filing_date PIT 索引、Contract IR
- X0 / EMPTY / RAW_EVENT / E1/E2 表模型约束、grain/cardinality fan-out guard、跨市场歧义 fail-closed
- 查询缓存 / coverage / serving 分层
- FE 算子 evidence / capability、numerical operator 修复、plan-cost 路由 + offline benchmark baseline

---

## 二、P0 资源治理（Task #61）—— 最高优先级

### R1 真实资源发现 `ExecutionResourceManager`
新建 `runtime/resource_governor.py`：
- `effective_cpu_slots()`：env override → cgroup v2 `cpu.max` → sched affinity → host CPU
- `effective_memory_limit_bytes()`：显式配置 → cgroup v2 `memory.max` → cgroup v1
  `memory.limit_in_bytes` → SLURM `SLURM_MEM_PER_NODE` → RLIMIT_AS → host RAM
- `spill_disk_available()` / `spill_disk_speed_class()`
- `ExecutionResourcePlan`（frozen dataclass）：
  `effective_memory_limit/process_budget/reserve/duckdb_budget/data_cache_budget/
  cse_budget/panel_budget/result_budget/spill_budget/max_workers/duckdb_threads/
  polars_threads/io_concurrency`
- 默认分配：process ≈ 75% M；其中 DataAccess/DuckDB ≈ 45%、Factor/cache/CSE ≈ 30%、
  final result ≈ 10%、其他 ≈ 15%（可配置，非写死）

`runtime/execution_resources.py` 重构为委托 `resource_governor`，保留
`physical_cores()/resource_plan()/set_duckdb_max_threads()/apply_live_duckdb_threads()` 旧 API。

### R2 运行时 RSS Governor
`MemoryGovernor`：
- 每个 cache 层独立 budget，`reserve(owner, bytes)` / `release()` / `evict()`
- 运行时 RSS 分档：<70% 正常 / 70-80% 停预热 / 80-85% 逐出 LRU / 85-90% spill 冷 CSE /
  90-95% 降并发拆 batch / >95% fail-fast `ResourceBudgetExceeded`
- 禁止把 `gc.collect()` 当治理手段

### R3 worker 数 CPU×RAM 双约束
`max_workers = min(cpu_allowed, floor(available_execution_memory / per_worker_peak))`
接入 `run_many_parallel`。

### R4 run_many 结果流（Task #62）
- `run_many(result_policy="return"|"yield"|"sink"|"materialize")` + `run_many_iter`
- `sink`：每算完一个 → DQ → 落盘 → 释放 result → 下一个（生产挖因子推荐）

### R5 CSE 引用计数（Task #62）
- `apply_cse` 输出每个 sid 的 consumer_count；`_materialize_shared_subplan` 后
  root 每消费一次 refcount--，为 0 立即 evict；`CSEPolicy(lazy/memory/spill/recompute)`

### R6 统一缓存预算（Task #63）
- `GlobalMemoryGovernor` 统一 ColumnCache / PanelCache / CSECache / PlanCache /
  SQL 物化缓存 / ResultBuffer；所有 `put()` 走 `reserve→evict/spill→insert`
- `cache/expression_cache.py` `cache/panel_cache.py` `storage/cache.py(CacheManager/
  PersistentPlanCache)` 接入 governor + 字节 LRU

### R7 DataAccessSource 缓存预算 + 字节计算（Task #63）
- `FACTOR_ENGINE_DATA_CACHE_MAX_BYTES` 默认 8GB → `process_budget × 0.15`（env 仍可覆盖）
- `_series_bytes`：Series `memory_usage(index=True,deep=True)`、
  DataFrame `memory_usage(index=True,deep=True).sum()`、ndarray/Arrow `nbytes`、
  polars `estimated_size()`；RSS calibration factor

### R8 LazyColumnBundle 只持 Lazy Plan（Task #63）
- 去掉 `_materialized` 永久 Series 缓存 → 改为受 governor 约束的有界 LRU；
  materialized 列只由 DataAccessSource `_column_cache` 拥有

### R9 Plan Cache 数据版本污染（Task #64）
- `compute_data_scope` 已含 `data_snapshot_id`；补强：`CacheManager/PersistentPlanCache`
  加字节治理 + 显式 `data_snapshot_id` 校验；snapshot 更新后旧 factor result cache 绝不命中
  （`refresh_snapshot` 已有 manifest-token 失效，扩展覆盖 plan cache 作用域）

### R10 分钟聚合全面接入 DataAccess AggregationBundle（Task #64）
- `minute_at/minute_range/minute_bar/minute_resample/VWAP/多窗口/多字段` 统一编译成
  `aggregate_minute_bundle`；VWAP 的 Amount/Volume **同一 scan** 聚合
- minute SourceRef 显式继承 market/session/timezone；production 禁止把完整多年分钟拉进 Pandas

### R11 严格异常分类禁止危险 fallback（Task #64）
- 新异常：`CapabilityMiss / CompilationUnsupported / ResourceBudgetExceeded /
  DeadlineExceeded / OutOfMemory / SemanticContractError / PITViolation / SchemaError /
  DataQualityError / TransientIOError / BackendExecutionError`
- 仅 CapabilityMiss / CompilationUnsupported 允许跨 backend fallback；
  检查并清除 SQL/minute pushdown 中的 `except Exception → fallback pandas`；production fail-closed

### R12 SourceRef 编译为 DataRequest（Task #64，本轮开始落地）
- SourceRef 是 Logical Data Requirement，不是 Pandas boundary；含 SourceRef 的因子
  逐步可走 SQL / Polars-long / hybrid 原生执行

### R13 消除跨包私有 API（Task #64）
- FE 不再访问 `ScanHandle._lf` / `._conn` / `._engine`；DataAccess 增加正式
  `ScanHandle.lazyframe_native()`（composition-only，collect 仍走 budget）
- `backend/polars_lazy.py:35` 改为走正式接口

### R14 one-to-many SourceRef fail-closed（Task #64）
- TopTen 等 relation 表无 rank selector / aggregate 时 production 拒绝标量化

### R15 并发共享 mutable state 收口（Task #65）
- `ExecutionResourceScope` context manager：设置资源 → 执行 → finally 恢复
- run_many_parallel 内 PRAGMA threads 等动态修改包裹，结束恢复原值

### R16 panel_native 减内存（Task #65）
- template 只保存 axis/ordering metadata + validity mask，不保留完整 factor Series

### R17 流式物化 + materialize_many 流式（Task #65）
- 去掉 `list(iter_partition_groups())`；partition-at-a-time：新分区直接写，
  overlap 才 read+merge；每分区完成立即释放
- `materialize_many`：compile → wave → DQ → write → release → next

### R18 最终结果字节预算（Task #65）
- `result_budget` 进 ResourcePlan；超限：production 强制 sink/stream/materialize，
  research 告警

---

## 三、P1 治理/配置/多市场（Task #66）

- **P1-1 Router 峰值内存**：`plan_cost_router` 候选 backend 估计 peak memory >
  budget 时移出候选；`operator_cost` 增加 `peak_memory` 估计
- **P1-2 benchmark fingerprint**：加 CPU model / effective cores / RAM class /
  local-remote storage / NVMe-COS / worker config；无匹配 baseline → estimated
- **P1-3 cgroup CPU**：`effective_cpu_slots()` 支持 `cpu.max` quota
- **P1-4 warmup 按成本聚类**：short/medium/long/full-history 分 wave，不互相拖累
- **P1-5 Input DQ 下推**：null/finite/min/max/duplicate/coverage 在 DataAccess/DuckDB
  聚合层完成，不把输入 materialize 成 Pandas
- **P1-6 native/streaming result**：`result_mode=pandas|arrow|polars|lazy|stream|sink`
- **P1-7 ClickHouse 流式**：Arrow RecordBatch streaming，不全量返回后再 `.lazy()`
- **P1-8 合并 LQTP legacy/v2**：`lqtp_logical_source.py` → 薄 shim 委托 v2
- **P1-9 anchor index-only**：`_anchor_index()` 走 index-only scan，清理 broad except
- **P1-10 production managed data 统一 DataAccess**：raw ParquetSource 仅 research
  legacy；绕过 DataAccess 需显式 allowlist
- **P1-11 DerivedField 进 DAG**（探索性，非阻塞）
- **P1-12 Matrix 按因子列分块**：date partition × factor block，块大小由 governor 决定
- **P1-13 spill disk governance**：执行前检查 temp dir 可写/free space/quota；
  DuckDB `temp_directory` / `max_temp_directory_size`
- **P1-14 统一 YAML `resources:` 配置**：memory limit/fraction/reserve、workers、
  DuckDB threads/memory、cache fractions、result limit、spill、admission；
  env override；production 非法配置启动失败不静默忽略
- **P1-15 统一 telemetry**：RSS start/peak/end、各 cache bytes、CSE live/peak、
  result bytes、spill、evictions、throttles、files/bytes scanned、backend route、
  fallback reason → runtime audit / lineage
- **P1-16 CI gate**：`make audit-factor-engine`（FE 全量 + DA 全量 + 语义/PIT/parity/
  SourceRef/minute/financial/incremental/resource/OOM-injection/concurrency/cache 失效/
  benchmark 回归）

---

## 四、实施顺序

1. `runtime/resource_errors.py` + `runtime/resource_governor.py`（R1-R3、P1-3、P1-13）
2. `PerfConfig` / `EngineConfig` 资源字段 + YAML `resources:`（P1-14）
3. `cache/` 全层接 governor（R6）、`data_access_source.py` 字节修复 + 预算（R7）、
   `polars_lazy.py` LazyColumnBundle（R8）
4. `execution_resources.py` 委托重构 + `ExecutionResourceScope`（R15）
5. `planner/cse.py` consumer count + `batch_service.py` refcount（R5）、
   `run_many` result_policy + `run_many_iter`（R4）、result budget（R18）
6. `materializer.py` 流式（R17）、`engine.py materialize_many` 流式（R17）、
   `panel_native.py` template metadata（R16）
7. `plan_cost_router.py` + `operator_cost.py` peak memory（P1-1）、
   benchmark fingerprint（P1-2）、warmup clustering（P1-4）、
   DQ 下推（P1-5）、ClickHouse streaming（P1-7）、LQTP shim（P1-8）、
   anchor index（P1-9）、matrix blocks（P1-12）、telemetry（P1-15）
8. SourceRef/minute/financial（R10/R12/R14）、异常分级清理 fallback（R11）、
   `_lf` 正式接口（R13）、版本污染补强（R9）
9. 测试 + `make audit-factor-engine`（P1-16）+ 全量回归

## 完成标准（对照审查 §九 的 20 条）
1. `resources.mode=auto` 不错配容器资源 2. 任何 DataAccess I/O 前已建 ExecutionResourcePlan
3. QueryBudget 覆盖 prefetch/SourceRef/SQL/minute/financial 4. Resource/OOM/Deadline/PIT/
Schema 错误绝不跨 backend fallback 5. 1000+ 因子物化最终结果不必全部 resident
6. CSE 按 last-use 回收 7. 所有缓存统一受 global budget 8. snapshot 更新后旧 result cache
绝不命中 9. 财务 SourceRef 不再 `(None,end)→全历史 Pandas` 10. 财务生产 PIT 走 DataAccess
latest-period serving 11. minute SourceRef 显式带 market/session + 一次 scan
12. minute_bar/resample 不再默认 full-Pandas 13. SourceRef 逐步进 DataRequest / native 执行
14. FE 不访问 `._lf` 15. one-to-many 无 selector/aggregate 就 fail-closed
16. 不同内存服务器自动调整 workers/cache/DuckDB/spill 17. DuckDB memory_limit 受 process
budget 约束 18. spill disk 有 quota + free-space guard 19. materialize 流式按 partition 写
20. CI 全绿

## 状态（2026-08-08）
- 实施中：见 Task #60-#66。
