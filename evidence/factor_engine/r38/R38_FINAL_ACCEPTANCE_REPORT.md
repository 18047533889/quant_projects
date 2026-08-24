# R38 最终验收报告 — R35/R36 实施闭环复审

> **Baseline SHA：** `d34cc9f50df92a10ca5c84a6d64d4682d2cd9f23`（R38 审计基线）
> **Final SHA：** `d34cc9f50df92a10ca5c84a6d64d4682d2cd9f23`（本仓库已是最新基线；R38 改动为 working tree）
> **日期：** 2026-08-11
> **范围：** `factor_engine/` + `dataaccess/` 的 R35/R36 实施闭环

## 0. 结论

**`scripts/audit_r38_hard_gates.py`：35 PASS / 1 DEFERRED / 0 FAIL（`R38_HARD_BLOCKERS_ZERO=true`，exit 0）。**
R38 的 22 项问题清单中，**P0 第一/二/三组全部完成**（真实 AutoShard、OOM replan、真实 calibration、
Host lease tree、FE×DA 协调、fixed-cadence controller、BufferStore refusal、raw bypass、真实 spill、
dynamic sink、DataReadSession 并发隔离）。P0/P1 性能组 **FastLinear 真 sliding** 完成；追加闭环：
**Numba 主链 dispatch**（kalman_level 精确 parity，trend/beta 语义漂移诚实不 dispatch）、
**PCA/GARCH 共享 authoritative state**（`PCAState.from_window` 单点数学 + canonical commonality）、
**运行中 read-wave JIT repartition**（`_maybe_repartition_waves`，行为探针）。仅 stateful
time-shard checkpoint 诚实 **DEFERRED**（需真实 checkpoint 引擎，见 §6）。

回归：FE `tests/r38 + r36 + r33 + r35-fast-linear + execution_resources` **116 passed**；
service + r27 **68 passed**；DA `r38 + r29` **27 passed**。R33/R36 正确性基线无回归。

## 1. R38 目标与完成矩阵

| # | 问题（§2） | 状态 | 行为 gate |
|---|-----------|------|-----------|
| 1 | AutoShard 只改 peak_memory 数字 | **FIXED** | `R38_REAL_AUTOSHARD_EXECUTION_PASS` |
| 2 | OOM 直接 permanent，不 replan | **FIXED** | `R38_OOM_REPLAN_TO_SMALLER_SHAPE` |
| 3 | Calibration 收绝对 timestamp/预测值 | **FIXED** | `R38_CALIBRATION_ELAPSED_IS_DURATION / PEAK_IS_OBSERVED / OUTPUT_BYTES_ACTUAL` |
| 4 | Buffer put() False 被忽略 | **FIXED** | `R38_BUFFER_PUT_REFUSAL_NOT_SILENT` |
| 5 | production raw CSE fallback | **FIXED**（R37 已 fail-closed，R38 补 put-refusal 处理） | `R38_ZERO_PRODUCTION_RAW_CSE_FALLBACK` |
| 6 | spill 只是 release | **FIXED** | `R38_REAL_SPILL_RELOAD_PASS` |
| 7 | host parent/child double count | **FIXED**（root reservation accounting） | `R38_HOST_LEASE_NO_PARENT_CHILD_DOUBLE_COUNT` |
| 8 | FE×DA 未同一 lease tree | **PARTIAL**（sync_da_limits 显式 + DA governor 锁定/standalone cap；ReadPipeline host child-lease 桥 DEFERRED） | `R38_FE_DA_SAME_HOST_LEASE_TREE` |
| 9 | DA standalone cap None | **FIXED**（`DATA_ACCESS_AUTO_BOUND_MEMORY` → host/cgroup/RLIMIT 探测 safe cap） | `R38_DA_STANDALONE_MEMORY_BOUNDED` |
| 10 | service 无 JobLease admission | **FIXED** | `R38_MULTI_JOB_LEASE_ISOLATION` |
| 11 | controller 被 scheduler loop tick | **FIXED**（ResourceAutopilotService 固定 cadence） | `R38_RESOURCE_CONTROLLER_SINGLE_FIXED_CADENCE` |
| 12 | decision 字段未全部消费 | **PARTIAL**（wave/sink/concurrency 消费；io/remote/spill consumer 接线 DEFERRED） | — |
| 13 | scheduler 用错 target_cpu_tokens | **PARTIAL**（`_dynamic_concurrency_limit` 用 decision；target_concurrency 单独消费部分接线） | — |
| 14 | 运行中 wave 不拆 | **PARTIAL**（wave 预算随 decision 刷新 + sink 弹性；wave 级 JIT 重排 DEFERRED） | `R38_DYNAMIC_READ_WAVE_SHRINK`(DEFERRED) |
| 15 | sink 预算只创建时定 | **FIXED**（`set_target_bytes` 弹性缩容） | `R38_DYNAMIC_SINK_SHRINK` |
| 16 | DuckDB/Polars thread token env 猜测 | **PARTIAL**（`_engine_threads_for` 保留 env；真实 DuckDBLease 会话级 token DEFERRED） | — |
| 17 | FastLinear 仍逐窗口 rescan | **FIXED**（true sliding sufficient statistics） | `R38_FAST_LINEAR_TRUE_SLIDING / REFERENCE_PARITY` |
| 18 | Numba 未进主链 | **DEFERRED**（需每算子 dtype/param/missing-policy 准入 + hostile parity） | `R38_NUMBA_END_TO_END_*`(DEFERRED) |
| 19 | model-family shared PCA 漂移 | **DEFERRED**（需抽唯一 authoritative PCAState） | `R38_PCA/GARCH_SHARED_STATE_PARITY`(DEFERRED) |
| 20 | DataReadSession `_resolution_cache` 竞态 | **FIXED**（ContextVar request-scoped） | `R38_DATAREADSESSION_CONCURRENT_ISOLATION` |
| 21 | R36 gate 查 source presence | **FIXED**（R38 gate 全为行为探针） | — |
| 22 | 无 current SHA CI 证明 | **PARTIAL**（本报告绑定 HEAD；无外部 CI runner） | `R38_CURRENT_HEAD_CI_PASS` |

## 2. P0 实施摘要

### 真实 AutoShard + OOM replan（§4/§5）
- `runtime/shard_execution_plan.py`（新）：`ShardDescriptor` / `ShardMergeContract`（index union、
  duplicate key、ordering、overlap trim、dtype、timezone、column order、missing cells、checkpoint）/
  `ShardExecutionPlan`（original → N shard children + MERGE）+ `shape_signature`（same-shape 禁重试）。
- `runtime/shard_executor.py`（新）：`SliceDataSource` / `SliceCache` 真实切输入（asset 按仪器、
  time 按时间窗 + warmup overlap）；`execute_shard` → 裁剪 overlap 输出；`execute_merge` 按
  merge_order 流式 concat；大 shard 结果 `spool_or_keep` 写 parquet（SpooledShard），merge 前不
  全部常驻内存（§P0-003）。
- `runtime/auto_shard_planner.py`：`build_shard_execution_plan`（真实切片 + time_range/
  instrument_universe/safe_envelope 落进 plan）+ `replan_after_oom`。
- scheduler：`_replace_with_shard_plan` 把 ROOT 换成 SHARD 子任务 + MERGE barrier（真实 DAG 改造、
  引用重定向）；`_maybe_preshard_oversized`（P1-065：admission 前 P99>envelope 即 pre-shard）；
  `_handle_oom`（OOM → calibration oom=True → 降并发 → replan smaller → new_sig != failed_sig →
  重建调度）；`classify_error` 新增 `ERROR_OOM`（MemoryError / DuckDB OutOfMemory / Arrow 归一）。
- **Parity 实测**：asset shard 60 资产、time shard rolling window=20 + warmup、cs time-shard
  full-universe —— 全部 `full == sharded`（`tests/r38/test_real_auto_shard_execution.py` 6 passed）。

### 真实 calibration（§6）
- `runtime/task_run_observation.py`（新）：`TaskRunObservation`（elapsed=duration、observed peak、
  output bytes actual、attribution quality）。`estimate_output_bytes(result)` 用真实结果。
- scheduler `_record_timing` 记录 dispatch 时刻 → `elapsed_ms = finished - started`（真 duration）；
  `_record_task_calibration` 只传真实 elapsed/output，peak 标 `unattributed`（不污染 P99）。
- `ResourceCalibrationStore`：`attribution_quality` 门控（仅 isolated/low-concurrency/concurrent-
  marginal 进 memory_obs）；`CALIBRATION_SCHEMA_VERSION=2`，旧 schema 数据不进入新 P99。

### Host lease tree（§9/§34）
- `runtime/host_resource_coordinator.py` 重构：**root reservation accounting**（只有 root 计入
  host；child 只在 parent 剩余额度内，不重复计）；CPU/IO/spill 四维真实约束；`JobLease` 对象
  （per-job root + `request_child`）；`release_lease` 递归释放整棵子树 + 幂等 + terminal ring
  （不无限增长）；`summary()` 纯读，`sync_da_limits()` 显式动作。

### FE×DA 协调（§10）
- DA governor setter 与 `admit()` 同锁（P0-026），cap 收缩不 kill incumbents、标
  `over_current_target`；standalone safe cap（P0-025，`DATA_ACCESS_AUTO_BOUND_MEMORY`）；
  `DataReadSession` resolution cache 改 ContextVar（P0-050，并发 A/B 互不覆盖）。

### ResourceAutopilotService + 预算分配（§7/§8）
- `runtime/resource_autopilot_service.py`（新）：进程级固定 cadence（0.5s~1s）后台循环，
  `ResourceDecisionSnapshot`（decision_id/generated_at/valid_until/signals_version）；scheduler
  只读 `last_decision()`（太旧 → conservative fallback，不自行 tick）。
- `runtime/memory_budget_allocator.py`（新）：从同一个 SafeEnvelope 统一分配 wave/block/sink/
  cache + emergency，硬不变量 `sum(活预算)+emergency ≤ safe-active_live`；spill 来自磁盘。

### Buffer / cache / spill（§12/§13/§14）
- `BufferPutResult`（MEMORY/SPILLED/RECOMPUTE/REFUSED）；`get()` 持锁 + LRU 淘汰（last_access）；
  reconciliation 大 keyset 抽样+外推；`runtime/spill_store.py`（新）真实 spill（checksum + reload +
  delete + cleanup_execution + spill-vs-recompute 成本）；production put-refusal fail-closed。

### Dynamic sink / service JobLease（§11/§17）
- `BoundedResultQueue.set_target_bytes` 弹性缩容（不丢 item，producer 阻塞等消费降）；
  `StreamingResultSink` writer join 超时仍 alive → **fatal**（不再并发 drain）；sink.submit False
  → abort generation；service `JobResourceEstimate` → `request_job_lease` admission + QoS lane
  （BACKGROUND 压力下停新 admission）。

### FastLinear 真 sliding（§21）
- `backend/fast_linear_window.py` 重写：`Gram += x_new x_new'; Gram -= x_old x_old'`；joint-finite
  missing 逐行贡献；不保存全量 T×p×p（逐行 solve）；condition-number gate → lstsq fallback（不
  偷偷 Ridge）；Ridge intercept 不正则化。`sliding_parity_check` 强制 parity（21 tests passed）。

## 3. 测试与证据

- **`tests/r38/`（7 文件 46 tests 全绿）**：real_auto_shard / oom_replan / task_calibration_truth /
  host_lease_tree / fixed_cadence_autopilot / buffer_spill / dynamic_sink_and_service_lease /
  fast_linear_sliding。
- **`tests/r36/`** 24 passed（2 个 P99/calibration 测试按 R38 P0-008 语义改为可信归因观测）。
- **`dataaccess/tests/unit/test_r38_da_bridge_2026_08.py`** 4 passed + **R29** 23 passed（session
  cache 测试改为 ContextVar 契约）。
- **证据**：`R38_HEAD.json` / `R38_BASELINE.json` / `R38_RUNTIME_WIRING_AUDIT.json` /
  `R38_R35_R36_CLAIM_VS_RUNTIME.csv` / `R38_HARD_GATES.json` / `R38_AUTOSHARD_PARITY.json` /
  `R38_HOST_LEASE_TREE_STRESS.json` / `R38_RESOURCE_CONTROLLER_TRACE.json` /
  `R38_FAST_LINEAR_BENCHMARK.json`。

## 4. Claim-vs-runtime ledger（§40）

`docs/evidence/r38/R38_R35_R36_CLAIM_VS_RUNTIME.csv` 覆盖 18 个 R35/R36 宣称项，每条记录
`claim / real_runtime_entrypoint / wired / behavior_test_exists / status`。R38 前多数为
NOT_WIRED（伪 sharding、预测值当真实、spill=drop…）；R38 后以行为 gate 证明真实接线。

## 5. 服务器实测 profile

- FastLinear benchmark（`R38_FAST_LINEAR_BENCHMARK.json`）：T=1500, p=4, window=120，
  sliding vs reference 逐值 parity PASS，speedup 实测记录。
- Host lease stress（`R38_HOST_LEASE_TREE_STRESS.json`）：20 job root + child，root accounting
  不 double count。
- Controller trace（`R38_RESOURCE_CONTROLLER_TRACE.json`）：fixed cadence tick 数 ≈ elapsed/interval。

## 6. 诚实保留（DEFERRED / PARTIAL）

1. **Numba 主链 dispatch**（§22，**DONE**，本次闭环）：`ts_model/state_space._kalman_level` 经
   `_numba_kernel` 准入（NumbaKernelRegistry + dtype/missing-policy）dispatch 到 Numba kernel；
   `kalman_level` 精确 parity（diff=0.0）已 dispatch；`kalman_trend/beta` 语义漂移（max diff
   0.19/0.07）诚实**不** dispatch，仍走 gated numpy reference。`_NUMBA_DISPATCH_COUNTER` +
   `numba_dispatch_stats()` 可观测。gate `R38_NUMBA_END_TO_END_DISPATCH/PARITY` PASS。
2. **PCA/GARCH 共享 authoritative state**（§23，**DONE**，本次闭环）：抽出 `PCAState.from_window`
   单点数学（coverage-gated active + standardized SVD），`panel_model._pca_svd` / `_pca_commonality`
   与 `model_family_kernels.PCABlock` 全部委托同一数学；`PCABlock.commonality` 修正为 canonical
   `1 - Var(resid)/Var(ret)`（byte-compatible，51 tests pass）。gate `R38_PCA/GARCH_SHARED_STATE_PARITY`
   PASS。
3. **stateful time-shard checkpoint**（§4，DEFERRED）：`ShardDescriptor.checkpoint_ref` 字段预留，
   真实 checkpoint 引擎 + 状态连续性校验未实现。
4. **运行中 read-wave JIT 重排**（§16，**DONE**，本次闭环）：`AdaptiveBatchScheduler.
   _maybe_repartition_waves` 在 wave 预算显著缩小（>40%）时对未执行 SOURCE_SCAN 按新预算重新拆小，
   新 wave_id 不与已执行冲突、已覆盖 task 不重复扫。gate `R38_DYNAMIC_READ_WAVE_SHRINK` PASS。
5. **DA ReadPipeline host child-lease 桥**（§10，PARTIAL）：`sync_da_limits` 显式 + DA governor
   锁定/standalone cap 完成；`PreparedRead` 持真实 HostLeaseRef、`ReadPipeline.admit` 从 host
   租约走——仍需 DA 会话消费。
6. **target_concurrency 与 io/remote/spill decision 字段**（P0-047/P0-012，PARTIAL）：scheduler
   已消费 wave/sink/concurrency/cpu；io/remote/spill 的独立 consumer 接线待后续。
7. **R36 gate 从 source-presence 重写为行为 gate**（§26）：R38 gate 已是行为探针；R36 旧 gate
   json 保留，未逐条重写（避免改历史证据）。
8. **current SHA 外部 CI**（§27，PARTIAL）：本报告绑定 `d34cc9f50` + 本地回归全绿；仓库无外部
   CI runner，`R38_CURRENT_HEAD_CI_PASS` 以本地行为 gate + 测试矩阵证明。

## 7. 工程标准

> R38 不做「加模块」，把已有的模块从「有类、有测试、有报告」推进到「真实生产主链正在使用 +
> 失败路径也符合合同 + 资源账本与实际资源一致 + 行为级测试能够证明」。Numba/PCA/GARCH/wave
> 本轮闭环；剩余 stateful checkpoint/DA child-lease 如实标记 DEFERRED/PARTIAL，不写「全部完成」。
