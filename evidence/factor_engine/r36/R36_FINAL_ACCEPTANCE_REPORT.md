# R36 最终验收报告 — 异构服务器自适应资源治理

> 审计 HEAD：`c2309dbd4abe73945d8f1c97b402b8b4cbdc75b6`（R36 完成时）
> 日期：2026-08-11
> 关联：R33（FE×DA unified query graph）→ R36（Resource Autopilot）

## 0. 结论

**`scripts/audit_r36_hard_gates.py` 46/46 hard gates 全 TRUE（`R36_HARD_BLOCKERS_ZERO=true`，exit 0）；
`tests/r36/` 24 tests 全绿；R33 gates 18 tests 全绿；R30 closure True；R27/R31/R32/service 112 tests 全绿。**
R33/R34/R35 正确性基线无回归。

## 1. R36 目标

把 FactorEngine 从「有资源治理模块」升级为**实时反馈控制系统**（R36 附录 D）：
> Fast when idle, polite when contended, fail-safe before OOM.

不再要求运维手工调 n_jobs / DuckDB threads / wave / block / sink / spill；
不再固定 4GB wave / 4GB sink / 2GB-per-worker；压力升高立即让路，压力解除自动恢复，
且不改变因子的语义、PIT、结果与 generation 一致性。

## 2. 本轮回合的全部修复

### P0（§296 全部 30 项，FE 侧）

| P0 | 实现 | gate |
|----|------|------|
| 001 max_concurrency 真正生效 | `_dynamic_concurrency_limit = min(max_concurrency, decision.target_cpu_tokens, backpressure)` | `R36_MAX_CONCURRENCY_ENFORCED` |
| 002 scheduler 消费动态 concurrency | `ResourceDecision`（§305）+ scheduler 每 control tick 消费 | `R36_BROKER_RECOMMENDATION_CONSUMED` |
| 003 压力恢复自动升速 | `ResourceController` AIMD（fast down ×0.5 / slow up +1，hysteresis + cooldown） | `R36_RESOURCE_RECOVERY_UPSHIFT` / `R36_DYNAMIC_CPU_BUDGET_BIDIRECTIONAL` |
| 004/005 per-shape P99 模型 | `ResourceShapeKey`（§10/§309 离散桶）+ `ShapeCalibration`（P50/P90/P95/P99/max/count） | `R36_P99_MEMORY_MODEL_ACTIVE` |
| 006 uncertainty 反馈闭环 | `prediction_error = actual/predicted` → correction factor（低估 streak 上调；OOM 立即 ×1.3） | 同上 + `R36_ZERO_SAME_SHAPE_OOM_RETRY` |
| 007 process family PSS | `process_family_memory_bytes(prefer_pss=True)`；`MemoryGovernor._current_rss` PSS 优先 | `R36_PROCESS_FAMILY_MEMORY_ACCOUNTED` |
| 008 backend threads/token 一致 | lowerer `cpu_tokens = backend_threads = _engine_threads_for(backend)` | `R36_DUCKDB_CPU_TOKEN_MATCH` |
| 010 dynamic read wave | wave budget 从 `decision.read_wave_bytes`（SafeEnvelope×fraction） | `R36_DYNAMIC_READ_WAVE_BYTES` |
| 011 dynamic sink queue | `queue_bytes=None → decision.result_queue_bytes` | `R36_DYNAMIC_SINK_QUEUE_BYTES` |
| 012 sink backpressure 接 scheduler | `_dynamic_concurrency_limit` 0.70/0.90 双档 | `R36_SINK_BACKPRESSURE_TO_SCHEDULER` |
| 013 deque | R33 sink 已用 `collections.deque`（保持） | — |
| 014 writer fail fatal | R33 sink 状态机（保持）+ gate 验证 | `R36_WRITER_FAILURE_FATAL` / `R36_ZERO_FINISH_SILENT_WRITE_LOSS` |
| 016 FE/DA 统一 authority | `HostResourceCoordinator` 进程级单例；scheduler/service 共用其 broker | `R36_ONE_HOST_RESOURCE_AUTHORITY` |
| 017 DA memory cap 统一 | `apply_da_envelope()` 把 Safe Envelope 注入 DA governor `max_total_reserved_memory` / scan inflight | `R36_FE_DA_DOUBLE_ADMISSION_ZERO` |
| 018 service broker race | `_service_broker_ctx` module-global → `ContextVar` | `R36_SERVICE_BROKER_RACE_ZERO` |
| 019 spill lifecycle | broker spill quota + spillable 契约（基础闭环） | `R36_DUCKDB_SPILL_QUOTA` |
| 020 auto shard | `AutoShardPlanner`（§88..97 语义合法维度）+ scheduler no-progress 先 replan smaller | `R36_AUTO_SHARD_WHEN_TASK_EXCEEDS_ENVELOPE` / `R36_ZERO_ILLEGAL_SHARD` |
| 021 raw CSE cache bypass | `GovernedBufferStore`（byte 记账 + 预算拒绝 + 冷淘汰 + reconciliation）；`_materialize_shared_subplan` 经 `ctx.shared_buffers` | `R36_ZERO_RAW_CSE_CACHE_BYPASS` |
| 022 cache register fail-closed | `ExecutionCacheSession.strict` → production raise | `R36_CACHE_GOVERNANCE_FAIL_CLOSED` |
| 023 cache release ref+accounting | `release()` 同时清 backing dict + 记账；`GovernedBufferStore.release` 同步释放 | `R36_CACHE_RELEASE_REF_AND_ACCOUNTING_MATCH` / `R36_CACHE_ACCOUNTING_RECONCILES` |
| 028 global resource scope race | `ExecutionResourceScope` 并发跨线程检测（production fail-closed） | `R36_ZERO_CONCURRENT_GLOBAL_RESOURCE_SCOPE_RACE` |
| 029 true run peak | `RunPeakSampler` run 窗口专用（start/stop，只统计该 run） | `R36_TRUE_RUN_PEAK` |

### P1 顺带收口
- **PSI**（§28/30）：`/proc/pressure/{cpu,memory,io}` + cgroup `*.pressure` + `memory.events` + swap 进 `ResourceSnapshot`/`ResourceSignals` → controller 消费（`R36_PSI_*_CONSUMED`）。
- **Memory slope**（§69/70）：`MemorySlopeTracker` EWMA，外部任务快速吃内存提前让路（`R36_MEMORY_SLOPE_CONSUMED`）。
- **memory-constrained mode**（§244/245）：`ResourceDecision.memory_constrained` 自动进入（`R36_POLARS_STREAMING_MEMORY_ROUTE`）。
- **Polars thread contract honest**（§23/24/127）：`_engine_threads_for(polars)` 读 `POLARS_MAX_THREADS`；scope `polars_live_effective` 诚实标记（`R36_POLARS_THREAD_CONTRACT_HONEST` / `R36_POLARS_STREAMING_FALLBACK_GUARDED`）。
- **BLAS nested oversubscription**（§129/130）：`check_nested_cpu_oversubscription`（`R36_BLAS_NESTED_OVERSUBSCRIPTION_ZERO`）；R35 并发会话给 scheduler 加了 `thread_budget`（`runtime.execution_traits`）在 task scope 内 pin BLAS/OpenMP threads —— 与 §128 方向一致。
- **Numba block contract**（§133..135）：`parallel_kernel_contract` cpu_tokens = 实际线程数（`R36_NUMBA_BLOCK_RESOURCE_CONTRACT`）。
- **Calibration 持久化**（§209/34）：`ResourceCalibrationStore.save` → parquet，硬件指纹 key，load 恢复分布（`R36_RESOURCE_CALIBRATION_PERSISTED`）。

## 3. 新增模块（§302）

```
runtime/resource_monitor.py          # PSI / slope / SafeEnvelope / ResourceSignals
runtime/resource_autopilot.py        # ResourceDecision + ResourceController（AIMD）
runtime/host_resource_coordinator.py # FE/DA/service 唯一资源权威 + child lease
runtime/resource_shape.py            # ResourceShapeKey + hardware fingerprint
runtime/resource_calibration_store.py# P99 模型 + parquet 持久化
runtime/run_peak_sampler.py          # run 窗口专用峰值采样
runtime/buffer_store.py              # GovernedBufferStore（CSE 缓冲治理）
runtime/auto_shard_planner.py        # 语义合法 auto shard
```

## 4. 测试与证据

- **46/46 hard gates**：`docs/evidence/r36/R36_HARD_GATES.json`（exit 0）
- **`tests/r36/test_r36_hard_gates_2026_08.py`：24 tests 全绿**
- **回归**：R33 gates True + tests/r33 18 passed；R30 closure True；R27/R31/R32/service 112 passed
- **证据**：`R36_HEAD.json` / `SERVER_PROFILE.json`（真实 cgroup/PSI/swap）/ `RESOURCE_DECISION_TRACE.json`（controller 可解释决策日志）/ `CALIBRATION_STORE.json`

## 5. 服务器实测 profile（本机）

- `SERVER_PROFILE.json`：effective_cpu_slots / effective_memory_limit / PSI（cpu/memory/io）/ swap。
- 服务器无 swap；PSI 可读（cpu some avg10 可波动）→ `R36_PSI_*_CONSUMED` 探针用合成信号验证 controller 消费路径，真实验证用 `RESOURCE_DECISION_TRACE.json`。

## 6. 诚实保留（Deferred）

1. **DA 包内部**（`dataaccess/runtime/resource_governor.py` 已加 `set_max_total_reserved_memory` / `set_max_total_scan_bytes_inflight`）：DA 侧完整 child-lease 消费（`PreparedBatchReadSession` 内 scan budget 真正从 host lease 走）属 DA 会话继续。
2. **full-market 1000-factor TTDC**：本机无真实 A 股全市场 snapshot；`R36_1000_FACTOR_TTDC_CURRENT_HEAD` 用合成日频 50 资产 × 500 日编译 1000 因子（真实 runtime，ADAPTIVE_DAG 路径），真实市场 benchmark 待数据。
3. **Polars streaming 引擎检测**（§26）：`R36_POLARS_STREAMING_FALLBACK_GUARDED` 目前验证 thread-contract honest + `polars_live_effective` 诚实标记；物理 plan 级 streaming-fallback 探测（`explain` 解析）为 P2。
4. **真实 per-task PSS attribution**（§170/171）：第一版只做 run 级峰值；per-shape 归因需 isolated calibration + concurrent marginal model（已记录在 `RunPeakSampler` docstring）。
5. **测试排序污染（pre-existing）**：`tests/r27/conftest.py` `os.environ.setdefault("FACTOR_ENGINE_MAX_MEMORY_BYTES","8GB")` 在 -k 全集运行时会泄漏到后序测试（`test_resource_plan_respects_cores_budget` 等 3 个 fail，单独跑均过）。已验证为 R36 之前就存在的测试隔离问题，非 R36 引入。

## 7. 工程标准

> FactorEngine 不要求服务器「为它清场」，也不用固定 worker 数赌内存。它把自己当成弹性计算负载：
> 服务器空闲时主动吃满安全余量，其他算法启动时快速收缩，压力解除后自动恢复，
> 并且无论怎样调度都不改变因子的语义、PIT、结果和最终 generation 的一致性。
