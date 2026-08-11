# R39 全链路极速落值终审 —— 闭包报告（2026-08-11）

> 整改依据：`FactorEngine_R39_全链路极速落值终审_扫描转换计算调度写放大与批量物化优化方案_20260811.md`
> 审计基线 HEAD：`8e9893b5`；实施期间并发会话持续推进 HEAD（当前 `c9c08ff5`）。
> 完成判定遵循 §34：只标记有真实代码改动 + 单元测试 + benchmark + before/after + correctness parity 的 issue。
> 交付物：84 项 PERF 状态 + 12 硬门 + 16 实现 agent + 4 补刀 agent + 13 份 §32 evidence + §33 Q&A + 2 个 benchmark 期真实 bug 修复。

---

## 一、结果总览

| 类别 | 数量 |
|---|---|
| PERF 项 CLOSED（含 P0 全闭） | **78** |
| PERF 项 PARTIAL（PERF-017：native backend slice 未做） | 1 |
| PERF 项 NOT_CLOSED（P1 深架构/深内核，诚实记录） | 5（020/073/078/079/080） |
| 硬门 PASS | **12 / 12** |
| 新增 R39 测试 | **295**（tests/r39/ 全绿） |
| 既有回归 | 217 passed / 4 pre-existing（baseline 亦失败，非 R39） |

**§27 同机 before/after（QUICK，300 股 × 252 日 × 100 因子，干净 lake）**
- baseline `8e9893b5`：B1 total **129.3s**（run 3.7s + mat 125.5s）
- HEAD：B1 total **96.3s**（run 3.4s + mat 92.8s）→ **TTDC 改善 ~26%**
- `batch_write_transaction_count`：100 → **2**
- R39 前 `materialize_many_fast` 100 因子在固定 10s join 下**必然 writer-alive-abort**（该规模不可用）；修复后正常落值。

## 二、实现覆盖（§2-§25 分簇）

16 个实现 agent 按文件不相交分簇 + 4 个补刀 agent（PERF-030/027/051/063），全部中央串行单进程回归。

- **§2 control-plane**：typed SourceBinding/TimeRange/ReadWave source+consumer tasks/ProjectedColumnFootprint/cost-based superset coalescing/ReadOnlyOverlayMap（零拷贝）/BatchWarmupPlan/PlanExecutionCertificate 缓存/MicroBatchTask（16-128 roots 单 Future）→ future_per_factor 0.05
- **§10 分钟→日**：Counter 去重/整数分钟 EXTRACT/FilterSignature dedup/单条 DuckDB SQL → B6 实测 100 聚合 **121ms** 单扫
- **§11-14 materialize/catalog**：O(1) factor dict/batch 事务（tx 100→2）/flush count+bytes+age/sink 攒批 64/least-loaded partition pinning/opt-in delta/Watermark 无 read-back（正常路径 0 次 get_watermark）
- **§15-19 storage/matrix**：matrix 轴相等 column-stack 0 join/block 布局 COW 未触 block hardlink/PartitionObjectRef inventory（rglob 不增）/checksum proof/load pushdown/streaming materialize
- **§20-25 其余**：applied_config_fingerprint PRAGMA 跳过/DeadlineManager 单 timer 线程/QueryClassCohort/ScanShapeKey/TtdcEstimate/PerfCounters 22 槽/ProductionExecutionCertificate

## 三、硬门 12/12 PASS

Gate-01..12 全部有静态断言或运行时计数证据（详见 `R39_ISSUE_CLOSURE_LEDGER.md` 硬门表）。

## 四、Benchmark 期真实 bug 修复（超出 84 项）

1. **sink join timeout 误杀活 writer**（`streaming_result_sink.py`）——PERF-040 batch=64 后一次 flush 写 64 因子 >10s，`finish()` 固定 10s join 把活着的慢 writer 判 fatal。修复：join timeout 随 batch_size 缩放 + finish 时按队列剩余动态放宽；真死锁语义保留。8 个单测。
2. **R40 交叉 bug**（`runtime/config.py`）——`TypedDataSourceOptions.to_dict()` 把 `read_auto/snapshot_only/normalize_timestamp` 缺省 False 泄漏进所有 source 类型，parquet_kline 被 `_ensure_no_extra_options` 误拒。修复：缺省改 None。6 个 enterprise 测试恢复。
3. **测试适配**（PERF-068 行为变化）：`materialize_sharded` 走 batch 写路径，`test_materialize_sharded_selects_subset` 改 patch 批量入口。

## 五、诚实局限（§34）

- **5 项 NOT_CLOSED**：PERF-020（process worker lane）、073（DA ScopedConnectionPool，dataaccess 并发活跃区）、078（NativePipelineRegion optimizer 建模）、079（RollingStateBlock Numba 内核族）、080（PrimitiveBlock 跨因子物化）。均需深度架构/内核改造，本轮未做，如实记录。
- **PERF-017 PARTIAL**：OutputSlice 零拷贝交付完成；native backend 源头 emit slice 未做。
- **benchmark 数字为 QUICK 规模**（并发会话持续占机，全量 1500×504×300 无法在安静窗口完成）；before/after 同机同 workload 干净 lake，环境负载记录在 evidence。
- **baseline 首测 42s 为 resume 跳写假象**，已用干净 lake 复测 125.5s 并记录（不虚报 3x 加速）。

## 六、交付物清单

- `docs/R39_ISSUE_CLOSURE_LEDGER.md` —— 84 项状态 + 12 硬门 + §33 16 问 Q&A
- `docs/R39_CLOSURE_REPORT_2026_08_11.md` —— 本报告
- `build/r39_evidence/r39_*.json` —— 13 份 §32 evidence
- `scripts/r39_benchmark_suite.py`（after）+ `scripts/r39_baseline_bench.py`（baseline，可复现）
- `tests/r39/` —— 295 个新测试（含补刀 027/030/051/063 与 sink join timeout）
- 新模块：`planner/`（source_binding/fusion_size_model/negative_fusion_cache/projected_column_footprint/output_slice）、`runtime/`（readonly_overlay_map/batch_warmup_plan/deadline_manager/query_class_cohort/scan_shape/column_footprint/factor_block_ref/execution_identity/factor_campaign_session/materialization_identity_certificate/micro_batch_task/plan_execution_certificate/production_execution_certificate/temp_array_arena/spool_policy/perf_counters/performance_run_summary）、`storage/`（delta_store/write_amplification/partition_stats/matrix_block_layout/block_lake/parquet_batch_writer/partition_object_ref）、`backend/rep_transition.py`、`dataaccess/read/minute_filter.py`

未 commit / 未 push（R39 §0 要求直接改 main 树，由用户决定提交时机）。
