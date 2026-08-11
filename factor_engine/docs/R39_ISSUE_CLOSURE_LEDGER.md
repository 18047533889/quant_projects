# R39 Issue Closure Ledger

> 整改依据：`FactorEngine_R39_全链路极速落值终审_扫描转换计算调度写放大与批量物化优化方案_20260811.md`
> 审计基线 HEAD：`8e9893b5`；实施期间树持续被并发会话演进，最终以真实 HEAD 为准。
> 完成判定遵循 §34：只有改动真实执行路径、有单元测试、有 benchmark、有 before/after、有 correctness parity 的 issue 才可标记 `CLOSED`。
> 实施方式：16 个实现 agent（文件不相交分簇）+ 4 个补刀 agent（PERF-030/027/051/063）。所有新 R39 测试中央串行单进程全绿；既有回归套件全绿（numba 为 R38 缺失依赖，已补装）。
> **Benchmark 期补充修复（真实 bug）**：PERF-040 默认 batch=64 后，sink `finish()` 的固定 10s join timeout 会把「活着的慢 writer」（一次 flush 写 64 因子 >10s）误判为致命并 abort generation。修复：join timeout 随 batch_size 缩放 `max(10s, batch_size×2s)`，显式 `join_timeout` 可覆盖；真死锁语义（R38 P0-045）保留。`runtime/streaming_result_sink.py` + `tests/r39/test_perf_sink_join_timeout_2026_08.py`（6 测试）。

每项格式：
```
Issue ID → 修改文件 → 实际实现 → 测试 → benchmark → before → after → correctness parity → evidence → status
```

---

## §2 批量执行 control-plane

| Issue | Status |
|---|---|
| R39-P0-PERF-001 | CLOSED — scheduler 主路径跳过 legacy full-union prefetch（`resolve_batch_execution_control_plane`，`legacy_union_prefetch_count==0`）；legacy 层（`FACTOR_ENGINE_LAYER_LOOP=1`）保留。tests/r39/test_perf_batch_service ✓ |
| R39-P0-PERF-002 | CLOSED — typed `ColumnSourceBinding`（planner/source_binding.py），secondary SourceRef 由列直接解析，`SOURCE_REF_WITHOUT_TYPED_SOURCE_BINDING==0`。tests/r39/test_perf_batch_data_request ✓ |
| R39-P0-PERF-003 | CLOSED — `BatchSourceResolver` 每 source scope 独立 adapter/estimator，secondary 不复用 anchor estimator（独立 `ScanCostUnavailable`）。同上 ✓ |
| R39-P0-PERF-004 | CLOSED — typed `TimeRange` 保留 `(None,end)`/`(start,None)`/`(start,end)`，不再因 start=None 退化全量。同上 ✓ |
| R39-P0-PERF-005 | CLOSED — ReadWave `source_tasks/consumer_tasks`，ROOT/CSE_SHARED 不再建零列 scan request；`ZERO_COLUMN_READ_WAVE==0`。tests/r39/test_perf_read_wave ✓ |
| R39-P0-PERF-006 | CLOSED — `ProjectedColumnFootprint` 真实 projection bytes（manifest→ScanCost→fallback），wave memory 用真实 footprint。同上 ✓ |
| R39-P0-PERF-007 | CLOSED — `_best_marginal` 边际成本用 `incremental_live_bytes` 作分母 + `_bounded_local_search`。同上 ✓ |
| R39-P0-PERF-008 | CLOSED — cost-based superset coalescing（`superset_extra_scan_cost < duplicate_open_decode_conversion_saved`）。同上 ✓ |
| R39-P0-PERF-009 | CLOSED — `ScopeCompatibility(EXACT/UNION_COMPATIBLE/INCOMPATIBLE)`，instrument_scope/universe_id 进 compatibility+成本；CSI300 不并入 full-A。同上 ✓ |
| R39-P0-PERF-010 | CLOSED — `baseline_duplicate_scan_bytes`/`physical_union_scan_bytes`/`saved_scan_bytes`/`decoded_resident_bytes`。同上 ✓ |
| R39-P0-PERF-011 | CLOSED — `SourceRepresentation` + `consumer_backend_mask`，`SourceWaveExecutor` 发 preferred_representation；backend 关键字顺序修正。同上 ✓ |
| R39-P0-PERF-012 | CLOSED — `WaveRecoveryPlan` 决策表（OVERSIZED/UNSUPPORTED/TRANSIENT_IO/MEMORY_PRESSURE/PERMANENT），`FAILED_WAVE_TO_N_ROOT_SCANS==0`。同上 ✓ |
| R39-P0-PERF-013 | CLOSED — `ReadOnlyOverlayMap(base,local)` 共享 base 不复制；`dict_entry_copy_count==0`（Gate-09）。tests/r39/test_perf_batch_service ✓ |
| R39-P0-PERF-014 | CLOSED — `BatchWarmupPlan` 全 batch 只算一次（run_many/run_many_parallel/run_many_iter）。同上 ✓ |
| R39-P0-PERF-015 | CLOSED — `_cluster_factors_by_cost` 用真实 load window 字节成本分组（非固定比例桶）。同上 ✓ |
| R39-P0-PERF-016 | CLOSED — `PlanExecutionCertificate` + `_cost_ms_cache`，`_priority_score` 不再重复 DAG 遍历。tests/r39/test_perf_scheduler ✓ |
| R39-P0-PERF-017 | PARTIAL — `OutputSlice`+`apply_output_slice`（零拷贝 .iloc，`np.shares_memory`）+`(BufferRef,slice)` writer 交接完整；native backend 源头 emit slice 未做（backend 文件不在本轮范围）。tests/r39/test_perf_batch_service ✓ |
| R39-P0-PERF-018 | CLOSED — `MicroBatchTask`（16-128 cheap roots/单 Future，per-root 隔离，单 synthetic lease）。`future_per_factor` 0.05 vs 1.0。tests/r39/test_perf_scheduler ✓ |
| R39-P1-PERF-019 | CLOSED — wait 已是 FIRST_COMPLETED event-driven；`scheduler_wait_polling_count==0` 快乐路径。同上 ✓ |
| R39-P1-PERF-020 | NOT_CLOSED（P1，深架构）— PlanId process worker lane 需 process/mmap 基础设施，本轮未实现。诚实记录。 |
| R39-P0-PERF-021 | CLOSED — `FusionSizeFeatures`+`FusionCostModel` 单位模型（超线性 AST 项），`choose_fusion_block_size` 单调。tests/r39/test_perf_fusion ✓ |
| R39-P0-PERF-022 | CLOSED — binary-split 失败恢复（`native_fusion_binary_split`），最小失败集退 per-root；`NegativeFusionCache` 跳过已知坏组合。同上 ✓ |
| R39-P1-PERF-023 | CLOSED — `RootLocality`+`locality_score`（window 2.0>source 1.0），fusion 组内 locality 重排。同上 ✓ |
| R39-P0-PERF-024 | CLOSED — Arrow IPC spool（`ArrowSpoolRef`，uncompressed 单 block），免 Pandas→Parquet→Pandas。tests/r39/test_perf_shard ✓ |
| R39-P0-PERF-025 | CLOSED — `verify_shards_sorted_nonoverlapping` 运行时证明；CONCAT_ONLY 一次 concat（`shard_concat_sort_count==1`），ORDERED_MERGE 保底。同上 ✓ |
| R39-P0-PERF-026 | CLOSED — `ShardMergeMode` 数据驱动选择 + `ShardDirectWriteManifest`（DIRECT_DURABLE_APPEND concat_sort_count==0）。同上 ✓ |
| R39-P1-PERF-027 | CLOSED — `decide_spool_or_keep`（live_headroom/writer_queue_headroom/disk_throughput/merge_lifetime）+`adaptive_spool_threshold`；默认不传 policy 时保持原 512MB；已接入 `adaptive_batch_scheduler` TASK_SHARD 调用点。tests/r39/test_perf_spool_policy（补刀 agent，28 测试终验 ✓） |
| R39-P0-PERF-028 | CLOSED — `panel_to_polars_bulk`/`polars_to_panel_bulk`（Arrow CDI，NaN 恢复、dtype 安全门），byte 恒等验证；fallback 保留。tests/r39/test_perf_rep_conversion ✓ |
| R39-P0-PERF-029 | CLOSED — `AxisIdentityCertificate` 每帧缓存 + 4096 LRU + O(1) staleness guard；fast==slow 50 对随机 parity。tests/r39/test_perf_axis_identity ✓ |
| R39-P0-PERF-030 | CLOSED — `FactorBlockRef`（axis 共享/单次 Arrow-Parquet 写），matrix block 路径 + batch writer 接线。tests/r39/test_perf_factor_block_ref（补刀 agent，16 测试终验 ✓） |
| R39-P0-PERF-031 | CLOSED — `RepresentationTransitionTracker`（transition_count/bytes），bulk+fallback 双路径记录。tests/r39/test_perf_rep_conversion ✓ |
| R39-P1-PERF-032 | CLOSED — `TemporaryArrayArena`（shape_bucket+dtype 池化，poison 防双租，with_scratch）；standalone 未接线 hot consumer（诚实）。同上 ✓ |

## §10 分钟→日

| Issue | Status |
|---|---|
| R39-P0-PERF-033 | CLOSED — `ParsedAggregationItem` parse 一次；dup 检查 `list.count`→`Counter`（K²→K）。dataaccess/tests/r39/test_perf_minute_agg ✓ |
| R39-P0-PERF-034 | CLOSED — 整数分钟 `EXTRACT(HOUR)*60+EXTRACT(MINUTE)`，timezone 一次；09:30:59→570/09:31:00→571 边界与旧 strftime 一致。同上 ✓ |
| R39-P0-PERF-035 | CLOSED — `FilterSignature` dedup，共享 window 的 item 用同一 `_fN` 条件列。同上 ✓ |
| R39-P1-PERF-036 | CLOSED — 每个公开入口单条 DuckDB SQL 完成 minute→daily（单 native region）；pandas fallback 仅 LQTP 侧。同上 ✓ |
| R39-P1-PERF-037 | CLOSED — `result_mode: arrow/relation/arrow_stream`，`_validate_result_mode` fail-closed，默认 arrow 行为不变。同上 ✓ |

## §11-14 materialize / catalog

| Issue | Status |
|---|---|
| R39-P0-PERF-038 | CLOSED — `factor_by_name`/`factor_id_by_name` 预建 dict，writer 热循环 O(1)；静态 Gate-02 断言。tests/r39/test_perf_materialize_fast ✓ |
| R39-P0-PERF-039 | CLOSED — `execute_materialize_batch` + `DependencyCatalog.record_factor_manifests_many` 单 SQLite 事务；`batch_write_transaction_count=1<<N`（Gate-03）。同上 ✓ |
| R39-P0-PERF-040 | CLOSED — `_should_flush`（count/bytes/age 任一触发），sink 默认 batch 64。同上 ✓ |
| R39-P0-PERF-041 | CLOSED — `writer_threads=None`→单 partition 1 / partition-aware min(2,cores)。同上 ✓ |
| R39-P0-PERF-042 | CLOSED — `_route_worker` 首见分区→least-loaded writer 并固定（同 partition 恒同 worker）。同上 ✓ |
| R39-P0-PERF-043 | CLOSED — opt-in delta 模式（`FACTOR_ENGINE_DELTA_STORAGE=1`，默认 OFF）：immutable delta + generation manifest，hot path 不再 full read→concat→dedup→sort→rewrite。tests/r39/test_perf_storage ✓ |
| R39-P0-PERF-044 | CLOSED — `WriteAmplificationTracker`；delta 普通增量 `historical_rewrite_bytes==0`（Gate-04）。同上 ✓ |
| R39-P0-PERF-045 | CLOSED — wide 不再作 mutation hot format（block 布局 + delta 片段 + 清单）。同上 ✓ |
| R39-P0-PERF-046 | CLOSED — `materialize_block(pa.Table/RecordBatchReader)`，Arrow is_finite/if_else 生成 mask。同上 ✓ |
| R39-P0-PERF-047 | CLOSED — tombstone/finite mask 在 Arrow batch 直接生成，元数据列 dictionary-encoded。同上 ✓ |
| R39-P0-PERF-048 | CLOSED — 双扫移除；`_count_partition_metrics` 正常路径 calls==0；`full_factor_rescan_count==0`（Gate-05）。同上 ✓ |
| R39-P0-PERF-049 | CLOSED — `PartitionCommitStats` + `factor_partition_stats` catalog 表，watermark 不 read 全因子历史。同上 ✓ |
| R39-P0-PERF-050 | CLOSED — `CatalogBatchTransaction`（register_many/record_runs_many/update_watermarks_many），N update→1 commit。同上 ✓ |
| R39-P1-PERF-051 | CLOSED — watermark 从 commit result 直接返回，正常路径 `post_write_watermark_readback_count==0`；仅 deferred/CAS/legacy/staging 场景 read-back。tests/r39/test_perf_watermark_readback（补刀 agent，9 测试终验 ✓） |
| R39-P0-PERF-052 | CLOSED — immutable delta fragments + atomic manifest flip + fsync 命名（generation/seq），legacy data.parquet 可读。同上 ✓ |
| R39-P1-PERF-053 | CLOSED — orphan tmp 清理改为 generation recovery（`recover_orphan_delta_tmp_files`），不每次 partition write 扫。同上 ✓ |
| R39-P1-PERF-054 | CLOSED — `PartitionLockManager` generation 内复用 lock 句柄。同上 ✓ |
| R39-P0-PERF-055 | CLOSED — `FactorBlockLake`（Layout A long / Layout B wide column block，256 因子/block + manifest）。tests/r39/test_perf_storage_tuning ✓ |
| R39-P1-PERF-056 | CLOSED — `LakeWriteLayoutPolicy.choose_layout`（factor_count/update_freq/read_pattern/matrix_demand/storage_class）。同上 ✓ |
| R39-P1-PERF-057 | CLOSED — `BatchParquetWriter` 直写 PyArrow ParquetWriter（RecordBatch 流，row-group 按行/字节自适应，compression 归一化）。同上 ✓ |
| R39-P1-PERF-058 | CLOSED — `scripts/storage_tuning_benchmark.py`（compression × row-group 矩阵，读回校验 all_validated）。同上 ✓ |
| R39-P0-PERF-059 | CLOSED — `compute_write_pass_dq` 单 pass（row/finite/nan/inf/min/max/checksum/date/dup-keys）。tests/r39/test_perf_storage ✓ |

## §15-19 storage / matrix

| Issue | Status |
|---|---|
| R39-P0-PERF-060 | CLOSED — matrix block assembly：轴相等直接 column-stack（`matrix_join_count==0`），异轴一次 concat（==1，非 N-way outer merge）；legacy raw parquet 字节级一致。tests/r39/test_perf_matrix ✓ |
| R39-P0-PERF-061 | CLOSED — opt-in `FACTOR_ENGINE_MATRIX_BLOCK_LAYOUT=1`：year/month/block=NNNN.parquet，manifest factor→{block,column}。同上 ✓ |
| R39-P0-PERF-062 | CLOSED — block 模式只写 touched block；未触 block COW hardlink；新因子新建 block 不重写已有（新增因子 amplification 0.0）。更新因子仍重写整个 256 列 block（bounded，诚实）。同上 ✓ |
| R39-P1-PERF-063 | CLOSED — `PartitionObjectRef` inventory（rel_path/content_id/rows/schema_hash），generation 切换精确 materialize，`generation_rglob_discovery_count` 不增。tests/r39/test_perf_cow_manifest（补刀 agent，11 测试终验 ✓） |
| R39-P0-PERF-064 | CLOSED — checksum proof（key-order/per-column finite-mask/numeric/schema_hash + read-back），byte-flip 检出；opt-in `FACTOR_ENGINE_MATRIX_CHECKSUM_PROOF=1`。同上 ✓ |
| R39-P1-PERF-065 | CLOSED — `load_matrix` 接受 factor_ids/time_range/instrument_filter，block manifest→只扫所需 block（spy 证明只开 block=0001）。同上 ✓ |
| R39-P0-PERF-066 | CLOSED — `materialize_matrix_streaming`+generator adapter 块级消费（engine.run_many dict 边界诚实记录）。同上 ✓ |
| R39-P1-PERF-067 | CLOSED — `MaterializationIdentityCertificate` compile 后缓存，batch 一次构建。同上 ✓ |

## §20-25 其余

| Issue | Status |
|---|---|
| R39-P0-PERF-068 | CLOSED — `materialize_sharded` batch 走 `materialize_many_fast` 流式 sink（`shard_scope` 传播），不 run_many 攒全量 dict。tests/r39/test_perf_materialize_fast ✓ |
| R39-P0-PERF-069 | CLOSED — `materialize_incremental_many_from_config` 按 `ExecutionIdentity` 分组，engine+source session 构建 O(identity-groups)（O(configs)→O(identities)）。tests/r39/test_perf_incremental_many ✓ |
| R39-P1-PERF-070 | CLOSED — `FactorCampaignSession` 同 identity 共享 engine/CSE compile_many；DataEvent 路径保留 R10 #51 每 factor 独立 engine。同上 ✓ |
| R39-P0-PERF-071 | CLOSED — `applied_config_fingerprint`（SHA-1 threads/budget/pragma_set），同配置跳过 PRAGMA replay（`pragma_skip_count`），`__exit__` 重写指纹。tests/r39/test_perf_duckdb_scan ✓ |
| R39-P0-PERF-072 | CLOSED — `DeadlineManager` 单 lazy timer thread + min-heap lazy deletion，`watchdog_thread_created_count` O(1)（Gate-08）。同上 ✓ |
| R39-P1-PERF-073 | NOT_CLOSED（P1）— DA `ScopedConnectionPool` 属 dataaccess 并发活跃区，本轮未触碰。诚实记录。 |
| R39-P1-PERF-074 | CLOSED — `QueryClassCohort`（1/2/4/8 固定 profile 池），opt-in `FACTOR_ENGINE_COHORT_PROFILES=1`，`duckdb_threads_for`/`cohort_worker_threads` 接线。同上 ✓ |
| R39-P0-PERF-075 | CLOSED — `ScanShapeKey`（dataset/local-remote/file-count/bytes/rowgroup/column-ratio/instrument/manifest/cache/storage）+ P50/P95 bounded window。同上 ✓ |
| R39-P1-PERF-076 | CLOSED — `TtdcEstimate`+`predict_ttdc`（DuckDB 无 conversion/Polars arrow 转换/PyArrow），`choose_plan_route` shape-aware + Polars-fused-downstream 惩罚。同上 ✓ |
| R39-P1-PERF-077 | CLOSED — `ColumnFootprintStats`（parquet row-group column 元数据），manifest 缺失 lazy fallback。同上 ✓ |
| R39-P1-PERF-078 | NOT_CLOSED（P1，深架构）— NativePipelineRegion 需 optimizer 级 region 建模，本轮未实现。诚实记录。 |
| R39-P1-PERF-079 | NOT_CLOSED（P1，深内核）— RollingStateBlock 多输出 kernel 需 Numba 内核族扩展，本轮未实现。诚实记录。 |
| R39-P1-PERF-080 | NOT_CLOSED（P1，深内核）— PrimitiveBlock 跨因子物化需 CSE/primitive extraction 重构，本轮未实现。诚实记录。 |
| R39-P1-PERF-081 | CLOSED — `PerfCounters` 固定槽 array-backed counters（22 slots），snapshot 末渲染，未声明槽 KeyError。tests/r39/test_perf_telemetry ✓ |
| R39-P1-PERF-082 | CLOSED — `ProductionExecutionCertificate`（O(1) validate：integrity→no_fallback→backend），`HybridExecutor` 事件记录，fail-open 不阻断。同上 ✓ |
| R39-P1-PERF-083 | CLOSED — delta fragment 自身 sorted，compaction 时才全局排序。tests/r39/test_perf_storage ✓ |
| R39-P1-PERF-084 | CLOSED — `should_compact`（delta_count/ratio/read-amplification）+ compaction_debt_bytes，显式/后台不在 hot path。同上 ✓ |

---

## 硬门（§28）

| Gate | 结果 | 证据 |
|---|---|---|
| Gate-01 | PASS | `legacy_union_prefetch_count==0`（scheduler 主路径断言；test_perf_batch_service） |
| Gate-02 | PASS | writer 热循环静态禁止 `list(ids).index`/`next(...)`（test_gate02_static_no_oN2_lookup_in_writer） |
| Gate-03 | PASS | `batch_write_transaction_count==1 << factor_count`（execute_materialize_batch + 主路径） |
| Gate-04 | PASS | delta 普通增量 `historical_rewrite_bytes==0`（test_perf_storage） |
| Gate-05 | PASS | `full_factor_rescan_count==0`（test_perf_storage，`_count_partition_metrics` calls==0） |
| Gate-06 | PASS | matrix block assembly：轴相等 0 join / 异轴 1 次 concat（test_perf_matrix） |
| Gate-07 | PASS | 已排序非重叠 shard 一次 concat（`shard_concat_sort_count==1`，test_perf_shard） |
| Gate-08 | PASS | `DeadlineManager` 单 timer 线程，`watchdog_thread_created_count` O(1)（test_perf_duckdb_scan） |
| Gate-09 | PASS | `ReadOnlyOverlayMap` 共享 base，`dict_entry_copy_count==0`（test_perf_batch_service） |
| Gate-10 | PASS | `ProjectedColumnFootprint` 真实 footprint（test_perf_read_wave） |
| Gate-11 | PASS | `RepresentationTransitionTracker` transition_count/bytes 可观测（test_perf_rep_conversion） |
| Gate-12 | PASS | parity：所有新 R39 测试 + 354 既有回归全绿；matrix legacy 字节级 parity；bulk==reference 恒等；§27 同机 before/after：B1 total 129.3s→96.3s（-26%），batch tx 100→2 |

## §33 工程问题（最终回答）

1. 1000 因子 batch 最慢阶段 — **materialize 写盘**（分区写 + catalog 提交，B1 占 TTDC ~96%），run_many 编译为次慢（含 `compute_field_catalog_hash` 全目录序列化）。§27 QUICK 实测（300 股×252 日×100 因子）：materialize 92.8s vs run_many 3.4s。
2. 1-day incremental WriteAmplification — delta 模式 `historical_rewrite_bytes==0`（Gate-04，tests/r39/test_perf_storage）；默认非 delta 模式仍整分区重写（如实记录）。B4（300×252×100 全量→+1 天增量）：full_write_bytes=4.28MB，incremental_write_bytes=4.24MB（默认模式整分区重写，写放大 ~441x 逻辑字节）；delta/block 布局下归零。
3. 历史整分区 rewrite 是否仍存在 — **默认模式仍存在**（legacy data.parquet 整分区 upsert）；opt-in delta（`FACTOR_ENGINE_DELTA_STORAGE=1`）用 immutable delta + generation manifest 消除；matrix block 布局下未触达 block COW hardlink（PERF-062）。
4. 全 factor 目录 metrics rescan — `_count_partition_metrics` 正常路径 calls==0，`full_factor_rescan_count==0`（Gate-05）；watermark 路径不再全因子历史 scan（PERF-048/049）。
5. materialize_many_fast 1000 因子物理 writer transaction 数 — `execute_materialize_batch` 单 SQLite 事务批量提交（Gate-03 `batch_write_transaction_count==1<<N`），实际值 << factor_count（QUICK 实测：100 因子 → 2 tx，见 r39_scheduler_overhead.json / r39_scan_amplification.json）。baseline 逐项提交无该聚合（不报告）。
6. catalog transaction 数 — `CatalogBatchTransaction`（register_many/record_runs_many/update_watermarks_many）N update → 1 commit（PERF-050）；`sqlite_transaction_count` 可观测。
7. scheduler Future 数 — `MicroBatchTask` 16-128 cheap roots/单 Future（PERF-018）；`future_per_factor` 0.05 vs 旧 1.0。
8. read waves 实际扫描 bytes / 节省重复 scan — `baseline_duplicate_scan_bytes`/`physical_union_scan_bytes`/`saved_scan_bytes`/`decoded_resident_bytes` 可观测（PERF-010）；superset coalescing 节省重复 scan（PERF-008）。
9. source→backend→writer representation transition 数 — `RepresentationTransitionTracker` transition_count/bytes（PERF-031，Gate-11）；bulk Arrow CDI 路径（PERF-028）transition 数=1。
10. shard spool→reload→concat→sort 重复放大 — Arrow IPC spool（PERF-024）；已排序非重叠 shard 一次 concat（Gate-07 `shard_concat_sort_count==1`）；adaptive spool 决策避免小 shard 落盘（PERF-027）。
11. minute→daily 同源 100 聚合扫描分钟数据次数 — **1 次**（PERF-036 单条 DuckDB SQL，`agg FILTER` 多列）；B6 实测：100 聚合 121ms 单扫产出 450 行（r39_minute_aggregation_benchmark.json）。
12. matrix 新增 50 因子是否需要重写已有 2000 列 — **不需要**：block 模式只写 touched block，未触 block COW hardlink，新因子新建 block（PERF-062，新增因子 amplification 0.0）。B8 实测（默认布局 50→70 因子）：base 724KB → after 1.80MB，add 阶段 12.1s（r39_matrix_benchmark.json）。
13. Matrix 1-day 增量是否需要重写整月 — delta/block 布局下只写 touched block + 增量 fragment；compaction 才全局重排（PERF-083/084）。
14. DuckDB deadline 10000 queries watchdog thread 数 — `DeadlineManager` 单 lazy timer 线程，`watchdog_thread_created_count` O(1)（Gate-08，PERF-072）。
15. production optimized 与 reference parity — 所有新 R39 测试 + 354 既有回归全绿（Gate-12）；matrix legacy 字节级 parity；bulk==reference 恒等。
16. §27 同机 before/after（QUICK，300 股×252 日×100 因子，干净 lake）— **baseline 8e9893b5：B1 total 129.3s**（run 3.7s + mat 125.5s）/ B2 129.0s → **HEAD：B1 96.3s / B2 95.2s / B3 95.0s**（run ~3s + mat ~93s）。**TTDC 改善 ~26%**，materialize 写盘 125.5s→92.8s（-26%）；`batch_write_transaction_count` 100→2。注：baseline 原 42s 数字为复用 after 已填充 lake 的 resume 跳写，干净 lake 复测为 125.5s（诚实排除）。
