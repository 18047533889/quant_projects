# FactorEngine R39：全链路极速落值终审——扫描、转换、计算、调度、写放大与批量物化优化方案

> **用途**：本文件是一份可直接交给 coding AI 执行的增量整改提示词。  
> **审计基线**：`18047533889/quant_projects`，`main`，HEAD=`8e9893b562f80e083c4baed2e0a15eb4e040cdc9`。  
> **核心目标**：在不降低正确性、PIT、安全、证据、DQ、原子发布和资源治理要求的前提下，把 FactorEngine + DataAccess 的 **Time-to-Durable-Commit（TTDC，因子从提交计算到可靠落值完成）** 做到尽可能低。  
> **本轮性质**：只做 **R39 新增性能优化**。不要重复 R37/R38 已经处理的资源治理、PIT/参数认证、Evidence Truth、HostResourceCoordinator、AutoShard 合法性等整改；也不要重复上一轮已经列出的 75 个问题。若本文件与旧整改方向有接口交叉，以“新增性能语义”为准，不要把旧问题重新包装成新 issue。

---

# 0. 执行要求：不要只写分析，直接改代码

收到本文件后，请直接在最新 `main` 上实施，而不是只输出建议。

必须遵守：

1. **先读取最新代码再改**，不得按旧版本猜文件结构。
2. 所有优化都必须保持：
   - 相同 PIT 语义；
   - 相同 source snapshot / calendar / universe / price basis；
   - 相同 missing / NaN / Inf / dtype 语义；
   - 相同 operator semantics；
   - 相同 production admission；
   - 相同 durable commit / generation atomicity。
3. **禁止伪优化**：
   - 禁止关闭 DQ；
   - 禁止关闭 read-back / durability proof 而不提供等价证明；
   - 禁止删除 `fsync` 只为 benchmark 好看；
   - 禁止把 production 改成 research；
   - 禁止提高内存上限掩盖写法低效；
   - 禁止把 Pandas 换成 Polars 但仍做相同次数全量 copy / merge / rewrite；
   - 禁止只修改 cost metadata、不修改真实执行路径；
   - 禁止只新增 class / enum / telemetry，但主链仍走旧路径。
4. 每一项必须至少交付：
   - 真实代码改动；
   - 单元测试；
   - integration / benchmark；
   - 当前 HEAD 的 evidence；
   - before/after 指标。
5. 所有性能 benchmark 必须同时校验输出正确性，不能只比时间。

---

# 1. R39 的性能北极星：TTDC，而不是单个算子的 microbenchmark

## 1.1 定义

整批因子落值的主指标：

```text
TTDC =
    Compile
  + Batch Planning
  + Source Resolution
  + Physical Scan
  + Decode
  + Representation Conversion
  + Compute
  + Scheduler / Queue Wait
  + Result Normalization
  + DQ
  + Serialization / Compression
  + Filesystem / Object-store Write
  + Durability
  + Catalog / Manifest Commit
```

优化时必须记录以下“放大系数”：

```text
ScanAmplification
  = physical_scanned_bytes / minimum_required_source_bytes

ConversionAmplification
  = bytes_crossing_representation_boundary / final_output_bytes

ComputeAmplification
  = actual_operator_work / minimum_semantically_required_work

WriteAmplification
  = physical_bytes_written / changed_logical_output_bytes

RewriteAmplification
  = historical_bytes_rewritten / changed_logical_output_bytes

MetadataAmplification
  = metadata_or_footer_bytes_read / changed_logical_output_bytes

SchedulerAmplification
  = scheduled_task_count / minimum_coalesced_execution_units

MaterializationAmplification
  = temporary_intermediate_bytes / final_durable_bytes
```

## 1.2 必须新增统一 PerformanceRunSummary

建议新增：

```python
@dataclass
class PerformanceRunSummary:
    factor_count: int
    compile_ms: float
    planning_ms: float
    scan_ms: float
    compute_ms: float
    conversion_ms: float
    writer_wait_ms: float
    serialize_ms: float
    fsync_ms: float
    catalog_commit_ms: float
    total_ttdc_ms: float

    scan_bytes: int
    minimum_required_scan_bytes: int
    conversion_bytes: int
    output_bytes: int
    write_bytes: int
    rewrite_bytes: int
    metadata_scan_bytes: int

    future_count: int
    physical_query_count: int
    parquet_file_open_count: int
    parquet_file_write_count: int
    fsync_count: int
    sqlite_transaction_count: int
    full_factor_rescan_count: int
    watchdog_thread_created_count: int
```

不要依赖日志文本反推这些指标，关键路径直接打结构化计数。

---

# 2. 第一优先级：先消灭“明明批量执行，却又提前做一次全量准备”的重复工作

## R39-P0-PERF-001：Adaptive Scheduler 主路径前仍调用 legacy full-union batch prefetch

### 当前问题

当前 `runtime/batch_service.py` 中：

- `_execute_run_many_scheduler()` 明确声明 scheduler 主路径不再使用 full-union prefetch，应该由 `BatchDataRequest + ReadWave` 成为唯一读取控制面；
- 但 `execute_run_many()` 在决定是否进入 scheduler 之前仍先执行 `_maybe_prepare_batch_data(...)`；
- `execute_run_many_parallel()` 也在进入 scheduler 前做同样准备。

这会导致：

1. 全批列 union 先被准备一次；
2. scheduler 又规划 read waves；
3. 即使底层 cache 避免第二次磁盘 scan，也已经付出了：
   - 全 union 的路径；
   - 全 union 的列 materialization；
   - 大工作集内存；
   - 可能的 DQ；
   - source cache 污染；
   - read-wave 内存有界设计被破坏。

### 必须修改

先决定 control plane：

```python
mode = resolve_batch_execution_control_plane(...)
if mode == "adaptive_scheduler":
    # 不调用 _maybe_prepare_batch_data
    input_report = None
    return _execute_run_many_scheduler(...)
else:
    input_report = _maybe_prepare_batch_data(...)
```

`execute_run_many_parallel()` 同样处理。

### 新 invariant

```text
ADAPTIVE_SCHEDULER_LEGACY_UNION_PREFETCH_COUNT == 0
```

### 测试

构造 100 个因子：

- 50 个依赖 `close, volume`
- 50 个依赖 `fundamental_x`
- 不同 lookback

mock source 统计：

- `prefetch_columns`
- `load_columns`
- DataAccess scan

确认 adaptive scheduler 进入 read wave 前不存在额外全列 union read。

---

# 3. BatchDataRequest：真正做成 multi-source，而不是“名义 multi-source”

## R39-P0-PERF-002：secondary SourceRef 分组的 typed binding 应直接由列解析得到

### 当前问题

`planner/batch_data_request.py` 当前 secondary source 发现逻辑存在明显的类型混用风险：

- `_source_ref_datasets()` 的 key/value 实际含义与 `_source_ref_dataset_for()` 的消费方式不一致；
- source-ref 列、dataset、market 被不同函数通过字符串 dict 间接匹配；
- 这不仅可能分错 source，也会导致 secondary 字段被错误塞进 anchor group，造成过度扫描。

### 必须重构

不要再使用：

```text
dict[str, str]
```

承载多个语义。

新增：

```python
@dataclass(frozen=True)
class ColumnSourceBinding:
    encoded_column: str
    dataset: str
    field: str
    market: str
    source_scope: SourceScopeId
```

扫描 PlanNode 时直接：

```python
binding_by_column[encoded_column] = ColumnSourceBinding(...)
```

随后：

```python
binding = binding_by_column.get(column_name)
if binding:
    group_by_scope[binding.source_scope].add(binding.field)
else:
    anchor_group.add(column_name)
```

### Hard Gate

所有 SourceRef：

```text
SOURCE_REF_WITHOUT_TYPED_SOURCE_BINDING == 0
```

---

## R39-P0-PERF-003：secondary source 不能继续用 anchor source 的 cost estimator

### 当前问题

`build_batch_data_request()` 当前获取：

```python
estimator = getattr(source, "estimate_scan_cost", None)
```

之后对所有 source group 使用这一 estimator。

如果 batch 同时包含：

- 日频行情；
- 财务；
- 行业；
- 分钟；
- universe；

secondary source 的扫描成本不应由 anchor source 来估。

### 必须修改

实现：

```python
class BatchSourceResolver:
    def resolve_source(scope: SourceScopeId) -> DataSourceAdapter
```

每个 `SourceScanGroup` 都绑定：

```python
source_adapter
cost_estimator
storage_kind
snapshot_id
```

调用：

```python
group.source_adapter.estimate_scan_cost(
    dataset=group.dataset,
    fields=group.fields,
    time_range=group.time_range,
    instruments=group.instrument_scope,
)
```

### Acceptance

mixed-source benchmark 中每个 dataset 都产生独立 ScanCost evidence，不能全部来自同一个 anchor adapter。

---

## R39-P0-PERF-004：保留 end-only / start-only 时间窗，不允许因为 start=None 退化全量扫描

### 当前问题

BatchDataRequest 构造 group 的 time_range 时，应完整保留：

```text
(None, end)
(start, None)
(start, end)
```

任何一端为空都不是“没有时间过滤”。

### 必须修改

统一 typed：

```python
@dataclass(frozen=True)
class TimeRange:
    start: Timestamp | None
    end: Timestamp | None
```

禁止依赖：

```python
if time_range[0] is not None
```

决定整个 range 是否存在。

### 测试

- `start=None, end=2025-12-31`
- `start=2025-01-01, end=None`

都必须产生 partition / row-group prune。

---

# 4. ReadWave：从“固定 500k×8B”升级为真正的物理 footprint optimizer

## R39-P0-PERF-005：ROOT / CSE_SHARED 不应该作为零列 scan request 创建 wave

### 当前问题

`build_waves_from_dag()` 当前会把：

- SOURCE_SCAN
- CSE_SHARED
- ROOT

都注册进 `ReadWavePlanner`。

后两类使用：

```text
columns=()
scan=0
memory=0
```

这会：

- 产生空 wave 或无意义 request；
- 增加 planner/scheduler bookkeeping；
- 模糊 “wave.task_ids” 到底是 source scan 还是 consumer。

### 必须修改

ReadWave 只包含真实 SOURCE_SCAN。

消费者单独表达：

```python
ReadWave(
    source_tasks=...,
    consumer_tasks=...,
)
```

consumer 由 dependency graph 的 downstream closure 计算。

### Hard Gate

```text
ZERO_COLUMN_READ_WAVE == 0
NON_SOURCE_TASK_IN_READ_WAVE_SOURCE_TASKS == 0
```

---

## R39-P0-PERF-006：wave memory packing 必须消费真实 projection bytes

### 当前问题

虽然 `register_scan_task()` 接收 `estimated_memory_bytes`，实际 wave packing 又回到：

```python
_default_scan_bytes(union_cols, rows_estimate)
```

固定：

- 500k rows；
- 8B/column。

这会同时导致：

- float64 / string / large panel 低估；
- 小窗口/少股票高估；
- wave 切得太大造成 spill/OOM；
- 或切得太碎损失 scan reuse。

### 必须新增

```python
@dataclass(frozen=True)
class ProjectedColumnFootprint:
    name: str
    decoded_bytes: int
    compressed_scan_bytes: int
    null_bitmap_bytes: int
    offsets_bytes: int
    dictionary_bytes: int
```

wave memory：

```text
axis_bytes
+ union(decoded column bytes)
+ Arrow/Polars metadata
+ downstream_live_reserve
+ output_reserve
```

真实数据优先来自：

1. manifest column stats；
2. parquet row-group metadata；
3. ScanCost；
4. 最后才 fallback heuristic。

---

## R39-P0-PERF-007：marginal-overlap 评分当前应改成真正的“边际成本”

### 当前问题

当前 `_best_marginal()` 命名为 marginal，但 denominator 使用的是整个 union 大小，而不是新增大小。

应改：

```python
incremental_live_bytes = bytes(union) - bytes(current)
```

收益：

```text
avoided_scan
+ avoided_decode
+ avoided_open
+ avoided_conversion
```

成本：

```text
incremental_live_bytes
+ memory_rent * expected_lifetime
+ extra_scan_due_to_superset
+ scheduling_delay
```

score：

```text
benefit / max(cost, epsilon)
```

首轮 greedy 后允许一次 bounded local-search / bin-swap，解决明显坏 packing。

---

## R39-P0-PERF-008：允许“重叠时间窗”进行 cost-based superset coalescing

### 当前问题

当前 wave scope key 把完整 `time_range` 作为 exact key。

所以：

- 20d；
- 60d；
- 120d；

即使来源完全相同，也绝不会进入同一 wave。

这能避免 full-history 污染短因子，但也错过大量便宜的 superset reuse。

### 必须改成

不是无脑 union，而是评估：

```text
Cost(separate scans)
vs
Cost(one superset scan + slice views)
```

只有：

```text
superset_extra_scan_cost < duplicate_open_decode_conversion_saved
```

才合并。

consumer 使用 zero-copy time slice view。

---

## R39-P0-PERF-009：instrument_scope / universe_id 必须进入 wave compatibility 和成本模型

### 当前问题

ReadWave request 已保存：

- `instrument_scope`
- `universe_id`

但 packing 的核心 key / score 并没有真正消费它们。

### 修改

引入：

```python
ScopeCompatibility:
    EXACT
    UNION_COMPATIBLE
    INCOMPATIBLE
```

对于不同股票池：

- 如果 union 很小，可共享 scan 后 consumer slicing；
- 如果一个全 A、一个 CSI300，通常不应为了共享把小请求扩大成全 A，除非成本模型证明有收益。

### Acceptance

报告：

```text
instrument_scope_union_extra_bytes
universe_scope_union_extra_rows
```

---

## R39-P0-PERF-010：区分 baseline duplicate scan bytes 与真实 physical scan bytes

### 当前问题

ReadWave 当前的 `estimated_scan_bytes` 容易表达为各 request scan 成本求和，而实际 wave 只进行一次 union scan。

必须拆成：

```python
baseline_duplicate_scan_bytes
physical_union_scan_bytes
saved_scan_bytes
decoded_resident_bytes
```

资源 admission 只按：

```text
physical_union_scan_bytes
```

申请 IO token，不能按 baseline sum 过度限流。

---

## R39-P0-PERF-011：read wave 输出不能固定为 pandas-column cache

### 当前问题

`SourceWaveExecutor` 当前最终把 source wave 描述为类似：

```text
representation = "pandas_column"
location = "source.column_cache"
```

这样即使下游是：

- DuckDB native；
- Polars lazy；
- NumPy/Numba block；

read wave 也先落到 column-cache/Pandas 语义，再转一次。

### 新设计

```python
class SourceRepresentation(Enum):
    DUCKDB_RELATION
    ARROW_BATCH_BLOCK
    POLARS_LAZY
    NUMPY_COLUMN_BLOCK
    PANDAS_COLUMNS   # reference fallback
```

`ReadWave` 加：

```python
preferred_representation
consumer_backend_mask
```

`SourceWaveExecutor` 直接产出 `BufferRef`。

原则：

```text
source-native → largest native compute region → terminal writer
```

而不是：

```text
source → pandas → native → pandas → writer
```

---

## R39-P0-PERF-012：read-wave 失败不能退化成 N 个 root 自己重复读

### 当前问题

coalesced wave 一旦失败，如果后续 root 回到 on-demand read，会发生：

```text
1 次失败的大 scan
+ N 次 root 重复 scan
```

### 必须做 typed recovery

```text
OVERSIZED
    → split wave 2-way

REPRESENTATION_UNSUPPORTED
    → downgrade representation only

TRANSIENT_IO
    → bounded retry

MEMORY_PRESSURE
    → shrink wave

PERMANENT_SOURCE_ERROR
    → abort
```

禁止 uncontrolled per-root fallback。

### Hard Gate

```text
FAILED_WAVE_TO_N_ROOT_SCANS == 0
```

---

# 5. ExecutionContext 与 Python control-plane：把 O(N×shared) 降到 O(N)

## R39-P0-PERF-013：每个 root 不要复制整份 shared cache dict

### 当前问题

`_execute_root_with_path()` 当前为每个 root：

```python
dict(ctx.shared_result_cache)
dict(ctx.shared_long_lazy_cache)
dict(ctx.materialized_long_lazy)
dict(ctx.materialized_series)
```

如果：

```text
5000 roots × 1000 shared keys
```

仅复制 dict 就是数百万级 Python hash-entry work。

### 新设计

```python
class ReadOnlyOverlayMap:
    base: Mapping
    local: dict
```

读：

```text
local → base
```

写只进 local。

或者：

```python
BatchSharedState(frozen)
RootScratchState(mutable)
```

root 不再 clone base。

### Benchmark

5000 trivial roots：

```text
context_creation_ms
allocated_python_bytes
dict_entry_copy_count
```

必须接近 O(root_count)。

---

## R39-P0-PERF-014：warmup planning 全 batch 只算一次

### 当前问题

不同路径会重复：

```python
prepare_run_warmup(...)
```

例如：

- cost clustering；
- batch union warmup；
- recursive wave。

### 新增

```python
@dataclass(frozen=True)
class BatchWarmupPlan:
    per_factor: dict[str, RunWindow]
    groups: list[WarmupGroup]
```

一次计算：

```text
analysis/history requirement
→ factor RunWindow
→ scan-cost-aware grouping
```

后续所有路径只消费这一个结果。

---

## R39-P0-PERF-015：warmup clustering 不再使用固定 short/medium/long 比例桶

当前按 lookback / total rows 的固定比例分类只适合粗 heuristic。

改为：

```text
合并两个 group 的额外 source scan bytes
vs
分开 scan 的重复成本
```

使用真实 partition boundary、manifest bytes、source scope。

目标是：

```text
min(scan bytes + open/decode + duplicated compute)
```

而不是 lookback ratio。

---

## R39-P0-PERF-016：PlanCost / scheduling hints / path eligibility 不要运行后重复遍历整棵 DAG

当前 scheduler planning 已经计算 cost，运行结束又可能：

- `estimate_plan_cost(fp.root)`
- `summarize_plans`
- `derive_scheduling_hints`
- production fast-path 再 collect ops

### 改为

编译阶段产生：

```python
PlanExecutionCertificate:
    structural_hash
    bound_ops
    cost
    backend_eligibility
    output_shape
    history
    estimated_memory
```

task 持证书。

运行时：

```text
O(1) read certificate
```

最终 report 复用 planner 已有 cost，不重新 walk plan。

---

## R39-P0-PERF-017：output warmup trim 下推，禁止每个 factor 结果再做 Pandas copy/slice

### 当前

root 算完后：

```python
_trim_batch_result(...)
```

逐 factor slicing。

### 改法

Plan 输出契约带：

```python
OutputSlice(start_offset, end_offset)
```

native backend：

- full warmup 仍参与计算；
- terminal result 返回 zero-copy slice/view。

writer 接受：

```python
BufferRef + slice
```

尽量不产生新的 Series。


---

# 6. 调度：大批 tiny factors 不应该等于大量 Python Future

## R39-P0-PERF-018：增加 Scheduler MicroBatchTask

### 场景

10000 个低成本：

```text
rank/add/div/lag/简单 rolling
```

即使整个 batch 不满足 DIRECT_VECTOR，小 root 一个 Future 也非常浪费。

### 新结构

```python
MicroBatchTask(
    roots=(...),
    same_backend=True,
    same_axis=True,
    same_source_buffers=True,
    estimated_total_work < threshold,
)
```

一个 Future 内连续执行 16–128 个 cheap roots。

这和 backend native fusion 不同：

- native fusion = 一个物理表达式/SQL；
- microbatch = scheduler dispatch coalescing。

### 指标

```text
future_count / factor_count
scheduler_cpu_ms / TTDC
```

---

## R39-P1-PERF-019：scheduler 等待改成 completion/resource/sink event，不依赖周期 polling

统一 event source：

```text
FutureCompleted
ResourceLeaseReleased
WriterQueueSpaceAvailable
ReadWaveCompleted
ControllerDecisionChanged
```

scheduler 阻塞在 Condition/EventMux。

定时 timeout 只用于：

- watchdog；
- deadlock detection；

不能成为正常调度节拍。

---

## R39-P1-PERF-020：GIL-heavy legacy root 使用 PlanId worker lane，而不是 pickle 整个 ctx

不要直接把巨大：

- PlanNode；
- DataSource；
- ExecutionContext；
- cache dict；

pickle 到 process。

设计：

```python
WorkerRuntimeRegistry:
    plan_id -> compiled executable
    buffer_ref_id -> shared mmap/Arrow ref
    source_session_id -> prepared read session
```

dispatch：

```python
ProcessTask(plan_id, buffer_refs, parameter_block)
```

目标：

- legacy Python/GIL 算子不阻塞 scheduler thread；
- 同一 worker warm cache；
- 避免大型对象 serialization。

---

# 7. Native Fusion：当前要从“能融合”升级成“融合规模可自动寻优”

## R39-P0-PERF-021：重写 adaptive_fusion_block_size 的单位模型

当前 fusion block 的 complexity/output/budget 比较存在维度不一致风险。

必须换成可测量变量：

```text
sql_chars
ast_node_count
window_expression_count
projection_count
estimated_intermediate_bytes
compile_ms_model
optimizer_ms_model
output_bytes
```

目标：

```text
min(
    compile_ms
  + execute_ms
  + output_materialization_ms
)
```

block 不必只限制 32/64/128/256，可保留 bucket，但由校准模型选。

---

## R39-P0-PERF-022：fusion 失败不要直接全组退成 per-root

### 新恢复策略

如果 128 roots fused fail：

```text
128
→ 64 + 64
→ 必要时 32...
```

仅失败子组继续拆。

维护：

```python
NegativeFusionCacheKey(
    backend,
    backend_version,
    plan_family_hash,
    parameter_shape,
    source_shape,
)
```

已知失败组合后续不再先付一次失败成本。

---

## R39-P1-PERF-023：fusion grouping 加入 kernel/window locality

同 backend/source 还不够。

优先把：

- 相同 rolling window；
- 相同 rank/group scope；
- 相同 source columns；
- 相同 derived primitive；

聚在一起。

目标不仅共享 scan，还共享：

- sort；
- rolling state；
- local expressions；
- window frame；
- group partition。

---

# 8. Shard：当前 spool/merge 本身可能成为第二个瓶颈

## R39-P0-PERF-024：spool 不要 Pandas→Parquet→Pandas

当前大 shard：

```text
Series/DataFrame
→ to_parquet
→ pd.read_parquet
```

临时 spool 追求的是快速、低内存，不是长期压缩率。

### 新实现

优先：

```text
Arrow IPC stream/file
+ memory map
```

或：

```text
uncompressed / light-compression Arrow
```

`SpooledShard` 保存：

```python
ArrowSpoolRef(
    path,
    schema,
    row_count,
    sortedness_certificate,
)
```

---

## R39-P0-PERF-025：merge 禁止每加一个 shard 就 concat+sort 一次

当前模式：

```python
acc = concat(acc, chunk)
acc = sort_index(acc)
```

K 个 shard 会产生重复 copy/sort。

### 改法

如果 merge contract 已证明：

- shard ranges 不重叠；
- 每 shard 内已排序；
- merge_order 正确；

则：

```python
pd.concat(all_chunks, copy=False)
```

一次即可，甚至无需 sort。

更大结果使用：

```text
k-way streaming merge
```

---

## R39-P0-PERF-026：能直接落盘的 shard 不要先 merge

增加：

```python
ShardMergeMode:
    DIRECT_DURABLE_APPEND
    CONCAT_ONLY
    ORDERED_MERGE
    REDUCE_STATE
```

time shard / asset shard 如果 final storage 分区兼容，可：

```text
shard完成
→ trim overlap
→ writer
```

merge task 只提交 manifest，不重建大内存结果。

---

## R39-P1-PERF-027：spool threshold 自适应

512MB 固定阈值应改成：

```text
live_headroom
writer_queue_headroom
spool_disk_throughput
estimated_merge_lifetime
```

如果 writer 很快，直接写；

如果 merge 很近且内存足，保留；

否则 spool。

---

# 9. 表示层：真正做到 Arrow / Polars / DuckDB / NumPy 零拷贝链

## R39-P0-PERF-028：重写 panel_to_polars / polars_to_panel 的 copy-heavy 路径

当前 wide Pandas → Polars 会逐列：

```python
panel[c].to_numpy()
```

再构造 dict。

反向又：

```python
result.select(...).to_numpy()
→ pd.DataFrame
```

### 必须做

优先 Arrow C Data Interface：

```text
Pandas/Arrow owner
→ pa.Table / RecordBatch
→ Polars from_arrow
```

如果是同 dtype contiguous numeric block：

```text
single 2D block transfer
```

禁止 per-column Python loop 成为批量 root 热路径。

---

## R39-P0-PERF-029：PanelIdentity/AxisIdentity 只生成一次并传播

当前 identity 校验不应该每次：

- time `to_list()`；
- MultiIndex levels/codes hash；
- 大量 repr/SHA。

新增：

```python
AxisIdentityCertificate:
    source_snapshot_id
    axis_hash
    datetime_count
    instrument_count
    sortedness
```

axis-preserving operator：

```text
certificate passthrough
```

axis-changing operator 才重算。

### 指标

```text
axis_identity_ms / TTDC < 1%
```

---

## R39-P0-PERF-030：引入 FactorBlockRef，多个因子共享一份 axis

建议：

```python
@dataclass(frozen=True)
class FactorBlockRef:
    axis_ref: AxisBufferRef
    factor_ids: tuple[str, ...]
    values_ref: BufferRef      # rows × factors
    dtype: str
    row_count: int
    factor_count: int
    validity_ref: BufferRef | None
```

好处：

- 不重复 MultiIndex；
- 不重复 ticker string；
- writer queue 少 Python object；
- 多因子一次 Arrow/Parquet write；
- matrix 天然接收。

---

## R39-P0-PERF-031：native representation 必须跨多个相邻 root 保持，不在每 root 边界强制回 Pandas

新原则：

```text
DuckDB relation
  → DuckDB fused subtree
  → Arrow/Polars block
  → Polars/Numba subtree
  → FactorBlock writer
```

只有真正需要 Pandas reference 的 root 才回 Pandas。

新增：

```text
representation_transition_count
representation_transition_bytes
```

作为 release KPI。

---

## R39-P1-PERF-032：临时数值 buffer 使用 arena/pool，减少重复 malloc/free

针对 NumPy/Numba：

```python
TemporaryArrayArena(
    shape_bucket,
    dtype,
)
```

对滚动/排名/neutralize 的 scratch：

- 用完 release；
- 相同 shape 复用；
- 不跨 semantic result 生命周期。

必须防止别名修改 final output。

---

# 10. 分钟→日：这是 A 股因子挖掘中最容易形成 TB 级重复扫描的地方

## R39-P0-PERF-033：AggregationSpec 只 parse 一次

当前 bundle 上层已 parse items，SQL builder 内又再次 parse。

改为：

```python
ParsedAggregationItem
```

从 validation 到 SQL builder 一路传递。

重复输出名检查使用 `Counter`，不要 `list.count()` O(N²)。

---

## R39-P0-PERF-034：timezone 转换只做一次；HH:MM 使用整数分钟而不是字符串

当前 SQL 应重构为：

```sql
WITH _raw AS (
  SELECT
    timezone(...) AS _local,
    inst,
    v1, v2 ...
  FROM source
),
_clock AS (
  SELECT
    CAST(_local AS DATE) AS _date,
    EXTRACT(HOUR FROM _local) * 60 + EXTRACT(MINUTE FROM _local) AS _minute,
    ...
  FROM _raw
)
SELECT ...
```

过滤：

```text
09:31 => 571
14:30 => 870
```

不要每行 `strftime('%H:%M')` 后做字符串比较。

---

## R39-P0-PERF-035：同一 minute filter signature 的多个输出共享 condition

例如几十个：

```text
09:31-10:00
14:30-15:00
last 30m
```

先 canonicalize：

```python
FilterSignature
```

SQL 中 condition 只定义一次。

必要时先产生 bucket / session-slot id，再做 conditional aggregation。

---

## R39-P1-PERF-036：minute aggregate → daily factor 能下推的继续留在 DuckDB

目标：

```text
minute parquet
→ filter
→ daily aggregate
→ simple daily arithmetic/rolling
→ factor output
```

尽可能在一条/一个 native region 内完成。

不要：

```text
minute → daily Arrow → Pandas Series → FactorEngine → writer
```

---

## R39-P1-PERF-037：aggregate_minute_bundle 支持 relation/stream output

增加：

```python
result_mode = "relation" | "arrow_stream" | "arrow"
```

FE 如果立即消费，不需要先把完整 daily aggregate table 全量 materialize 成 Arrow Table。

---

# 11. materialize_many_fast：现在最大的“假 batch”位置之一

## R39-P0-PERF-038：writer 内的 factor / id 查找改成 O(1)

当前每个 item：

```python
list(ids).index(item.name)
next((f for f in factors if f.name == item.name), None)
```

必须预建：

```python
factor_by_name = {f.name: f for f in factors}
factor_id_by_name = {
    factor.name: fid
    for factor, fid in zip(factors, ids)
}
```

writer O(1)。

这也解决自定义 `factor_id != factor.name` 时的身份映射歧义。

---

## R39-P0-PERF-039：StreamingResultSink 的 batch 必须变成“物理批量写”，不是 batch 内继续逐因子 execute_materialize

### 当前

writer 收：

```python
batch: list[ResultItem]
```

但：

```python
for item in batch:
    execute_materialize(...)
```

所以即使 `writer_batch_size=64`：

- 仍 64 次 materializer；
- 64 次 normalize；
- 64 次 catalog；
- 64 次 partition write；
- 64 轮 metadata。

### 新 API

```python
execute_materialize_batch(
    engine,
    items: list[MaterializeItem],
    generation: GenerationTransaction,
)
```

批量做：

1. lineage/certificates vectorized；
2. FactorBlock normalization；
3. 按 storage partition 聚类；
4. 批写；
5. batch catalog commit；
6. 单 generation publish。

---

## R39-P0-PERF-040：writer batch size 必须按 bytes + partition locality 自适应

不是只按 factor_count。

flush 条件：

```text
block_bytes >= target_bytes
OR factors >= max_factors
OR oldest_item_age >= latency_budget
OR partition_group changes materially
```

初始可设：

```text
64–256 MiB logical block
```

但必须根据 benchmark 自动校准。

---

## R39-P0-PERF-041：writer_threads 不能永远默认 1，也不能无脑加线程

自动选择：

```text
local NVMe:
    several writers, partition-disjoint

network/object store:
    concurrency based on throughput/latency

same partition:
    single writer
```

使用 HostResourceCoordinator writer lease，但本项只关注 throughput routing。

---

## R39-P0-PERF-042：静态 hash partition→writer 改为动态 least-loaded ownership

保持：

```text
same partition never concurrent write
```

但第一次出现 partition 时：

```python
owner = least_loaded_writer()
```

之后 pin。

partition epoch 完成后可释放 ownership。

避免多个热点 partition hash 碰到同一个 worker。

---

# 12. ParquetMaterializer：真正的核心瓶颈是“增量写却重写历史”

## R39-P0-PERF-043：停止 hot path 的 full-partition read→concat→dedup→sort→rewrite

### 当前 long upsert

每个触达 partition：

```text
pd.read_parquet(old whole partition)
→ concat(old,new)
→ drop_duplicates
→ sort_values(asset,datetime)
→ write whole partition
```

对于“1 天增量写 5 年历史因子”：

```text
changed rows 很小
rewrite rows 很大
```

### 新物理模型：Base + Delta

```text
factor partition
├── base/
│   └── base_<compaction_gen>.parquet
├── delta/
│   ├── gen_001.parquet
│   ├── gen_002.parquet
│   └── ...
└── manifest.json
```

每条 delta 带：

```text
datetime
asset
value
is_valid
generation
operation (UPSERT/TOMBSTONE)
```

正常增量：

```text
只写 changed rows
```

read：

```sql
row_number() over (
  partition by datetime, asset
  order by generation desc
)
= 1
```

后台/阈值 compaction 合并 base+deltas。

---

## R39-P0-PERF-044：WriteAmplification 必须成为物化器硬 KPI

记录：

```text
logical_changed_bytes
physical_new_write_bytes
historical_rewrite_bytes
delta_file_count
compaction_debt_bytes
```

目标：

普通 1-day incremental：

```text
historical_rewrite_bytes == 0
```

在不触发 compaction 时：

```text
WriteAmplification <= 3~5x
```

具体阈值按编码/compression benchmark确定。

---

## R39-P0-PERF-045：wide storage 不再作为 mutation hot format

当前 wide 更新：

```text
read wide
→ unpivot
→ concat/dedup
→ pivot
→ rewrite wide
```

这种格式适合 read-serving，不适合高频增量 mutation。

规则：

```text
canonical mutation format = append-only long/delta or FactorBlock
wide = compaction/materialized view
```

---

## R39-P0-PERF-046：materializer terminal 输入从 pd.Series 升级为 Arrow/FactorBlock

新增：

```python
materialize_block(
    block: FactorBlockRef | pa.Table | RecordBatchReader
)
```

仍保留：

```python
materialize(pd.Series)
```

作为兼容 reference。

生产 fast path：

```text
backend output
→ FactorBlock/Arrow
→ writer
```

不要每因子：

```text
Series.reset_index()
astype("string")
DataFrame.copy()
loc assignments
```

---

## R39-P0-PERF-047：tombstone/finite mask 在 Arrow batch 中直接生成

使用：

```text
is_finite
if_else
cast
dictionary encode
```

production metadata 常量列用 Arrow scalar/dictionary，不为每一行构造 Python string object。

---

# 13. 当前存在明确的落值后“双重全目录扫描”

## R39-P0-PERF-048：禁止 `_count_total_rows()` + `_count_partition_metrics()` 双扫

当前逻辑：

```python
total_rows = self._count_total_rows(factor_dir)
partition_metrics = self._count_partition_metrics(factor_dir)
```

而 `_count_total_rows()` 内部又调用 `_count_partition_metrics()`。

因此一个 materialize 正常可能把 factor 目录完整统计两次。

必须第一步立刻改成：

```python
partition_metrics = self._count_partition_metrics(...)
total_rows = partition_metrics["physical_row_count"]
```

但这只是止血。

---

## R39-P0-PERF-049：正常写路径彻底禁止为了统计而 pd.read_parquet 全因子历史

### 正确设计

每次 partition commit 返回：

```python
PartitionCommitStats:
    rows_before
    rows_after
    rows_added
    rows_replaced
    valid_before
    valid_after
    min_date
    max_date
    file_bytes
```

catalog / manifest 事务增量更新 aggregate stats。

需要 rebuild 时：

优先：

```text
Parquet footer / metadata
DuckDB parquet_metadata
```

而不是加载 value cells。

### Hard Gate

```text
NORMAL_MATERIALIZE_FULL_FACTOR_RESCAN_COUNT == 0
```


---

# 14. Catalog：5000 个因子不应该对应 5000 组独立 SQLite 提交

## R39-P0-PERF-050：新增 CatalogBatchTransaction

对一批 factor：

```python
with catalog.batch_transaction(generation_id):
    register_many(...)
    record_runs_many(...)
    update_watermarks_many(...)
    update_partition_stats_many(...)
```

使用 prepared statements / executemany。

确保：

```text
数据 durable
→ batch catalog transaction
→ generation visible
```

避免每 factor transaction / fsync。

---

## R39-P1-PERF-051：避免写后立即 get_watermark 的重复查询

如果 transaction 已知本次写入的新 watermark：

直接从 commit result 返回。

只有 CAS / external modification 场景才需要 read-back。

---

# 15. 文件级 durability：不是“删 fsync”，而是通过 immutable generation 减少 fsync 次数

## R39-P0-PERF-052：从“每因子覆盖一个 data.parquet”改为 immutable delta + generation manifest

当前：

```text
tmp write
fsync file
replace
fsync dir
```

对 10000 factor partition 会产生大量 sync。

新模型：

1. 多个 immutable delta file 并行写；
2. 文件 durable；
3. generation manifest 单次切换；
4. catalog commit。

仍保证 crash safety，但不需要不断改写同一大文件。

---

## R39-P1-PERF-053：orphan tmp cleanup 从“每次 partition write 扫一次”改为 generation recovery

temp / delta 名字带：

```text
run_generation
writer_id
sequence
```

启动/恢复时一次扫描未提交 generation。

正常每次写 partition 不需要 `_cleanup_orphan_tmp_files()` 反复扫目录。

---

## R39-P1-PERF-054：partition lock 文件描述符复用

新增 `PartitionLockManager`：

- active generation 内复用 lock handle；
- batch 完成统一 release；
- 外部并发语义不变。

但在 append-only generation 模型下，同 generation 内部可以更多依靠 orchestrator ownership，而不是每 cell batch 重入 flock。

---

# 16. 文件布局：不要让“因子数量”线性变成“小文件数量”

## R39-P0-PERF-055：引入 FactorBlockLake layout

现有逻辑 API 可继续是：

```text
factor_id → Series
```

物理写可改为 block：

```text
factor_blocks/
  date_bucket=2026-08/
    block=<id>/
      data.parquet
```

两种候选：

### Layout A：long block

```text
datetime
instrument
factor_id
value
generation
```

### Layout B：column block

```text
datetime
instrument
factor_0001
factor_0002
...
factor_0256
```

由真实读取模式 benchmark 决定。

Catalog 维护：

```text
factor_id → block_id / column
```

---

## R39-P1-PERF-056：增加 LakeWriteLayoutPolicy

根据：

```text
factor_count
update_frequency
read pattern
matrix training demand
object store/local
```

选择：

```text
SINGLE_FACTOR_DELTA
FACTOR_BLOCK_LONG
FACTOR_BLOCK_WIDE
MATRIX_COLUMN_BLOCK
```

不要一个 layout 统治全部 workload。

---

# 17. Parquet encoding / row group：按 TTDC 自动调，不凭经验写死

## R39-P1-PERF-057：直接使用 PyArrow ParquetWriter / RecordBatch，不经过 Pandas to_parquet

实现：

```python
BatchParquetWriter.write_batches(reader_or_batches)
```

支持：

- row-group streaming；
- schema 固化；
- statistics；
- dictionary encoding；
- compression options。

---

## R39-P1-PERF-058：建立 storage tuning benchmark

测试：

```text
SNAPPY
ZSTD-1
ZSTD-3
uncompressed(temp/spool)
```

测试 row group：

```text
64MB
128MB
256MB
```

必要时测试：

```text
BYTE_STREAM_SPLIT
dictionary columns
```

优化目标必须是：

```text
write CPU + bytes + later read cost
```

不是只追求最小文件。

---

# 18. DQ / Validation：不降低强度，但必须 single-pass

## R39-P0-PERF-059：DQ statistics 在 result emission / writer 过程中顺便算

对于每 batch 同步积累：

```text
row_count
finite_count
nan_count
inf_count
min/max
checksum
date_min/max
duplicate key stats
```

不要：

```text
compute
→ write
→ 再扫 DataFrame 做 DQ
→ 再扫 parquet 做 metrics
```

---

# 19. FactorMatrix：当前 pairwise outer merge 是另一处非常大的 Python 热点

## R39-P0-PERF-060：禁止按 factor 逐个 pandas outer merge 构造 matrix

当前模式：

```text
factor1 long
merge factor2
merge factor3
...
```

即使每 256 个分块，block 内和 block 间仍是 pairwise merge。

### 新方案

如果所有结果 axis 相同：

```text
AxisIdentityCertificate equality
→ 直接 column-stack
```

无需 join。

如果 axis 不完全相同：

1. 构造一次 canonical row key index；
2. 每 factor vectorized reindex；
3. column block assembly。

或使用：

```text
DuckDB/Polars pivot / join plan
```

但不能 N 次 Pandas merge。

---

## R39-P0-PERF-061：matrix 物理布局使用 factor-column blocks，禁止 2000/10000 列巨型单 parquet

例如：

```text
year=2026/month=08/
  block=0000_0255.parquet
  block=0256_0511.parquet
  ...
```

manifest：

```text
factor_id → block_id
```

好处：

- 新增少量因子不必重写全部 columns；
- 训练只读需要的 blocks；
- compaction 可 block-local。

---

## R39-P0-PERF-062：matrix touched partition 不再 read-merge-rewrite 整个月宽表

使用：

```text
generation delta block
```

或者：

```text
new column block
new row delta
```

只改动的 factor/date block 写新 generation fragment。

---

## R39-P1-PERF-063：COW generation 不要每次 `rglob` 全旧 generation 再 hardlink/copy

manifest 维护完整 partition inventory：

```python
PartitionObjectRef(
    rel_path,
    content_id,
    rows,
    schema_hash,
)
```

新 generation manifest 可引用旧 object，而不是物理复制整个目录树。

如果底层必须目录化，可根据 manifest 精确 materialize links，不做 `rglob()` discovery。

---

## R39-P0-PERF-064：production staging validation 避免 Pandas 全量 read+sort+逐列循环

当前 matrix production 校验会：

```text
pd.read_parquet
sort expected
sort readback
for every factor column:
    NaN mask
    astype float64
    allclose
```

强度不能降，但执行方式必须改。

### 新 proof

writer 在写出时计算：

```text
key-order checksum
per-column finite-mask checksum
per-column numeric checksum / hash
row count
schema hash
```

read-back 使用 Arrow streaming：

- footer/schema；
- row groups；
- streaming checksum；

与 expected checksum 比。

不需要把两个巨大 matrix 同时在 Pandas 排序。

如果必须完整数值 compare：

使用 block-stream compare，不构造第二个 full DataFrame。

---

## R39-P1-PERF-065：matrix load 进入 DataAccess pushdown

`load_matrix()` 不应：

```text
读取全部月 parquet
pd.concat
全局 sort
最后再选 factor_ids
```

应支持：

```text
columns=factor_ids
time_range
instrument_filter
```

DuckDB/Polars pushdown，只扫对应 factor blocks 和 date partitions。

---

## R39-P0-PERF-066：materialize_matrix 不要先 run_many 收全量 results 再写

新增：

```python
materialize_matrix_streaming(...)
```

流程：

```text
scheduler
→ FactorBlock
→ matrix block writer
```

结果不在 Python dict 中驻留整批。

---

## R39-P1-PERF-067：matrix semantic identity 在 compile 后就缓存，不依赖完整 result 才重复构建 lineage

identity 所需的：

- factor；
- analysis；
- source contract；
- scope；

大部分编译后已知。

构造：

```python
MaterializationIdentityCertificate
```

writer 直接消费，不为每 factor 再重建大量 metadata dict。

---

# 20. materialize_sharded / incremental：批量入口必须真的走 fast sink 路径

## R39-P0-PERF-068：materialize_sharded 的 batch 模式不能 run_many 返回全部结果后再逐因子写

当前 batch shard：

```text
run_many(...)
→ results dict
→ for factor execute_materialize
```

应提供：

```text
materialize_many_fast(..., shard_scope=...)
```

计算完成即 sink。

分片的目的之一就是控制内存/TTDC，不应该分片后又在 shard 内攒全结果。

---

## R39-P0-PERF-069：materialize_incremental_many_from_config 不要逐配置串行执行

当前流程本质：

```text
for config:
    materialize_incremental(...)
```

### 新流程

先加载所有 config，按严格 execution identity 分组：

```text
source snapshot
dataset config
market
universe
frequency
calendar
PIT policy
writer target
```

同组：

```text
batch compile
shared read waves
CSE
batch incremental windows
streaming batch writer
```

不同组分开。

---

## R39-P1-PERF-070：DataEvent 受影响 factor 可按完全相同 source/execution identity 安全批处理

仍然保留“不同 source config 绝不共享 engine”的 correctness 原则。

但不是：

```text
每 factor 永远一个 engine
```

而是：

```text
identical immutable ExecutionIdentity
→ one FactorCampaignSession
```

factor-level semantic identity 仍单独校验。

---

# 21. DuckDB 执行成本：当前 per-query setup 仍可明显下降

## R39-P0-PERF-071：deadline pool connection PRAGMA 只在创建/配置变化时设置

当前 idle connection 每次 acquire 不应重复应用所有 PRAGMA。

connection 保存：

```python
applied_config_fingerprint
```

只有 lease 需要的动态项变化才 update。

---

## R39-P0-PERF-072：deadline 不要每个 query 新建一个 watchdog thread

新增进程级：

```python
DeadlineManager
```

内部：

- 一个 timer thread；
- min-heap / timer wheel；
- query register；
- cancel；
- 超时调用 `conn.interrupt()`。

指标：

```text
watchdog_thread_created_count
```

应该从 O(query_count) 变成 O(1)。

---

## R39-P1-PERF-073：scoped/isolated DuckDB connection 使用 resettable pool

当前某些 scoped SQL 每次：

```python
duckdb.connect(":memory:")
```

新增：

```python
ScopedConnectionPool
```

归还前：

- drop temp objects；
- rollback/reset；
- clear request credentials；
- health check。

禁止 credential 泄漏。

---

## R39-P1-PERF-074：DuckDB threads 使用 query-class cohort，而不是静态 `total_threads // max_concurrency`

避免不停修改全局 PRAGMA。

建立固定 profile 连接池：

```text
1-thread
2-thread
4-thread
8-thread
```

Resource lease 根据：

- current concurrency；
- query shape；
- scan bytes；

选择 profile。

重 scan 在机器空闲时可以吃更多核；并发高时自动选小 profile。

---

# 22. ScanCost / Router：从 dataset-only EMA 升级成 shape-aware TTDC predictor

## R39-P0-PERF-075：ScanCost calibration 不能只用 dataset 一个乘子

新增：

```python
ScanShapeKey(
    dataset,
    local_or_remote,
    file_count_bucket,
    selected_bytes_bucket,
    rowgroup_count_bucket,
    projected_column_ratio_bucket,
    instrument_count_bucket,
    manifest_hit,
    cache_heat,
    storage_class,
)
```

记录：

```text
open_ms
scan_ms
decode_ms
rows
bytes
throughput
```

输出 P50/P95。

---

## R39-P1-PERF-076：read strategy 不再使用固定 1M rows / 512MB 阈值

router 评估：

```text
DuckDB TTDC
Polars TTDC
PyArrow TTDC
```

包括：

```text
source scan
conversion into downstream FE backend
expected reuse
result materialization
writer representation
```

如果后续是 DuckDB fused factor，就不要为了“大数据”先选 Polars lazy。

---

## R39-P1-PERF-077：manifest 保存真实 column decoded footprint

不要：

```text
string = 16B
unknown = 8B
```

新增 per-column：

```text
compressed_bytes
uncompressed_bytes
null_count
dictionary_bytes
avg_variable_width
```

Wave memory 使用 decoded footprint。

---

# 23. 跨层极致优化：减少“边界”，不是只加快每个边界

## R39-P1-PERF-078：建立 NativePipelineRegion

planner 尽量形成：

```text
SourceScan
→ Filter
→ PIT Join
→ Minute Aggregate
→ Daily Operators
→ Cross-sectional Operators
→ Output Projection
```

一段尽可能长的 native region。

每次切 backend 都计：

```text
transition_penalty
conversion_bytes
materialization_bytes
```

optimizer 优化的是：

```text
whole-region TTDC
```

而不是单 operator speedup。

---

## R39-P1-PERF-079：重复 rolling primitive 做 Multi-Output Kernel

常见：

```text
ts_mean(x,20)
ts_std(x,20)
ts_zscore(x,20)
ts_sum(x,20)
```

或同一个 window 上多个输入。

可以提供：

```python
RollingStateBlock(window=20)
```

一次维护：

```text
sum
sumsq
count
min/max where useful
```

派生多个输出。

只在语义完全一致、证据通过的 family 使用。

---

## R39-P1-PERF-080：跨 factor source primitive 直接物化 block，不先生成多个独立 Series

例如同 source：

```text
returns
log_returns
volume_change
turnover_change
```

如果在 CSE / primitive extraction 中可识别：

```text
PrimitiveBlock
```

下游因子通过 column ref 消费。

减少：

- Python Series；
- MultiIndex；
- dict；
- allocator。

---

# 24. 不要让 telemetry 本身成为 10k factor 的瓶颈

## R39-P1-PERF-081：runtime_stats 从高频 dict mutation 改为 structured counters

热路径使用：

```python
PerfCounters
```

固定字段 / array / integer enum。

最终才渲染 dict/JSON。

详细 per-factor event：

- slow factor；
- fallback；
- error；
- sampled；

才展开完整 payload。

---

## R39-P1-PERF-082：production hard gate 使用 compile certificate，不要每 root 再 import + tree walk

compile 阶段生成：

```python
ProductionExecutionCertificate
```

runtime 校验：

```text
certificate_hash
runtime backend event
no fallback
```

常数时间。

---

# 25. Storage write ordering：为后续读取优化，但不能为排序而每次重排历史

## R39-P1-PERF-083：delta fragment 自身 sorted，base compaction 时才做全局排序

增量 delta：

```text
sort changed rows only
```

不要为了保持整个 annual file sorted，每天全历史 sort。

compaction：

```text
base + deltas
→ one sorted new base
```

---

## R39-P1-PERF-084：compaction 由 read/write amplification 联合触发

触发条件示例：

```text
delta_count > N
OR delta_bytes/base_bytes > ratio
OR read_amplification > threshold
OR query latency P95 rises
```

不要固定“每天 compaction”。

后台 compaction 必须资源受控，不能抢 production compute。

---

# 26. 推荐的新核心对象

为了避免几十个 patch 继续形成多套语义，建议新增/统一以下对象。

## 26.1 FactorBlockRef

```python
@dataclass(frozen=True)
class FactorBlockRef:
    axis: AxisBufferRef
    factor_ids: tuple[str, ...]
    values: BufferRef
    validity: BufferRef | None
    dtype: str
    rows: int
    factors: int
    representation: str
```

## 26.2 BatchWriteTransaction

```python
class BatchWriteTransaction:
    generation_id: str

    def stage_block(...)
    def stage_delta(...)
    def record_partition_stats(...)
    def stage_catalog_updates(...)
    def validate(...)
    def durable_commit(...)
    def abort(...)
```

## 26.3 DeltaManifest

```python
@dataclass
class DeltaManifest:
    base_generation: str | None
    delta_objects: list[ObjectRef]
    tombstone_objects: list[ObjectRef]
    compaction_generation: str | None
```

## 26.4 PerformanceRunSummary

见 §1。

## 26.5 MaterializationIdentityCertificate

compile 后绑定：

```text
FactorSemanticIdentity
OperatorManifest
SourceSnapshot
Calendar
Universe
Frequency
StoragePrecision
```

writer 不再重复构造。

---

# 27. 必须新增的 benchmark workload

不能只跑 toy 10×100。

## B1：Daily Small

```text
100 factors
5 years
~5000 stocks
price/volume only
```

看：

- scheduler overhead；
- context cost；
- writer latency。

## B2：Daily Large

```text
1000 factors
5 years
mixed rolling/cs
```

看：

- CSE；
- read waves；
- conversion；
- FactorBlock；
- TTDC。

## B3：Mining Scale

```text
5000–10000 cheap/medium factors
```

看：

- Python Future；
- dict copy；
- catalog；
- small files；
- writer batch。

## B4：Incremental 1 Day

```text
1000/5000 factors
only latest trading day changes
```

这是 **R39 最重要 benchmark**。

必须测：

```text
changed logical bytes
physical write bytes
historical rewrite bytes
TTDC
```

## B5：Historical Revision 20 Days

财务/行情修订造成一个 20 日影响窗。

看：

- minimal compute；
- delta write；
- compaction debt。

## B6：Minute→Daily

```text
A-share minute
100 aggregation outputs
multiple time windows
5y sample or representative large range
```

看：

- physical scan count；
- timezone expression CPU；
- SQL compile；
- Arrow roundtrip。

## B7：Mixed Source

```text
daily price
minute
fundamentals
industry
universe
```

验证真正 multi-source BatchDataRequest。

## B8：Factor Matrix

```text
500 factors
2000 factors
monthly partitions
incremental 1 day
add 50 new factors
```

看 matrix rewrite amplification。

## B9：Shard Oversize

大 panel 触发：

- time shard；
- asset shard；

测试 spool / merge / direct writer。

---

# 28. R39 Hard Performance Gates

以下不是“最好做到”，是本轮完成判定。

## Gate-01：Adaptive scheduler 无 legacy union prefetch

```text
legacy_union_prefetch_count == 0
```

## Gate-02：materialize_many_fast 无 O(N²) factor lookup

静态测试禁止：

```python
list(ids).index(...)
next(f for f in factors ...)
```

出现在 writer hot loop。

## Gate-03：Fast writer 必须真实 batch physical commit

不能 batch 入口内部仍逐 item 完成全部物化。

至少提供：

```text
batch_write_transaction_count << factor_count
```

## Gate-04：普通增量不重写历史分区

在 delta mode：

```text
historical_rewrite_bytes == 0
```

## Gate-05：正常 materialize 不全目录重扫统计

```text
full_factor_rescan_count == 0
```

## Gate-06：matrix 不用 N 次 Pandas outer merge

提供 matrix block assembly 路径。

## Gate-07：shard merge 不做每 shard concat+sort

已排序、非重叠 shard 必须一次 concat / direct write。

## Gate-08：deadline thread 数 O(1)

大量 deadline queries 不得大量创建 watchdog threads。

## Gate-09：root context shared mapping 不做全量 dict clone

5000 roots benchmark 证明。

## Gate-10：ReadWave memory 使用真实 projection footprint

不得仅依赖固定：

```text
500000 × columns × 8
```

## Gate-11：Native representation transition 可观测

输出：

```text
transition_count
transition_bytes
```

## Gate-12：Correctness parity

所有性能 benchmark 同时做：

```text
old reference vs optimized
```

比较：

- index；
- values；
- NaN；
- Inf；
- dtype policy；
- PIT；
- universe；
- lineage；
- factor version；
- durable reload。

---

# 29. 性能验收目标

硬件差异很大，因此不要只用绝对秒数。

必须以同一服务器、同一数据 snapshot、同一 commit family 做 A/B。

## 29.1 无回退目标

任何 benchmark：

```text
optimized_TTDC <= baseline_TTDC
```

除非明确证明是 correctness bug fix 导致额外必要工作。

## 29.2 推荐目标

### Daily 1000 factors

```text
TTDC speedup >= 1.5x
```

### Incremental 1-day

delta writer 完成后目标：

```text
TTDC speedup >= 2x
```

大型历史分区下应远高于 2x。

### Write amplification

普通 incremental：

```text
historical_rewrite_bytes == 0
```

### Metadata

```text
full_factor_rescan_count == 0
```

### Scheduler

cheap-factor benchmark：

```text
future_count / factor_count 显著下降
scheduler_cpu_ratio 显著下降
```

### Conversion

native batch：

```text
ConversionAmplification
```

必须相对 baseline 明显下降。

---

# 30. 分阶段实施顺序

不要一上来同时重写所有 storage。

## Phase A：低风险高收益 control-plane

先完成：

- PERF-001；
- PERF-005~017；
- PERF-018；
- PERF-021/022；
- PERF-038；
- PERF-048；
- PERF-071/072。

这批不需要改变 factor lake 对外格式，就能先拿一轮速度。

## Phase B：FactorBlock + 真 batch writer

完成：

- PERF-028~032；
- PERF-039~042；
- PERF-046/047；
- PERF-050。

建立 batch-native 内存与 writer 接口。

## Phase C：Delta storage

完成：

- PERF-043~058；
- PERF-083/084。

这是增量落值速度的关键跃迁。

## Phase D：FactorMatrix

完成：

- PERF-060~067。

## Phase E：Minute / DuckDB / router

完成：

- PERF-033~037；
- PERF-071~080。

## Phase F：Incremental & sharded production

完成：

- PERF-068~070；
- PERF-024~027。

---

# 31. 兼容性迁移要求

## 31.1 旧 factor lake 必须可读

新 reader：

```text
legacy monolithic partition
delta-generation partition
```

都支持。

## 31.2 不要在线原地破坏迁移

提供：

```python
migrate_factor_partition_to_delta(...)
```

流程：

```text
read old
→ write immutable base generation
→ validate
→ manifest atomic switch
```

## 31.3 双格式期间 writer 默认策略

research：

可 opt-in delta。

production：

先完成 parity/evidence 后再切 default。

---

# 32. 必须输出的新 evidence / report

完成整改后生成：

```text
r39_performance_baseline.json
r39_performance_after.json
r39_ttdc_breakdown.json
r39_scan_amplification.json
r39_conversion_amplification.json
r39_write_amplification.json
r39_scheduler_overhead.json
r39_materialization_layout.json
r39_delta_compaction_report.json
r39_matrix_benchmark.json
r39_minute_aggregation_benchmark.json
r39_shard_benchmark.json
r39_correctness_parity.json
```

每个必须绑定：

```text
git_sha
FactorEngine build identity
DataAccess build identity
Python
NumPy
Pandas
Polars
DuckDB
PyArrow
CPU
RAM/cgroup
storage class
source snapshot
```

---

# 33. 最终必须回答的工程问题

完成 R39 后，在总结里逐条给出证据：

1. 1000 因子 batch 最慢阶段是什么？
2. 1-day incremental 的 WriteAmplification 是多少？
3. 是否仍存在历史整分区 rewrite？
4. 是否仍存在全 factor 目录 metrics rescan？
5. materialize_many_fast 一批 1000 因子产生多少物理 writer transaction？
6. catalog transaction 数是多少？
7. scheduler 创建多少 Future？
8. read waves 实际扫描多少 bytes？节省多少重复 scan？
9. source→backend→writer 共发生多少 representation transition？
10. shard 是否存在 spool→reload→concat→sort 的重复放大？
11. minute→daily 同源 100 个聚合扫描几次分钟数据？
12. matrix 新增 50 因子是否需要重写已有 2000 列？
13. Matrix 1-day 增量是否需要重写整月？
14. DuckDB deadline 10000 queries 创建多少 watchdog thread？
15. production optimized output 与 reference 是否完全通过 parity？

---

# 34. 最终禁止用以下话术宣布完成

以下不能算完成：

```text
“新增了一个 fast path”
“增加了一个 batch API”
“新增了一个 DeltaManifest class”
“加了性能 telemetry”
“测试通过”
“理论上更快”
```

必须证明：

```text
真实 main path 命中
+
benchmark 可复现
+
TTDC 实际下降
+
扫描/转换/写放大实际下降
+
reload parity 正确
```

---

# 35. R39 最核心的架构终态

目标最终链路应接近：

```text
Factor Batch
    │
    ▼
Batch Compile Certificate
    │
    ▼
Typed Multi-Source Request
    │
    ▼
Cost-Based Read Waves
    │
    ├── DuckDB Relation
    ├── Arrow Block
    └── Polars/NumPy Native Block
    │
    ▼
Largest Native Pipeline Region
    │
    ▼
FactorBlockRef
    │
    ▼
Adaptive Byte-Batched Writer
    │
    ▼
Immutable Delta Objects
    │
    ▼
Single Generation Manifest Commit
    │
    ▼
Batch Catalog Commit
    │
    ▼
Durable Visible Generation
```

而不是：

```text
每因子
→ Series
→ reset_index
→ DataFrame
→ 读旧全年 parquet
→ concat
→ dedup
→ sort
→ 重写全年
→ 再读全目录统计
→ SQLite 更新
```

**R39 的真正目标就是把后一条链彻底退出大规模生产 fast path。**

---

# 36. 收官标准

只有同时满足以下条件，才允许宣布 R39 完成：

- [ ] R39-P0 全部实现；
- [ ] adaptive scheduler 不再提前 full-union prefetch；
- [ ] BatchDataRequest secondary source binding typed 化；
- [ ] ReadWave 使用真实 footprint；
- [ ] FactorBlockRef 已进入生产快路径；
- [ ] `materialize_many_fast` 是真实 batch physical writer；
- [ ] writer hot loop 不存在 O(N²) factor lookup；
- [ ] 正常增量不读写整个历史 partition；
- [ ] full factor metrics rescan 为 0；
- [ ] catalog 支持 generation batch transaction；
- [ ] matrix 不再 factor-by-factor Pandas outer merge；
- [ ] matrix 增量不重写全部宽列；
- [ ] shard merge 无 repeated concat+sort；
- [ ] minute bundle timezone/clock 计算降重；
- [ ] DuckDB deadline timer O(1) thread；
- [ ] ScanCost shape-aware；
- [ ] 所有 A/B benchmark correctness parity 通过；
- [ ] 当前 git SHA evidence 完整；
- [ ] TTDC、ScanAmplification、ConversionAmplification、WriteAmplification 均有 before/after。

完成后请提交一份：

```text
R39_ISSUE_CLOSURE_LEDGER.md
```

格式：

```text
Issue ID
→ 修改文件
→ 实际实现
→ 测试
→ benchmark
→ before
→ after
→ correctness parity
→ evidence
→ status
```

任何只改注释、metadata、默认值或测试 fixture 而未改变生产执行路径的 issue，一律不得标记 `CLOSED`。
