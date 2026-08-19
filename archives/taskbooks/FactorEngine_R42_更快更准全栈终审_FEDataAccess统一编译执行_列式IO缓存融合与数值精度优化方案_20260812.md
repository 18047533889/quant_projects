# FactorEngine × DataAccess R42 更快更准全栈终审
## —— 统一编译执行、列式 IO、缓存局部性、算子融合、增量计算、零拷贝表示与数值精度全量优化任务书

> **用途：可直接整份发送给 Coding AI / Codex / Cursor / Claude Code 执行。**
>
> 仓库：`https://github.com/18047533889/quant_projects`
>
> 本轮 GitHub connector 可见的 `main`：
>
> ```text
> 4b1577fce14c097fba48619885b38d6c596242eb
> ```
>
> 重要说明：
>
> 1. 用户已经把 R41 任务提交给其他 AI，但截至本轮审计时，GitHub connector 可见的 `main` 仍是 `4b1577f...`，没有看到新的 R41 落地 commit。
> 2. **执行本文件前必须重新 `git fetch` 并读取执行时真实最新 `main` HEAD**。若 R41 已经落地，R42 中与 R41 相交的项必须重新判定为 `FIXED_ALREADY_WITH_CURRENT_HEAD_PROOF`，不得重复重构。
> 3. 本文件不替代 R27–R41。它专门继续挖 **R41 之外的“更快、更准、更少重复工作”系统优化空间**。
> 4. 本轮的核心不是再堆安全开关，而是把 FactorEngine + DataAccess 收敛为一个可证明的 **Quant Query Compiler + Columnar Execution Runtime**。
>
> 本文中：
>
> ```text
> [VERIFIED] = 在审计基线 HEAD 直接看到的代码事实/缺陷
> [ARCH]     = 架构级优化方向，需要先映射最新 HEAD 再实施
> [PERF]     = 主要提升速度/资源效率
> [ACC]      = 主要提升数值/语义准确性
> [BOTH]     = 同时提升速度与准确性
> ```

---

# 0. R42 的北极星目标

最终要做到：

```text
Factor DSL / Model Feature Request
        ↓
Typed Semantic Compiler
        ↓
Dependency-normalized Global QueryGraph
        ↓
Source/Column/History/Universe/Calendar Resolution
        ↓
Cost + Accuracy Constrained Physical Optimizer
        ↓
ReadWave / RowGroup / ColumnChunk Plan
        ↓
DuckDB / Polars / Arrow / Numba Native Regions
        ↓
Shared Primitive / RollingState / CSE Buffer Graph
        ↓
FactorBlock / FeatureBlock
        ↓
Streaming DQ + Quantization/Precision Certificate
        ↓
Delta / Matrix / FactorLake Durable Commit
```

性能目标不再只看单个函数耗时，而看：

```text
TTFC = Time To First Correct Factor
TTDC = Time To Durable Commit

TTDC =
  Compile
+ Semantic Binding
+ Source Resolution
+ Snapshot
+ Planning
+ Scan
+ Decode
+ Conversion
+ Compute
+ Scheduler Wait
+ DQ
+ Serialize
+ Write
+ Catalog Commit
```

同时长期监控：

```text
Scan Amplification
Decode Amplification
Conversion Amplification
Compute Amplification
Python Object Amplification
Scheduler Amplification
Write Amplification
Rewrite Amplification
Metadata Amplification
Memory Residency Amplification
```

准确性目标：

```text
Semantic Error = 0
PIT Leakage = 0
Axis Misalignment = 0
Uncertified Approximation = 0
Uncontrolled Precision Loss = 0
Unexpected Backend Drift = 0
```

---

# 1. R42 实施总原则

## 1.1 速度优化不能破坏语义

任何：

```text
fusion
CSE
cache reuse
approximation
float32
SIMD
parallel reduction
row-group skipping
incremental recompute
```

都必须绑定：

```text
SemanticIdentity
SourceSnapshot
UniverseSnapshot
CalendarSnapshot
PriceBasis
MissingPolicy
NumericPolicy
```

否则宁可不复用。

## 1.2 优先消灭重复工作，而不是先加线程

优化顺序：

```text
少扫
→ 少解码
→ 少转换
→ 少计算
→ 少写
→ 最后才增加并发
```

## 1.3 Pandas 是 reference/debug，不应是生产公共中间表示

目标：

```text
DataAccess
→ Arrow/Relation
→ DuckDB / Polars / Numba
→ Arrow FactorBlock
→ Writer
```

Pandas 仅：

```text
reference oracle
research ergonomics
compatibility fallback
```

## 1.4 “更准”包括数值精度和经济语义精度

不仅是：

```text
float64 比 float32 更准
```

还包括：

```text
交易日 vs 自然日
动态股票池 vs 静态股票池
raw price vs adjusted price
YTD flow vs single-period flow
knowledge time vs effective time
minute session index vs clock minute
```

---

# 2. 当前 HEAD 已直接确认的新增问题总览

以下是 R42 本轮重新审计直接看到、而且不应只靠 R41 兜底的问题。

## R42-001 [VERIFIED][P0][ACC] `OutputShapeContract` 定义了，但没有真正进入 `OperatorSpec`

### 当前代码

文件：

```text
factor_engine/cleaned_operators/operator_spec.py
```

当前已经定义：

```python
@dataclass(frozen=True)
class OutputShapeContract:
    input_grain: str
    output_grain: str
    preserves_index: bool = False
    preserves_columns: bool = False
    aggregation_keys: tuple[str, ...] = ()
```

`OperatorSpec` 也有：

```python
output_shape: OutputShapeContract | None = None
```

但当前 `build_operator_spec()` 构造 `OperatorSpec(...)` 时**没有传 `output_shape`**。

### 后果

机器契约存在，但：

```text
真实 registry metadata
→ OperatorSpec
```

链断了。

生产代码仍主要靠：

```text
catalog.input_grain
catalog.output_grain
shape_preserving
```

判断。

### 必须怎么改

增加唯一：

```python
resolve_output_shape_contract(canonical, op, catalog, policy)
```

来源只能是：

```text
显式 OperatorMetadata
显式 OperatorSemanticContract
```

不允许再从名字猜。

并让：

```text
build_operator_spec
spec_to_manifest_entry
OperatorSemanticContractDigest
ProductionExecutionCertificate
IR transfer function
```

全部消费同一个 contract。

---

## R42-002 [VERIFIED][P0][ACC] `OperatorSpec.to_dict()` 和 manifest 都丢失 `output_shape`

即使未来 `build_operator_spec()` 填了 contract，当前：

```text
to_dict()
spec_to_manifest_entry()
```

也没有序列化它。

### 必须怎么改

manifest 至少加入：

```json
{
  "output_shape": {
    "input_grain": "...",
    "output_grain": "...",
    "preserves_index": false,
    "preserves_columns": false,
    "aggregation_keys": [...]
  }
}
```

并进入 semantic digest。

### Hard Gate

```text
SHAPE_CHANGING_PRODUCTION_OPERATOR_WITHOUT_SERIALIZED_OUTPUT_CONTRACT == 0
```

---

## R42-003 [VERIFIED][P0][ACC] production shape-changing admission 仍读 loose catalog 字符串，不读 typed contract

当前：

```python
if not shape_preserving:
    declared_grain = bool(
        catalog.get("input_grain")
        and catalog.get("output_grain")
        and catalog.get("input_grain") != catalog.get("output_grain")
    )
```

### 问题

仅两个字符串存在就可能通过 shape gate，但没有证明：

```text
aggregation keys
index semantics
column semantics
session mapping
grain transform
```

### 必须怎么改

production：

```text
shape_preserving=False
→ OutputShapeContract mandatory
→ validate_output_shape_contract()
```

Loose catalog 字符串只作 migration input，不是 runtime authority。

---

## R42-004 [VERIFIED][P1][ACC] `deterministic` 仍靠名字/tag 猜

当前近似：

```python
deterministic =
    "shuffle" not in canon
    and "rand_" not in canon
    and "random" not in tags
```

### 问题

以下都可能不是随机名字但仍不完全 deterministic：

```text
unordered reduction
BLAS race-sensitive reduction
hash iteration
tie-order-sensitive rank
unseeded sklearn/scipy path
parallel approximate algorithm
```

### 必须怎么改

新增：

```python
DeterminismContract(
    level = BITWISE | TOLERANCE | SEEDED | NONDETERMINISTIC,
    seed_policy,
    tie_policy,
    reduction_policy,
    backend_constraints,
)
```

任何 production operator 必须显式声明/继承一个机器 contract。

---

## R42-005 [VERIFIED][P1][ACC] `numerical_stability` 仍是 scope/name heuristic

当前：

```text
hypothesis → low
cs + name含regression → medium
其它 → high
```

这不是数值证据。

### 必须怎么改

拆成：

```python
NumericStabilityContract(
    conditioning_class,
    cancellation_risk,
    overflow_risk,
    division_risk,
    accumulation_policy,
    tolerance_class,
    preferred_dtype,
)
```

由算子 author 明确声明，或由 math certificate 生成。

---

## R42-006 [VERIFIED][P1][ACC] `_infer_panel_params()` 仍依赖签名 + 参数名黑名单

当前用：

```text
required positional parameter
+
lower/upper/threshold/alpha/... hardcoded scalar names
```

判断 panel/scalar。

### 风险

新算子如果：

```text
required scalar 参数叫别的名字
或
panel 输入带 default
```

就可能误分类。

### 必须怎么改

所有 production operator registration：

```text
panel_params
scalar_params
context_params
```

必须显式。

signature inference 仅 research/migration audit。

---

## R42-007 [VERIFIED][P1][BOTH] Analyzer 仍维护大量 canonical-name history tables

文件：

```text
factor_engine/ir/analyzer.py
```

仍有大量：

```text
_LAG_PARAM_NAMES
_FIXED_LAGS
_WINDOW_PLUS_ONE_CANONICALS
_STRUCTURE_PREFIXES
_FIN_REPORT_PERIOD_CANONICALS
...
```

### 问题

这是典型的：

```text
operator semantics 一部分在 registry
一部分在 analyzer
一部分在 runtime.execution_contract
```

### 必须怎么改

每个 operator 统一提供：

```python
OperatorTemporalTransfer(
    history_requirement,
    forward_reach,
    includes_current,
    state_requirement,
    full_history_requirement,
)
```

Analyzer 变成通用解释器，不再维护 canonical 清单。

---

## R42-008 [ARCH][P1][PERF] Analyzer 的 canonical name/set 判断应编译成整数 operator IDs

当一次批量有：

```text
1k–10k factors
几十万 IR nodes
```

大量：

```text
string canonical lookup
set membership
dict lookup
```

会形成明显 Python control-plane 开销。

### 改法

Registry freeze 时建立：

```text
CanonicalId: uint32
FieldId: uint32
ContractId: uint32
```

Typed IR 内部用整数 ID。

外部 explain/render 时再映射回名字。

---

## R42-009 [VERIFIED][P0][ACC] `SemanticLattice` 只有 5 维，但 `SemanticIdentityDigest` 已经十几维

当前 lattice：

```text
domains
frequencies
source_vintages
universe_ids
semantic_kinds
```

但 identity 已包含：

```text
market
concept_id
field_id
provider_id
dataset
unit_semantic
price_basis
flow_semantics
period_duration
timeframe
universe_id
availability
knowledge_time
revision_policy
snapshot
missing_policy
group_fallback
temporal_model
...
```

### 风险

出现：

```text
identity 最后能区分
但 compiler 中间阶段没有足够 type information 提前 reject
```

例如：

```text
raw price + adjusted price
annual flow + quarterly flow
different units
different availability
```

### 必须怎么改

建立统一 product type：

```python
SemanticTypeBundle(
    value_type,
    axis_type,
    market_type,
    frequency_type,
    unit_type,
    price_basis_type,
    flow_type,
    period_duration_type,
    availability_type,
    knowledge_time_type,
    revision_type,
    universe_type,
    missing_policy_type,
)
```

operator 定义 transfer function：

```text
inputs semantic types
→ output semantic type
```

---

## R42-010 [VERIFIED][P0][ACC] semantic canonical serializer 会把 dict key 统一 `str(k)`

当前：

```python
if isinstance(value, dict):
    return {str(k): ...}
```

### 风险

理论上：

```python
{1: "a", "1": "b"}
```

会产生 key collision。

### 必须怎么改

production semantic mappings：

```text
只允许 string key
```

或者 typed key encoding：

```text
("int",1)
("str","1")
```

推荐前者，语义 schema 本身就应使用字符串机器字段名。

---

## R42-011 [VERIFIED][P1][ACC] SemanticIdentityDigest 内部只保留 SHA256 前 16 hex

当前：

```text
hexdigest()[:16]
```

即约 64-bit。

### 建议

内部 identity：

```text
128-bit 或完整 SHA256
```

UI/display 才缩写 12–16 位。

原因不是当前马上会碰撞，而是：

```text
数万 factors
× 参数点
× source snapshots
× artifact generations
× 多年长期运行
```

没必要主动降低身份安全裕度。

---

## R42-012 [VERIFIED][P1][PERF] Optimizer 目前仍是轻量串行 rewrite，不是完整 compiler pass manager

当前：

```text
literal fold
→ param validate
→ composite lower
→ literal fold
→ fastpath rewrite
→ canonicalize params
```

### 缺失

没有统一：

```text
TypeInference
SemanticLegality
HistoryNormalization
ProjectionPushdown
PredicatePushdown
PrimitiveExtraction
CSE
WindowFusion
NativeRegionFormation
BackendPartition
CostOptimization
PhysicalLowering
Liveness
BufferPlanning
```

### 必须怎么改

新增：

```python
CompilerPassManager
```

每个 pass：

```text
input IR kind
output IR kind
invariants
cost
semantic-preservation proof
trace
```

---

## R42-013 [ARCH][P1][BOTH] optimizer 应引入“正确性约束下的 CBO”，不是纯 rule-based

目标：

```text
legal plans
→ cost ranking
```

而不是：

```text
先选 backend
→ 再硬套优化
```

Cost 至少：

```text
scan bytes
decoded bytes
conversion bytes
kernel work
memory lifetime
scheduler queue
write bytes
```

同时硬约束：

```text
PIT
parameter cert
backend cert
numeric error budget
axis contract
```

---

## R42-014 [VERIFIED][P1][PERF] `materialize_shared_nodes_parallel()` 每次创建新 `ThreadPoolExecutor`

文件：

```text
factor_engine/runtime/batch_service.py
```

### 问题

批量执行中不断：

```text
create thread pool
submit
shutdown
```

而系统已经有：

```text
HybridExecutor
ResourceBroker
AdaptiveBatchScheduler
```

### 必须怎么改

删除临时线程池。

Shared CSE task 统一进入主 scheduler DAG。

legacy entry 也借用 process-long-lived executor。

---

## R42-015 [VERIFIED][P1][PERF] CSE shared materialization 运行期还会临时重算 plan cost

当前 `_materialize_shared_subplan`：

```text
try estimate_plan_cost(sub)
except → 0
```

这是执行时控制面重复工作。

### 必须怎么改

compile/plan 阶段预生成：

```text
CSEExecutionCertificate(
    size_estimate,
    recompute_cost_ms,
    consumer_count,
    lifetime_interval,
)
```

运行时 O(1) 读取。

---

## R42-016 [ARCH][P1][BOTH] CSE eviction 不应只以 LRU 为中心，应使用 reuse distance

对于 factor DAG，compiler 知道每个 shared node 的消费者顺序。

因此可构造：

```text
first_use
last_use
next_use_distance
consumer_count
recompute_cost
size_bytes
```

实现接近 Belady 的：

```text
keep / spill / recompute
```

决策。

评分示例：

```text
value = recompute_cost × future_reuse
cost  = bytes × residency_time
```

---

## R42-017 [VERIFIED][P1][PERF] `ReadWavePlanner` 仍保留固定 500,000 rows fallback

当前：

```text
_DEFAULT_ROWS_ESTIMATE = 500_000
```

即使只是 fallback，也可能影响：

```text
packing
wave memory
coalescing
```

### 必须怎么改

fallback 顺序：

```text
exact manifest
→ partition index
→ historical ScanShape calibration
→ dataset density model
→ conservative class estimate
```

不使用一个全局 rows 常数。

---

## R42-018 [VERIFIED][P1][ACC] ReadWave 时间范围成本使用自然日长度，不是市场 session

当前 `_range_length_days()`：

```text
datetime.fromisoformat
→ (d1-d0).days
```

### 问题

A 股：

```text
周末
节假日
停市
```

都不是 scan rows。

### 必须怎么改

使用：

```text
CalendarSnapshot.session_ordinal(start/end)
```

估计：

```text
sessions × eligible universe × row density
```

---

## R42-019 [VERIFIED][P1][PERF] 不同 `universe_id` 默认直接 INCOMPATIBLE，可能错失合法 scan 复用

当前：

```text
CSI300 vs CSI500
```

只要 universe_id 不同，保守不合并。

### 更优设计

`UniverseSnapshot` 提供：

```text
membership bitmap
subset/superset relation
intersection size
union extra rows
```

Planner 可以做成本判断：

```text
union scan cheaper?
```

而不是按名字拒绝。

注意：

```text
输出 factor 计算仍使用各自 universe
```

只共享物理 scan，不共享 cross-sectional calculation universe。

---

## R42-020 [ARCH][P1][PERF] instrument filter 应使用 bitmap/roaring bitmap，而不是 tuple/string scope

大股票池时：

```text
tuple 5000 codes
排序/hash/比较
```

控制面开销高。

建议：

```text
SecurityMasterId → dense uint32
RoaringBitmap
```

用于：

```text
universe equality
subset
intersection
scan pushdown
```

---

## R42-021 [ARCH][P1][BOTH] ReadWave 应优化“decoded lifetime”，不只是 wave static bytes

现在 memory 主要按：

```text
wave estimated memory
```

更准确应建：

```text
column decoded at t1
consumer last-use at t2
```

Memory cost：

```text
bytes × lifetime
```

Planner 可把消费者相近的 roots 放一起，提高 cache locality。

---

## R42-022 [VERIFIED][P1][PERF] DataAccess `ScanCost` 在线校准只有 dataset 级单 EMA

当前：

```python
_calibration: dict[str, float]  # dataset -> factor
```

### 问题

同一 dataset：

```text
本地 vs COS
1列 vs 50列
1天 vs 5年
100股 vs 全A
hot cache vs cold cache
1并发 vs 8并发
```

不能共用一个乘子。

### 必须怎么改

key 至少：

```python
ScanShapeKey(
    dataset,
    source_class,
    storage_class,
    file_format,
    projected_column_ratio_bucket,
    selected_bytes_bucket,
    selected_file_count_bucket,
    rowgroup_count_bucket,
    selectivity_bucket,
    instrument_count_bucket,
    cache_state,
    concurrency_bucket,
)
```

---

## R42-023 [ARCH][P1][PERF] ScanCost 应保存分位数，不是单 EMA

记录：

```text
P50
P75
P90
P95
MAD
sample_count
```

路由：

```text
latency-sensitive → P90/P95
batch throughput → P50
```

---

## R42-024 [VERIFIED][P1][PERF] `suggest_read_strategy()` 仍主要是固定 rows/bytes 阈值

当前核心：

```text
prefer_polars && rows >= 1M → polars
rows >= 1M or bytes >= 512MiB → stream
否则 duckdb/arrow
```

### 必须怎么改

改为比较：

```text
PredictedTTDC(duckdb→arrow)
PredictedTTDC(duckdb→stream)
PredictedTTDC(polars lazy)
PredictedTTDC(pyarrow)
```

包括 downstream consumer backend。

例如后续是 Polars native region：

```text
Polars 扫描即便 scan 略慢
也可能因 0 conversion 总体更快
```

---

## R42-025 [VERIFIED][P1][PERF] `_avg_row_width()` 对 string/varchar 固定 16B，decoded footprint 很粗

### 必须怎么改

从 Parquet metadata / Arrow schema statistics 获取：

```text
compressed bytes
uncompressed bytes
null density
dictionary encoding
average binary length
```

建立：

```text
ColumnPhysicalFootprint
```

---

## R42-026 [VERIFIED][P1][PERF] 无 manifest 时 time selectivity 仍用固定 0.3 / 0.5

### 必须怎么改

使用：

```text
dataset coverage start/end
calendar sessions
partition min/max
instrument density
```

不能“一段闭区间=30%”。

---

## R42-027 [ARCH][P1][PERF] Scan calibration 应区分 cold-cache / warm-cache

同一 query：

```text
首次 parquet footer / object-store
第二次 object cache
```

差异大。

Benchmark/evidence 必须分：

```text
COLD
WARM
STEADY_STATE
```

---

## R42-028 [ARCH][P1][PERF] Scan calibration 应持久化，但必须绑定 host/storage class

进程重启后完全丢掉 calibration 会重复冷启动。

可保存：

```text
host_class
filesystem/storage_class
DuckDB version
DataAccess build
```

作为 calibration generation。

---

## R42-029 [VERIFIED][P0][BOTH] DataAccess deadline Arrow 与 stream 两条路径语义不完全一致

`_execute_isolated_with_deadline()` 已经：

```text
pool wait uses remaining
deadline exhausted → fail-fast
post execution re-check
```

但 `_execute_isolated_reader_with_deadline()` 当前仍：

```python
conn = self._deadline_pool.acquire(self._config)
```

使用默认 5s pool wait。

### 风险

请求：

```text
deadline = 100ms
pool 满
```

可以先等几秒，再进入 stream path。

### 必须怎么改

所有 public query 共用：

```python
AbsoluteDeadline
```

pool：

```text
wait_seconds = remaining
```

---

## R42-030 [VERIFIED][P0][ACC] stream deadline 在 acquire 后如果已经过期，watchdog 可能不立即 interrupt

当前 stream watchdog：

```python
wait = absolute_deadline - time.monotonic()
if wait > 0 and not finished.wait(wait):
    interrupt()
```

如果：

```text
wait <= 0
```

没有立即 timeout 分支。

### 必须怎么改

和 Arrow path 一致：

```python
if wait <= 0:
    timed_out.set()
    conn.interrupt()
    return
```

并在真正执行前再次 `deadline.check()`。

---

## R42-031 [VERIFIED][P0][ACC] stream deadline 缺少“查询已经超时但刚好返回”的 post-execute 丢弃检查

Arrow path 已做：

```text
timed_out or now >= deadline
→ discard result
```

stream setup 也应做。

在：

```text
reader 创建完成
第一 batch
每 batch before read
```

三个点检查。

---

## R42-032 [VERIFIED][P1][PERF] DataAccess scoped SQL 每次都新建 `duckdb.connect(":memory:")`

当前：

```text
execute_scoped_sql_arrow
execute_scoped_sql_stream
```

均有 per-call fresh connection。

### 成本

```text
connection create
PRAGMA
extension/S3 setup
catalog temp view setup
cold object/footer cache
```

### 必须怎么改

实现 R39 PERF-073 真正的：

```python
ScopedConnectionPool
```

租出干净 sandbox connection。

归还前：

```text
rollback txn
drop temp views
unregister Arrow
reset interrupt state
restore config fingerprint
```

---

## R42-033 [VERIFIED][P1][PERF] DataAccess 仍存在“每 deadline/scoped query 一个 watchdog thread”

在高并发下：

```text
100 queries
→ 100 watchdog threads
```

### 必须怎么改

全进程：

```python
DeadlineManager(
    min_heap(deadline, connection/token),
    one timer thread
)
```

注册/取消 O(logN)。

---

## R42-034 [VERIFIED][P1][PERF] deadline pool 每次 acquire 都 `apply_pragmas`

即使同一 physical connection 配置没变，也重复设置。

### 必须怎么改

连接保存：

```text
AppliedConfigFingerprint
```

只有：

```text
threads
memory
temp_directory
object_cache
S3 config version
```

变化时重放。

---

## R42-035 [VERIFIED][P1][PERF] 主 DuckDB `_conn` 与 deadline pool 并不是真正同一个 DatabaseInstance

当前：

```text
main _conn = :memory:
deadline pool = shared temp .db
```

所以注释中的：

```text
one process one DuckDB instance / shared cache
```

只在 pool 内成立，不覆盖 main path。

### 必须怎么改

做明确架构选择：

A：

```text
一个 process database instance
多 connection/cursor
```

或 B：

```text
两个 instance
但分别用于不同 workload
```

然后 benchmark 证明。

不要注释说共享、实际不共享。

---

## R42-036 [VERIFIED][P1][ACC] 远程存储需求仍靠 SQL/params sniff `"s3://"`

### 必须怎么改

PhysicalReadPlan 显式：

```python
StorageRequirement(
    local,
    remote_s3,
    credential_scope,
    endpoint,
)
```

runtime 不再解析 SQL 字符串判断是否配 S3。

---

## R42-037 [VERIFIED][P1][BOTH] `ReadPipeline.counters` 是 Store 级共享对象，不是 request-scoped

并发请求都修改：

```text
snapshot
budget
governor
verify_before
execute
...
```

### 问题

“每 public path exactly once”的 evidence 在并发下不再成立。

### 必须怎么改

```text
ReadExecutionTrace
```

绑定 request/execution ID。

全局只聚合 metrics。

---

## R42-038 [VERIFIED][P1][ACC] `PipelineCounters.assert_all_exactly_once()` 使用 Python `assert`

生产验证不要依赖：

```text
python -O
```

显式：

```python
raise PipelineInvariantError(...)
```

---

## R42-039 [VERIFIED][P1][ACC] `ReadPipeline.resolve_snapshot(strict=...)` 参数目前没有实际参与逻辑

### 必须怎么改

要么删参数，避免假 capability；

要么 strict 明确：

```text
unresolved
empty-unknown
fallback
```

的 fail-closed 规则。

---

## R42-040 [ARCH][P1][BOTH] FE ReadWave 与 DA PreparedRead/ScanCost 仍是“两层 planner”

当前：

```text
FactorEngine ReadWavePlanner
+
DataAccess prepare/read routing
```

都在分别推断：

```text
columns
time range
memory
engine
representation
```

### 长期目标

统一：

```python
UnifiedReadIntent
→ DataAccessPhysicalReadPlan
→ FE consumer graph
```

FE 不再自己估底层 storage cost；
DA 不再不知道 downstream backend。

---

## R42-041 [ARCH][P1][PERF] DataAccess 应把 relation/scan 作为一等结果，而不仅是 Table/Frame

如果 downstream 是 DuckDB native region：

```text
不要 scan → Arrow Table → register → DuckDB
```

直接：

```text
DataAccess RelationHandle
→ FE DuckDB native region
```

---

## R42-042 [ARCH][P1][PERF] Arrow stream 应成为 FE SourceWave 的默认大数据表示

目标：

```text
DA RecordBatchReader
→ FactorBlock builder / Polars stream / Numba chunk kernel
```

避免整 wave 一次物化。

---

## R42-043 [VERIFIED][P1][PERF] Polars axis verification 当前 `to_list()` 做 Python timestamp 比较

当前：

```text
result[__fe_time__].to_list()
expected = Python list
got != expected
```

大面板每个 representation boundary 都 O(N) Python boxing。

### 必须怎么改

BufferRef 携带：

```text
AxisIdentityCertificate
```

转换时：

```text
length
dtype
monotonic
digest
```

O(1)/vectorized verify。

只有 certificate miss 时做完整 compare。

---

## R42-044 [VERIFIED][P1][PERF] Polars fast path 入口仍以 Pandas panel 为源

当前：

```text
pa.Table.from_pandas(panel)
→ pl.from_arrow
```

### 真正零拷贝目标

DataAccess 直接给：

```text
Arrow Table / RecordBatch
```

Polars：

```text
pl.from_arrow
```

避免：

```text
DA → Pandas → Arrow → Polars
```

---

## R42-045 [VERIFIED][P1][ACC] `ContextVar` 的默认值是一个模块级共享 mutable `ExecutionPerfCounters()`

当前：

```python
ContextVar(..., default=ExecutionPerfCounters())
```

如果调用方没有显式 bind：

```text
多个 context 共用同一个 mutable default counter
```

### 必须怎么改

default：

```text
None
```

getter：

```text
若 None → 新建当前 context counter 并 set
```

---

## R42-046 [ARCH][P1][PERF] representation 不应以 DataFrame type 为核心，而应以 BufferRef 为核心

建议：

```python
BufferRef(
    storage=Arrow|Polars|NumPy|DuckDBRelation,
    axis_certificate,
    semantic_certificate,
    source_snapshot,
    ownership,
    bytes,
)
```

backend 之间尽量“借用”同一 buffer。

---

## R42-047 [VERIFIED][P0][ACC] 分钟聚合默认 metric 仍靠字段名后缀猜

当前：

```text
*open → first
*high → max
*low → min
*volume / *amount → sum
其它 → last
```

### 风险

自定义字段：

```text
buy_volume_x
turnover_value
mid_high_indicator
```

语义可能被名字误导。

### 必须怎么改

FieldContract 增：

```python
AggregationSemantic(
    open_like,
    high_like,
    low_like,
    additive_flow,
    stock_last,
)
```

production 不能从字段名猜。

---

## R42-048 [VERIFIED][P0][ACC] `AggregationSpec.market` 可以为空，随后使用 A 股式 fallback

无 market 时：

```text
minute elapsed 从 09:30 算
session bars 默认 240
timezone 不转换
```

这对 US/未来其它市场不安全。

### 必须怎么改

production：

```text
market + calendar/session contract mandatory
```

research legacy 才允许 fallback。

---

## R42-049 [VERIFIED][P0][ACC] `AggregationSpec(aggregation="minute_at", hhmm=None)` 可被直接构造

当前 `__post_init__` 对 minute_at 的检查只在：

```text
hhmm is None AND start/end 非空
```

时 raise。

纯：

```python
AggregationSpec(aggregation="minute_at")
```

仍可能成立。

### 必须怎么改

```text
minute_at → hhmm mandatory
minute_range → either explicit FULL_SESSION or start+end mandatory
minute_of_day → period/index mandatory
```

---

## R42-050 [VERIFIED][P1][ACC] “无过滤条件 = full session” 是隐式语义

目前 `_spec_filter_sql()` 返回 None：

```text
SQL 不加 WHERE
```

这可以是合法 full-session，但应显式：

```text
SessionWindow.FULL_REGULAR_SESSION
```

不能由“参数缺失”代表经济语义。


---

# 3. Streaming Sink、缓存与内存：当前 HEAD 可直接修的速度/正确性问题

## R42-051 [VERIFIED][P0][PERF] StreamingResultSink 初始总 queue budget 会被 writer 数放大

当前：

```python
self._worker_queues = [
    BoundedResultQueue(queue_bytes)
    for _ in range(self._writer_threads)
]
```

如果：

```text
queue_bytes = 4 GiB
writer_threads = 2
```

理论总 queue capacity 变成：

```text
8 GiB
```

而 `set_target_bytes(total)` 后才会均分。

### 必须怎么改

构造时也按总预算分：

```python
per_worker = total_queue_bytes // writer_threads
```

并保证 remainder 有确定分配。

### Hard Gate

```text
sum(worker_queue.max_bytes) <= configured_total_queue_bytes
```

---

## R42-052 [VERIFIED][P0][ACC] `submit()` 先增加 accepted，再实际 enqueue

当前：

```python
with self._lock:
    self._accepted += 1
return queue.put(item)
```

如果：

```text
queue closed
timeout
writer fatal
```

`put=False`，accepted 仍多 1。

### 后果

最终：

```text
accepted != committed + failed
```

而且 durable accounting 的含义被污染。

### 必须怎么改

```python
ok = queue.put(item)
if ok:
    accepted += 1
return ok
```

需要避免 fatal 与 enqueue 的 race，最好由 queue 返回 typed admission receipt。

---

## R42-053 [VERIFIED][P0][PERF] `BoundedResultQueue.put()` 的 timeout 每次 wait 都重新给完整 600s

当前 while：

```python
while current + item > max:
    if not cond.wait(timeout):
        return False
```

如果多次被 notify 又仍然放不下：

```text
总等待时间可远大于 timeout
```

### 必须怎么改

绝对 deadline：

```python
deadline = monotonic() + timeout
remaining = deadline - monotonic()
```

---

## R42-054 [VERIFIED][P0][PERF] 单个 ResultItem 大于 queue budget 时可能永远无法入队

条件：

```text
item.bytes > max_bytes
```

无论消费多少：

```text
current + item.bytes > max_bytes
```

一直成立。

### 必须怎么改

定义 OversizedResultPolicy：

```text
DIRECT_HANDOFF
SPILL_THEN_QUEUE_REF
TEMPORARY_SINGLE_ITEM_BORROW
SHARD_RESULT
```

production 禁止无限 wait。

---

## R42-055 [VERIFIED][P1][ACC] writer 对未知异常默认判 transient

当前：

```python
return "transient"
```

### 问题

未知：

```text
TypeError
logic bug
unexpected schema bug
programming error
```

会重试三遍。

### 必须怎么改

默认：

```text
PERMANENT_UNKNOWN
```

只对明确：

```text
timeout
connection reset
temporary lock
retryable remote status
```

重试。

---

## R42-056 [ARCH][P1][PERF] writer batch 应按 partition/locality 分组，不应只按 worker queue 到达顺序

一个 worker 可能拥有多个 partition。

当前 batch：

```text
连续拿到什么就一起写什么
```

更优：

```text
partition-key microbuffer
```

按：

```text
partition
factor block
storage target
```

聚合。

减少：

```text
lock churn
small file
catalog transaction
Parquet writer reopen
```

---

## R42-057 [ARCH][P1][PERF] backpressure 不应只看 queue ratio

应估：

```text
writer drain bytes/sec
incoming compute bytes/sec
queue bytes
ETA to saturation
```

Scheduler 控制：

```text
incoming_rate > drain_rate
→ 提前降 admission
```

而不是等 70%/90% 才减速。

---

## R42-058 [VERIFIED][P1][ACC] `finish()` 错误文字仍写 `join(timeout=10)`，但实际 timeout 已动态

不是大 correctness bug，但会误导运维/evidence。

所有错误信息输出：

```text
actual join_timeout
queue depth
batch size
writer throughput
```

---

## R42-059 [VERIFIED][P0][ACC] Cache release reconciliation 在 clear 之后再算 actual retained bytes

当前 `ExecutionCacheSession.release()`：

```text
shared_result_cache.clear()
panel_cache.clear()
然后 estimate_dict_bytes(shared_result_cache)
```

因此：

```text
actual ≈ 0
```

### 问题

所谓：

```text
accounting reconciliation
```

不能证明释放前的账实一致。

### 必须怎么改

顺序：

```text
before_actual
before_declared
reconcile
unregister
clear/release
after_actual
assert after≈0
```

并分别记录：

```text
released_bytes
orphan_bytes
governor_declared_bytes
```

---

## R42-060 [ARCH][P1][PERF] 每个 ExecutionCacheSession 自己创建 SpillStore，长期可改为 Host SpillManager

目标：

```text
HostSpillManager
├─ execution A namespace
├─ execution B namespace
└─ execution C namespace
```

共享：

```text
disk quota
throughput budget
cleanup worker
spill index
```

避免每 job 都创建独立管理对象/扫描目录。

---

## R42-061 [ARCH][P1][BOTH] cache key 与 buffer ownership 要彻底统一

当前不同层有：

```text
plan cache
CSE
panel cache
DataAccess query cache
DA resolution cache
Arrow buffers
```

长期建议一个：

```python
CacheObjectIdentity(
    semantic_identity,
    source_snapshot,
    physical_representation,
    axis_identity,
    build_generation,
)
```

各 cache 只是不同 persistence/lifetime tier。

---

## R42-062 [ARCH][P1][PERF] CSE 应编译出 exact lifetime，consumer 完成后无需 runtime 重走 root 收集 sid

compile 时生成：

```text
consumer_count
last_consumer_task
```

scheduler task complete：

```text
O(1) decrement/release
```

而不是临时遍历/collect。

---

## R42-063 [ARCH][P1][PERF] CSE 可进行“部分列共享”，不必共享整 panel

如果：

```text
A 需要 close,volume
B 只需要 close
```

可共享：

```text
axis + close buffer
```

而不是一个大 DataFrame object。

因此 CSE 粒度应向：

```text
ColumnBufferRef
PrimitiveBufferRef
```

发展。

---

## R42-064 [ARCH][P1][PERF] PanelCache 的 unstack 结果应列式保存

不要只缓存：

```text
一个 Pandas wide frame
```

可以缓存：

```text
AxisRef
ColumnGroupRef
```

不同算子按需要借列。

---

## R42-065 [ARCH][P1][PERF] 对高复用 input 做“read-once + vectorized primitive pack”

例如日线：

```text
close
open
high
low
volume
amount
```

一次构造 primitive pack：

```text
return_1d
log_return
true_range
dollar_volume
overnight_gap
intraday_return
```

下游几千因子共享。

但 primitive 必须有独立 semantic identity。

---

## R42-066 [ARCH][P1][PERF] 公共 rolling sufficient statistics 应缓存 state，不缓存最终 N 个高度相似 rolling outputs

例如同一窗口：

```text
sum
mean
variance
std
zscore
cov
beta
```

共享：

```text
count
sum
sum_sq
sum_xy
```

RollingStateBlock 一次 traversal。

---

## R42-067 [ARCH][P1][ACC] RollingStateBlock 必须定义数值算法

不要为了共享改成：

```text
sum_sq - sum^2/n
```

却在大数/低方差下失真。

允许：

```text
Welford
Chan-Golub-LeVeque
compensated rolling update
```

性能和稳定性一起 benchmark。

---

## R42-068 [ARCH][P1][PERF] cross-sectional 同日算子也可多输出融合

同一 date/universe：

```text
rank
zscore
demean
winsor
quantile bucket
```

可共享：

```text
finite mask
sort/order
median/quantile
group partition
```

尤其 rank/quantile 家族。

---

## R42-069 [ARCH][P1][PERF] group/industry operators 应一次构造 group offsets

不要每个因子：

```text
groupby industry
```

重新 build hash table。

构造：

```text
GroupPartitionRef
```

绑定 date/universe/industry snapshot。

---

## R42-070 [ARCH][P1][PERF] neutralization/regression 家族共享 design matrix decomposition

同一个 date：

```text
industry dummies
log mcap
beta
```

很多因子做 neutralize 时：

```text
X 相同，y 不同
```

可以：

```text
QR(X) / projection operator
```

一次算，多个 y 批量投影。

---

# 4. 存储和增量物化：减少写放大，同时提高精度

## R42-071 [VERIFIED][P1][PERF] Delta storage 在当前 materializer 仍是 DEFAULT OFF

当前：

```text
FACTOR_ENGINE_DELTA_STORAGE default = 0
```

也就是 R39 新的 delta 架构没有自动成为正常生产路径。

### 建议

不要立即无条件翻默认。

先跑：

```text
7天/30天 shadow
read amplification
write amplification
compaction debt
recovery
```

通过后：

```text
production default = delta
legacy rewrite = compatibility
```

---

## R42-072 [ARCH][P1][PERF] Base+Delta 需要读路径 merge index，而不仅是写快

写 delta 后如果读每次：

```text
base + 100 deltas 全扫
```

最终把成本从 write 移到了 read。

新增：

```text
DeltaIndex
key-range
min/max date
factor columns
row count
tombstone count
```

读取只选相关 fragments。

---

## R42-073 [ARCH][P1][PERF] compaction 触发应基于 read cost，而不仅是 delta count/ratio

计算：

```text
predicted read amplification
P95 read latency
fragment open cost
```

在：

```text
后台资源充足
```

时 compaction。

---

## R42-074 [ARCH][P1][PERF] compaction 可以 partial，不必每次整 partition

例如：

```text
year partition
```

内部按：

```text
month / key-range / row-group
```

局部 compact。

---

## R42-075 [ARCH][P1][PERF] 因子值和重复 metadata 应物理分离

当前 long format 每行可能重复：

```text
factor_version
snapshot
calc_time
policy
```

更优：

```text
data parquet:
  datetime, asset, value, validity

manifest/catalog:
  factor-level metadata
generation-level metadata
```

仅真正 row-level 的字段留数据表。

---

## R42-076 [VERIFIED][P0][ACC] production float32 无量化证据当前仍可能只标 `float32_legacy` 而不是 hard reject

“更准”的生产目标建议：

```text
production default float64
```

float32 只有：

```text
QuantizationCertificate
```

才能发布。

---

## R42-077 [ARCH][P1][ACC] QuantizationCertificate 不应只看 absolute numeric diff

因子最重要的是：

```text
rank
sign
top/bottom membership
IC
```

float64→float32 认证至少测：

```text
max abs error
relative error
rank correlation
top-decile overlap
sign flip rate
IC delta
```

---

## R42-078 [ARCH][P1][BOTH] dtype 可以按算子/因子动态选择

例如：

```text
rank / normalized signal → float32 往往足够
small residual / near-zero spread → float64
large notional intermediates → float64
boolean/mask → bit/uint8
group id → int32
```

不必全系统一个 dtype。

---

## R42-079 [VERIFIED][P1][PERF] FactorMatrix legacy merge path 仍使用 Pandas outer merge + sort

当前 `_merge_matrix_frames()`：

```text
existing.merge(new, how="outer")
drop_duplicates
sort_values
astype
```

对大月分区 × 数千因子仍贵。

### 必须怎么改

优先：

```text
Arrow Table keyed overlay
DuckDB MERGE-like relational overlay
column-block delta
```

避免把整月所有列装进 Pandas。

---

## R42-080 [ARCH][P1][PERF] Matrix column blocks 应支持“只新增因子列，不重写旧列”

布局：

```text
axis block
factor block 000
factor block 001
...
```

新增 50 个 factor：

```text
只写新 block + manifest
```

不触碰前 2000 列。

---

## R42-081 [ARCH][P1][PERF] FactorMatrix row update 与 column update 应分两种 delta

```text
RowDelta:
  新日期/修订日期

ColumnDelta:
  新因子
```

两者物理优化不同。

---

## R42-082 [ARCH][P1][PERF] Matrix load 应 projection pushdown 到 factor block

请求：

```text
20 factors
```

不能打开：

```text
2000-factor wide parquet
```

如果 block layout 已有，DataAccess/FactorMatrix reader 要真正消费。

---

## R42-083 [ARCH][P1][PERF] FactorBlockRef 可以直接进入 Parquet writer，避免 DataFrame reconstruction

流程：

```text
backend native output
→ Arrow RecordBatch
→ add factor columns
→ parquet
```

不再：

```text
Series
→ DataFrame
→ merge
→ Arrow
```

---

## R42-084 [ARCH][P1][BOTH] Writer 使用 sorted-axis certificate 跳过重复 sort

如果 compute output 已证明：

```text
(date, asset) sorted unique
```

writer 不应每层重复：

```text
sort
drop_duplicates
```

携带：

```text
AxisOrderCertificate
UniquenessCertificate
```

---

## R42-085 [ARCH][P1][PERF] incremental materialization 应利用 DataChangeSet 精确决定重算范围

变化：

```text
一家公司一条财报 revision
```

不能默认：

```text
所有股票 × 全窗口
```

ChangeImpact 输出：

```text
affected instruments
earliest affected date
affected dependent factors
required warmup
```

---

## R42-086 [ARCH][P1][ACC] ChangeImpact 的最小重算边界必须基于 operator history transfer

例如：

```text
revision at t
rolling 60
→ t..t+59 或相应因果影响
```

不能用固定 `lookback_extra=5` 一类经验值作为 correctness authority。

---

## R42-087 [ARCH][P1][PERF] 水位线从单日期升级为 source-aware dependency watermark

每个 factor 保存：

```text
source A watermark
source B watermark
calendar generation
universe generation
```

只有变化 source 触发相应 recompute。

---

## R42-088 [ARCH][P1][PERF] 高频/分钟聚合物化可以按“交易日 block”缓存

分钟→日的聚合：

```text
某日完成后通常不可变
```

可生成：

```text
MinuteDailyAggregateBlock(date, snapshot)
```

多个日频因子共用。

---

## R42-089 [ARCH][P1][PERF] 对稳定原始数据建立物理 column bundle

例如日频价格：

```text
OHLCVA
```

常一起读。

可以 Parquet row group/column layout 结合真实 workload 调优：

```text
hot column bundle
cold auxiliary columns
```

不要只靠逻辑 projection。

---

## R42-090 [ARCH][P2][PERF] 热/冷分层

```text
最近 60 天 → NVMe/hot compact blocks
历史多年 → compressed cold parquet
remote archive → COS
```

DataAccess planner 知道 storage tier。

---

# 5. DataAccess 读取：从“会选引擎”提升为“物理访问路径优化器”

## R42-091 [ARCH][P1][PERF] PartitionMetadataIndex 要真正支持 row-group pruning

不仅文件级：

```text
date min/max
instrument range/bloom
```

还要：

```text
row-group min/max
selected columns physical bytes
```

Planner 直接估实际读取 row groups。

---

## R42-092 [ARCH][P1][PERF] instrument filter 可使用 row-group bitmap/bloom

全 A 文件中只读：

```text
CSI300
```

如果文件布局不按 stock 分区，纯 SQL predicate 仍要读大量 pages。

可评估：

```text
Bloom filter
sorting by date+instrument
secondary row-group index
```

哪种更合适。

---

## R42-093 [ARCH][P1][PERF] manifest 要记录 compressed / uncompressed column bytes

这样内存估计不再：

```text
rows × 8
```

而能估：

```text
IO bytes
decoded bytes
```

分别进入 cost。

---

## R42-094 [ARCH][P1][PERF] remote scan 要把 HEAD/LIST/open/read 分开计成本

```text
metadata latency
request count
data bytes
```

不同优化策略。

---

## R42-095 [ARCH][P1][PERF] 小文件问题要有自动 compact / ingestion policy

DataAccess 观测：

```text
file_count
median file size
rowgroups/file
open latency share
```

若小文件过多，后台 compact。

---

## R42-096 [ARCH][P1][PERF] 对重复查询 pattern 建 workload-aware partitioning advisory

系统可以离线建议：

```text
按 year/month
按 trade_date
cluster by instrument
row group size
```

但不在 query 时自动重排数据。

---

## R42-097 [ARCH][P1][PERF] DataAccess 本地 NVMe read-through cache

COS 热数据：

```text
immutable snapshot objects
```

可按：

```text
etag/version_id/content digest
```

缓存到本地。

必须：

```text
content-addressed
quota governed
LRU/ARC
```

---

## R42-098 [ARCH][P1][PERF] remote multi-object prefetch 应受 bandwidth-aware scheduler 控制

不是固定 remote concurrency。

观测：

```text
throughput
latency
errors
headroom
```

动态调整并发。

---

## R42-099 [ARCH][P1][PERF] 数据读取可以异步预热 metadata，但计算线程不应阻塞等待无关 source

Global QueryGraph 知道 critical source。

优先：

```text
critical path sources
```

低优先 source 后读。

---

## R42-100 [ARCH][P1][PERF] source resolution cache 应缓存 typed resolution，不只是 path list

缓存：

```text
ResolvedPhysicalScope
Snapshot
PartitionStats
ColumnFootprint
SecurityDecision
```

key 必须绑定：

```text
source generation
principal scope
request predicates
```


---

# 6. ReadWave / BufferRef / Scheduler：把“计划上的 native”变成“真实 native”

## R42-101 [VERIFIED][P0][ACC] SourceWave scan 异常被吞掉并返回 `None`

文件：

```text
factor_engine/runtime/buffer_ref.py
```

当前：

```python
try:
    prefetch(...)
    ...
except Exception as exc:
    event["reason"] = ...
return ref  # 失败时 None
```

### 问题

这不是 typed recovery，而是：

```text
scan failed
→ no exception
→ caller 得到 None
```

### 必须怎么改

返回 typed：

```python
WaveExecutionResult =
    Success(BufferRef)
  | Failed(WaveFailure)
```

production：

```text
Permanent / PIT / Snapshot / Schema / Permission
→ hard fail
```

仅：

```text
explicit recoverable category
```

进入 `WaveRecoveryPlan`。

---

## R42-102 [VERIFIED][P0][ACC] scheduler 会把失败 wave 覆盖的 SOURCE_SCAN task 标 committed

当前 `_execute_read_waves()`：

```text
ref = executor.execute_wave(...)
refs[wid] = ref
...
buffer_results[t] = ref
committed.add(t)
```

没有先要求：

```text
ref is valid success
```

### 风险

控制面认为：

```text
source dependency complete
```

真实：

```text
scan failed
buffer = None
```

然后 root 可能再走隐式按需读取/fallback。

### 必须怎么改

```text
Success → committed
RecoverableFailure → typed replan
FatalFailure → raise
```

绝不能 `None` 同时标 committed。

### Hard Gate

```text
FAILED_SOURCE_WAVE_COMMITTED_TASK_COUNT == 0
```

---

## R42-103 [VERIFIED][P0][BOTH] BufferRef 的 `preferred_representation` 当前可能只是“标签”，不是实际物理产物证明

当前 wave 执行：

```text
source.prefetch_columns(...)
或 source.load_columns(...)
```

之后直接：

```text
representation = preferred_representation
location = "source.native"
```

但没有看到：

```text
prefetch 返回 native Arrow/Polars buffer
```

的验证。

### 必须怎么改

Source API 改为：

```python
source.execute_read_wave(
    ReadWavePhysicalPlan
) -> SourceBufferRef
```

由 source 自己返回：

```text
actual representation
actual buffer handle
actual bytes
actual snapshot
axis certificate
```

FE 不自己猜。

---

## R42-104 [P0][ACC] BufferRef 的 representation 必须是枚举 + capability，而不是自由字符串

定义：

```python
PhysicalRepresentation:
    ARROW_TABLE
    ARROW_STREAM
    POLARS_LAZY
    POLARS_FRAME
    DUCKDB_RELATION
    NUMPY_BLOCK
    PANDAS_PANEL
    SPILL_REF
```

并有：

```text
materialization state
zero-copy compatibility
streamability
mutability
```

---

## R42-105 [P0][ACC] BufferRef `meta: dict` 在 frozen dataclass 内仍可变

类似 artifact deep-freeze 问题，但这是运行时 buffer identity。

### 必须怎么改

semantic metadata 与 telemetry 分离：

```text
BufferSemanticCertificate = immutable
BufferRuntimeStats = mutable telemetry
```

不要把可变 dict 放 identity carrier。

---

## R42-106 [P0][ACC] BufferRef 的 `location: Any` 缺 typed ownership contract

当前可能是：

```text
cache key
object ref
relation
字符串 "source.native"
```

统一：

```python
BufferLocation:
    InMemoryObjectRef
    CacheObjectRef
    ArrowStreamRef
    DuckDBRelationRef
    SpillFileRef
```

每类有显式 close/release 生命周期。

---

## R42-107 [P1][PERF] SourceWaveExecutor 的 event list 可以改为固定统计 + bounded trace

1万因子/很多 waves 时无需永久保存所有 runtime dict。

保留：

```text
recent N detailed events
+
aggregate counters/histograms
```

---

## R42-108 [VERIFIED][P1][PERF] scheduler 每个 control loop 都遍历全部 waves 找 ready wave

`_execute_read_waves()`：

```text
for wave in waves:
    检查 inputs committed
```

随着：

```text
wave_count × scheduler ticks
```

增长。

### 必须怎么改

建立：

```text
dependency_count_by_wave
task→waiting_waves adjacency
ready_wave_queue
```

当 dependency commit 时 O(out-degree) 更新。

---

## R42-109 [VERIFIED][P1][PERF] scheduler 仍实际每 50ms timeout 一次

当前：

```python
_EVENT_WAIT_TIMEOUT_S = 0.05
wait(..., timeout=0.05, FIRST_COMPLETED)
```

注释说“不是轮询”，但没有 future 在 50ms 内完成时仍会：

```text
wake
resource decision
scan ready sets
再 wait
```

这就是低频 polling。

### 必须怎么改

事件源统一：

```text
Future completion
ResourceDecision change
Writer backpressure threshold crossing
Cancellation
Deadline
```

用 condition/event 唤醒。

另有较长 watchdog heartbeat，不要 20Hz 控制轮询。

---

## R42-110 [VERIFIED][P0][ACC] cancellation 检查的 broad `except` 当前会进入 `break`

当前代码形态：

```python
try:
    get_active_cancellation_token(...)
    ...
except Exception:
    pass
    self._explain("CANCELLED...")
    break
```

### 后果

不是“取消异常忽略”，而是：

```text
读取 cancellation infrastructure 出任意异常
→ scheduler 停止
```

### 必须怎么改

绝对禁止这类 `pass + break` 模糊语义。

```text
CancellationInfrastructureError
```

production：

```text
fail run
```

research：

```text
warning + continue only if policy permits
```

---

## R42-111 [VERIFIED][P0][PERF] `_plan_cost_bytes()` 失败时返回 `peak_memory=0`

当前：

```python
except Exception:
    return {"total_work": 0.0, "peak_live_memory_bytes": 0}
```

### 风险

cost estimation failure：

```text
最危险的大任务
→ 被当成零成本
→ 最容易被 admission
```

### 必须怎么改

production：

```text
CostUnknown
→ conservative contract
```

不能零。

---

## R42-112 [VERIFIED][P1][ACC] scheduler ERROR_UNKNOWN 会自动 retry 一次

未知错误很可能是：

```text
deterministic bug
programming bug
schema bug
```

### 建议

默认：

```text
UNKNOWN → no retry
```

只允许 typed transient retry。

research 可以 opt-in one retry 做诊断。

---

## R42-113 [VERIFIED][P0][ACC] Scheduler 把所有 `OSError` 归 transient

`OSError` 包含：

```text
ENOSPC
EROFS
permission
invalid path
```

这些不是 transient。

### 必须怎么改

按 errno：

```text
EAGAIN / ETIMEDOUT / ECONNRESET → transient
ENOSPC / EACCES / EROFS / EINVAL → permanent
```

---

## R42-114 [VERIFIED][P1][PERF] MicroBatch 当前强制 `prefer="thread"`

即使某 backend/task：

```text
Python/GIL-bound
```

也可能被 microbatch 强制线程执行。

### 必须怎么改

MicroBatch contract 聚合：

```text
ExecutionTraits
```

只有全组：

```text
releases_gil / native
```

才 thread。

否则：

```text
process lane
或不 microbatch
```

---

## R42-115 [ARCH][P1][PERF] MicroBatch 大小应由 launch overhead / work ratio 动态决定

固定：

```text
16–128
```

改为：

```text
estimated kernel work
future overhead
Python dispatch cost
buffer locality
```

目标：

```text
dispatch overhead < 3–5% batch time
```

---

## R42-116 [ARCH][P1][PERF] fusion 和 microbatch 不应是两套互不知情机制

统一：

```text
ExecutionGroup
```

候选：

```text
NativeFusionGroup
MicroBatchGroup
RollingStateBlock
PrimitiveBlock
```

Optimizer 选收益最高、不冲突的 group。

---

## R42-117 [ARCH][P1][PERF] Scheduler priority 要考虑“释放多少内存”

在 critical path/reuse priority 之外加入：

```text
completion_frees_bytes
```

高内存压力下优先运行：

```text
完成后能释放大 CSE buffer
```

的 consumer。

---

## R42-118 [ARCH][P1][PERF] Scheduler priority 应考虑 writer output size

当 writer 压力高：

```text
优先小输出任务
暂停大 output task
```

防止 queue saturation。

---

## R42-119 [ARCH][P1][PERF] Scheduler 应区分 scan-bound / compute-bound / write-bound 阶段

实时判断 bottleneck：

```text
IO bound
CPU bound
memory bound
writer bound
```

不同调节：

```text
scan concurrency
compute concurrency
writer threads
wave size
```

不是一个 target_concurrency 控全部。

---

## R42-120 [ARCH][P1][PERF] Scheduler 的 resource decision 可以由 throughput feedback 驱动

例如：

```text
increase concurrency
→ throughput 无提升 + memory ↑
→ 回退
```

在线 hill-climbing/AIMD。

---

# 7. 编译器：从轻量 optimizer 升级为 Quant Query Compiler

## R42-121 [ARCH][P1][BOTH] 建立明确编译层级

推荐：

```text
ParsedExpr
↓
BoundExpr
↓
TypedSemanticIR
↓
NormalizedIR
↓
LogicalQueryGraph
↓
OptimizedLogicalGraph
↓
PhysicalQueryGraph
↓
ExecutableRegions
```

不要一个 `PlanNode` 同时承担所有阶段。

---

## R42-122 [ARCH][P0][ACC] 每次 rewrite 必须声明 semantic equivalence class

```python
RewriteRule(
    pattern,
    replacement,
    preconditions,
    semantic_equivalence,
    numeric_equivalence,
)
```

例如：

```text
(a+b)+c ↔ a+(b+c)
```

对 floating point：

```text
不是 bitwise equivalent
```

不能默认生产改写。

---

## R42-123 [ARCH][P1][ACC] 区分数学等价与浮点等价

等级：

```text
EXACT_VALUE
IEEE_EQUIVALENT
TOLERANCE_EQUIVALENT
RANK_EQUIVALENT
RESEARCH_APPROXIMATE
```

Optimizer 根据 `NumericPolicy` 决定可用 rewrite。

---

## R42-124 [ARCH][P1][PERF] Algebraic canonicalization 可以大幅提升跨因子 CSE

例如统一：

```text
mul(x, 2)
2*x
x+x
```

是否可视为同 primitive，要由数值策略决定。

安全情况下 canonicalize 后：

```text
CSE 命中率显著提高
```

---

## R42-125 [ARCH][P1][PERF] commutative operators 的输入排序应使用 semantic hash

适用：

```text
add
mul
min/max where semantics permit
```

但 NaN/tie/overflow 可能影响，需要 contract。

---

## R42-126 [ARCH][P1][PERF] 常量传播不只 literal folding

例如：

```text
where(mask, x, x) → x
x * 1 → x
x + 0 → x
delay(x,0) → x（仅参数 contract 允许时）
```

每条都需要 edge semantics proof。

---

## R42-127 [ARCH][P1][PERF] Predicate Pushdown

如果 factor 外层有：

```text
instrument subset
time range
validity mask
```

尽可能下推 DataAccess scan。

但 cross-sectional operator：

```text
不能错误把最终输出筛选提前成计算 universe 筛选
```

必须区分：

```text
computation universe
output projection universe
```

---

## R42-128 [ARCH][P1][PERF] Projection Pushdown

最终只依赖：

```text
close, volume
```

不能读取完整 OHLCV+fundamental wide source。

投影从：

```text
Typed IR field dependency
```

一直下推到 Parquet column chunk。

---

## R42-129 [ARCH][P1][BOTH] Temporal Pushdown 要区分 warmup range 与 output range

```text
requested: 2026-01-01..2026-06-30
lookback: 60
```

scan：

```text
warmup_start..end
```

输出：

```text
requested_start..end
```

不要后端生成更长最终 frame 再多次 trim。

---

## R42-130 [ARCH][P1][PERF] History requirement compositional transfer

每个 operator：

```text
required_history(output_interval)
→ input_interval
```

把整 DAG 的 input ranges 精确反推。

比：

```text
global max lookback
```

更省 scan。

---

## R42-131 [ARCH][P1][PERF] 多源 DAG 可以分别求 history range

例如：

```text
price rolling 60
×
latest fundamental
```

价格需要 60 天；
财务 source 未必需要同样的 daily rows。

避免所有 source 用同一个 expanded range。

---

## R42-132 [ARCH][P1][PERF] NativeRegionFormation

找：

```text
最大连续 backend-compatible subgraph
```

例如 DuckDB：

```text
scan
filter
simple arithmetic
window
group
join
```

一次 SQL 执行。

---

## R42-133 [ARCH][P1][PERF] PolarsRegionFormation

保留：

```text
LazyFrame
```

跨多个 operator，不要每个 operator collect。

---

## R42-134 [ARCH][P1][PERF] NumbaRegionFormation

多个逐元素/rolling native kernels：

```text
尽量共享 ndarray block
```

避免每 kernel Pandas Series wrapper。

---

## R42-135 [ARCH][P1][PERF] backend partition 要最小化 transition cut

代价：

```text
cut_cost = conversion_bytes + materialization + axis verify
```

选：

```text
收益 - cut_cost
```

而非逐 node 独立选 backend。

---

## R42-136 [ARCH][P1][PERF] DuckDB SQL emitter 应多输出 roots

同一 scan：

```sql
SELECT
  expr_a AS factor_a,
  expr_b AS factor_b,
  expr_c AS factor_c
FROM ...
```

比 3 个 query 更快。

---

## R42-137 [ARCH][P1][PERF] 多 window function 可共享 partition/order

DuckDB：

```text
WINDOW w AS (PARTITION BY asset ORDER BY date)
```

多 rolling outputs 共享。

---

## R42-138 [ARCH][P1][PERF] Polars 也应通过一次 `.with_columns()` 生成多个 roots

避免：

```text
N 个 LazyFrame collect
```

---

## R42-139 [ARCH][P1][PERF] expression DAG intern

编译时结构相同 node：

```text
只存在一个 interned node
```

降低：

```text
Python object count
hash computation
memory
```

---

## R42-140 [ARCH][P1][PERF] compile_many 应缓存“模板 IR + 参数绑定”

很多因子：

```text
同 operator family
不同字段/窗口
```

可共享 parser/type-check skeleton。

但参数改变经济语义时 identity 仍独立。

---

## R42-141 [ARCH][P1][PERF] 参数组合可做 vectorized kernel family

例如：

```text
ts_mean(x, 5)
ts_mean(x, 10)
ts_mean(x, 20)
ts_mean(x, 60)
```

不一定四次 rolling。

可一次 prefix sum：

```text
多个窗口 O(N + K*N small constant)
```

---

## R42-142 [ARCH][P1][PERF] 多 lag delay/delta/return 用 shared shift buffer

```text
lag 1,5,20
```

一次构建 shifted views。

---

## R42-143 [ARCH][P1][PERF] 多 EWMA span 可 vectorize states

对同 input：

```text
span 5/10/20/60
```

一个 loop 更新 K 个 state。

---

## R42-144 [ARCH][P1][PERF] 多 quantile/rank requests 共享一次 sort

同横截面：

```text
p10
p20
median
p80
p90
rank
```

一次 order statistic/sort。

---

## R42-145 [ARCH][P1][PERF] 多 regression targets 共享 X decomposition

见前述 neutralization，同样适用于：

```text
rolling beta family
cross-sectional residualization
```

---

## R42-146 [ARCH][P1][PERF] PCA family 共享 fit block

同 input/window：

```text
pc1
pc2
explained variance
residual
```

一个 PCA fit，多输出。

---

## R42-147 [ARCH][P1][PERF] GARCH/Kalman/state models 共享 state trajectory

一个 stateful fit：

```text
level
trend
variance
innovation
```

多输出。

---

## R42-148 [ARCH][P1][PERF] Primitive extraction 应通过 graph pattern，不靠 factor names

例如：

```text
return = close / delay(close,1)-1
```

结构识别为：

```text
RETURN_1D primitive
```

但只在 exact semantic equivalence 证明后。

---

## R42-149 [ARCH][P1][PERF] restricted e-graph 可用于研究候选，但 production 只选择 certified rewrites

可以离线枚举：

```text
等价表达式
```

找最便宜物理实现。

最终 production：

```text
只用 proof/certificate 白名单
```

---

## R42-150 [ARCH][P1][BOTH] `engine.explain()` 要升级成真正编译解释器

输出：

```text
logical IR
semantic types
field/source dependencies
history intervals
CSE
primitive blocks
read waves
backend regions
conversion cuts
memory lifetimes
estimated TTDC
numeric precision
production certificates
```

这样性能问题可以定位，不靠猜。


---

# 8. 数值准确性：把“算得快”升级为“有误差预算地算得快”

## R42-151 [ARCH][P0][ACC] 建立统一 `NumericPolicy`

建议：

```python
NumericPolicy(
    compute_dtype,
    accumulation_dtype,
    output_dtype,
    determinism_level,
    reduction_algorithm,
    division_policy,
    overflow_policy,
    underflow_policy,
    degeneracy_policy,
    tolerance_profile,
)
```

进入：

```text
FactorSemanticIdentity
BackendEligibility
ProductionCertificate
Materialization metadata
```

---

## R42-152 [ARCH][P1][ACC] compute dtype 与 storage dtype 分离

例如：

```text
compute float64
storage float32 certified
```

不能因为最终写 float32，就中间也都 float32。

---

## R42-153 [ARCH][P1][ACC] accumulation dtype 单独声明

即使 input/output float32：

```text
sum/mean/covariance
```

可用：

```text
float64 accumulation
```

代价通常可接受，精度显著改善。

---

## R42-154 [ARCH][P1][ACC] 大窗口 sum/mean 使用 compensated summation

候选：

```text
Kahan
Neumaier
pairwise reduction
```

benchmark：

```text
speed
ULP error
rank impact
```

---

## R42-155 [ARCH][P1][ACC] rolling variance/covariance 不使用不稳定的差平方形式作为唯一 fast path

对：

```text
大 level + 小波动
```

容易 catastrophic cancellation。

优先：

```text
Welford/Youngs-Cramer
```

或双通道：

```text
fast stable kernel
reference high precision oracle
```

---

## R42-156 [ARCH][P1][ACC] regression solve 不要默认 normal equations

```text
(X'X)^-1 X'y
```

会放大 condition number。

production 优先：

```text
QR
SVD
regularized solve
```

根据 condition policy 路由。

---

## R42-157 [ARCH][P1][BOTH] condition number 可以决定算法路径

例如：

```text
condition < threshold
→ fast QR

condition high
→ SVD / ridge / NaN
```

但 branch 结果要进入 diagnostics/certificate。

---

## R42-158 [ARCH][P1][ACC] division policy 要统一覆盖 NumPy / Pandas / Polars / SQL

真值表至少：

```text
finite / 0
0 / 0
Inf / finite
finite / Inf
NaN
signed zero
very small denominator
```

所有 backend 一张 oracle 表。

---

## R42-159 [ARCH][P1][ACC] signed zero 语义要明确

```text
+0.0
-0.0
```

在：

```text
division
sign
log
copysign
ranking
```

可能不同。

如果经济上不区分：

```text
normalize signed zero
```

也应显式。

---

## R42-160 [ARCH][P1][ACC] subnormal/underflow policy

极小残差/概率：

```text
flush-to-zero
保留 subnormal
```

会受硬件/BLAS影响。

Determinism evidence 记录。

---

## R42-161 [ARCH][P1][ACC] finite mask 应是公共 primitive

不要每个算子：

```text
np.isfinite(...)
```

重复扫描大 panel。

对一个 input block：

```text
FiniteMaskRef
```

多个统计算子共享。

---

## R42-162 [ARCH][P1][PERF] null bitmap 直接复用 Arrow validity bitmap

如果源是 Arrow：

```text
不要先转 NumPy 再重新 isna
```

结合：

```text
Arrow validity
+
numeric Inf mask
```

生成 finite policy。

---

## R42-163 [ARCH][P1][ACC] NaN 与“业务无效/tombstone/out-of-coverage”不要物理混成一个语义

内部 validity 至少区分：

```text
MISSING_SOURCE
OUT_OF_UNIVERSE
INSUFFICIENT_HISTORY
NUMERIC_INVALID
DELETED
NOT_APPLICABLE
```

最终 factor value 可都是 NaN，但 DQ/lineage 不一样。

---

## R42-164 [ARCH][P1][BOTH] validity reason 可使用 compact uint8 code

不要每行 string `invalid_reason`。

Parquet：

```text
value
validity_code uint8
```

字典在 manifest。

更省空间、更快。

---

## R42-165 [ARCH][P1][ACC] cross-sectional rank tie policy 进入 identity

选：

```text
average
min
max
dense
deterministic security-id tie break
```

不能 backend 各自默认。

---

## R42-166 [ARCH][P1][ACC] quantile interpolation policy 进入 identity

NumPy/Pandas/DuckDB/Polars 对 quantile interpolation 可能不同。

显式：

```text
linear
nearest
lower
higher
midpoint
```

---

## R42-167 [ARCH][P1][ACC] ddof/bias/fisher 等 moment convention 必须从 operator contract 直达 backend emitter

不要每 backend 自己默认。

---

## R42-168 [ARCH][P1][ACC] EWM 的 adjust/ignore_nulls/seed semantics 跨 backend 对齐

尤其：

```text
warmup
NaN gap
first valid
```

要有 golden oracle。

---

## R42-169 [ARCH][P1][ACC] float equality gate 不统一用 `allclose` 一个阈值

按 operator class：

```text
exact discrete
rank/order
stable arithmetic
iterative solver
stochastic seeded
```

不同 tolerance。

---

## R42-170 [ARCH][P1][ACC] tolerance 应随 scale 变化

使用：

```text
absolute + relative + ULP
```

不能只 `atol=1e-8`。

---

## R42-171 [ARCH][P1][ACC] backend parity 加 rank-level metrics

对于 alpha：

```text
Spearman
top-k overlap
sign agreement
```

有时比绝对误差更重要。

但绝对 numerical gate 仍保留。

---

## R42-172 [ARCH][P1][ACC] 近似算法必须有 ApproximationCertificate

未来如果用：

```text
approx quantile
randomized SVD
sketch
sampling
```

必须记录：

```text
error bound
confidence
seed
workload domain
production eligibility
```

---

## R42-173 [ARCH][P1][BOTH] 大横截面 rank 可研究 radix/argsort 优化，但不能默认 approximate

A 股约 5000 股票：

```text
exact sort
```

本身不算很大。

先共享 sort、多因子融合，通常比 approximate rank 更划算。

---

## R42-174 [ARCH][P1][PERF] 多列 rank 可以用 block sort / batched argsort

如果 100 factors 同 date：

```text
NumPy/Numba column block
```

减少 Python loops。

---

## R42-175 [ARCH][P1][ACC] 全局 Numeric Reference Suite

构造极端 corpus：

```text
NaN
±Inf
±0
tiny
huge
constant
near-constant
ties
monotonic
single valid
gaps
alternating signs
```

每个 production operator/backend 自动跑。

---

# 9. Data semantic accuracy：减少“数据本身读错”的误差

## R42-176 [ARCH][P0][ACC] 交易日 ordinal 应成为 FE/DA 公共底层类型

不要：

```text
business day
calendar day
observed row index
```

三套概念混用。

定义：

```python
SessionOrdinal(calendar_snapshot_id, ordinal)
```

---

## R42-177 [ARCH][P0][ACC] timestamp 统一存 UTC instant + market-local session mapping

不要只携带 naive datetime。

内部：

```text
UTC timestamp
timezone
session_date
session_ordinal
bar_index_in_session
```

---

## R42-178 [ARCH][P0][ACC] minute bar index 从 SessionCalendar 生成

不能默认：

```text
240 bars
午休固定
```

未来：

```text
半日市
特殊交易日
US early close
```

需要 snapshot-aware session schedule。

---

## R42-179 [ARCH][P0][ACC] financial period 使用 typed FiscalPeriodId

不要仅：

```text
2025Q1 string
```

至少：

```text
issuer
fiscal_year
period_type
period_end
duration
```

---

## R42-180 [ARCH][P0][ACC] YTD→single-quarter 转换必须基于 fiscal calendar/period continuity

缺上一期、会计年度变更时：

```text
不能直接 diff
```

---

## R42-181 [ARCH][P0][ACC] corporate action/复权 policy 要 source-snapshot 化

同一历史日：

```text
今天后复权
当时可知复权
```

不同。

production PIT 因子需要明确：

```text
adjustment knowledge policy
```

---

## R42-182 [ARCH][P0][ACC] 股票代码不是长期 instrument identity

内部统一：

```text
SecurityMasterId
```

代码/交易所/上市段是属性。

防：

```text
退市重上
代码复用
市场迁移
```

---

## R42-183 [ARCH][P0][ACC] UniverseSnapshot 应是 membership interval，不是日期列表快照拼接

```text
valid_from
valid_to
knowledge_from
knowledge_to
```

支持双时点。

---

## R42-184 [ARCH][P0][ACC] group/industry membership 同样双时点

行业调整不能 retroactively 作用过去，除非 research retrospective。

---

## R42-185 [ARCH][P0][ACC] DataAccess source contract 应表达 update/revision semantics

字段：

```text
append-only
mutable-history
restated
late-arriving
correction-window
```

帮助 ChangeImpact 和 cache invalidation。

---

## R42-186 [ARCH][P1][PERF] append-only source 可以更激进缓存

如果 source contract：

```text
historical objects immutable
```

旧 snapshot block 永久 content-addressed cache。

---

## R42-187 [ARCH][P1][BOTH] late-arriving source 用 revision frontier

记录：

```text
latest stable date
mutable tail length
```

增量重算只覆盖 tail。

---

## R42-188 [ARCH][P0][ACC] knowledge time 缺失不能用 effective time 替代

必须：

```text
UNKNOWN
```

production fail。

---

## R42-189 [ARCH][P0][ACC] provider/source change 可能改变经济口径，不能只当存储位置变化

`ProviderIdentity` 进入 field/source semantic contract。

---

## R42-190 [ARCH][P1][BOTH] 对 provider 等价性做 explicit mapping

只有经过验证：

```text
provider A close
provider B close
```

定义相同，才允许 failover/cache sharing。

---

# 10. 进程、线程、共享内存与硬件：减少 Python 与复制开销

## R42-191 [ARCH][P1][PERF] GIL-bound process lane 用 shared Arrow/mmap，不传大 DataFrame

任务 payload：

```text
PlanId
BufferRef IDs
small params
```

worker 从共享内存/mmap 取数据。

---

## R42-192 [ARCH][P1][PERF] Process pool worker 预加载 immutable RegistrySnapshot

避免每 task：

```text
import/load registry
```

worker generation 与 main build 一致。

---

## R42-193 [ARCH][P0][ACC] worker 接收的 registry/evidence generation 必须校验

main 更新/热重载后：

```text
旧 worker
```

不能继续跑新任务。

使用：

```text
WorkerGenerationToken
```

不一致则 recycle worker。

---

## R42-194 [ARCH][P1][PERF] Arrow Plasma 已不推荐，可用 mmap/Arrow IPC/共享内存块管理器

选择最小依赖的：

```text
memory mapped Arrow IPC
shared_memory + Arrow buffer
```

重点是 ownership/fencing。

---

## R42-195 [ARCH][P1][PERF] 内存池统一

可评估：

```text
PyArrow memory pool
NumPy arena
TemporaryArrayArena
```

给 HostResourceCoordinator 统一观测。

---

## R42-196 [ARCH][P1][PERF] 大 ndarray scratch 复用要按 shape bucket + dtype

避免频繁：

```text
malloc/free
```

但：

```text
poison on release
generation token
```

防 use-after-release。

---

## R42-197 [ARCH][P1][PERF] NUMA 机器上大 panel 可按 socket affinity

仅在：

```text
双路/多路服务器
数据量大
```

值得。

调度：

```text
buffer first-touch
worker affinity
```

需要 benchmark 后启用。

---

## R42-198 [ARCH][P1][PERF] CPU feature dispatch

可利用：

```text
AVX2
AVX-512
```

的 Numba/C++ kernel，但 build/runtime evidence 记录 CPU feature。

---

## R42-199 [ARCH][P1][ACC] 不同 CPU/BLAS 下 determinism level 可能不同

Evidence matrix：

```text
x86_64 AVX2
x86_64 AVX512
```

至少验证 tolerance。

---

## R42-200 [ARCH][P1][PERF] threadpoolctl 不只在 root 执行时设置，应覆盖所有 native nested thread regions

包括：

```text
NumPy linalg
SciPy
model fit
writer compression if threaded
```

统一 `NativeThreadBudgetScope`。


---

# 11. 自动化因子挖掘吞吐：为 AlphaProbe / CogAlpha / LLM Mining 专门优化

## R42-201 [ARCH][P1][PERF] Candidate 进入执行前先做 semantic canonicalization

对：

```text
语法不同、经济定义相同
```

的候选先归一。

例如安全等价时：

```text
add(x,y)
add(y,x)
```

得到同 canonical semantic hash。

避免浪费：

```text
compile
scan
compute
evaluate
```

---

## R42-202 [ARCH][P1][BOTH] 去重必须分层

定义：

```text
TextDuplicate
ASTDuplicate
CanonicalIRDuplicate
ParameterNormalizedDuplicate
SemanticEquivalentDuplicate
HighCorrelationButNotEquivalent
```

只有前几类可直接跳执行。

高相关不能当语义重复删除。

---

## R42-203 [ARCH][P1][PERF] Negative Compile Cache

如果候选因为：

```text
unknown field
invalid param
type mismatch
PIT violation
unsupported operator combination
```

确定性失败，缓存：

```text
canonical formula hash
compiler generation
error class
```

LLM 重复产同坏结构时 O(1) 拒绝。

---

## R42-204 [ARCH][P1][PERF] Parameter-domain invalid cache

大量模型会不断尝试：

```text
window=0
lag<0
不合法组合
```

在 parser/binder 早期拒绝，不进入 planner。

---

## R42-205 [ARCH][P1][PERF] compile_many 应批量 bind field/source contracts

例如 1000 candidates 都用：

```text
close, volume, amount
```

FieldResolver/SourceResolver 一次批量解析。

---

## R42-206 [ARCH][P1][PERF] Candidate dependency signature

每个 candidate 编译得到：

```python
DependencySignature(
    sources,
    fields,
    history,
    universe_semantics,
    backend_capability,
    primitive_families,
)
```

scheduler 按 signature 聚类。

---

## R42-207 [ARCH][P1][PERF] source-first clustering

比按 candidate 顺序跑更快：

```text
同 source/columns/history
→ 一组 ReadWave
```

让随机生成的 LLM candidate 顺序不影响 IO locality。

---

## R42-208 [ARCH][P1][PERF] window-family clustering

同一个 input：

```text
rolling 5/10/20/60
```

优先放到一个 primitive/multi-window group。

---

## R42-209 [ARCH][P1][PERF] backend-region clustering

将大量候选分：

```text
DuckDB-native
Polars-native
Numba-native
Pandas-reference-only
```

减少进程内 representation 来回切。

---

## R42-210 [ARCH][P1][PERF] Mining Campaign Session

一次算法挖掘 run 创建：

```python
MiningCampaignSession(
    pinned source snapshot,
    pinned universe,
    compiler snapshot,
    shared CSE,
    factor evaluator,
    resource lease,
)
```

几千 candidate 全部复用。

---

## R42-211 [ARCH][P1][ACC] Campaign 内 snapshot 必须冻结

不能：

```text
第1个 candidate 读上午 source
第1000个读下午更新 source
```

导致比较不可比。

---

## R42-212 [ARCH][P1][PERF] 流式 factor evaluation，不必所有候选先落 lake

流程：

```text
compute candidate block
→ evaluator
→ DQ / coverage / IC pre-metrics
→ only accepted candidate durable materialize
```

减少垃圾因子写放大。

---

## R42-213 [ARCH][P1][ACC] pre-screen 与 final evaluation 必须严格分层

允许便宜预筛：

```text
短时间窗口
股票子样本
较低精度
```

但只能：

```text
reject obviously bad
```

最终生产因子必须：

```text
full exact evaluation
full universe
full precision
full PIT
```

---

## R42-214 [ARCH][P1][ACC] Pre-screen 需要 False-Negative Budget

不能为了速度把可能好因子提前误杀太多。

对历史候选回放：

```text
pre-screen reject
vs
full evaluation quality
```

估：

```text
false-negative rate
```

并设上限。

---

## R42-215 [ARCH][P1][PERF] Multi-fidelity evaluation

层级：

```text
L0 static legality
L1 small sample numeric/DQ
L2 recent period
L3 full historical
L4 robust regime/OOS
```

越往后成本越高。

---

## R42-216 [ARCH][P1][BOTH] Candidate 结构新颖度可在执行前评估，但只能用于调度优先级

基于：

```text
operator graph
field mix
economic mechanism tags
```

优先测试更不同的候选。

不能直接替代实证表现。

---

## R42-217 [ARCH][P1][PERF] Candidate correlation evaluation 也应 block/vectorized

1000 factor matrix：

```text
不要 Python pairwise corr loops
```

用：

```text
block matrix
incremental covariance
approx nearest-neighbor first pass
```

---

## R42-218 [ARCH][P1][ACC] 相关性去重使用 PIT-consistent 同一 coverage sample

两个因子 correlation：

```text
必须在共同有效股票日期上比较
```

避免 missing pattern 造成虚假低相关。

---

## R42-219 [ARCH][P1][PERF] 因子评价结果 cache

key：

```text
factor semantic identity
dataset snapshot
universe snapshot
evaluation convention
```

同一 candidate 被多个算法提出时直接复用。

---

## R42-220 [ARCH][P1][PERF] Mining API 支持 `compile_many` / `run_many_iter` 流式返回

算法可以：

```text
边收到结果边生成下一批
```

避免等待整个 campaign。

---

## R42-221 [ARCH][P1][PERF] 动态 batch size 根据候选复杂度

简单候选：

```text
更大 batch
```

复杂 rolling/model：

```text
更小 batch
```

避免固定 factor count。

---

## R42-222 [ARCH][P1][PERF] Candidate memory footprint 进入 batch grouping

同 batch 不只看 work，还看：

```text
peak buffers
output bytes
CSE lifetime
```

---

## R42-223 [ARCH][P1][PERF] Factor evaluator 直接消费 FactorBlock，不拆成 N 个 Series

例如计算：

```text
coverage
mean/std
IC
cross-factor correlation
```

block 化。

---

## R42-224 [ARCH][P1][PERF] Eval 与 writer 并行，但同一 FactorBlock 尽量零复制多播

```text
FactorBlock
├→ evaluator read-only
└→ writer read-only
```

refcount 完成后释放。

---

## R42-225 [ARCH][P1][ACC] Factor evaluator 的时间/股票轴也使用 AxisCertificate

不能 evaluator 自己重建/猜 index。

---

## R42-226 [ARCH][P1][PERF] 搜索算法读取已有 factor lake 统一走 DataAccess

把：

```text
raw fields
materialized factors
model scores
```

统一成 Feature Query API。

算法不再自己 `pd.read_parquet()`。

---

## R42-227 [ARCH][P1][BOTH] Factor lake 里的因子也要 source-like capability metadata

例如：

```text
frequency
coverage
PIT
universe
precision
snapshot
```

算法可在 compile 前知道是否可复用。

---

## R42-228 [ARCH][P1][PERF] materialized factor reuse

如果候选 DAG 的一个 primitive 已经是 durable factor：

```text
成本模型比较
重算
vs
读取已物化结果
```

选择更便宜的。

---

## R42-229 [ARCH][P1][ACC] durable factor reuse 必须 exact semantic match

不能：

```text
名字一样
```

就复用。

必须：

```text
SemanticIdentityDigest
SourceSnapshot compatibility
precision
calendar
universe
```

---

## R42-230 [ARCH][P2][PERF] Mining Campaign 可以做 locality-aware queue

多个 LLM agent 并行提交候选时：

```text
系统不按 agent 隔离批次
```

可跨 agent 合并相同 dependency signature，但结果权限/归属仍隔离。

---

# 12. FE × DataAccess 统一 QueryGraph：下一阶段最值得做的系统重构

## R42-231 [ARCH][P1][BOTH] 建立 `UnifiedQueryGraph`

节点类型：

```text
SourceResolve
SourceScan
Projection
Filter
PITJoin
TemporalAlign
Panelize
Primitive
RollingState
CrossSection
ModelScore
DQ
Materialize
```

FE 和 DA 共用一个 physical execution model。

---

## R42-232 [ARCH][P1][PERF] DataAccess 不再只作为“返回 DataFrame 的库”

它应提供：

```text
physical scan nodes
relation handles
stream handles
snapshot handles
```

供 FE optimizer 组合。

---

## R42-233 [ARCH][P1][ACC] FE 不直接决定底层文件路径

所有路径/对象：

```text
VerifiedPhysicalScope
```

来自 DA。

---

## R42-234 [ARCH][P1][PERF] QueryGraph CSE 跨越 FE/DA 边界

例如：

```text
同一 source scan
同一 PIT join
同一 panelization
```

也能共享，不只 FE 算子 subtree。

---

## R42-235 [ARCH][P1][PERF] PIT join 作为 native physical operator

财务/event source：

```text
as-of join
```

在 DuckDB 中一次做完，避免拉回 Pandas merge_asof。

---

## R42-236 [ARCH][P0][ACC] PIT join physical operator 带双时点 contract

```text
effective_time
knowledge_time
decision_time
```

缺任一 required axis 生产拒绝。

---

## R42-237 [ARCH][P1][PERF] Calendar/session mapping 也成为可复用 physical node

多 source：

```text
timestamp → session_date
```

一次映射共享。

---

## R42-238 [ARCH][P1][PERF] Universe membership join 也可下推

如果 DataAccess 支持：

```text
membership interval table
```

在 scan/join 阶段过滤 physical rows。

但 cross-sectional universe semantics 必须保持。

---

## R42-239 [ARCH][P1][PERF] Source block 统一 ColumnarBatch

建议 Arrow schema：

```text
session_ordinal
security_id
value columns...
validity
```

作为日频公共物理格式。

---

## R42-240 [ARCH][P1][PERF] 从 long→wide panel 的 pivot 尽量延迟

很多操作：

```text
DuckDB relation
long Polars
```

都能在 long format 做。

只有真正需要 wide TS kernel 时 pivot。

---

## R42-241 [ARCH][P1][PERF] 同一 source 不要反复 long↔wide

Compiler 规划：

```text
representation phase
```

把需要 wide 的连续 region 放一起。

---

## R42-242 [ARCH][P1][PERF] Panelization cache 绑定 AxisIdentity

同 source/snapshot/universe/time range：

```text
long→wide
```

一次，多个 native wide kernels 复用。

---

## R42-243 [ARCH][P1][PERF] wide panel 可以使用 contiguous 2D ndarray + separate axis

Pandas column objects 不适合高吞吐 kernel。

物理：

```text
values: C/F contiguous ndarray
time_axis
security_axis
```

Numba 直接消费。

---

## R42-244 [ARCH][P1][PERF] 根据 kernel access pattern 选择 C/F order

```text
沿 time 每股票 rolling
→ column/asset locality

横截面每日期
→ row/date locality
```

可维护两种 representation 的成本模型，而非盲目 transpose。

---

## R42-245 [ARCH][P1][PERF] transpose 也要进入 conversion cost

有时：

```text
复制一次 F-order
换后续 100 个 TS kernel 更快
```

CBO 应能选择。

---

## R42-246 [ARCH][P1][PERF] Arrow→NumPy 尽可能 zero-copy

满足：

```text
single chunk
fixed-width
no incompatible null representation
```

时借 buffer。

否则 consolidate 成 block，并记录 copy bytes。

---

## R42-247 [ARCH][P1][PERF] Chunk consolidation 只做一次

不要每 kernel：

```text
combine_chunks()
```

SourceBlock 创建阶段决定。

---

## R42-248 [ARCH][P1][PERF] dictionary encoded security IDs 保持 encoded

不要每层转 Python string。

内部 dense ID。

---

## R42-249 [ARCH][P1][PERF] date/session axis 使用 int32/int64 ordinal

计算 kernel 不需要 Python Timestamp。

最终输出再映射。

---

## R42-250 [ARCH][P1][BOTH] `FeatureBlock` 成为 FE/DA/Model 共用交付对象

```python
FeatureBlock(
    values,
    time_axis,
    entity_axis,
    feature_ids,
    semantic_certificate,
    source_snapshot,
    validity,
)
```

因子训练、模型训练、回测统一消费。


---

# 13. Catalog / Metadata / Control Plane：避免 10k 因子时元数据本身变瓶颈

## R42-251 [ARCH][P1][PERF] Catalog 读写与 value data path 分离

factor run 热路径不要频繁：

```text
SQLite SELECT/INSERT per factor
```

批量：

```text
prepare manifest rows
→ transaction batch
```

R39 已做部分，继续确保所有入口统一。

---

## R42-252 [ARCH][P1][PERF] Catalog connection 使用长生命周期 WAL 模式

如果 SQLite 仍是本地 catalog：

```text
WAL
busy_timeout
prepared statements
single writer queue
multiple readers
```

benchmark 并发写。

---

## R42-253 [ARCH][P1][PERF] Factor lookup 以 numeric internal ID 为主

```text
factor_id string
```

只在边界。

Catalog/Matrix 内部：

```text
factor_pk int64
```

减少宽字符串索引。

---

## R42-254 [ARCH][P1][PERF] Manifest 不反复 JSON parse

热读取：

```text
manifest → typed object cache
```

key：

```text
inode/etag/version/generation
```

变更才重 parse。

---

## R42-255 [ARCH][P1][PERF] 大 manifest 可以二级索引

例如 factor matrix 几万 objects：

```text
top manifest
→ partition manifest
→ block manifest
```

请求局部范围不用 parse 全量。

---

## R42-256 [ARCH][P1][PERF] Catalog stats incremental maintain

不要每次：

```text
COUNT / MIN / MAX 扫全部历史
```

commit 返回：

```text
delta stats
```

聚合。

---

## R42-257 [ARCH][P1][ACC] metadata cache 必须 generation-aware

Catalog update 后：

```text
旧 manifest object cache
```

立即失效。

---

## R42-258 [ARCH][P1][PERF] DataAccess registry freeze 后预编译 hot lookup tables

```text
dataset name → numeric ID
field ID → physical column
dataset → contract
```

避免每 query 重 parse YAML/dict。

---

## R42-259 [ARCH][P1][PERF] OperatorRegistry 同理生成 immutable compact snapshot

生产执行不碰：

```text
几十层 Python metadata dict
```

生成：

```text
arrays / compact dataclasses / integer IDs
```

---

## R42-260 [ARCH][P1][BOTH] Registry snapshot digest 直接进入 compile cache generation

任一 semantic registry 变化：

```text
相关 compile cache invalidation
```

dependency-scoped 优于全局失效。

---

## R42-261 [ARCH][P1][PERF] Dependency-scoped invalidation

如果只改：

```text
一个未被 factor 使用的 operator
```

不应让所有 factor compile/materialization cache 全失效。

Compiler 记录：

```text
used operator contract digests
used field digests
```

---

## R42-262 [ARCH][P1][PERF] Plan cache 分成 parsed / bound / optimized / physical 多层

不同变化：

```text
source snapshot 变
```

不一定需要重 parse/typecheck formula。

可复用：

```text
Parsed/Typed IR
```

只重做 physical source plan。

---

## R42-263 [ARCH][P1][PERF] 编译缓存避免缓存巨大 runtime objects

缓存：

```text
immutable compact IR
```

不要缓存：

```text
live source handles
DataFrames
connections
```

---

## R42-264 [ARCH][P1][PERF] Compile cache 使用 content-addressed storage

跨进程/服务 restart 可复用：

```text
formula semantic hash
compiler generation
```

但 production 先验证 build compatibility。

---

## R42-265 [ARCH][P1][PERF] Negative fusion cache 也应 generation-aware + bounded

避免某版本 backend bug 修复后：

```text
旧 negative cache
```

永久阻止融合。

---

## R42-266 [ARCH][P1][PERF] cost calibration store 也 versioned

绑定：

```text
backend version
host class
storage class
```

软件升级后降低旧样本权重。

---

## R42-267 [ARCH][P1][BOTH] Catalog 记录真实 physical execution summary

每次 factor generation：

```text
scan bytes
backend regions
conversion bytes
compute ms
write bytes
precision
```

未来 optimizer 用自己的历史。

---

## R42-268 [ARCH][P2][PERF] 用真实 workload 自动建议下一轮优化优先级

例如：

```text
ConversionAmplification P95 高
→ 优先 native region

WriteAmplification 高
→ delta/block layout

SchedulerAmplification 高
→ microbatch/intern
```

---

# 14. 多机/分布式：先把边界设计对，再决定是否上

## R42-269 [ARCH][P2][PERF] 不要过早把单机任务拆成分布式

A 股日频：

```text
5000 stocks × 数年
```

很多 workload 单机列式优化后已足够。

分布式只在：

```text
分钟全市场
几万候选
超大历史
多团队并发
```

有明确收益时启用。

---

## R42-270 [ARCH][P2][PERF] 分布式最自然的 shard 维度是可证明独立的 axis

例如 TS 因子：

```text
instrument shard
```

CS 因子不能按 instrument shard 独立算。

---

## R42-271 [ARCH][P0][ACC] Cross-sectional operator 必须有 `requires_full_cross_section=True`

分布式 planner 看到后：

```text
按 date shard
或 shuffle/gather
```

不能错误按股票拆。

---

## R42-272 [ARCH][P2][PERF] 分布式 source scan 尽量 data-local

如果有多服务器本地数据镜像：

```text
task → 数据所在节点
```

比网络拉大 panel 快。

---

## R42-273 [ARCH][P2][PERF] 分布式中间数据用 Arrow Flight/对象引用，不用 Python pickle

仅在确实需要跨节点时。

---

## R42-274 [ARCH][P0][ACC] distributed CSE 需要 content-addressed immutable object

同 semantic identity 同 snapshot 才共享。

---

## R42-275 [ARCH][P2][PERF] 全局 scheduler 与 host scheduler 两层

```text
GlobalCampaignScheduler
→ HostResourceCoordinator
```

global 分配 job；
host 内部仍自己做线程/内存/IO调度。

---

# 15. 测试：下一轮必须从“例子测试”升级到系统性 differential / metamorphic

## R42-276 [P0][ACC] 全生产算子 differential oracle

每个 canonical：

```text
reference backend
vs
all production backends
```

覆盖：

```text
default params
边界参数
随机合法参数
edge-value corpus
```

---

## R42-277 [P0][ACC] Metamorphic properties

按算子声明可用性质：

```text
permutation equivariance
scale invariance/equivariance
translation invariance
prefix invariance
chunk invariance
monotonicity
sign symmetry
```

不是所有算子都强行同一属性。

---

## R42-278 [P0][ACC] Temporal causality mutation test

对未来 `t+1...` 数据随机改动：

```text
factor value at <=t 必须不变
```

覆盖：

```text
daily
minute→daily
fundamental
event
model score
```

---

## R42-279 [P0][ACC] Universe mutation test

只修改：

```text
未来 universe membership
```

过去 factor 不变。

---

## R42-280 [P0][ACC] Source revision mutation test

修改：

```text
未来才知道的 restatement
```

历史 PIT factor 不变。

---

## R42-281 [P0][ACC] Backend route mutation test

强制：

```text
Pandas
Polars
DuckDB
Numba
```

结果应在 operator tolerance contract 内一致。

---

## R42-282 [P0][ACC] Representation-boundary fuzz

随机：

```text
row order
column order
dtype
null pattern
chunking
```

Arrow↔Polars↔NumPy 不得换轴/错列。

---

## R42-283 [P0][ACC] ReadWave failure negative tests

注入：

```text
source timeout
schema error
snapshot change
permission error
OOM
```

验证：

```text
失败 wave 不得 committed
typed recovery only
```

---

## R42-284 [P0][ACC] BufferRef representation truth test

如果 ref 宣称：

```text
ARROW_TABLE
```

其 location 必须真的 resolve 到 Arrow object。

禁止“标签 native，实际 pandas cache”。

---

## R42-285 [P0][PERF] Writer queue budget test

2/4/8 writers：

```text
sum capacity == configured total
```

---

## R42-286 [P0][ACC] Writer submit failure accounting

模拟：

```text
queue closed
timeout
oversized
```

确认：

```text
accepted 只计成功 enqueue
```

---

## R42-287 [P0][PERF] Oversized result test

```text
item > queue target
```

必须：

```text
spool/shard/direct
```

有限时间完成或明确失败，绝不无限等。

---

## R42-288 [P0][ACC] DataAccess stream deadline parity

同一个慢 query：

```text
arrow
stream
scoped arrow
scoped stream
```

都在同 absolute deadline 语义下失败。

---

## R42-289 [P1][PERF] Deadline thread count hard gate

1000 concurrent deadlines：

```text
deadline manager threads = O(1)
```

不是 1000。

---

## R42-290 [P1][PERF] ScopedConnectionPool contamination test

Query A：

```text
TEMP VIEW A
PRAGMA X
register Arrow
```

归还后 Query B：

```text
看不到 A 的临时状态
```

---

## R42-291 [P1][PERF] ScanCost routing replay test

保存真实历史：

```text
shape → actual TTDC
```

离线比较：

```text
旧 router
新 router
oracle best route
```

评估 regret。

---

## R42-292 [P1][BOTH] Optimizer differential plan test

同一 logical factor：

```text
optimizer OFF
optimizer ON
```

结果一致，TTDC 比较。

---

## R42-293 [P1][ACC] Rewrite fuzz

随机生成小 expression trees：

```text
before rewrite
after rewrite
```

edge corpus 上验证 equivalence level。

---

## R42-294 [P1][PERF] CSE liveness test

记录：

```text
buffer created
next use
last use
released
peak live bytes
```

确认无提前释放/迟释放。

---

## R42-295 [P1][PERF] Multi-output kernel parity

RollingStateBlock / rank block / regression block：

```text
multi-output
==
逐算子 reference
```

---

## R42-296 [P1][ACC] Float32 QuantizationCertificate test

至少：

```text
numeric diff
rank corr
top-k overlap
sign flip
IC delta
```

未达标不能 production publish。

---

## R42-297 [P1][PERF] Cold/Warm benchmark split

每个 benchmark：

```text
cold process + cold cache
warm metadata
warm data cache
steady state
```

分开报告。

---

## R42-298 [P1][PERF] Resource contention benchmark

组合：

```text
1000 factors + model train
1000 factors + compaction
2 mining campaigns
remote scan + local scan
```

观察：

```text
throughput
P95
RSS
fairness
```

---

## R42-299 [P1][ACC] Cross-process determinism

同 build：

```text
fresh process A
fresh process B
```

production factor 输出在 determinism contract 内一致。

---

## R42-300 [P1][ACC] Cross-build compatibility golden set

每次升级：

```text
上一 release
当前 release
```

对固定 snapshots 跑 golden corpus。

任何结果变化：

```text
必须有 semantic/numeric version explanation
```

---

# 16. R42 Benchmark Matrix

## B1 — 单因子基线

```text
50 个典型 daily factors
5年
全A
```

记录：

```text
compile
read
compute
write
```

---

## B2 — 1000 因子混合

包含：

```text
elementwise
TS rolling
CS rank
fundamental
mixed
```

重点：

```text
scan amplification
CSE
conversion
scheduler futures
```

---

## B3 — 10,000 mining candidates

重点：

```text
compile throughput
semantic dedup
negative cache
future count
Python object memory
```

---

## B4 — Incremental 1 day

```text
1000 / 5000 factors
```

最重要的每日生产场景。

目标：

```text
historical rewrite bytes = 0
full factor history rescan = 0
```

---

## B5 — Revision impact

```text
1个财务字段
10只股票
过去20个交易日 revision
```

目标：

```text
只重算真正受影响 dependency closure
```

---

## B6 — Minute→Daily 100/500 aggregations

目标：

```text
scan count ≈ source scopes
而不是 aggregation count
```

---

## B7 — 远程 COS

```text
cold
warm metadata
warm local cache
```

---

## B8 — FactorMatrix

```text
2000 factors
新增50列
新增1天
修订20天
```

分别测试：

```text
column delta
row delta
```

---

## B9 — Backend region

相同 100 factors：

```text
per-op routing
vs
native region routing
```

比较 conversion bytes/TTDC。

---

## B10 — RollingStateBlock

```text
5/10/20/60 windows
mean/std/zscore/min/max
```

---

## B11 — Cross-sectional block

```text
rank/zscore/winsor/quantiles
```

100因子 × 5年。

---

## B12 — Process lane

比较：

```text
pickle Pandas
vs
shared Arrow/mmap BufferRef
```

---

# 17. R42 关键性能 KPI

每次 benchmark 必须输出：

```text
TTFC
TTDC
Peak RSS
CPU utilization
IO throughput
Remote request count

ScanAmplification
DecodeAmplification
ConversionAmplification
ComputeAmplification
WriteAmplification
RewriteAmplification
SchedulerAmplification
PythonObjectAmplification
```

定义建议：

```text
ScanAmplification =
  physical_scanned_bytes / minimum_required_source_bytes

ConversionAmplification =
  bytes_crossing_representation_boundaries / final_unique_input_bytes

ComputeAmplification =
  executed_primitive_work / unique_required_primitive_work

WriteAmplification =
  physical_written_bytes / logical_new_or_changed_bytes

SchedulerAmplification =
  futures_or_tasks / final_factor_count
```

---

# 18. R42 Hard Gates

## HG-R42-01 Operator Contract

```text
PROD_OUTPUT_SHAPE_CONTRACT_MISSING == 0
PROD_HEURISTIC_PANEL_PARAM_COUNT == 0
PROD_HEURISTIC_DETERMINISM_COUNT == 0
```

## HG-R42-02 Semantic Type

```text
MIXED_PRICE_BASIS_UNDETECTED == 0
MIXED_PERIOD_DURATION_UNDETECTED == 0
UNSUPPORTED_SEMANTIC_DICT_KEY == 0
```

## HG-R42-03 ReadWave

```text
FAILED_WAVE_COMMITTED == 0
PREFERRED_REPRESENTATION_FALSE_CLAIM == 0
SOURCE_WAVE_UNTYPED_FAILURE == 0
```

## HG-R42-04 Deadline

```text
STREAM_DEADLINE_BUDGET_RESET == 0
SCOPED_QUERY_PER_QUERY_WATCHDOG_THREAD == 0
DEADLINE_QUERY_THREADS = O(1)
```

## HG-R42-05 Writer

```text
SUM_QUEUE_CAPACITY <= CONFIGURED_TOTAL
FAILED_ENQUEUE_ACCEPTED_COUNT == 0
OVERSIZED_RESULT_INFINITE_WAIT == 0
UNKNOWN_WRITE_ERROR_RETRIED == 0
```

## HG-R42-06 Cache

```text
CACHE_RELEASE_RECONCILIATION_REAL
CSE_RELEASE_AFTER_LAST_USE
NO_UNGOVERNED_SHARED_BUFFER
```

## HG-R42-07 Columnar Path

```text
DATAACCESS_ARROW_TO_POLARS_VIA_PANDAS_COUNT == 0
NATIVE_REGION_UNNECESSARY_MATERIALIZATION_COUNT == 0
AXIS_FULL_PYTHON_LIST_VERIFY_HOT_PATH == 0
```

## HG-R42-08 Numerical

```text
UNCERTIFIED_FLOAT32_PRODUCTION_WRITE == 0
BACKEND_NUMERIC_PARITY_FAIL == 0
UNCERTIFIED_APPROXIMATION == 0
```

## HG-R42-09 Incremental

```text
UNCHANGED_HISTORY_REWRITE_BYTES == 0
CHANGE_IMPACT_OUTSIDE_DEPENDENCY_CLOSURE == 0
```

## HG-R42-10 Mining

```text
SEMANTIC_DUPLICATE_EXECUTION_RATE < target
DETERMINISTIC_INVALID_CANDIDATE_REEXECUTION == 0
CAMPAIGN_SNAPSHOT_DRIFT == 0
```

---

# 19. 推荐实施顺序

## Phase A — 先修当前可确认 P0

优先：

```text
R42-001~006
R42-009~011
R42-029~031
R42-037~039
R42-045
R42-047~050
R42-051~055
R42-059
R42-101~105
R42-110~113
```

这些是不需要大架构重构也应该先改的。

---

## Phase B — FE/DA Read Path 统一

```text
R42-017~044
R42-091~100
R42-231~250
```

目标：

```text
Arrow/Relation first
一次 resolution
一次 scan
少 conversion
```

---

## Phase C — Compiler / CSE / Native Regions

```text
R42-121~150
R42-061~070
```

这是提升批量因子速度的核心。

---

## Phase D — Storage / Incremental

```text
R42-071~090
```

---

## Phase E — Numeric Accuracy

```text
R42-151~190
```

---

## Phase F — Mining Throughput

```text
R42-201~230
```

---

## Phase G — Control Plane / Scale

```text
R42-251~275
```

---

# 20. Coding AI 必须生成的 Closure Ledger

文件：

```text
factor_engine/docs/R42_ISSUE_CLOSURE_LEDGER.md
```

每一项格式：

```text
ID
classification: VERIFIED | ARCH
baseline status
current HEAD proof
files/functions
root cause
implementation
tests
negative controls
benchmark
before
after
semantic parity
status
```

状态只允许：

```text
FIXED
FIXED_ALREADY_WITH_CURRENT_HEAD_PROOF
NOT_APPLICABLE_WITH_PROOF
BLOCKED_BY_EXTERNAL_DEPENDENCY
ARCHITECTURE_BACKLOG_WITH_MEASURED_JUSTIFICATION
```

对于：

```text
VERIFIED + P0
```

不允许最终留：

```text
PARTIAL
OPEN
NOT_RUN
```

---

# 21. 执行中的再次审计要求

Coding AI 不要机械只做这 300 项。

整改完成后必须再次全仓搜索：

```text
pandas conversion
to_pandas
from_pandas
collect
combine_chunks
sort_values
drop_duplicates
merge(how="outer")
concat
duckdb.connect
ThreadPoolExecutor
ProcessPoolExecutor
threading.Thread
wait(timeout=
except Exception
default=str
repr(
str(k)
isfinite
groupby
rolling
rank
quantile
to_list
```

对 hot path 做 flamegraph / sampling profiler。

新发现：

```text
R42-NEW-301+
```

继续加到 ledger。

---

# 22. Definition of Done：什么叫“FactorEngine 和 DataAccess 更快、更准”

## 更快

不是单个 operator 快 10%，而是：

```text
同 source 尽量只扫一次
同 column 尽量只 decode 一次
同 representation 尽量少转换
同 primitive 尽量只算一次
同 rolling state 尽量多输出
同 partition 尽量少写/少重写
```

并且：

```text
1000-factor TTDC
5000-factor TTDC
1-day incremental TTDC
```

都有真实改善。

## 更准

必须：

```text
交易日/日历正确
PIT/knowledge time正确
动态股票池正确
字段单位/flow duration正确
price basis正确
axis正确
missing/validity正确
backend数值一致
float32/approximation有证书
```

## 更稳

必须：

```text
资源预算是真预算
deadline 是全链一个 deadline
writer失败不会假成功
source wave失败不会假 committed
cache账实一致
并发不会 OOM 失控
```

---

# 23. R42 最终交付清单

Coding AI 完成后必须交：

```text
1. execution-start HEAD
2. final HEAD
3. R42_ISSUE_CLOSURE_LEDGER.md
4. R42-001..300 current status
5. R42-NEW-301+ findings
6. exact changed files
7. unit/integration tests
8. differential/metamorphic tests
9. hard-gate report
10. benchmark raw JSON/Parquet
11. before/after summary
12. flamegraph / profiler summary
13. FE↔DA end-to-end run
14. cold/warm benchmark split
15. correctness parity evidence
16. remaining architecture backlog
```

---

# 24. 给 Coding AI 的最终执行指令

```text
A. 重新读取执行时最新 main，不假设本文 baseline 仍最新
B. 先读取 R41 closure，避免和已落地修复打架
C. 对 R42 VERIFIED 项逐一在 current HEAD 重新证明
D. 先修 VERIFIED P0
E. 建立 profiler baseline
F. 修 ReadWave/BufferRef representation truth
G. 修 DataAccess deadline/scoped connection
H. 修 writer/cache 当前 bug
I. 建立统一 semantic compiler transfer contract
J. 建 NativeRegion / CSE liveness / multi-output primitives
K. 把 DataAccess read plan 与 FE consumer graph联动
L. 推进 Arrow/Relation-first，减少 Pandas
M. 优化 Base+Delta / Matrix block
N. 建 NumericPolicy / QuantizationCertificate
O. 优化 mining campaign 1000/10000 candidate 场景
P. 跑 differential/metamorphic/PIT negative tests
Q. 跑 B1-B12 benchmark
R. 用 profiler 再找新的热点，补 R42-NEW-301+
S. 所有 VERIFIED P0 闭环后再结束
```

---

# 25. 最终架构判断

当前 FactorEngine / DataAccess 已经不是“缺几个算子、再加几个线程”的阶段了。

下一阶段最值得投入的不是继续横向堆 feature，而是把现有大量能力收拢成四个真正统一的核心：

```text
1. 一个 Semantic Compiler
2. 一个 Unified QueryGraph
3. 一个 Columnar Buffer Runtime
4. 一个 Cost + Accuracy Constrained Optimizer
```

如果这四层真正收拢，速度提升会来自结构性减少重复工作，而不是参数调优：

```text
扫描更少
转换更少
重复计算更少
重写更少
Python 对象更少
```

准确性也会同步上升，因为：

```text
history
calendar
universe
unit
price basis
availability
missing
numeric policy
```

不再散落在 analyzer / registry / backend / DataAccess 的多套 heuristic 中。

R42 的最终原则：

> **性能优化首先减少不必要的工作，而不是用更多资源完成重复工作。**

> **任何加速只有在 semantic identity、PIT、axis、numeric parity 都有机器证据时，才允许进入 production。**
