# FactorEngine R33（修订完整版）：DuckDB 优先但成本驱动——FactorEngine × DataAccess Unified Query Graph、零拷贝扫描复用、批量计算与极速准确落值方案

- 仓库：`18047533889/quant_projects`
- 审计基线 HEAD：`499078505edff30b1268b60862dbcaab2f9ab31e`
- 日期：2026-08-10
- 核心目标：**不是让单个 operator 再快一点，而是让一批 100 / 1,000 / 10,000 个因子从“依赖解析 → 读数 → 计算 → DQ → 落盘”整条链的总 wall time 尽可能短，同时保证 PIT、snapshot、语义、数值和物化结果不变。**
- 本轮重点：FactorEngine（FE）与 DataAccess（DA）真正合并成上下两层查询优化器，而不是各自规划、各自缓存、各自资源准入，再在 Python 边界拼起来。
- 前置约束：
  - 不回退 R30/R31/R32 已建立的 correctness / PIT / production safety。
  - 不为了速度绕开 DataAccess snapshot / auth / query budget / PIT / schema gate。
  - 不通过“关闭检查”换吞吐。
  - 不把“模块存在”当“主路径真的执行”。
  - 所有性能结论必须以当前 HEAD + 当前服务器 + 当前数据快照 benchmark 为证据。

---


# 零、修订版强制说明：本版覆盖上一版 R33

本文件是 R33 的修订完整版，后续整改以本文件为准。若与上一版 R33 存在表述差异，**以本版为准**。

本次修订最重要的变化有四个：

1. **不再把后端策略理解成固定的 `DuckDB > Polars > Pandas` 排名。**
2. 改成 **DuckDB-first / relation-first / cost-driven**：DataAccess 本身以 DuckDB 为核心读与多表关系执行层，因此当一批因子能够把 `scan + filter + PIT/asof join + aggregation + window + multi-root projection + direct write` 留在同一个 DuckDB query graph 内时，默认优先 DuckDB；但只要 Polars 能形成更大的 native streaming subgraph、减少 materialization / conversion，或者实测更快，必须允许 Polars胜出。
3. 后端路由从“**单个 root 选最快 backend**”进一步升级成“**整批因子的全局物理计划选最短 time-to-durable-commit**”。
4. 把“快”定义为：

```text
不是 operator benchmark 最快
而是：
数据读完 + PIT 对齐 + 因子算完 + DQ 完成 + 数据可靠落盘 + generation 原子提交
这一整条链最快。
```

## 0.1 DuckDB 是否一定比 Polars 快？结论：不能写死

官方 benchmark 本身就说明两者会随 workload / 数据规模变化：

- Polars 官方 2025 PDS-H：SF-10 中 Polars streaming 总时间约 3.89s，DuckDB 约 5.87s；SF-100 中 DuckDB 约 19.65s，Polars streaming 约 23.94s。
- DuckDB 官方 performance benchmark 文档也明确强调跨系统 benchmark 很难公平，结果强依赖 workload、版本、硬件和配置。
- Polars lazy engine 同样具备 predicate pushdown、projection pushdown、common subplan elimination、join ordering、streaming 等查询优化能力。

因此本项目不应硬编码：

```text
DuckDB 永远第一
Polars 永远第二
Pandas 永远第三
```

而应该硬编码以下原则：

```text
1. correctness / PIT / snapshot / backend certification 先决定“能不能跑”；
2. 尽量让数据留在当前 source-native backend，避免转换；
3. 尽量让更大的连续 subgraph 在一个 native engine 内完成；
4. 优先能够 multi-root / direct sink 的计划；
5. 在同等或成本不确定时，对 DataAccess parquet/relational workload 给 DuckDB tie-break preference；
6. 有可信 measured baseline 时由实测成本决定；
7. Pandas 只保留 reference + specialized fallback，不作为大批量默认数据面。
```

## 0.2 推荐的默认 Backend Policy

### Lane A：DuckDB Relational Native Lane —— **默认第一候选，不是强制第一名**

满足以下条件时，优先把整批放在 DuckDB：

```text
DataAccess source 是 parquet / DuckDB relation
+ source scan 可以下推
+ PIT/asof join 可以在 SQL 内完成
+ universe/filter/aggregation 可以下推
+ factor operators SQL production certified
+ 多个 factor 可以同一个 SELECT / WINDOW graph 输出
+ 最终结果可以直接流向 Arrow batches / staging writer
```

这里 DuckDB 的核心优势不是“单个 add 比 Polars 快”，而是：

```text
从文件扫描一直到最终多因子输出都不用离开关系执行引擎。
```

### Lane B：Polars Streaming/Lazy Native Lane —— **与 DuckDB 竞争的一级后端**

更适合：

```text
大量纯列式表达式
复杂 expression DAG
projection/filter 非常强
wide multi-output with_columns
streaming pipeline
部分 rolling / group / reshape 在 Polars native 表达更自然
DuckDB SQL emitter coverage 不完整但 Polars coverage 完整
```

如果把 DuckDB结果先 materialize 成 Arrow/Pandas，再去 Polars，通常不值；
但如果 DA 可以直接给 Polars 一个 governed LazyFrame，且后续大段计算都留在 Polars，一次 collect/sink，则完全可能比 DuckDB更优。

### Lane C：Specialized Native Kernel Lane

有些 recursive / stateful / numerical operators 不适合硬塞 SQL/Polars：

```text
Kalman
复杂 path-dependent state machine
部分 rolling regression
特殊优化算法
```

这时不要简单理解成“Pandas 慢所以不用”。正确做法是：

```text
Pandas 作为 reference semantics
NumPy / SciPy / Bottleneck / Numba 等作为 specialized execution kernel
```

输出仍进入统一 BufferRef / FactorBlock。

### Lane D：Pandas Reference Lane

Pandas承担：
- canonical reference；
- differential testing；
-极少数无 native implementation 的 fallback；
-小数据 / debug / research。

**不允许大批量 production 因子因为一个很小的 unsupported operator整棵 root退回 Pandas。**

## 0.3 真正的优先级不是 backend 名字，而是“少一次大转换”

例如：

```text
计划 A：
Parquet → DuckDB scan → PIT join → 400 factor SQL → direct Arrow batch sink

计划 B：
Parquet → DuckDB read → Arrow full materialize → Polars → 400 factor → Arrow → writer
```

即使某些 expression 在 Polars microbenchmark 更快，A 很可能整体更快，因为少了：
- full materialization；
- DuckDB→Arrow→Polars 转换；
-额外内存峰值；
-重复 scheduling。

反过来：

```text
Parquet → Polars scan → 600 Polars-native factors → streaming sink
```

如果无需 DuckDB join，Polars也可能是更优主路径。

所以后端选择必须以：

```text
END_TO_END_COST
```

而不是：

```text
OPERATOR_COST_ONLY
```

---

# 一、这轮的核心判断

最新 HEAD 已经比 R31 审计时前进很多：

```text
FactorEngine
  ├─ BatchDataRequest
  ├─ PhysicalLowerer
  ├─ AdaptiveBatchScheduler
  ├─ ResourceBroker
  ├─ NativeFusion
  └─ StreamingResultSink

DataAccess
  ├─ DataRequest / ReadPlan
  ├─ PreparedRead
  ├─ ReadPipeline
  ├─ PhysicalPlan
  ├─ ScanCost
  ├─ sql_relation / ScanHandle
  └─ generation pointer
```

但是现在仍然存在一个很明显的结构问题：

> **FE 与 DA 已经分别长出了“查询规划器”，但还没有真正变成一个统一的执行图。**

当前更接近：

```text
FE Analyzer
  ↓
FE BatchDataRequest
  ↓
FE Physical DAG（部分 stage 是规划视图 / no-op）
  ↓
root backend.execute(whole_plan)
  ↓
DataAccessSource
  ↓
DA read / scan / ReadPlan
  ↓
Arrow / Pandas / Polars
  ↓
FE operator recursion
  ↓
per-factor result
  ↓
writer
```

最终应改成：

```text
FactorBatch
  ↓
BatchAnalyze + IR Interning
  ↓
UnifiedDataDemandGraph
  ↓
DataAccess PreparedBatchPlan
  ↓
Unified Physical Query Graph
  ├─ SourceScan
  ├─ PIT/ASOF Join
  ├─ Universe Filter
  ├─ Minute→Daily Aggregate
  ├─ FE SQL/Polars Native Subgraph
  ├─ Shared Rolling / Group / CS Stage
  ├─ Specialized Pandas Kernel
  └─ MultiRoot Output Block
  ↓
Streaming DQ
  ↓
Partition-aware / Matrix-block Writer
  ↓
DataAccess Generation Commit
```

一句话：

> **下一阶段最大的性能收益，不来自“更多线程”，而来自“少读、少转、少排队、少重复算、少落小文件、少在 FE/DA 边界来回物化”。**


## 二、R31 之后已经改善的地方：这些不要推倒重来

当前 HEAD 已经有明显进展，整改 AI 不要重新实现第二套：

### 2.1 默认 run_many 已经切到 AdaptiveBatchScheduler

除显式 `FACTOR_ENGINE_LAYER_LOOP=1` 外，默认批跑已进入 scheduler。

这项保留。

### 2.2 FE 已经有 BatchDataRequest

说明“整批因子先 union 依赖”的方向是对的。

问题不是删除它，而是：

```text
把它改成 DataAccess DataRequest / ReadPlan 的 adapter / compiler input，
不要继续发展成一套与 DA 平行的数据规划系统。
```

### 2.3 FE 已经有 PhysicalLowerer

保留 physical task / resource / backend / shard 等概念。

但要从：

```text
planning-only stage
```

升级成：

```text
executable data-plane stage
```

### 2.4 DA 已经有 DataRequest / ReadPlan

这是最应该被 FE 深度复用的接口：

```text
field resolve
dataset coalesce
multi-dataset join
PIT
universe
aggregation
scan cost
snapshot policy
physical plan
```

不要 FE 再重复写一遍。

### 2.5 DA 已经有 PreparedRead / ReadPipeline

生产治理链方向正确：

```text
contract
snapshot
budget
governor
verify
execute
release
```

下一步是让这套安全机制支持：

```text
PreparedBatchReadSession
```

而不是每个 column/root 反复走完整生命周期。

### 2.6 Native Fusion / Streaming Sink 已有骨架

保留，但需要：
- 真 backend capability；
- 真 multi-root execution；
- 真 matrix/block output；
- 真 backpressure feedback；
- 真 writer failure propagation。

### 2.7 Generation Pointer 思路正确

“完整新代 → 原子 pointer flip”是正确的生产语义。

但物理实现要改成：

```text
manifest-level copy-on-write
```

避免更新少量分区却复制整代历史。


## 三、当前 HEAD 新确认的性能与融合问题总表

下面这些是本轮最需要优先处理的项目。

---

### R33-P0-001｜FE BatchDataRequest 目前实际只构造一个 anchor SourceScanGroup

当前 `build_batch_data_request()` 最终仍然：

```text
source = engine.data_source
dataset = source.dataset
group_id = 0
```

没有真正把：

```text
secondary SourceRef
fundamental dataset
industry dataset
universe dataset
corporate action
minute source
event source
```

拆成独立 source group。

因此“one/few DataRequests”目前更接近“one anchor request”。

**整改：**

BatchDataRequest 不再从 `engine.data_source` 推断全局源，而从：

```text
compiled plans
source dependency graph
field semantic plans
SourceRef nodes
```

生成：

```python
SourceDemand(
    dataset,
    fields,
    time_range,
    instruments/universe,
    filters,
    source_params,
    pit_policy,
    joins,
    transforms,
    aggregations,
    snapshot_policy,
)
```

---

### R33-P0-002｜FE BatchDataRequest 与 DA DataRequest 是重复规划层

DA 的 `DataRequest / ReadPlan` 已经明确设计成：

```text
FactorEngine Analyzer 一次性提交的逻辑数据需求
```

并且已经支持：
- logical field resolution；
- 多 dataset coalesce；
- join；
- PIT；
- universe；
- minute aggregation；
- ScanCost；
- engine/result routing。

因此 FE 不应该继续维护一套独立的 data planner。

**目标：**

```text
FE BatchDataRequest
= FE-side demand manifest
→ compile_to_dataaccess_requests()
→ DA DataRequest / ReadPlan
```

DA 成为唯一 Source Planning Authority。

---

### R33-P0-003｜BatchDataRequest 的 ScanCost failure 会被宽泛吞掉

当前：

```python
try:
    scan_cost = estimator(...)
except Exception:
    scan_cost = None
```

性能规划失去成本数据时不应该无痕降级。

改成：

```text
ScanCostUnavailable(
  dataset,
  reason,
  fallback_estimate,
  confidence
)
```

Production correctness可继续；
performance evidence必须记录 degraded planning。

---

### R33-P0-004｜BatchDataRequest ScanCost 没带真实 time_range / instruments

当前只：

```text
estimate_scan_cost(fields=ordered)
```

没有把每批真实：

```text
warmup range
requested range
instrument/universe
filters
source params
```

传入。

这样得到的 selected bytes可能接近全数据集成本，无法指导 wave。

---

### R33-P0-005｜BatchDataRequest source_scope 与 PhysicalLowerer source_scope 不一致

BatchDataRequest可能构造：

```text
dataset:foo::snapshot:xxx::market:ashare
```

PhysicalLowerer 当前常见：

```text
dataset:foo
```

而 `scope_scan_cost_map` 按完整字符串查 key。

结果：

```text
ScanCost已经算出来，
但 Physical task lookup 很可能 miss。
```

统一使用 typed `SourceScopeId`，禁止字符串拼接/拆分。

---

### R33-P0-006｜`_extract_dataset()` 可能返回 `dataset:foo` 而不是 `foo`

不要通过：

```text
split("::")
```

解析业务身份。

用 typed fields。

---

### R33-P0-007｜PhysicalLowerer 仍明确说明 production 计算是 whole-root execute

目前 physical stage更多是：

```text
cost/admission/explain
```

ROOT才真正：

```text
backend.execute(root_plan)
```

这意味着 physical DAG还不是实际数据平面。

**核心整改：**

每一个 executable stage必须有：

```text
input BufferRef(s)
output BufferRef
execute_stage()
```

---

### R33-P0-008｜scheduler `_dispatch()` 对 SOURCE_SCAN / ROLLING / GROUP / CS / STATEFUL / OPERATOR 都是 no-op

目前真正执行的主要只有：

```text
CSE_SHARED
ROOT
```

其它 task：

```text
return (task_id, None)
```

但是这些 no-op task仍参加：
- resource admission；
- scheduling；
- timing；
- dependency。

这增加控制面开销，却没有减少 root compute。

**整改：**
要么：
1. 真正实现 executable stage；
要么：
2. 在未实现前不要把虚拟 stage当 runtime task admission。

最终必须选 1。

---

### R33-P0-009｜SOURCE_SCAN 算出的 source_cols 没存进 PhysicalFactorTask

Lowerer有：

```text
source_cols = _walk_source_columns(plan)
```

但 task schema没有 `required_columns`，
最终 source task丢失真实 IO需求。

新增：

```python
SourceScanSpec(
    dataset,
    required_columns,
    time_range,
    instrument_scope,
    filters,
    source_params,
    snapshot_id,
    pit_policy,
    ordering,
)
```

---

### R33-P0-010｜ReadWavePlanner 当前拿 SOURCE_SCAN.inputs 当 columns

SOURCE_SCAN.inputs 是 predecessor task ids，通常为空。

所以：

```text
ReadWave.columns
```

不是实际物理列。

这是当前明确 bug。

---

### R33-P0-011｜ReadWavePlanner 对 ROOT/CSE 用 `(task.op,)` 当 columns

可能把：

```text
ts_mean
add
rank
```

这样的 operator canonical当成物理字段。

必须彻底删除这种推断方式。

---

### R33-P0-012｜ReadWave 的 time_range 当前固定 None

因此：

```text
5日短因子
250日因子
full-history因子
```

无法通过 wave planner用真正历史需求区分。

---

### R33-P0-013｜ReadWave dataset 从 source_scope字符串反推

typed source identity必须替代：

```python
task.source_scope.split("::")[0]
```

---

### R33-P0-014｜ReadWave memory 将每个 request 内存直接相加

共享列被重复计。

例如：

```text
100 factors 都需要 close
```

真实 wave只需 close buffer一份，
当前可能按100份 request memory相加。

导致：
- wave过度拆分；
- 共享扫描减少；
-吞吐下降。

应按：

```text
union projected columns
actual representation bytes
```

计算。

---

### R33-P0-015｜ReadWave greedy 实际不是按 marginal overlap

当前主要按：

```text
-len(columns)
```

排序。

应优化：

```text
marginal_scan_bytes
marginal_memory_bytes
shared_column_overlap
source locality
history overlap
consumer criticality
```

---

### R33-P0-016｜ReadWavePlan 当前生成了但 scheduler.run 没真正执行

这是本轮最重要的 wiring gap 之一。

scheduler plan里有：

```text
plan.read_waves
```

run loop却主要只消费：

```text
physical_dag
fusion_groups
```

没有：

```text
execute wave → produce source buffer → unblock tasks
```

所以 ReadWave现在仍是 planning artifact。

---

### R33-P0-017｜默认 scheduler 前仍有 `_maybe_prepare_batch_data` 全 union prefetch

当前 run_many scheduler之前仍会：

```text
all referenced columns union
→ engine._prepare_batch_data(...)
```

除 fully SQL/native skip外。

这和 ReadWave形成两套路径：

```text
全 union prefetch
+
wave planning
```

必须收敛成一套。

最终生产默认：

```text
read waves
```

而不是 batch-wide eager prefetch。

---

### R33-P0-018｜Batch warmup 对所有因子取统一最早窗口

一批里：

```text
990个 20日因子
10个 full-history因子
```

现在可能让整批 source window被 full-history拉长。

应按：

```text
source scope
history class
lookback band
backend
```

聚类。

建议：

```text
short
medium
long
full-history
```

但不是只分类用于调度，而是真正生成不同 source wave。

---

### R33-P0-019｜DataAccessSource 每次 load/scan仍会 refresh_snapshot

虽然 manifest token较便宜，
但一个 batch内不应在各层重复确认 snapshot。

建立：

```text
BatchReadSession.snapshot_token
```

整批 pin一次，必要 terminal verify一次。

---

### R33-P0-020｜DataAccess `prepare_read()` 会在 prepare阶段就拿 governor reservation

对单请求合理。

对大 batch如果先 prepare很多 wave：

```text
还没 ready就持资源 reservation
```

可能：
- 资源被预占；
- 其它 ready任务进不来；
- 形成过度保守甚至互等。

拆成：

```text
PreparedReadTemplate     # 无资源 lease
ExecutionLease           # scheduler JIT admission才获取
```

---

### R33-P0-021｜FE ResourceBroker 与 DA GlobalResourceGovernor 是两个资源准入系统

当前可能发生：

```text
FE reserve memory/IO
↓
调用 DA
↓
DA 再 reserve scan/memory/remote
```

同一份资源被 double-account。

严重时：
- 过度降并发；
- 吞吐下降；
- 甚至形成资源等待闭环。

建立：

```text
HierarchicalResourceLease
```

FE host-global broker是总预算，
DA scan从 FE task lease中派生子 lease，
而不是二次独立占同一份预算。

---

### R33-P0-022｜DA prepare_read 的路径/manifest/schema/snapshot工作可能在 batch重复

prepare_read包含：
- resolve paths；
- glob expand；
- file manifest；
- budget；
- snapshot；
- schema；
- schema epoch；
- contract；
- predicates；
- temporal plan。

同 dataset/snapshot/window的一批因子应共享：

```text
PreparedDatasetScope
```

---

### R33-P0-023｜DA resolve_fields(names) 虽是 batch API，内部仍逐 name处理

对每个 unresolved physical field还可能扫描 registry datasets。

建立：
- `SemanticFieldCatalog.resolve_many()`；
- alias index；
- `(dataset, physical_name) -> SemanticField` reverse index；
- global `physical_name -> datasets` index。

目标复杂度：

```text
O(fields)
```

而不是近似：

```text
O(fields × datasets)
```

---

### R33-P0-024｜DA ReadPlan已经能做 multi-dataset coalesce，但 FE没把全批 source graph交给它

应该由 FE产生：

```text
one/few DataRequest per compatible query scope
```

而不是每个 DataAccessSource自己 read。

---

### R33-P0-025｜DA physical plan也仍部分是 explain tree，不是完全 executable IR

当前：

```text
aggregation + multi-dataset join
```

有组合执行器，
其它场景仍回旧 read/read_joined路径。

需要把 DA physical plan也升级成：

```text
ExecutableRelationalPlan
```

---

### R33-P0-026｜DA 聚合 + Join 组合路径会写临时 Parquet再让 DuckDB重读

当前组合执行大意：

```text
minute aggregate
→ Arrow
→ temp parquet
→ DuckDB read parquet
→ join
```

这是明显的 materialization barrier。

整改优先级极高：

优先：
```text
single DuckDB SQL CTE
```

其次：
```text
connection-scoped registered Arrow relation/view
```

禁止 hot path：

```text
Arrow → temp Parquet → DuckDB
```

---

### R33-P0-027｜read_joined 会把最终结果先 materialize 成 Arrow

如果后面 FE仍是 SQL-emittable subgraph，
此时不应该离开 DuckDB。

增加：

```text
ReadPlan.open_relation()
RelationHandle.append_project()
RelationHandle.append_window()
RelationHandle.collect_to_sink()
```

---

### R33-P0-028｜sql_relation 会重新做 snapshot/path解析

如果 FE已经拿到 PreparedBatchPlan，
应复用：

```text
exact physical scope
snapshot
contract
budget lineage seed
```

禁止 relation路径二次 resolve。

---

### R33-P0-029｜Eager DataAccessSource 过早 Arrow→Pandas MultiIndex Series

当前：

```text
DA Arrow
→ arrow_table_to_multiindex_columns
→ dict[str, Series]
→ FE cache
```

后面很多 operator又可能：
- unstack；
-转 Polars；
-转 NumPy。

建议 canonical in-memory batch representation：

```text
Arrow/Polars long block
```

仅在 Pandas-only stage边界转换。

---

### R33-P0-030｜`_column_cache` + `_panel_cache` 双份 representation

同一字段可能同时存在：
- MultiIndex Series；
- wide DataFrame。

浪费内存和转换时间。

更进一步，当前 `_cache_bytes` 是两缓存共用，
但 `_put_cache(cache,...)` 超预算时只从“当前这个 cache”淘汰，
因此总预算有可能仍超限。

整改：
- 统一 `RepresentationCache`；
- 一个 global LRU；
- value按 representation登记；
- conversion view尽量不复制；
- value/recompute cost-aware eviction。

---

### R33-P0-031｜Polars long路径可能重复 sort

如果 DA manifest/storage已经能证明：

```text
(inst, ts)
```

排序，
FE不应该每次 re-sort。

增加：

```text
OrderingCertificate
```

只有 ordering未知/不满足时才 sort。

---

### R33-P0-032｜Physical stage cost仍反复使用 whole-plan cost

barrier stage的 contract：

```text
contract_for_plan(plan)
```

传的是完整 root plan。

导致每个虚拟 stage都可能被估成“整棵 root成本”。

必须 stage-local：
- subgraph cost；
- input/output bytes；
- materialization；
- conversion；
- scan；
- write。

---

### R33-P0-033｜CSE shared task backend仍硬编码 pandas

`lower_batch_dag` shared nodes：
- backend_candidates空；
- preferred pandas；
- resource contract按 pandas。

CSE shared也必须 route。

---

### R33-P0-034｜PhysicalLowerer 的 scan_cost_map参数没有真正用于 task lowering

把 DA ScanCost直接写进：

```text
SourceScanTask.resource_contract
```

---

### R33-P0-035｜backend_threads目前基本 hardcode 1

对于：
- DuckDB；
- Polars；
- NumPy/BLAS；
- Arrow scan；

实际 thread budget应来自 host-global broker和阶段规模。

---

### R33-P0-036｜shardability metadata与 resource contract不一致

SOURCE_SCAN task可能声明：

```text
ShardSpec(time, legal=True)
```

但 resource contract仍：

```text
shardable=False
```

统一一份事实源。

---

### R33-P0-037｜scheduler计划时 instruments数目前仍接近 0/未知

需要从：
- resolved universe；
- DataAccess ScanCost；
- request instruments；
- dataset stats；

得到 cardinality estimate。

---

### R33-P0-038｜scheduler `max_concurrency` 没成为显式 admission上限

即使 ResourceBroker能间接控制，也应有明确：

```text
running_futures < dynamic_concurrency_limit
```

避免大量廉价 task同时 submit。

---

### R33-P0-039｜scheduler ready set 使用 set，优先级没有真正主导 admission

虽然代码已有 `task_priority` 概念，
当前主循环仍应改成真正 priority queue：

```text
critical path
reuse
data-ready
marginal memory
sink backpressure
```

---

### R33-P0-040｜scheduler 采用固定 50ms polling wait

大量短 task时，50ms控制周期非常昂贵。

改：
```text
FIRST_COMPLETED
Condition/Event
broker resource-release notification
sink-backpressure notification
```

---

### R33-P0-041｜CSE release failure在 scheduler路径被 `except: pass`

内存回收失败不能静默。

Production：
```text
CSE_RELEASE_FAILURE
```

至少 telemetry + fail-safe cleanup。

---

### R33-P0-042｜Native Fusion 对整批 root先做一次 global can_fuse

只要有一个不同 scope/backend root，
可能整批 fusion关闭。

应该：
```text
先 partition by compatible scope
再每组 can_fuse
```

---

### R33-P0-043｜fusion block只在第一个 scope group自适应

`fusion_block=None` 在第一个 group算出后，
后续 group可能复用同一个 block。

每 group独立计算。

---

### R33-P0-044｜scheduler调用 fusion planner时没有真实 backend capability map

只有确实实现并认证：

```text
execute_multi_roots
```

的 backend才能创建 fusion group。

否则直接不计划，不要 runtime再 fallback。

---

### R33-P0-045｜Fusion fallback异常被吞后自动 per-root，缺性能原因证据

如果 multi-root执行失败：
- correctness可以 fallback；
- production perf evidence必须记录 exception type/reason/op/scope；
- certified backend若无合理 transient原因，不应长期静默 fallback。

---

### R33-P0-046｜Streaming sink backpressure没有真正反馈给 scheduler admission

队列有 `backpressure_ratio`，
scheduler需把它放进：

```text
dynamic concurrency
new root admission
wave prefetch
```

---

### R33-P0-047｜Streaming sink默认 batch_size=1 / writer_threads=1

对大量因子小结果：
- open/write/close；
- catalog calls；
- partition metadata；
会成为瓶颈。

需要 benchmark自适应：
```text
factor block width
writer count
partition ownership
```

---

### R33-P0-048｜BoundedResultQueue 使用 list.pop(0)

队列长时是 O(n)移动。

改：
```text
collections.deque
```

---

### R33-P0-049｜writer失败的 requeue路径不可靠

当前失败：
```text
requeue
continue
```

问题：
- poison batch无限循环；
- requeue失败返回值未成为 fatal；
- 没有 retry budget；
- 没有 error propagation；
- finish可能看不到失败。

建立 writer state machine：
```text
ACTIVE
RETRYING
FAILED
DRAINED
```

---

### R33-P0-050｜sink.finish 只 join timeout，不证明写完

生产 run不能在 writer仍活着/未落完时返回成功。

必须：
```text
all accepted items committed
writer threads terminated
fatal_error is None
```

---

### R33-P0-051｜writer多线程不能随便增加，必须 partition-aware

否则两个 writer可能同时写：
```text
同 factor / 同 date partition
```

需要：
```text
partition owner / shard key
```

同 partition单 writer，
不同 partition并行。

---

### R33-P0-052｜DA generation writer会 Arrow→Pandas 做 partition groupby

避免大表整张 `to_pandas()`。

改：
- Arrow compute；
- PyArrow dataset write_dataset；
- Polars partition；
- DuckDB COPY PARTITION_BY；
按 benchmark选。

---

### R33-P0-053｜DA generation multi-key upsert会 Pandas + astype(str) + row-wise agg

对百万/千万行很慢。

改：
- DuckDB anti join；
- Arrow struct key；
- Polars join；
禁止 Python row-wise字符串拼 key。

---

### R33-P0-054｜Generation copy设计可能产生整代写放大

`copy_generation()` 会物理复制整代文件。

必须审计所有 hot-path call sites。

最终建议：

```text
Generation Manifest
  partition A -> immutable object v1
  partition B -> immutable object v8
  partition C -> new object v9
```

新 generation只写变更 partitions，
未变 partitions复用 object reference。

---

### R33-P0-055｜Factor output仍以 per-factor Series为主要边界

1000因子：

```text
1000 Series
→ 1000 normalize/materialize calls
→ 1000 writer items
```

最终应：

```text
FactorBlock / MatrixBlock
```

例如 32/64/128个 root一组，
直接输出同一 `(date, asset)` block。

---

### R33-P0-056｜Cross-sectional operator仍缺 batch-matrix execution lane

大量：
- zscore
- rank
- winsorize
- demean
- group normalize

可以对：

```text
date × asset × factor_block
```

批量执行，而不是每 factor重复 groupby。

---

### R33-P0-057｜Rolling family需要 multi-window / multi-output kernel

同一 input：
```text
ts_mean(x,5)
ts_mean(x,10)
ts_mean(x,20)
ts_std(x,20)
ts_sum(x,20)
```

应共享：
- rolling state；
- cumulative sums；
- count；
- mean/std intermediates。

---

### R33-P0-058｜Fundamental PIT Join应该整批一次

如果100个因子都需要：
```text
revenue
assets
cashflow
```

不要每因子触发独立 asof join。

DA计划层：
```text
join once → relation/buffer reused
```

---

### R33-P0-059｜Minute→Daily aggregation要“一次 scan 多输出”

DA已有 Aggregation bundle基础。

继续扩：
- VWAP；
- volume profile；
- realized vol；
- imbalance；
- intraday trend；
- open/close windows；
多个 daily aggregates一次 minute scan。

---

### R33-P0-060｜当前 R27 throughput evidence不能证明 fast path真的快

当前仓库报告的轻量 synthetic：
- serial约 0.183s；
- parallel约 0.862s；
- fast约 0.997s。

而报告 HEAD也不是当前 HEAD。

结论：

```text
presence gate != throughput proof
```

必须建立当前 HEAD真实 benchmark，
并增加“小任务自动不并行”的成本门。


## 四、最终目标架构：Unified Factor Query Graph（UFQG）

建议新增一个真正统一的批量执行 IR：

```text
UnifiedFactorQueryGraph
```

它不是把 DA代码搬进 FE，
而是让两边各管自己擅长的语义：

```text
FE owns:
  factor expression
  operator semantics
  CSE
  factor output identity
  factor-level cost
  backend operator capabilities

DA owns:
  fields → datasets
  exact physical scope
  snapshot
  PIT joins
  universe source
  filters
  source transforms
  minute aggregation
  file/row-group pruning
  scan cost
  governed IO

Unified Physical Planner owns:
  where source processing ends
  where factor compute begins
  which subgraph stays SQL
  which subgraph stays Polars
  where Pandas/NumPy is necessary
  when materialize
  when convert
  when spill
  when write
```

建议逻辑结构：

```python
UnifiedFactorBatchPlan:
    factor_roots
    data_demands
    source_plans
    shared_subgraphs
    execution_scopes
    output_specs

PreparedBatchDataPlan:
    dataset_scopes
    exact_snapshots
    physical_objects
    scan_costs
    joins
    aggregations
    ordering_certificates
    source_capabilities

UnifiedPhysicalDAG:
    SourceScanTask
    SourceJoinTask
    SourceAggregationTask
    NativeSqlStage
    NativePolarsStage
    RollingSharedStage
    CrossSectionBlockStage
    StatefulStage
    PandasKernelStage
    ConversionStage
    SpillStage
    MultiRootOutputStage
    WriteBlockTask
    CommitGenerationTask
```


## 五、不要再让 FE BatchDataRequest 与 DA DataRequest 平行发展

建议把 FE 侧改成：

```python
@dataclass(frozen=True)
class FactorDataDemand:
    factor_ids: tuple[str, ...]
    logical_fields: tuple[str, ...]
    source_refs: tuple[SourceRefDemand, ...]
    start: Timestamp
    end: Timestamp
    history: HistoryContract
    universe: UniverseDemand
    pit: PitDemand
    frequency: Frequency
```

然后：

```python
def compile_factor_demands_to_da_requests(
    demands: Sequence[FactorDataDemand]
) -> list[DataRequest]:
    ...
```

聚类 key：

```text
dataset set
snapshot policy
PIT/join semantics
universe
time range compatibility
frequency
source params
filters
normalization semantics
```

兼容的因子合成一个 DataRequest；
不兼容的拆几个。

不要追求“整个10000因子永远一个 DA request”。
目标是：

```text
one/few optimal requests
```


## 六、PreparedBatchReadSession：FE 与 DA 融合的关键接口

建议 DA 增加：

```python
PreparedBatchReadSession
```

生命周期：

```text
1. compile requests
2. authorize datasets once
3. compile contracts once
4. resolve fields once
5. pin snapshots once
6. resolve exact objects once
7. schema epoch check once
8. build scan costs once
9. expose executable source stages
10. scheduler JIT acquire lease
11. execute waves
12. terminal revalidate
13. close/release
```

注意：

```text
prepare != reserve
```

应该：

```python
PreparedDatasetScan
    immutable plan
    no resource lease

execute_scan(prepared, lease)
```

这样 scheduler能按 ready状态获取资源。

### Batch session应暴露

```python
session.dataset_scope(dataset)
session.open_duckdb_relation(scope)
session.open_polars_lazy(scope)
session.scan_arrow_batches(scope)
session.snapshot_id(dataset)
session.ordering_certificate(dataset)
session.scan_cost(dataset)
session.close()
```

安全机制仍存在，
只是从“每个读调用一次”提升到“每个 query-scope一次”。


## 七、SourceScanTask 必须成为真正 executable task

给 `PhysicalFactorTask` 增加：

```python
@dataclass(frozen=True)
class SourceScanContract:
    dataset: str
    columns: tuple[str, ...]
    time_range: tuple[Timestamp, Timestamp] | None
    instrument_scope: InstrumentScope
    universe_id: str | None
    filters_digest: str
    source_params_digest: str
    snapshot_id: str
    expected_rows: int
    expected_bytes: int
    projected_bytes: int
    remote: bool
    ordering: OrderingCertificate | None
    output_representation: str
```

执行：

```python
BufferRef = execute_source_scan(contract, batch_session, lease)
```

而不是：

```text
SOURCE_SCAN → None
```

SourceScan完成后：
- output BufferRef写入 task result table；
- consumer收到 ref；
-最后一个 consumer完成释放/evict。


## 八、BufferRef：不要再让 task之间传巨大 DataFrame

新增统一数据引用：

```python
class BufferRef:
    representation
    schema
    grain
    rows
    bytes
    source_snapshot
    ordering
    partitioning
    location
    ownership
    refcount
```

具体：
```text
ArrowTableRef
ArrowBatchStreamRef
DuckDBRelationRef
PolarsLazyRef
PolarsFrameRef
PandasPanelRef
NumpyBlockRef
SpillFileRef
```

原则：
- task间传 ref，不复制大对象；
-真正 materialize由 consumer要求；
-同一 buffer可被多个 root消费；
-转换产生新的 BufferRef；
- refcount归零释放；
- spilled buffer由 lease管理生命周期。


## 九、ReadWave 2.0：真正执行，并做异步双缓冲

ReadWave最终流程：

```text
Wave N:
  DA async scan
  ↓
  BufferRef
  ↓
  compute consumers

同时：

Wave N+1:
  prefetch / remote range read
```

形成：

```text
IO(N+1) || CPU(N) || WRITE(N-1)
```

即三段流水线。

### wave clustering目标函数

不要只看列数：

```text
benefit =
  saved_scan_bytes
+ saved_decompression
+ saved_snapshot_prepare
+ saved_join
+ saved_conversion
- extra_resident_memory
- extra_unneeded_rows
- blocking_on_long_history
```

### wave memory

必须按：

```text
union physical projection
```

不是每个 request memory求和。

### wave分裂维度

可按：
- source；
- history band；
- time partition；
- instrument shard；
- remote/local；
- expected bytes；
- backend consumer；
- ordering need。

### 自动 wave size

根据实际观测：
```text
estimated vs actual bytes
scan throughput
decompression throughput
consumer latency
sink pressure
```

动态修正下一 wave。


## 十、History-aware Batch Clustering：避免一个 full-history 因子拖全批

对每 factor提取：

```text
required_history_start
lookback_bars
recursive_state_start
full_history_required
incremental_forward_impact
```

聚类：

```text
H0: no/very short history
H1: short
H2: medium
H3: long
H4: full-history
```

阈值不要固定写死，
根据 source partition和成本自适应。

例：

```text
1000 factors:
  800 need 20d
  150 need 120d
  40 need 500d
  10 need full history
```

不要全读 full history。

可执行：

```text
latest short wave
medium extension wave
long extension wave
full-history lane
```

较长 wave可以复用较短 wave重叠部分，
或者直接按 partition cache复用。


## 十一、DuckDB-first Relational Fast Lane：默认优先把 DA scan + PIT join + FE 多因子表达式留在一个 Query Graph

这一节修订上一版“DuckDB SQL Fast Lane”的表述。

**原则：DuckDB 是本项目的默认第一候选，但不是无条件第一。**

原因来自当前架构，而不是抽象 benchmark：

```text
DataAccess 本身已经大量使用 DuckDB
→ parquet scan / multi-dataset read / join / SQL relation 都天然靠近 DuckDB
→ 如果 FE 的 SQL operator coverage 足够高
→ 最省成本的路径通常是继续留在 DuckDB
```

### 11.1 优先目标：一次 query 直接算一个 FactorBlock

不要：

```text
for factor in 1000 factors:
    backend.execute(one_root)
```

要：

```sql
WITH
base AS (...DataAccess exact snapshot scans...),
joined AS (...PIT/asof/universe joins...),
shared_window AS (...shared expensive windows...),
factor_block AS (
    SELECT
        ts,
        inst,
        expr_001 AS f001,
        expr_002 AS f002,
        ...,
        expr_128 AS f128
    FROM shared_window
)
SELECT * FROM factor_block
```

一个 block 32/64/128/256 roots，自适应决定。

### 11.2 SQL operator 不应只是 root-level all-or-nothing

当前 whole-plan routing要求整棵 root所有 operator 都 SQL supported 才形成 `duckdb_sql` candidate。

这会产生：

```text
99% SQL-native
+ 1 个 Pandas-only operator
= 整 root失去 pure SQL candidate
```

应该做 maximal native subgraph partition：

```text
SourceScan
→ Join
→ SQL-native operators 1..N
→ ONE conversion boundary
→ Specialized kernel
→ output
```

不要整个退回 Pandas。

### 11.3 SourceRef 必须先 lower，不能直接毒死 SQL eligibility

当前 plan router对包含 SourceRef 的 root会显著限制 SQL/Polars candidate。

整改顺序：

```text
Symbolic SourceRef
→ DataAccess resolve
→ SourceDemand / JoinPlan
→ relational source stage
→ 再进行 backend partition
```

backend router看到的应该是：

```text
已经合法解析的 source relation
```

而不是 opaque SourceRef string。

### 11.4 DuckDB direct source scan

尽量：

```text
read_parquet exact object list
+ projection
+ predicate
+ partition pruning
+ PIT joins
```

都在 SQL中。

禁止默认：

```text
DA Arrow full table
→ register to DuckDB
```

除非 source本来已经在 Arrow memory中。

### 11.5 DuckDB direct output

优先顺序：

```text
DuckDB relation
→ Arrow RecordBatch stream
→ block DQ / generation writer
```

或在 DataAccess保证原子 staging的前提下：

```text
DuckDB COPY SELECT → staging parquet partitions
→ validate
→ generation manifest commit
```

不要：

```text
DuckDB → Pandas → Parquet
```

### 11.6 Window coalescing

大量因子共享：
- partition by inst；
- order by ts；
-同 window长度；
-同 rolling statistic family。

SQL compiler应：
-合并 named WINDOW spec；
-复用中间 CTE；
-同一 scan输出多列；
-避免同一个 rolling expression重复生成。

### 11.7 PIT/asof join 尽量作为 source relation的一部分

对：
-财务；
-估值；
-行业；
-指数成分；
-event state；

一个 batch只做一次正确 PIT join，后续 factor block复用。

### 11.8 DuckDB 线程策略

DuckDB query内部本身会并行。

所以不要：

```text
16个 Python futures × 每个 DuckDB 16线程
```

造成 256 runnable threads。

GlobalResourceBroker必须控制：

```text
concurrent_duckdb_queries × duckdb_threads_per_query <= effective CPU budget
```

在大 batch里通常更倾向：

```text
更少、更胖的 multi-root DuckDB query
```

而不是很多小 query并行。

### 11.9 DuckDB tie-break policy

当：
- DuckDB与Polars都是 production-certified；
-成本模型置信区间重叠；
- DataAccess source本身已经是 DuckDB relation/parquet SQL plan；
- DuckDB可以 direct sink；

允许 DuckDB作为 tie-break winner。

但是如果 measured baseline明确证明 Polars更快，则不能强行 DuckDB。

## 十二、Polars Streaming/Lazy Fast Lane：不是“第二名”，而是与 DuckDB 竞争的 native execution lane

Polars的定位需要修订：

```text
不是 DuckDB失败才随便试 Polars
而是：
对 Polars-native 的连续大 subgraph，它是一级候选。
```

### 12.1 使用 governed lazy source

DA应暴露：

```text
PreparedPolarsSource / GovernedLazyFrame
```

它必须携带：
- snapshot；
- exact object scope；
- budget；
- lineage；
- terminal verify hook。

FE不能直接绕过治理调用裸 `scan_polars().collect()`。

### 12.2 一个 LazyFrame算一组 factors

优先：

```python
lf = source
lf = lf.with_columns([
    expr_f001.alias('f001'),
    expr_f002.alias('f002'),
    ...,
])
```

然后：

```text
streaming collect / sink
```

不要每 factor独立 collect。

### 12.3 最大 Polars-native connected component

Physical optimizer需要找：

```text
maximal Polars-native subgraph
```

而不是 Python递归一 operator一 dispatch。

### 12.4 利用 Polars optimizer，而不是和它对抗

保持 lazy直到最后：
- projection pushdown；
- predicate pushdown；
- common subplan elimination；
- join ordering；
- streaming。

过早 materialize 会让这些优化失效。

### 12.5 Polars 不允许 Pandas delegate伪装 native

如果某 `polars` slot内部：

```text
Polars → Pandas → operator → Polars
```

它不是 Polars-native。

成本、capability、telemetry必须明确标记 delegate，不能让 router因为 backend名字选择它。

### 12.6 Polars streaming与内存

大结果优先 streaming engine，避免 wide full in-memory panel。

但必须 benchmark：
- group/join类型；
-窗口支持；
- streaming fallback；
- source cardinality。

### 12.7 Polars direct sink

如果某 factor block全程 Polars：

```text
LazyFrame
→ sink / Arrow batch stream
→ DataAccess generation staging
```

不要为了统一接口先转 Pandas。

### 12.8 DuckDB ↔ Polars 边界

两者都使用 Arrow生态，但“可零拷贝”不能当作永远零成本。

真实记录：
-转换字节；
- rechunk；
- schema cast；
- sort；
- materialization；
- ownership/lifetime。

只有测出来的转换成本才进 cost model。

## 十三、Specialized Kernel / Pandas Reference Lane：不要把所有非 SQL/Polars 算子都按“Pandas DataFrame 算”处理

上一版写成 “Pandas/NumPy Specialized Lane”，本版进一步拆清：

### 13.1 Reference semantics

Pandas实现保留 canonical correctness reference。

### 13.2 Production specialized kernels

对不适合 SQL/Polars 的算子，优先开发：
- NumPy contiguous kernel；
- SciPy；
- Bottleneck；
- Numba（如果实际 benchmark和部署稳定性支持）；
-必要时自定义 Rust/C++ extension。

### 13.3 禁止 row-wise Python hot loop

生产大面板禁止：
- `DataFrame.apply(axis=1)`；
- Python per-row lambda；
- Python per-date group循环；
- Python per-stock循环；

除非 operator本身不可避免且已有性能证据。

### 13.4 Pandas fallback只发生在最小 subgraph

例如：

```text
DuckDB relation
→ 90% factor expressions
→ Arrow/NumPy block
→ one specialized operator
→ output
```

不要：

```text
因为最后一个 operator不支持 SQL
→ 整个 source scan + join + rolling 全回 Pandas
```

### 13.5 小任务例外

很小的数据/很少因子，Pandas/NumPy直接跑可能比建立 native query graph更快。

仍由 Auto Granularity router决定。

## 十四、Batch Representation Planner：横截面与时序需要不同布局

同一数据有两种主要访问：

### 时序算子
适合：
```text
asset-major / inst,ts sorted
```

### 横截面算子
适合：
```text
date-major / ts,inst sorted
```

不能对每个 operator反复 sort/unstack/stack。

PhysicalPlanner应：
1. 识别一批消费者 orientation；
2. 选择主 canonical layout；
3.必要时创建一个共享 transpose/reorder barrier；
4.后续多个 CS factor复用。

记录：

```text
orientation_conversion_bytes
sort_bytes
transpose_count
```

Hard gate：
```text
同一批同一 source无意义重复 sort = 0
```


## 十五、Cross-sectional Factor Block Kernel

对同 timestamp 上几十/几百个 factor：

```text
X: assets × factors
```

批量执行：
- demean；
- zscore；
- rank；
- winsor；
- group demean；
- group zscore；
- neutralization。

不要：

```text
for factor:
    groupby(date)
```

可建立：

```python
CrossSectionBlockStage(
    factor_block,
    group_labels,
    operations,
)
```

### rank
可根据 factor列分块并行，
tie/null语义严格复用 canonical rank spec。

### neutralize
同一日同一 exposure matrix可对多个 factor RHS一起做：
```text
multi-RHS regression
```

这是非常明显的共享机会。


## 十六、Rolling Multi-output Kernel

建立共享 rolling primitive：

```text
rolling state:
  count
  sum
  sumsq
  min/max deque
  cov accumulators
```

可支持多个输出：

```text
mean
sum
std
var
zscore
cov
corr
beta
```

同 input/window不要重复扫描。

### 多窗口

可以：
- cumulative prefix；
- window families；
- deque/rolling state；
按算子选择最合适算法。

### Multi-output技术指标

继续扩：
- MACD line/signal/hist；
- DMI+/DMI-/DX/ADX；
- Bollinger mean/std/up/down；
- regression beta/intercept/resid/r2/tstat；
- Keltner等。

Physical node输出：
```text
OutputRef["beta"]
OutputRef["resid"]
...
```


## 十七、Fundamental / Event / Universe Join Cache

这些 source join常比算子本身贵。

建立 batch query级：

```text
JoinedSourceBuffer
```

identity：
- source snapshots；
- PIT policy；
- knowledge-time；
- universe；
- date range；
- projected fields。

同批100个 fundamental factors：
```text
ASOF join一次
```

不同 factor只在 join后的 relation上继续运算。

禁止将 PIT joined table做跨 snapshot不安全全局缓存；
可以做 query/session cache或带完整 identity的持久缓存。


## 十八、分钟数据→日频：一次扫描尽量把一整天能用的统计都算完

分钟数据最贵的是：
- 扫描；
-解压；
-排序；
- group；
而不是最后一个公式。

建立 `IntradayDailyFeatureBundle`：

同一次 minute scan输出：
```text
VWAP
TWAP
open/close window returns
high-low
realized vol
bipower
volume concentration
turnover
imbalance
price impact
intraday trend
jump metrics
session features
```

这些 daily primitives再供大量 factor组合。

冷启动/挖掘算法应优先复用 daily bundle，
而不是每条公式都直接反复扫 minute parquet。


## 十九、Cache 重新分层：缓存真正昂贵且可复用的东西

建议：

```text
L0 Parse/AST
L1 TypedIR/LogicalPlan
L2 DataAccess PreparedDatasetScope
L3 SourceScan Buffer
L4 PIT/Join/Aggregate Buffer
L5 Expensive Shared Operator
L6 Final Factor Result（可选）
L7 Materialized Factor Lake
```

价值函数：

```text
cache_value =
  expected_reuse
  × recompute_cost
  / bytes
```

不要因为节点结构重复就全部 cache。

### Representation cache

同一 semantic buffer：
```text
Arrow
Polars
Pandas
```

必须知道它们是同一个 logical value的不同 representation，
并避免三份常驻。

### Global LRU

修复 column/panel两个 cache共享 bytes、却局部 eviction的问题。


## 二十、Snapshot 与 Manifest Fast Path

一批因子必须有：

```text
QuerySnapshotGeneration
```

开始时：
- 每 dataset拿 manifest token；
- pin exact scope；
-记录 generation。

执行期间：
-不在每个 load_column重复 refresh；
-所有 source stage消费同 token。

结束前：
- production terminal revalidate；
-如果 changed → batch fail/retry/replan；
-不发布混合 snapshot结果。

无 manifest数据集：
-成本更高；
-建议优先为高频使用 source建立 manifest；
-避免每 wave describe全 dataset。


## 二十一、FE + DA 统一资源治理

最终只保留一个 host-level capacity authority。

建议：

```text
HostResourceBroker
  ├─ CPU tokens
  ├─ Memory commitments
  ├─ Local IO
  ├─ Remote IO
  ├─ DuckDB threads
  ├─ Polars threads
  ├─ BLAS threads
  ├─ Writer bandwidth
  └─ Spill capacity
```

DA不再重复独立算同一内存预算；
DA获得：

```text
ChildResourceLease
```

### Source scan资源
看：
- scan bytes；
- decompression；
- projected bytes；
- remote requests。

### Compute资源
看：
- CPU；
- output bytes；
- backend threads。

### Write资源
看：
- queue bytes；
- compression CPU；
- disk bandwidth；
- metadata commit。


## 二十二、Scheduler 2.0：调度真实工作，而不是调度规划占位符

ready queue中的 task必须是真实 work unit。

### Priority

```text
priority =
  critical_path_gain
+ reuse_unlock_gain
+ data_locality_gain
+ downstream_fanout
- marginal_memory
- sink_pressure_penalty
```

### Auto Granularity

对于特别便宜的 nodes/roots：
```text
不要拆 task
```

合成 stage。

### Small-batch bypass

若估计：

```text
scheduler overhead > compute savings
```

自动：
```text
serial fused
```

当前轻量 evidence已经证明这一点很重要。

### No 50ms poll

Future completion / lease release / writer queue事件驱动。

### Dynamic concurrency

显式：

```text
min(max_concurrency, broker_limit, sink_limit)
```

### Work stealing

只有真实 executable shards才启用。


## 二十三、Native Fusion 修复

当前立即改：

1. 删除 batch-global `can_fuse_roots(all_roots)` gate；
2. 先按 `(backend, source_scope, execution_scope, snapshot)` 分组；
3.每组独立 `can_fuse`；
4.每组独立计算 fusion block；
5. scheduler传真实 backend capability map；
6. certified backend失败必须记录原因；
7. repeated failure自动暂时熔断该 native fusion capability；
8.成功 group直接输出 FactorBlock，不拆成 N个Series后再拼。

### Fusion cost

考虑：
- compile time；
- query size；
- shared windows；
- output width；
- memory；
- sink block width。

不能固定只按 roots count。


## 二十四、Streaming Sink 2.0

### 24.1 deque

替代 list pop(0)。

### 24.2 Partition-aware queues

例如：
```text
partition/day hash
```

路由 writer。

### 24.3 BatchBlock

writer输入：
```python
FactorBlock(
    factor_ids,
    ts,
    inst,
    values,
    metadata,
)
```

而不是单 factor item。

### 24.4 Error propagation

writer fatal：
```text
scheduler stop admitting
cancel remaining
abort generation
surface original error
```

### 24.5 Retry taxonomy

只 retry：
- transient IO；
- temporary lock；
- remote retryable。

不 retry：
- schema；
- invalid factor；
- disk full；
- deterministic write bug。

### 24.6 Backpressure

scheduler读取：
```text
queue bytes ratio
writer throughput
oldest item age
```

动态减少：
- compute roots；
- read prefetch；
- fusion output width。

### 24.7 finish

必须证明：
```text
accepted == committed + failed
queue empty
writers joined
fatal_error none
```


## 二十五、Factor Matrix Block Writer：大量因子落值的主路径

如果下游训练/回测经常同时读取很多因子，
建议主产物不只是：

```text
factor_id/year=data.parquet
```

增加经过 benchmark验证的：

```text
factor_matrix
```

布局：

```text
date
asset
f001
f002
...
f064
```

或者 Arrow fixed block。

### block width

32/64/128/256不要固定，
按：
- memory；
- compression；
- query pattern；
- Parquet column pruning；
benchmark选择。

### metadata

block manifest记录：
```text
factor_id -> column
factor_version
snapshot
schema
```

### 单因子读取

DataAccess通过 projection仍可只读某一 factor列。

### 增量

每天新增 date partitions，
避免每个 factor单独开文件。


## 二十六、Generation Pointer 改成 Manifest-level Copy-on-Write

目标：

```text
Generation g1:
  2025 -> objA
  2026-01 -> objB
  2026-02 -> objC

Generation g2:
  2025 -> objA      # reference reuse
  2026-01 -> objB   # reference reuse
  2026-02 -> objD   # only changed
```

commit只：
```text
write objD
write g2 manifest
flip current pointer
```

不复制 objA/B。

### local filesystem

可用：
- immutable shared object directory；
- hardlink（谨慎跨FS）；
- manifest references。

### object store

天然按 key复用。

### GC

引用计数/mark-sweep：
只删不被任何 retained generation引用的 object。


## 二十七、DA Generation Writer 去 Pandas 化

以下 hot path应尽量不再：
```text
Arrow → Pandas → groupby → Arrow
```

优先候选：

### DuckDB
```sql
COPY (...) TO ... PARTITION_BY (...)
```

### PyArrow
```python
dataset.write_dataset(...)
```

### Polars
按 partition streaming/sink。

Multi-key upsert：
```text
anti join old vs new
UNION ALL new
```

用 DuckDB/Polars完成。

所有实现必须做：
```text
old reference vs new backend parity
```


## 二十八、编译阶段也要 batch 化

10k LLM因子可能在读数据前就花很多时间。

### Expression Intern Pool

相同字符串/token/AST子树共享。

### parse_many

批量 parse。

### analyze_many

一次加载 registry/operator/field catalogs。

### typed IR cache

key：
```text
canonical formula + dialect + catalog dependency digest
```

### lower_many

批量生成 DAG。

### dependency extraction一次

不要：
```text
analysis
DAG walk
BatchDataRequest walk
ReadWave walk
source columns walk
```

各走一遍整棵树。

生成一个：

```text
PlanDependencyManifest
```

所有后续复用。


## 二十九、PlanDependencyManifest：一个依赖清单喂所有子系统

建议编译时一次生成：

```python
PlanDependencyManifest:
    operators
    logical_fields
    physical_fields_by_dataset
    source_refs
    lookback_by_source
    required_ordering
    pit_requirements
    universe_requirements
    group_requirements
    backend_capabilities
    output_grain
    cost_features
```

后续：

```text
identity
BatchDataRequest
DataAccess plan
read wave
cost router
cache
materializer
mining grammar
lineage
```

都消费同一 manifest。

这样既快，又避免“各模块自己 walk 得到不同结论”。


## 三十、DataAccess Field Resolution 真正 O(N)

为 SemanticFieldCatalog维护：

```text
logical_name index
alias index
(dataset, physical_name) index
physical_name -> datasets index
market index
```

提供：

```python
resolve_many(names, dataset=None, market=None)
```

一次返回：
- resolved；
- clean_miss；
- ambiguous；
- invalid。

不要 batch API内部又循环做昂贵全 registry搜索。


## 三十一、Unified Batch Cost Model：不要用静态 DuckDB > Polars > Pandas；要用“DuckDB tie-break + 全链路实测成本”

当前 `choose_plan_route()` 已经有一个正确的基础思想：

```text
从多个 eligible backend candidate 中选择 estimated cost 最小者
```

这个思想**应该保留**，不要改成简单：

```python
if duckdb_supported:
    use_duckdb()
elif polars_supported:
    use_polars()
else:
    use_pandas()
```

但现有 cost router仍主要是：

```text
single root
operator occurrences
+ heuristic conversion
+ heuristic memory
```

R33要升级成：

```text
batch-global physical plan cost
```

### 31.1 总成本

```text
TotalCost =
    CompileCost
  + SourceResolveCost
  + ScanCost
  + DecodeCost
  + FilterCost
  + JoinCost
  + AggregateCost
  + SortCost
  + OperatorCost
  + ConversionCost
  + MaterializationCost
  + SchedulerOverhead
  + DQCost
  + WriteCost
  + GenerationCommitCost
```

最终优化：

```text
TimeToDurableCommit
```

### 31.2 加入 Shared Benefit

对于一组 roots：

```text
SharedBenefit =
  saved source scans
+ saved joins
+ saved windows
+ saved groupby
+ saved conversions
+ saved writer opens
```

单 root看来 DuckDB 可能略慢；
100个 roots fusion后 DuckDB可能明显更快。

### 31.3 Source affinity

如果 source已经是：

```text
DuckDB relation
```

切到 Polars必须支付真实 boundary cost。

如果 source已经是：

```text
Polars LazyFrame
```

反之同理。

### 31.4 Writer affinity

如果 DuckDB可 direct COPY / batch Arrow stream，
或者 Polars可 streaming sink，
写端收益必须进入 route。

### 31.5 Conversion penalty不能写死常数

当前类似：

```text
DuckDB 5.0 + coeff
Polars 2.0 + coeff
```

只能作为 seed fallback。

真正 production model应该测：

```text
bytes / throughput + fixed overhead
```

并按：
- Arrow→Polars；
- Arrow→Pandas；
- DuckDB→Arrow；
- wide↔long；
- sort；
分别建模。

### 31.6 Memory model也必须来自真实 shape

不能只：

```text
rows × 8 × generic factor
```

需要：
-实际 projection width；
-dtype；
-null bitmap；
-string/dictionary；
-window state；
-output block width；
-concurrent buffers。

### 31.7 Cost uncertainty

每个 estimate带：

```text
mean
p50/p90
confidence
sample_count
hardware_fingerprint
source_fingerprint
```

当两个 backend差异小于 uncertainty：

```text
优先 stay-in-engine
```

若 DataAccess source-native 是 DuckDB，则 DuckDB tie-break。

### 31.8 Online calibration

每次真实 batch把：

```text
predicted vs actual
```

回写 runtime calibration store。

随着运行次数增加，router越来越接近你的真实服务器/真实 A股数据，而不是通用 benchmark。

## 三十二、Representation Conversion 必须进入一等优化目标

记录每一条 edge：

```text
Arrow → Pandas
Arrow → Polars
Polars → Pandas
DuckDB → Arrow
Long → Wide
Wide → Long
Sort
Transpose
Copy
```

指标：
```text
conversion_count
conversion_bytes
conversion_ms
sort_bytes
transpose_bytes
```

优化目标：
```text
minimize total conversion bytes
```

而不是只减少 backend transition count。


## 三十三、Output DQ 也做 block 化

不要每 factor做一遍所有 Python级检查。

可 block计算：
- null ratio；
- finite ratio；
- cross-sectional coverage；
- variance；
- constant；
- extreme ratio。

高阶 factor-specific DQ再单独执行。

DQ输出：
```text
vectorized metrics per factor
```

这样可以：
```text
FactorBlock → DQ block → writer
```
不拆 Series。


## 三十四、Input DQ 与 DataAccess 统计融合

如果 DataAccess manifest已有：
- null stats；
- min/max；
- row counts；
- coverage；

优先用 metadata做 preflight。

只有需要真实值检查时才加载。

避免：
```text
input DQ先扫一遍
正式 compute再扫一遍
```

DataAccess source scan应同时产生：
```text
data buffer + DQ stats
```


## 三十五、远程/对象存储 IO 优化

如果数据在 COS/S3：

### 35.1 row-group pruning
确保 filters/time/universe真正下推。

### 35.2 range request coalescing
避免大量小 range。

### 35.3 bounded async prefetch
与 compute overlap。

### 35.4 NVMe local cache
key：
```text
object etag/version + rowgroup + projection
```

### 35.5 remote/local独立 IO tokens

### 35.6 compression awareness
有时瓶颈是解压CPU而非网络，
ScanCost要区分：
```text
compressed bytes
decoded bytes
```


## 三十六、Parquet / 数据布局

DataAccess应按真实 workload benchmark：

### Daily price/fundamental
可能：
```text
date partition
asset/time sort
```

### Minute
已有 route_minute_storage基础，
继续基于：
- universe width；
- date range；
- typical factor scan；
选择 date-major / bucket-major。

### Factor matrix
通常：
```text
date partition
asset sorted
factor columns
```

### row group
调：
- row group size；
- compression codec；
- dictionary；
- statistics。

目标不是最小文件，
是：
```text
scan prune + decode + write总时间最优。
```


## 三十七、避免小文件

1000因子 × 每日 × 单 factor文件会导致：
- open/close；
- metadata；
- filesystem；
- remote object count；
成为主瓶颈。

策略：
- factor blocks；
- partition compaction；
- minimum target file size；
- writer batching；
- generation manifest。

同时保留：
```text
单 factor column projection
```
能力。


## 三十八、Incremental 与批量计算真正融合

每天新数据时不要：

```text
1000 factors × 各自计算自己的 window
```

先做：

```text
changed source partitions
→ ChangeImpactDAG
→ affected factor subgraphs
→ group by required history
→ read waves
→ compute blocks
→ COW generation
```

递归 stateful：
- checkpoint；
- replay range。

rolling N：
- source变更T影响[T, T+N-1]。

source revision：
只重写 affected output partitions。


## 三十九、Auto Execution Mode：小任务不并行，大任务不逐 root，按 workload 自动选择粒度

当前 synthetic evidence已经说明：

```text
更复杂的 scheduler并不自动等于更快。
```

建立：

```python
ExecutionMode =
    DIRECT_VECTOR
    DUCKDB_FUSED
    POLARS_FUSED
    SPECIALIZED_KERNEL_BLOCK
    ADAPTIVE_DAG
```

### DIRECT_VECTOR

极少量简单 factors：
-不建大量 Future；
-不拆虚拟 stage；
-一次 vector block。

### DUCKDB_FUSED

大量 relation-native factors：
-少数 fat query；
-multi-root；
- direct sink。

### POLARS_FUSED

大量 lazy-native expressions：
-single/multiple controlled LazyFrame block；
-streaming。

### SPECIALIZED_KERNEL_BLOCK

同类特殊 operator成块执行。

### ADAPTIVE_DAG

真正存在：
-多 source；
- barrier；
-不同 history；
-复杂资源约束；
-多个 backend partition；
才启用完整 scheduler。

Router依据：

```text
factor count
IR node count
source count
history classes
estimated scan bytes
shared subgraph ratio
native coverage
expected compile cost
expected scheduling cost
output bytes
```

Hard Gate：

```text
AUTO不得在 representative workload上持续显著慢于最佳可选模式。
```

## 四十、不要急着做多机分布式

在单机内以下问题没有收口前，不建议先上 Ray/Dask：

```text
重复扫描
重复物化
FE/DA双规划
虚拟 physical stages
Arrow→Pandas
临时 Parquet
per-factor写
read wave没执行
```

这些修好后，
如果单机 CPU/IO已经线性吃满，
再考虑 multi-node。

否则分布式只会把重复工作搬到更多机器。


## 四十一、建议新增/修改的核心文件

建议但不强制路径：

### FactorEngine

```text
factor_engine/planner/unified_batch_plan.py
factor_engine/planner/plan_dependency_manifest.py
factor_engine/planner/dataaccess_bridge.py
factor_engine/planner/physical_lowerer.py
factor_engine/planner/read_wave_planner.py
factor_engine/planner/representation_planner.py
factor_engine/planner/batch_cost_optimizer.py

factor_engine/runtime/batch_read_session.py
factor_engine/runtime/adaptive_batch_scheduler.py
factor_engine/runtime/stage_executor.py
factor_engine/runtime/buffer_ref.py
factor_engine/runtime/streaming_result_sink.py
factor_engine/runtime/block_dq.py

factor_engine/backend/sql_multi_root.py
factor_engine/backend/polars_multi_root.py
factor_engine/backend/cross_section_block.py
factor_engine/backend/rolling_multi_output.py
```

### DataAccess

```text
dataaccess/runtime/prepared_batch_read.py
dataaccess/runtime/read_pipeline.py
dataaccess/read/data_request.py
dataaccess/read/physical_plan.py
dataaccess/read/semantic_catalog.py
dataaccess/read/relation_handle.py
dataaccess/write/generation.py
dataaccess/write/factor_block_writer.py
```

不要为了满足文件名机械拆模块，
核心是保持 single authority。


## 四十二、建议的新接口草图

```python
# FE
batch = engine.compile_batch(factors)

# FE -> DA
demands = batch.data_demands()
da_plan = store.plan_batch(demands.to_data_requests())

# query-scoped
with store.open_batch_session(da_plan) as session:
    physical = engine.physical_plan(
        batch,
        data_plan=da_plan,
        runtime_capabilities=session.capabilities,
    )

    engine.execute_batch(
        physical,
        data_session=session,
        sink=FactorGenerationSink(...),
    )
```

Source stage：

```python
ref = session.execute_scan(
    source_scan_spec,
    lease=child_lease,
)
```

SQL stage：

```python
ref2 = session.extend_relation(
    ref,
    sql_expressions=stage.expressions,
)
```

Polars stage：

```python
ref3 = session.extend_lazy(
    ref,
    polars_exprs=stage.exprs,
)
```

Write：

```python
sink.submit_block(
    FactorBlockRef(...)
)
```


## 四十三、Benchmark 体系必须重做

当前 benchmark不能只用：
```text
20日 × 2 instrument × 20 factors
```

至少建立：

### B1 Small cheap
```text
20 / 100 simple factors
daily
```
目标：验证 scheduler overhead router。

### B2 1000 daily price-volume
```text
A股全市场
5年
1000 factors
```

### B3 Mixed history
```text
1000 factors
20d / 120d / 500d / full mix
```

### B4 Fundamental PIT
```text
500 factors
price + income + balance + cashflow + valuation
```

### B5 Minute→daily
```text
A股分钟
100/500 daily outputs
```

### B6 Mixed backend
```text
70% native SQL/Polars
20% CS/group
10% specialized Pandas/stateful
```

### B7 Incremental one day
```text
1000 factors
latest market day
```

### B8 Historical correction
```text
source revision at T
change impact recompute
```

### B9 Materialization
```text
1000 outputs
single-factor vs matrix-block
```

### B10 Remote source
```text
COS/S3
rowgroup prune / cache / prefetch
```


## 四十四、Benchmark 每次必须采集的 KPI

### Compile
```text
factors/sec
parse ms
analyze ms
lower ms
DA plan ms
```

### IO
```text
physical objects opened
row groups scanned
compressed bytes
decoded bytes
projected bytes
scan wall
remote requests
```

### Reuse
```text
scan reuse bytes
join reuse count
source buffer consumers
CSE saved work
rolling shared work
```

### Conversion
```text
conversion count
conversion bytes
conversion ms
sort bytes
transpose bytes
```

### Compute
```text
CPU utilization
wall
backend time
root throughput
stage throughput
```

### Scheduler
```text
task count
real task count
virtual task count
queue wait
admission wait
scheduler overhead
```

### Memory
```text
peak RSS
source buffers
cache
CSE
result queue
spill
```

### Write
```text
write MB/s
files created
avg file size
writer queue
backpressure time
catalog commit time
generation commit time
```

### End-to-end
```text
factor outputs/sec
cells/sec
total wall
time-to-first-factor
time-to-last-commit
```


## 四十五、性能正确性 Differential Gates

任何加速路径都必须与 reference比：

### Read Wave
```text
wave execution == full union reference
```

### SQL Fusion
```text
fused SQL == per-root reference
```

### Polars Fusion
```text
one lazy graph == reference
```

### Matrix CS
```text
block rank/zscore == canonical per-factor
```

### Rolling Multi-output
```text
shared kernel == canonical operator
```

### Warmup clustering
```text
clustered windows == individual correct warmup
```

### Sink
```text
streaming/block materialized reload == live results
```

### Generation COW
```text
new generation logical dataset == full rewrite generation
```

### Different wave sizes
结果一致。

### Different concurrency
结果一致。

所有 mismatch先修 correctness，不能通过 tolerance无脑扩大解决。


## 四十六、性能 Hard Gates

建议建立：

```text
R33_BATCHDATAREQUEST_MULTI_SOURCE_REAL
R33_DA_DATAREQUEST_IS_SOURCE_PLANNING_AUTHORITY
R33_SOURCE_SCOPE_TYPED_ZERO_STRING_PARSE

R33_SOURCE_SCAN_REQUIRED_COLUMNS_REAL
R33_SOURCE_SCAN_TASK_EXECUTES
R33_VIRTUAL_STAGE_RESOURCE_RESERVATION_ZERO
R33_READ_WAVE_COLUMNS_ARE_PHYSICAL
R33_READ_WAVE_TIME_RANGE_REAL
R33_READ_WAVE_EXECUTED_IN_MAIN_PATH
R33_FULL_UNION_PREFETCH_MAIN_PATH_ZERO

R33_SCAN_COST_SCOPE_MATCH_100PCT
R33_SCAN_COST_HAS_WINDOW_AND_UNIVERSE
R33_SCAN_COST_SILENT_FAILURE_ZERO

R33_PREPARED_BATCH_SESSION_PRESENT
R33_SNAPSHOT_RESOLVED_ONCE_PER_DATASET_SCOPE
R33_SCHEMA_GATE_ONCE_PER_DATASET_SCOPE
R33_FIELD_RESOLVE_MANY_LINEAR
R33_RESOURCE_DOUBLE_ADMISSION_ZERO

R33_ARROW_TO_PANDAS_BEFORE_REQUIRED_BOUNDARY_ZERO
R33_COLUMN_PANEL_DUPLICATE_CACHE_ZERO
R33_GLOBAL_REPRESENTATION_CACHE_BUDGET_RESPECTED
R33_REDUNDANT_SORT_ZERO

R33_DA_AGG_JOIN_TEMP_PARQUET_ZERO
R33_SQL_RELATION_REUSES_PREPARED_SCOPE
R33_SQL_MULTI_ROOT_CERTIFIED
R33_POLARS_MULTI_ROOT_CERTIFIED

R33_NATIVE_FUSION_MIXED_SCOPE_PARTITIONS_CORRECT
R33_NATIVE_FUSION_BLOCK_PER_GROUP
R33_UNCERTIFIED_FUSION_PLANNED_ZERO

R33_SCHEDULER_REAL_TASK_RATIO_REPORTED
R33_MAX_CONCURRENCY_ENFORCED
R33_FIXED_50MS_POLL_ZERO
R33_CSE_RELEASE_SILENT_FAILURE_ZERO

R33_SINK_BACKPRESSURE_FEEDS_ADMISSION
R33_WRITER_FATAL_PROPAGATES
R33_WRITER_FINISH_DURABLE
R33_RESULT_QUEUE_O1_POP
R33_PARTITION_WRITER_RACE_ZERO

R33_FACTOR_BLOCK_OUTPUT_SUPPORTED
R33_BLOCK_DQ_SUPPORTED
R33_CROSS_SECTION_BLOCK_PARITY
R33_ROLLING_MULTI_OUTPUT_PARITY

R33_GENERATION_HOT_PATH_PANDAS_GROUPBY_ZERO
R33_MULTIKEY_UPSERT_ROW_STRING_AGG_ZERO
R33_GENERATION_COPY_UNCHANGED_PARTITIONS_ZERO
R33_GENERATION_COW_PARITY

R33_SMALL_BATCH_AUTO_OVERHEAD_GATE_PASS
R33_1000_FACTOR_BENCH_CURRENT_SHA
R33_MINUTE_TO_DAILY_BENCH_CURRENT_SHA
R33_FUNDAMENTAL_PIT_BENCH_CURRENT_SHA
R33_INCREMENTAL_BENCH_CURRENT_SHA

R33_BATCH_VS_SINGLE_NUMERIC_PARITY_PASS
R33_WAVE_VS_FULL_PREFETCH_PARITY_PASS
R33_FUSED_VS_REFERENCE_PARITY_PASS
R33_STREAM_MATERIALIZE_RELOAD_PARITY_PASS

R33_HARD_BLOCKERS_ZERO
```


## 四十七、建议的性能验收原则

不要提前伪造“提升 3x/5x”。

正式验收：

### Small cheap workload
AUTO应该选择低开销路径。

要求：
```text
auto wall <= best available reference × 合理小容差
```

重点是不再出现：
```text
fast path比 serial慢数倍
```

### 1000+ factors
必须相对当前 HEAD记录：
```text
wall time
scan bytes
conversion bytes
peak memory
files written
```

不仅看 factors/min。

### IO-heavy
优先看：
```text
scan bytes reduction
physical scans reduction
```

### Write-heavy
优先看：
```text
files count
write amplification
write MB/s
generation commit
```

### Mixed workload
看：
```text
end-to-end time-to-durable-commit
```
这是最终指标。


## 四十八、需要重新审视当前 R27/R31 evidence 的真实性

当前 evidence有部分属于：

```text
“代码里存在某模块/关键字”
```

但不能证明：
```text
主路径真的执行了它。
```

例如需要区分：

```text
ReadWavePlanner present
vs
ReadWave executed

Physical stage present
vs
stage did actual compute

NativeFusionGroup present
vs
multi-root backend actually ran

Bounded sink present
vs
scheduler consumed backpressure

ScanCost bridge present
vs
scope key actually matched and affected admission
```

R33 evidence全部改成：

```text
runtime event + counter + benchmark + differential
```

不要 static presence gate作为最终性能证据。


## 四十九、建议新增 Evidence

输出：

```text
factor_engine/docs/evidence/r33/
```

至少：

```text
R33_HEAD.json
R33_CURRENT_END_TO_END_PATH.md
R33_FE_DA_BOUNDARY_AUDIT.json

R33_BATCH_DATA_DEMAND_GRAPH.json
R33_DA_REQUEST_COALESCING.json
R33_SOURCE_SCOPE_MATCH_AUDIT.json

R33_EXECUTABLE_PHYSICAL_STAGE_AUDIT.json
R33_VIRTUAL_TASK_AUDIT.json

R33_READ_WAVE_RUNTIME_TRACE.json
R33_READ_WAVE_COLUMN_AUDIT.json
R33_READ_WAVE_MEMORY_ACCOUNTING.json
R33_READ_WAVE_REUSE_REPORT.json

R33_PREPARED_BATCH_SESSION_AUDIT.json
R33_SNAPSHOT_CALL_COUNT.json
R33_SCHEMA_GATE_CALL_COUNT.json
R33_FIELD_RESOLUTION_BENCH.json

R33_RESOURCE_DOUBLE_ADMISSION_AUDIT.json
R33_HOST_RESOURCE_TIMELINE.json

R33_REPRESENTATION_TRANSITIONS.csv
R33_CONVERSION_BYTES.json
R33_SORT_TRANSPOSE_AUDIT.json

R33_DUCKDB_MULTIROOT_PARITY.json
R33_POLARS_MULTIROOT_PARITY.json
R33_NATIVE_FUSION_RUNTIME.json

R33_CROSS_SECTION_BLOCK_PARITY.json
R33_ROLLING_MULTI_OUTPUT_PARITY.json
R33_INTRADAY_BUNDLE_PARITY.json

R33_STREAM_SINK_FAILURE_INJECTION.json
R33_STREAM_SINK_BACKPRESSURE.json
R33_FACTOR_BLOCK_WRITE_BENCH.json

R33_GENERATION_WRITE_AMPLIFICATION.json
R33_GENERATION_COW_PARITY.json

R33_SMALL_BATCH_BENCH.csv
R33_100_FACTOR_BENCH.csv
R33_1000_FACTOR_BENCH.csv
R33_10000_FACTOR_COMPILE_BENCH.csv
R33_MIXED_HISTORY_BENCH.csv
R33_FUNDAMENTAL_PIT_BENCH.csv
R33_MINUTE_TO_DAILY_BENCH.csv
R33_INCREMENTAL_ONE_DAY_BENCH.csv
R33_HISTORICAL_REVISION_BENCH.csv
R33_REMOTE_IO_BENCH.csv

R33_END_TO_END_SPEEDUP_REPORT.md
R33_FINAL_ACCEPTANCE_REPORT.md
```

全部绑定：
- current SHA；
- DataAccess SHA/version；
- dependency versions；
- hardware；
- data snapshot；
- universe；
- date range。



## 五十、实施顺序：按收益/风险排序

### Phase 0｜先定正确的后端策略，不做静态 DuckDB > Polars > Pandas

1. 保留现有 cost-based router 思想；
2. 增加 DuckDB source-affinity tie-break；
3. route从 single-root 升级 batch-global；
4. SourceRef先 lower成 DA source relation；
5. 加 scan/join/conversion/write/shared benefit；
6. 建 current hardware measured backend benchmark；
7. 禁止一个 unsupported op 导致整 root Pandas fallback。


### Phase A｜先修当前“规划接了但没执行”的问题

1. SourceScanContract；
2. read-wave真实 columns/time/source；
3. read-wave接 main scheduler；
4.移除 scheduler前 full-union prefetch；
5. ScanCost scope key统一；
6. no-op physical stage不再占资源；
7. stage-local cost。

这一步完成后才有真正的：
```text
scan once → consumers many
```

### Phase B｜FE 与 DA 统一规划

8. FE demand → DA DataRequest；
9. PreparedBatchReadSession；
10. multi-source requests；
11. query snapshot一次；
12. field resolve_many；
13. FE/DA resource lease统一。

### Phase C｜减少物化/转换

14. DuckDB relation fast lane；
15. Polars governed lazy fast lane；
16. DA aggregate+join去 temp parquet；
17. Arrow/Polars long canonical buffer；
18. representation planner；
19. sortedness certificate。

### Phase D｜批量算子

20. native multi-root；
21. CS block；
22. rolling multi-output；
23. fundamental join cache；
24. intraday daily bundle。

### Phase E｜批量落值

25. FactorBlock；
26. block DQ；
27. partition-aware writer；
28. sink backpressure；
29. generation Arrow-native writer；
30. COW generation。

### Phase F｜自动调度

31. small batch bypass；
32. priority queue；
33. event-driven scheduler；
34. adaptive waves；
35. adaptive fusion；
36. dynamic write-aware concurrency。

### Phase G｜Benchmark + Evidence

37. representative workloads；
38. current SHA baseline；
39. differential；
40. regression CI。


## 五十一、AI 修改时的禁止事项

1. 不允许为了 benchmark关 PIT/snapshot/schema检查。
2. 不允许把 DataAccess private path/glob直接暴露给 FE绕治理。
3. 不允许再新增第三套 field resolver。
4. 不允许 FE/DA分别维护同一 source identity规则。
5. 不允许以更多线程替代重复扫描修复。
6. 不允许把 DataFrame pickle给 process worker作为常规数据传输。
7. 不允许 virtual task继续消耗真实 resource lease。
8. 不允许 writer错误静默。
9. 不允许为了少代码继续用 temp parquet作为高频中间IPC。
10. 不允许大 batch先 materialize所有结果再统一写。
11. 不允许 generation增量更新复制多年未变化历史。
12. 不允许把“hard gate=True”建立在 grep/module existence上。
13. 不允许 benchmark只跑2只股票20天。
14. 不允许速度路径结果与 reference不一致后直接扩大 tolerance。
15. 不允许对所有算子强制 SQL/Polars；specialized operator保留最佳专用 lane。


## 五十二、完成后的理想性能路径

典型 1000 A股日频因子：

```text
1000 formulas
↓
batch parse/analyze/intern
↓
dependency manifest
↓
3 compatible DA requests
  - daily market
  - PIT fundamentals
  - universe/industry
↓
pin 3 source snapshots
↓
read waves
  - 20d short
  - 120d medium
  - 500d long
↓
DuckDB:
  scan + filter + PIT join + 200 SQL-native factors
↓
Polars:
  same source buffers + 500 native factors
↓
CS block:
  200 rank/zscore/group factors
↓
Pandas specialized:
  100 difficult/stateful factors
↓
FactorBlocks
↓
Block DQ
↓
partition-aware async writers
↓
COW generation
↓
atomic publish
```

整个过程最重要的性质：

```text
同一物理 source partition尽量只读一次
同一 PIT join尽量只做一次
同一 rolling intermediate尽量只算一次
同一 representation尽量只转换一次
同一批 factor尽量 block输出
同一未变历史 partition绝不重写
```


## 五十三、最终 Definition of Done

R33 完成，不是“新建了 Unified Planner 类”，也不是“把 DuckDB 写成第一优先级”就结束。

最终必须证明 backend policy 对真实 workload 是全链路最优，而不是静态偏好。

必须实际证明：

```text
1. 当前 run_many 主路径真正消费 DataAccess batch plan；
2. read wave真正发生 IO，而不是只 explain；
3. SOURCE_SCAN task有真实 BufferRef输出；
4. physical stages不再是大量 no-op；
5. FE/DA没有重复 source planning；
6. 同一 dataset/snapshot/window没有多次无意义 prepare；
7. full-union prefetch不再和 read wave同时存在；
8. ScanCost确实影响 wave/admission；
9. DuckDB/Polars native batch能一次算多个 root；
10. minute aggregate + join不再写 temp parquet；
11. Arrow→Pandas只发生在确实需要的边界；
12. column/panel双缓存问题关闭；
13. small batch AUTO不会被 scheduler overhead拖慢；
14. 1000-factor真实 workload比当前 HEAD更快；
15. write path不产生海量小文件；
16. generation增量不复制未变化历史；
17. writer失败会使整个 publish失败；
18. PIT/snapshot/schema/numeric parity全部保持；
19. evidence绑定当前 HEAD和真实 data snapshot；
20. 最终指标使用 time-to-durable-commit，而不是只计算到内存结果。
```

最终要达到的状态：

> **FactorEngine 不再是“从 DataAccess 取到 Pandas 后逐因子算”的引擎，而是一个能把成千上万个因子的共同数据依赖和共同计算结构整体编译成查询图、让 DataAccess 与计算后端共同执行、并直接流式写入版本化因子湖的批量因子执行系统。**

这才是后续继续扩大因子搜索空间时真正能形成基础设施优势的方向。


---

## 五十四、当前 `plan_cost_router` 还要继续改的具体点

这是本次修订新增的代码级重点。

### R33-P0-061｜Whole-plan route 粒度仍太粗

当前主要返回：

```text
one PlanRoute per root
```

最终需要：

```text
one BatchPhysicalRoute
with multiple backend regions
```

### R33-P0-062｜SourceRef 在 route 前未完全 relational-lower

包含 SourceRef时，SQL/Polars candidate可能直接受限。

必须在 router前：

```text
SourceRef → DA relation dependency
```

### R33-P0-063｜row estimate仍可能通过 pandas bdate_range + 3000 stocks启发式

这对：
-真实交易日；
-分钟频；
-universe；
-停牌；
-筛选；
不够准确。

优先使用 DA ScanCost的：

```text
estimated_rows
selected_bytes
projection_bytes
```

### R33-P0-064｜DuckDB / Polars conversion penalty仍是固定启发式

如果最终 direct sink，conversion可能接近 0；
如果 full materialize + rechunk，则可能非常大。

必须 edge-level measured。

### R33-P0-065｜Peak memory是后端粗粒度公式

需要 actual physical stage liveness simulation。

### R33-P0-066｜Measured baseline还不够 batch-specific

operator baseline需要加入：
- window；
- rows；
- instruments；
- factor block width；
- backend threads；
- source type；
- output representation。

### R33-P0-067｜Hybrid cost需要加入真实 source / write tasks

当前 mixed backend更关注 operator DAG。

最终：

```text
source + operator + writer
```

统一 cost。

---

## 五十五、Multi-Query Optimization：真正把“1000个因子”当一条 workload，而不是1000条 query

这是 FactorEngine 要形成基础设施优势最重要的算法层之一。

### 55.1 Hash-consing

在 parser / IR阶段就对相同子表达式 intern：

```text
同一逻辑表达式只创建一个 node identity
```

### 55.2 Algebraic canonicalization

只做**语义可证明安全**的重写：
-交换律/结合律的 canonical order；
-常数折叠；
-重复 cast消除；
-重复 fill/clip消除（语义严格相同时）；
-重复 delay/window输入共享。

不能为了 CSE随便把 NaN/Inf语义不同的表达式视为相等。

### 55.3 Common Window Extraction

对：

```text
ts_mean(close,20)
ts_std(close,20)
ts_zscore(close,20)
```

提取共同 rolling state。

### 55.4 Shared Affine/Return Primitive

例如大量因子依赖：

```text
ret1
logret
turnover
vwap
market_neutral_ret
```

这些成为 versioned batch primitives。

### 55.5 Cost-based materialized CSE

不是重复就 materialize。

只当：

```text
(recompute_cost × reuse - materialize_cost) / bytes
```

足够高才缓存。

### 55.6 Cross-factor optimization

允许 optimizer选择：

```text
“先算一个共享 expensive intermediate，再给100个 roots复用”
```

而不是每个 root局部最优。

---

## 五十六、把字符串资产代码移出 hot path：InstrumentId / GroupId 整数化

A股全市场计算中大量成本来自：
-字符串比较；
-hash；
-join key；
-MultiIndex；
-group label。

在 BatchReadSession开始时建立稳定映射：

```text
external instrument id
→ canonical InstrumentId
→ batch dense int32 code
```

计算 hot path使用：

```text
int32 asset_code
int32 group_code
int32 industry_code
Date32 / int64 timestamp
```

最终 writer/lineage再映射回 canonical instrument。

要求：
-映射版本化；
-不可混 snapshot；
-退市/更名不改变历史身份；
-与 R32 SecurityMaster / InstrumentId一致。

收益：
-更小内存；
-更快 sort/join/group；
-更容易 contiguous matrix；
-避免 object dtype。

---

## 五十七、Universe / Industry / Missing Mask 共享编码

同一 batch通常共享：

```text
universe mask
industry group
ST/suspension mask
limit-up/down mask
validity mask
```

这些不要每 factor重新构建。

使用：
- bitset / boolean block；
- categorical integer codes；
- per-date reusable mask；

由 source stage一次生成。

CrossSectionBlock直接消费 mask/group code。

---

## 五十八、Sort / Ordering 是一等资产：能不排序就绝不重复排序

建立：

```python
OrderingCertificate(
    keys=('inst','ts'),
    ascending=True,
    stable=True,
    snapshot_id=...,
)
```

每个 BufferRef携带 ordering。

operator声明：

```text
required_ordering
preserves_ordering
changes_partitioning
```

optimizer只有必要时插 SortStage。

对于 parquet serving layer，尽可能物理写成常用排序。

Hard Gate：

```text
同一个 source/snapshot/order 在同一 batch重复全量 sort == 0
```

---

## 五十九、Join 也要做 Multi-Consumer Optimization

PIT fundamental join通常比简单 factor贵得多。

建立：

```text
PITAlignedRelation
```

一次对齐：
- filing availability；
- period selection；
- revisions；
- industry；
- universe；

然后多个 factor block消费。

如果日常挖掘长期重复同一历史窗口，可以建设 versioned serving layer：

```text
pit_daily_serving
```

但必须：
- snapshot/version绑定；
- lineage；
-不存在未来数据；
- source revision可触发增量 rebuild。

---

## 六十、Factor Mining Campaign Session：针对 AlphaProbe / AlphaMiner 等连续候选挖掘做长会话复用

你的真实 workload不是只执行一次1000 factors，还可能是：

```text
LLM / symbolic mining
→ 一轮100候选
→ 评估
→ 下一轮100候选
→ ...
```

如果每轮都重新：
- resolve DA；
- pin snapshot；
- scan历史；
- build joins；
- compile common primitives；
会浪费大量时间。

新增：

```python
FactorCampaignSession
```

在固定：
- market；
- universe；
- date range；
- data snapshot；
下复用：

```text
PreparedBatchReadSession
source relations
expensive PIT joins
intraday daily bundle
shared primitive cache
IR intern pool
measured cost calibration
```

新一轮候选只：

```text
compile delta
→ reuse existing DAG nodes
→ compute new roots
```

一旦 source snapshot变更，session失效/重建。

这是自动化因子挖掘场景极高价值的优化。

---

## 六十一、Incremental Formula Compilation：新增因子不要重编整个大库

维护：

```text
FormulaRegistryDigest
CompiledIRCache
DependencyManifestCache
```

当新增20个 factors：

```text
只 parse/analyze/lower 新的20个
```

并把其 subtree与已有 intern pool合并。

删除/修改factor只影响对应依赖。

---

## 六十二、Native Operator Coverage 的优化目标：不是“算子越多越好”，而是“高频 workload native coverage越高越好”

建立 operator usage telemetry：

```text
usage_count
runtime_share
fallback_share
conversion_cost_caused
fanout
```

优先补 SQL/Polars实现的算子：

```text
high usage × high cost × high fallback impact
```

而不是按 operator catalogue顺序盲目全补。

每周生成：

```text
TOP_NATIVE_COVERAGE_GAPS
```

---

## 六十三、Operator Macro Lowering：高层算子尽量展开成已认证 primitives

很多 operator本质上可写成：

```text
primitive expression graph
```

例如：
-比例；
-zscore；
-简单 technical indicators；
-条件强度；

如果直接作为 opaque Python operator，会阻断 SQL/Polars fusion。

增加：

```text
OperatorLoweringRecipe
```

在语义完全一致时：

```text
high-level canonical
→ primitive IR
```

这样 backend只需要认证 primitive，自动扩大 native coverage。

要求 differential parity严格。

---

## 六十四、避免 Python UDF：这是 DuckDB/Polars 快路径的硬约束

如果把自定义 Python function塞进：
- DuckDB Python UDF；
- Polars map_elements / Python callback；

往往会破坏 native vectorization和 optimizer。

生产 policy：

```text
Python UDF in native fast lane = 默认禁止
```

除非：
- benchmark证明仍有收益；
- semantics不可替代；
-明确 telemetry。

优先：
- SQL expression；
- Polars Expr；
- native Rust/NumPy kernel boundary。

---

## 六十五、BatchBlock 数据结构：不要用1000个 Series 做内部 API

定义：

```python
FactorBlockRef:
    factor_ids
    date_axis
    asset_axis
    values  # contiguous / Arrow columns
    dtype
    validity
    snapshot
    lineage
```

好处：
- multi-root output天然承载；
- block DQ；
- block normalization；
- block write；
-模型训练可直接读取；
-少1000次 Python object allocation。

单 factor API仍可由 block view提供兼容。

---

## 六十六、Factor Lake 读取也要反向服务模型训练

落值不仅要“写快”，还要保证后面：

```text
模型读取也快
```

因此 factor matrix/block布局要根据下游访问模式 benchmark：
-横截面训练：某时间区间 × 很多因子；
-单因子研究：一个factor × 长历史；
-在线预测：最新date × 全factor。

可以保留：

```text
canonical long identity/index
+
optimized wide factor blocks
```

但避免双份无版本管理数据。

通过 manifest映射同一逻辑 generation。

---

## 六十七、Write Amplification 必须成为核心 KPI

定义：

```text
write_amplification = physical_bytes_written / logical_changed_bytes
```

增量一天更新如果逻辑变化1GB，
却因为 generation copy写了100GB，
这是明确失败。

目标：

```text
接近 1 + 必要 metadata/compaction overhead
```

COW generation是关键。

---

## 六十八、压缩算法不要固定：以 decode+write 总时间选择

Parquet codec不是越压缩越好。

对：
- daily source；
- minute source；
-factor matrix；
分别 benchmark：

```text
write throughput
compressed size
decode throughput
CPU
```

生产选：

```text
end-to-end scan+write最优
```

而不是单纯最小文件。

---

## 六十九、Metadata / Footer / Manifest 也可能成为 10k 因子瓶颈

小文件太多时：
- parquet footer读取；
- stat；
-manifest parse；
-catalog lookup；
可能超过真实计算。

优化：
- block file；
- manifest index；
- metadata cache；
-批量 stat；
-immutable generation metadata；
-避免每 factor单独 footer。

---

## 七十、编译与执行的 Persistent Warm Process

对于高频挖掘服务器，尽量复用进程：
- operator registry；
-field catalog；
-DuckDB connection pool；
-Polars thread pool；
-IR cache；
-cost baseline；
-manifest cache。

避免每个 batch冷启动 import和registry rebuild。

但所有 cache必须由：

```text
implementation hash
catalog hash
snapshot/generation
```

正确失效。

---

## 七十一、CPU Oversubscription 是必须防的性能陷阱

同时存在：
- scheduler threads；
-DuckDB internal threads；
-Polars rayon pool；
-NumPy BLAS/OpenMP；
-writer compression threads。

HostResourceBroker必须统一控制。

目标：

```text
TotalRunnableThreads
```

接近有效物理核，而不是成倍 oversubscribe。

建立 runtime telemetry：

```text
python_workers
duckdb_threads
polars_threads
blas_threads
writer_threads
```

并硬门禁 nested oversubscription。

---

## 七十二、Memory Liveness Simulation：先算什么时候可以释放，再决定并行多少

Physical planner在执行前做：

```text
buffer birth
last consumer
release point
```

得到 peak live bytes。

scheduler不只看每task memory，
还看：

```text
当前活跃 buffers + candidate task output + writer queue
```

这样比简单 token更准。

---

## 七十三、Spill 策略：优先 recompute cheap node，spill expensive shared node

不要内存满就全 spill。

决策：

```text
spill_cost
vs
recompute_cost
vs
consumer wait
```

廉价 elementwise：
```text
通常重算
```

昂贵 PIT join / large rolling shared：
```text
更可能 spill/cache
```

Spill format也使用 Arrow/Parquet，不转 Pandas。

---

## 七十四、准确性不能只靠最终数值 tolerance：还要比较语义路径

高速 backend parity证据要比较：
- row grain；
-index/date/asset；
-null mask；
-+Inf/-Inf；
-dtype；
-ordering；
-PIT snapshot；
-universe；
-factor version；

浮点 tolerance只是其中一项。

对于 rank/top-k等排序敏感输出，增加：
- per-date rank correlation；
-top-k overlap；
-tie parity。

---

## 七十五、最终 Backend Selection 决策树

生产默认：

```text
Step 1: correctness eligibility
    ↓
PIT / operator / source / output semantics certified?
    no → 禁止该 backend
    yes
    ↓
Step 2: relational pushdown coverage
    ↓
scan+join+aggregation+factor+write 能否大段留 DuckDB?
    yes → DuckDB strong candidate
    ↓
Step 3: Polars native graph coverage
    ↓
Polars是否能形成更大 streaming/lazy subgraph且少转换?
    yes → Polars strong candidate
    ↓
Step 4: specialized kernel
    ↓
是否存在更快精确 native kernel?
    yes → hybrid boundary
    ↓
Step 5: measured batch cost
    ↓
选择 total predicted durable-commit cost最低计划
    ↓
Step 6: uncertainty tie-break
    ↓
source-native stay-in-engine
DA parquet/relational默认 DuckDB tie-break
    ↓
Step 7: runtime feedback
    ↓
actual cost回写 calibration
```

因此最终 policy可以简化成一句：

> **DuckDB 默认优先承接 DataAccess 的关系型主链；Polars 与其竞争纯列式/流式 native 子图；Specialized kernel 处理两者不擅长的算子；Pandas只做最小 fallback/reference。最终由批量全链路成本而不是后端名称决定。**

---

## 七十六、R33 修订版新增 Hard Gates

在原 R33 hard gates基础上增加：

```text
R33_BACKEND_STATIC_PRIORITY_HARDCODE_ZERO
R33_DUCKDB_FIRST_NOT_DUCKDB_ONLY
R33_BATCH_GLOBAL_ROUTE_PRESENT
R33_ROUTE_COST_INCLUDES_SCAN_JOIN_WRITE
R33_ROUTE_COST_INCLUDES_SHARED_BENEFIT
R33_ROUTE_COST_INCLUDES_SCHEDULER_OVERHEAD
R33_ROUTE_COST_INCLUDES_DQ
R33_ROUTE_COST_INCLUDES_GENERATION_COMMIT

R33_SOURCEREF_LOWERED_BEFORE_BACKEND_ROUTE
R33_SOURCEREF_NATIVE_POISON_ZERO
R33_MAXIMAL_NATIVE_SUBGRAPH_PARTITIONED
R33_SINGLE_UNSUPPORTED_OP_FULL_PANDAS_FALLBACK_ZERO

R33_DUCKDB_MULTIROOT_DIRECT_SOURCE_SCAN
R33_DUCKDB_TEMP_PARQUET_HOT_PATH_ZERO
R33_DUCKDB_DIRECT_BATCH_OUTPUT_SUPPORTED
R33_DUCKDB_OVERSUBSCRIPTION_ZERO

R33_POLARS_GOVERNED_LAZY_SOURCE
R33_POLARS_ONE_COLLECT_MULTIROOT
R33_POLARS_PYTHON_UDF_FASTPATH_ZERO
R33_POLARS_STREAMING_FALLBACK_VISIBLE

R33_SPECIALIZED_KERNEL_BLOCK_SUPPORTED
R33_PANDAS_ROW_APPLY_PRODUCTION_ZERO
R33_PANDAS_FULLROOT_FALLBACK_FROM_ONE_UNSUPPORTED_OP_ZERO

R33_INSTRUMENT_STRING_HOTPATH_MINIMIZED
R33_UNIVERSE_MASK_SHARED
R33_GROUP_CODE_SHARED
R33_ORDERING_CERTIFICATE_CONSUMED
R33_REDUNDANT_BATCH_SORT_ZERO

R33_MULTIQUERY_CSE_COST_BASED
R33_COMMON_WINDOW_EXTRACTION
R33_OPERATOR_MACRO_LOWERING_PARITY
R33_FACTOR_CAMPAIGN_SESSION
R33_INCREMENTAL_FORMULA_COMPILE

R33_CPU_TOTAL_RUNNABLE_GOVERNED
R33_MEMORY_LIVENESS_SIMULATED
R33_WRITE_AMPLIFICATION_REPORTED
R33_WRITE_AMPLIFICATION_THRESHOLD_PASS

R33_END_TO_END_DURABLE_COMMIT_BENCH
R33_BACKEND_A_B_C_CURRENT_SHA
R33_ROUTER_PREDICTED_VS_ACTUAL_ERROR_REPORTED
R33_RUNTIME_CALIBRATION_CURRENT_HARDWARE
```

---

## 七十七、R33 修订版最终性能目标

不在文档里伪造固定“5x/10x”。

但优化方向必须让以下数字持续下降：

```text
physical scans / batch
source bytes decoded
PIT joins / batch
sorts / batch
representation conversion bytes
Python objects created
scheduler overhead ratio
peak live bytes
writer queue wait
files created
write amplification
TimeToFirstFactor
TimeToAllComputed
TimeToDurableCommit
```

并让以下数字持续上升：

```text
scan reuse ratio
native backend coverage
roots per native query
factors per source scan
cells/sec
write MB/sec
cache benefit density
CPU useful utilization
```

最终不是追求某一个 backend“赢”，而是让：

```text
你的 FactorEngine + DataAccess 整体系统赢。
```

---


# 附录 A｜本版给整改 AI 的一句总指令

> 在当前 `499078505edff30b1268b60862dbcaab2f9ab31e` 基线上，把 FactorEngine 与 DataAccess 从“两个分别规划、Python 边界拼接的系统”改造成一个批量 Unified Query Graph。DataAccess 负责 source/snapshot/PIT/join/scan 的单一事实源；FactorEngine负责 factor/operator semantics；Physical Optimizer 在全批量层面做 cost-based backend partition。默认尽量让 DataAccess 的 parquet/relational 主链继续留在 DuckDB，并给 DuckDB source-affinity tie-break，但绝不写死 DuckDB 永远快于 Polars；Polars streaming/lazy 对完整 native expression graph是一级候选；Pandas只做 reference和最小 fallback，难算子优先 specialized native kernel。真正优化目标是最小化 `TimeToDurableCommit`，同时保证 PIT、snapshot、universe、null/Inf、dtype、grain、排序和 numerical parity。所有新增模块必须在 production 主路径真实执行，并用 runtime evidence而不是 grep/static presence证明。

# 附录 B｜修订版优先级速查

```text
P0-1  真 ReadWave + 真 SourceScan BufferRef
P0-2  FE demand → DA DataRequest/PreparedBatchReadSession
P0-3  SourceRef relational lowering
P0-4  batch-global backend optimizer
P0-5  DuckDB multi-root end-to-end query
P0-6  Polars governed lazy multi-root
P0-7  one unsupported op 不再 full Pandas fallback
P0-8  Arrow/Polars canonical block，MultiIndex退出 hot path
P0-9  query-scoped snapshot / source relation reuse
P0-10 PIT join / minute aggregate multi-consumer reuse
P0-11 FactorBlock + block DQ + partition-aware writer
P0-12 COW generation，消灭整代复制
P0-13 CPU/内存/内部线程统一治理
P0-14 CampaignSession 给自动挖掘循环复用
P0-15 current SHA真实100/1000/10000 workload benchmark
```

# 附录 C｜Backend 策略证据说明

本版不采用固定 `DuckDB > Polars > Pandas` 的原因：

1. DuckDB官方文档明确指出系统间公平 benchmark 很困难，结果依赖 workload与环境。
2. Polars官方 2025 PDS-H benchmark：SF-10 streaming Polars快于DuckDB，而SF-100 DuckDB快于Polars streaming，说明不存在稳定的绝对排序。
3. Polars lazy optimizer具备 projection/predicate pushdown、common subplan elimination、join ordering和streaming。
4. DuckDB与Polars之间已有Arrow高效互操作能力，因此两者更合理的关系是“native region竞争 + 尽量少边界转换”。
5. 本项目 DataAccess 当前关系执行与 parquet SQL能力天然更靠近DuckDB，所以**在成本不确定且完整关系链可留在DuckDB时，DuckDB应作为默认 tie-breaker**；这与“DuckDB无条件最快”是两回事。
