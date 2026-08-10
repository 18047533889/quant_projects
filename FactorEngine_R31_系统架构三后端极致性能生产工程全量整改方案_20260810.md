# FactorEngine R31：系统架构、三后端、极致性能与生产工程全量整改方案

- 仓库：`18047533889/quant_projects`
- 审计对象：`factor_engine`，必要时联动 `dataaccess`
- 当前基线 HEAD：`58490a634eef36efc92ecaec945fc4a26016661c`
- 日期：2026-08-10
- 说明：用户明确当前仍在修改、尚未产生新的 GitHub 提交，因此本轮继续以该 HEAD 为基线。
- 本文件是 **R31 独立整改轮次**。R30 主要解决逐算子隐形语义、随机/未来物理删除、PIT/交易时钟、复权/股票池等；R31 不重复展开这些内容，而是继续审 **FactorEngine 作为完整生产计算系统** 的架构、后端、编译、调度、数据平面、缓存、增量、服务、CI、发布与性能。

---

## 总体结论

当前 FactorEngine 已经不是一个简单的“公式解释器”，而是已经具备：

- DSL / Analyzer / IR / Logical Plan；
- CSE / rolling CSE；
- Pandas reference；
- Polars panel / Polars long；
- DuckDB / ClickHouse SQL emitter 与 partial pushdown；
- whole-plan backend cost router；
- DataAccess integration；
- execution resource plan；
- AdaptiveBatchScheduler；
- ResourceBroker；
- HybridExecutor；
- read-wave planner；
- native multi-root fusion；
- incremental materialization；
- persistent cache；
- HTTP service / queue / policy / observability。

因此下一阶段不应该继续以“再多写几个模块”为主要思路，而应该转向：

> **把已有组件接成一条真正统一的 physical execution pipeline，并消灭 duplicated control planes、未接线优化器、表示转换、错误成本估计和失败路径资源泄漏。**

当前最值得优先修的不是算子数量，而是：

1. **默认主执行链没有真正使用已经写好的 AdaptiveBatchScheduler。**
2. **所谓 PhysicalFactorDAG 当前仍未真正 lower 到 source/operator/group/state/write 粒度。**
3. **scheduler 创建 task 时 backend_candidates 为空、preferred_backend 固定 Pandas。**
4. **whole-plan cost router 以“去重 canonical 集合”估价，丢失重复节点、参数和真实 DAG 拓扑。**
5. **354 个 SQL emitter 已存在，但当前生成 evidence 显示 DuckDB parity verified=0 / production safe=0。**
6. **DataAccess 已经有 DataRequest/ReadPlan/read_joined/sql_relation，但 FactorEngine 还没有把整批 DAG 数据需求完整下推给 DataAccess。**
7. **线程/进程/resource broker 已实现，但失败、重试、pickling、token accounting、IO pressure 等仍有隐患。**
8. **Optimizer 还非常轻，离真正的 columnar query optimizer 还有较大空间。**
9. **Pandas/Series/wide panel/Polars long/Arrow/SQL relation 多表示并存，转换仍可能成为主要成本。**
10. **README / backend coverage / dependency declarations已有明显漂移，需要改成 generated truth。**

R31 的目标是把 FactorEngine 从“功能齐全”升级为：

> **统一编译器 + 统一 physical planner + 统一资源调度器 + 统一列式数据平面 + 多后端子图执行 + 增量 change propagation + 可验证性能系统。**


## 1. R31 与 R30 的边界

R30 已经覆盖：

- 随机数/未来函数物理删除；
- Research 隔离；
- 每个算子语义/PIT/缺失/数值；
- Availability/Decision/Execution Clock；
- corporate-action vintage；
- as-of universe/survivorship；
- label/feature firewall；
- semantic identity/evidence invalidation。

R31 重点覆盖：

```text
Compiler
IR / Logical Plan
Optimizer
Physical Planner
Backend partition
Pandas / Polars / DuckDB / ClickHouse
Batch execution
CSE
Fusion
Scheduler
Thread/process
CPU/RAM/IO/spill
DataAccess
Arrow/columnar representation
Cache
Incremental recompute
Materialization
Failure/retry
Service concurrency
Observability
Benchmark
CI
Packaging
Release
Mining integration
```


## 2. 当前架构最大的系统问题：新 scheduler 没真正成为默认执行器

当前 `execute_run_many()` 主路径依然：

```text
compile DAG
→ materialize_shared_nodes_parallel()
→ build_factor_batch_graph()
→ layer 1 roots
→ layer 2 roots
...
```

`execute_run_many_parallel()` 也仍然：

```text
先 shared CSE 全部物化
→ 再每层 ThreadPoolExecutor 跑 roots
```

而代码库已经有：

```text
AdaptiveBatchScheduler
ResourceBroker
HybridExecutor
ReadWavePlan
PhysicalFactorDAG
NativeFusionGroup
```

两套执行 control plane 并存。

### 必须整改

生产默认：

```text
FactorEngine.run_many
FactorEngine.run_many_parallel
materialize_many
service job
```

最终都进入：

```text
BatchCompiler
→ PhysicalPlanner
→ AdaptiveBatchScheduler
→ StreamingSink
```

旧 `batch_service` layer-loop 只保留：

```text
reference / compatibility / debug mode
```

不能继续是默认 production 路径。

### Hard Gate

```text
R31_DEFAULT_BATCH_EXECUTOR_IS_ADAPTIVE_SCHEDULER = true
R31_PRODUCTION_LAYER_LOOP_EXECUTOR_COUNT = 0
```


## 3. 当前 PhysicalFactorDAG 还是“声明很完整、实际 lower 不完整”

`physical_factor_dag.py` 声明支持：

```text
SOURCE_SCAN
SOURCE_JOIN
SOURCE_AGG
OPERATOR
CSE_SHARED
ROLLING_SHARED
GROUP
CROSS_SECTION
STATEFUL
ROOT
WRITE
```

但当前 `AdaptiveBatchScheduler.plan()` 实际主要创建：

```text
CSE_SHARED task
ROOT task
```

而且 shared task：

```text
inputs=()
```

并没有把 shared node 内部 predecessor 真实拆出来。

因此当前 scheduler 即使启用，也还不是完全的 operator/stage-level physical DAG。

### 正确做法

新增：

```text
planner/physical_lowerer.py
```

把 optimized logical DAG lower 为真实 physical stages：

```text
SourceScanTask
SourceJoinTask
SQLSubgraphTask
PolarsLazySubgraphTask
PandasKernelTask
RollingSharedTask
CrossSectionBarrierTask
StatefulTask
ConversionTask
MaterializeTask
SpillTask
WriteTask
```

### 原则

不是“每个 operator 一定一个 task”。

应根据：

```text
backend
representation
barrier
statefulness
materialization need
reuse
memory
```

合并成物理 stage。

目标是：

```text
logical node很多
physical stage尽量少
representation transition尽量少
```


## 4. Scheduler 当前 backend 信息没有真正接进 task

当前 scheduler task 默认：

```text
backend_candidates=()
preferred_backend="pandas_numpy"
```

resource contract 也默认：

```text
backend="pandas_numpy"
cpu_tokens=1
gil_bound=False
releases_gil=True
```

这与实际任务可能完全不符。

### 必须改

PhysicalPlanner 在生成 task 时填：

```text
backend_candidates
preferred_backend
alternative_backends
backend_costs
backend_threads
gil_bound
releases_gil
streamable
materializes_full_panel
spillable
shardable
```

这些必须来自：

```text
OperatorCapability
PlanCostRouter
DataAccess capability
physical subgraph
```

Scheduler 不能自己猜 backend。


## 5. 当前 scheduler 失败路径存在 task_id 与资源归还风险

当前 pattern：

```python
try:
    tid, result = future.result()
except Exception:
    retries = self._retries_remaining.get(tid, 1)
```

如果 `future.result()` 直接抛异常，当前 future 的 `tid` 没有从返回值成功绑定。

同时失败/retry 分支若没有对应：

```text
broker.release(contract, task_id)
```

会造成：

```text
CPU token leak
IO token leak
memory reservation leak
```

最终可能出现：

```text
系统有资源
但所有新 task 都 admission rejected
```

### 正确设计

维护：

```python
future_to_task_id: dict[Future, str]
```

所有 future 终态：

```text
SUCCESS
FAILED
CANCELLED
BROKEN_WORKER
TIMEOUT
```

统一走：

```text
finally:
    release reservation exactly once
```

增加：

```text
ReservationLease
```

对象，支持幂等 release。

### Hard Gate

```text
R31_TASK_FAILURE_RESOURCE_LEAK_ZERO
R31_RETRY_RESERVATION_BALANCE_PASS
```


## 6. Retry 不能对所有异常一视同仁

分类：

```text
TRANSIENT_IO
REMOTE_TIMEOUT
WORKER_CRASH
OOM
BACKEND_ENGINE_FAILURE
BAD_DATA
SEMANTIC_ERROR
PIT_ERROR
INVALID_PARAM
CODE_BUG
```

只有 transient 类自动 retry。

以下禁止 retry：

```text
PIT violation
semantic violation
invalid parameter
unsupported operator
schema mismatch
deterministic numeric error
```

否则只是重复浪费资源。

OOM 应：

```text
reduce shard / lower concurrency / spill
```

而不是原尺寸直接再跑一次。


## 7. HybridExecutor 当前真正多进程路径还不够可靠

当前架构方向正确：

- Polars/DuckDB/NumPy native → thread；
- Pandas/Python loop → process；
- long-lived process pool。

但需要修几个关键点。

### 7.1 Process initializer

当前 `_process_initializer()` 返回 nested function。

在 spawn 环境（特别是 macOS/Windows）有 pickling 风险。

改：

```python
def _worker_initializer(threads: int):
    ...

ProcessPoolExecutor(
    initializer=_worker_initializer,
    initargs=(threads,),
)
```

### 7.2 不传大 backend/context/closure

当前 scheduler dispatch payload包含：

```text
backend
ctx
execute_root callable
materialize_shared callable
```

其中 lambda/closure、DataSource、cache、backend instance 很可能：

```text
不可 pickle
或 pickle 成本巨大
```

正确模型：

```text
主进程：
PhysicalTaskSpec（小、可序列化）
        ↓
worker:
worker-local BackendRuntime
worker-local DataAccess handle
        ↓
ResultReference
```

不要跨进程传大 DataFrame。

### 7.3 共享数据

Pandas process worker需要：

```text
Arrow IPC
memory mapped Arrow
shared memory
或 worker-local reread filtered partitions
```

而不是：

```text
pickle 3GB wide DataFrame。
```

### 7.4 BrokenProcessPool

worker crash 后：

```text
重建 process pool
熔断连续失败 backend
```

不能让整个 Engine 永久不可用。


## 8. HybridExecutor 的 cpu_tokens 参数目前语义不完整

`submit(... cpu_tokens=1)` 暴露了 cpu token 参数，但 Executor 自身没有真正使用它做 admission。

推荐职责清晰化：

```text
ResourceBroker = 唯一 admission authority
HybridExecutor = 只负责执行已 admission 的 task
```

因此 Executor API直接去掉 `cpu_tokens`，避免让人以为它有第二套资源治理。

或反过来由 Executor持有 lease，但只能有一个 authority。


## 9. 不要只按 backend 名字判断 GIL

当前：

```text
pandas_numpy → process
polars/duckdb → thread
```

过于粗。

同一个 pandas_numpy operator可能是：

```text
NumPy vectorized / scipy C
```

可以释放 GIL。

同一个 Polars slot可能：

```text
map_groups Python UDF
```

反而 GIL-heavy。

应以每个 physical task 的：

```text
gil_bound
releases_gil
backend_threads
python_callback
```

为准，不以 backend name为唯一依据。


## 10. ResourceBroker：can_admit 不应该有副作用

当前 `can_admit()` 会实际：

```text
try_acquire CPU tokens
try_acquire IO tokens
```

如果调用方只是询问：

```python
broker.can_admit(task)
```

而没有随后 reserve/release，token会泄漏。

### 必须改

```text
can_admit = pure function
reserve = atomic check + acquire
release = idempotent
```

最好：

```python
lease = broker.try_reserve(task)
if lease:
    ...
    lease.release()
```


## 11. ResourceBroker 当前 memory accounting可能过度双算

`live_headroom` 已经反映：

```text
cgroup current
host MemAvailable
process family RSS
```

之后 `can_admit()` 又：

```text
sum(running_task_predicted_peaks)
```

全部再扣一次。

这会非常保守，尤其 running task 已经进入实际 RSS 时。

建议：

```text
reserved_remaining =
max(0, predicted_peak - observed_increment)
```

或：

```text
memory_guard = max(actual_process_usage, reserved_commitment)
```

不要：

```text
actual usage + full predicted peak again
```

否则机器容易只跑 1~2 个 task。


## 12. ResourceBroker 的“external CPU”当前会把自己的负载算进去

当前 comment 描述：

```text
external CPU ≈ system CPU - our CPU share
```

实际却平滑：

```text
system_cpu_util
```

FactorEngine 自己把 CPU跑到 80% 时，broker可能误以为：

```text
外部负载很高
```

然后自我降并发。

### 修复

采样：

```text
system cpu
process-family cpu
```

计算：

```text
external_cpu = max(0, system - factor_engine_family_share)
```

再做 EMA。


## 13. ResourceBroker 的 disk pressure 目前基本没有实现

当前：

```python
_disk_busy() -> 0.0
```

所以 IO pressure stage并不知道磁盘是否已经饱和。

应支持：

```text
psutil.disk_io_counters delta
Linux /proc/diskstats
iostat optional
remote object-store latency
ClickHouse query latency
```

并区分 IO token pool：

```text
local_nvme
network_fs
object_store
clickhouse
```

不能全部用：

```text
hard_cpu_slots
```

作为 IO concurrency。


## 14. Spill reserve 当前存在量纲问题

当前可用 spill 预留里，reserve fraction 使用了：

```text
hard_memory_limit * spill_min_free_fraction
```

spill free 的 fraction 应针对：

```text
spill filesystem total capacity
```

而不是 RAM。

修：

```text
spill_total_bytes
spill_free_bytes
spill_reserve=max(20GB, spill_total*10%)
```


## 15. Resource sampling 本身也要轻量

当前每次 refresh：

```text
recursive process RSS
recursive process PSS
```

分两次遍历 process tree；

`psutil.cpu_percent(interval=0.05)` 还会阻塞约 50ms。

大量 scheduler tick 时会产生控制面成本。

建议独立 background sampler：

```text
500ms~1s
```

一次采：

```text
RSS/PSS/CPU/IO
```

scheduler只读取 lock-free/latest snapshot。


## 16. Cost Router 当前最大的准确性问题：把 DAG压成“去重 canonical 列表”

当前 `_canonical_ops(plan)` 使用 `seen`，一个 canonical只保留一次。

于是：

```text
ts_mean(close,5)
+ ts_mean(volume,20)
+ ts_mean(amount,60)
+ ts_mean(close,120)
```

可能在 whole-plan backend cost中只看到：

```text
ts_mean × 1
```

这是严重低估。

同时：

```text
window=5 vs 120
```

参数信息也消失。

### 必须改

cost model基于：

```text
BoundPhysicalNode
```

而不是 canonical set。

每个 occurrence单独有：

```text
canonical
bound params
rows
cols
window
group size
feature_dim
backend
input representation
output representation
reuse count
```


## 17. CostContext 已经写了，但 whole-plan route没有充分使用

`operator_cost.py` 已经有：

```text
window
k
feature_dim
regressors
group_count
session_bars
density
```

这是好设计。

R31 要求：

```text
Analyzer / binder
→ CostContext
→ PhysicalPlanner
→ BackendRouter
→ Scheduler
```

全链路传递。

不能 CostContext只存在于 API，却 whole-plan route仍按：

```text
canonical + row_count
```

估价。


## 18. mixed/hybrid cost 目前不是实际 DAG partition optimization

当前 mixed cost主要：

```text
按 canonical first occurrence
每个 op选最低 backend
统计 backend transition次数
```

这无法表达：

```text
branch A Polars
branch B SQL
join at Pandas
CSE reused 20 times
```

### 应改成真正的 DAG dynamic programming

对每个 node计算：

```text
cost[node, output_representation/backend]
```

包括：

```text
child execution
conversion
materialization
spill
network
reuse
```

然后选最小 physical plan。

类似 query optimizer 的：

```text
Volcano / Cascades-lite
```

不需要复杂到数据库级，但必须是：

```text
subgraph-aware
```


## 19. Backend conversion penalty 需要统一记账一次

当前 Polars panel 成本存在重复 conversion penalty 的风险：

```text
单算子 _cost(... requires_conversion=True)
```

已经加一次，

whole-plan Polars 后又加一次 plan conversion。

应拆成：

```text
OperatorExecutionCost
EdgeConversionCost
ScanCost
MaterializationCost
```

operator 本身不携带“每次都转换”的假设。

只有 physical edge：

```text
Arrow → Pandas
Pandas → Polars
SQL → Arrow
Long → Wide
```

真正发生时才计一次。


## 20. Memory cost 必须 backend-specific，而不是一份 generic peak

当前 plan route先算一个 backend-independent：

```text
rows * 8 * memory factor
```

若它超过预算：

```text
所有 backend候选一起拒绝。
```

但现实：

```text
DuckDB streaming SQL
```

可能无需整个 wide panel驻留；

```text
Polars lazy
```

也可能 projection pushdown + streaming；

Pandas则可能真的需要全量 materialization。

所以：

```text
PeakMemory(plan, backend, physical_partition)
```

必须分 backend。

### 还应考虑

```text
input bytes
live intermediates
dtype
n columns
window buffers
sort buffers
hash/group buffers
Arrow conversion
CSE outputs
result output
```


## 21. 当前 measured benchmark hardware fingerprint 检查不完整

benchmark baseline记录了：

```text
cpu_model
effective_cores
ram_gb
storage_class
```

但当前兼容性判断主要比较：

```text
python/numpy/pandas/polars/duckdb
```

软件版本。

这意味着不同服务器仍可能错误使用同一 measured baseline。

### 修复

至少建立 hardware family：

```text
CPU architecture/model family
core bucket
RAM bucket
storage class
remote/local data
```

允许：

```text
exact
compatible_family
estimate_only
```

三种级别。


## 22. Runtime calibration应该真正反哺 physical planner

当前已有 `record_plan_actual()` / runtime calibration。

下一步：

```text
predicted vs actual
```

按：

```text
operator/backend/shape/window/market/frequency
```

持续校准：

```text
elapsed
peak RSS
output bytes
conversion time
scan bytes
```

而不仅修正一个 plan total work。

路由器优先级：

```text
recent local measured
> compatible baseline
> static model
```


## 23. 三后端战略：不要要求所有 1362 个都三后端

推荐把“三后端”定义成：

```text
Pandas/NumPy reference
Polars native
DuckDB SQL native
```

ClickHouse 是独立 SQL dialect production certification，不自动继承 DuckDB。

### Tier A：高 ROI，应该三后端

优先全部实现 + parity：

```text
四则运算
比较/逻辑
abs/log/exp/sqrt/sign/clip
protected math
where/coalesce/is_null/is_finite
ts_delay/delta/pct/log_return
rolling sum/mean/min/max/std/var/count
rolling covariance/correlation
rolling rank/quantile（SQL语义可严谨对齐时）
cs sum/mean/std/count/rank/zscore/demean
group sum/mean/std/count/rank/zscore
simple OHLC transforms
price/volume ratios
common volatility estimators
simple fundamental ratios
common candlestick geometry
```

### Tier B：有价值，逐批三后端

```text
rolling regression/beta
weighted group transforms
neutralization
technical indicators with bounded rolling definition
some EWM if SQL/native semantics can exactly match
simple intraday daily aggregations
event count/rolling state statistics
```

### Tier C：不要为了三后端而硬写

```text
recursive state machines
KAMA/PSAR/Supertrend复杂递归
HMM/Kalman/GARCH
DMD/spectral/wavelet
path signatures
matrix profile
graph/PageRank/network
dynamic KNN
high-dimensional model fitting
complex shareholder/relation identity logic
custom PIT source transforms
```

这些应：

```text
Pandas/NumPy/Numba reference
或 specialized native backend
```

而不是制造低质量 SQL。


## 24. 当前 SQL 最大机会：354 emitter 已有，但认证为 0

当前生成 `docs/sql_pushdown_coverage.md`：

```text
active SQL emitter implementations = 354
DuckDB parity verified = 0
DuckDB production safe = 0
```

这说明短期最有效任务是：

```text
SQL certification factory
```

而不是继续写 emitter。

### 自动化 parity harness

每个 candidate：

```text
Pandas reference
vs
DuckDB SQL
```

在以下 fixtures对比：

```text
normal
NaN
Inf
ties
zero denominator
short history
group singleton
constant
large magnitude
random-but-fixed synthetic
A-share realistic fixture
```

输出：

```text
value parity
NaN mask parity
dtype/unit parity
index order
warmup
```

### 先认证 50~100 个核心

再逐批扩到：

```text
200+
300+
```

而不是一次把354全开 production。


## 25. ClickHouse 不得自动继承 DuckDB

同一个 SQL emitter在：

```text
DuckDB
ClickHouse
```

可能存在：

```text
NULL
NaN
window frames
quantile
stddev ddof
argMax
division
timestamp
```

差异。

必须单独：

```text
DUCKDB_PARITY
CLICKHOUSE_PARITY
```

ClickHouse更适合：

```text
远程大规模 scan
简单 projection
group/window
批量因子生产
```

而不是所有复杂数学。


## 26. Polars 的机会：让 native expr 成为正常 planner 路径，不是 env flag

当前 whole-tree Polars expression fast path仍有：

```text
FACTOR_ENGINE_POLARS_EXPR
```

这种 opt-in。

对于已经 production-certified 的 native expression，不应靠运维手工开关。

应由 PhysicalPlanner自动：

```text
识别 maximal Polars-native subgraph
→ compile LazyFrame expressions
→ one collect
```

环境变量仅用于：

```text
debug force-disable/force-enable
```


## 27. PolarsBackend 继承 Pandas recursive evaluator，仍有 Python dispatch 成本

即使 operator kernel是 Polars，当前路径仍可能：

```text
Python recursive _eval
→ kernel lookup
→ child eval
→ Polars op
```

对于大规模表达式：

```text
编译成一个 Polars Expr DAG
```

比逐节点 Python dispatch更好。

目标：

```text
90% native Polars factor root
= one LazyFrame plan + one collect
```

而不是：

```text
100个 Python operator dispatch + 100个小 Polars对象。
```


## 28. Polars long 应成为主要 columnar compute representation

对于：

```text
datetime
instrument
field/value
```

或：

```text
datetime
instrument
多列字段
```

长表更适合：

```text
scan
filter
group
window
join
```

不要频繁：

```text
long
→ wide pandas
→ unstack
→ Polars
→ long
```

### 推荐

Data plane canonical：

```text
Arrow / Polars Long
```

Pandas wide只在：

```text
reference / specialized operator boundary
```

临时 materialize。


## 29. DataAccess 已经有高级计划能力，FactorEngine 要真正使用

DataAccess 已有：

```text
SemanticFieldCatalog
DataRequest
ReadPlan
read_joined
pit_asof
manifest snapshot token
sql_relation
ScanCost
```

FactorEngine 应在整批 compile 后生成：

```text
BatchDataRequest
```

内容：

```text
all required logical fields
physical datasets
time windows
universe
PIT joins
filters
source transforms
required ordering
snapshot policy
```

然后：

```text
DataAccess.plan(request)
```

一次完成：

```text
projection coalesce
file pruning
PIT join
filter
snapshot
```

再把一个或少量 relation交给 compute layer。


## 30. 不要让每个 factor/column驱动数据读取

目标流程：

```text
1000 factors
→ union source dependencies
→ group by source scope
→ one/few DataRequests
→ one scan per physical source wave
```

而不是：

```text
factor1 load close
factor2 load close+volume
factor3 load volume
...
```

虽然 column cache会缓解，但仍不如 planner级 projection/read coalescing。


## 31. DataAccess catalog field resolution应批量

当前适配器对 catalog字段存在逐字段：

```text
resolve_fields([name])
```

调用。

改为：

```text
resolve_fields(all_names)
```

一次返回 typed map。

对 1000 因子、数百 field dependency可以显著降低 Python/API overhead。


## 32. Semantic catalog version不要每次全量重新 hash

当前可见实现会遍历整个 catalog field map计算版本。

应让 DataAccess直接提供：

```text
semantic_catalog_generation
semantic_catalog_digest
```

FactorEngine只读取一个 token。

catalog mutation时更新 generation。


## 33. Column cache + Panel cache 会重复驻留同一数据

当前 DataAccessSource可能同时缓存：

```text
MultiIndex Series
wide DataFrame panel
```

同一 close 数据可能保留两份。

分钟级数据尤其昂贵。

### 推荐

执行 session 里选一个 canonical representation：

```text
Arrow/Polars long
or
wide panel
```

根据 physical plan决定。

避免同一 field长期双缓存。

若必须双缓存：

```text
shared buffer / zero-copy view
```

而不是复制。


## 34. Wide panel pivot/unstack 是大成本，需要批量/延迟

`load_column_panel()`：

```text
Series → unstack
```

每个字段单独做会重复构造 index/columns。

改：

```text
prefetch wide panels once
```

一次构造：

```text
MultiFieldPanel
```

或坚持 Polars long，不 pivot。

Pandas specialized operator只取其需要的 wide view。


## 35. Ordering/sort 也要 planner-aware

Polars long path当前为保证：

```text
instrument,time
```

顺序会重新 enforce sort。

如果 DataAccess manifest / dataset contract已保证排序：

```text
ordering_guarantee
```

PhysicalPlanner可以跳过冗余 sort。

如果不能证明：

```text
必须 sort。
```

不要靠假设。


## 36. DataAccess snapshot revalidation 存在 production fail-open路径

`revalidate_for_long_collect()` 在无 manifest时回退 `describe_dataset()`。

如果 describe本身异常，目前路径可能直接 return。

Production 应：

```text
SnapshotRevalidationUnavailable
```

hard fail。

Research才：

```text
warning + continue。
```

否则“collect前重新验证”的安全承诺不完整。


## 37. SourceRef 不应永久阻断所有后续 SQL pushdown

当前含 SourceRef 的 subtree按 opaque non-SQL处理是安全的。

但可以引入：

```text
SourceResolveTask
```

先把 SourceRef解析成一个：

```text
typed materialized relation
```

然后其下游纯数值子图重新进入：

```text
DuckDB/Polars pushdown
```

而不是 SourceRef一出现，整条上游链永远 Python。


## 38. SQL partial pushdown 后 residual backend需要再次 cost-route

当前 SqlBackend：

```text
SQL subtrees
→ materialized_series/long
→ residual root
```

residual `_eval_hybrid()` 默认基本：

```text
Pandas指定时Pandas
否则Polars
```

应重新判断：

```text
residual DAG
input representation
remaining ops
conversion cost
```

选择：

```text
Polars
Pandas/Numba
```

不能固定“SQL后就Polars”。


## 39. SQL partial pushdown 可以跨多个 factor 一次 batch query

现在 batch SQL subtrees主要发生在一个 root 的 physical plan中。

进一步做：

```text
same DataRequest
same source
same scope
multiple factor roots
```

把多个 SQL roots：

```sql
SELECT
  ..., factor_a_expr AS fa,
  factor_b_expr AS fb,
  ...
```

一次 query。

这与 NativeMultiRootFusion合并为同一个概念。


## 40. NativeFusion 当前有 scope 校验缺口

`can_fuse_roots()` 当前检查：

```text
backend
source_scope
snapshot
```

但应同时检查：

```text
execution_scope
market
universe
calendar
decision-time policy
frequency
availability
```

而 scheduler当前 root task的 `source_scope/source_snapshot_id` 又默认空。

因此 fusion安全身份必须从 FactorExecutionScope/DataRequest snapshot真实填入。


## 41. adaptive_fusion_block_size 要真正接入

已经有：

```text
32/64/128/256
```

自适应函数，但 group planner仍常用固定 block。

实际 block取决于：

```text
backend compile latency
SQL expression length
Polars optimizer latency
output width
memory
root complexity
```

用 runtime calibration自动选。

不要因一次 SELECT 塞 2000 个表达式导致 compile本身变慢。


## 42. Fusion groups 必须真正执行，而不是只出现在 plan metadata

R31验收必须证明：

```text
planned fusion group
→ backend.execute_multi_roots
→ one physical scan/query
```

有实际 runtime event。

指标：

```text
native_fusion_planned
native_fusion_executed
native_fusion_fallback
roots_per_fusion
scan_reduction
```


## 43. Optimizer 当前太轻，需要升级为真实 query-style optimizer

当前主要：

```text
constant folding
composite lowering
fastpath rewrite
canonical params
```

下一步加：

```text
required-column propagation
projection pruning
filter pushdown
predicate simplification
common rolling-window fusion
group aggregation fusion
multi-output operator fusion
representation-aware materialization
backend subgraph partition
dead node elimination
CSE after lowering
CSE after physical rewrite
```

但所有代数 rewrite必须：

```text
semantic-contract proven safe
```

NaN/Inf/zero语义不允许凭数学恒等式乱改。


## 44. Multi-output operator 是非常大的提速机会

很多算子族重复做同一中间计算：

```text
MACD line / signal / hist
PPO / signal / hist
PVO / signal / hist
DMI+ / DMI- / DX / ADX
rolling regression beta/intercept/resid/r2/tstat
Keltner mid/upper/lower/position
Bollinger mid/upper/lower/width/position
```

当前如果每个 canonical独立 root：

```text
同一 rolling/EMA/regression可能重复算多次。
```

### 引入 MultiOutputKernel

```text
PhysicalKernel:
    outputs = {a,b,c,d}
```

DAG多个 canonical output只是：

```text
projection/ref
```

这样比靠结构 CSE猜共同子式更可靠。


## 45. Rolling state fusion

对于同一：

```text
input
window
missing policy
```

同时需要：

```text
mean
std
sum
count
```

可以用一套 rolling accumulator。

同理：

```text
cov/corr/beta
```

共享：

```text
sum_x
sum_y
sum_x2
sum_y2
sum_xy
count
```

Polars/SQL可交给引擎优化；

Pandas/Numba backend可自己做 fused rolling kernel。


## 46. Cross-sectional aggregate fusion

同一日期截面同时算：

```text
mean/std/count/zscore
```

不要每个 factor重新 group/date scan。

物理层生成：

```text
CrossSectionAggregateTask
```

一次产生多个统计量。


## 47. 编译本身也会成为 1万/10万因子规模的瓶颈

当前多因子：

```text
每个 Factor 独立 Analyzer.lower
独立 Lowerer
独立 Optimizer
之后再 CSE
```

对 LLM自动生成海量表达式，Python编译开销不小。

新增：

```text
ExpressionInternPool
TypedIRCache
CompiledPlanCache
```

key：

```text
canonical expression
semantic catalog generation
operator generation
market/scope
```

同一子表达式在 compile阶段就 intern。

不是执行后才发现 CSE。


## 48. Batch Analyzer

支持：

```python
analyzer.lower_many(expressions)
```

一次：

```text
parse
field resolve
operator signature resolve
semantic typing
history analysis
```

共享 registry/catalog lookup。

尤其 AlphaProbe/LLM每轮生成几千公式时收益很大。


## 49. Compile cache 与 execution cache要分开

建议明确：

```text
L0 parse cache
L1 typed IR cache
L2 logical/optimized plan cache
L3 physical plan cache
L4 runtime subplan result cache
L5 materialized factor cache
```

每层有独立 identity。

不能一个 `plan_cache_key` 同时承担所有层。


## 50. Physical plan cache必须绑定机器/能力

Logical plan可以跨机器。

Physical plan可能依赖：

```text
Polars installed
DuckDB version
hardware
memory budget
source capabilities
backend evidence
```

所以 physical plan cache key额外绑定：

```text
RuntimeCapabilityDigest
HardwareFamily
SourceCapabilityDigest
```


## 51. Persistent execution cache 继续收紧 fail-closed namespace

当前 persistent cache已有 schema/checksum/semantic namespace，方向很好。

R31再要求：

```text
operator semantic digest
field catalog digest
source snapshot
universe snapshot
corporate-action vintage
compiler version
backend numerical policy
```

全部进入正确层级 key。

`unknown_ops` / unknown numeric semantics：

```text
production禁止写 persistent cache。
```


## 52. Cache admission要按 reuse/value，不只是对象大小

缓存一个 2GB CSE：

```text
只被消费1次
```

通常不划算。

缓存价值：

```text
recompute_cost * future_reuse
/
resident_bytes
```

实现：

```text
value-aware cache admission
```

高 reuse昂贵 CSE优先。

低 reuse大对象：

```text
stream / recompute
```


## 53. CSE liveness/refcount应该进入 scheduler

当前 batch_service有 CSE refcount release，这是好设计。

统一 scheduler后：

```text
consumer task committed
→ decrement reference
→ last consumer
→ release/spill/cache-policy
```

不应在 root layer结束后批量释放。

这样真正实现：

```text
最短 live range。
```


## 54. Spill 应成为真正执行策略，不只是资源字段

当前已有：

```text
spillable
spill_bytes
spill free
```

但要完整实现：

```text
Arrow IPC / Parquet temp spill
checksum
task-owned spill handle
refcount
automatic cleanup
crash recovery
```

优先 spill：

```text
large CSE
materialized SQL subtrees
intermediate Arrow tables
```

不要 spill：

```text
tiny scalar
cheap-to-recompute node。
```


## 55. 增量计算：从“factor watermark”升级到 Change Impact DAG

当前已有：

```text
watermark
backward_history
forward_impact
state requirement
checkpoint
```

下一步输入一个变化集合：

```text
(dataset, field, instrument, time interval, source_version)
```

沿 dependency DAG传播：

```text
source change
→ affected operator intervals
→ affected root partitions
```

只重算真正受影响的数据。

### 示例

`ts_mean(close,20)`：

```text
close 在 T 修订
→ factor affected [T, T+19]
```

recursive EMA：

```text
T → until checkpoint/replay end
```

fundamental revision：

```text
announcement/revision time开始
→ next vintage before replacement
```


## 56. Incremental recompute需要 partition-aware

物化层不要：

```text
有一个2024年日期修订
→ 整个2024 partition读写重建
```

如果分区太粗：

```text
year
```

upsert成本会很大。

根据 factor frequency /规模：

```text
year
month
trade_date bucket
hash(asset)
```

选择 partition policy。

Daily大因子库通常：

```text
year/month
```

比单 year更适合频繁增量。


## 57. Materializer 的 read-concat-rewrite 需要注意大分区成本

当前本地 Parquet upsert思路是：

```text
old + new
dedup
sort
replace partition
```

语义简单可靠，但大 partition频繁更新时 IO放大。

生产推荐：

```text
DataAccess staging
→ transactional publish
→ manifest
```

本地 factor lake主要：

```text
research/dev
```

生产已有禁止 direct-local，这是正确方向。

继续让：

```text
write target
snapshot manifest
watermark
catalog
```

统一事务。


## 58. Result dtype不要为了省盘默认牺牲因子排序

当前 production物化已倾向 float64，这是正确方向。

只有经过：

```text
rank stability
IC stability
portfolio turnover stability
```

量化证明的 factor才允许 float32。

Quantization certificate进入 factor identity。


## 59. Batch result streaming 应成为默认

对于上万因子：

```text
results: dict[name -> full Series]
```

会线性吃内存。

默认提供：

```text
result_policy = stream / materialize / callback
```

生产：

```text
root完成
→ DQ
→ write/sink
→ release
```

只有交互式 research才：

```text
return all results。
```


## 60. Main API 不要再维护 serial/parallel 两套执行语义

最终：

```python
engine.run_many(..., execution="auto")
```

Scheduler根据：

```text
factor count
cost
resources
backend
```

自动选择。

`run_many_parallel` 只保留 compatibility alias。

否则：

```text
serial path
parallel path
scheduler path
```

三套容易语义漂移。


## 61. Data representation boundary要显式成为 physical edge

定义：

```text
ARROW_LONG
POLARS_LONG
POLARS_WIDE
PANDAS_WIDE
PANDAS_SERIES_MULTIINDEX
DUCKDB_RELATION
CLICKHOUSE_RELATION
```

每个 task声明 input/output representation。

Planner明确插入：

```text
ConversionTask
```

这样：

```text
转换次数
转换bytes
转换cost
```

都能观察和优化。


## 62. “零拷贝”必须有证据，不能靠名义

Arrow → Polars很多情况可 zero-copy。

Arrow → Pandas不一定。

Pandas unstack通常复制。

因此 telemetry记录：

```text
conversion_type
input_bytes
output_bytes
elapsed_ms
zero_copy_claim
```

benchmark验证。


## 63. Pandas reference backend也能继续快很多

不必放弃 Pandas。

### 建议

- `KernelRegistry` 做 process-wide immutable singleton；
- Engine实例不重复注册1362 kernels；
- 把递归 `_eval` 编译成 topological execution tape；
- kernel lookup在 compile阶段绑定；
- 常见 rolling用 bottleneck/Numba；
- multi-output/fused rolling；
- contiguous NumPy arrays；
- column prefetch；
- 避免反复 MultiIndex构造；
- 仅最终输出恢复 labeled Series。


## 64. Numba 应作为 Pandas reference 的“kernel acceleration tier”，不是第四 DSL backend

适合：

```text
rolling custom loops
state machines
path-dependent kernels
technical recursive loops
```

不需要暴露：

```text
backend=numba
```

给用户。

PhysicalPlanner可以选择：

```text
pandas_numpy kernel variant
numba kernel variant
```

它们属于同一 reference semantic backend。


## 65. TA-Lib / scipy 也应视作 kernel provider

如果 TA-Lib已安装且：

```text
语义与 canonical exact parity
```

可选 C implementation。

但必须：

```text
parity evidence
version binding
missing/warmup parity
```

否则不能“装了库就自动换结果”。


## 66. Modin 建议重新评估是否值得继续支持

FactorEngine已经有自己的：

```text
batch scheduler
process pool
memory governor
```

Modin/Ray又引入一层分布式 scheduler。

容易：

```text
oversubscription
object store memory
debug复杂
```

如果没有明确 benchmark收益：

```text
deprecate pandas_modin backend
```

比维护第五套性能路径更合理。


## 67. Service 外层线程池会与内部 scheduler 叠加

HTTP service有独立：

```text
ThreadPoolExecutor(max_workers=N)
```

每个 job内部 FactorEngine又可能开：

```text
threads/processes/DuckDB threads/Polars threads
```

需要 host-global：

```text
ResourceBroker
```

把：

```text
job admission
task admission
```

统一。

否则 4个HTTP job × 16内部workers：

```text
64 runnable
```

很容易过载。


## 68. 多任务公平调度

服务场景：

```text
1个超大10000 factor job
```

不能饿死：

```text
小的交互 validate/compute job。
```

scheduler加入：

```text
job_id
tenant
priority
quota
weighted fair scheduling
```

资源 token按 job隔离。


## 69. Cancellation 要一路传到底层

用户取消 HTTP job：

```text
queue status cancelled
```

还不够。

需要：

```text
Scheduler CancellationToken
→ pending task不再admit
→ DuckDB interrupt
→ remote query cancel
→ Polars collect best-effort
→ process worker cooperative cancel
→ sink abort
```

已写出的 partition必须按事务语义处理。


## 70. Timeouts 也必须 task-aware

总 job deadline 与 task deadline区分。

避免：

```text
一个 SQL query卡30分钟
整个worker占着资源
```

QueryBudget + scheduler deadline统一。


## 71. Error taxonomy 应跨所有层统一

不要每层重新包装字符串。

统一异常 families：

```text
DSL
TYPE
SEMANTIC
PIT
SOURCE
SNAPSHOT
DQ
BACKEND_UNSUPPORTED
BACKEND_RUNTIME
RESOURCE
OOM
TIMEOUT
CANCELLED
MATERIALIZE
INTERNAL_BUG
```

每个错误：

```text
stable code
factor
task
canonical
backend
source
execution_id
```

服务只返回 sanitized message。


## 72. Fallback 必须显式且可计费

Production fallback不是简单：

```text
SQL失败 → Pandas继续。
```

每个 fallback有：

```text
from
to
reason
semantic_allowed
performance_cost
```

高价值指标：

```text
fallback_rate
fallback_bytes
fallback_time
```

如果某 backend已认证却频繁 fallback：

```text
CI/production alarm。
```


## 73. Backend circuit breaker

若一个 backend在当前运行连续：

```text
N 次 engine crash / compiler crash
```

暂时从候选中移除。

例如 DuckDB connection损坏：

```text
recreate connection
```

而不是每个 factor都先失败一次再fallback。


## 74. DuckDB connection / thread model要统一

生产建议：

```text
process-local shared DuckDB engine
```

但 query应：

```text
connection/session scoped
```

避免不同 task修改：

```text
PRAGMA threads
temp_directory
memory_limit
```

互相污染。

ResourceScope通过显式 connection setting，不靠全局环境。


## 75. ClickHouse pushdown应该批量而不是细碎 query

远程 ClickHouse最大的敌人通常不是CPU，是：

```text
network round trip
query startup
```

所以：

```text
multi-factor SELECT
large projection
server-side window/group
```

优于：

```text
每个factor一个query。
```

CostModel加入：

```text
network RTT
selected bytes
remote result bytes
```


## 76. Read-wave planner要真正由 ScanCost驱动

DataAccessSource已经有：

```text
estimate_scan_cost()
```

但主 batch path没有真正走 AdaptiveBatchScheduler/read waves。

统一后：

```text
ScanCost.selected_bytes
projection_bytes
rows
files
remote
selectivity
```

进入：

```text
read wave
IO token
memory admission
backend selection
```


## 77. Prefetch 不能无脑

只 prefetch：

```text
critical-path soon
high reuse
small enough
```

压力上升：

```text
PRESSURE_1
```

首先停止 speculative prefetch。

这应有真实 action，不只是 pressure label。


## 78. Pressure stages要真正绑定动作

例如：

```text
NORMAL:
  normal admission + prefetch

P1:
  stop speculative prefetch

P2:
  evict low-value cache

P3:
  no new memory-heavy task
  spill eligible intermediates

P4:
  shrink concurrency

CRITICAL:
  only running completion + sink
```

每个 stage写集成测试。


## 79. Physical sharding 要按语义

可 shard：

```text
纯时序、股票独立
→ asset shard

独立交易日截面
→ time/date shard

group独立
→ group shard

minute session独立
→ session shard
```

不可乱 shard：

```text
cross-sectional按asset切
recursive state按time无checkpoint切
global model按asset切
```

ShardSpec由 operator execution contract推导。


## 80. Shard size需要 adaptive

目标：

```text
predicted peak < memory target
```

同时不能过小造成 overhead。

根据校准：

```text
rows/sec
bytes/sec
startup cost
```

动态调整。


## 81. Stateful operator checkpoint要从“支持”升级为普遍可用

对：

```text
EMA/Wilder
state machine
episode
```

如果能安全 checkpoint：

```text
每天增量只恢复状态
```

无需 full replay。

Checkpoint包括：

```text
schema_version
operator semantic digest
source snapshot
last timestamp
state payload
```

任何 digest变化：

```text
checkpoint invalid。
```


## 82. Stateful checkpoint不能只存数值状态

还应存：

```text
missing/gap state
warmup counters
left/right censor
session state
```

否则 restart与单次全历史运行不等价。


## 83. Cross-backend checkpoint通常不应该共享

Pandas/Numba与某 native backend若内部状态定义不同：

```text
checkpoint backend-specific。
```

只有经过证明的 portable checkpoint schema才能跨 backend。


## 84. Incremental correctness必须做 full-vs-incremental parity

所有宣称 incremental-safe operator：

```text
full_history_run
==
partitioned incremental run
==
restart from checkpoint run
```

在：

```text
normal
gap
source revision
corporate action
universe change
```

上逐项比较。


## 85. 编译器的 semantic rewrite要有 rewrite proof

Optimizer每个 rewrite rule声明：

```text
preconditions
semantic dimensions preserved
NaN behavior
Inf behavior
unit
availability
history
```

生产 rewrite只有：

```text
proof predicate == true
```

才触发。

避免：

```text
x*0 → 0
```

在 x=NaN时改变语义。


## 86. CSE key要区分结构相同但 execution scope不同

当前已经有 scope namespace，这是正确方向。

继续确保 key覆盖：

```text
market
universe
calendar
source snapshot
decision policy
data vintage
corporate action
```

R30 semantic identity完成后让 CSE直接消费它。


## 87. CSE 不是越多越好

提取一个：

```text
便宜且只复用2次
```

的小节点并 materialize可能更慢。

Cost-based CSE：

```text
benefit =
(recompute_cost * (reuse-1))
- materialize_cost
- memory_cost
```

只有 benefit>threshold才提取。


## 88. Rolling CSE也要 parameter-aware

例如：

```text
ts_mean(x,20)
ts_std(x,20)
```

可 fused，

但：

```text
ts_mean(x,20)
ts_mean(x,21)
```

不能只因为结构相似就制造昂贵共享。

可以做：

```text
prefix rolling state reuse
```

但需专门算法。


## 89. Cold-start/LLM mining 与 runtime capability要一个 manifest

给 AlphaProbe/AlphaMiner 等输出：

```text
MiningOperatorManifest
```

只包含：

```text
production-visible
role legal
input semantic legal
param search grade
backend/cost class
history cost
```

LLM不应该自己读1362个raw operator猜哪些能用。


## 90. Mining grammar应感知成本

当生成 factor时控制：

```text
max AST nodes
max estimated runtime
max memory
max high-cost operators
max source joins
```

这样不会先生成大量：

```text
理论能算但批量跑不起
```

的表达式。


## 91. Mining阶段先做 semantic/cost dedupe，再执行

候选公式：

```text
canonical AST
semantic equivalence
param equivalence
CSE fingerprint
```

先去重。

再做：

```text
static cost
data availability
backend availability
```

最后才真正跑数。


## 92. 因子批量执行可以按 locality，而不是只按 factor name

分组依据：

```text
same source
same fields
same window family
same backend
same representation
same universe
```

让 cache locality/scan reuse最大。

当前 dependency_graph已经正确把“列重叠”从 dependency改成 locality hint，这个方向应保留。


## 93. Warmup clustering还可以更细

当前按 lookback聚类已经比全批最大lookback好。

进一步按：

```text
source
bar frequency
history kind
stateful/full history
```

分 wave。

一个 full-history stateful factor不应把1000个20日 rolling factor拖成全历史读取。


## 94. Minutely + daily mixed batch不要强行一个 execution wave

minute→daily和pure daily：

```text
source bytes
representation
session semantics
```

完全不同。

PhysicalPlanner分 source wave，最终在 daily boundary汇合。


## 95. DataAccess 读因子本身可以成为 FE input

DataAccess已有 `read_factors()`。

FactorEngine应正式支持：

```text
MaterializedFactorRef
```

作为 typed input。

这样：

```text
二阶因子 / stacking / model
```

可直接读已有 factor lake，不绕 raw parquet。


## 96. Factor dependency graph要支持“因子依赖因子”

当前跨 factor真实依赖较弱。

未来支持：

```text
FactorRef("mom20")
```

后形成：

```text
materialized dependency
or
inline dependency
```

Planner决定：

```text
read materialized
vs recompute
```

基于 freshness/cost。


## 97. Materialized-vs-recompute cost

如果某中间因子：

```text
已有同 snapshot物化
```

且读取成本 < recompute：

```text
读取。
```

否则 inline计算。

类似 database：

```text
materialized view substitution。
```


## 98. 数据质量 DQ要在正确层执行

输入 source DQ：

```text
DataAccess层
```

semantic field/unit/PIT：

```text
typing/source contract层
```

factor output DQ：

```text
root/sink层
```

不要每个 root重复全输入 DQ。


## 99. Output DQ 可以用于自动隔离坏算子

生产运行发现某 canonical频繁：

```text
99.9% NaN
Inf
constant
异常跳变
```

不是立即删除，但进入：

```text
runtime health quarantine
```

阻止自动 mining直到调查。


## 100. Observability：每个执行都要能 explain “为什么这么跑”

`explain_physical_plan()` 应输出：

```text
source scans
backend partitions
conversion edges
fusion groups
CSE
shards
memory estimates
CPU tokens
SQL pushdown
Polars native
fallback reason
write strategy
```

用户能看到：

```text
为什么这个 factor走Pandas而不是Polars。
```


## 101. 必须建立 performance trace

每个 task记录：

```text
queue wait
admission wait
scan time
compute time
conversion time
sink time
rows
bytes
peak memory
backend
thread/process
cache hit
spill
```

然后 flame-like timeline。


## 102. 核心性能 KPI

至少持续跟踪：

```text
compile formulas/sec
factor roots/sec
rows/sec
input GB/sec
output MB/sec
peak RSS
CPU utilization
IO utilization
cache hit bytes
CSE saved work
SQL pushed node ratio
Polars-native node ratio
backend transition count
conversion bytes
fallback rate
incremental recompute ratio
```

不要只测：

```text
总耗时。
```


## 103. 转换成本应成为第一等 KPI

尤其记录：

```text
Pandas→Polars bytes
Polars→Pandas bytes
Arrow→Pandas bytes
long→wide bytes
wide→long bytes
SQL→client bytes
```

目标不是“Polars占比越高越好”，而是：

```text
total conversion bytes最小。
```


## 104. Backend transition hard metric

一个 factor root理想：

```text
0~1次主要 representation transition。
```

如果：

```text
SQL → Pandas → Polars → Pandas
```

即使各 operator单独快，整体也可能慢。

CI性能测试直接记录：

```text
backend_transition_count。
```


## 105. Benchmark suite要分 workload，而不是单 operator

至少：

```text
micro:
  单算子

meso:
  10~30 node真实factor

macro:
  100/1000/10000 factors batch

incremental:
  daily update

minute:
  minute→daily

fundamental:
  multi-source PIT join
```

单算子 benchmark不能指导整个 DAG router。


## 106. 真实 benchmark corpus

建立固定因子集合：

```text
simple price-volume
rolling stats
technical
cross-sectional/group
fundamental
minute aggregation
mixed source
stateful
high-cost research
```

每次 PR自动跑小版；

nightly跑大版。


## 107. Performance regression CI

阈值不要太死，使用：

```text
median
p95
relative baseline
```

检测：

```text
runtime +20%
memory +20%
conversion bytes突增
fallback突增
```

再人工确认。


## 108. SQL / Polars parity与性能认证分开

先：

```text
correctness parity
```

再：

```text
performance certification
```

一个 backend虽然正确但比 Pandas慢：

```text
可以 implemented/parity
但 auto router不选。
```

不要为了覆盖率强制 production auto。


## 109. 三后端 matrix 建议自动生成

生成：

```text
R31_BACKEND_TARGET_MATRIX.csv
```

字段：

```text
canonical
family
surface
pandas_impl
pandas_prod
polars_impl
polars_parity
polars_prod
polars_long_native
duckdb_emitter
duckdb_parity
duckdb_prod
clickhouse_parity
recommended_target
reason
priority
benchmark_class
```

不要人工维护“哪些该三后端”名单。


## 110. SQL emitter存在但不该 production 的算子不要认证

例如 R30要求删除/隔离的：

```text
unsafe
random/future
research-only
```

即使 SQL emitter存在：

```text
也不进入 production certification。
```

先做 R29/R30 disposition，再做 backend target。


## 111. Backend coverage目标按价值而不是数量

建议优先覆盖：

```text
80%+ 实际 cold-start/mining expression 节点
```

而不是：

```text
80% canonical count。
```

通过真实挖掘日志统计：

```text
operator usage frequency
runtime share
```

挑 top operators。


## 112. Profile-guided backend development

每周统计：

```text
top CPU operators
top conversion offenders
top pandas fallback operators
top unsupported SQL operators
```

下一批开发按真实瓶颈。

不要凭感觉移植。


## 113. Backend development顺序建议

### 第一批

```text
elementwise + rolling core + cs/group core
```

### 第二批

```text
price-volume + common technical + regression
```

### 第三批

```text
fundamental simple + intraday aggregation
```

### 第四批

只处理：

```text
真实 profile证明很贵且可 native 的特殊 operator。
```


## 114. 当前 README / docs 数量已经漂移

README仍能看到历史数字：

```text
canonical 626
daily 223
extended 920
SQL 147
```

而当前 R30 catalog是：

```text
1362 canonical
daily 1185
...
```

SQL generated coverage又是：

```text
354 emitters。
```

说明文档不是 single source of truth。

### 修复

README中的数字全部：

```text
CI生成
```

或者：

```text
不写固定数字，只链接 generated artifact。
```


## 115. 文档不能继续描述 joblib 如果实现已切 concurrent.futures

当前 README仍有旧的：

```text
joblib root parallel
shared serial
```

描述，而当前代码已经改过。

对新人非常危险。

建立：

```text
docs generated from code/evidence
```

并在 CI做 doc drift test。


## 116. Packaging dependencies需要与实际 API版本对齐

当前 FE `full` extra允许：

```text
data-access>=0.3.1
```

但当前 DataAccess README 已是：

```text
0.5.0
```

且 FE使用多项 0.5-era高级 API。

应把最低版本改成：

```text
实际 CI验证过的最小兼容版本
```

并建立：

```text
FactorEngine × DataAccess compatibility matrix。
```


## 117. Polars/DuckDB 等生产环境建议锁 tested minor family

backend evidence与：

```text
Polars/DuckDB/Pandas/NumPy
```

版本强相关。

生产 deployment使用：

```text
constraints.txt / uv.lock
```

锁定。

新 minor版本：

```text
先 parity + performance再升级。
```


## 118. `parallel=joblib` extra 若已不再使用就删除

当前核心 parallel路径已经主要依赖：

```text
concurrent.futures
```

如果仓库无真实 joblib production依赖：

```text
删除 optional dependency
更新 README。
```

减少无效依赖。


## 119. 建议增加 psutil/threadpoolctl 到 performance runtime

ResourceBroker现在已有 psutil optional使用。

如果生产资源治理依赖它：

```text
应成为 performance/full明确依赖。
```

`threadpoolctl` 用于：

```text
BLAS/OpenMP运行时线程限制
```

比仅设置环境变量可靠。


## 120. Kernel thread oversubscription必须统一控制

外层：

```text
N workers
```

内层：

```text
MKL
OpenBLAS
NumExpr
Polars
DuckDB
```

总 runnable必须受：

```text
CPU token budget。
```

例如：

```text
4 DuckDB tasks × 8 threads
```

不能在16核机上同时开。


## 121. Runtime thread设置不要依赖“导入后改 env”

某些 BLAS库在 import时已读取线程配置。

用：

```text
threadpoolctl
DuckDB connection SET threads
Polars process config（在worker启动前）
```

配合 environment。


## 122. Engine instance构建成本可以降低

`run_many_from_config` 当前会为每个 YAML：

```text
from_config
→ engine/backend/data_source
```

即使最终同 data scope再合并。

优化：

```text
先 parse configs
→ group by engine/data source/backend config
→ 每组创建一个 engine
```

而不是先创建 N个再只取第一组 engine执行。


## 123. Backend singleton / shared immutable registry

以下可以 process-wide复用：

```text
OperatorRegistry frozen snapshot
KernelRegistry
compiled emitter registry
static capability maps
```

避免每个 Engine重新构造。


## 124. DataSource session生命周期要 batch级

同一个 data scope的1000 factor：

```text
one read session
one snapshot
one cache
```

不要每 factor自己 refresh snapshot /新 handle。

执行结束统一 close。


## 125. Query-scoped snapshot必须固定整个 batch

开始 batch：

```text
resolve snapshot token S
```

所有 scan：

```text
S
```

结束前再validate。

不能：

```text
factor A读S1
factor B读S2
```

然后还说是同一次 batch。


## 126. Batch lineage记录 source snapshot set

多源：

```text
price snapshot
fundamental snapshot
universe snapshot
benchmark snapshot
```

都记录。

不是一个字符串 `data_snapshot_id` 就结束。


## 127. DataAccess source cost不要 catch-all 后静默 None

`estimate_scan_cost`失败 research可以 fallback。

Production planner若依赖 cost做 resource admission：

```text
应该区分
COST_UNAVAILABLE
vs
真实0成本。
```

高风险远程/大数据源如果完全无法估成本：

```text
保守 admission。
```


## 128. ScanCost要进入 source task，而不是 root plan猜 rows

当前 plan router `estimate_plan_rows()` 有：

```text
默认3000 instruments
business days
500k fallback
```

对：

```text
minute
event
irregular panel
```

可能非常不准。

SourceScanTask直接使用：

```text
DataAccess ScanCost.estimated_rows / bytes
```

再向上估 shape。


## 129. Row count之外还要 cardinality

CostContext扩：

```text
rows
unique_dates
unique_instruments
groups
avg group size
session slots
missing density
```

因为：

```text
cs_rank
group_rank
rolling
```

成本不只由总 rows决定。


## 130. Cardinality statistics 可以来自 manifest

DataAccess manifest增加：

```text
row count
date count
symbol count
null rates
min/max
partition stats
```

planner无需真正扫描就能估。


## 131. Statistics catalog

类似数据库 optimizer：

```text
DatasetStats
FieldStats
```

包括：

```text
density
null rate
distinct count
sortedness
```

backend cost更准确。


## 132. Physical planner可做 adaptive replan

实际 scan后发现：

```text
rows远超估计
```

在尚未运行后续 expensive tasks前：

```text
重新评估shard/backend/concurrency。
```

不要整个 plan从头固定死。


## 133. 失败后的 fallback也应该触发 cost feedback

例如：

```text
Polars OOM
```

记录：

```text
shape/backend failure envelope
```

下次类似 shape直接避开，不每次重试。


## 134. 编译期检查 backend viability

Mining候选如果：

```text
只有高成本 Pandas
且预计超过 budget
```

可提前：

```text
reject / defer
```

不需要读完数据再失败。


## 135. 多后端 semantic parity contract必须共用一个 OperatorContract

Pandas/Polars/SQL不能各自声明：

```text
param specs
units
missing policy
```

同一 canonical有且只有一个 logical contract。

backend只声明：

```text
physical capability
numeric implementation policy
```

当前 registry已经在往这个方向做，继续彻底化。


## 136. Backend-specific floating tolerance要版本化

Parity不能所有 operator统一：

```text
atol=1e-8
```

对于：

```text
returns
probability
large money
regression
```

尺度不同。

OperatorContract声明：

```text
absolute/relative tolerance
or reference property
```

但生产输出仍应尽量 exact semantic parity。


## 137. SQL NULL vs NaN特别需要一套转换层

DuckDB/ClickHouse与Pandas/Polars：

```text
NULL
NaN
Inf
```

不同。

建立：

```text
BackendNullSemanticsAdapter
```

并在 emitter boundary统一。

否则 parity会出现大量“数值一样但mask不同”。


## 138. Sort/tie semantics在三个后端必须固定

重点：

```text
rank
quantile
argmax/argmin
top-k
winsorize
```

必须声明：

```text
ties method
stable ordering
NaN placement
```

SQL emitter不能依赖数据库默认排序。


## 139. Window frame要显式

SQL rolling必须：

```text
ROWS BETWEEN N PRECEDING AND CURRENT ROW
```

等明确 frame。

不能用数据库默认：

```text
RANGE
```

导致重复 timestamp时语义变化。


## 140. Minute数据有重复 timestamp时 backend parity

统一 identity：

```text
(timestamp, instrument, session_slot)
```

必要时增加 source sequence key。

不允许 Pandas按row order、SQL按RANGE、Polars按sort后各算不同结果。


## 141. Production backend fallback必须保护“认证过的快路径”

如果：

```text
某 root全部是 Polars certified
```

却 runtime fallback Pandas：

```text
必须报警/失败
```

当前已有一部分 gate，继续推广到：

```text
SQL
multi-root fusion
scheduler
```


## 142. 部分 SQL pushdown失败不要悄悄制造大量小 Python subtrees

记录：

```text
number of SQL subtrees
number of fallback subtrees
query count
```

若：

```text
query_count过大
```

可能比整 root Pandas更慢。

Router需要比较：

```text
partial pushdown plan
vs
pure Polars/Pandas plan。
```


## 143. SQL subtree抽取要 cost-based

不是：

```text
能下推就一定下推。
```

一个很小子树：

```text
SQL执行 + materialize + transfer
```

可能更慢。

Physical optimizer应只抽：

```text
beneficial subtree。
```


## 144. DataAccess `sql_relation()` 与 FactorEngine SQL compiler可以融合

理想路径：

```text
DataAccess scan relation
+ FE SQL expression
```

组合成一条 query。

避免：

```text
DataAccess先collect Arrow
→ FE再注册 DuckDB relation
→ 第二次query。
```


## 145. Scan + factor fusion 是极高价值方向

尤其：

```text
simple daily factors
```

可以：

```sql
SELECT
  date,
  symbol,
  ...
  rolling_expr AS factor
FROM parquet_scan(...)
WHERE ...
```

直接输出 factor long table。

这样：

```text
不进Pandas
不进Polars
不构建wide panel
```


## 146. 但 SQL pushdown 不能损害 CSE

多因子同 source：

```text
一次 multi-root SQL
```

而不是：

```text
100条完全独立SQL。
```

CSE / fusion / pushdown应统一规划。


## 147. Factor output matrix writer

大量因子同批：

```text
date × symbol × factor
```

不要每 factor一列一文件反复 open/close。

支持：

```text
matrix block writer
```

例如 32/64 factors一块，再由 factor catalog索引。

是否采用需与读取模式平衡。


## 148. 因子存储 layout需要显式 benchmark

比较：

```text
one-factor long parquet
multi-factor wide parquet
columnar factor blocks
ClickHouse long
```

对你的真实下游：

```text
模型训练
单因子研究
批量读取
```

分别 benchmark后决定。


## 149. FactorCatalog不要成为写锁瓶颈

1000 factor并行物化时 SQLite catalog可能序列化。

建议：

```text
write manifest batch
single catalog commit thread
```

避免每 factor一事务。


## 150. Atomic publication应该批级

一批 factor：

```text
all staging done
validation done
→ publish generation
```

如果中途失败：

```text
旧 generation仍可读。
```

而不是用户看到半批新、半批旧。


## 151. Run Generation

每次生产 batch生成：

```text
run_generation
```

所有：

```text
factor partitions
catalog
watermark
snapshot
```

绑定它。

模型训练可以指定：

```text
generation=N
```

保证可复现。


## 152. Observability的数据不要无限增长在 runtime_stats dict

大量 event若塞内存 dict：

```text
长 batch可能膨胀。
```

采用：

```text
bounded counters
ring buffer
structured trace sink
```

详细事件流外部写日志/OTel。


## 153. OpenTelemetry 可考虑

统一：

```text
trace_id
execution_id
job_id
factor
task
backend
```

后续很方便做：

```text
Grafana/Tempo/Prometheus。
```

不是必须，但生产成熟度高时值得。


## 154. Log不要在热循环输出

operator cell / row级：

```text
严禁 logging。
```

task级：

```text
sampled。
```

否则 IO锁会吃掉性能。


## 155. Service cost estimate不能只靠 formula length

HTTP admission最好直接调用：

```text
Analyzer + CostModel + DataAccess ScanCost
```

得到：

```text
expected rows
memory
CPU
IO
```

而不是只用 AST节点/字符串近似。


## 156. Multi-tenant安全

如果以后团队多人共用：

```text
最大并发
最大内存
最大scan bytes
最大output bytes
```

按 user/job设 quota。

避免一个错误 formula把整台机吃满。


## 157. Research 与 production可以共用编译器，但不能共用无界资源策略

Research允许：

```text
fallback
experimental operator
```

但仍需要资源 budget。

“research”不能等于：

```text
无限内存/无限线程。
```


## 158. CI分层

建议：

### PR fast

```text
unit
contract
small parity
small scheduler
compile smoke
```

### PR backend

```text
Pandas↔Polars core
Pandas↔DuckDB promoted set
```

### nightly

```text
full canonical
large panel
performance
incremental
multi-process
failure injection
```

### release

```text
all hard gates
evidence regeneration
docs regeneration
package compatibility
```


## 159. Failure injection tests

除了普通 pytest，主动模拟：

```text
worker crash
DuckDB interrupt
disk full
spill full
snapshot changes mid-run
cache checksum corrupt
network timeout
ClickHouse unavailable
process OOM
cancellation
SIGTERM
```

验证：

```text
资源释放
事务不半提交
重启可恢复。
```


## 160. Scheduler stress test

随机生成：

```text
1000 tasks
random DAG
random resource contracts
random failures
```

检查不变量：

```text
no deadlock
no leaked tokens
no task executes before predecessors
no duplicate commit
no missing release
```


## 161. Deterministic scheduling

相同：

```text
plan
resources
inputs
```

优先保证：

```text
结果完全确定。
```

执行顺序可以变，但不能影响 floating semantic。

对非 associative reduction：

```text
backend definition固定。
```


## 162. Backend fallback concurrency tests

同时10个 root：

```text
Polars成功
SQL部分失败
Pandas fallback
```

检查：

```text
runtime_stats隔离
source状态不污染
cache不竞态
```

当前代码已做一些 thread-local，继续覆盖所有新 scheduler路径。


## 163. Source object不要由 backend临时 mutate

Polars lazy目前仍需要 enable/restore source模式。

长期改成：

```text
ExecutionReadOptions
```

不可变传入：

```text
read_mode
lazy
representation
snapshot
```

DataSource本身保持 immutable/thread-safe。


## 164. Context也应 immutable + scoped mutable state

ExecutionContext分：

```text
ImmutableExecutionConfig
MutableExecutionState
```

避免 dataclass replace复制大 dict，也减少线程共享疑惑。


## 165. Runtime state store

共享 mutable state集中：

```text
ExecutionState:
  caches
  metrics
  cancellation
  snapshot
  leases
```

不要散落：

```text
ctx._xxx
runtime_stats
data_source attrs
global env。
```


## 166. 环境变量不能成为主要 runtime configuration bus

环境变量适合：

```text
deployment defaults
```

不适合：

```text
每 task路由。
```

所有：

```text
Polars expr
thread count
fallback
```

进入 typed `PerfConfig/ExecutionPolicy`。

环境变量只在启动时解析一次。


## 167. Config schema也要版本化

YAML：

```text
config_schema_version
```

旧字段自动 migration 或拒绝。

避免几年后：

```text
同一个yaml在新引擎含义变化。
```


## 168. Factor expression dialect/version

已经有 dialect/version基础，继续让：

```text
operator alias
parameter default
semantic version
```

稳定。

LLM生成器绑定：

```text
dialect_version。
```


## 169. Public API缩小

对普通用户：

```text
FactorEngine
Factor
api operators
validate
materialize
```

足够。

不要让用户依赖：

```text
OperatorRegistry._operators
internal physical classes
```

内部接口加 `_` / private package。


## 170. Compatibility aliases要有生命周期

所有 alias：

```text
introduced
deprecated
remove_after
replacement
```

自动 warn。

过期后 tombstone。


## 171. Registry initialization成本/风险

1362 operators import-time注册越来越重。

可考虑 generated manifest：

```text
metadata registry
```

与：

```text
lazy implementation import
```

分开。

编译/validation只需要 metadata，不必 import 所有 scipy/model modules。


## 172. Lazy operator implementation loading

Production manifest预生成：

```text
canonical → implementation module
```

真正 plan用到时 import。

收益：

```text
CLI启动
service worker启动
validation
```

更快，也减少 optional dependency import失败。


## 173. 但 registry manifest必须由 CI生成，不人工编辑

源代码 operator定义是 authority。

CI：

```text
load all
validate
export frozen manifest
```

runtime读取 manifest。

版本不匹配：

```text
fail/rebuild。
```


## 174. Optional dependencies要隔离

Research model/spectral package缺少依赖：

```text
不能影响 production core import。
```

R29 Research loader拆分后自然解决。


## 175. Python version建议收窄生产支持矩阵

当前 >=3.10。

如果 CI实际只稳定跑：

```text
3.11/3.12
```

生产只认证这两个。

不要声称所有未来Python版本自然兼容。


## 176. Cross-platform

多进程、resource、cgroup明显偏 Linux。

明确：

```text
Linux production certified
macOS developer
Windows maybe research
```

不要让平台差异偷偷走另一套语义。


## 177. Benchmark artifacts绑定 git SHA

每次 performance baseline：

```text
git_sha
operator digest
backend versions
hardware
data snapshot
```

全部记录。

否则数字不可比较。


## 178. 自动生成一份 Backend Readiness Scorecard

每 release输出：

```text
Pandas production coverage
Polars implemented/parity/prod
Polars long native
DuckDB emitter/parity/prod
ClickHouse parity/prod
fallback rates
```

并按：

```text
actual usage-weighted coverage
```

排序。


## 179. 自动生成 Performance Opportunity Report

从 nightly/profile自动列：

```text
Top 20 CPU nodes
Top 20 memory nodes
Top 20 conversion edges
Top 20 unsupported native operators
Top 20 repeated scans
Top 20 CSE candidates
```

让后续优化有依据。


## 180. 这轮建议新增的 current-HEAD issue 清单

以下是基于 `58490a...` 当前代码确认或高度明确的系统项：

```text
R31-P0-001  Production run_many仍未以AdaptiveBatchScheduler为默认主执行链
R31-P0-002  Adaptive scheduler physical DAG当前主要只有CSE/root tasks
R31-P0-003  Scheduler tasks backend_candidates为空、preferred_backend固定Pandas
R31-P0-004  Scheduler default resource contract错误硬编码Pandas/GIL属性
R31-P0-005  Scheduler future异常路径task_id绑定不可靠
R31-P0-006  Scheduler retry/failure存在ResourceBroker reservation泄漏风险
R31-P0-007  HybridExecutor process payload含closure/context/backend，真实多进程可序列化性不足
R31-P0-008  Process initializer使用nested function，spawn平台有风险
R31-P0-009  ResourceBroker.can_admit具有token acquisition副作用
R31-P0-010  ResourceBroker external CPU估计把本进程负载算作外部负载
R31-P0-011  ResourceBroker spill reserve fraction使用RAM而非spill disk容量
R31-P0-012  ResourceBroker disk_busy恒为0，IO pressure不可见
R31-P0-013  Whole-plan cost router按去重canonical估价，重复节点被漏算
R31-P0-014  Whole-plan backend routing丢失bound params/window/feature_dim
R31-P0-015  Mixed backend cost不是基于真实DAG topology
R31-P0-016  Polars conversion penalty存在重复计费风险
R31-P0-017  Memory route使用backend-independent粗估，可能误杀streaming backend
R31-P0-018  Measured cost baseline兼容检查未充分校验已记录hardware fingerprint
R31-P0-019  Native fusion scope校验未完整包含execution_scope
R31-P0-020  Scheduler的source_scope/source_snapshot_id未真实填充
R31-P0-021  Adaptive fusion block存在但未成为实际默认group size逻辑
R31-P0-022  Native fusion planning与main batch execution没有形成统一执行链
R31-P0-023  SQL emitter已有354，但当前generated evidence DuckDB parity/prod均为0
R31-P0-024  SQL residual backend默认倾向Polars而非重新cost-route
R31-P0-025  DataAccess高级DataRequest/ReadPlan未成为FE batch数据计划主入口
R31-P0-026  DataAccess ScanCost虽有adapter API，但主batch scheduler链未真正消费
R31-P0-027  DataAccess catalog field resolution逐字段调用，可批量
R31-P0-028  Column cache + panel cache可能重复驻留同一字段
R31-P0-029  DataAccess long collect无manifest时snapshot revalidation异常存在fail-open路径
R31-P0-030  README/backend coverage数字与当前实现/evidence明显漂移
R31-P0-031  FE data-access最小依赖版本与当前使用API代际可能不一致
R31-P1-032  Optimizer仅轻量rewrite，缺projection/filter/fusion/physical partition
R31-P1-033  PandasBackend每实例建立KernelRegistry，可共享/预编译
R31-P1-034  Polars whole-expression fastpath仍偏env opt-in
R31-P1-035  Polars recursive Python dispatch仍限制大型表达式吞吐
R31-P1-036  SQL SourceRef resolver boundary后可进一步恢复pushdown
R31-P1-037  CSE尚未全面cost-based
R31-P1-038  Incremental仍可升级为source-change impact DAG
R31-P1-039  HTTP job并发与内部scheduler资源治理尚未统一
R31-P1-040  Job cancellation需要传播至底层执行
R31-P1-041  Resource sampling可改background single-pass
R31-P1-042  Modin支持应重新benchmark/可能删除
R31-P1-043  runtime configuration仍有环境变量控制路径，可进一步typed化
R31-P1-044  docs/dependencies应由generated truth治理
```


## 181. R31 三后端整改阶段

### Phase A：Backend Truth

```text
生成 current backend matrix
确认354 SQL emitter
确认真实Polars native coverage
删除R30已禁用算子
```

### Phase B：SQL Certification

```text
先50~100核心
再200+
再300+
```

### Phase C：Polars Native Expansion

重点：

```text
高频、可Expr、可window/group的算子。
```

### Phase D：Physical Subgraph Planner

```text
Pandas / Polars / SQL子图分区
conversion edge
```

### Phase E：Multi-root Fusion

```text
same source/scope
```

### Phase F：Cost Calibration

```text
真实benchmark + runtime feedback。
```


## 182. R31 性能整改阶段

### Phase 1

接线：

```text
AdaptiveBatchScheduler → production default
```

### Phase 2

真实：

```text
PhysicalFactorDAG lowering
```

### Phase 3

修：

```text
failure/retry/resource lease
```

### Phase 4

统一：

```text
thread/process/resource
```

### Phase 5

DataAccess：

```text
BatchDataRequest + ScanCost
```

### Phase 6

列式 representation：

```text
Arrow/Polars long first
```

### Phase 7

optimizer：

```text
fusion/projection/partition
```

### Phase 8

incremental：

```text
change impact DAG
```

### Phase 9

cache/spill/liveness

### Phase 10

service global resource scheduling


## 183. R31 必须生成的 artifacts

```text
factor_engine/docs/evidence/r31/

R31_CURRENT_ARCHITECTURE.json
R31_MAIN_EXECUTION_PATH_AUDIT.md

R31_BACKEND_TARGET_MATRIX.csv/json
R31_PANDAS_COVERAGE.json
R31_POLARS_COVERAGE.json
R31_DUCKDB_COVERAGE.json
R31_CLICKHOUSE_COVERAGE.json

R31_SQL_EMITTER_CERTIFICATION.csv
R31_TRIPLE_BACKEND_PARITY.csv

R31_PHYSICAL_DAG_AUDIT.json
R31_SCHEDULER_WIRING_AUDIT.json
R31_SCHEDULER_FAILURE_INJECTION.json
R31_RESOURCE_LEASE_AUDIT.json

R31_COST_MODEL_NODE_COVERAGE.csv
R31_BACKEND_ROUTING_ACCURACY.json
R31_CONVERSION_EDGE_REPORT.csv

R31_DATAACCESS_BATCH_PLAN_AUDIT.json
R31_SCAN_COST_INTEGRATION.json

R31_CACHE_LIVENESS_REPORT.json
R31_SPILL_AUDIT.json

R31_INCREMENTAL_CHANGE_IMPACT_AUDIT.json

R31_MULTIROOT_FUSION_REPORT.json
R31_MULTI_OUTPUT_KERNEL_MATRIX.csv

R31_PERFORMANCE_BASELINE.json
R31_PERFORMANCE_REGRESSION.json
R31_WORKLOAD_BENCHMARKS.csv

R31_DOC_DRIFT_AUDIT.json
R31_DEPENDENCY_COMPATIBILITY.json

R31_FINAL_ACCEPTANCE_REPORT.md
```


## 184. R31 Hard Gates

最终至少：

```text
R31_DEFAULT_BATCH_EXECUTOR_IS_ADAPTIVE_SCHEDULER

R31_PHYSICAL_DAG_HAS_REAL_SOURCE_OPERATOR_WRITE_STAGES
R31_PHYSICAL_TASK_BACKEND_CONTEXT_REAL
R31_PHYSICAL_TASK_RESOURCE_CONTRACT_REAL

R31_TASK_FAILURE_RESOURCE_LEAK_ZERO
R31_TASK_CANCEL_RESOURCE_LEAK_ZERO
R31_RETRY_RESOURCE_BALANCE_PASS
R31_PROCESS_POOL_SPAWN_SAFE
R31_BROKEN_WORKER_RECOVERY_PASS

R31_CAN_ADMIT_IS_PURE
R31_RESOURCE_EXTERNAL_CPU_CORRECT
R31_SPILL_CAPACITY_DIMENSION_CORRECT
R31_IO_PRESSURE_OBSERVABLE

R31_COST_COUNTS_EVERY_NODE_OCCURRENCE
R31_COST_USES_BOUND_PARAMS
R31_COST_IS_DAG_AWARE
R31_CONVERSION_COST_COUNTED_ONCE
R31_BACKEND_SPECIFIC_MEMORY_COST

R31_MEASURED_BASELINE_HARDWARE_BOUND

R31_NATIVE_FUSION_EXECUTION_SCOPE_SAFE
R31_NATIVE_FUSION_ACTUALLY_EXECUTED

R31_DATAACCESS_BATCH_REQUEST_ACTIVE
R31_SCAN_COST_USED_BY_SCHEDULER
R31_DUPLICATE_SOURCE_SCAN_MINIMIZED

R31_SQL_EMITTERS_CURRENT
R31_DUCKDB_CORE_PARITY_CERTIFIED
R31_CLICKHOUSE_SEPARATE_CERTIFIED
R31_POLARS_NATIVE_AUTO_PLANNED

R31_BACKEND_TRANSITION_TELEMETRY
R31_CONVERSION_BYTES_TELEMETRY

R31_CSE_COST_BASED
R31_CSE_LIVENESS_RELEASE_PASS
R31_SPILL_LIFECYCLE_PASS

R31_INCREMENTAL_FULL_PARITY_PASS
R31_CHANGE_IMPACT_RECOMPUTE_PASS

R31_BATCH_RESULT_STREAMING_DEFAULT_PRODUCTION

R31_SERVICE_AND_ENGINE_SHARE_RESOURCE_GOVERNANCE
R31_CANCELLATION_PROPAGATES_TO_EXECUTION

R31_DOCS_GENERATED_FROM_CURRENT_EVIDENCE
R31_PACKAGE_DEPENDENCIES_MATCH_TESTED_APIS

R31_PERFORMANCE_BASELINE_CURRENT_SHA
R31_NO_UNEXPLAINED_MAJOR_REGRESSION

R31_HARD_BLOCKERS_ZERO
```


## 185. 最推荐的最终架构

理想主链：

```text
Factor DSL / Generated Expressions
        ↓
Batch Parser + Typed Analyzer
        ↓
Shared Typed IR DAG
        ↓
Semantic Optimizer
        ↓
Logical DAG
        ↓
Batch DataRequest Planner
        ↓
DataAccess ReadPlan / Source Snapshot
        ↓
Physical Subgraph Optimizer
  ┌─────────────┬──────────────┬───────────────┐
  │ DuckDB/CH   │ Polars Lazy  │ Pandas/Numba  │
  │ SQL stages  │ Expr stages  │ special stage │
  └─────────────┴──────────────┴───────────────┘
        ↓
Real PhysicalFactorDAG
        ↓
AdaptiveBatchScheduler
  ResourceBroker
  HybridExecutor
  Read Waves
  Fusion
  Sharding
  Spill
        ↓
Streaming DQ / Sink
        ↓
Staging + Atomic Publish
        ↓
Factor Catalog / Lineage / Watermark
```

这时：

```text
BackendRouter不是一个孤立组件
Scheduler不是一个孤立组件
DataAccess不是一个按列读取工具
CSE不是一个后处理技巧
```

而是一个统一优化器的一部分。


## 186. 如果只按收益优先做，建议先做这 12 项

如果人力有限，按对吞吐/稳定性最可能产生大收益的顺序：

```text
1. 把 AdaptiveBatchScheduler 真正接成 production run_many 默认路径
2. 修 scheduler failure/task_id/resource lease
3. 真正 lower PhysicalFactorDAG，不只 CSE/root
4. 把 backend candidates/cost/resource contract 填进 physical tasks
5. 修 whole-plan cost：不去重节点、带 bound params、DAG-aware
6. DataAccess BatchDataRequest，一批因子一次规划scan/join
7. SQL 354 emitters先完成核心 parity certification
8. Polars certified Expr自动 physical planning，减少Python dispatch
9. Multi-output + rolling/group fusion
10. columnar representation统一，压低 Pandas/Polars/Arrow转换
11. change-impact incremental recompute
12. service/job与内部scheduler共用一个资源broker
```

这 12 项完成后，再去继续做边缘性能优化。


## 187. 不建议做的事情

以下不要为了“看起来高级”而做：

```text
给所有1362算子硬写SQL
给所有stateful算子硬写Polars
再引入Dask/Ray作为第4个scheduler
为了并行给每个factor开独立进程
每个operator都自己决定backend
每个模块都有自己的memory limit
用Modin同时再叠加FE process pool
为了CSE把所有共享节点都提前永久物化
所有因子默认全部return到内存
```

系统成熟的核心是：

```text
少 control plane
少 representation
少 conversion
少 scan
少重复计算
一个资源authority
一个physical planner
一个生产scheduler。
```


## 188. 最终定义：什么叫“很完善”

FactorEngine 真正成熟，不是：

```text
1362 operators
4个backend名字
20个performance模块。
```

而是：

```text
一个公式进来后，
系统能确定它的语义，
知道数据在哪里，
知道要读多少，
知道哪个子图在哪个backend最合适，
知道需要多少CPU/RAM/IO，
知道哪些结果能共享，
知道何时释放，
知道失败如何恢复，
知道每天哪些部分必须重算，
并且能证明最后结果与reference完全一致。
```

这是 R31 的完成标准。
