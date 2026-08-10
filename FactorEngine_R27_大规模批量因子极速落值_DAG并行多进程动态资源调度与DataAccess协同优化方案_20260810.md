# FactorEngine R27：大规模批量因子极速落值——DAG 并行、多进程、动态资源调度与 DataAccess 协同优化方案

> **用途**：直接交给代码 AI 执行。  
> **目标**：在不改变因子数值语义、PIT、安全边界和数据快照身份的前提下，把“一大批因子的计算 + 落值”做成一个真正的**资源感知、DAG 感知、数据扫描感知、自适应并行**执行系统。  
> **重点**：不是简单增加 `n_jobs`，而是让 FactorEngine 与 DataAccess 联合决定“哪些任务一起算、用线程还是进程、一次读多少数据、开多少 worker、每个 worker 分多少线程、何时 spill、何时降并发、何时立即落盘释放内存”。  
> **适用场景**：A 股/美股日频大批量因子、分钟→日频因子、基本面/事件/PIT 因子、自动因子挖掘后的批量落值、训练前 factor matrix 生产，以及不同 CPU/RAM 配置的服务器。  
> **非目标**：本轮不重新定义算子数学语义；R24/R25/R26 的 correctness/PIT/真实可用性要求继续作为硬前提。

---

# 0. 一句话目标

最终希望用户只需要调用类似：

```python
engine.materialize_many_fast(
    factors,
    mode="auto",
    coexist=True,
    target="factor_matrix",
)
```

FactorEngine 自动完成：

```text
编译全部因子
→ 构建真实多因子 DAG
→ CSE / rolling CSE / scan fusion
→ 估算每个节点 CPU / 内存 / I/O / 输出大小
→ 读取当前服务器真实资源余量
→ 自动选择线程 / 进程 / DuckDB / Polars / Pandas
→ 自动决定并发度
→ 自动决定分片方式和 shard size
→ 边算边写
→ 边写边释放
→ 内存紧张时降低 admission / evict / spill
→ 资源重新空闲后自动增加并发
→ 最终给出吞吐、资源、cache、scan、backend telemetry
```

核心指标：

```text
在 OOM=0、PIT/数值结果不变的前提下，
最大化 factors / minute 和 rows / second。
```

---

# 1. 当前 FactorEngine 已有的基础能力——不要推倒重来

当前代码已经有不少正确骨架，应复用：

```text
run_many
run_many_parallel
run_many_iter
compile_many
DAGPlan
structural CSE
rolling CSE
CSE refcount release
batch source prefetch
warmup clustering
materialize_many
materialize_sharded
factor_id / asset_bucket / time_month sharding
ExecutionResourcePlan
MemoryGovernor
DataAccess ScanCost
DataAccess QueryBudget
DuckDB shared engine
operator cost model
backend auto-routing
```

因此 R27 是：

```text
把已有能力从“组件存在”
升级成
“真正由一个统一 cost/resource scheduler 协同运行”。
```

---

# 2. 当前最重要的性能瓶颈

## R27-001：DAG 只是“roots + shared_nodes”，还不是真正的执行 DAG

当前：

```text
DAGPlan.roots
DAGPlan.shared_nodes
```

主要表达：

```text
共享子树
+
因子根
```

运行方式接近：

```text
先把所有 shared_nodes 物化
→ 再按 root layer 执行
```

需要升级成：

```text
所有 source scan / transform / CSE / rolling / group / root
都成为可调度 DAG Task。
```

每个 task：

```text
知道真正 predecessor
知道 consumer count
知道 cost
知道 memory
知道 backend
知道 output bytes
知道是否可 spill
知道是否可 shard
```

---

## R27-002：shared CSE 节点当前基本串行物化

当前 shared node 通过循环逐个：

```python
for sid, sub in dag.shared_nodes.items():
    materialize_shared_subplan(...)
```

这意味着：

```text
两个彼此完全独立的 shared nodes
也不会并行。
```

新调度器必须：

```text
shared node 和 root node统一进入 topological ready queue。
```

---

## R27-003：当前 parallel layer 把“列有重叠”当成冲突

当前依赖图中：

```text
referenced_columns 不相交
```

才允许同层并行。

这是性能上很大的 false dependency。

例如：

```text
factor A = ts_mean(close, 5)
factor B = ts_std(close, 20)
factor C = ts_rank(close, 60)
factor D = ts_corr(close, volume, 20)
```

它们都使用：

```text
close
```

但：

```text
共享只读 close
≠
执行依赖
```

恰恰因为共享 close：

```text
它们更适合放在同一 locality wave
避免重复扫描/解码/cache miss。
```

因此删除：

```text
column overlap => cannot parallel
```

这种规则。

真正 dependency 只能来自：

```text
PlanNode dependency
SourceTransform dependency
CSE dependency
state/checkpoint dependency
materialization ordering
```

---

# 3. 目标架构

新增核心模块：

```text
runtime/adaptive_batch_scheduler.py
runtime/resource_broker.py
runtime/task_resource_contract.py
runtime/hybrid_executor.py
runtime/adaptive_sharding.py
runtime/runtime_calibration.py
planner/physical_factor_dag.py
planner/read_wave_planner.py
planner/dag_cost_model.py
storage/streaming_result_sink.py
```

总体：

```text
                       ┌────────────────────┐
Factors ──compile─────▶│ PhysicalFactorDAG  │
                       └─────────┬──────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │ Unified Cost / Resource Model│
                  │ FE OperatorCost + DA ScanCost│
                  └──────────────┬──────────────┘
                                 │
                   ┌─────────────▼─────────────┐
                   │ Dynamic Resource Broker    │
                   │ CPU / RAM / IO / Spill     │
                   └─────────────┬─────────────┘
                                 │
                 ┌───────────────▼───────────────┐
                 │ Resource-Aware DAG Scheduler   │
                 └───────────────┬───────────────┘
                         ┌───────┴────────┐
                         │                │
                ┌────────▼───────┐ ┌─────▼───────────┐
                │ Thread Executor│ │ Process Executor │
                │ DuckDB/Polars  │ │ GIL-heavy Python│
                └────────┬───────┘ └─────┬───────────┘
                         │                │
                         └───────┬────────┘
                                 ▼
                        Bounded Result Sink
                                 │
                ┌────────────────┼────────────────┐
                ▼                ▼                ▼
             Parquet          FactorMatrix    ClickHouse
```

---

# 4. `PhysicalFactorDAG`

## R27-004

新增：

```python
@dataclass
class PhysicalFactorTask:
    task_id: str
    op: str
    inputs: tuple[str, ...]
    consumers: tuple[str, ...]

    execution_scope: str
    source_scope: str
    source_snapshot_id: str

    backend_candidates: tuple[str, ...]
    preferred_backend: str

    estimated_cost: TaskCost
    resource_contract: TaskResourceContract

    shard_spec: ShardSpec | None
    spillable: bool
    cacheable: bool
    deterministic: bool
```

---

## R27-005

Task 类型至少：

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

---

## R27-006

禁止：

```text
shared nodes一定先全部算完
```

改：

```text
某 shared node predecessor ready
且资源可 admission
就立刻运行。
```

---

# 5. DAG 优先级：critical path + reuse + locality

纯 FIFO 不够快。

## R27-007

为每 task 计算：

```text
estimated_ms
critical_path_remaining_ms
reuse_count
output_bytes
recompute_ms
scan_bytes
```

建议优先级：

```text
priority
=
critical_path_remaining_ms
+ λ1 * reuse_saved_ms
+ λ2 * locality_benefit
- λ3 * memory_pressure_cost
- λ4 * io_pressure_cost
```

其中：

```text
reuse_saved_ms
≈
(reuse_count - 1) * recompute_ms
```

---

## R27-008

高 reuse shared node优先：

例如：

```text
ts_mean(close,20)
```

被 100 个 factor使用，

应该比：

```text
只给1个 factor使用的 2GB 中间结果
```

更早执行、更值得 cache。

---

# 6. 统一 TaskResourceContract

## R27-009

每个 task执行前必须有：

```python
@dataclass
class TaskResourceContract:
    predicted_elapsed_ms: float

    cpu_tokens: int
    io_tokens: int

    input_bytes: int
    peak_memory_bytes: int
    output_bytes: int

    spill_bytes: int

    gil_bound: bool
    releases_gil: bool

    backend: str
    backend_threads: int

    shardable: bool
    shard_dimension: str | None

    uncertainty: float
```

---

# 7. FactorEngine CostModel 与 DataAccess ScanCost 合并

## R27-010

当前 FE 已有：

```text
OperatorCost
CostContext
time_complexity
memory_complexity
materialization_multiplier
```

不要废弃。

---

## R27-011

DataAccess 已有：

```text
selected_files
selected_bytes
selected_rowgroups
estimated_rows
projection_bytes
remote
selectivity
calibrated_factor
```

直接消费。

---

## R27-012

最终任务成本：

```text
TaskCost
=
SourceScanCost
+
OperatorComputeCost
+
ConversionCost
+
MaterializationCost
+
WriteCost
```

---

## R27-013

例如：

```text
close + volume
10 years
5000 stocks
daily
```

DataAccess先提供：

```text
selected_rows
projection_bytes
scan_bytes
remote/local
```

FE根据：

```text
operator family
window
backend
instrument count
rows
```

估算：

```text
CPU time
peak intermediate bytes
output bytes
```

---

# 8. 不要继续使用静态 low/medium/high 内存档作为最终 admission

## R27-014

最终必须得到：

```text
estimated_peak_bytes
```

数值。

初始可：

```text
panel_bytes = rows * instruments * dtype_size
```

再乘：

```text
input multiplicity
temporary multiplier
backend multiplier
window workspace multiplier
```

---

## R27-015

例如：

```text
2-input rolling correlation
```

可初估：

```text
peak
≈
inputA
+ inputB
+ output
+ rolling workspace
+ backend conversion
```

---

# 9. 在线学习实际成本

静态估计永远不够准。

## R27-016

每个 task运行后记录：

```text
actual_elapsed_ms
rss_before
rss_after
rss_peak_delta
read_bytes
output_bytes
spill_bytes
backend
threads
rows
instruments
window
```

---

## R27-017

建立 key：

```text
operator_canonical
backend
shape_bucket
window_bucket
market
frequency
```

---

## R27-018

维护 EMA：

```text
elapsed_factor
memory_factor
```

类似 DataAccess 已有的 scan calibration。

---

## R27-019

之后：

```text
predicted_peak
=
static_peak
* calibrated_memory_factor
* uncertainty_margin
```

---

# 10. 当前 `_per_worker_peak_mb` 必须修

## R27-020

当前资源代码中有一个明显问题：

注释：

```text
默认 3GB / worker
```

但函数返回：

```python
3.0
```

而名称：

```text
_per_worker_peak_mb
```

意味着：

```text
3 MB
```

语义明显不一致。

---

## R27-021

更关键：

当前 `resource_plan()`：

```text
并没有真正消费 _per_worker_peak_mb()
```

所以注释里声称：

```text
n_jobs <= process_budget / per_worker_peak
```

实际上没有真正形成 worker admission硬约束。

必须删除假合同或真正实现。

---

# 11. 动态内存余量：这点非常有用，而且必须做

用户的服务器：

```text
内存大小不同
同时还可能跑模型训练、算法、回测、别的任务
```

因此只看：

```text
host total RAM
```

不够。

只看：

```text
cgroup hard limit
```

也不够。

---

# 12. ResourceBroker 必须区分 Hard Limit 与 Live Headroom

## R27-022

定义：

```text
HardMemoryLimit
```

来自：

```text
cgroup memory.max
SLURM
RLIMIT
host RAM
用户配置
```

---

## R27-023

定义：

```text
LiveMemoryHeadroom
```

动态来自：

```text
cgroup memory.current
host MemAvailable
当前主进程 RSS
child worker RSS
system memory pressure
spill free disk
```

---

# 13. Live memory公式

## R27-024

建议：

```text
cgroup_headroom
=
memory.max - memory.current
```

若 cgroup无限：

```text
∞
```

---

## R27-025

主机：

```text
host_headroom
=
MemAvailable - external_reserve
```

---

## R27-026

配置：

```text
configured_headroom
=
configured_limit - FE_process_family_RSS
```

---

## R27-027

最终：

```text
live_headroom
=
min(
    cgroup_headroom,
    host_headroom,
    configured_headroom
)
```

---

# 14. 必须给其他程序留内存

## R27-028

新增：

```yaml
resources:
  coexist:
    enabled: true
    min_host_reserve_gb: 8
    min_host_reserve_fraction: 0.15
```

实际 reserve：

```text
max(
  min_host_reserve_gb,
  RAM * min_host_reserve_fraction
)
```

---

## R27-029

默认不要：

```text
把 MemAvailable吃到接近0。
```

---

# 15. 三种资源风格

## R27-030

用户可选：

```text
aggressive
balanced
coexist
```

### aggressive

```text
目标 CPU 90~98%
内存 reserve较小
```

### balanced

```text
目标 CPU 80~95%
内存保留约15~20%
```

### coexist

```text
服务器还跑模型/算法
CPU自动避让
内存 reserve约20~30%
```

默认：

```text
balanced / coexist-aware
```

---

# 16. 当前 MemoryGovernor 不够动态

## R27-031

当前 governor主要比较：

```text
this process RSS
vs
static process budget
```

这对：

```text
别的算法突然吃掉 30GB RAM
```

反应不够。

---

## R27-032

新增：

```python
ResourceSnapshot
```

至少：

```text
timestamp
hard_cpu_slots
system_cpu_util
cpu_psi
loadavg

hard_memory_limit
cgroup_memory_current
host_mem_available
process_rss
worker_rss
memory_psi
memory_events_high
memory_events_oom

spill_free_bytes
spill_write_mbps
disk_busy
```

---

# 17. 资源采样频率

## R27-033

建议：

```text
500ms ~ 2s
```

不要每 operator cell采样。

默认：

```text
1s
```

---

# 18. 当前 resource telemetry 的 RSS start/end要修

## R27-034

当前 telemetry的 `_rss_bytes()` 使用：

```text
ru_maxrss
```

它是：

```text
历史峰值
```

不是：

```text
当前 RSS
```

因此：

```text
rss_start
rss_end
```

实际上都可能是：

```text
同一个历史峰值
```

不能用于 adaptive scheduling。

---

## R27-035

改：

```text
current RSS -> psutil.Process().memory_info().rss
fallback /proc/self/status VmRSS
```

单独保留：

```text
ru_maxrss -> lifetime_peak_rss
```

---

# 19. child process RSS

## R27-036

若启用多进程：

```text
主进程 RSS
```

不够。

需要：

```text
sum(parent + recursive children RSS/PSS)
```

建议同时记录：

```text
rss_sum
pss_sum
```

如果 Linux `/proc/*/smaps_rollup` 可读：

优先 PSS做共享内存估算。

---

# 20. Memory Token Admission

## R27-037

调度器不能只：

```text
开 N worker
```

必须：

```text
每个 task申请 memory token。
```

---

## R27-038

例如：

```text
live headroom = 30 GB

task A = 8 GB
task B = 6 GB
task C = 12 GB
task D = 4 GB
```

可以：

```text
A+B+D = 18GB
```

若留 uncertainty/reserve，

而不是：

```text
按 32核一口气开32任务。
```

---

# 21. admission条件

## R27-039

只有：

```text
sum(predicted_peak_of_running_tasks)
+
candidate_peak * uncertainty
<=
admissible_memory
```

才启动 candidate。

---

## R27-040

初始 uncertainty：

```text
1.25~1.50
```

随着 calibration样本增加：

```text
下降到 1.10~1.20
```

---

# 22. 内存压力阶段动作

## R27-041

建议：

### NORMAL

```text
正常 admission
prefetch
cache
```

### PRESSURE_1

```text
停止新的 speculative prefetch
```

### PRESSURE_2

```text
evict低价值 cache
```

### PRESSURE_3

```text
停止新 task admission
spill高价值中间结果
```

### PRESSURE_4

```text
降低并发目标
```

### CRITICAL

```text
只允许 running task完成/写出
禁止新计算
紧急逐出 cache
```

---

# 23. 不建议直接 kill运行中的 task

## R27-042

因为：

```text
浪费已做计算
可能留下 staging artifacts
```

优先：

```text
停止 admission
→ evict
→ spill
→ writer drain
```

只有：

```text
明确可重试的 isolated worker
```

才能考虑 kill/requeue。

---

# 24. CPU Token，不只是 worker数量

## R27-043

一个：

```text
DuckDB task threads=4
```

应申请：

```text
4 CPU tokens
```

一个：

```text
Python single-thread operator
```

申请：

```text
1 CPU token
```

---

## R27-044

保证：

```text
sum(all running task CPU tokens)
<= dynamic CPU budget
```

而不是只有：

```text
n_jobs * fixed_duckdb_threads <= cores
```

---

# 25. 避免 nested parallelism

这是性能里非常重要的一点。

## R27-045

worker内部统一控制：

```text
OMP_NUM_THREADS
MKL_NUM_THREADS
OPENBLAS_NUM_THREADS
NUMEXPR_NUM_THREADS
DUCKDB threads
Polars Rayon threads
```

---

## R27-046

原则：

如果：

```text
外层开 16 worker
```

内层 NumPy BLAS再各开 16 threads：

```text
256 runnable threads
```

会比 16线程更慢。

---

# 26. backend thread budget

## R27-047

对于每 task：

```text
task_cpu_tokens = k
```

设置：

```text
DuckDB threads = k
BLAS threads = k / 1
```

具体按 operator backend。

---

# 27. 多线程 vs 多进程：不要二选一，要 Hybrid Executor

用户问：

```text
多线程 / 多进程是否支持？
```

目标答案：

```text
两者都支持，
但调度器自动选择。
```

---

# 28. Thread Executor适合什么

## R27-048

优先线程：

```text
DuckDB native query
Polars native expression
NumPy vectorized kernel
Numba nogil kernel
Arrow IO
其他明确 releases GIL 的实现
```

优势：

```text
共享 DataAccess cache
共享 DuckDB buffer/footer cache
共享 CSE panel
无 pickle
低内存复制
```

---

# 29. Process Executor适合什么

## R27-049

优先进程：

```text
纯 Python loop
Pandas-heavy GIL-bound operator
Python UDF
某些昂贵 Research/statistical kernel
需要独立 Polars thread pool配置的 task
```

---

# 30. 当前只用 joblib threading还不够

## R27-050

当前 root parallel主要：

```python
Parallel(..., backend="threading")
```

这对：

```text
DuckDB/NumPy/Polars
```

不错，

但对：

```text
GIL-bound Python operator
```

不能利用多核。

---

# 31. 不能一刀切全部进程

## R27-051

全部用 multiprocessing会出现：

```text
panel pickle
数据复制
每进程 DuckDB cache重复
每进程 Polars memory重复
CSE共享困难
RSS暴涨
```

因此：

```text
Hybrid
```

才合理。

---

# 32. Process Pool必须 long-lived

## R27-052

禁止：

```text
每个 factor启动一个新进程。
```

用：

```text
长期 worker pool
```

并在 worker启动前：

```text
设置 OMP/MKL/Polars threads。
```

---

# 33. Process间传数据禁止普通 pickle大 DataFrame

## R27-053

优先：

```text
Arrow IPC
memory mapped Arrow
shared memory
read-only NumPy mmap
```

---

## R27-054

任务传：

```text
buffer reference
column ids
row range
```

而不是：

```text
整个 5GB DataFrame。
```

---

# 34. Linux fork的 CoW不能当硬保证

## R27-055

虽然：

```text
fork + read-only memory
```

可能省复制，

但：

```text
Pandas block mutation
allocator行为
GC
temporary arrays
```

会破坏 CoW。

所以 production不要依赖：

```text
“fork应该不会复制”。
```

---

# 35. Polars线程池问题

## R27-056

当前代码自己已经知道：

```text
Polars import后修改 POLARS_MAX_THREADS不保证生效。
```

因此真正动态 per-task Polars CPU quota：

```text
不能只修改 env。
```

---

## R27-057

方案：

```text
Polars-heavy worker process按固定 thread budget启动。
```

可以准备 pool：

```text
1-thread pool
2-thread pool
4-thread pool
```

但不要过度复杂。

第一版：

```text
统一固定 Polars threads/worker
外层调 worker数。
```

---

# 36. DataAccess DuckDB应该优先共享线程执行

## R27-058

当前 DataAccess：

```text
进程内共享 DuckDB connection
query用 cursor()
buffer pool/footer cache共享
```

这是优势。

---

## R27-059

因此：

```text
不要为了“多进程”让每个 factor自己开DuckDB。
```

否则：

```text
缓存重复
内存重复
parquet footer重复
```

---

# 37. DataAccess ScanCost接入 scheduler admission

## R27-060

新增 API：

```python
DataAccessSource.estimate_scan_cost(
    fields,
    time_range,
    instruments,
)
```

返回已有：

```text
ScanCost
```

---

## R27-061

FE scheduler在真正 read前就知道：

```text
selected_bytes
projection_bytes
estimated_rows
remote/local
files
rowgroups
```

---

# 38. 不要再一次性 prefetch 全 batch 所有列

## R27-062

当前：

```text
all referenced columns union
→ 一次 prefetch
```

当 batch非常大：

```text
1000 factors
几十/几百字段
```

会让：

```text
源 panel/cache突然爆内存
```

---

# 39. ReadWavePlanner

## R27-063

将 factor tasks按：

```text
source_scope
dataset
snapshot
time_range
universe
column overlap
lookback
```

聚成：

```text
Read Waves
```

---

## R27-064

每 wave 满足：

```text
predicted_scan_output_bytes
+
predicted_compute_peak
<= wave_memory_budget
```

---

# 40. Read wave追求“共享最大、内存有限”

## R27-065

wave grouping不是：

```text
列不重叠
```

而是：

```text
共享列越多越值得一起
但总驻留字节不能超过budget
```

可定义：

```text
reuse_density
=
shared_scan_bytes_saved
/
wave_memory_bytes
```

---

# 41. DataAccess只读一次，一份 source buffer喂多个 task

## R27-066

同：

```text
dataset + snapshot + time range + universe
```

下：

```text
close
volume
amount
```

一次 scan，

多个 factor复用。

---

# 42. Projection pushdown

## R27-067

严禁：

```text
为了 batch方便 SELECT *
```

每 wave只投影：

```text
该 wave真实需要列。
```

---

# 43. filter / partition pushdown

## R27-068

继续利用 DataAccess：

```text
time range
instrument filter
manifest prune
row-group prune
```

让：

```text
scan bytes
```

先降下来。

---

# 44. Fully SQL / Polars subtree优先“整段下推”

## R27-069

如果：

```text
col → transform → rolling → arithmetic → root
```

整段可以 DuckDB/Polars native：

尽量一次执行。

不要：

```text
每个小 operator Materialize → Python → 再转回 native。
```

---

# 45. Materialization Boundary optimizer

## R27-070

选择中间 materialization点时考虑：

```text
reuse_count
output_size
recompute_cost
backend conversion cost
spillability
```

---

## R27-071

只有：

```text
值得共享
```

的节点才物化。

---

# 46. CSE不能“出现2次就一定物化”

## R27-072

当前结构 CSE：

```text
出现 >1
```

就会成为 shared node。

但：

```text
add(close,1)
```

可能极便宜，

物化一个巨大 panel反而更慢。

---

# 47. Cost-aware CSE

## R27-073

定义：

```text
CSE benefit
=
(reuse_count - 1) * recompute_cost
- materialize_cost
- memory_cost
```

---

## R27-074

只有：

```text
benefit > threshold
```

才 materialize。

否则：

```text
允许重复计算。
```

---

# 48. Cheap recompute vs expensive cache

## R27-075

例如：

```text
neg(x)
abs(x)
x + const
```

往往：

```text
重算比存一个5GB intermediate更划算。
```

---

# 49. Cache admission按 Benefit Density

## R27-076

定义：

```text
cache_value
=
expected_future_reuse * recompute_ms
```

---

## R27-077

定义：

```text
benefit_density
=
cache_value / bytes
```

---

## R27-078

高：

```text
保留。
```

低：

```text
优先 evict。
```

---

# 50. 多层 cache统一预算

目前可能同时：

```text
DataAccess query/result cache
FE column cache
panel cache
CSE cache
DuckDB buffer
Polars temporary
results
```

各自觉得自己“没超预算”，

总和却超内存。

---

## R27-079

新增：

```text
GlobalResourceBroker
```

统一为每层发 budget。

---

## R27-080

至少登记：

```text
dataaccess_scan_cache
dataaccess_query_cache
fe_column_cache
fe_panel_cache
fe_cse_cache
duckdb
result_queue
writer_queue
process_pool_shared_buffers
```

---

# 51. 避免 DataAccess + FactorEngine双缓存同一数据

## R27-081

如果 source已经：

```text
Arrow buffer
```

被 DataAccess cache持有，

FE不要再复制成：

```text
第二份 Pandas DataFrame
```

除非 backend确实需要。

---

# 52. 优先 Arrow / Polars lazy共享

## R27-082

尽量保持：

```text
Arrow / Polars
```

直到最后真正需要 Pandas reference算子。

---

# 53. Panel conversion计入 cost

## R27-083

以下不是免费：

```text
Arrow → Pandas
Pandas → Polars
long → wide
wide → long
stack/unstack
```

要进入：

```text
TaskCost
```

---

# 54. 动态自动分片

当前已有：

```text
factor_id
asset_bucket
time_month
```

这是很好的基础。

R27要让：

```text
scheduler自己选。
```

---

# 55. Shardability是语义属性，不只是性能属性

## R27-084

不能随便：

```text
按股票拆
```

如果 factor含：

```text
rank
zscore
neutralize
group
cross-sectional regression
market/global state
relation graph
```

按 asset shard会改变结果。

---

# 56. 自动 shard legality

## R27-085

编译计划后得到：

```text
TIME_SHARD_SAFE
ASSET_SHARD_SAFE
GROUP_SHARD_SAFE
FACTOR_SHARD_SAFE
SESSION_SHARD_SAFE
STATEFUL_CHECKPOINT_REQUIRED
```

---

# 57. factor_id sharding

## R27-086

适合：

```text
因子彼此共享少
或者多服务器分发。
```

缺点：

```text
会破坏跨因子CSE
会重复数据scan。
```

所以不要默认。

---

# 58. time sharding

## R27-087

适合：

```text
截面因子
group因子
全市场因子
```

因为每个日期保留完整 universe。

需要：

```text
lookback overlap
```

---

# 59. asset sharding

## R27-088

只允许：

```text
纯 instrument-separable时序因子。
```

---

# 60. intraday因子

## R27-089

优先：

```text
trade_date/session shard
```

每个 shard保留：

```text
完整一天
完整 session
```

不能切断 session。

---

# 61. stateful operator

## R27-090

若已有 checkpoint：

```text
segment incremental
```

否则：

```text
time shard必须携带足够状态历史
或禁止任意切分。
```

---

# 62. Hybrid shard

## R27-091

大量因子常见最优不是单维：

```text
time × factor_group
```

例如：

```text
一年一个 time shard
每 shard内算200个高度共享的因子
```

---

# 63. shard size动态决定

## R27-092

不是固定：

```text
每月
64资产bucket
```

而是：

```text
predicted_peak_memory <= shard_memory_target
```

---

# 64. 动态 shard公式

## R27-093

若：

```text
current shard estimated 20GB
memory target 8GB
```

自动：

```text
split_factor ≈ ceil(20/8)
```

然后再满足：

```text
calendar boundary
session boundary
group legality
```

---

# 65. 自动 work stealing

本地多 worker：

## R27-094

不要静态：

```text
worker0分A
worker1分B
```

而是：

```text
ready queue
谁空谁拿。
```

高成本task结束慢：

其他 worker自动拿剩余小task。

---

# 66. 多服务器场景

用户有：

```text
内存大服务器
内存小服务器
```

R27第二阶段建议支持：

```text
heterogeneous worker agents。
```

---

# 67. ServerCapability

## R27-095

每台服务器启动 agent时上报：

```text
cpu_slots
memory_hard
memory_live
spill_disk
disk_speed
local_datasets
backend capabilities
```

---

# 68. 大机器自然拿更大的 shard

## R27-096

例如：

```text
128GB server
→ 20GB wave / 12 concurrent tasks

32GB server
→ 5GB wave / 3 concurrent tasks
```

不需要用户手工写两套配置。

---

# 69. 多服务器调度第一版不要过度引入重型框架

## R27-097

本地单机 scheduler先做好。

多节点第一版可以：

```text
shared task manifest
+
atomic claim
+
heartbeat
+
lease
```

就够。

---

## R27-098

如果以后规模很大，

再考虑：

```text
Ray / Dask / Redis coordinator
```

不要为了“可能以后用”现在就把核心 engine绑死。

---

# 70. DistributedTask identity

## R27-099

task identity必须包含：

```text
factor semantic digest
data snapshot
market
universe
time range
shard
backend semantic version
```

避免不同 server写混。

---

# 71. task lease

## R27-100

多 server：

```text
PENDING
CLAIMED
RUNNING
COMMITTED
FAILED_RETRYABLE
FAILED_FATAL
```

---

## R27-101

worker crash：

```text
lease expire
→ task可重新 claim。
```

---

# 72. 结果必须边算边写，不要整个 batch驻留

这是很重要的速度+内存优化。

## R27-102

当前：

```text
run_many_iter
```

已经支持一个一个 yield。

继续强化。

---

# 73. parallel路径改成 as_completed

## R27-103

当前 joblib：

```text
一个 layer raw results先形成 list
```

再统一处理。

这会产生：

```text
layer result burst memory。
```

---

## R27-104

改：

```text
future completes
→ result validation
→ DQ
→ sink/write
→ release
```

---

# 74. Bounded Result Queue

## R27-105

计算线程和writer之间：

```text
bounded queue
```

按：

```text
bytes
```

限制，

不是只按：

```text
item count。
```

---

# 75. write backpressure

## R27-106

如果磁盘写不过来：

```text
queue达到bytes上限
→ scheduler降低新compute admission
```

不要：

```text
继续算100个结果堆内存。
```

---

# 76. compute与write pipeline重叠

## R27-107

目标：

```text
CPU 正在算 factor B/C
同时 writer落 factor A
```

而不是：

```text
全部算完
→ 再开始写。
```

---

# 77. 写端 batching

## R27-108

大量小因子单独：

```text
open file
write tiny parquet
close
```

会很慢。

---

## R27-109

按：

```text
date partition
factor group
matrix block
```

批量写。

---

# 78. Factor Matrix优先场景

如果后续：

```text
模型训练
回测
IC分析
```

会一次读大量因子，

应该：

```text
factor_matrix
```

而不是：

```text
1000个单独factor parquet再join。
```

---

# 79. Matrix block

## R27-110

建议：

```text
date × asset × factor_block
```

每 block：

```text
64 / 128 / 256 factors
```

按真实读模式 benchmark选。

---

# 80. float32

## R27-111

因子最终落值若：

```text
精度要求允许
```

默认：

```text
float32
```

可以：

```text
减半存储
减半I/O
减小训练读取
```

---

## R27-112

但：

```text
中间高精度统计
```

仍可 float64。

只有：

```text
最终materialization
```

转换。

---

# 81. 增量优先

如果因子已经落过历史：

```text
绝不能每天重算十年。
```

---

## R27-113

所有生产批量任务优先：

```text
watermark
+
affected source dependency
+
lookback
+
recompute tail
```

---

# 82. Data update → affected factor DAG

## R27-114

数据更新：

```text
只重算受影响 factor。
```

当前已有：

```text
plan_incremental_from_event
```

继续纳入极速路径。

---

# 83. 自动识别“不需要重新算”的因子

## R27-115

基于：

```text
factor semantic digest
source snapshot
source dependency hash
watermark
```

完全未变：

```text
SKIP。
```

---

# 84. 先减少工作量，再谈并行

性能优先级应该是：

```text
1. 不算不需要算的
2. 不重复读
3. 不重复算
4. native pushdown
5. 并行
6. spill
```

而不是：

```text
先把32核开满。
```

---

# 85. Warmup wave要从“lookback分类”升级为 cost-aware

当前：

```text
short
medium
long
full_history
```

是不错的初版。

---

## R27-116

新 wave还要考虑：

```text
scan bytes
memory
source columns
backend
CSE overlap
output bytes
```

---

# 86. full-history factor不要拖累普通因子

## R27-117

保留：

```text
full-history独立wave
```

甚至单独：

```text
low-concurrency lane。
```

---

# 87. high-cost lane

## R27-118

昂贵高级算子：

```text
DMD
kernel
entropy
complex PCA
tail
graph
```

不应该被删。

应该：

```text
high-cost queue
```

并受：

```text
memory token / CPU token
```

严格控制。

---

# 88. IO token

## R27-119

如果同时开：

```text
20个大 parquet scan
```

即使 CPU/RAM够，

磁盘也可能打满：

```text
random read
page cache churn
吞吐反而下降。
```

---

## R27-120

定义：

```text
IO tokens
```

不同 storage：

```text
NVMe -> 高
SATA SSD -> 中
network/COS -> 单独remote tokens
```

---

# 89. spill token

## R27-121

spill不是无限。

必须检查：

```text
free disk
reserved disk
spill speed
```

---

## R27-122

定义：

```text
usable_spill
=
free_disk - disk_reserve
```

---

# 90. 不要把服务器盘写满

## R27-123

默认至少保留：

```text
20GB
或磁盘10%
```

取较大者，

配置可调。

---

# 91. 只有昂贵、可复用 intermediate才值得 spill

## R27-124

不要 spill：

```text
cheap elementwise transform
```

因为：

```text
重新算可能比写盘+读盘快。
```

---

# 92. Recompute vs Spill planner

## R27-125

估算：

```text
spill_roundtrip_ms
vs
recompute_ms
```

若：

```text
recompute < spill
```

就：

```text
evict and recompute。
```

---

# 93. 避免 cache污染

## R27-126

一次性 root result：

```text
只写出
不进入长期 cache。
```

---

# 94. Data locality scheduling

## R27-127

同一个：

```text
SourceScope
same columns
same time range
```

的 tasks尽量连续/同时执行，

增加：

```text
OS page cache
DuckDB object cache
Arrow buffer
FE source cache
```

命中。

---

# 95. locality不能压过内存安全

## R27-128

如果同 wave合并后：

```text
驻留过大
```

拆 wave。

---

# 96. Scheduler scoring

## R27-129

可采用：

```text
HEFT-like
```

但不用引入复杂论文系统。

第一版：

```text
Priority =
critical_path
+ reuse_benefit
+ locality_score
- memory_penalty
```

即可。

---

# 97. External workload awareness

用户明确说：

```text
服务器可能同时跑算法。
```

这必须做。

---

# 98. CPU coexist

## R27-130

监控：

```text
system CPU utilization
load average
CPU PSI
```

如果：

```text
外部负载升高
```

降低：

```text
soft CPU tokens
```

---

# 99. 内存 coexist最重要

## R27-131

如果其他任务内存突然上涨：

```text
MemAvailable下降
```

FE：

```text
停止 admission
而不是等自己RSS到90%才反应。
```

---

# 100. 推荐 cgroup QoS

## R27-132

如果部署环境支持：

最稳的是给 FactorEngine独立：

```text
cgroup / systemd scope
```

设置：

```text
MemoryHigh
MemoryMax
CPUWeight
IOWeight
```

---

## R27-133

这样：

```text
操作系统也能帮你和训练任务隔离。
```

Python内部 governor仍继续工作。

---

# 101. MemoryHigh比直接OOM更友好

## R27-134

配置：

```text
memory.high
```

达到时：

```text
kernel进行回收/节流
```

比：

```text
直接碰 memory.max然后OOM
```

更适合 coexist。

---

# 102. Resource profile自动化

## R27-135

第一次在服务器启动：

生成：

```text
ServerFingerprint
```

：

```text
CPU model
CPU slots
RAM
NUMA nodes
disk class
local data path
DuckDB version
Polars version
```

---

# 103. per-server calibration

## R27-136

持久化：

```text
~/.cache/factor_engine/perf/<server_fingerprint>.json
```

保存：

```text
scan calibration
operator/backend calibration
memory multiplier
write throughput
```

---

# 104. 不同服务器不用同一固定 n_jobs

## R27-137

同一份配置：

```yaml
parallelism: auto
```

在：

```text
8C16G
```

和：

```text
32C128G
```

自动得到不同 plan。

---

# 105. NUMA

对于非常大的 2-socket server：

## R27-138

第二阶段加入：

```text
NUMA-aware worker placement
```

避免：

```text
跨NUMA memory traffic。
```

---

## R27-139

第一版可以：

```text
detect NUMA
只 telemetry
```

不必立刻复杂化。

---

# 106. Config设计

建议：

```yaml
execution:
  scheduler: adaptive

  parallel:
    mode: hybrid
    max_workers: auto
    cpu_target: 0.90
    allow_processes: true

  memory:
    mode: live_headroom
    process_fraction: 0.75
    min_host_reserve_gb: 8
    min_host_reserve_fraction: 0.15
    task_uncertainty_factor: 1.30

  coexist:
    enabled: true
    profile: balanced

  io:
    max_concurrency: auto
    remote_concurrency: auto

  spill:
    enabled: true
    directory: auto
    min_free_gb: 20
    min_free_fraction: 0.10

  cache:
    admission: benefit_density
    global_budget: auto

  batching:
    read_wave: adaptive
    auto_shard: true
    result_policy: sink
    writer_queue_bytes: auto

  calibration:
    enabled: true
    persist: true
```

---

# 107. 简化用户 API

## R27-140

不要逼用户每次配置：

```text
worker=17
DuckDB threads=2
chunk=421
```

---

## R27-141

提供：

```python
engine.materialize_many_fast(
    factors,
    execution="auto",
    coexist=True,
)
```

---

# 108. dry-run planner

## R27-142

非常建议新增：

```python
engine.plan_many_fast(factors)
```

返回：

```text
预计读取多少GB
预计计算多少 task
CSE省多少
预计峰值内存
预计worker
预计shards
预计backend分布
预计落盘GB
```

---

# 109. 示例 dry run

```text
Factors                 2,430
Unique DAG tasks          812
CSE materialized          103
CSE recompute-cheaper      76

Data scan               182 GB raw
Projected scan           41 GB
Scan waves                11

CPU slots                 32
Live memory headroom      71 GB
Planned memory budget     49 GB

Thread workers             9
Process workers            3
DuckDB task threads      1–4

Estimated factor output   18 GB
Writer queue max            4 GB
```

---

# 110. 调度器一定要可解释

## R27-143

每个 task记录：

```text
为什么现在执行
为什么选线程
为什么选进程
为什么选4 threads
为什么不并发另一个task
为什么spill/evict
```

---

# 111. telemetry

最终至少：

```text
total_wall_time
factors_per_minute
rows_per_second

cpu_util_avg
cpu_util_p95
cpu_psi

rss_current
rss_peak
process_family_rss
mem_available_min
cgroup_headroom_min
memory_pressure_events

scan_bytes
scan_files
scan_count
source_reuse

CSE_candidates
CSE_materialized
CSE_recompute
CSE_saved_ms

thread_task_count
process_task_count

spill_bytes
spill_read_bytes
spill_write_bytes

cache_hits
cache_evictions
cache_saved_ms

write_bytes
write_mbps
writer_backpressure_seconds
```

---

# 112. 当前 telemetry不能把 start/end都当peak

## R27-144

明确拆：

```text
rss_current_start
rss_current_end
rss_peak_lifetime
rss_peak_run
```

---

# 113. per-task peak memory

## R27-145

最理想：

```text
worker process单task
```

可以较准确算：

```text
peak delta。
```

线程task共享进程时：

只能：

```text
采样 + attribution approximation。
```

---

# 114. 优先做 task级 calibration，而不是追求绝对完美 attribution

## R27-146

只要能做到：

```text
后续 admission越来越准
```

即可。

---

# 115. benchmark suite

必须建立：

```text
factor_engine/benchmarks/r27/
```

---

# 116. workload A：普通日频

## R27-147

```text
100
1000
5000
```

因子。

包含：

```text
price-volume
rolling
rank
group
组合
```

---

# 117. workload B：高共享

## R27-148

大量：

```text
close
returns
rolling means
rolling std
```

共享子式。

验证：

```text
CSE + locality。
```

---

# 118. workload C：低共享

## R27-149

验证：

```text
调度器不会为了CSE浪费memory。
```

---

# 119. workload D：分钟→日频

## R27-150

验证：

```text
scan
session partition
IO
memory。
```

---

# 120. workload E：基本面/PIT

## R27-151

验证：

```text
多源join
PIT
source reuse。
```

---

# 121. workload F：高成本算子

## R27-152

```text
DMD
entropy
PCA
kernel
tail
```

验证：

```text
high-cost lane
memory token。
```

---

# 122. workload G：混合后台负载

## R27-153

在 benchmark同时运行：

```text
CPU burner
memory holder
```

模拟：

```text
模型训练/其他算法。
```

---

# 123. server profiles

至少：

```text
8C / 16GB
16C / 32GB
16C / 64GB
32C / 128GB
```

能实际测哪些测哪些。

---

# 124. 基线

必须同时跑：

```text
BASELINE-1: sequential run
BASELINE-2: current run_many
BASELINE-3: current run_many_parallel
R27 adaptive
```

---

# 125. 正确性 benchmark

## R27-154

任何性能优化：

```text
不能只比速度。
```

必须对：

```text
serial reference
```

做：

```text
exact or tolerance equality
NaN mask equality
index equality
PIT equality
```

---

# 126. shard正确性

## R27-155

每种合法 shard：

```text
sharded compute
merge
```

必须：

```text
等价 full compute。
```

---

# 127. CSE正确性

## R27-156

```text
CSE on
CSE off
```

结果一致。

---

# 128. thread/process正确性

## R27-157

```text
thread path
process path
serial path
```

结果一致。

---

# 129. backend正确性

## R27-158

只对：

```text
已有正确证据支持的 backend
```

做自动 routing。

---

# 130. PIT绝不为速度退让

## R27-159

以下禁止：

```text
为了复用把不同 data snapshot 混在同一个 CSE
为了少读历史缩短真实 warmup
为了分片丢掉财务 knowledge-time
为了并发绕过 source contract
```

---

# 131. source scope仍是硬边界

## R27-160

共享必须满足：

```text
same market
same universe membership
same calendar
same source snapshot
same decision-time policy
same source dependency identity
```

---

# 132. 性能目标不要写死绝对倍数

不同 workload差异太大。

R27 acceptance使用：

```text
无明显回退 + 大 batch显著提升
```

而不是承诺：

```text
一定10倍。
```

---

# 133. 建议的目标区间

## R27-161

对 1000+ factor大 batch：

希望：

```text
CPU-bound workload:
CPU utilization ≥ 80%（无外部竞争时）

memory:
不触发OOM
live available不跌破reserve

scan:
相同 source scope重复scan显著下降

large batch throughput:
相对 current run_many_parallel有实质提升
```

---

# 134. OOM是硬失败

## R27-162

所有 benchmark：

```text
OOM = 0
swap storm = 0
disk full = 0
```

---

# 135. Coexist benchmark

## R27-163

当外部程序占用：

```text
30~50% CPU
30~50% RAM
```

R27：

```text
自动降低并发
不抢到系统OOM。
```

---

# 136. 第一阶段实现顺序

不要一次全部重构。

### Phase A：最快收益

```text
1. 修 dependency_graph false column conflict
2. shared nodes按真实依赖并行
3. parallel result as_completed + sink
4. dynamic live memory headroom
5. 真正 memory-token worker admission
6. DataAccess ScanCost接 FE scheduler
7. bounded read waves
```

---

# 137. Phase B：Hybrid executor

```text
8. operator GIL classification
9. thread/process hybrid
10. Arrow shared buffers
11. nested parallelism closure
```

---

# 138. Phase C：Adaptive sharding / calibration

```text
12. semantic shard legality
13. adaptive shard size
14. per-server calibration
15. benefit-density cache
16. spill-vs-recompute
```

---

# 139. Phase D：多服务器

```text
17. worker capability agent
18. distributed task manifest
19. lease / retry / work stealing
```

---

# 140. 当前代码具体整改点

## R27-164

`planner/dependency_graph.py`

删除：

```text
column intersection == execution dependency
```

改为：

```text
真实 Plan dependency graph。
```

保留：

```text
column overlap
```

作为：

```text
locality/reuse hint。
```

---

## R27-165

`runtime/batch_service.py`

把：

```text
shared_nodes serial loop
```

升级：

```text
DAG-ready shared nodes parallel。
```

---

## R27-166

并行 root路径：

从：

```text
joblib layer returns list
```

升级：

```text
Future/as_completed
→ immediate sink。
```

---

## R27-167

`runtime/execution_resources.py`

修：

```text
per_worker_peak unit
```

并让：

```text
memory genuinely limit concurrency。
```

---

## R27-168

`runtime/resource_governor.py`

新增：

```text
live_headroom
external load
process-family memory
dynamic task admission。
```

---

## R27-169

`runtime/resource_telemetry.py`

修 current RSS。

---

## R27-170

`storage/sources/read_session.py`

不要：

```text
全 batch一次性 union prefetch。
```

由：

```text
ReadWavePlanner
```

按 wave调用。

---

## R27-171

DataAccess `scan_cost.py`

暴露稳定 public API，

FE不能访问：

```text
private store internals。
```

---

## R27-172

`backend/operator_cost.py`

扩：

```text
actual numeric peak bytes
backend calibration
```

---

# 141. 新 scheduler接口

## R27-173

```python
class AdaptiveBatchScheduler:
    def plan(...)
    def run(...)
    def materialize(...)
```

---

# 142. ResourceBroker API

## R27-174

```python
snapshot()
can_admit(task)
reserve(task)
release(task)
pressure_stage()
recommended_concurrency()
```

---

# 143. Ready Queue

## R27-175

task只有：

```text
all predecessors committed
```

才 ready。

---

# 144. Writer也属于 DAG

## R27-176

不要把 write当：

```text
计算结束后附带动作。
```

WriteTask：

```text
消费 root result
```

完成后：

```text
root output memory可释放。
```

---

# 145. Source scan也属于 DAG

## R27-177

这样调度器可以：

```text
prefetch下一wave
```

与：

```text
当前wave compute
```

重叠，

但受：

```text
memory/IO tokens
```

控制。

---

# 146. Double buffering

## R27-178

允许：

```text
compute wave N
同时 read wave N+1
```

前提：

```text
两者总 memory安全。
```

---

# 147. 不要无限ahead prefetch

## R27-179

默认：

```text
prefetch depth = 1 wave
```

动态内存大时可：

```text
2
```

---

# 148. Remote/COS读取

## R27-180

remote数据需要单独：

```text
network tokens。
```

---

# 149. local vs remote读取选择

## R27-181

继续利用：

```text
local mirror
remote fallback
hybrid
```

如果 DataAccess已有。

---

# 150. remote scan coalescing

## R27-182

同一个远程 parquet：

```text
不要20个factor各发一遍HTTP range。
```

尽量：

```text
wave合并。
```

---

# 151. page cache friendliness

## R27-183

同 dataset tasks：

```text
尽量接近执行。
```

---

# 152. 大量小文件

## R27-184

DataAccess cost已经惩罚小文件。

调度器应：

```text
优先合并scan
```

并通过：

```text
compaction
```

治理长期问题。

---

# 153. Writing tiny files也要治理

## R27-185

物化层：

```text
target file size
```

建议 benchmark：

```text
128MB~512MB
```

按 storage调整。

---

# 154. 编译缓存

## R27-186

大量自动挖掘因子：

```text
表达式相似。
```

compile/analyze也有成本。

---

## R27-187

对：

```text
semantic expression digest
```

缓存：

```text
IR
Plan
cost summary
```

---

# 155. 但 compile cache必须绑定 registry/evidence版本

## R27-188

否则：

```text
算子升级后复用旧计划。
```

---

# 156. Incremental compile

## R27-189

新增100个因子：

```text
不需要重新compile已有900个。
```

---

# 157. DAG append

## R27-190

允许：

```text
旧 DAG + new roots
```

重新做：

```text
incremental CSE merge。
```

第二阶段即可。

---

# 158. 容量预测 dry-run必须读 manifest，不读实际大数据

## R27-191

planner阶段：

```text
只用metadata/manifest
```

估算。

不要为了估成本：

```text
先把数据全读一遍。
```

---

# 159. 数据统计可采样

## R27-192

需要精确估计时：

```text
小 sample
```

而不是 full scan。

---

# 160. 编排算法建议

第一版不必追求理论最优。

## R27-193

采用：

```text
topological ready queue
+
resource admission
+
priority scoring
+
work stealing
```

足够。

---

# 161. 不建议第一版直接引入复杂 distributed scheduler

## R27-194

先做到：

```text
单机1000~5000因子高吞吐稳定
```

再多机。

---

# 162. 多机以后可复用 TaskResourceContract

## R27-195

因为：

```text
server agent
```

只是在不同机器上做：

```text
同一种 admission。
```

---

# 163. Metrics dashboard

## R27-196

建议输出：

```text
JSON
CSV
简单HTML
```

至少让用户看：

```text
CPU有没有吃满
内存为什么没吃满
为什么只开3个worker
哪几个factor最慢
哪张表scan最多
哪个operator拖后腿
```

---

# 164. Top bottlenecks报告

## R27-197

每批结束自动：

```text
Top 20 slowest factors
Top 20 expensive operators
Top 20 largest intermediates
Top 20 largest scans
Top 20 cache opportunities
```

---

# 165. 自动建议

## R27-198

例如：

```text
“78%时间消耗在 Pandas fallback”
“StockMinuteBar重复scan 12次”
“CSE candidate X 4.8GB但只省2.1s，不值得cache”
“writer占总时间34%，建议factor_matrix batching”
```

---

# 166. profiler sampling

## R27-199

默认不要全量 Python profiler，

开销大。

用：

```text
task timing
backend telemetry
resource samples
```

即可。

---

# 167. 专项 profile模式

需要时：

```text
py-spy
scalene
DuckDB EXPLAIN ANALYZE
Polars profile
```

离线诊断。

---

# 168. Fail-safe

## R27-200

资源估计错了：

```text
不能直接OOM。
```

需要：

```text
runtime pressure feedback。
```

---

# 169. estimation overrun

## R27-201

如果 task实际RSS增长：

```text
> predicted * tolerance
```

立即：

```text
raise its calibration factor
stop new admission
```

---

# 170. repeated underestimate

## R27-202

连续3次：

```text
underpredict
```

将该 operator/backend uncertainty：

```text
提高。
```

---

# 171. repeated overestimate

## R27-203

长期过度保守：

```text
逐渐降低 uncertainty
```

让机器吃得更满。

---

# 172. 自动并发不是固定一次算出

## R27-204

这是核心：

```text
并发度应在 run 过程中改变。
```

例如：

```text
开始 memory空 -> 8 tasks
外部模型启动 -> 降到3
模型结束 -> 回到7
```

---

# 173. 运行中不要强行减少已运行task

## R27-205

只改变：

```text
new admission。
```

---

# 174. 任务大小异质

## R27-206

不能假定：

```text
每factor一样重。
```

一批里可能：

```text
10ms
100ms
10s
```

混在一起。

---

# 175. Cost-aware scheduling避免长尾

## R27-207

尽量提前启动：

```text
critical long tasks
```

避免：

```text
最后只剩1个超慢factor拖整批。
```

---

# 176. Small task batching

## R27-208

大量：

```text
1~5ms elementwise roots
```

不要每个提交一个 Future。

合成：

```text
micro-batch
```

降低调度开销。

---

# 177. task granularity

## R27-209

设：

```text
min_task_estimated_ms
```

低于阈值：

```text
cluster execution。
```

---

# 178. thread/process pool size

## R27-210

不是：

```text
各开CPU数量
```

否则双倍 oversubscribe。

统一受：

```text
CPU token broker。
```

---

# 179. Query concurrency

## R27-211

DataAccess concurrent DuckDB cursor是好基础。

但：

```text
同时query数
```

也必须受：

```text
IO + memory token。
```

---

# 180. DuckDB memory_limit要和并发一致

## R27-212

如果：

```text
4个并行 query
```

不能每个都以为自己有：

```text
40GB memory limit。
```

---

# 181. Shared-process DuckDB

## R27-213

若是共享一实例：

定义：

```text
global DuckDB budget
```

不是 per-query重复预算。

---

# 182. isolated process DuckDB

## R27-214

如果某 process确实独立 DuckDB：

```text
per-process memory_limit
```

必须：

```text
sum <= total DuckDB budget。
```

---

# 183. Polars lazy fusion

## R27-215

尽量：

```text
多个表达式在一个 LazyFrame select
```

一次 scan输出多个 factor。

---

# 184. Multi-output native execution

## R27-216

这是非常有价值的提速方向。

同：

```text
source scope
backend
```

下多个根：

```text
一次 native query/select
```

而不是：

```text
每根调用backend.execute。
```

---

# 185. DuckDB multi-root query

## R27-217

对可 SQL lowering的多个因子：

生成：

```sql
SELECT
  ... AS factor_a,
  ... AS factor_b,
  ... AS factor_c
FROM ...
```

共享：

```text
scan
sort/window
```

由数据库优化器优化。

---

# 186. Polars multi-root select

## R27-218

类似：

```python
lf.select([
    expr_a.alias("factor_a"),
    expr_b.alias("factor_b"),
    expr_c.alias("factor_c"),
])
```

---

# 187. 这比 root-level threading更重要

## R27-219

因为：

```text
并行10个相同scan
```

不如：

```text
一次scan算10列。
```

---

# 188. Native Fusion Group

## R27-220

新增：

```text
NativeFusionGroup
```

约束：

```text
same backend
same source
same time/universe
same semantic scope
```

---

# 189. fusion group size受 memory限制

## R27-221

不要：

```text
5000个factor一次SQL SELECT
```

可能：

```text
planner爆
结果超宽
内存大。
```

---

# 190. fusion block

## R27-222

自适应：

```text
32 / 64 / 128 / 256 roots
```

根据：

```text
expression complexity
output bytes
backend compile time
```

调整。

---

# 191. multi-root output直接送 matrix writer

## R27-223

最理想：

```text
DuckDB/Polars算出一个 factor block
→ writer直接写 factor_matrix block
```

避免：

```text
拆成100个Series再拼回来。
```

---

# 192. 这是最终大批量因子落值的主路径

推荐执行优先级：

```text
1. Native multi-root fusion
2. CSE DAG execution
3. Thread parallel
4. Process fallback
```

---

# 193. Pandas fallback的优化

## R27-224

如果某些 operator只能 Pandas：

尽量：

```text
将所有 native前缀一次算好
只在最小边界转换到 Pandas。
```

---

# 194. Pandas process worker输入最小化

## R27-225

只传：

```text
该 subtree所需 columns/intermediates。
```

---

# 195. Numba

## R27-226

对大量固定 rolling kernel：

若已有 Numba路径：

```text
优先nogil/vectorized。
```

---

# 196. 不建议大规模使用 Modin作为核心策略

## R27-227

原因：

```text
额外调度层
与已有 DAG/resource scheduler重叠
Pandas兼容语义复杂
```

可保留 optional，

但不是 R27主路线。

---

# 197. GPU暂不作为主方案

## R27-228

大部分当前因子：

```text
数据搬运
rolling
cross-section
PIT join
```

未必自然 GPU友好。

先把：

```text
CPU / IO / DAG
```

做好。

---

# 198. 未来GPU可以作为backend candidate

## R27-229

TaskResourceContract设计应允许：

```text
gpu_tokens
gpu_memory
```

但当前不实现也行。

---

# 199. 验收脚本

新增：

```text
scripts/benchmark_r27_batch_throughput.py
scripts/audit_r27_resource_admission.py
scripts/audit_r27_dag_parallelism.py
scripts/audit_r27_shard_equivalence.py
scripts/audit_r27_native_fusion.py
scripts/audit_r27_external_pressure.py
```

---

# 200. 报告

生成：

```text
docs/R27_BATCH_THROUGHPUT_REPORT.md
docs/R27_RESOURCE_SCHEDULER_AUDIT.json
docs/R27_DAG_EXECUTION_AUDIT.json
docs/R27_SERVER_CALIBRATION.json
```

---

# 201. 性能矩阵

每 benchmark：

```text
workload
server
baseline
wall_time
factors_per_min
CPU avg/p95
peak RSS
min MemAvailable
scan GB
scan count
spill GB
write GB
CSE reuse
backend mix
```

---

# 202. R27 hard gates

## R27-230

```text
R27_TRUE_DAG_NODE_SCHEDULER=true
```

## R27-231

```text
R27_SHARED_NODES_PARALLELIZED=true
```

## R27-232

```text
R27_COLUMN_OVERLAP_NOT_FALSE_DEPENDENCY=true
```

## R27-233

```text
R27_NATIVE_MULTI_ROOT_FUSION=true
```

## R27-234

```text
R27_HYBRID_THREAD_PROCESS_EXECUTOR=true
```

## R27-235

```text
R27_NESTED_PARALLELISM_CLOSED=true
```

## R27-236

```text
R27_DYNAMIC_MEMORY_HEADROOM=true
```

## R27-237

```text
R27_EXTERNAL_WORKLOAD_AWARE=true
```

## R27-238

```text
R27_PROCESS_FAMILY_MEMORY_ACCOUNTED=true
```

## R27-239

```text
R27_DATAACCESS_SCAN_COST_INTEGRATED=true
```

## R27-240

```text
R27_OPERATOR_COST_CALIBRATED=true
```

## R27-241

```text
R27_MEMORY_TOKEN_ADMISSION=true
```

## R27-242

```text
R27_CPU_TOKEN_ADMISSION=true
```

## R27-243

```text
R27_IO_TOKEN_ADMISSION=true
```

## R27-244

```text
R27_SPILL_TOKEN_ADMISSION=true
```

## R27-245

```text
R27_AUTO_SHARD_SEMANTIC_SAFE=true
```

## R27-246

```text
R27_ADAPTIVE_SHARD_SIZE=true
```

## R27-247

```text
R27_READ_WAVE_MEMORY_BOUNDED=true
```

## R27-248

```text
R27_GLOBAL_PREFETCH_UNION_REMOVED=true
```

## R27-249

```text
R27_AS_COMPLETED_STREAM_MATERIALIZE=true
```

## R27-250

```text
R27_BOUNDED_WRITE_BACKPRESSURE=true
```

## R27-251

```text
R27_COMPUTE_WRITE_OVERLAP=true
```

## R27-252

```text
R27_CACHE_BENEFIT_DENSITY=true
```

## R27-253

```text
R27_DOUBLE_CACHE_BUDGET_CLOSED=true
```

## R27-254

```text
R27_SPILL_VS_RECOMPUTE_DECISION=true
```

## R27-255

```text
R27_PIT_SEMANTICS_PRESERVED=true
```

## R27-256

```text
R27_SHARD_FULLRUN_EQUIVALENCE=true
```

## R27-257

```text
R27_THREAD_PROCESS_SERIAL_EQUIVALENCE=true
```

## R27-258

```text
R27_OOM_ZERO=true
```

## R27-259

```text
R27_DISK_FULL_ZERO=true
```

## R27-260

```text
R27_LARGE_BATCH_THROUGHPUT_BENCH_PASS=true
```

---

# 203. 最终 production-ready条件

R27只解决：

```text
大规模计算吞吐和资源调度。
```

最终 FactorEngine production ready仍要求：

```text
R24
AND R25
AND R26
AND R27
```

对应 hard flags全部通过。

---

# 204. 推荐给 AI 的实际实现优先级

如果只能先做最重要的：

### P0-1
修：

```text
parallel layer false column dependency。
```

### P0-2
实现：

```text
live memory headroom + memory token admission。
```

### P0-3
修：

```text
per-worker memory constraint实际上未生效。
```

### P0-4
实现：

```text
parallel as_completed → immediate sink。
```

### P0-5
实现：

```text
ReadWavePlanner
而不是全 batch列union preload。
```

### P0-6
接：

```text
DataAccess ScanCost
+
FE OperatorCost。
```

### P0-7
实现：

```text
native multi-root fusion。
```

### P0-8
shared DAG node：

```text
按 dependency并行。
```

这八项做完，

大批量 factor落值吞吐应该就会出现最明显的第一波提升。

---

# 205. 推荐执行入口最终形态

```python
result = engine.materialize_many_fast(
    factors,
    factor_ids=factor_ids,

    scheduler="adaptive",
    parallel="hybrid",

    resource_profile="coexist",
    auto_shard=True,
    enable_cse=True,
    native_fusion=True,

    result_policy="sink",
    storage_format="matrix",

    pit_enforce=True,
)
```

---

# 206. 用户不应该手工做的事

用户以后不需要自己算：

```text
这台机开8线程还是16线程
DuckDB开几线程
Polars开几线程
一次算200还是500因子
分月还是分股票
什么时候spill
还有多少内存
```

这些都应该由：

```text
FactorEngine + DataAccess
```

自动决定。

---

# 207. 最终设计原则

**最快，不等于把 worker 数调到最大。**

真正的最大吞吐来自：

```text
少读
少算
少转换
多共享
native fusion
正确并行
动态资源 admission
边算边写
及时释放
```

最终要做到：

```text
内存大的机器自动多吃任务，
内存小的机器自动缩 batch；
服务器空闲时尽量吃满，
服务器有别的算法运行时自动退让；
相似因子共享扫描与中间结果，
不相似因子才真正并行；
native能一条 query算几十个因子，
就不要启动几十个独立 Python task；
所有结果完成一个就写一个/写一个 block，
不让几千个因子结果一起堆在 RAM。
```

---

# 208. AI 执行要求

收到本文件后不要只输出设计说明。

按以下顺序直接改：

```text
1. audit current batch execution path
2. benchmark current baseline
3. build PhysicalFactorDAG
4. fix false column dependency
5. integrate ScanCost + OperatorCost
6. implement live ResourceBroker
7. implement CPU/RAM/IO token admission
8. parallelize independent shared nodes
9. implement as_completed streaming sink
10. implement bounded ReadWavePlanner
11. implement native multi-root fusion
12. add hybrid thread/process routing
13. add adaptive shard legality + sizing
14. integrate cache/spill admission
15. run equivalence tests
16. run external-load benchmark
17. produce R27 acceptance report
```

不得以：

```text
“已有 run_many_parallel”
```

作为完成。

也不得只：

```text
把 n_jobs 调大。
```

最终必须证明：

```text
large-batch factor materialization throughput
在真实资源约束下显著优于当前路径，
且无 OOM、无 PIT变化、无数值变化。
```
