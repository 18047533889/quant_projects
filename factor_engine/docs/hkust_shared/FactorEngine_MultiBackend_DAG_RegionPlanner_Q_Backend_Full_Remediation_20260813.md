
# FactorEngine 多后端执行 / DAG / Backend Region Planner / Polars / DuckDB / Pandas / q(K) 全量补充整改任务书

> **仓库**：`18047533889/quant_projects`  
> **审计基线**：`main@854bdc2278678e3db03c894c059cd5fdbb7dac1b`  
> **日期**：2026-08-13  
> **对象**：Claude Code 主代理 + subagents  
> **范围**：最近几轮新增发现、此前 Model/Filter 整改文档未完整覆盖的多后端执行、DAG、Region Planner、Polars Lazy、DuckDB、Pandas、未来 q/K、CSE、内存、调度、分片、PIT 与生产治理问题。  
> **执行要求**：开始时重新读取最新 `main`。若某项已被后续提交真实修复，标记 `ALREADY_FIXED` 并给出代码/测试证据，不得重复造第二套 abstraction。  
> **数据要求**：不要以“服务器没有大规模真实因子/内存小”为理由停工。本任务应以静态代码、metadata-only shape、synthetic fixtures 和小数据 correctness tests 完成；真实 benchmark 以后只用于校准成本系数。

---

## 0. 总目标

不要只“再加一个 q backend”，而要把当前系统正式收口成：

```text
Canonical Factor DAG
        ↓
Semantic / PIT / Scope / ExecutionAxis
        ↓
Capability Annotation
        ↓
DataShapeEstimate（metadata only）
        ↓
Batch-Global Backend Region Optimizer
        ↓
Executable PhysicalRegionPlan
        ↓
Region Scheduler
        ↓
Backend-native execution
        ↓
Native buffer / streaming / spill / direct sink
```

生产环境必须满足：

```text
Planner 决定 backend
Executor 只执行计划
Operator 不再临时自选 backend
跨 backend 只发生在显式 Region boundary
```

---

## 1. Polars 自己有 DAG，但 FactorEngine DAG 仍然必须保留

Polars `LazyFrame` 自己有 query plan / DAG，会做 projection pushdown、predicate pushdown、common subplan/subexpression elimination、join/order 等内部优化。

这不意味着 FactorEngine DAG 多余。

正确结构：

```text
LEVEL 1: FactorEngine Global DAG
- 跨因子 CSE
- PIT
- universe
- DataAccess/source
- backend assignment
- ExecutionAxis
- state
- memory
- sharding
- materialization

              ↓

LEVEL 2: Backend-native plan
- Polars Lazy DAG
- DuckDB SQL optimizer
- q native execution
- Pandas/NumPy kernels
```

FactorEngine 只决定“哪些节点交给 Polars”，不要重复实现 Polars 内部 optimizer；Polars Region 内则尽量给它一整块 Lazy DAG，让其自己融合优化。

---

## 2. q/K 的架构结论

**不推荐把 FactorEngine 整体改写成 q/K。**

推荐：

```text
q/K = optional physical execution backend
```

保持以下 canonical authority 不变：

```text
DSL
IR
OperatorSemanticRegistry
PIT
DecisionClock
Unit
Factor identity
Evidence
DataAccess source authority
```

q/K 只负责：

```text
compile_to_q(region)
execute_q_region()
```

如果有人把 Factor DSL 直接改成 q expression，或让 q 的默认语义反过来决定 FactorEngine rolling/null/PIT 定义，必须拒绝。

---


## 3. P0 — Correctness / 直接架构缺陷

| ID | 问题 |
|---|---|
| MB-P0-001 | `_dag_aware_mixed_cost()` 引用未定义 `source_lowered`，mixed path 有 NameError 风险 |
| MB-P0-002 | mixed cost 使用 generic `sql` 而不是真实 `duckdb_sql/clickhouse_sql`，成本和能力身份错误 |
| MB-P0-003 | `plan_native_subgraph_fraction(..., backend=...)` 实际只检查 `supports_sql()`，Polars 路由也会报 SQL native fraction |
| MB-P0-004 | `PlanRoute` 只能表达单 backend；`hybrid` 只有一个 float cost，没有 node→backend assignment |
| MB-P0-005 | DAG-aware mixed cost 目前主要是“估价”，没有产出可执行的多 Region 物理 DAG |
| MB-P0-006 | Executor/fallback 会重新 routing；Planner 决策不是生产执行唯一真值 |
| MB-P0-007 | Polars fallback 仍可能逐 operator Pandas↔Polars 往返 |
| MB-P0-008 | 当前 `_best(node)` 只保留一个局部最佳 backend，不是 `DP[node][output_backend]`，父边 transfer affinity 未正确参与 |
| MB-P0-009 | shared DAG 成本可能重复累计；memo 只避免重复求解，不等于 shared compute 只计一次 |
| MB-P0-010 | shared node revisit 时生成的 node id 可能与首次 id 不一致，必须稳定映射 |
| MB-P0-011 | cross-sectional operator 不能按 asset bucket 分片，否则 rank/zscore/neutralize 在错误股票池计算 |
| MB-P0-012 | recursive stateful operator 不能无 checkpoint 按时间并行切片 |
| MB-P0-013 | Region boundary 必须保留 PIT / universe / grain / available_at / source_snapshot |
| MB-P0-014 | Pandas/Polars/DuckDB/q 的 null/NaN/inf/ties/ddof/window/quantile/div0 等必须由 canonical contract 统一 |
| MB-P0-015 | q/K 只能是 execution backend，不得成为 FactorEngine semantic authority |
| MB-P0-016 | Factor DSL/IR 不得退化成 q code；q 只能是 Canonical IR 的编译目标 |
| MB-P0-017 | Production 不得执行中 try backend A、失败后静默换 B |
| MB-P0-018 | 生产主路径禁止 operator 自选 backend；Region Planner 是唯一 backend decision authority |

## 4. P1 — 性能 / 内存 / 大规模真实场景

| ID | 问题 |
|---|---|
| MB-P1-001 | `cost_ctx(...,3000)` 固定股票数，A 股全市场/指数池/分钟场景误差大 |
| MB-P1-002 | unknown shape 默认 500k rows / 3000 instruments 过粗，需 metadata-only DataShapeEstimate |
| MB-P1-003 | `pd.bdate_range()` 不是 A 股交易日历，日期数应优先用 session calendar metadata |
| MB-P1-004 | peak memory 只按 rows×8 粗估，忽略 live columns/dtype/sort/hash/conversion overlap |
| MB-P1-005 | `_one_conversion_penalty` 与 TTDC conversion/materialize 可能重复计费 |
| MB-P1-006 | operator compute cost 与 TTDC execute_ms 可能重复计 compute |
| MB-P1-007 | mixed backend transition 用统一 delegate penalty，没有 edge type/bytes/sort/repartition |
| MB-P1-008 | hybrid candidate 的内存 eligibility 不应套 Pandas peak estimate |
| MB-P1-009 | `plan_batch_route` 逐 root 选 backend 后汇总，不是真正 batch-global optimizer |
| MB-P1-010 | shared benefit 固定 10%/5ms heuristic 过粗，应从真实 DAG shared work 推导 |
| MB-P1-011 | scheduler/DQ/write/generation commit 的 heuristic 要拆成可校准 component |
| MB-P1-012 | CSE cache 不区分 native representation，可能 Polars shared node 先转 Pandas 再被反复转回 |
| MB-P1-013 | 同 backend shared node 应保持 native buffer，跨 representation 只转换一次并缓存 |
| MB-P1-014 | 缺 liveness/refcount 驱动的 early free，容易让大量 intermediate 活到 batch 结束 |
| MB-P1-015 | 缺 KEEP_NATIVE / RECOMPUTE / SPILL_ARROW / SPILL_PARQUET / DIRECT_SINK 物化决策 |
| MB-P1-016 | 低内存服务器必须 streaming-first，不能默认整批全部驻留 RAM |
| MB-P1-017 | Polars 多 root 应优先统一 Lazy DAG / collect_all，而非每 factor eager collect |
| MB-P1-018 | DuckDB Region 边界优先 Arrow/Relation，不应无必要 `.df()` 到 Pandas |
| MB-P1-019 | Source scan 要进入 DAG/成本模型，不能把 `column` 永远当 O(1) |
| MB-P1-020 | 远程 Parquet/COS 要共享 scan、projection/predicate/row-group pruning |
| MB-P1-021 | TransferEdge 成本必须包含 sort / repartition / wide↔long / dtype cast |
| MB-P1-022 | 外层并发 × Polars/DuckDB/BLAS 内并发可能 oversubscribe |
| MB-P1-023 | Region Scheduler 需要 CPU/RAM/IO/backend-thread token |
| MB-P1-024 | 10k factor batch 禁止所有最终因子同时保存在 RAM |
| MB-P1-025 | 最终落 factor lake 时支持 backend direct sink，避免 native→Pandas→Parquet |
| MB-P1-026 | Semantic result cache 与 Physical plan cache 必须分层 |
| MB-P1-027 | backend/capability/cost model/resource profile 改变只失效 physical plan cache |

## 5. P2 — 治理 / q/K / 可维护性

| ID | 问题 |
|---|---|
| MB-P2-001 | 统一 BackendCapability authority，避免 registry/cost/polars-long/sql-lowerer 各有一套 |
| MB-P2-002 | BackendRegion / TransferEdge / NativeBuffer / PhysicalRegionPlan typed contract |
| MB-P2-003 | ExecutionAxis 一等化：TIME_PER_INSTRUMENT / CROSS_SECTION_PER_DATE / GROUP_PER_DATE / GLOBAL_PANEL / RECURSIVE_TIME |
| MB-P2-004 | Representation 一等化：PANDAS_LONG/WIDE、POLARS_LAZY_LONG/WIDE、ARROW、DUCKDB_RELATION、Q_TABLE |
| MB-P2-005 | sorted_by / partitioned_by / grouped_by / unique_key 作为物理属性 |
| MB-P2-006 | q process/session/version/license availability 显式治理 |
| MB-P2-007 | PyKX zero-copy 只能做优化，不能成为 correctness 前提 |
| MB-P2-008 | q null/symbol/date/timestamp/as-of adapter 必须版本化 |
| MB-P2-009 | Region/transfer/sort/spill/source scan 全链路 telemetry |
| MB-P2-010 | ExplainPlan：解释每个 Region 为何选择该 backend |
| MB-P2-011 | static prior 为主体，runtime actual 仅做 bounded calibration |
| MB-P2-012 | Polars Python UDF/delegate 必须从 native capability 中剥离 |
| MB-P2-013 | Research fallback 与 Production fallback policy 分离 |
| MB-P2-014 | PhysicalRegionPlan hash/certificate |
| MB-P2-015 | Region-level failure/retry/idempotency |
| MB-P2-016 | spill generation/cleanup/crash recovery |


---

## 6. P3 — 后续高级优化

P3 可以一次性做，但不得阻塞 P0/P1：

```text
cost-based graph coarsening
更精细的 DAG graph partition solver
representation-specific code generation
adaptive materialization
cross-job source cache
NUMA-aware scheduler
Arrow Flight / q IPC 优化
未来 GPU backend hook
distributed Region scheduler
```

---

# 第一部分：当前代码中的具体问题与明确改法

## 7. `_dag_aware_mixed_cost()` 的 `source_lowered` 直接 bug

当前函数参数没有 `source_lowered`，内部 `_eligible()` 却引用它。

不要只写：

```python
source_lowered = False
```

敷衍修复。

推荐统一成：

```python
class SourceRelationStatus(Enum):
    NO_SOURCE_REF = ...
    LOWERABLE = ...
    OPAQUE = ...
    INVALID = ...
```

Planner 和 SQL/Polars/q capability 都读取这个状态。

**Hard Gate**

```text
BACKEND_MIXED_SOURCE_RELATION_NAMEERROR_ZERO
```

**Test**

```text
test_mixed_cost_source_relation_no_nameerror
test_lowerable_source_ref_keeps_native_candidates
test_opaque_source_ref_rejects_invalid_pushdown
```

---

## 8. generic `"sql"` 不是物理 backend

mixed planner 已经知道真实：

```text
duckdb_sql
clickhouse_sql
```

但 `_eligible()` / `_cost()` 仍可能用 `"sql"`。

物理 cost / capability / telemetry 必须用具体 backend。

逻辑层可以有：

```text
SQL_FAMILY
```

但进入 Physical Plan 后必须是：

```text
DUCKDB_SQL
CLICKHOUSE_SQL
```

---

## 9. `plan_native_subgraph_fraction()` 必须真正 backend-specific

现在 backend 参数不能只是装饰。

统一 API：

```python
supports_backend(
    canonical,
    backend,
    mode,
    data_source_kind,
)
```

返回：

```text
NATIVE
DELEGATE
RESEARCH_ONLY
UNSUPPORTED
```

native fraction 至少同时输出：

```text
node_count_fraction
estimated_compute_fraction
native_nodes
delegate_nodes
unsupported_nodes
```

delegate 不得算 native。

---

## 10. 从 `PlanRoute(backend="hybrid")` 升级为真正 PhysicalRegionPlan

建议建立：

```python
@dataclass(frozen=True)
class BackendRegion:
    region_id: str
    backend: str
    representation: str
    node_ids: tuple[str, ...]
    execution_axis: str
    required_properties: ...
    state_contract: ...

@dataclass(frozen=True)
class TransferEdge:
    edge_id: str
    producer_region: str
    consumer_region: str
    source_representation: str
    target_representation: str
    estimated_rows: int
    estimated_bytes: int
    requires_sort: bool
    requires_repartition: bool
    requires_reshape: bool
    semantic_contract: ...

@dataclass(frozen=True)
class PhysicalRegionPlan:
    regions: tuple[BackendRegion, ...]
    edges: tuple[TransferEdge, ...]
    topological_order: tuple[str, ...]
    peak_memory_estimate: int
    estimated_ttdc_ms: float
    plan_hash: str
```

`hybrid` 不再只是一个名字。

---

## 11. Planner 必须成为生产执行唯一真值

生产主路径禁止：

```text
RegionPlan 选 Polars
→ PolarsBackend 内某 op 再 BackendRouter.select()
→ 临时转 Pandas
```

Production：

```text
plan_regions()
→ execute_region_plan()
```

Research/debug 单算子 API 可以保留 `BackendRouter.select(canonical)`。

---

## 12. 消灭 operator-level Pandas↔Polars ping-pong

错误：

```text
Pandas panel
→ Polars op A
→ Pandas
→ Polars op B
→ Pandas
→ Polars op C
→ Pandas
```

正确：

```text
Pandas/Arrow boundary
→ Polars Region(A→B→C)
→ 一次边界输出
```

**Hard Gate**

```text
BACKEND_ZERO_OPERATOR_LEVEL_CROSS_ENGINE_PINGPONG
```

---

## 13. DP 必须考虑“父节点希望什么 backend”

当前如果：

```text
child Pandas = 9ms
child Polars = 10ms
Pandas→Polars transfer = 20ms
parent = Polars
```

全局应该选 child Polars。

所以不能：

```python
best[node] = one_backend
```

树结构至少应：

\[
DP(v,b)=Compute(v,b)+\sum_u \min_{b_u}[DP(u,b_u)+Transfer(b_u,b)]
\]

真实共享 DAG 还要进一步处理 shared-node 全局唯一 compute。

---

## 14. 真 DAG 成本不能按树重复加 shared node

Diamond：

```text
      A
     / \
    B   C
     \ /
      D
```

A 只算一次。

需要：

```text
stable node set
edge set
fanout
consumer count
buffer identity
```

成本规则：

```text
每 node compute 只计一次
每 representation conversion 只计一次并可供同类 consumers 共享
```

---

## 15. shared node 的 id 必须稳定

遍历不能第一次用 `n3`，第二次 revisit 又用 `n{id(obj)}`。

建立：

```python
object_id_to_stable_node_id
```

更好是由 logical plan 本身携带稳定 node id / structural identity。

---

## 16. Region 切分不是“最大兼容块”，而是 Cost-Optimal Region

即使 A-F 全支持 Polars，如果 D 在 DuckDB 快很多，且 transfer 后仍有净收益，可以单独切 Region。

目标：

```text
Compute
+ Transfer
+ Sort
+ Reshape
+ Memory Risk
+ Materialization
```

总和最小。

---

# 第二部分：Polars Region 的正确设计

## 17. Polars Region 内要给 Polars 自己一棵大 Lazy DAG

同 source scope 的 100/1000 因子，如果 Polars-compatible：

```text
一个 base LazyFrame
+ 多个 derived columns
+ shared expressions
+ 一次 collect / collect_all / sink
```

不要一个 factor 一个 eager collect。

---

## 18. Polars UDF 必须单独分类

能力状态至少：

```text
POLARS_NATIVE_EXPR
POLARS_NATIVE_GROUP
POLARS_NATIVE_STREAMING
POLARS_PYTHON_UDF_DELEGATE
UNSUPPORTED
```

`map_elements`、Python lambda、opaque `map_groups` 不得计成 native Polars 快路径。

---

## 19. Streaming capability 不是所有 Polars op 都一样

每个 canonical 标：

```text
streaming_safe
requires_global_sort
requires_full_group
requires_recursive_state
```

Planner 才能决定：

```text
collect
stream
sink_parquet
```

---

## 20. Polars shared node 保持 native

错误：

```text
ADV20 Polars
→ to_pandas
→ cache
→ 100 consumers 再转 Polars
```

正确：

```text
ADV20 Polars native buffer
→ 100 Polars consumers
```

第一次有 Pandas consumer 时：

```text
Polars→Pandas 一次
→ 缓存第二 representation
```

这叫：

```text
representation-aware CSE
```

---

# 第三部分：DuckDB / Arrow / Pandas / q

## 21. DuckDB Region

DuckDB 优先候选：

```text
Parquet scan
projection/filter
large join
relational aggregation
SQL-native window
远程/本地 columnar source
```

Region 内保持：

```text
DuckDB Relation / SQL
```

跨 Region 优先：

```text
DuckDB → Arrow
```

不要无意义：

```text
DuckDB → Pandas
```

---

## 22. SQL ordering 不能假设

时间序列 downstream 如果需要：

```text
sorted_by=(instrument, datetime)
```

producer 没有该 property：

```text
TransferEdge.requires_sort=True
```

并计 sort cost。

---

## 23. Pandas/NumPy 的定位

保留：

```text
reference backend
SciPy/NumPy specialized kernels
复杂 regression
研究型特殊算法
小数据
```

不要为了“Polars 覆盖率”强行把所有复杂算法迁 Polars。

---

## 24. q/K 第一期只做高适配场景

优先：

```text
arithmetic
comparison
lag/delta
rolling mean/sum/std/min/max
corr/cov
rank/group basic
VWAP/basic aggregation
time/as-of preparation
minute→daily aggregation
```

暂缓：

```text
复杂 model training
复杂 checkpoint state
topology/特殊科研算法
```

---

## 25. q/K boundary 也必须 Region 级

错误：

```text
Python→q op1→Python→q op2
```

正确：

```text
50/100/500 q-native nodes
→ q Region
→ 一次传输
```

---

## 26. q/K 类型与 PIT

显式定义：

```text
q null
float NaN/inf
int null
symbol
string
timestamp
date
timespan
boolean null
```

以及 Arrow/Pandas/Polars 转换。

q as-of join 再快，也必须由 DataAccess 给定：

```text
available_at
revision vintage
source snapshot
DecisionClock
```

---

# 第四部分：DataShape / Cost Model

## 27. 不加载大数据也能规划：DataShapeEstimate

新增 metadata-only：

```python
@dataclass(frozen=True)
class DataShapeEstimate:
    estimated_rows: int
    estimated_dates: int
    estimated_instruments: int
    estimated_columns: int
    estimated_bytes: int
    average_row_width_bytes: float
    density: float
    frequency: str
    bars_per_session: int | None
    group_count: int | None
    remote: bool
    storage_kind: str
    sorted_by: tuple[str, ...]
    partition_by: tuple[str, ...]
    projected_columns: tuple[str, ...]
```

来源：

```text
Parquet metadata
DataAccess manifest
schema
date bounds
universe metadata
row-group stats
calendar sessions
```

---

## 28. 去掉固定 3000 instruments

主路径使用：

```text
shape.estimated_instruments
```

没有 metadata 时使用 conservative universe prior：

```text
CSI300 ≈ 300
CSI500 ≈ 500
CSI1000 ≈ 1000
ALL_A ≈ 5500 conservative
```

---

## 29. 日期数优先真实交易日历

不要主路径：

```python
pd.bdate_range()
```

应：

```text
calendar_service/session_count
```

fallback 才普通工作日估计。

---

## 30. 成本函数拆成单一 authority 的 components

\[
C_{total}
=
C_{source}
+C_{compute}
+C_{transfer}
+C_{reshape}
+C_{sort}
+C_{materialize}
+C_{spill}
+C_{schedule}
+C_{memory-risk}
+C_{sink}
\]

任何 component 只能加一次。

---

## 31. 审计当前 double-count

重点检查：

```text
operator backend cost
+ _one_conversion_penalty
+ predict_ttdc conversion
+ predict_ttdc execute
+ materialize
```

禁止 conversion/compute 同时在两层重复。

---

## 32. TransferEdge 要按 edge type 和 bytes 计价

至少：

```text
pandas_to_polars
polars_to_pandas
duckdb_to_arrow
arrow_to_polars
arrow_to_pandas
arrow_to_q
q_to_arrow
wide_to_long
long_to_wide
sort
repartition
dtype_cast
```

---

## 33. Representation 一等化

不要只有：

```text
backend=polars
```

还要：

```text
representation=POLARS_LAZY_LONG
```

建议 Enum：

```text
PANDAS_LONG
PANDAS_WIDE
NUMPY_PANEL
POLARS_LONG
POLARS_WIDE
POLARS_LAZY_LONG
ARROW_TABLE
DUCKDB_RELATION
Q_TABLE
Q_VECTOR
```

---

## 34. LONG/WIDE 不能合并

现有 `polars_panel` / `polars_long` 区分应保留并正式进入 physical representation。

---

## 35. Physical properties

Region output 声明：

```text
sorted_by
partitioned_by
grouped_by
unique_key
grain
```

下游不满足则显式插入 property enforcement，并计成本。

---

# 第五部分：ExecutionAxis / 分片 / state

## 36. ExecutionAxis 一等化

建议：

```text
TIME_PER_INSTRUMENT
CROSS_SECTION_PER_DATE
GROUP_PER_DATE
GLOBAL_PANEL
EVENT_STREAM
RELATIONAL
RECURSIVE_TIME_PER_INSTRUMENT
```

---

## 37. 分片安全规则

### TIME_PER_INSTRUMENT
- asset shard 通常安全
- time shard 需要 lookback overlap

### CROSS_SECTION_PER_DATE
- date shard 安全
- asset shard **禁止**

### GROUP_PER_DATE
- 必须保证整个 group 在同一 shard

### RECURSIVE_TIME
- asset shard 通常安全
- time shard 只有 checkpoint chain 才合法

---

## 38. TS→CS→TS 是调度 barrier

例如：

```text
ts_mean(close,20)
→ rank
→ ts_mean(rank,5)
```

必须：

```text
TIME region
→ CROSS_SECTION barrier
→ TIME region
```

不能整个 pipeline 用一种 shard 维度。

---

## 39. rolling 时间分片

window W：

```text
下一个 shard 读取 W-1 overlap
只输出 primary interval
```

---

## 40. recursive 时间分片

默认：

```text
SEQUENTIAL_SHARD_CHAIN
```

只有已验证：

```text
checkpoint seed
```

才允许独立 shard continuation。

---

# 第六部分：Source DAG / CSE / 内存 / 物化

## 41. Source scan 也是 DAG 节点

新增/明确：

```text
SourceScanNode
```

包含：

```text
dataset
snapshot
projected columns
date range
universe
predicate
partitions
remote/local
```

10000 因子共用 close/volume/amount：

```text
scan 一次
```

而不是每 factor 重新 COS/Parquet scan。

---

## 42. DataAccess 接口

建议：

```text
describe_scan()
estimate_scan()
scan_capabilities()
```

只读 metadata。

---

## 43. Batch-global planner

不要：

```text
for root:
    choose_plan_route(root)
```

真正优化对象应是：

```text
DAGPlan(all roots + shared nodes)
```

因为 shared scan/CSE 可能改变单 root 最优 backend。

---

## 44. `shared_benefit` 不能固定 10%

从真实 DAG 推导：

```text
avoided source scans
avoided shared compute
avoided transfer
```

---

## 45. NativeBufferStore

同 semantic node 可以有多个 representation：

```text
Polars
Arrow
Pandas
q
```

但每个 representation 最多转换一次，多个 consumer 共享。

key 至少：

```text
semantic_node_id
source_snapshot
scope
representation
dtype_schema
backend_semantic_version
```

---

## 46. Liveness / refcount

每 buffer 维护：

```text
remaining_consumer_count
```

最后一个 consumer 完成：

```text
release
```

---

## 47. Peak memory 必须考虑 boundary overlap

跨 backend 转换通常同时存在：

```text
source buffer
target buffer
scratch
```

所以：

\[
M_{edge}\approx M_{source}+M_{target}+M_{scratch}
\]

---

## 48. Materialization policy

每个 intermediate 选择：

```text
KEEP_NATIVE_MEMORY
RECOMPUTE
SPILL_ARROW
SPILL_PARQUET
DIRECT_SINK
```

CSE 不等于一定 materialize。

---

## 49. 低内存服务器原则

必须：

```text
streaming-first
projection pushdown
early release
adaptive batch size
preemptive spill
direct sink
avoid giant dense factor cube
```

禁止构造：

```text
date × stock × 10000 factors
```

完整 dense 立方体。

---

## 50. Direct Sink

如果最终目标是 factor lake：

```text
Polars Region → sink parquet
DuckDB Region → Arrow/COPY/writer
q Region → Arrow/Parquet adapter
```

不要统一先回 Pandas。

---

## 51. factor batch size 自适应

根据：

```text
memory budget
source sharing
CSE sharing
sink bandwidth
```

决定 batch 大小。

---

## 52. 跨 batch 高复用 shared node

如：

```text
returns
ADV20
vol20
VWAP
market cap
```

可以成为 durable intermediate，但只有当：

```text
reuse benefit > store/read cost
```

且绑定：

```text
semantic identity
source snapshot
version
```

---

# 第七部分：资源调度

## 53. 防 oversubscription

统一 `ResourceGovernor`：

```text
CPU tokens
RAM tokens
IO tokens
backend thread tokens
```

---

## 54. Backend thread policy

### 大 Polars Region
```text
outer concurrency 低
Polars internal threads 高
```

### Pandas 单线程小 Region
```text
outer concurrency 可高
```

### DuckDB
显式设置 DuckDB threads。

### BLAS
治理：

```text
OMP_NUM_THREADS
MKL_NUM_THREADS
OPENBLAS_NUM_THREADS
```

---

## 55. `n_jobs` 只作为上限

真实执行：

```text
Region ready queue
+ resource tokens
+ dependency DAG
```

---

## 56. IO backpressure

远程 source/sink 不能 32 个任务同时扫爆。

需要：

```text
IO tokens
sink backpressure
```

---

## 57. Scheduler priority

综合：

```text
critical path
fanout
memory release benefit
sink readiness
```

高 fanout shared producer 通常优先。

---

# 第八部分：Backend Semantic Parity

## 58. 必须统一的语义

逐 family 建 fixture：

```text
NaN/null
inf/-inf
division by zero
0/0
boolean null logic
integer overflow
float precision
ddof
rolling min_periods
current included/excluded
ties
rank pct denominator
quantile interpolation
group null key
sort stability
timestamp/timezone
```

---

## 59. rolling parity

明确：

```text
window observation count
current bar included?
min_periods
skip null?
physical bar gap vs compressed observation time
```

---

## 60. std/cov parity

固定：

```text
ddof
pairwise missing
sample/population
```

---

## 61. rank parity

固定：

```text
tie method
pct denominator
NaN handling
group behavior
stable deterministic ordering
```

---

## 62. quantile parity

固定 interpolation：

```text
linear / nearest / lower / higher / midpoint
```

只能一个 canonical definition。

---

## 63. timezone / calendar / price basis

Region boundary 不得丢：

```text
timezone
calendar
session
available_at
RAW/ADJUSTED/CONTINUOUS price_basis
```

---

## 64. wide↔long duplicate key

`(datetime,instrument)` duplicate 时：

```text
fail closed
```

禁止 pivot 自动 aggregate。

---

## 65. dtype policy

默认因子 numeric：

```text
float64
```

如果允许 float32：

```text
必须显式 policy + evidence
```

backend 不得自己降精度。

---

# 第九部分：Production fallback / cache / certificate

## 66. Production fallback

Production：

```text
unsupported → planning fail
```

或：

```text
在执行前就规划 certified alternative region
```

禁止 runtime try/except 后偷偷换 backend。

Research 可以 warn+fallback，但必须 telemetry。

---

## 67. Semantic cache vs Physical plan cache

### Semantic result cache key
```text
factor/node semantic identity
source snapshot
scope
```

backend 改变但语义相同，可以复用结果。

### Physical plan cache key
```text
logical hash
scope hash
capability hash
cost model version
resource profile
backend versions
hardware/storage family
```

---

## 68. PhysicalRegionPlan hash

包含：

```text
logical structural hash
region assignment
representation
edges
required properties
backend implementation semantic versions
```

不包含实际 runtime ms。

---

## 69. ProductionExecutionCertificate

扩展绑定：

```text
region_plan_hash
region_count
backend assignments
edge count
semantic authority hash
source snapshot policy
```

生产如果 certificate 是 hard gate，构造失败不能一律 fail-open。

---

# 第十部分：Telemetry / Explain / Calibration

## 70. 每次执行必须记录

```text
region_count
nodes per region
backend switches
transfer edge types
estimated/actual transfer bytes
sort count
reshape count
spill bytes
peak RSS
source scan bytes
compute ms
transfer ms
sink ms
planned backend
actual backend
fallback/replan reason
```

---

## 71. Explain Plan

例如：

```text
Region R2 → Polars
- 43 native expressions
- input already Arrow/Polars
- no additional global sort
- estimated 310MB
- DuckDB saves only 4ms
- boundary cost estimated 12ms
=> stay Polars
```

---

## 72. Static cost 是主体

本轮不要求下载/加载几十 GB 数据。

路由依据：

```text
capability
operator complexity
DataShape metadata
source affinity
memory budget
transfer estimate
DAG topology
```

现有 runtime calibration 继续记录 actual，后续只做 bounded EMA 修正。

---

## 73. Stay-in-engine hysteresis

只有：

```text
expected gain > transfer cost + safety margin
```

才切 backend。

防止：

```text
Polars→DuckDB→Polars→DuckDB
```

---

## 74. q unavailable policy

如果 plan 需要 q 但 q runtime/license unavailable：

```text
执行前选择 certified alternative
```

或：

```text
fail
```

不能 runtime 偷换 Pandas。

---

# 第十一部分：建议代码修改范围

## 75. `planner/physical_plan.py`

从：

```text
SQL / PYTHON / MATERIALIZED
```

扩成真正 Region model，或把旧 `PhysicalPlan` 包装/迁移到：

```text
BackendRegion
TransferEdge
PhysicalRegionPlan
```

---

## 76. 新增 `planner/backend_region.py`

放：

```text
PhysicalBackend
Representation
BackendRegion
TransferEdge
PhysicalRegionPlan
```

---

## 77. 新增 `planner/execution_axis.py`

放：

```text
ExecutionAxis
partition safety
barrier rules
```

---

## 78. 新增 `planner/data_shape.py`

metadata-only DataShapeEstimate。

---

## 79. `backend/plan_cost_router.py`

重构：

```text
单 PlanRoute
→ batch-global region optimizer
```

修：

```text
source_lowered
generic sql
native fraction
fixed 3000
double count
hybrid memory
```

---

## 80. `backend/operator_cost.py`

成本拆成：

```text
compute only
```

增加 shape：

```text
dates
instruments
group size
window
representation
```

不要同时承担 transfer/source/sink。

---

## 81. `backend/cleaned_bridge.py`

Production Region 内禁止每 operator normalize 到 Pandas。

只用于：

```text
explicit boundary
debug
reference tests
```

---

## 82. Polars backend/emitter

一个 Region：

```text
一棵 Lazy DAG
```

多 roots：

```text
collect_all / unified with_columns
```

共享节点 native 保留。

---

## 83. DuckDB backend

接受明确 DuckDB Region。

优先：

```text
Relation/Arrow
```

不要自己再决定 hybrid backend。

---

## 84. `runtime/engine.py`

主链改成：

```text
compile_many
→ CSE
→ DAGPlan
→ plan_regions
→ execute_regions
```

---

## 85. 新增 `runtime/native_buffer_store.py`

representation-aware buffer/cache + refcount/liveness。

---

## 86. Scheduler

从 root/job 粒度转向 Region DAG ready queue。

---

# 第十二部分：必须添加的 Hard Gates

```text
BACKEND_REGION_PLANNER_IS_EXECUTION_AUTHORITY
BACKEND_ZERO_OPERATOR_LEVEL_CROSS_ENGINE_PINGPONG
BACKEND_EVERY_LOGICAL_NODE_ASSIGNED_EXACTLY_ONCE
BACKEND_EVERY_TRANSFER_EDGE_EXPLICIT

BACKEND_MIXED_SOURCE_RELATION_NAMEERROR_ZERO
BACKEND_GENERIC_SQL_COST_ZERO
BACKEND_NATIVE_FRACTION_BACKEND_SPECIFIC

BACKEND_DAG_SHARED_COMPUTE_COUNTED_ONCE
BACKEND_DAG_NODE_IDS_STABLE
BACKEND_DP_PARENT_TRANSFER_AWARE

BACKEND_TRANSFER_COST_SINGLE_AUTHORITY
BACKEND_COMPUTE_COST_SINGLE_AUTHORITY
BACKEND_SORT_REPARTITION_COST_EXPLICIT

BACKEND_PHYSICAL_REPRESENTATION_TYPED
BACKEND_NATIVE_INTERMEDIATE_PRESERVED
BACKEND_CSE_REPRESENTATION_AWARE

BACKEND_CROSS_SECTION_ASSET_SHARD_ZERO
BACKEND_STATEFUL_UNSEEDED_TIME_SHARD_ZERO
BACKEND_EXECUTION_AXIS_ENFORCED

BACKEND_PIT_CONTRACT_PRESERVED_ACROSS_EDGES
BACKEND_UNIVERSE_CONTRACT_PRESERVED_ACROSS_EDGES
BACKEND_GRAIN_CONTRACT_PRESERVED_ACROSS_EDGES
BACKEND_SOURCE_SNAPSHOT_STABLE

BACKEND_MEMORY_PEAK_INCLUDES_CONVERSION_OVERLAP
BACKEND_REGION_PLAN_WITHIN_MEMORY_BUDGET
BACKEND_LIVENESS_RELEASES_DEAD_BUFFERS

BACKEND_RESOURCE_OVERSUBSCRIPTION_ZERO
BACKEND_DIRECT_SINK_NO_FORCED_PANDAS
BACKEND_BATCH_GLOBAL_PLANNING_ENABLED

BACKEND_POLARS_REGION_COMPILES_ONE_LAZY_DAG
BACKEND_POLARS_DELEGATE_NOT_NATIVE
BACKEND_DUCKDB_REGION_ARROW_BOUNDARY_PREFERRED

Q_BACKEND_ZERO_SEMANTIC_AUTHORITY
Q_BACKEND_CANONICAL_IR_ONLY
Q_BACKEND_OPERATOR_LEVEL_PINGPONG_ZERO
Q_BACKEND_NULL_TIME_SEMANTICS_CERTIFIED
Q_BACKEND_PIT_PARITY
```

---

# 第十三部分：必须添加的 synthetic tests

## 87. DAG fixtures

```text
Linear: A→B→C→D

Branch:
A→B
 \→C

Diamond:
A→B→D
 \→C↗

Multi-root:
shared A
├ factor1
├ factor2
└ factor3

TS-CS-TS:
ts_mean→rank→ts_mean

State:
EMA→deadband
```

---

## 88. P0 bug tests

```text
test_mixed_cost_source_relation_no_nameerror
test_mixed_cost_uses_duckdb_sql_not_generic_sql
test_native_fraction_polars_uses_polars_capability
test_native_fraction_duckdb_uses_sql_capability
test_delegate_not_counted_native
test_revisited_shared_node_has_same_id
```

---

## 89. Region Plan tests

```text
test_hybrid_route_returns_explicit_regions
test_every_logical_node_assigned_exactly_once
test_every_cross_region_edge_explicit
test_executor_does_not_reroute_region_nodes
```

---

## 90. Shared DAG tests

```text
test_diamond_shared_compute_count_once
test_shared_conversion_count_once_per_representation
```

---

## 91. Parent transfer affinity test

构造：

```text
child Pandas 9ms
child Polars 10ms
Pandas→Polars edge 20ms
parent Polars
```

必须选 child Polars。

---

## 92. CS shard tests

```text
test_rank_rejects_asset_bucket_partition
test_rank_date_partition_matches_full_run
```

---

## 93. state shard tests

```text
test_recursive_time_shard_without_checkpoint_rejected
test_recursive_checkpoint_chain_matches_full_run
```

---

## 94. Polars no-ping-pong

全 Polars-compatible chain：

```text
backend switches = 0
Pandas conversions = 0
```

---

## 95. cost no-double-count fixture

手工：

```text
source=10
compute=20
transfer=5
sink=3
```

总成本必须 38。

---

## 96. memory overlap fixture

```text
source=500MB
target=400MB
scratch=100MB
```

boundary peak 至少约 1GB。

---

## 97. representation-aware CSE fixture

```text
100 Polars consumers
3 Pandas consumers
```

期望：

```text
producer once
Polars→Pandas conversion once
```

---

## 98. direct sink test

Polars final region：

```text
sink directly
```

assert 无 Pandas final materialization。

---

## 99. Batch-global case

构造一个 synthetic DAG，使：

```text
两个 root 单独选 DuckDB
但共享 Polars producer 后全局 Polars 更优
```

Planner 必须选 batch-global 方案。

---

# 第十四部分：Subagents 并行执行设计

## 100. Agent 0 — Architecture / Shared Contracts

先落：

```text
BackendRegion
TransferEdge
PhysicalRegionPlan
Representation
ExecutionAxis
DataShapeEstimate
NativeBuffer protocol
```

**其他 agents 必须基于 Agent 0 接口，不得自行再造。**

---

## 101. Agent 1 — 当前 P0 bugs

负责：

```text
source_lowered
generic sql
native fraction
stable node id
cost double count initial audit
```

---

## 102. Agent 2 — Region Optimizer

负责：

```text
capability annotation
DP[node][backend]
Region coarsening
shared DAG accounting
batch-global planning
```

---

## 103. Agent 3 — Polars

负责：

```text
Lazy Region compiler
multi-root/collect_all
streaming/direct sink
native shared buffers
delegate classification
production no per-op bridge
```

---

## 104. Agent 4 — DuckDB / Arrow

负责：

```text
DuckDB Region compiler
Relation/Arrow boundary
sortedness properties
direct sink
```

---

## 105. Agent 5 — Pandas / Transfer

负责：

```text
Pandas Region compiler
Pandas↔Polars boundary
Arrow adapters
wide↔long
dtype/null boundary
```

---

## 106. Agent 6 — q/K Scaffold

负责：

```text
q capability
q emitter
q runtime adapter
Arrow/q bridge
q type/null/time contracts
q availability policy
```

没有 q runtime：

```text
live q tests = NOT_APPLICABLE
```

但结构/mocks 必须完成。

---

## 107. Agent 7 — Cost / DataShape / Source

负责：

```text
metadata-only shape
source scan cost
transfer/sort/reshape
memory estimate
static priors
```

禁止要求大真实数据。

---

## 108. Agent 8 — CSE / Memory / Materialization

负责：

```text
representation-aware CSE
NativeBufferStore
liveness
spill
recompute/materialize
direct sink integration
```

---

## 109. Agent 9 — Scheduler / Sharding / State

负责：

```text
ExecutionAxis enforcement
resource tokens
TS/CS sharding
rolling overlap
recursive checkpoint chain
backpressure
```

---

## 110. Agent 10 — Semantic Parity

负责：

```text
NaN/null/inf
rank ties
std ddof
rolling
quantile
group
timezone
price basis
PIT/snapshot
```

---

## 111. Agent 11 — Evidence / Acceptance

最后执行：

```text
hard gates
synthetic DAG tests
telemetry assertions
CURRENT_HEAD binding
final acceptance
```

不得提前把 NOT_RUN 写 PASS。

---

## 112. 合并顺序

```text
Agent 0
  ↓
Agents 1~10 parallel
  ↓
主代理 integration / conflict resolution
  ↓
full tests
  ↓
Agent 11 evidence
  ↓
final HEAD acceptance
```

---

## 113. 文件 ownership

核心文件：

```text
plan_cost_router.py
physical_plan.py
engine.py
```

不要多个 subagents 同时大改。

优先：

```text
新模块各自实现
最后 integration agent 接线
```

---

# 第十五部分：完成标准

## 114. P0 DoD

```text
[ ] source_lowered bug 修复
[ ] generic SQL identity 修复
[ ] native fraction backend-specific
[ ] shared node stable id
[ ] PhysicalRegionPlan 真实存在
[ ] 每 logical node 恰好分配一次
[ ] 每跨 Region edge 显式存在
[ ] Executor 完全服从 RegionPlan
[ ] production per-op rerouting 关闭
[ ] 全 Polars chain 0 Pandas roundtrip
[ ] shared DAG compute 不重复
[ ] DP 考虑 parent transfer
[ ] rank/zscore/neutralize asset shard 被禁止
[ ] recursive state 无 seed time shard 被禁止
[ ] PIT/universe/grain/snapshot 边界保持
```

---

## 115. P1 DoD

```text
[ ] metadata-only DataShapeEstimate
[ ] 主路径无 fixed 3000
[ ] memory representation-aware
[ ] compute/transfer 不 double count
[ ] edge cost 包含 bytes/sort/reshape
[ ] batch-global planner
[ ] representation-aware CSE
[ ] liveness early release
[ ] spill/direct sink
[ ] no forced Pandas final materialization
[ ] ResourceGovernor
[ ] source scan shared
[ ] small-memory profile
```

---

## 116. P2 DoD

```text
[ ] unified capability authority
[ ] typed Representation / ExecutionAxis
[ ] physical plan cache/versioning
[ ] ExplainPlan/telemetry
[ ] q backend scaffold
[ ] q semantic authority = zero
[ ] q PIT/type adapter
[ ] production fallback explicit
[ ] evidence 绑定 final HEAD
```

---

## 117. 最终 Acceptance Report 必须能回答

```text
这个 batch 有多少 logical nodes？
多少 shared nodes？
多少 Regions？
每个 Region 选哪个 backend，为什么？
有多少 backend switches？
每个 boundary 搬多少数据？
多少 sort / reshape？
哪个 shared node 保持 native？
哪些 buffer spill？
peak memory 多少？
是否超 budget？
是否有 cross-sectional shard violation？
stateful Region 怎么续 state？
source snapshot 是什么？
planned backend 与 actual backend 是否一致？
是否发生 runtime fallback？
最终是否被强制转 Pandas？
q 是 execution backend 还是 semantic authority？
evidence 是否属于最终 HEAD？
```

---

## 118. 最终执行要求

Claude Code 主代理最后必须提交：

```text
1. issue ledger：每个 MB-P* -> PASS / ALREADY_FIXED / NOT_APPLICABLE
2. changed-file map
3. 最终架构说明
4. tests 命令与结果
5. hard-gate report
6. final HEAD SHA
7. remaining FAIL / NOT_RUN
```

只要 Production P0/P1 仍存在：

```text
FAIL
NOT_RUN
silent fallback
operator-level ping-pong
```

就不得宣布完成。

---

# 最终原则

最终目标不是：

```text
“FactorEngine 支持 Pandas / Polars / DuckDB / q 四个 backend”
```

而是：

> **FactorEngine 成为一个语义统一、PIT 安全、DAG 感知、内存感知、资源感知的多引擎物理执行系统。**

四个 backend 只是 physical engines。

真正决定：

```text
算什么
何时可用
在哪个 universe
共享什么
如何分片
状态如何延续
什么时候跨 backend
是否应该物化/重算/spill
```

的 authority 必须始终属于 FactorEngine canonical planner。
