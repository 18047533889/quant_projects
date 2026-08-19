# DataAccess R30：全维度成熟度、极致读取性能与 FactorEngine 深度协同补强任务书

> **仓库**：`https://github.com/18047533889/quant_projects`  
> **执行时必须先读取最新 `main`**  
> **本次审计观察到的仓库 HEAD**：`8e9893b562f80e083c4baed2e0a15eb4e040cdc9`  
> **DataAccess 最近完整收口基线**：R29 `7c2430c524bc2e00afaeb28e6303990c828556aa`  
> **R29 后重要桥接更新**：`DataReadSession` resolution cache 已切到 ContextVar；DataAccess ResourceGovernor 已继续向 FactorEngine HostResourceCoordinator / shared resource envelope 协作方向演进。  
> **主体范围**：`dataaccess/`。只有在 DataAccess ↔ FactorEngine 边界、批量数据需求合并、资源协调、增量重算接口等确有必要时修改 `factor_engine/`。  
> **总体目标**：在 R24–R29 已经完成的治理、安全、PIT、snapshot、事务和执行链基础上，把 DataAccess 进一步提升为面向 A 股/美股、多频率、多数据源、自动因子挖掘、模型训练、模拟盘和未来生产环境的高性能量化数据运行时。

---

# 0. 给 coding agent 的最高级指令

本轮不是重新发明 DataAccess，也不是把已经闭环的 R24–R29 再写一遍。

当前 DataAccess 已经拥有并应继续复用：

```text
Dataset Registry
SemanticFieldCatalog
RuntimeDatasetContract
DataRequest / ReadPlan / PhysicalPlan
PIT / Temporal Join / Availability
Calendar
SourceSnapshotResolver / SnapshotVerifier
Manifest / Generation / COW
QueryBudget / GlobalResourceGovernor
DataReadSession
Cache / Local Mirror / COS
Security Principal / Policy / CredentialProvider
DuckDB / Polars / Arrow
Factor Lake / Factor Matrix
Lineage / Audit
HTTP Service
```

## 硬约束

1. **先读取执行时最新 HEAD。** 本文列出的项目若已被后续提交真实完成，应标记 `ALREADY CLOSED`，禁止重复实现。
2. 不允许为了速度绕过：
   - PIT；
   - snapshot；
   - schema/semantic contract；
   - authorization；
   - QueryBudget / resource lease。
3. 不允许 FactorEngine production 路径重新裸读 parquet/COS。
4. 不允许 DataAccess 重新实现 FactorEngine 的 operator / alpha 逻辑。
5. A 股与美股必须继续保留各自真实：
   - 单位；
   - currency；
   - PIT；
   - financial flow semantics；
   - calendar/session；
   - instrument identity。
6. Pandas 可以保留在 reference/debug/small metadata，但不得重新进入大型热路径。
7. FE 侧已经有 HostResourceCoordinator / ResourceBroker / auto-shard 等资源体系。DataAccess 只作为共享资源树中的数据读取子系统，禁止再造互相竞争的第二套全局主资源权威。
8. 所有性能优化必须通过 correctness-qualified benchmark 与 regression test。
9. **当前阶段的核心性能原则：同一批因子共享的数据，只物理准备一次。**

---

# 1. 本轮 Definition of Done

完成后应形成下面的主链：

```text
A股 / 美股 / Factor / Event / Minute / Fundamental
Local / COS / CSV / Parquet / future DB
        ↓
Dataset + Semantic Contract
        ↓
Concept / Unit / Grain / PIT / Coverage
        ↓
DataReadSession + ExperimentDataSnapshot
        ↓
FactorBatchDataPlan
        ↓
FieldRequestCoalescer
        ↓
Partition Metadata Index
        ↓
最少物理扫描
        ↓
session-level source reuse
        ↓
DuckDB / Polars / Arrow 成本驱动执行
        ↓
FactorEngine operator DAG
```

必须达到：

```text
factor 数量线性增长
≠
source scan 数量线性增长
```

如果 10,000 个 factor 实际只有 20 个兼容的数据需求组，物理数据准备复杂度应尽量接近 20，而不是 10,000。

---

# 2. 优先级

## P0
当前立即做，直接影响：
- 批量因子吞吐；
- 重复 IO；
- FE×DA 协同；
- 正确性可观测；
- 增量重算。

## P1
当前适合补强，显著提高：
- 多数据源；
- 跨市场；
- 可维护性；
- 生产成熟度。

## P2
只铺接口，不做过度工程化：
- 期货/期权；
- 大规模分布式；
- 更复杂训练数据平台。

---


# R30-P0-001 — [P0] Benchmark Suite：建立固定、可复现的 DataAccess 性能基准

**为什么改**

现在 DataAccess 功能与治理层已经很复杂，继续凭感觉调 DuckDB threads、缓存、batch size 或 manifest 很容易增加复杂度却没有真实收益。后续所有“极致速度”优化必须先有固定基准。

**怎么改**

新增 `dataaccess/benchmarks/`，至少包含：

```text
fixtures.py
benchmark_local_daily.py
benchmark_local_minute.py
benchmark_cos.py
benchmark_join.py
benchmark_factor_batch.py
benchmark_session_reuse.py
benchmark_incremental.py
report.py
```

固定 workload：

- B01：A 股 10 年 daily，3000~5000 股票，OHLCV/Amount/VWAP；
- B02：A 股分钟：单日、单月、一年、分钟→日频；
- B03：daily + fundamental PIT join + universe；
- B04：100 factors shared-source；
- B05：1000 factors mixed-source；
- B06：10000 factors mining-like workload；
- B07：COS cold；
- B08：COS warm；
- B09：单日增量更新。

每次记录：

```text
TTFC
TTDC/data-ready
wall time
rows/s
GB/s
physical object count
physical scan count
bytes scanned
bytes returned
resolution_ms
snapshot_ms
schema_ms
calendar_ms
remote_list_ms
remote_head_ms
governor_wait_ms
duckdb_wait_ms
duckdb_execute_ms
polars_execute_ms
cache hit
DataReadSession reuse
peak RSS
spill bytes
```

**验收**

不要使用跨机器无意义的绝对秒数，使用相对基准：

```text
warm local governed scan >= direct-equivalent backend throughput 的 90%
same-session repeated source 不重复 glob/stat/footer/source-resolution
1000 factor 的 scan 数量接近 unique source demand 数量，而不是 factor 数量
主 benchmark P95 回退 >10% 触发 CI 警告/失败
peak RSS 增加 >15% 触发检查
```

**主要位置**

`dataaccess/benchmarks/`、CI workflow、evidence/report 目录。


# R30-P0-002 — [P0] FactorSourcePlan：让每个 Factor 的数据需求成为机器可读 IR

**为什么改**

FactorEngine 知道 expression 和 operator，DataAccess 知道数据语义。两者之间还应有一个稳定的数据依赖合同，避免 FE 一个 factor 一个 factor 地临时读数据。

**怎么改**

在 FE 侧从表达式 DAG 提取：

```python
FactorSourcePlan(
    factor_id,
    market,
    leaf_concepts,
    source_datasets,
    required_frequency,
    required_grain,
    pit_requirements,
    price_basis,
    aggregations,
    joins,
    coverage_requirements,
)
```

禁止 DA 反向解析 factor expression；dependency extraction 仍属于 FE。

**用途**

- source demand CSE；
- 成本估算；
- 可解释性；
- production 防旁路；
- 后续批量计划输入。

**主要位置**

```text
factor_engine/planner/
factor_engine/storage/sources/data_access_source.py
dataaccess/read/
```

**验收**

同一 factor 编译多次应生成稳定 digest；source/market/PIT/price basis 变化必须改变 plan identity。


# R30-P0-003 — [P0] FactorBatchDataPlan：把 1000/10000 factors 的数据需求先批量规划

**为什么改**

真实工作负载不是“读一张表”，而是大量共享 source 的因子。如果 FE 按 factor 逐个请求，单次 DataAccess 再快也会被固定开销拖死。

**怎么改**

将多个 `FactorSourcePlan` 合并为：

```python
FactorBatchDataPlan(
    factors,
    source_groups,
    canonical_fields,
    time_range,
    universe_id,
    market,
    experiment_snapshot,
)
```

`SourceDemandGroup` 至少按以下维度分组：

```text
dataset
market
source snapshot
frequency
timeframe
PIT policy
universe
security scope
price basis
aggregation recipe
filters
```

**不要错误合并**

例如：
- US quarterly 与 TTM；
- raw price 与 backward-adjusted price；
- 不同 universe；
- 不同 decision policy；
- 不同 security scope。

**验收**

1000 factors 输出：
```text
factor_count
unique_concepts
source_group_count
physical_scan_count
```
并证明 source_group_count 远小于 factor_count。


# R30-P0-004 — [P0] FieldRequestCoalescer：兼容请求自动做列集合并

**怎么改**

在 DataAccess 层新增/扩展通用 request coalescing：

```text
R1 price [Close, Volume]
R2 price [Amount, VWAP]
R3 price [Close, High, Low]
       ↓
one compatible scan:
[Close, Volume, Amount, VWAP, High, Low]
```

合并 key 必须包含：

```text
dataset
source snapshot
market
time range
instrument universe
PIT policy
timeframe
security digest
frequency
price basis
aggregation semantics
```

只对完全兼容请求做 `columns=union(columns)`。

**主要位置**

`dataaccess/read/`、`DataReadSession`、FE batch data bridge。

**验收**

构造 100 个共享 `Close` 的请求，物理读取 Close 只出现一次；不同 timeframe 的请求绝不合并。


# R30-P0-005 — [P0] DataReadSession：从 resolution cache 升级到 source-block reuse

R29 已加入 `DataReadSession`，后续 R38 桥接已经将 resolution cache 改为 ContextVar；这些必须保留。

**下一步**

Session 不仅复用：
```text
path resolution
```
还应可复用：
```text
PreparedRead
ResolvedSource
canonical Arrow block
DuckDB relation/reference
```

建议：

```python
DataReadSession:
    resolution_cache
    prepared_cache
    source_block_cache
```

source-block cache key 必须包含：

```text
dataset
snapshot
time range
universe
canonical fields
filters
PIT contract
security digest
price basis
```

**不能全部预载 RAM**

由成本模型根据：

```text
reuse_count
estimated_size
remote cost
scan cost
available memory
```

决定：
```text
memory materialize
relation reuse
rescan
spill
```

必须服从 FE HostResourceCoordinator → DA ResourceGovernor 的共享资源 envelope。

**验收**

同一 session 中 100 个因子重复依赖 OHLCV：
- source resolution 一次；
- schema/snapshot 证明一次或按 generation 复用；
- 物理扫描次数按 plan 而不是 100 次；
- 并发 session A/B ContextVar 不串。


# R30-P0-006 — [P0] Partition Metadata Index：把 query-time 文件枚举变成 metadata lookup

**为什么改**

大量日文件/分钟文件下，反复 glob、stat、footer 仍会形成显著固定成本。

**怎么改**

为每个大型 dataset 构建 versioned metadata index：

```text
object_uri
partition_values
min_time
max_time
row_count
byte_size
schema_epoch
etag/version/checksum
source_generation
optional key stats
```

可以使用 `dataset_manifest.parquet` 或当前 manifest 扩展，优先复用已有 manifest，不要再造第二套 source truth。

更新发生在：

```text
publisher commit
generation publish
mirror refresh
external source manifest ingestion
```

Query-time：

```text
time range / partition predicate
→ metadata index
→ exact object set
```

而不是 filesystem full glob。

**验收**

同一 immutable generation 的第二次/第 N 次请求，不再触发全目录 glob/stat/footer。


# R30-P0-007 — [P0] CoverageService：把数据 coverage 事实从 FE 下沉为 DataAccess 权威

当前 `factor_engine/storage/sources/data_access_source.py` 已有很细的 coverage contract，包括按年、按日、按股票、按 universe、market/provider/timeframe/source_version。

**职责重新收敛**

DataAccess 负责：
```text
“这个 source/concept 的 coverage 客观是多少”
```

FactorEngine 负责：
```text
“这个 operator/mining policy 最低允许多少 coverage”
```

新增/整合：

```python
CoverageService.describe(
    concept,
    market,
    dataset,
    provider,
    timeframe,
    universe,
    source_snapshot,
)
```

返回：

```text
first_valid_date
last_valid_date
overall coverage
by-year
by-date
by-instrument
quantiles
source_version
snapshot_id
```

迁移期必须做 FE 旧 coverage contract parity test，确认事实没有改变。

**不要**

把 operator-specific threshold 下沉到 DA。


# R30-P0-008 — [P0] CalendarSnapshot：把 PIT calendar 世界变成真正 snapshot

当前 calendar/PIT 已很成熟，下一步重点不是继续加规则，而是可复现。

**定义**

```python
CalendarSnapshot(
    market,
    source_version,
    timezone,
    trading_days_digest,
    session_schedule_digest,
    early_close_digest,
    snapshot_id,
)
```

Digest 必须覆盖：

```text
全部交易日
全部 session open/close
午休
early close
timezone
calendar source/version
```

不要只 hash：
```text
count / first / last
```

**DataReadSession**

进入 job 时冻结 calendar snapshot，同一个 job 所有 factor 使用同一 calendar 世界。

**Factor identity**

所有 PIT-sensitive factor/materialization 带 `calendar_snapshot_id`。

**验收**

只修改中间某一个交易日/early-close，snapshot digest 必须改变。


# R30-P0-009 — [P0] ExperimentDataSnapshot：一次实验冻结完整的数据世界

当前已经有 dataset/source snapshot、registry/contract/build identity，但实际一次回测/挖因子任务需要的是集合 snapshot。

**新增**

```python
ExperimentDataSnapshot(
    experiment_id,
    market,
    datasets={dataset: source_snapshot},
    calendar_snapshot,
    universe_snapshot,
    semantic_contract_digest,
    registry_digest,
    code_build_sha,
    security_scope_digest,
)
```

**使用场景**

```text
factor mining
backtest
model training
weekly batch
simulation replay
```

最终结果只需保存一个 `experiment_data_snapshot_id`，即可追溯完整数据世界。

**验收**

任一 dataset snapshot、calendar、universe、contract 或 build SHA 改变，ExperimentDataSnapshot identity 必须改变。


# R30-P0-010 — [P0] QueryTrace + Observability：精确知道时间和 IO 花在哪里

Audit 解决“谁做了什么”，Observability 解决“系统为什么慢/坏”。

**实现 QueryTrace**

```python
QueryTrace(
    request_id,
    job_id,
    dataset,
    stage_timings,
    scan_bytes,
    result_bytes,
    cache_status,
    resource_wait,
    backend,
    source_snapshot,
)
```

stage 至少：

```text
auth
contract
resolution
mirror
remote_list
remote_head
schema
snapshot
calendar
prepare
governor_wait
duckdb_wait
duckdb_execute
polars_execute
normalize
join
serialize
post_verify
```

**Metrics**

OpenTelemetry/Prometheus 至少支持：

```text
query_duration
scan_bytes
rows
cache_hit
resolution_cache_hit
session_source_reuse
remote_head_latency
remote_list_latency
duckdb_wait
resource_wait
pit_join_latency
errors_by_type
```

FE batch job trace 必须关联 DA request IDs。

**最终要能回答**

```text
1000-factor job = 60s
DA = 12s
operator = 38s
writer = 10s
```

而不是继续凭感觉优化。


# R30-P0-011 — [P0] DataChangeSet：让数据变化驱动 FactorEngine 最小重算

**定义**

```python
DataChangeSet(
    dataset,
    source_before,
    source_after,
    changed_objects,
    changed_partitions,
    changed_time_range,
    changed_instruments,
    changed_columns,
    change_kind,
    revision_availability,
)
```

`change_kind`：

```text
append
correction
revision
delete
schema_change
calendar_change
universe_change
```

DataAccess 负责生成“数据变化事实”。

FactorEngine：
```text
DataChangeSet
→ ChangeImpactDAG
→ affected factors
→ affected windows
→ minimal recompute
```

财务 revision 不能只看物理文件日期，还要结合 knowledge time 与 factor lookback。

**验收**

单公司一条财报修订不允许默认触发“全 A 股全历史所有因子”重算。


# R30-P0-012 — [P0] Canonical ConceptId + UnitType：减少跨市场/LLM 搜索歧义

不要一次性重写所有字段；先落地最核心 typed primitives。

**ConceptId 示例**

```text
price.close
price.vwap
return.close_to_close
valuation.market_cap
financial.revenue
financial.net_income.consolidated
financial.net_income.attributable
financial.total_assets
fundamental.roe
```

**UnitType**

```text
Return(decimal)
Ratio
Price(currency)
Money[CNY]
Money[USD]
Shares
Volume
Count
Days
```

Compile-time 检查：

```text
Money[CNY] + Money[USD]
```
无 FX contract 必须 reject。

原始数据本身不改，仅升级 semantic metadata/compiler。

**FE**

Production expression leaf 最终应绑定 ConceptId + MarketContext，而不是漂亮但模糊的 alias。


# R30-P0-013 — [P0] Production CI + 性能回归 Gate

**CI 分层**

```text
clean checkout/import/wheel
unit
contract
PIT
security
snapshot
backend parity
FE integration
destructive
benchmark smoke
```

性能结果必须绑定：

```text
commit SHA
CPU
RAM
storage
DuckDB
Polars
PyArrow
```

Benchmark regression：

```text
warm throughput -10%
1000 factor TTDC +10%
peak RSS +15%
physical scan amplification 增加
```

应触发 fail 或至少 release-blocking review。

如果当前 GitHub SHA 没有 CI status，不允许 closure report 写“CI green”。


# R30-P1-001 — [P1] SourceAdapter Protocol：统一未来 Parquet/COS/CSV/DB 数据源接入

先审计当前 FormatAdapter/backend abstraction；若已有就扩展，禁止并行重建。

建议接口：

```python
class SourceAdapter(Protocol):
    capabilities()
    resolve(request, contract)
    estimate(resolved)
    scan(prepared)
    snapshot(resolved)
    healthcheck()
```

`SourceCapabilities`：

```text
projection_pushdown
filter_pushdown
partition_pruning
streaming
remote
snapshot
write
transaction
asof
```

以后 ClickHouse/Postgres/vendor HTTP 不需要往 Store 继续堆 if/else。


# R30-P1-002 — [P1] Format Support：把 CSV/Arrow IPC 纳入统一读取而非旁路

用户目标是各种 parquet/CSV/factor data 都统一由 DataAccess 读。

至少正式支持：

```text
Parquet
CSV
Arrow IPC/Feather
```

CSV 主实现优先：

```text
PyArrow CSV
Polars scan_csv
DuckDB read_csv
```

不要大型数据 `pandas.read_csv`。

Production CSV contract 必须显式定义：

```text
schema
delimiter
encoding
header
null representation
timestamp parse
```

不能完全依赖自动 dtype inference。


# R30-P1-003 — [P1] MarketProfile：把 A股/美股市场规则收敛成一等对象

建议：

```python
MarketProfile(
    market_id,
    timezone,
    default_currency,
    calendar_id,
    session_policy,
    instrument_namespace,
    settlement_policy,
)
```

当前只落地：
```text
ASHARE
US
```

未来 HK/futures/crypto 扩展时，不需要到处加 market if。

注意：MarketProfile 不能替代 Dataset Contract，只承载稳定的市场级规则。


# R30-P1-004 — [P1] InstrumentIdentity / InstrumentType：为 ticker rename 与未来多资产铺底

区分：

```text
display ticker/symbol
stable security id
```

至少定义：

```text
EQUITY
ETF
INDEX
FUTURE
OPTION
BOND
FX
CRYPTO
```

当前无需完整实现 futures/options；只保证核心 identity 不假设“所有 instrument 永远是一个 ticker”。

美股 ticker reuse/rename 尤其需要 stable identity。


# R30-P1-005 — [P1] FrequencySpec：替代散落字符串

定义：

```python
FrequencySpec(
    unit="minute",
    multiplier=5,
    session="regular",
    timezone="America/New_York",
)
```

支持：
```text
tick/event
1m/5m/30m
daily
weekly
snapshot
```

FrequencySpec 仅描述频率；真正 source→target 转换仍必须有 AggregationSpec，不能因为 requested frequency 不同就隐式猜 resample。


# R30-P1-006 — [P1] GrainSpec：把“一行代表什么”机器化

定义：

```python
GrainSpec(("trade_date","instrument"))
GrainSpec(("knowledge_date","instrument","period_end"))
GrainSpec(("trade_date","instrument","industry_source"))
```

JoinPlanner 在执行前检查：

```text
1:1
N:1
1:N
N:N
```

例如 IndustrySource 未过滤导致右表多行时，应在 join 前直接发现 grain 不唯一，而不是结果行数爆炸后才发现。


# R30-P1-007 — [P1] Semantic Schema Registry：不仅管 dtype，也管定义版本

扩展现有 SchemaEpoch/semantic contract：

```python
SemanticColumnVersion(
    concept_id,
    physical_column,
    dtype,
    unit,
    definition_version,
    valid_from,
    valid_to,
)
```

这样供应商如果：
```text
列还是 double
但 % → decimal
```
也形成 semantic epoch，不会被普通 schema union 掩盖。


# R30-P1-008 — [P1] RevisionFidelity：把“PIT”拆成真实能力等级

明确枚举：

```text
NONE
KNOWLEDGE_DATE
INGESTION_VINTAGE
TRUE_VENDOR_VINTAGE
```

A 股当前若没有完整历史 revision vintage，不得用 `PIT=True` 模糊表达。

FactorArtifact/Experiment snapshot 应记录 revision fidelity。


# R30-P1-009 — [P1] UniverseSnapshot：把 universe construction 也变成可复现对象

定义：

```python
UniverseSnapshot(
    universe_id,
    market,
    source_snapshot,
    membership_policy_version,
    tradability_policy_version,
    snapshot_id,
)
```

同样叫 CSI300，如果：
```text
ST处理
停牌处理
上市天数
可交易规则
```
不同，snapshot 必须不同。


# R30-P1-010 — [P1] PriceBasisSpec：把 raw/复权/总收益价显式化

定义：

```text
RAW
FORWARD_ADJUSTED
BACKWARD_ADJUSTED
TOTAL_RETURN
POINT_IN_TIME_ADJUSTED
```

FactorSourcePlan 带 price basis。

尤其：
```text
breakout
52-week high
gap
return
volatility
```
不能隐式混用不同 basis。


# R30-P1-011 — [P1] Join Cost Planner：在 temporal correctness 上再优化 physical join strategy

输入：

```text
row count
bytes
selectivity
unique keys
time range
coverage
local/remote
```

输出：

```text
join order
build side
filter-before-join
pre-aggregation
materialize/not
backend
```

第一版只需 heuristic：

```text
small dimension first
filter before join
minute aggregate before fundamental join
```

不要一开始写复杂优化器。


# R30-P1-012 — [P1] Aggregation Recipe Identity：分钟→日频的所有隐含政策进入 identity

Aggregation identity 至少包含：

```text
source frequency
market session
minute window
timezone
halt policy
missing-bar policy
early-close policy
price basis
aggregation function
```

进入：
```text
FactorSourcePlan
cache key
materialization metadata
```

避免两个都叫 VWAP，但计算时段/停牌处理不同却复用缓存。


# R30-P1-013 — [P1] CacheHierarchy：明确每层缓存的责任与失效机制

当前已有多种 cache，下一步需要统一模型。

建议：

```text
L0 plan/request cache
L1 DataReadSession source-block cache
L2 process metadata/query cache
L3 local SSD/COS cache
L4 remote source
```

每层文档化并实现：

```text
identity
security scope
snapshot scope
TTL
max bytes
eviction
invalidation trigger
```

禁止四个 cache 各自猜 freshness。


# R30-P1-014 — [P1] ExecutionLease：继续统一 DA 与 FE 的资源树

当前 FE 已有 HostResourceCoordinator，DA governor 也已有 bridge 更新。

最终资源链：

```text
HostResourceCoordinator
    ↓ job envelope
DA ResourceGovernor
    ↓
DA ExecutionLease
```

Lease 包含：

```text
memory
scan bytes
remote slots
duckdb slots
temp disk
spill budget
absolute deadline
```

DataAccess 不建立第二个独立 auto-sharder，与 FE 的 shard/replan 保持清晰边界。


# R30-P1-015 — [P1] Location-independent Snapshot：逻辑数据身份与读取位置分离

FactorEngine 应只知道：

```text
source snapshot S123
```

而不是：
```text
local or COS
```

同一个 snapshot 可由：
```text
local mirror hit
remote object
```
执行。

location 进入 execution provenance，但不改变 logical source identity。


# R30-P1-016 — [P1] 重要 mutable dataset 统一 immutable generation + manifest pointer

优先覆盖：

```text
factor matrix
published factor artifacts
derived clean datasets
```

使用：

```text
immutable objects
generation manifest
atomic pointer
```

减少原地 overwrite。

Vendor/raw mirror 是否 generation 化由 mutation_owner/source contract 决定，不要强行全部改。


# R30-P1-017 — [P1] Policy Manifest：把服务器权限策略版本化

定义：

```python
PolicyManifest(
    policy_version,
    principal_mappings,
    dataset_classifications,
    factor_entitlements,
    digest,
)
```

进入：
```text
ExecutionContext
audit
lineage
experiment snapshot
```

这样 server A/B policy 漂移可以直接检测。


# R30-P1-018 — [P1] CredentialScopeId：让云权限范围进入 cache / execution identity

不保存 secret，只保存：

```text
credential_scope_id
```

表示：
```text
这份 credential 的权限范围
```

进入：
```text
security digest
cache scope
audit
source execution identity
```

长期优先 server role / STS，减少静态 Secret。


# R30-P1-019 — [P1] Lineage Store：让 lineage 可查询，而不是只存在日志里

先用简单的 DuckDB/SQLite/Parquet index 即可，不需要上复杂图数据库。

记录：

```text
run_id
factor_id
dataset
concept
source_snapshot
experiment_snapshot
principal
build_sha
timestamp
```

支持：
```text
某 factor 用过哪些 dataset？
某 dataset revision 影响哪些 factor？
某 premium source 被谁读过？
```

后续可直接服务 ChangeImpact。


# R30-P1-020 — [P1] BackendCapabilities：Planner 不再散落 backend == duckdb 判断

每 backend 声明：

```text
projection_pushdown
filter_pushdown
partition_pruning
asof_join
streaming
lazy
remote
window
groupby
write
```

Planner 根据 capability + cost 决策。

正确性仍由 backend parity certification 保证。


# R30-P1-021 — [P1] Production FactorEngine 只允许 Canonical Concept/Field

生产 FE 不应直接依赖 physical column name。

允许 physical name 的范围：
```text
DataAccess adapter
migration/admin
debug/reference
```

Expression leaf 最终绑定：
```text
ConceptId + MarketContext
```

这样 A/US 同名/别名不会在 FE 内重新猜。


# R30-P1-022 — [P1] MiningFieldProfile：让自动挖因子知道数据成本与可用边界

DataAccess 提供数据事实：

```python
MiningFieldProfile(
    concept_id,
    mining_allowed,
    PIT_fidelity,
    coverage_class,
    cost_class,
    frequency,
    grain,
)
```

可补：
```text
recommended operator families
forbidden operator families
```
但最终 operator policy 权威仍在 FE。

例如 sparse event / snapshot-only field 不应被 LLM 当普通 daily panel 随便滚 252 日。


# R30-P1-023 — [P1] Field Capability Catalog：给 LLM/冷启动库机器可读合法搜索空间

Catalog 输出：

```text
concept
market
dataset
frequency
grain
unit
PIT
coverage
cost class
source availability
```

LLM/AlphaProbe 先在合法 field space 内生成表达式，减少：
- 不存在字段；
- PIT 不合格字段；
- coverage 太低字段；
- 代价极高却无必要的数据需求。


# R30-P1-024 — [P1] FactorArtifact Metadata：物化结果保存完整数据与语义身份

Factor artifact metadata 至少绑定：

```text
factor definition
operator semantic versions
FactorSourcePlan digest
ExperimentDataSnapshot ID
universe snapshot
calendar snapshot
coverage policy
security classification
build SHA
```

不能只保存 factor expression/value。


# R30-P1-025 — [P1] Data Quality Service：把 source certification 变成正式服务

复用已有 DQ，不重建平行体系。

输出：

```python
DataQualityResult(
    dataset,
    source_snapshot,
    checks,
    severity,
)
```

关键检查：

```text
row-count drift
missing partition
duplicate key
critical null
OHLC consistency
negative volume
financial period <= knowledge
currency enum
coverage
schema/semantic epoch
```

DQ 不自动篡改数据；负责 PASS/WARN/BLOCK。


# R30-P1-026 — [P1] mutation_owner 变成 runtime contract，不只是配置备注

正式枚举：

```text
DATAACCESS
EXTERNAL_VERSIONED
EXTERNAL_MUTABLE
IMMUTABLE
```

处理策略：

```text
DATAACCESS -> epoch/generation
EXTERNAL_VERSIONED -> publisher manifest/source generation
EXTERNAL_MUTABLE -> refresh/list/checkpoint
IMMUTABLE -> content identity
```

避免外部 writer 改了文件但 DA 自己的 source_epoch 没变化。


# R30-P1-027 — [P1] Current-state 文档：让 R24-R29 退出主使用路径

新增并长期维护：

```text
dataaccess/docs/DATAACCESS_ARCHITECTURE.md
dataaccess/docs/DATAACCESS_DEVELOPER_GUIDE.md
dataaccess/docs/DATAACCESS_OPERATIONS_RUNBOOK.md
```

Architecture 只描述当前系统，不讲历史修复过程。

Developer Guide：
```text
如何新增 dataset
如何新增 semantic concept
如何新增 market/source adapter
如何写 PIT contract
如何写 benchmark/test
```

Operations：
```text
COS 403
manifest stale
snapshot mismatch
calendar unavailable
disk full
cache corruption
```

R24-R29 进入 archive/reference。


# R30-P1-028 — [P1] 版本治理拆分：package version 之外增加 contract/schema/API 版本

明确：

```text
API_VERSION
CONTRACT_SCHEMA_VERSION
REGISTRY_SCHEMA_VERSION
SEMANTIC_SCHEMA_VERSION
STORAGE_FORMAT_VERSION
```

不要所有 compatibility 都用 Python package `0.x.y` 表示。

build SHA 继续进入 snapshot/lineage。


# R30-P2-001 — [P2] TrainingDatasetSpec：为未来模型训练建立正式数据快照合同

定义接口，不需要当前大改模型体系：

```python
TrainingDatasetSpec(
    features,
    labels,
    universe,
    time_range,
    train_valid_test,
    purge,
    embargo,
    missing_policy,
    normalization_policy,
    experiment_snapshot,
)
```

DataAccess 负责 dataset assembly / snapshot；
Model layer 负责训练。


# R30-P2-002 — [P2] 多资产 Instrument Contract：只铺 futures/options 所需字段

预留：

```text
underlying
contract_id
expiry
multiplier
strike
option_type
```

当前不开发完整期货/期权数据平台，只确保核心 identity/grain 模型不会假设所有资产都是永久股票 ticker。


# R30-P2-003 — [P2] Distributed Provider Interfaces：只铺接口，不现在上重分布式

定义：

```python
DistributedLeaseProvider
DistributedLockProvider
DistributedMetadataStore
```

默认 local implementation。

等未来多机规模真正需要时再替换 Redis/Postgres/其他实现，不要当前阶段就引入复杂基础设施。


# R30-P2-004 — [P2] API Surface 收敛

主文档只推荐：

```text
store.read()
store.scan()
store.session()
store.plan()
store.write()
store.publish()
```

`read_arrow/read_auto/read_frame/...` 可保留 compatibility，但不要继续新增同义 public entry。

目标是减少维护与用户心智成本。


---

# 49. FactorEngine × DataAccess 最终责任边界

## DataAccess 负责

```text
dataset identity
source identity
physical storage
schema
market
instrument identity
unit/currency
grain
PIT clocks
availability
calendar
universe source
coverage facts
snapshot
security
source aggregation
temporal join
data read execution
data change facts
```

## FactorEngine 负责

```text
factor expression
operator semantics
operator parameter domain
factor DAG
data dependency extraction
FactorSourcePlan
batch factor grouping
operator execution
alpha/model logic
materialization request
```

## 共同边界

```text
FactorSourcePlan
FactorBatchDataPlan
DataReadSession
ExperimentDataSnapshot
DataChangeSet
shared resource envelope
```

---

# 50. FE × DA 高吞吐主链

最终应尽量接近：

```text
10,000 expressions
        ↓
FE compile / analyze
        ↓
extract Concept requirements
        ↓
FactorSourcePlan
        ↓
FactorBatchDataPlan
        ↓
FieldRequestCoalescer
        ↓
DataReadSession
        ↓
ExperimentDataSnapshot
        ↓
Partition Metadata Index
        ↓
one/few physical scans
        ↓
session source-block reuse
        ↓
FE operator DAG
        ↓
materialization
```

## 核心成功指标

```text
factor_count ↑
physical_scan_count 不线性 ↑
```

---

# 51. 小/中/大任务不同 fast path

## 1~5 factors

不要为了 batch planner 产生巨大固定 overhead。

直接 fast path：
```text
compile → coalesce → read
```

## 10~500 factors

```text
field coalesce
DataReadSession reuse
```

## 500+

```text
FactorBatchDataPlan
source grouping
selective source-block materialization
```

## 超大任务

使用 FE 当前的：

```text
HostResourceCoordinator
AutoShard/Replan
```

DataAccess 只消费数据侧 resource envelope，不独立建立第二套全局 auto-sharder。

---

# 52. DuckDB / Polars 路由

不要规定“全部 DuckDB”或“全部 Polars”。

## DuckDB 更适合

```text
Parquet pushdown
large join
ASOF/SQL
aggregation
remote scan
```

## Polars 更适合

```text
in-memory expression
column transform
Arrow-adjacent chain
```

Planner 根据：

```text
capability
cost
source size
reuse
```

路由。

所有 optimized route 都必须有 parity evidence。

---

# 53. Zero-copy / 少物化

热路径尽量：

```text
DuckDB ↔ Arrow ↔ Polars
```

避免：

```text
Arrow → pandas → Arrow
```

尤其：

```text
minute
factor matrix
generation writer
large join
```

可以加静态 audit：

```text
production hot path 出现 .to_pandas()
=> 未在 allowlist 则 fail
```

允许：

```text
reference
debug
small metadata
```

---

# 54. DataReadSession Memory 策略

source block cache 不能把“少 IO”换成 OOM。

所有 session cache entry 都有：

```text
size
reuse_count
refcount
last_access
spillable
security scope
snapshot identity
```

服从 shared resource envelope。

成本决策：

```text
reuse_count × rescan_cost > materialize_cost
=> materialize
```

否则继续 scan。

---

# 55. COS / Remote 性能

理想路径：

```text
source generation manifest
→ exact objects
→ pooled client
→ bounded parallel access
```

同一个 job：

```text
generation identity 只证明一次
```

不要：
```text
每个 factor 重新 LIST / HEAD。
```

如果 upstream generation immutable 可证明，优先 generation-level proof，而不是重复 2N HEAD。

---

# 56. Data Read Cost Model

每个 SourceDemandGroup 至少估：

```text
object count
scan bytes
rows
projection ratio
selectivity
local/remote
estimated memory
reuse count
```

用于决定：

```text
scan once+materialize
vs cheap rescan
join order
backend
```

第一版 heuristic 即可。

---

# 57. Incremental 示例

一条 A 股利润表 revision：

```text
dataset = ashare_stock_income
instrument = A
knowledge_date = 2026-08-10
period = 2026Q2
columns = NetProfit...
change_kind = revision
```

DataAccess 只描述变化。

FactorEngine：
```text
find dependent factors
+ rolling lookback
+ cross-sectional impact
→ minimal recompute
```

---

# 58. Multi-market 明确规则

未来如果做 A+US global factor：

必须显式：

```text
MarketContext
CalendarAlignmentPolicy
CurrencyPolicy
CrossSectionPolicy
```

不能默认：
```text
相同 YYYY-MM-DD = 同一个可比截面。
```

---

# 59. Event / News

Event 表不能当普通 daily panel。

需要显式：

```text
event_id
entity mapping
knowledge timestamp
source id
```

Event→daily 必须带：

```text
aggregation window
decision cutoff
timezone
```

进入 source recipe identity。

---

# 60. Historical + Streaming Hybrid

未来模拟盘需要：

```text
historical immutable generation
+
live sequence checkpoint
```

定义：

```text
HybridReadSnapshot
```

这样能复现：
```text
截至 live sequence N 的数据世界。
```

当前先铺接口即可。

---

# 61. Production Source Certification

每个 production-readable dataset/concept 建议最终具备：

```text
schema status
PIT status
revision fidelity
coverage
freshness
DQ
security classification
snapshot capability
mining eligibility
```

FE mining catalog 默认只暴露 certified source。

---

# 62. 不要做的伪优化

禁止：

1. 为 benchmark 关闭 PIT；
2. 为快跳过 snapshot；
3. FE 重新直接 scan parquet；
4. 把全部历史数据一次性载入 RAM；
5. 无脑增加 DuckDB threads；
6. 每个 source 再写一个 `read_xxx()`；
7. 新建第二套 FE DataLoader；
8. 删除 semantic/schema validation 来换速度；
9. 用 pandas 把 typed join 简化掉；
10. 用新的 cache 掩盖 source identity 问题。

正确方向：

```text
compile once
prove once
reuse many
```

---

# 63. 实施顺序

## Phase 1 — 测量

```text
Benchmark Suite
QueryTrace
```

没有测量，不做性能重构。

## Phase 2 — FE 数据需求批量化

```text
FactorSourcePlan
FactorBatchDataPlan
FieldRequestCoalescer
```

## Phase 3 — Session 深化

```text
PreparedRead reuse
source-block reuse
memory/spill policy
```

## Phase 4 — Metadata 热路径

```text
Partition Metadata Index
CoverageService
CalendarSnapshot
```

## Phase 5 — 完整复现

```text
UniverseSnapshot
ExperimentDataSnapshot
```

## Phase 6 — Incremental

```text
DataChangeSet
FE ChangeImpact integration
```

## Phase 7 — Semantic typing

```text
ConceptId
UnitType
GrainSpec
MarketProfile
PriceBasisSpec
```

## Phase 8 — Production maturity

```text
Observability
Perf CI
Lineage Store
Current-state docs
```

---

# 64. 如果工作量过大，先做这 10 项

按 ROI 排：

```text
1. Benchmark Suite
2. QueryTrace
3. FactorSourcePlan
4. FactorBatchDataPlan
5. FieldRequestCoalescer
6. DataReadSession source-block reuse
7. Partition Metadata Index
8. CoverageService
9. CalendarSnapshot + ExperimentDataSnapshot
10. DataChangeSet → FE minimal recompute
```

这 10 项最直接提升：

```text
速度
可解释
可复现
自动挖掘效率
增量效率
```

---

# 65. 验收矩阵

## Correctness

必须保持：

```text
PIT regression = 0
A/US unit regression = 0
snapshot regression = 0
security regression = 0
backend parity retained
```

## Batch efficiency

1000 factors：

```text
factor_count = 1000
source_group_count << 1000
physical_scan_count ≈ unique compatible source groups
```

## Session metadata

同 DataReadSession：

```text
第二次相同 source demand
不再 full glob/stat/footer/source resolve
```

## Memory

达到 session cache 上限：

```text
evict / spill / rescan
```

不能 OOM。

## Incremental

单一 source revision：

```text
不默认全历史重算。
```

---

# 66. Permanent A/US Regression

A 股至少：

```text
Return bp → decimal
ROE % → decimal
financial PubDate PIT
YTD financial flow
industry source grain
tradability/universe
```

美股至少：

```text
Ret decimal unchanged
ROE decimal unchanged
filing_date PIT
timeframe exactly one
USD money
StockCapital split vs shares
dividend declaration/effective
```

---

# 67. 静态审计

新增/扩展：

```text
audit_dataaccess_source_contracts.py
audit_factorengine_raw_io.py
audit_semantic_concept_coverage.py
audit_dataaccess_perf_paths.py
```

FE production 发现：

```text
pd.read_parquet
pl.scan_parquet
raw duckdb read_parquet path
raw COS path
```

不在 allowlist：

```text
CI fail
```

---

# 68. Benchmark Evidence

每次正式 release 生成：

```text
dataaccess/docs/evidence/r30/
    BENCHMARK_ENV.json
    BENCHMARK_RESULTS.parquet
    BENCHMARK_SUMMARY.md
    FE_DA_1000_FACTOR_TRACE.json
    FE_DA_10000_FACTOR_TRACE.json
```

必须绑定最终 `commit_sha`。

---

# 69. 最终 Closure Report

coding agent 完成后必须输出：

```text
DATAACCESS_R30_FINAL_ACCEPTANCE_REPORT.md
```

至少包含：

## A. Baseline

```text
start SHA
final SHA
git status
changed files
```

## B. Closure Ledger

每项：

```text
DONE
ALREADY CLOSED
DEFERRED
BLOCKED
```

## C. Benchmark

前后：

```text
B01
B02
B03
B05
B06
```

## D. FE Batch Integration

至少：

```text
100 factors
1000 factors
10000 factors
```

记录：

```text
unique concepts
source groups
physical scans
scan bytes
DA time
peak RSS
```

## E. Correctness

```text
PIT
unit
coverage
snapshot
security
backend parity
```

## F. CI

必须是当前最终 SHA。

如果当前 SHA 没有 GitHub CI evidence：

```text
明确写 NO CURRENT-HEAD CI EVIDENCE
```

不能用本地 pytest 冒充。

---

# 70. 最终目标状态

DataAccess 不再被定义为：

```text
“统一读 parquet 的库”
```

而应达到：

> **面向量化计算的高性能、PIT-safe、snapshot-safe、多市场、多频率、多数据源 semantic data runtime。**

最终最重要的系统分工：

```text
FactorEngine 负责算什么
DataAccess 负责把正确的数据以最少物理 IO 准备好
```

因此本轮最核心的一句话：

> **不是继续让每次读取更快，而是让同一批因子中重复的数据准备尽可能消失。**
