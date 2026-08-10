# R25 — DataAccess 全平台最终收口：权限、PIT、物理分区、Source Snapshot、一致性、跨市场语义、并发、缓存、服务化与 FactorEngine 协作整改提示词

> **仓库**：`https://github.com/18047533889/quant_projects`  
> **当前审计基线**：`main@7b15a5e7a7a769734f7d1e003a6b0867c1ce88c9`  
> **审计时间**：2026-08-10  
> **执行对象**：直接修改代码的编码 AI  
> **本轮性质**：DataAccess 全平台收口，不是继续堆功能。  
> **主体**：`dataaccess/`；仅在 DataAccess ↔ FactorEngine 边界需要时修改 `factor_engine/`。  
> **目标**：一次性解决当前已验证的 P0/P1、把已经出现的多套契约真正收敛成 runtime 单一事实源，并为多服务器、多权限、多市场、自动因子挖掘、大规模并发运行建立不会静默出错的生产不变量。

---

# 0. 开始前：这次不要“继续重写一遍 DataAccess”

当前 DataAccess 已经有很多正确的基础设施：

- Dataset Registry
- DataRequest / CompiledDataRequest
- ReadPlan
- QueryBudget
- SemanticFieldCatalog
- COSDatasetContract
- ContractIR
- PIT / temporal join
- session calendar
- manifest / snapshot
- generation
- mutation lock
- schema validation
- Polars / DuckDB / PyArrow 多后端
- local mirror / remote httpfs / CLI cache
- factor lake / matrix integration
- lineage / audit
- strict semantics

**这些不是本轮要推翻的。**

本轮最重要的思想是：

> 不再继续造第 4、第 5、第 6 套配置，而是把已经存在的正确抽象真正串起来，关闭“每一层单独正确、组合起来仍然出错”的漏洞。

尤其不要：

```text
再新建一套新的 YAML
再复制一份 PIT 表
再复制一份 market/unit mapping
再让 mirror.py 自己维护一套 layout
再让 remote.py 自己猜一套 layout
再让 FactorEngine 自己重新猜 DataAccess 已经知道的语义
```

---

# 1. 本轮问题分级

为了避免 AI 把所有问题都当成同等严重，必须按下面三类处理。

## A. VERIFIED CURRENT DEFECT

当前 HEAD 已经能从代码直接验证的缺陷或契约矛盾。

这些必须优先修，不能只留 TODO。

## B. VERIFIED ARCHITECTURAL GAP

当前代码已经存在这个结构性缺口，只是未必已经造成可观察事故。

这些需要本轮把不变量补上。

## C. HIGH-PROBABILITY SCALE RISK

当前规模可能暂时不出问题，但在：

```text
更多 worker
更多服务器
更大 COS
更多 Factor
HTTP 服务化
长时间运行
自动化挖因子
```

以后高概率出事故。

这类不要求把 DataAccess 变成一个超大型云平台，但必须把最小生产护栏补上。

---

# 2. 最终系统必须满足的 12 条核心不变量

本轮所有改动都围绕下面 12 条。

## INV-01：代码仓库不等于数据权限

```text
clone GitHub repo
≠
获得 COS 数据权限
```

真正权限必须由：

```text
DataPrincipal
∩ DataAccess AccessPolicy
∩ Registered Dataset Boundary
∩ COS CAM/IAM/STS
∩ Local Cache Ownership
```

共同决定。

---

## INV-02：一个 production read 必须绑定唯一可证明的 source snapshot

不能出现：

```text
Jan = local old
Feb = remote new
```

却宣称这是一个统一 snapshot。

---

## INV-03：Physical partition clock 与 query/PIT clock 必须分离

必须明确：

```text
partition_time
knowledge_time
effective_time
period_time
decision_time
```

不能把 request 的 `time_range` 直接拿来拼任何 `{date}.parquet`。

---

## INV-04：PIT 无法证明时 production 必须失败

尤其：

```text
calendar 不可用
knowledge clock 不可信
date-only filing 没有盘中时间
revision availability 不存在
```

不能通过 fallback 猜。

---

## INV-05：同一 logical field 必须只有一套 authoritative source semantics

DataAccess runtime 不允许同时分别从：

```text
datasets.yaml
COSDatasetContract
SemanticFieldCatalog
FactorEngine Provider
```

各自重新解释一次。

---

## INV-06：required filter 是数据语义，不是 UI 提示

例如：

```text
US finance timeframe
IndustrySource
IndexSymbol
currency
```

缺失必须真正阻止错误数据进入结果。

---

## INV-07：自动化研究不能享受“warning 后继续”的宽松语义

AlphaProbe / LLM mining 是机器，不会看 warning。

自动化研究在：

```text
PIT
field ambiguity
unit
required filters
source identity
```

方面必须接近 production strict。

---

## INV-08：Derived data 权限不得低于最敏感输入

```text
restricted raw data
→ factor
→ matrix
→ model feature
```

不能通过派生数据“洗白”。

---

## INV-09：单位相同不等于经济定义相同

跨市场必须同时证明：

```text
unit comparable
currency compatible
definition comparable
temporal semantics compatible
provider coverage compatible
```

---

## INV-10：一个查询合法不代表全服务器合法

必须有 server-level admission / resource governance。

---

## INV-11：本地缓存完整不代表远端仍是同一版本

Freshness 必须绑定 source identity，不只是本地 checksum。

---

## INV-12：任何执行结果必须可回答

```text
谁读的？
从哪台 server？
用什么权限？
读了哪些 exact source objects/generation？
用了什么 PIT？
什么 calendar？
什么 semantic/unit contract？
什么 backend/version？
最后生成了什么？
```

---

# 3. 当前最优先的 P0 总表

下面这些先解决，之后再做 P1/P2。

| ID | 问题 | 类型 | 当前判断 |
|---|---|---|---|
| R25-P0-001 | US finance `period_files` 与 MirrorSpec `daily_parquet` 冲突 | VERIFIED CURRENT DEFECT | 必修 |
| R25-P0-002 | US StockCapital split/shares filename pattern 未进入 MirrorSpec | VERIFIED CURRENT DEFECT | 必修 |
| R25-P0-003 | Store dataset-level required filters 漏 `required_event_filters` | VERIFIED CURRENT DEFECT | 必修 |
| R25-P0-004 | `timeframe` 允许多值，没有 exactly-one 约束 | VERIFIED CURRENT DEFECT | 必修 |
| R25-P0-005 | AvailabilityCompiler 在 calendar=None 时 strict 仍 return knowledge | VERIFIED CURRENT DEFECT | 必修 |
| R25-P0-006 | production 自动解析 `~/.cos.yaml` / 注入全局 env secret | VERIFIED ARCHITECTURAL GAP | 必修 |
| R25-P0-007 | registered COS prefix 被当成 authorization | VERIFIED ARCHITECTURAL GAP | 必修 |
| R25-P0-008 | local mirror/cache 不继承 server/principal 权限 | VERIFIED ARCHITECTURAL GAP | 必修 |
| R25-P0-009 | remote wildcard 被当成一个 FileVersion/snapshot object | VERIFIED CURRENT DEFECT | 必修 |
| R25-P0-010 | remote wildcard 可绕过 `max_scan_files` | VERIFIED CURRENT DEFECT | 必修 |
| R25-P0-011 | direct DuckDB remote read 不统一执行 strict snapshot verification | VERIFIED ARCHITECTURAL GAP | 必修 |
| R25-P0-012 | local mirror freshness 不验证当前 COS source identity | VERIFIED ARCHITECTURAL GAP | 必修 |
| R25-P0-013 | auto hybrid 可 local/remote 混 source epoch | VERIFIED CURRENT ARCHITECTURE | 必修 |
| R25-P0-014 | full `cos sync` 直接写 live mirror，非 dataset-level atomic | VERIFIED ARCHITECTURAL GAP | 必修 |
| R25-P0-015 | missing object 被 downloader 统一静默跳过 | VERIFIED CURRENT DEFECT | 必修 |
| R25-P0-016 | event/period datasets 被 trade-day expected partition 逻辑错误建模 | VERIFIED CURRENT ARCHITECTURE | 必修 |
| R25-P0-017 | US filing_date date-label 被当 UTC instant | VERIFIED CURRENT SEMANTIC RISK | 必修 |
| R25-P0-018 | A UpdateTime 被混用为 revision order，易造成 revision hindsight | VERIFIED SEMANTIC GAP | 必修 |
| R25-P0-019 | Factor 派生数据权限未形成强制 lineage inheritance | VERIFIED ARCHITECTURAL GAP | 必修 |
| R25-P0-020 | Automated mining 仍可能走 research warning fallback | VERIFIED ARCHITECTURAL GAP | 必修 |

---

# 4. P0-001 — US 财务物理文件布局与 remote/mirror planner 冲突

## 当前事实

`cos_contract_us.py` 已明确：

```text
us_stock_balance
us_stock_income
us_stock_cashflow

temporal_model = E2
knowledge = filing_date
period = period_end
storage_layout = period_files
```

`datasets.yaml` 也明确：

```text
文件名 = period_end
PIT = filing_date
```

但是 `cos/mirror.py` 当前 MirrorSpec：

```python
_spec("us_stock_balance", ..., table="StockBalance")
_spec("us_stock_income", ..., table="StockIncome")
_spec("us_stock_cashflow", ..., table="StockCashFlow")
```

没有指定 layout，所以落到：

```text
daily_parquet
```

`remote.py` 又根据 `MirrorSpec.layout` 生成：

```text
base/{request_day}.parquet
```

这把：

```text
query/decision/knowledge date
```

错误当成：

```text
physical filename date
```

## 为什么危险

例如：

```text
2024-05-10 才 filing 的 Q1 财务
实际对象：
StockIncome/2024-03-31.parquet
```

用户请求：

```text
as_of 2024-05-10
```

当前 remote planner 可能找：

```text
2024-05-10.parquet
```

而不是：

```text
2024-03-31.parquet
```

最终不是简单“慢”，而可能：

```text
漏读真实 filing
或者扫错 physical period
```

## 正确整改

不要只：

```text
把 layout 字符串从 daily 改成 period
```

必须引入：

```python
@dataclass(frozen=True)
class PhysicalPartitionSpec:
    layout: PhysicalLayout
    partition_clock: str | None
    filename_template: str | None
    prefix: str | None
    completeness: MissingPartitionSemantics
```

`PhysicalLayout` 至少：

```text
DAILY_TRADE_DATE
DAILY_CALENDAR_DATE
EVENT_DATE_FILE
PERIOD_END_FILE
PREFIXED_DATE_FILE
HIVE_DATE
HIVE_YEAR
STATIC_SINGLE
PLAIN_GLOB
GENERATION_POINTER
```

US financial：

```text
layout = PERIOD_END_FILE
partition_clock = period_end
filename_template = "{period_end}.parquet"
query_clock = filing_date / decision_time
```

## 关键原则

当：

```text
predicate_clock != partition_clock
```

不能：

```text
用相同 request start/end 展开文件名
```

需要以下之一：

### 方案 A — 推荐

建立 source-side physical index：

```text
source_manifest
object_key
period_min/max
knowledge_min/max
schema_hash
etag/version
```

request：

```text
decision window
```

先用：

```text
knowledge_min/max
```

选物理 `period_end` 文件。

### 方案 B — 过渡方案

对 US financial：

```text
根据 requested knowledge window
扩大一个保守 period lookback
扫描相关 period_end files
然后严格用 filing_date filter
```

宁可多扫，不能漏。

## 测试

构造：

```text
StockIncome/
  2024-03-31.parquet
      filing_date=2024-05-10
```

请求：

```text
time_range=[2024-05-10,2024-05-10]
```

必须能读到这一行。

并验证：

```text
remote
mirror
auto
local
```

结果一致。

---

# 5. P0-002 — US StockCapitalDaily split/shares 物理文件模板冲突

## 当前事实

同一个 COS 目录：

```text
StockCapitalDaily/
```

有两类文件：

```text
2024-01-01.parquet
shares_2024-01-01.parquet
```

它们 schema 和语义完全不同。

Registry 已经正确拆成：

```text
us_stock_capital_split
us_stock_capital_shares
```

但 MirrorSpec 没有 filename pattern。

## 必须改

使用同一个 `PhysicalPartitionSpec`：

### Split

```text
layout = PREFIXED_DATE_FILE
filename_template = "{date}.parquet"
file_selector = numeric-date
```

### Shares

```text
layout = PREFIXED_DATE_FILE
filename_template = "shares_{date}.parquet"
file_selector = shares_
```

所有：

```text
mirror
remote
auto
CLI cache
completeness
```

必须调用**同一个 locator**。

禁止：

```text
mirror.py 自己拼
remote.py 再拼一遍
registry glob 又是一套
```

## CI

加入：

```text
Registry glob
PhysicalPartitionSpec filename_template
MirrorSpec/compiled physical spec
```

一致性审计。

---

# 6. P0-003 — Store 漏掉 `required_event_filters`

## 当前缺陷

US finance contract 已声明：

```python
required_event_filters=("timeframe",)
```

但 Store 当前 dataset gate 只收：

```python
required_panel_filters
required_dimension_filters
```

`_validate_joined_contract_filters()` 同样漏掉 event filters。

所以当前注释虽然写：

```text
US 财务 timeframe 会 dataset-level 强制
```

实际并不完整。

## 改法

废弃：

```python
_dataset_required_filters()
```

这种手工拼 tuple。

改成 ContractIR 编译：

```python
@dataclass(frozen=True)
class FilterRequirement:
    field: str
    scope: Literal["panel","event","dimension","all"]
    required: bool
    cardinality: Literal["any","exactly_one","at_most_one"]
    allowed_values: tuple[Any, ...] = ()
```

RuntimeDatasetContract：

```python
filter_requirements: tuple[FilterRequirement, ...]
```

然后所有路径调用：

```python
validate_filter_requirements(
    contract,
    read_mode,
    params,
    filters,
)
```

必须覆盖：

```text
read_result
read_arrow
read_frame
read_arrow_stream
scan_polars
PyArrow scanner
read_joined
event helper
ReadPlan.execute
SQL registered view
FactorEngine DataAccessSource
HTTP
```

---

# 7. P0-004 — US `timeframe` 必须 exactly-one

当前 `_validate_filters()`：

```text
只要每个元素在 allowed enum 内
```

所以：

```python
timeframe=["quarterly","annual"]
```

会合法。

财务语义上这是错误的。

## 改法

US financial：

```python
FilterRequirement(
    field="timeframe",
    scope="event",
    required=True,
    cardinality="exactly_one",
    allowed_values=(
        "quarterly",
        "annual",
        "trailing_twelve_months",
    ),
)
```

## identity

`timeframe` 必须进入：

```text
CompiledDataRequest digest
DataSnapshot params
ReadLineage
Factor semantic identity
Factor full definition
cache key
coverage key
materialization identity
```

否则季度因子和 TTM 因子可能重用错误 cache/version。

---

# 8. P0-005 — AvailabilityCompiler calendar=None 时存在 fail-open

## 当前代码问题

`compile_available_from()` 当前逻辑先：

```python
if calendar is None or not calendar.has_data:
    return knowledge
```

然后才决定 strict。

这意味着：

```text
availability=next_trading_day
strict=True
calendar=None
```

仍然返回：

```text
knowledge
```

这等价于：

> “无法证明下一交易日，就当现在已经可用。”

PIT 方向完全反了。

## 改法

顺序必须：

```python
strict = strict if strict is not None else is_strict_semantics()

if availability_requires_calendar(availability):
    if calendar is None or not calendar.has_data:
        if strict:
            raise CalendarUnavailableForAvailability(...)
        return DegradedAvailability(...)
```

不要 raw return knowledge。

推荐：

```python
@dataclass(frozen=True)
class AvailabilityResult:
    available_from: Any
    authoritative: bool
    calendar_snapshot_id: str | None
    degradation_reason: str | None
```

Production：

```text
authoritative=false
→ reject
```

Research：

```text
可以返回 degraded
但 lineage 必须记录
```

---

# 9. P0-006/P0-007 — COS credential boundary 与真正授权

## 当前问题

Production 现在仍会：

```text
解析 ~/.cos.yaml
读取 cos.base.secretid / secretkey
写入 COS_SECRET_* env
```

而用户实际希望：

```text
不同 server 有不同 COS 权限
```

如果 `clean-cos-ro` 本身是受限 wrapper，而 `.cos.yaml` base 权限更大，DataAccess 会形成 privilege escalation 路径。

## 最终模型

```python
@dataclass(frozen=True)
class DataPrincipal:
    principal_id: str
    server_id: str | None
    roles: tuple[str, ...]

@dataclass(frozen=True)
class AccessPolicy:
    allowed_datasets: frozenset[str]
    allowed_factor_namespaces: frozenset[str]
    actions: frozenset[str]

@dataclass(frozen=True)
class CredentialMaterial:
    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)
    session_token: str | None = field(default=None, repr=False)
    expires_at: datetime | None = None
    scope_id: str | None = None
```

Credential provider：

```text
Server Role / STS
Explicit deployment env
CLI wrapper
```

## Production 禁止

```text
自动 parse ~/.cos.yaml
AccessDenied 后换 credential
自动 privilege fallback
secret 注入 process-global env
```

## `clean-cos-ro`

保持：

```text
CLI credential boundary
```

DataAccess 只：

```text
subprocess clean-cos-ro ...
```

不解析它的秘密。

---

# 10. P0-008 — Local mirror/cache 必须继承权限

## 必须实现

cache root：

```text
<cache_root>/<principal_scope_id>/<dataset>/<generation>/
```

权限：

```text
directory 0700
file 0600
```

如果团队明确 Unix group：

```text
0750 / 0640
```

需要显式部署配置。

Manifest 写：

```text
principal_scope_id
access_policy_digest
credential_scope_id
source_snapshot_id
```

低权限 principal：

```text
不能复用高权限 cache
```

权限降级后：

```text
旧 cache 自动 invalid
```

---

# 11. P0-009/P0-010 — Remote wildcard 不是物理 snapshot，也不能算“1 个文件”

## 当前严重问题

`build_file_manifest()` 当前遇到：

```text
s3://bucket/table/*.parquet
```

会创建一个：

```python
FileVersion(path="s3://bucket/table/*.parquet")
```

而不是 enumerate 所有真实对象。

结果两个问题：

### Snapshot 假

Query 实际读：

```text
10000 个 object
```

lineage 却只有：

```text
一个 wildcard URI
```

### QueryBudget 被绕

`max_scan_files` 看：

```text
len(FileVersion)=1
```

而实际可能：

```text
10000
```

## 改法

Production remote physical planning 必须先得到：

```python
@dataclass(frozen=True)
class ResolvedObject:
    uri: str
    etag: str | None
    version_id: str | None
    content_length: int
    last_modified: datetime | None

@dataclass(frozen=True)
class ResolvedSourceSnapshot:
    dataset: str
    source_generation: str | None
    objects: tuple[ResolvedObject, ...]
    content_digest: str
```

### 两条合法来源

A. COS LIST/HEAD exact objects  
B. 上游权威 source manifest

Production 不允许：

```text
wildcard URI 直接进入 executor
```

除非 wildcard 指向一个 immutable generation 并且有 manifest 证明 exact object set。

## 预算

查询入场前：

```text
actual_object_count
sum(content_length)
estimated rows
```

都进入 budget。

增加：

```text
max_scan_objects
max_scan_bytes
max_remote_list_objects
```

---

# 12. P0-011 — Snapshot verification 必须全 backend 统一

当前 Polars `ScanHandle.collect()` 已经有不错的：

```text
collect 前 remote HEAD
etag/size/version check
```

但 direct DuckDB/Arrow 并没有完全统一使用相同 verifier。

## 改法

新建：

```python
class SnapshotVerifier:
    def verify_before_execute(snapshot): ...
    def verify_after_execute(snapshot): ...
```

至少：

### Local

```text
path
size
mtime
optional checksum/generation
```

### Remote

```text
exact object
etag/versionId/content_length
```

执行：

```text
resolve snapshot
→ verify
→ execute exact objects
→ optional final verify for long read
```

所有 backend：

```text
DuckDB
Polars
PyArrow
stream
read_joined
factor read
```

共用。

---

# 13. P0-012 — Local mirror freshness 必须比较远端 source identity

当前：

```text
local file vs local manifest
```

一致就视为 fresh。

但：

```text
COS same key 被 upstream overwrite
```

不会被发现。

## 最佳方案：上游发布 source manifest

例如：

```json
{
  "source_generation": "20260810T153000Z-abc",
  "objects": [
    {
      "key": "...",
      "etag": "...",
      "size": 123
    }
  ]
}
```

DataAccess mirror：

```text
下载 generation G
本地 manifest 绑定 G
```

查询指定：

```text
latest resolved G
或 pinned G
```

## 如果暂时没有 source manifest

短期至少：

```text
remote HEAD ETag
```

在：

```text
sync freshness probe
```

比较。

但不要对 TB 级每个 query 全 HEAD。

需要：

```text
TTL + source generation/index
```

---

# 14. P0-013 — Auto hybrid 禁止混 source epoch

当前：

```text
local complete partition → local
missing partition → remote
```

缺少：

```text
source_generation equality
```

## 改法

先：

```python
target = SourceSnapshotResolver.resolve(dataset, snapshot_policy)
```

再选 location：

```text
local object matches target object identity
→ local

local does not match
→ remote exact target object
或 refresh
```

最后：

```python
assert all(parts.source_generation == target.generation)
```

无法证明：

```text
production => 不允许 hybrid
```

fallback：

```text
全部 remote target generation
```

而不是混。

---

# 15. P0-014 — Full mirror 要做 generation publish

当前：

```text
cos sync → live directory
```

读者可能看到：

```text
old/new mixture
```

## 改成

```text
mirror/
  generation/
    A/
    B/
  manifest.json
```

流程：

```text
1. resolve upstream generation G
2. mkdir staging/G.tmp
3. sync all
4. verify object inventory/checksum/schema
5. fsync
6. rename G.tmp -> G
7. atomic manifest pointer -> G
8. readers only resolve pointer
9. old generation delayed GC
```

不要边 sync 边暴露给 reader。

---

# 16. P0-015 — Missing object 必须根据 dataset semantics 决定

当前 downloader：

```text
404 / NoSuchKey
→ debug
→ return
```

不能统一这样做。

## Contract

```python
class MissingPartitionSemantics(Enum):
    ERROR = "error"
    WARN = "warn"
    EMPTY_OK = "empty_ok"
```

### Dense market data

交易日：

```text
StockDailyBar missing
=> ERROR
```

### Event table

某天无 event：

```text
可能 EMPTY_OK
```

### Sparse provider

根据 contract。

Downloader 不自己决定。

Physical planner 返回：

```text
ExpectedObject(required=True/False)
```

404 时遵循。

---

# 17. P0-016 — Event/period 数据不能继续套 trade-day partition model

当前 COS contract 已有：

```text
calendar_domain=event_time
```

但是 mirror expected partition helper实质只区分：

```text
calendar_day
else trade_day
```

这是概念性错误。

## 改法

Physical completeness 与 logical calendar 分离。

### D1

```text
partition_domain=trade_day
```

### Calendar-state snapshot

```text
partition_domain=calendar_day
```

### Event

```text
partition_domain=event_sparse
```

### Finance period files

```text
partition_domain=period_end
```

### Static

```text
single object
```

不要让：

```text
event_time
```

退化成：

```text
trade day
```

---

# 18. P0-017 — US filing_date 必须声明 date_label

当前逻辑：

```text
naive datetime = UTC instant
```

对于：

```text
filing_date = 2024-05-10 00:00:00
```

会转纽约：

```text
2024-05-09 20:00
```

这不符合当前数据实际“只有 filing 日期”的语义。

## TemporalAxisSpec

```python
@dataclass(frozen=True)
class TemporalAxisSpec:
    column: str
    representation: Literal["date_label","instant"]
    precision: Literal["date","second","millisecond","nanosecond"]
    storage_timezone: str | None
    semantic_timezone: str | None
```

US finance：

```text
filing_date
representation=date_label
precision=date
semantic_timezone=America/New_York
availability=next_session_open
```

US news：

```text
published_utc
representation=instant
storage_timezone=UTC
semantic_timezone=America/New_York
```

---

# 19. P0-018 — A股 UpdateTime 与 revision vintage 分离

当前：

```text
revision_columns=(UpdateTime,)
```

不能解释成：

```text
market knew revision at UpdateTime
```

## Contract 拆分

```text
knowledge_time = PubDate
revision_availability_time = None
dedup_tiebreaker = UpdateTime
pit_fidelity = knowledge_date_pit
```

只有上游未来有：

```text
revision published/ingested vintage history
```

才能：

```text
pit_fidelity = vintage_pit
```

## 关键

如果同一：

```text
Symbol + PubDate + ReportPeriod
```

有冲突 revision：

没有真实 revision availability 时：

```text
historical replay 不能假装恢复当时版本
```

可以：

```text
pinned-current-snapshot research
```

但 lineage 必须明确。

---

# 20. P0-019 — Derived data 权限继承

Dataset：

```text
classification/access_tags
```

Factor：

```text
derived_access_tags = max/union(source access tags)
```

继续传播到：

```text
factor lake
factor matrix
features
model input
materialized reports
API
```

读取 factor 时：

```text
authorize factor classification
```

## 降密

只能显式：

```text
declassification_id
approved_by
reason
policy_version
```

---

# 21. P0-020 — Automated Research 模式

新增 runtime mode：

```text
interactive_research
automated_research
production
```

## interactive_research

允许：

```text
某些 warning/degraded
```

## automated_research

必须 strict：

```text
PIT
calendar
semantic ambiguity
required filters
units
source snapshot
authorization
unknown field
```

但：

```text
不允许 publish production
预算可比 production 宽
```

## production

strict + publish + security + SLO。

AlphaProbe 默认：

```text
automated_research
```

---

# 22. ContractIR v2：真正成为 Runtime Single Source of Truth

当前 ContractIR 方向正确，但 runtime 仍经常分别调用：

```text
registry
get_cos_contract
semantic_catalog
mirror registry
```

所以才出现：

```text
required_event_filters 漏了
storage_layout 漂了
mirror layout 不一致
```

## 新结构

```python
@dataclass(frozen=True)
class RuntimeDatasetContract:
    dataset: str
    market: str | None

    storage: StorageContract
    physical_partition: PhysicalPartitionSpec
    temporal_axes: Mapping[str, TemporalAxisSpec]
    pit: PITContract
    filters: tuple[FilterRequirement, ...]
    units: Mapping[str, UnitContract]
    cardinality: CardinalityContract
    coverage: CoverageContract
    security: DatasetSecurityContract
    schema: SemanticSchemaContract

    fingerprint: str
```

启动时：

```text
datasets.yaml
+
COSDatasetContract
+
SemanticFieldCatalog
+
Mirror/source storage declaration
+
security deployment policy
→ ContractCompiler
→ RuntimeDatasetContract
```

然后 runtime 只消费它。

---

# 23. ContractIR v2 必须审计的新冲突

CI 要抓：

```text
Mirror layout != COS contract layout
Registry glob != physical filename pattern
partition_clock != declared physical layout
required_event_filters not enforced
filter cardinality mismatch
field PIT != table PIT
time representation mismatch
unit/currency mismatch
flow semantics mismatch
access classification missing
duplicate policy mismatch
coverage/panel policy mismatch
instrument key mismatch
market mismatch
```

US finance period_files / StockCapital prefix 这种必须自动变成 CI failure。

---

# 24. PreparedRead：把多时间轴真正带进 planner

建议核心对象：

```python
@dataclass(frozen=True)
class PreparedRead:
    dataset: str
    predicate_clock: str
    pruning_clock: str
    join_clock: str | None
    physical_partition_clock: str | None

    resolved_source_snapshot: ResolvedSourceSnapshot
    exact_objects: tuple[ResolvedObject, ...]
    filter_contract: CompiledFilterContract
    temporal_contract: CompiledTemporalContract
    unit_contract: CompiledUnitContract
    security_context_digest: str
```

规则：

```text
pruning_clock != physical_partition_clock
```

时：

必须有：

```text
显式 mapping/index
```

否则：

```text
不裁剪
```

不能瞎映射。

---

# 25. MirrorSpec 不再是第二套业务契约

当前 `DATASET_MIRROR_REGISTRY` 手工重复：

```text
dataset
cos prefix
table
layout
```

与 registry/COS contract 漂移。

## 重构目标

MirrorSpec 只保留 deployment/location：

```text
local_root
CLI profile/tool
cache root
```

而：

```text
layout
filename template
partition clock
```

来自：

```text
RuntimeDatasetContract.physical_partition
```

最好 mirror registry 最终只是：

```text
dataset -> location override
```

不再定义业务语义。

---

# 26. QueryBudget v2：远程对象和扫描字节预算

增加：

```python
max_scan_objects
max_scan_bytes
max_remote_list_objects
max_remote_requests
max_estimated_memory
```

入场顺序：

```text
resolve exact objects
→ calculate object count / bytes
→ admission
→ execute
```

不要：

```text
执行完才知道扫了多少
```

---

# 27. GlobalResourceGovernor

QueryBudget 只限制单次。

实现最小全局 governor：

```python
@dataclass
class ResourceReservation:
    query_id: str
    principal_id: str
    estimated_scan_bytes: int
    estimated_memory: int
    remote_requests: int
```

全局限制：

```text
max_active_queries
max_total_reserved_memory
max_total_scan_bytes_inflight
max_remote_concurrency
max_duckdb_concurrency
per_principal_active
per_principal_hourly_cost
```

如果暂时单进程：

```text
进程级 governor
```

如果多 worker：

```text
明确 single-worker contract
或共享 Redis/Postgres limiter
```

不要假装一个 process semaphore 是全服务器限制。

---

# 28. CacheManager

新增统一 cache lifecycle：

```python
class CacheManager:
    max_bytes
    high_watermark
    low_watermark
    ttl
    per_principal_quota
```

对象状态：

```text
unpinned
pinned_by_queries
evictable
stale
quarantined
```

GC：

```text
never delete pinned generation
```

磁盘达到 high watermark：

```text
evict LRU 到 low watermark
```

无法释放：

```text
new admission fail
```

不要等磁盘 100%。

---

# 29. Full mirror 与 CLI remote cache 分开

需要明确：

## Mirror

长期、可复现：

```text
generation-based
source snapshot aware
```

## Remote CLI cache

短期：

```text
cache only
evictable
principal-scoped
```

不要把二者混成：

```text
“本地有个 parquet 就能读”
```

---

# 30. SourceSnapshotResolver

建议新组件：

```python
class SourceSnapshotResolver:
    def resolve(dataset, policy) -> ResolvedSourceSnapshot
```

policy：

```text
latest
pin(snapshot_id)
fail_if_changed
```

对 COS：

优先：

```text
publisher generation manifest
```

次选：

```text
exact object list + etag
```

禁止：

```text
wildcard only
```

---

# 31. DatasetManifest 的 content identity

当前：

```text
dataset_version
partition_version
```

可以继续保留作快速 operational token。

新增：

```text
content_set_digest
```

例如：

```text
hash(sorted(
    object_key,
    etag/versionId,
    content_length
))
```

本地 immutable generation：

```text
hash(file path + checksum)
```

这个才是 reproducibility identity。

---

# 32. External upstream 的 manifest ownership

DataAccess 不能认为：

```text
自己 sidecar source_epoch
```

天然代表上游 COS 当前内容。

如果上游是独立清洗 pipeline：

应由 publisher 写：

```text
source_generation
source_manifest
complete marker
```

DataAccess：

```text
只消费
```

如果暂时做不到：

DataAccess 必须明确：

```text
external_source_identity_unverified
```

Production 不允许把它标为 fully pinned。

---

# 33. Schema evolution

`union_by_name=True` 不能成为“什么 schema 都能拼”的万能开关。

新增：

```python
@dataclass(frozen=True)
class SchemaEpoch:
    epoch_id: str
    start: date | None
    end: date | None
    schema_hash: str
    semantic_version: str
```

读取跨 epoch：

```text
compatible declared migration
→ allow

required field missing
→ reject

dtype widening explicitly approved
→ allow

unit/definition changed
→ semantic migration required
```

---

# 34. 每个 partition 的 required field coverage

当用户请求：

```text
close
roe
```

不能只验证：

```text
union schema 里有 roe
```

要确认 selected partition：

```text
每一个应有 roe 的 schema epoch
```

都满足。

否则：

```text
old partitions 全变 null
```

可能静默进入 FactorEngine。

---

# 35. Unit/Definition Version

增加 field source contract：

```text
source_unit
canonical_unit
currency
definition_id
definition_version
semantic_valid_from
semantic_valid_to
```

如果供应商：

```text
percent → decimal
```

dtype 没变，但：

```text
definition_version
```

必须变。

跨 version request：

```text
显式 normalize migration
```

否则 fail。

---

# 36. A/US typed UnitSpec

统一底层单位对象：

```python
@dataclass(frozen=True)
class UnitSpec:
    dimension: str
    scale: float
    currency: str | None
    currency_column: str | None
    cross_market_comparable: bool
    requires_fx: bool
```

禁止 DataAccess 与 FactorEngine 各写一套。

---

# 37. Money 与 FX

以下不能直接 global rank：

```text
A market_cap CNY
US market_cap USD
A revenue CNY
US revenue USD
```

如果无 FX：

```text
cross_market_comparable = false
```

如果未来加 FX：

FX 本身必须 PIT：

```text
fx_rate knowledge/effective time
currency pair
decision time
```

不能用今天 FX 把过去金额重算。

---

# 38. US dividend

DataAccess logical field 必须：

```text
knowledge = declaration_date
effective = ex_dividend_date
```

不是：

```text
ex_dividend exact
```

并拆：

```text
cash_dividend_local
cash_dividend_usd
```

没有 FX 时：

```text
currency != USD
```

不能当 USD。

---

# 39. Financial attribution

至少分：

```text
net_income_consolidated
net_income_attributable
equity_consolidated
equity_attributable
```

ROE 等：

```text
numerator/denominator attribution match
```

不能靠 alias 自动合并。

---

# 40. Financial flow semantics

Contract：

```text
cumulative_ytd_flow
single_period_flow
point_in_time_stock
```

A股：

```text
Income/CashFlow often cumulative YTD
```

US quarterly：

```text
single period
```

FactorEngine operator 必须消费这个 contract。

---

# 41. Fiscal calendar

US issuer：

```text
fiscal Q1
```

不一定自然年 Q1。

YoY/QoQ：

必须按：

```text
issuer fiscal period sequence
```

不是简单：

```text
date - 3 months
```

---

# 42. Missing prior period

A股 quarterization：

```text
Q2 YTD - Q1 YTD
```

如果 Q1 缺：

```text
不能把 Q2 YTD 当单季
```

默认：

```text
NaN / unavailable
```

除非有明确 accounting reconstruction policy。

---

# 43. Null vs Zero

禁止 generic：

```text
fill_null(0)
```

财务字段增加：

```text
missingness_semantics:
  unavailable
  not_reported
  not_applicable
  true_zero_allowed
```

Factor fill policy 进入 factor identity。

---

# 44. Adjustment factor semantics

DataAccess 标：

```text
adjustment_convention
source
retrospective=true/false
level_safe=true/false
```

FactorEngine 已经对部分 US clamped/level_sensitive 做过限制，应继续消费 DataAccess provenance。

不要说：

```text
后复权因子 = historical knowledge PIT
```

这是两个概念。

---

# 45. Instrument Identity

长期主键建议：

```text
market
stable_security_id
date
```

ticker/symbol：

```text
作为 time-varying attribute
```

短期至少：

```text
canonical instrument = ashare:000001.SZ
canonical instrument = us:AAPL
```

未来接：

```text
SecurityMaster / TickerMap
```

做稳定 ID。

---

# 46. Universe PIT

Universe 必须包含：

```text
membership validity
listing/delisting
tradability
security type
index membership
ST/suspension
```

禁止：

```text
today's surviving list backfill to history
```

Factor/experiment identity：

```text
universe_id
universe contract hash
membership snapshot/policy
```

---

# 47. `frequency` 从 metadata 升级成执行契约

当前 DataRequest：

```text
frequency
```

只是 metadata。

改：

```text
source_frequency
requested_frequency
```

若不同：

```text
AggregationSpec mandatory
```

例如：

```text
minute → daily
```

必须写：

```text
OHLC aggregation
VWAP policy
timezone/session
missing bar policy
early close
```

Plan.explain 必须显示。

---

# 48. Raw read 与 Semantic read 分开

建议：

```text
read_raw_physical()
read_semantic()
```

## read_raw_physical

专家/debug：

```text
不自动 canonical unit
明确 physical column
```

## read_semantic

FactorEngine/AlphaProbe：

```text
canonical unit
market resolved
PIT contract
required filters
semantic identity
```

FactorEngine production/automated research：

```text
禁止 raw physical source
```

---

# 49. FactorEngine 旁路治理

CI 静态扫描：

在 production source path 中禁止：

```text
pd.read_parquet
pl.scan_parquet
duckdb read_parquet
raw COS URI
raw absolute root
```

除非：

```text
DataAccess approved adapter module
```

FactorEngine SourceRef：

```text
只接受 registered dataset/provider profile
```

---

# 50. GovernedFrame / Provenance Envelope

裸 pandas DataFrame 不携带可信 provenance。

推荐：

```python
@dataclass(frozen=True)
class GovernedFrame:
    table_or_frame: Any
    source_snapshot: DataSnapshot
    lineage: ReadLineage
    execution_environment: ExecutionEnvironmentIdentity
    security_digest: str
```

如果 FactorEngine 接裸 dataframe：

production：

```text
UnknownProvenanceError
```

Research：

显式：

```text
unsafe_external_frame=True
```

且不能 publish。

---

# 51. ExecutionEnvironmentIdentity

统一记录：

```text
dataaccess_version
factorengine_version
git_commit
registry_fingerprint
contract_ir_fingerprint
semantic_catalog_fingerprint
calendar_snapshot_id
source_snapshot_id
access_policy_digest
principal_scope_id
duckdb_version
polars_version
pyarrow_version
backend
run_mode
```

Factor identity / experiment manifest 消费它。

---

# 52. 多服务器配置漂移

Production startup 输出非 secret：

```text
server_id
registry fp
contract fp
semantic fp
package versions
calendar identity
policy digest
```

运维可以比较：

```text
Server A vs B
```

不要让：

```text
DATA_ACCESS_SEMANTIC_FIELDS
root env
DuckDB version
```

悄悄不一致。

---

# 53. Production 禁止 legacy `/home/shw/...` fallback

开发机可以。

Production：

如果 env 没显式配置：

```text
ASHARE_PARQUET_ROOT
US_MASSIVE_ROOT
US_CLEAN_ROOT
```

却会落到：

```text
/home/shw/...
```

直接启动失败。

不要在另一台 server 默默读错误目录。

---

# 54. Local path strict mode 信息泄露

`PathAuthorizer` 的错误脱敏条件应使用：

```text
is_strict_semantics()
```

而不是只看：

```text
QUANT_PRODUCTION_MODE
```

`DATA_ACCESS_STRICT_READ=1` 同样不应打印全部 roots。

---

# 55. HTTP Service：Principal + Scope

一个 API key：

```text
不能代表整台 server 全部权限
```

映射：

```text
API key hash → principal → scopes
```

Endpoint scope：

```text
dataset:list
dataset:read
factor:list
factor:read
factor:metadata_sensitive
uri:read:admin
metrics:read
```

---

# 56. `/v1/datasets`

只返回：

```text
principal authorized datasets
```

不能把 premium dataset 名全部暴露给低权限 caller。

---

# 57. `/v1/read_uri`

Production：

```text
默认关闭
```

或：

```text
admin-only
```

即使 URI 在 registered prefix：

还必须：

```text
resolve to dataset
authorize principal
```

---

# 58. `/v1/factors`

不要默认返回：

```text
catalog.root
完整 expression
敏感 source lineage
```

普通：

```text
summary
```

敏感 metadata：

```text
单独 scope
```

---

# 59. HTTP Model strictness

Pydantic：

```python
model_config = ConfigDict(extra="forbid")
```

所有 request。

增加：

```text
dataset name length
column count
column length
instrument_filter max cardinality
factor_ids max cardinality
params size/depth
filter AST complexity
time_range length
limit upper bound
URI length
```

---

# 60. HTTP JSON 内存放大

JSON 单独 budget：

```text
max_json_rows
max_json_arrow_bytes
estimated_json_bytes
```

例如远低于 Arrow stream。

Bulk：

```text
Arrow IPC stream
Parquet
```

不要 DataFrame→JSON object 大规模复制。

---

# 61. HTTP stream 生命周期

增加：

```text
max_stream_lifetime
idle_timeout
client_disconnect cancellation
max_stream_bytes
```

客户端断开：

```text
reader.close
DuckDB interrupt
release reservation
```

---

# 62. `/ready` 真正 readiness

轻量检查：

```text
ContractIR compile/audit
registry
engine SELECT 1
CredentialProvider resolution
critical calendar proof
critical dataset metadata/HEAD probe
cache/temp free space
source manifest reachability
```

不要读大表。

输出不能泄 secret。

---

# 63. Process-local semaphore 问题

如果多 worker：

```text
N workers × max_concurrency
```

真实并发被放大。

短期：

```text
强制 production single worker
```

或：

```text
shared ResourceGovernor
```

文档/部署检查必须明确。

---

# 64. Data Quality（DQ）成为一等公民

Schema check 之外增加：

```text
freshness
coverage
duplicate keys
critical null rate
row count drift
date monotonicity
OHLC consistency
negative volume
return consistency
finance period <= knowledge
currency enum
unit range
index weight sum
universe coverage
```

DQ：

```text
不负责自动修数据
```

只：

```text
PASS
WARN
BLOCK
```

---

# 65. 单位漂移 DQ

黄金 sentinel：

### A Return

```text
典型绝对值范围
bp semantics
```

### US Ret

```text
decimal semantics
```

监控分布突然缩放：

```text
100x / 10000x
```

能自动报警。

---

# 66. Coverage DQ

按：

```text
date
year
instrument
market
provider
```

统计。

特别是：

```text
X0
partial history
new source
```

不能用 overall coverage 掩盖某一段历史空洞。

---

# 67. Mirror/remote concurrency

 `_sync_cos_file`：

```text
固定 .tmp
```

改：

```text
per-object file lock
unique temp
fsync
atomic replace
manifest same lock
```

同一 object 多 worker：

```text
single flight
```

其他等待。

---

# 68. Cache eviction 与 in-flight query

Query resolve source snapshot 时：

```text
pin objects/generation
```

结束：

```text
unpin
```

GC：

```text
只删 refcount=0
```

---

# 69. Source Generation GC

不要：

```text
只保留 current + previous 后立即删
```

长期 read 可能仍 pin older generation。

策略：

```text
TTL
minimum retained generations
reader lease/refcount
```

---

# 70. Manifest freshness 与 external source

本地 DataAccess mutation 的：

```text
source_epoch
```

可以继续用。

但外部 COS：

必须区分：

```text
local_manifest_epoch
upstream_source_generation
```

不要混成一个概念。

---

# 71. Read `time_range=None`

Production 大表：

推荐：

```text
require_time_range=true
```

如果业务必须无界：

必须：

```text
complete source generation
exact manifest
scan budget
```

“目录非空”永远不能证明完整。

---

# 72. Remote LIST 成本

如果 source manifest 不存在，COS LIST 本身可能昂贵。

设置：

```text
max_list_pages
max_list_objects
list deadline
```

超限：

```text
要求更窄 time_range
或 source manifest
```

---

# 73. Query estimation

在执行前做：

```text
estimated_objects
estimated_bytes
estimated_rows
```

不用追求精确。

只要能阻止：

```text
明显 TB 级误扫
```

---

# 74. DuckDB object cache 与 source generation

当 COS same key 被覆盖：

DuckDB object cache 可能保留旧 footer/metadata。

Source snapshot 切 generation 时：

应：

```text
使用 immutable object key/generation
```

优于同 key overwrite。

如果必须 same key：

重新配置/invalid cache 或独立 generation path。

---

# 75. COS 最好启用 immutable generation/object version

上游建议：

```text
generation/<gid>/...
current manifest pointer
```

不要覆盖：

```text
2024-01-01.parquet
```

如果腾讯 COS bucket 开版本控制：

DataSnapshot 优先记录：

```text
VersionId
```

---

# 76. Retry semantics

IO retry 只适合：

```text
transient network
timeout
5xx
```

不能重试：

```text
AccessDenied
invalid filter
PIT reject
schema mismatch
snapshot changed
corruption
```

建立 typed retry classifier。

---

# 77. STS credential refresh

长任务：

```text
token expires
```

需要 CredentialProvider：

```text
refresh before expiry
```

但必须保证：

```text
refresh 后 principal/scope 不变或更窄
```

不能在 task 中途提升权限。

如果 refresh scope 改变：

```text
abort
```

---

# 78. Error taxonomy

建立 typed exceptions：

```text
AuthenticationError
AuthorizationError
CredentialExpiredError
SourceSnapshotUnavailable
SourceSnapshotChanged
PhysicalPartitionResolutionError
MissingRequiredPartition
CalendarUnavailable
PITContractError
FilterContractError
SchemaContractError
SemanticUnitError
ResourceAdmissionError
CacheSecurityError
```

HTTP：

```text
401 authentication
403 authorization
409 snapshot conflict
422 semantic contract
429 resource quota
503 dependency unavailable
```

不要都压成 400/422 + 内部字符串。

---

# 79. Audit Log 安全

Audit 自己也是敏感数据。

不要记录：

```text
Secret
signed URL
完整 credential
```

可能敏感的：

```text
premium dataset
factor expression
principal activity
```

审计存储也需要权限。

---

# 80. Audit completeness

每次 read：

```text
request_id
principal_id
server_id
dataset
action
contract fingerprint
source snapshot
object count
estimated/actual bytes
PIT mode
calendar snapshot
degradation flag
result rows
latency
```

不要只记前 5 个 path 当唯一 provenance。

---

# 81. Observability

至少 metric：

```text
dataaccess_queries_total
authorization_denied_total
pit_rejected_total
snapshot_changed_total
source_generation_mismatch_total
hybrid_fallback_total
mirror_refresh_total
mirror_stale_total
cache_bytes
cache_evictions
cache_pinned_bytes
remote_list_objects
remote_head_requests
scan_bytes_estimated
scan_bytes_actual
resource_admission_rejected
calendar_degraded_total
schema_drift_total
dq_block_total
```

---

# 82. SLO

可以先定义内部目标：

```text
critical read success rate
p50/p95 latency
PIT contract error rate
stale mirror rate
source snapshot unverifiable rate
```

不需要复杂 SLA 系统，但 regression 要有基准。

---

# 83. FactorEngine Contract Consumption

FactorEngine DataAccessSource：

只消费：

```text
RuntimeDatasetContract
Semantic Provider Binding
Governed Read Handle
```

不重新定义：

```text
filing/PubDate
unit scale
currency
required timeframe
source classification
```

---

# 84. Factor definition 需要加入 source contract digest

完整 factor identity：

```text
canonical expression
operators/version
market
frequency
universe
decision policy
source dataset/provider
source contract digest
unit/definition digest
PIT contract digest
source snapshot policy
```

---

# 85. Factor Lake / Matrix 访问权限

`read_factors()`：

必须：

```text
factor meta → derived classification → authorize principal
```

Matrix：

如果不同列有不同 classification：

### 推荐

同一 matrix generation 按 security domain 分开：

```text
matrix/basic
matrix/premium
```

不要让一个宽表混多个权限，之后难做列级权限。

---

# 86. Factor metadata list 的权限

低权限 caller：

不能看到：

```text
premium factor name
source fields
expression
```

除非 policy 允许 metadata discovery。

---

# 87. Automated Mining Grammar

只暴露：

```text
production/automated_research certified canonical fields
```

排除：

```text
raw physical
PIT unsupported
unit unknown
provider coverage insufficient
security inaccessible
derived compiler unavailable
```

---

# 88. Data leakage through model outputs

权限传播不能停在 factor。

如果：

```text
premium factor → model score
```

模型 score 也可能泄露信息。

至少 lineage 分类传播到：

```text
feature set
model artifact
prediction output
```

后续是否允许降密，需要独立政策。

---

# 89. Regression/Destructive Test 总原则

本轮不要只写：

```text
helper unit tests
```

要从真正公共入口测试。

至少：

```text
DataAccessStore
HTTP
FactorEngine DataAccessSource
mirror/remote
```

---

# 90. 必须新增：Physical Layout Tests

## T-PHY-001 US finance period file

物理：

```text
StockIncome/2024-03-31.parquet
filing_date=2024-05-10
```

request：

```text
knowledge window=2024-05-10
```

四模式都读到：

```text
local
mirror
remote
auto
```

## T-PHY-002 StockCapital shares

必须访问：

```text
shares_2024-01-01.parquet
```

不访问：

```text
2024-01-01.parquet
```

## T-PHY-003 split

反过来。

## T-PHY-004 event sparse

某天无 event object：

```text
EMPTY_OK
```

## T-PHY-005 dense D1 missing

交易日行情 object missing：

```text
hard error
```

---

# 91. Required Filter Tests

## T-FLT-001

US finance no timeframe：

```text
reject
```

所有：

```text
read_arrow
stream
Polars
PyArrow
read_joined
FactorEngine
```

## T-FLT-002

```text
timeframe=["quarterly","annual"]
```

reject。

## T-FLT-003

single quarterly：

success。

## T-FLT-004

physical columns / columns=None：

仍不能绕过 dataset-level requirement。

---

# 92. Availability Tests

## T-TIME-001

```text
next_trading_day + calendar=None + strict
```

must raise。

## T-TIME-002

US `filing_date=2024-05-10 00:00`

声明 date_label：

```text
semantic date remains 2024-05-10
```

## T-TIME-003

true UTC news instant：

正确 timezone conversion。

## T-TIME-004

Friday filing date-only：

next real session。

## T-TIME-005

calendar right boundary：

hard fail。

---

# 93. Source Snapshot Tests

## T-SNAP-001 remote wildcard

Production：

```text
wildcard unresolved
→ reject
```

## T-SNAP-002 exact object set

```text
objects + ETags
→ stable snapshot
```

## T-SNAP-003 same key overwrite

ETag changes：

```text
snapshot changes
```

## T-SNAP-004 local stale remote new

auto：

```text
cannot mix
```

## T-SNAP-005 exact same generation local+remote

hybrid allowed。

---

# 94. Mirror Atomicity Tests

## T-MIR-001

full sync halfway crash：

reader continues old generation。

## T-MIR-002

new generation verified：

atomic pointer switch。

## T-MIR-003

two workers same object：

only one physical download wins。

## T-MIR-004

cache principal mismatch：

reject.

## T-MIR-005

cache disk pressure：

pinned object never deleted。

---

# 95. Security Tests

继续保留 R24：

```text
server A basic
server B premium
```

同代码不同权限。

403：

```text
不得 fallback higher credential
```

Production：

```text
不 parse ~/.cos.yaml
不 global inject secret
```

STS token：

有效。

Derived premium factor：

basic principal reject。

---

# 96. Schema Evolution Tests

## T-SCH-001

2024 has field, 2025 missing：

request crossing both：

```text
fail unless explicit migration
```

## T-SCH-002

double → string：

reject。

## T-SCH-003

percent → decimal semantic version：

必须 normalize migration，不能只看 dtype。

---

# 97. Resource Governance Tests

## T-RES-001

20 legal queries：

global memory reservation exceed：

后面的 admission reject。

## T-RES-002

remote wildcard huge object list：

在 execution 前 reject。

## T-RES-003

cache disk near full：

new remote cache request reject/evict safely。

---

# 98. HTTP Tests

```text
unknown fields -> 422
basic principal cannot list premium
basic cannot read premium
read_uri admin only
JSON too large -> 413
slow stream timeout/cancel
client disconnect releases slot
multi-worker deployment guard
```

---

# 99. FactorEngine Integration Tests

## T-FE-001

FactorEngine production raw parquet path：

reject。

## T-FE-002

Automated research ambiguous semantic field：

reject，不 warning continue。

## T-FE-003

Factor from premium dataset：

materialized classification premium。

## T-FE-004

naked DataFrame unknown provenance：

production reject。

---

# 100. Data Quality Tests

至少 golden：

```text
A Return bp -> decimal
US Ret decimal unchanged
A ROE % -> decimal
US ROE unchanged
A CNY money != US USD money
US dividend non-USD blocked
A financial PubDate <= availability
US filing date-only next session
OHLC valid
negative volume blocked/flagged
duplicate semantic keys detected
```

---

# 101. Migration Phase 1 — 先修当前真实 P0

只改：

```text
US finance physical layout
StockCapital filename patterns
required_event_filters
timeframe exactly-one
calendar=None strict fail-open
```

要求现有普通 DataAccess regression 全过。

---

# 102. Phase 2 — Runtime ContractIR v2

实现：

```text
RuntimeDatasetContract
PhysicalPartitionSpec
FilterRequirement
TemporalAxisSpec
```

然后逐步把：

```text
store
mirror
remote
partition planner
```

切到这个 IR。

---

# 103. Phase 3 — Source Snapshot / Mirror Generation

实现：

```text
ResolvedSourceSnapshot
exact objects
source content digest
mirror generation
hybrid same-generation gate
```

---

# 104. Phase 4 — Security

实现：

```text
Principal
AccessPolicy
CredentialProvider
local cache security
HTTP scopes
derived classification
```

---

# 105. Phase 5 — Automated Research / FE Boundary

实现：

```text
run_mode=automated_research
semantic-only source
GovernedFrame
raw source blocking
factor classification
```

---

# 106. Phase 6 — Resource / Cache / Service

实现：

```text
ResourceGovernor
CacheManager
HTTP strict model
readiness
stream lifetime
```

---

# 107. Phase 7 — DQ / CI / Operations

实现：

```text
ContractIR audit
secret scan
schema/semantic drift
source freshness
DQ
metrics
runbook
```

---

# 108. 每个 Phase 都必须保持 backward compatibility 的范围

Research notebook API 可以保留旧方法：

```text
read_arrow
read_frame
```

但：

- production semantic floor 不允许被旧参数降级；
- deprecated raw behavior 打 warning；
- FactorEngine 不使用 deprecated surface。

---

# 109. CI 必须新增的静态审计

## Audit 1

`MirrorSpec/PhysicalSpec vs ContractIR`

## Audit 2

`required filters execution coverage`

扫描所有 public read surfaces。

## Audit 3

`no raw COS credential parse in production`

## Audit 4

`no FE production raw read bypass`

## Audit 5

`semantic fields vs FE providers`

## Audit 6

`cross-market unit/definition`

## Audit 7

`source snapshot identity completeness`

---

# 110. Secret Scan

CI：

```text
gitleaks/trufflehog
```

working tree + history。

发现历史 Secret：

```text
rotate
```

不是只删除。

---

# 111. Production Startup Gate

Production 启动前：

```text
registry load
ContractIR audit == 0 blocking problems
security context valid
credential provider valid
legacy home fallback unused
critical calendar authoritative
engine version supported
cache permissions safe
source snapshot provider reachable
```

任一 critical fail：

```text
startup fail
```

---

# 112. Break-glass

如果确实要临时降级：

```text
BREAK_GLASS_*
```

必须：

```text
explicit operator
reason
expiry
audit
```

不能永久 env 一开。

---

# 113. Disaster Recovery

## COS remote unavailable

如果 local pinned generation 完整：

```text
可以继续 pinned read
```

否则：

```text
fail
```

不能拿 partial cache。

## Mirror publish crash

旧 generation 继续服务。

## Source manifest corrupt

quarantine / reject。

## Calendar unavailable

PIT-required production reject。

---

# 114. Runbook：权限故障

403：

```text
1. 不换 credential
2. 记录 principal/dataset/action
3. 检查 AccessPolicy
4. 检查 CAM/IAM
5. 检查 token expiry
6. 不 fallback base ~/.cos.yaml
```

---

# 115. Runbook：结果不一致

两 server 因子不一致时比较：

```text
source snapshot
ContractIR fp
calendar snapshot
semantic fp
unit contract
backend versions
universe identity
factor identity
```

不要第一反应就是：

```text
浮点误差
```

---

# 116. Runbook：数据突然缺失

按：

```text
authorization
source generation
missing partition semantics
schema epoch
coverage DQ
calendar
```

排查。

---

# 117. Performance 原则

不要让治理把自动挖掘性能拖垮。

正确方式：

```text
ContractIR compile once
ResolvedSourceSnapshot cache
exact object manifest cache
schema epoch cache
unit compiled plan cache
```

不要每个 operator 重做。

---

# 118. Compile Once, Execute Many

因子批量挖掘：

相同：

```text
dataset
time range
universe
source snapshot
semantic contract
```

编译一次：

```text
PreparedRead
```

多 factor 共用。

---

# 119. Metadata Plane

建议提供内部诊断 API：

```text
describe_runtime_contract(dataset)
describe_source_snapshot(dataset)
explain_read(request)
explain_authorization(dataset, principal)
```

只给开发者/管理员。

方便 owner 不读代码理解系统。

---

# 120. `explain_read()` 应输出

例如：

```text
Dataset: us_stock_income
Market: US
Physical layout: PERIOD_END_FILE
Partition clock: period_end
Predicate clock: filing_date
Decision clock: anchor TradeDate
PIT: strict, date_label, next_session_open
Required filter: timeframe exactly-one
Source snapshot: G123
Objects: 8
Bytes: 120MB
Backend: remote httpfs
Principal: server-b / premium
Canonical unit: USD
```

这会大幅降低 DataAccess 的维护难度。

---

# 121. 最终代码模块建议

不要求文件名完全照抄，但职责最好收敛成：

```text
dataaccess/
  contract/
    runtime_contract.py
    physical_partition.py
    temporal_axis.py
    filters.py
    units.py

  security/
    principal.py
    policy.py
    credentials.py
    redaction.py

  snapshot/
    source_snapshot.py
    verifier.py

  cos/
    mirror_generation.py
    remote.py
    cache_manager.py

  runtime/
    prepared_read.py
    resource_governor.py
```

避免一个 `store.py` 继续无限变大。

---

# 122. 不要做的错误修法

### 错误 1

```text
US finance layout 改成 daily 就完了
```

不对，核心是 partition clock。

### 错误 2

```text
所有 remote read 先 LIST 全 bucket
```

会成本爆炸。

### 错误 3

```text
为了权限把 bucket 名藏起来
```

不是安全。

### 错误 4

```text
403 就 fallback clean-cos-ro/base credential
```

严重错误。

### 错误 5

```text
所有 event 缺文件都 hard fail
```

事件稀疏不合理。

### 错误 6

```text
所有 missing 都 empty_ok
```

行情数据泄漏。

### 错误 7

```text
所有 timestamp 都 UTC
```

filing date-label 会错。

### 错误 8

```text
所有 research 都 warning 放行
```

自动挖因子会静默污染搜索空间。

### 错误 9

```text
所有 factor 都取最高权限然后永远不降
```

安全但会把体系锁死；需要显式 declassification workflow。

### 错误 10

```text
为了可复现给每个 query 做几万个 HEAD
```

性能不可接受；优先 source manifest/generation。

---

# 123. 最终 Definition of Done — P0

以下全部满足之前，不允许声称 DataAccess 全面收官。

## Physical

- [ ] US finance period files 正确
- [ ] US StockCapital split/shares pattern 正确
- [ ] partition clock 与 predicate clock 分离
- [ ] event/period 不再被 trade-day enumerator 错误建模
- [ ] missing object 按 contract 处理

## Filters

- [ ] required_event_filters 全 surface 生效
- [ ] US timeframe exactly-one
- [ ] timeframe 进入 identity/cache/lineage

## PIT

- [ ] calendar missing strict fail
- [ ] US filing date-label
- [ ] US date-only filing next session
- [ ] A UpdateTime 不冒充 revision availability
- [ ] revision fidelity 明确

## Snapshot

- [ ] production remote 无 unresolved wildcard snapshot
- [ ] scan file/object budget 统计真实 object
- [ ] all backend shared SnapshotVerifier
- [ ] local mirror 能识别 upstream version
- [ ] hybrid same source generation
- [ ] full mirror atomic generation

## Security

- [ ] production 不 parse ~/.cos.yaml raw key
- [ ] no global secret env injection
- [ ] STS token
- [ ] principal/access policy
- [ ] server-specific permissions preserved
- [ ] cache principal isolated
- [ ] derived factor classification

## FE

- [ ] automated_research strict semantics
- [ ] production raw read bypass blocked
- [ ] governed provenance retained

---

# 124. Definition of Done — P1/P2 Production Readiness

- [ ] ContractIR v2 runtime single source
- [ ] schema epoch/migration
- [ ] unit/definition version
- [ ] global ResourceGovernor
- [ ] CacheManager quota/GC/pin
- [ ] HTTP principal/scopes
- [ ] strict Pydantic extra forbid
- [ ] stream timeout/cancel
- [ ] readiness probes
- [ ] DQ
- [ ] metrics
- [ ] secret scan
- [ ] runbooks
- [ ] environment identity
- [ ] multi-server config drift detection

---

# 125. AI 修改完成后必须提交的最终报告

不要只写：

```text
done
tests passed
```

必须严格输出。

## A. Baseline / Final HEAD

```text
starting HEAD
final HEAD
changed files
```

## B. Current Defects Fixed

按 R25 ID：

```text
R25-P0-001
...
```

逐条：

```text
root cause
implementation
test
```

## C. Runtime Architecture

画：

```text
Request
→ Principal/Auth
→ RuntimeDatasetContract
→ PreparedRead
→ SourceSnapshotResolver
→ ResourceGovernor
→ Backend
→ SnapshotVerifier
→ Result/Lineage
```

## D. Physical Layout Matrix

至少：

| Dataset | Logical clock | Physical partition clock | Layout | Filename template | Missing semantics |
|---|---|---|---|---|---|

包含：

```text
A daily
A finance
A dividend
US daily
US finance
US dividend
US capital split
US capital shares
US news
```

## E. PIT Matrix

| Market | Dataset | Knowledge | Representation | Availability | Period | Revision fidelity |
|---|---|---|---|---|---|---|

## F. Security Matrix

说明：

```text
Server A
Server B
Principal
AccessPolicy
CAM/IAM
clean-cos-ro
httpfs
local cache
factor classification
```

## G. Source Snapshot

说明：

```text
如何解决 remote wildcard
如何识别 upstream overwrite
如何保证 hybrid same epoch
如何做 mirror generation
```

## H. Resource Governance

列：

```text
per-query
global
cache quota
remote limits
HTTP limits
```

## I. A/US Unit/Definition Matrix

至少：

```text
return
ROE
market cap
revenue
net income
dividend
shares
adjusted price
```

## J. Tests

真实：

```text
file
test count
pass/fail
duration
```

分类：

```text
physical layout
PIT
security
snapshot
mirror
cross-market
FactorEngine integration
HTTP
resource
existing regressions
```

## K. Known Limitations

必须明确：

```text
A股是否有 historical revision-vintage PIT？
上游 COS 是否有 immutable source generation？
COS bucket 是否支持 VersionId？
哪些只能 knowledge-date PIT？
```

不能用模糊语言。

## L. Freeze Verdict

只能二选一：

```text
DATAACCESS FULL PLATFORM FREEZE = YES
```

或：

```text
NO
```

如果 NO：

只列剩余真实 blocker，不要再发散 P2。

---

# 126. 最终 Freeze 条件

只有：

```text
当前 P0 全部关闭
destructive tests 真通过
ContractIR runtime 不再漂
security server privilege 保留
PIT 无 fail-open
source snapshot 可证明
```

之后才允许：

```text
DATAACCESS CORE FREEZE
+
DATAACCESS COS/SECURITY FREEZE
+
DATAACCESS PIT/SEMANTIC FREEZE
+
DATAACCESS ↔ FACTORENGINE INTEGRATION FREEZE
```

之后除非：

```text
真实 regression
新数据源
新市场
新生产要求
```

不要继续为了“感觉可能还能优化”无止境重构 Core。

---

# 127. 本轮最重要的工程判断

当前 DataAccess 最大风险已经不是：

```text
不会读 parquet
```

而是：

```text
每个模块单独正确
+
模块之间对同一事实理解不同
=
组合结果静默错误
```

目前最典型的三个实例就是：

```text
COS contract: US finance = period_files
MirrorSpec: daily_parquet
```

```text
COS contract: required_event_filters=timeframe
Store dataset gate: 不包含 required_event_filters
```

```text
ContractIR: 已经有 partition/knowledge/period 多时钟
Physical planner: 仍可能直接用 request time_range 展开 filename
```

因此本轮真正的目标不是增加功能数量，而是建立：

> **一个 Runtime Contract + 一个 Source Snapshot + 一个 Security Context + 一个 PreparedRead。**

当这四个对象成为运行时唯一事实源后，DataAccess 才真正从“功能很多的数据读取库”变成“可以长期维护的量化数据平台”。

---

# 128. 最后一条执行指令

请不要分批询问我是否继续。

按以下顺序一次性完成：

```text
1. 修 verified P0
2. 重构 runtime contract
3. source snapshot / mirror generation
4. security
5. FE integration
6. resource/cache/service
7. DQ/CI
8. 全量测试
9. 生成最终 closure report
```

如果某个上游能力当前不存在，例如：

```text
COS source publisher 没有 immutable generation manifest
A股没有 revision vintage
COS 没有 VersionId
```

不要伪造。

必须：

```text
明确 limitation
采用保守 fallback
production fail closed
```

**禁止为了让测试“绿”而把无法证明的东西假装成已证明。**
