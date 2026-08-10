# R24 — DataAccess 安全、PIT、跨市场语义与 FactorEngine 协作专项整改提示词

> **仓库**：`https://github.com/18047533889/quant_projects`  
> **审计基线**：`main@7b15a5e7a7a769734f7d1e003a6b0867c1ce88c9`  
> **任务定位**：本轮不要继续扩大 DataAccess 架构。集中修复 **权限边界 / 数据泄露 / COS 多服务器分级权限 / 财务 PIT / A股与美股单位与会计语义 / DataAccess ↔ FactorEngine 契约漂移**。  
> **最终目标**：任何生产读取或因子值都必须能同时证明：**调用者有权读取、数据在决策时已经可知、单位与会计定义没有被误解、所用数据版本与权限来源可追溯**。无法证明时 production 必须 fail closed。

---

## 0. 修改前必须阅读

### DataAccess

重点阅读：

- `dataaccess/cos/remote.py`
- `dataaccess/cos/mirror.py`
- `dataaccess/cos_contract.py`
- `dataaccess/cos_contract_ashare.py`
- `dataaccess/cos_contract_us.py`
- `dataaccess/config/datasets.yaml`
- `dataaccess/config/semantic_fields.yaml`
- `dataaccess/read/semantic_catalog.py`
- `dataaccess/read/contract_ir.py`
- `dataaccess/read/temporal_join.py`
- `dataaccess/read/session_calendar.py`
- `dataaccess/read/data_request.py`
- `dataaccess/store.py`
- `dataaccess/service/app.py`
- `dataaccess/service/config.py`
- `.gitignore`

### FactorEngine

重点阅读：

- `factor_engine/storage/sources/data_access_source.py`
- `factor_engine/fields/concepts.py`
- `factor_engine/fields/providers.py`
- `factor_engine/fields/`
- `factor_engine/runtime/factor_identity.py`
- `factor_engine/runtime/materialize_service.py`
- FactorCatalog / lineage / factor lake / factor matrix 相关实现

### 数据事实源

必须以以下两份数据字典为准，不得靠字段名或市场常识猜：

1. `COS A股数据字典（lqtp_data）`
2. `COS 美股数据字典（massive_data）`

尤其关注：

- A股 `Return` 是 bp；美股 `Ret` 已经是 decimal；
- A股 ROE/换手率等大量比率是 `%`；美股对应比率多已是 decimal；
- A股财务 PIT 是 `PubDate`；
- 美股财务 PIT 是 `filing_date`，文件名是 `period_end`；
- 美股财务必须先选一个 `timeframe`；
- A股 `UpdateTime` 是供应商/清洗 freshness，不是历史市场可知时间；
- 美股分红 `cash_amount` 币种并不恒为 USD；
- 美股 Valuation/Indicator 是 X0 稀疏数据，不得伪装成全历史 D1；
- A股和美股同名表、同名字段不代表同语义。

---

# 1. 本轮禁止重新发明的模块

下列能力前面已经大量修过；除非新测试证明回归，本轮不要大改：

- DataRequest / CompiledDataRequest / ReadPlan 不可变性
- Registry 基础结构
- QueryBudget
- Snapshot / manifest / generation 主体机制
- factor_matrix generation publish
- factor semantic identity 主体机制
- PyArrow / DuckDB / Polars predicate parity
- EMPTY / X0 / E1 / E2 基本 temporal contract
- FactorEngine PIT eligibility 主框架
- A股 `Return/10000`
- A股 `% -> decimal`
- 美股 `Ret` 不二次除
- 美股 X0 valuation/indicator 禁止当 full-history panel
- 美股 StockCapitalDaily 双 schema 拆分
- matrix current generation 读取
- ClickHouse tombstone / factor version 主链

本轮修的是**新的安全/PIT/跨市场问题**。

---

# 2. 先统一最终安全模型

DataAccess 不能只有“路径白名单”，生产权限至少有四层：

```text
请求者 / 当前服务器身份
        │
        ▼
① DataAccess 逻辑授权
   dataset / factor / action scope
        │
        ▼
② Registry 路径边界
   请求只能落到注册 dataset / prefix
        │
        ▼
③ COS CAM / IAM / STS
   云端最终拒绝或允许
        │
        ▼
④ 本地 mirror / cache 权限
   已下载文件仍不能越权共享
```

必须严格区分：

- **认证 Authentication**：当前是谁；
- **授权 Authorization**：这个身份能读什么；
- **路径授权 Path Boundary**：这个 URI 是否属于已注册的数据空间；
- **COS CAM/IAM**：云端最终能否 GET 该对象；
- **本地文件 ACL**：对象下载后谁还能直接读 parquet。

`authorize_s3_path()` 当前只是在做注册 prefix 边界，**绝不能继续把它当真正的数据权限系统**。

---

# 3. P0-S1 — 禁止 production 直接解析 `~/.cos.yaml` 获取 base SecretKey

## 当前风险

`dataaccess/cos/remote.py` 当前有：

```python
_read_cos_cli_credentials()
resolve_s3_credentials()
load_cos_cli_credentials_into_env()
```

会读取：

```text
~/.cos.yaml
cos.base.secretid
cos.base.secretkey
```

并存在：

```python
os.environ.setdefault("COS_SECRET_ID", sid)
os.environ.setdefault("COS_SECRET_KEY", skey)
```

这会削弱用户原本依赖 `clean-cos-ro` 做的服务器权限分级。

典型危险路径：

```text
服务器 A
clean-cos-ro 使用受限身份
    ↓
只能读 basic_data

但是 DataAccess httpfs
    ↓
自己解析 ~/.cos.yaml 的 cos.base
    ↓
如果 base credential 权限更大
    ↓
可以绕过服务器原本的受限 wrapper
```

即便现在各服务器 `.cos.yaml` 恰好权限一致，也不允许留下这种架构能力。

## 必须修改

### 3.1 production 禁止 raw parse coscli private config

Production 下：

```text
resolve_s3_credentials()
```

只能从明确 CredentialProvider 获取。

允许来源：

```text
A. CVM / 容器 / 主机绑定角色
B. STS 临时凭证
C. 部署系统显式注入的 server-scoped env credential
D. 显式配置的 CredentialProvider
```

不要自动读：

```text
~/.cos.yaml -> cos.base.secret*
```

### 3.2 `clean-cos-ro` 保持为 credential boundary

CLI 路径：

```text
DataAccess
   ↓ subprocess
clean-cos-ro cp / ls
   ↓
coscli 自己选它被部署好的 profile / credential
```

DataAccess 不应该：

- 解析它的 SecretKey；
- 猜 profile；
- 切换到更高权限 profile；
- 遇到 403 后换 base credential 再试。

### 3.3 不要把 secret 写入全局环境

废弃 production：

```python
load_cos_cli_credentials_into_env()
```

禁止：

```python
os.environ["COS_SECRET_KEY"] = ...
```

原因：

- 子进程默认继承；
- crash dump / debug 工具可能暴露；
- 长驻 worker 其他模块也能读取；
- 测试进程容易污染。

### 3.4 支持临时 token

扩展：

```python
@dataclass(frozen=True)
class S3Credentials:
    access_key_id: str
    secret_access_key: str
    session_token: str | None
    expires_at: datetime | None
    principal_id: str | None
    credential_scope_id: str | None
```

STS/CAM 临时身份没有 session token 支持是不完整的。

### 3.5 Secret 自动脱敏

至少：

```python
secret_access_key: str = field(repr=False)
session_token: str | None = field(default=None, repr=False)
```

日志 / telemetry / exception / repr 不能打印：

- SecretId 全量（必要时只显示 hash/prefix）
- SecretKey
- session token
- Authorization header
- signed URL

### 兼容 research

如果开发机确实要读 `.cos.yaml`：

```text
DATA_ACCESS_ALLOW_COSCLI_CONFIG_PARSE=1
```

显式 opt-in。

但：

```text
production => 永远禁止
```

更推荐 research 也直接调用 `clean-cos-ro`。

---

# 4. P0-S2 — 增加 DataPrincipal + Dataset AccessPolicy

## 当前问题

`allowed_s3_prefixes()` 会收集所有注册 mirror prefix。

它证明的是：

```text
这个 URI 是 DataAccess 知道的数据
```

不是：

```text
当前 server/user 有权读这个 URI
```

如果仓库中注册：

```text
basic_market
fundamental
premium_alt
internal_private
```

路径白名单可能全部通过。

必须增加**逻辑授权层**。

## 最小实现

不要做大型 IAM 系统，只做 typed contract：

```python
@dataclass(frozen=True)
class DataPrincipal:
    principal_id: str
    roles: tuple[str, ...]
    server_id: str | None = None

@dataclass(frozen=True)
class AccessPolicy:
    allowed_datasets: frozenset[str]
    allowed_factor_namespaces: frozenset[str]
    allowed_actions: frozenset[str]
```

动作至少：

```text
dataset:list
dataset:read
factor:list
factor:read
uri:read
metadata:read
```

所有 backend 选择之前先：

```python
authorize_dataset(principal, dataset, action="dataset:read")
```

## 权限取交集

最终允许：

```text
DataAccess logical policy
∩ registry/path boundary
∩ COS CAM/IAM
∩ local cache ownership
```

任何一层 deny 都必须 deny。

## AccessDenied 不能 fallback 到更高身份

区分两种异常：

### 可以 backend fallback

```text
httpfs extension unavailable
S3 API capability unsupported
endpoint incompatibility
```

### 绝对不能 fallback

```text
403
AccessDenied
PermissionDenied
Forbidden
```

也就是说：

```text
受限 httpfs credential 403
→ 不准读取 ~/.cos.yaml 后换 credential
→ 不准换另一个 profile
→ 返回 AuthorizationError
```

---

# 5. P0-S3 — 本地 mirror / cache 也是数据权限边界

## 为什么严重

COS 权限只保护：

```text
cos://bucket/key
```

一旦执行：

```text
clean-cos-ro cp ...
```

数据已经成为本机：

```text
.../data.parquet
/tmp/data_access_cos_cache/...
```

其他用户可以完全绕过 COS：

```python
pd.read_parquet(local_file)
```

所以：

> 高权限服务器下载的数据，不能因为落盘就失去权限保护。

## 当前风险

目前看不到明确：

```text
cache root = 0700
data file = 0600
owner check
principal scope
policy digest
```

默认 `/tmp/data_access_cos_cache` 在共享服务器尤其危险。

## 必须修改

### 5.1 cache 按 user/principal 隔离

例如：

```text
$XDG_CACHE_HOME/quant-dataaccess/cos/<principal_scope>/
```

或：

```text
/tmp/data_access_cos_cache_<uid>/<principal_scope>/
```

不要所有 principal 共用一个 cache。

### 5.2 默认权限

Production：

```text
directory = 0700
file = 0600
```

如果团队明确使用 Unix group：

```text
0750 / 0640
```

必须显式配置，不要默认 world-readable。

### 5.3 安全临时文件

当前类似：

```text
file.parquet.tmp
```

改为随机唯一、原子创建：

```python
tempfile.NamedTemporaryFile(
    dir=dest.parent,
    delete=False,
)
```

或 `O_CREAT | O_EXCL`。

防止：

- symlink 替换；
- tmp 抢占；
- 并发 writer 覆盖。

### 5.4 production 检查 root

检查：

- root owner；
- group/world writable；
- world-readable；
- root 是否 symlink；
- principal scope 是否匹配。

过宽则 fail closed。

### 5.5 cache manifest 增加非 secret 权限身份

例如：

```json
{
  "principal_scope_id": "...",
  "access_policy_digest": "...",
  "credential_scope_id": "...",
  "remote_key": "...",
  "checksum": "..."
}
```

权限降低/策略变化后：

```text
旧的高权限 cache 不得自动复用
```

---

# 6. P0-S4 — FactorEngine 派生因子必须继承源数据权限

这是 DataAccess ↔ FactorEngine 协作里的重要泄露路径。

例如：

```text
premium_alt_data
      ↓
FactorEngine
      ↓
alpha_secret_001
      ↓
factor_lake
```

如果低权限服务器不能读 `premium_alt_data`，但能读 `alpha_secret_001`：

```text
restricted source
→ 通过因子结果被洗白
```

这是实际的数据泄露。

## 必须增加 classification/access tag 传播

Dataset contract 增加：

```text
access_tags
classification
```

例如：

```text
market.basic
fundamental
alt.premium
internal.restricted
```

Factor materialize 时：

```text
derived_access_tags =
union/max_sensitivity(source_access_tags)
```

写入：

- FactorCatalog
- full definition
- lineage
- factor lake metadata
- factor matrix manifest

读取：

```text
read_factors
factor_matrix
HTTP factor endpoints
```

统一做权限校验。

## 禁止自动降密

如果：

```text
premium source → public factor
```

必须存在显式审批：

```text
declassification_approved
reviewer
policy_version
reason
```

没有审批，派生数据权限不能低于输入。

---

# 7. P0-S5 — HTTP Service 不能继续“一把 API Key = 整台服务器权限”

当前服务只有一个共享：

```text
X-API-Key
```

这意味着拿到这个 key，就可能继承服务进程的所有 COS 和本地数据权限。

## 必须改为 key -> principal

例如：

```text
api_key hash
→ principal_id
→ roles
→ dataset scopes
→ factor scopes
```

不要求上 OAuth；配置文件/环境注入也可以。

## Endpoint 要求

### `/v1/datasets`

只返回 principal 可见 datasets。

Restricted dataset 名本身也属于 metadata，不应全部暴露。

### `/v1/read`

先 authorization，再调用 store/backend。

### `/v1/read_uri`

Production 默认：

```text
DISABLED
```

或要求：

```text
uri:read:admin
```

不能仅因为 URI 落在全局 registered prefix 就允许。

### `/v1/factors`

检查 FactorMeta 是否包含：

- expression
- source config
- local root
- lineage
- snapshot
- internal strategy metadata

普通 `factor:list` 只返回脱敏摘要。

敏感字段另需：

```text
factor:metadata_sensitive
```

### production 不允许 open mode

`DATA_ACCESS_API_ALLOW_OPEN` 只能 dev/research。

Production：

```text
auth required = true
```

不可绕过。

### runtime mode 一致

不能：

```python
ServiceSettings(production_mode=True)
```

但 DataAccess 底层还处于 research semantics。

建立统一：

```text
RuntimeSecurityContext(
    production,
    strict_semantics,
    principal,
    authorizer,
)
```

---

# 8. P1-S6 — 外部错误信息必须脱敏

现在某些异常会包含：

```text
allowed prefixes
完整 local path
COS prefix
```

HTTP 直接 `str(exc)` 返回时会泄露存储拓扑。

外部错误统一：

```text
resource is not authorized
request violates temporal contract
dataset unavailable
```

内部安全日志可以写：

```text
request_id
principal_id
dataset_id
policy_decision
```

不要写：

```text
secret
signed URL
完整 credential
高权限 dataset 列表
```

---

# 9. P1-S7 — 加 Git 历史 secret scan

本轮抽查当前源码没有发现直接硬编码真实 SecretKey，但这不能证明历史从未出现。

CI 增加：

```text
gitleaks
```

或：

```text
trufflehog
```

扫描：

```text
working tree + git history
```

`.gitignore` 至少加入：

```text
.cos.yaml
cos.yaml
*.credentials
credentials.*
*.secret
*.key
```

如果历史曾提交 credential：

```text
必须 rotate
```

不是只加 `.gitignore`。

测试 fixture 禁止出现真实格式的生产 key。

---

# 10. P0-PIT1 — 美股 `filing_date` 当前被错误当成“可同日使用”

## 数据事实

美股：

```text
StockBalance
StockIncome
StockCashFlow
```

正确 knowledge clock 是：

```text
filing_date
```

但是字典实际样本表现为：

```text
2026-05-27 00:00:00
2026-05-26 00:00:00
...
```

虽然 dtype 是 `timestamp[ns]`，但语义其实更像：

```text
只有 filing date
没有可信 intraday disclosure time
```

当前 `semantic_fields.yaml` 对美股财务大量：

```yaml
availability: same_day
```

于是：

```text
2026-05-27 filing
```

可能被：

```text
2026-05-27 开盘策略
```

使用。

这是 look-ahead。

---

# 11. P0-PIT2 — “timestamp dtype” 与 “时间语义”必须分离

`session_calendar` 当前存在一种通用假设：

```text
naive datetime
→ 当 UTC instant
→ 转交易所本地时区
```

对真正 UTC event timestamp 是对的。

对：

```text
filing_date = 2026-05-27 00:00:00
```

这种**日期标签伪装成 timestamp**则是错的。

如果转纽约：

```text
2026-05-27 00:00 UTC
→ 2026-05-26 evening New York
```

时间可能被提前一天。

## 增加明确 contract

至少：

```text
time_representation:
    date_label
    instant

time_precision:
    date
    timestamp

storage_timezone
semantic_timezone
```

## 推荐映射

### A股

```text
TradeDate:
  representation=date_label
  precision=date

PubDate:
  representation=date_label
  precision=date
  semantic_timezone=Asia/Shanghai

QuoteTime:
  representation=instant
  precision=timestamp
  storage_timezone=UTC
  semantic_timezone=Asia/Shanghai
```

### 美股

```text
TradeDate:
  representation=date_label
  precision=date

filing_date:
  representation=date_label
  precision=date
  semantic_timezone=America/New_York

FactNews.published_utc:
  representation=instant
  precision=timestamp
  storage_timezone=UTC
  semantic_timezone=America/New_York
```

以后如果有真实：

```text
SEC accepted_at timestamp
```

再声明：

```text
instant
```

**禁止以后再用 dtype 猜时间语义。**

---

# 12. P0-PIT3 — 当前 US date-only filing 必须保守到下一 session

对当前美股财务：

```text
knowledge_time = filing_date
precision = date
```

建议：

```text
availability = next_session_open
```

如果当前系统只支持日频：

```text
next_trading_day
```

也可以，但最好保留 session contract。

不要：

```text
same_day
```

这与 FactorEngine 当前金融时间策略里“date-only conservative next session”方向保持一致。

## 测试

```text
filing_date = Friday 2024-05-10

Friday open:
  unavailable

Friday close:
  unavailable under conservative date-only policy

next real trading session open:
  available
```

再测：

- holiday eve
- early close
- weekend
- calendar right boundary

---

# 13. P0-PIT4 — A股 `UpdateTime` 不是历史 revision availability

## 数据事实

A股字典明确：

```text
UpdateTime = 数据 freshness / 清洗更新时间
```

不能用来表示：

```text
市场何时知道修订
```

但现在：

```text
cos_contract_ashare.py
semantic_fields.yaml
```

普遍把：

```text
UpdateTime
```

放入 `revision_columns/revision_order`。

当前 ASOF 逻辑可能：

```text
same Symbol
same PubDate
same ReportPeriodEndDate
→ 取 UpdateTime 最新
```

在固定最新 snapshot 内做确定性去重可以。

但不能说：

```text
历史回测当时已知这个 2026 UpdateTime 的修订值
```

## 必须拆概念

```text
revision_availability_time
```

与：

```text
dedup_tiebreaker
```

完全分开。

当前 A股建议：

```text
knowledge_time = PubDate
revision_availability_time = None/unknown
dedup_tiebreaker = UpdateTime
```

## 增加 PIT fidelity

例如：

```text
pit_fidelity:
  knowledge_date_pit
  vintage_pit
  effective_only
  unsupported
```

如果 COS 没有历史 revision vintage：

A股财务只能承诺：

```text
PubDate knowledge-PIT
+ pinned source snapshot reproducibility
```

不能承诺：

```text
full bitemporal revision-vintage PIT
```

最终交付里必须明确说明这一点。

如果未来要真正支持重述：

上游需要保存：

```text
revision_available_at
source publication timestamp
immutable vintage/raw snapshot
```

然后：

```text
revision_available_at <= decision_time
```

---

# 14. P0-PIT5 — 美股 `timeframe` 必须 exactly-one

数据事实：

```text
quarterly
annual
trailing_twelve_months
```

做美股财务必须先选一个。

当前 contract 只要求：

```text
timeframe filter exists
```

且通用验证允许：

```python
timeframe=["quarterly", "annual"]
```

因为两个都属于 allowed enum。

这仍然会把：

```text
单季 + 年报 + TTM
```

混起来。

## 改成结构化 filter requirement

例如：

```python
FilterRequirement(
    field="timeframe",
    required=True,
    cardinality="exactly_one",
    allowed=(
        "quarterly",
        "annual",
        "trailing_twelve_months",
    ),
)
```

Production 下以下全部拒绝：

```text
missing
multiple
invalid
```

## `timeframe` 必须进入 identity

必须进入：

- DataRequest compiled identity
- source contract hash
- Factor semantic identity
- cache key
- full factor definition
- lineage

否则：

```text
quarterly factor
TTM factor
```

可能共享错误版本。

---

# 15. P0-PIT6 — calendar-required availability 不得 residual same-day fallback

真实日历加载已经比较严格，这是好事。

但检查 `read_joined` 里所有逻辑：

```text
next_trading_day
next_session_open
next_bar
after_close_next_open
```

如果 session/calendar 编译失败：

Production 必须：

```text
raise
```

不能：

```text
logger.warning
→ fallback same_day / >= knowledge
```

Research 若确实允许：

```text
allow_temporal_degradation=True
```

必须显式参数，并在 lineage：

```text
temporal_semantics_degraded=true
```

---

# 16. P1-PIT7 — 财务 duplicate/revision 冲突要 deterministic

对于：

### A

```text
(Symbol, PubDate, ReportPeriodEndDate)
```

### US

```text
(ticker, timeframe, period_end, filing_date)
```

如果同一 semantic key 有多行：

#### 完全相同

可以 dedup。

#### 值冲突

有可信 revision availability：

```text
按 revision availability PIT
```

没有：

```text
production =>
FinancialRevisionAmbiguityError
```

或者显式使用：

```text
current_snapshot_latest
```

但必须进入 lineage/identity。

绝不依赖 parquet scan order。

---

# 17. P0-XM1 — US Dividend 三套契约已经漂移

当前：

## `cos_contract_us.py`

已经正确表达：

```text
knowledge = declaration_date
effective = ex_dividend_date
strict PIT
```

## FactorEngine provider

也已经正确：

```text
declaration_date
missing declaration -> unavailable
currency=USD
```

## 但 DataAccess `semantic_fields.yaml`

仍有旧逻辑：

```yaml
cash_dividend:
  time_role: effective_time
  effective_time: ex_dividend_date
  join_policy: exact
  mining_allowed: true
```

这是明确的 contract drift。

## 修复

DataAccess semantic field 改为：

```text
knowledge_time = declaration_date
effective_time = ex_dividend_date
PIT = strict
missing declaration_date => unavailable
```

并加 currency 约束。

---

# 18. P0-XM2 — US dividend `cash_amount` 不是恒定 USD

字典实际有：

```text
USD
HKD
EUR
CAD
GBP
...
```

非 USD 占比不可忽略。

当前 COS 没有统一 FX 表。

所以禁止：

```text
cash_amount → 直接当 USD
```

## 推荐逻辑字段

### `cash_dividend_local`

返回：

```text
value
currency
```

并：

```text
cross_market_comparable=false
```

### `cash_dividend_usd`

要求：

```text
currency == USD
```

其他币种：

```text
unavailable / filtered
```

以后如果有 FX：

```text
cash_dividend_base_currency
```

必须显式依赖：

```text
FX dataset + FX PIT
```

---

# 19. P0/P1-XM3 — 单位系统从 scale 升级到 typed Unit + currency

当前：

```text
source_unit
canonical_unit
scale
```

对：

```text
Return bp → decimal
ROE % → decimal
```

足够。

对跨市场金额不够。

必须表达：

```text
dimension
currency
scale
cross_market_comparable
requires_fx
```

例如：

```python
UnitSpec(
    dimension="money",
    currency="CNY",
    scale=1.0,
    cross_market_comparable=False,
)
```

与：

```python
UnitSpec(
    dimension="money",
    currency="USD",
    scale=1.0,
    cross_market_comparable=False,
)
```

## 维度建议

```text
ratio
money
money_per_share
price
shares
identifier
boolean
count
```

## 强制规则

未经 FX 或市场内标准化，禁止跨市场直接联合：

```text
CNY money vs USD money
CNY/share vs USD/share
dynamic-currency dividend
```

允许统一：

```text
decimal return
dimensionless ratio
```

但也要显式决定：

```text
rank within market
还是
rank across markets
```

---

# 20. P0-XM4 — 财务同名概念不能静默 alias

特别注意：

### A股

```text
NetProfit
NpParentCompanyOwners
```

不是同义。

### US

```text
consolidated_net_income_loss
net_income_loss_attributable_common_shareholders
```

也不是同义。

DataAccess 不要让：

```text
net_profit
net_income
```

静默把：

```text
A consolidated
```

对到：

```text
US attributable common
```

建议 canonical concepts：

```text
net_income_consolidated
net_income_attributable
```

ROE：

```text
attributable numerator
```

必须配：

```text
attributable equity denominator
```

---

# 21. P0-XM5 — Financial flow semantics 要机器可读

A股 Income/CashFlow 很多字段：

```text
cumulative YTD
```

US：

```text
timeframe=quarterly
```

通常：

```text
single-period
```

所以：

```text
QoQ
quarterly margin
cash-flow growth
```

不能 rename 后直接共享公式。

增加：

```text
flow_semantics:
  cumulative_ytd_flow
  single_period_flow
  point_in_time_stock
```

FactorEngine：

### A

```text
quarterize cumulative YTD
```

### US quarterly

```text
already single-period
```

FactorEngine providers 已经有一部分这类语义；DataAccess/ContractIR 要与它同源。

---

# 22. P1-XM6 — Instrument ID 加 market namespace

跨市场统一结果中不要只有裸：

```text
asset
```

推荐：

```text
ashare:000001.SZ
us:AAPL
```

至少主键逻辑必须是：

```text
(market, instrument)
```

避免未来：

- 港股
- ETF
- index
- ADR
- 同 ticker/symbol 空间

发生冲突。

物理源仍保留原 vendor code。

---

# 23. P1-XM7 — 复权公式同类，不等于 raw field 可混

A：

```text
Close * Factor
```

US：

```text
Close * AdjFactor
```

公式都属于 backward adjustment multiplier。

但：

- 基期不同；
- source 不同；
- corporate action 覆盖不同；
- US 有 clamped marker。

所以 canonical concept 应该是：

```text
adjusted_price_backward
```

由 market provider 构造。

不要把：

```text
Factor
AdjFactor
```

直接统一成一个 raw column。

---

# 24. P1-S8 — `read_uri` 是 privileged API

Library 内 `read_uri()` 虽有 path whitelist，但它绕过：

```text
dataset id
dataset semantic contract
dataset classification
```

Production：

```text
HTTP read_uri default disabled
```

普通 FactorEngine 也不能自动 fallback `read_uri()`。

只有明确 privileged/admin 调用才允许。

---

# 25. P1-S9 — AuthorizationError 不得被 broad `except Exception` 吞掉

全面搜索：

```python
except Exception:
    return None
```

或：

```python
except Exception:
    table = None
```

尤其：

- universe resolution
- calendar
- metadata probe
- coverage
- remote fallback
- mirror

如果异常是：

```text
AuthorizationError / AccessDenied
```

必须原样传播。

绝不能：

```text
Forbidden
→ empty universe
→ fallback data
→ research default
```

必须区分：

```text
NotFound
NoRows
Forbidden
Corruption
TemporalContractError
```

---

# 26. 推荐的最小 security 模块

不要为了安全重新写 DataAccess。

增加薄层：

```text
dataaccess/security/
    principal.py
    policy.py
    credentials.py
    redaction.py
```

接口：

```python
class CredentialProvider(Protocol):
    def resolve(self) -> CredentialMaterial:
        ...

class DatasetAuthorizer(Protocol):
    def authorize(
        self,
        principal: DataPrincipal,
        dataset: str,
        action: str,
    ) -> None:
        ...
```

`DataAccessStore` 持有：

```text
principal
authorizer
credential_provider
```

共享给：

```text
read
scan
read_joined
read_factors
read_uri
mirror
remote
HTTP
```

不要每个 backend 自己猜身份。

---

# 27. 多服务器 COS 权限应该怎么部署

用户要求：

> 不同服务器访问 COS 的权限不同，必须保留。

正确模型：

## Server A

```text
principal = server-a

CAM/IAM:
  basic_market only

DataAccess AccessPolicy:
  basic datasets only
```

## Server B

```text
principal = server-b

CAM/IAM:
  basic + premium

DataAccess AccessPolicy:
  basic + premium
```

代码完全相同。

差异来自：

```text
部署身份 + 策略
```

不是：

```text
代码里不同 SecretKey
```

任何人 clone GitHub repo：

```text
只有代码
没有 server principal
没有 scoped credential
=> 不能读取 COS
```

---

# 28. `clean-cos-ro` 最终定位

如果 `clean-cos-ro` 是当前团队已配置好的只读入口：

```text
保留
```

而且 DataAccess CLI backend 应尊重它。

DataAccess：

```text
只调用：
clean-cos-ro ls/cp
```

不要：

```text
读取 ~/.cos.yaml
提取 base credential
猜 profile
替它决定身份
```

即：

```text
CLI = credential boundary
DataAccess = caller
```

---

# 29. Security + PIT 信息进入 lineage

每次 read 至少记录非 secret：

```text
principal_id
server_id
access_policy_digest
credential_scope_id
dataset
data_snapshot_id
temporal_contract_digest
unit_contract_digest
market
PIT mode
availability policy
time representation
```

Factor materialize 再记录：

```text
source access tags
derived access tags
source temporal fidelity
source unit/currency contract
```

以后才能追：

```text
谁
在哪台服务器
按什么权限
读了哪个 snapshot
使用什么 PIT
生成了什么 factor
```

---

# 30. 必须新增 Security destructive tests

## T-S01 — server scope

```text
principal only allows ashare
registry contains ashare + us
→ ashare success
→ us AuthorizationError
```

## T-S02 — 不得 privilege fallback

```text
restricted httpfs credential gets 403
machine has broader ~/.cos.yaml
→ MUST NOT retry with broader credential
→ final = AuthorizationError
```

## T-S03 — production secret boundary

```text
~/.cos.yaml contains credentials
production startup/read
→ DataAccess does not parse it
→ COS_SECRET_* not injected into os.environ
```

## T-S04 — repr redaction

```text
repr(S3Credentials)
logs
exception
→ no secret key/token
```

## T-S05 — STS

```text
access key + secret + session token
→ backend works
```

## T-S06 — high cache cannot be reused

```text
high principal writes cache
low principal later starts
→ cache scope mismatch
→ cannot read/reuse high cache
```

## T-S07 — unsafe permissions

```text
production cache root world-readable/world-writable
→ fail or securely correct before use
```

## T-S08 — symlink/tmp attack

```text
pre-create malicious symlink/tmp
→ download cannot overwrite target outside root
```

## T-S09 — filtered dataset catalog

```text
basic API principal GET /v1/datasets
→ premium dataset names absent
```

## T-S10

```text
basic principal POST /v1/read premium dataset
→ 403 before backend
```

## T-S11

```text
basic principal /v1/read_uri premium prefix
→ 403
```

## T-S12

```text
production + DATA_ACCESS_API_ALLOW_OPEN=1
→ still requires auth / startup rejects
```

## T-S13 — derived factor

```text
factor derived from premium source
basic principal read factor
→ 403
```

---

# 31. 必须新增 Financial PIT destructive tests

## T-P01 — A PubDate

```text
period_end = 2024-03-31
PubDate = 2024-04-30

decision < PubDate
→ unavailable
```

## T-P02 — A next session

```text
PubDate=T
decision=T
→ unavailable

decision=next real trading session
→ available
```

## T-P03 — A holiday/weekend

必须使用真实交易日历，不是 `+ 1 day`。

## T-P04 — A UpdateTime hindsight

构造：

```text
same Symbol/PubDate/Period
old UpdateTime=2024
revised UpdateTime=2026
```

如果没有真实 revision availability：

```text
历史 PIT 不得宣称 2024 已知 2026 revision
```

行为必须由：

```text
pit_fidelity + snapshot policy
```

明确控制。

## T-P05 — same PubDate multiple periods

```text
同一 PubDate：
2023 annual
2024 Q1
```

`latest_period` 必须按 period 规则选，不依赖文件顺序。

## T-P06 — US filing date only

```text
filing_date=2024-05-10 00:00
representation=date_label

same-day open
→ unavailable

next session open
→ available
```

## T-P07 — US filing 不 UTC date shift

```text
2024-05-10 00:00
date_label
→ semantic date remains 2024-05-10
```

不能变成纽约 `2024-05-09 20:00`。

## T-P08 — true UTC instant

`FactNews.published_utc`：

```text
representation=instant
→ UTC -> America/New_York
```

证明 date_label 与 instant 是两条不同代码路径。

## T-P09 — timeframe exactly-one

以下拒绝：

```text
None
["quarterly","annual"]
"foo"
```

仅接受一个 enum。

## T-P10 — US revised filing

```text
same period
filing t1
filing t2

decision between t1/t2
→ t1

decision after t2
→ t2
```

## T-P11 — calendar missing

production 下真实 calendar 读取失败：

```text
hard fail
```

不能 same_day fallback。

---

# 32. 跨市场单位 / 定义测试

## T-X01 Return

```text
A Return=100 bp
→ canonical 0.01

US Ret=0.01
→ canonical 0.01
```

## T-X02 ROE

```text
A Roe=5.4%
→ 0.054

US ROE=0.054
→ 0.054
```

## T-X03 money

```text
A market_cap CNY
US market_cap USD

direct joint cross-market rank without FX/market-normalization
→ reject
```

## T-X04 US dividend currency

```text
USD row -> allowed in usd concept
HKD/EUR/CAD -> not silently treated as USD
```

## T-X05 flow semantics

```text
A cumulative YTD revenue
US quarterly single-period revenue
```

必须通过 canonical quarterization 后才比较。

## T-X06 attribution

```text
consolidated net income
attributable net income
```

必须是两个概念。

## T-X07 identifier

```text
ashare:000001.SZ
us:AAPL
```

跨市场主键必须唯一。

---

# 33. ContractIR 必须成为真正的语义漂移 CI

当前 ContractIR 方向正确，但 audit 还不够强。

对：

```text
COSDatasetContract
SemanticFieldCatalog
FactorEngine Provider
```

比较至少：

```text
market
temporal_model
pit_policy
knowledge_time
knowledge_time_precision
time_representation
effective_time
period_time
availability
revision_availability_time
dedup_tiebreaker
period_selection
required_filters
required_filter_cardinality
source_unit
canonical_unit
currency
flow_semantics
cross_market_comparable
```

任意 authoritative mismatch：

```text
CI fail
```

它应该能自动抓到当前这种：

```text
US dividend:
COS contract = declaration_date strict PIT
FE provider = declaration_date strict PIT
DA semantic field = ex_dividend exact
```

---

# 34. DataAccess 与 FactorEngine 的职责边界

最终建议明确：

## DataAccess owns

```text
physical source
dataset identity
market
source unit/currency
temporal/PIT contract
knowledge/effective/period clocks
availability
required filters
data access classification
snapshot/version
```

## FactorEngine owns

```text
factor concepts
operator semantics
derived formulas
quarterization / growth formulas
neutralization
factor semantic identity
materialization
```

FactorEngine 不应该重新猜：

```text
PubDate vs filing_date
same_day vs next_session
CNY vs USD
source scale
dataset access classification
```

这些从 DataAccess Contract IR 消费。

---

# 35. 其他 P1/P2 清理

## P1-1

`dataaccess/cos/mirror.py` 有重复 `_expected_dates()` 定义。

删除重复，保留唯一 implementation。

## P1-2

`materialize_remote_via_cli()` 有连续重复 `raise ValidationError`。

删掉 unreachable duplicate。

## P1-3

类似：

```text
/home/shw/quant_projects/...
```

只作为 dev legacy fallback。

Production 要求显式 data root/workspace root。

## P1-4

HTTP 外部异常不要泄露完整 local/COS path。

## P1-5

security audit 日志分类：

```text
AUTHENTICATION_FAILED
AUTHORIZATION_DENIED
PIT_REJECTED
DATA_NOT_FOUND
DATA_CORRUPTION
QUERY_INVALID
```

不要全是泛化 400/422。

---

# 36. 必须保留的不变量

改完后保证：

```text
GitHub repo access != COS data access
```

拿到代码的人：

```text
没有 principal
没有 scoped credential
没有 policy
=> 不能读 COS
```

并保证：

```text
server A privilege < server B privilege
```

不会因为它们：

```text
使用同一份代码
同一份 datasets.yaml
```

而被抹平。

最终权限是：

```text
Principal
∩ AccessPolicy
∩ RegisteredDatasetBoundary
∩ CAM/IAM/STS
∩ LocalCacheOwnership
```

---

# 37. 实现顺序

不要同时大面积改。

## Phase 1 — Credential boundary

修：

```text
cos/remote.py
CredentialProvider
STS token
no production .cos.yaml parse
no process-global secret env injection
secret redaction
```

## Phase 2 — Dataset authorization

修：

```text
principal
policy
store entry gates
remote/mirror gates
HTTP scopes
```

## Phase 3 — Secure local cache

修：

```text
per-principal root
permissions
secure tmp
cache scope binding
```

## Phase 4 — Financial temporal semantics

修：

```text
date_label vs instant
US filing next session
A UpdateTime semantics
timeframe exactly-one
strict calendar
financial duplicates
```

## Phase 5 — Cross-market contracts

修：

```text
currency
typed units
flow semantics
net-income attribution
dividend PIT/currency
instrument namespace
```

## Phase 6 — Derived-data permission inheritance

修：

```text
Factor lineage
FactorCatalog
factor lake
factor matrix
HTTP factor auth
```

## Phase 7 — ContractIR + tests

最后统一跑。

---

# 38. Definition of Done

## Security

- [ ] production 不解析 `~/.cos.yaml` SecretKey
- [ ] production 不向 `os.environ` 注入 COS secret
- [ ] STS session token 支持
- [ ] credential repr/log 脱敏
- [ ] dataset authorization 在 backend 之前执行
- [ ] AccessDenied 不更换 credential 重试
- [ ] server-specific policy 有测试
- [ ] mirror/cache principal 隔离
- [ ] cache 0700/0600 或显式 group policy
- [ ] cache policy digest / principal scope 绑定
- [ ] HTTP API key 映射 principal/scopes
- [ ] dataset list 过滤
- [ ] production read_uri privileged/disabled
- [ ] factor derived access classification 传播
- [ ] git history secret scan CI

## PIT

- [ ] A finance knowledge = PubDate
- [ ] A date-level finance conservative next session
- [ ] A UpdateTime 不再充当 historical revision availability
- [ ] US finance knowledge = filing_date
- [ ] US filing_date = date_label
- [ ] US date-only filing = conservative next session
- [ ] US timeframe exactly one
- [ ] strict calendar unavailable = hard fail
- [ ] financial duplicate/revision ambiguity deterministic/fail closed
- [ ] US dividend = declaration_date strict PIT
- [ ] missing declaration_date = unavailable
- [ ] future ex-dividend 不产生前视

## Cross-market

- [ ] A Return bp / US Ret decimal parity
- [ ] A percent ratio / US decimal ratio parity
- [ ] CNY/USD money 不能静默混合
- [ ] US dividend dynamic currency 正确处理
- [ ] financial flow semantics machine-readable
- [ ] consolidated vs attributable 分开
- [ ] time representation machine-readable
- [ ] instrument ID market-scoped
- [ ] ContractIR CI 能抓 COS/Semantic/FE provider 漂移

---

# 39. 修改完成后必须提交的报告

不要只回复“改完了”。

必须输出：

## A. Changed Files

每个文件：

```text
path
修改内容
为什么
```

## B. 最终 Security Model

说明：

```text
server identity 从哪来
AccessPolicy 从哪来
COS credential 从哪来
clean-cos-ro 负责什么
httpfs 负责什么
本地 cache 怎么隔离
权限拒绝怎么处理
```

## C. PIT Matrix

至少：

| Market | Dataset | Knowledge time | Representation | Precision | Availability | Period | Revision fidelity |
|---|---|---|---|---|---|---|---|

## D. Unit Matrix

至少：

| Concept | A source unit | A canonical | US source unit | US canonical | Currency issue | Cross-market comparable |
|---|---|---|---|---|---|---|

## E. Tests

必须列：

```text
test file
test count
passed/failed
```

包括：

- security destructive tests
- finance PIT tests
- cross-market parity tests
- existing DataAccess regressions
- FactorEngine integration regressions

## F. Known Limitations

必须回答：

> A股当前是否真正支持 historical revision-vintage PIT？

如果 COS 没保存 historical revisions 的真实 availability：

必须明确写：

```text
No.
Current system can prove PubDate knowledge-PIT on a pinned source snapshot,
but cannot prove full historical revision-vintage PIT.
```

不要用 `UpdateTime` 假装已经解决。

---

# 40. 最终约束

1. **绝不能把真实 SecretId/SecretKey/token 写进源码、测试、文档、日志。**
2. 隐藏 bucket 名不是安全；真正权限来自 principal + policy + CAM/IAM。
3. 不要为了 httpfs 削弱 `clean-cos-ro` 原本的 server-specific 权限。
4. AccessDenied 不是 backend error，禁止 privilege fallback。
5. `timestamp[ns]` 不代表天然 UTC instant。
6. A/US 同名字段不代表同单位、同货币、同会计定义。
7. A股 `UpdateTime` 不得作为历史市场可知时间。
8. US financial `timeframe` 不允许多个值混用。
9. Restricted raw data 派生 factor 必须继承权限。
10. 不要破坏此前已稳定的 DataAccess Core / FactorEngine 主链。

---

# 41. 本轮最终期望状态

修完应达到：

```text
同一份 DataAccess 代码
+
不同 server principal
+
不同 CAM/IAM policy
=
不同的、不会被代码自动抹平的数据访问能力
```

PIT：

```text
A股财务
  PubDate(date label)
  → conservative next session
  → period selection
  → pinned snapshot

美股财务
  filing_date(date label)
  → conservative next session
  → exactly-one timeframe
  → period selection
```

跨市场：

```text
字段名映射
≠
语义相等

只有：
单位 + 币种 + 时间 + 会计口径 + availability
全部兼容
才允许进入通用 factor/operator
```

最终不变量：

> **生产环境里，任何一个 DataAccess read 或 FactorEngine factor value，如果不能证明“有权限、已可知、语义一致、版本可追溯”，就必须失败，而不是猜、降级、换身份或静默继续。**
