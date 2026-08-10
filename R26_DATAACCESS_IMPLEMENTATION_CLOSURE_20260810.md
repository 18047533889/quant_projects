# R26 — DataAccess Implementation Closure、Clean-Checkout Hardening 与真实执行链收口

> 仓库：`18047533889/quant_projects`  
> 审计基线：`main@58490a634eef36efc92ecaec945fc4a26016661c`  
> 日期：2026-08-10  
> 前序：R24 / R25 以及此前所有 DataAccess / FactorEngine × DataAccess 整改  
> 本轮性质：**R25 落地后的二阶审计与实现收口，不再扩功能。**

---

## 0. 本轮目标与总原则

R25 的架构方向基本正确，已经新增了：

- `RuntimeDatasetContract`
- `PhysicalPartitionSpec`
- `FilterRequirement`
- `TemporalAxisSpec`
- `AvailabilityResult`
- `ResolvedSourceSnapshot`
- `SnapshotVerifier`
- `GlobalResourceGovernor`
- `CacheManager`
- `RunMode`
- `GovernedFrame`
- `Production Startup Gate`
- mirror generation / source snapshot / resource budget 等机制

但当前 `main@58490a6` 仍存在一个核心问题：

> **部分架构只是“类型、helper、测试对象存在”，并没有成为所有 production public path 的唯一执行链；同时 current GitHub clean checkout 与 R25 closure report 的“875 passed / FULL PLATFORM FREEZE=YES”存在可验证的不一致。**

因此 R26 不再加新能力，只做以下四件事：

1. **Clean checkout 必须可安装、可 import、可测试。**
2. **R25 新架构必须真正接管所有 production execution path，不能只存在于 helper/test。**
3. **所有安全、PIT、snapshot、resource、schema gate 必须 fail-closed，不能“异常→旧逻辑→继续跑”。**
4. **所有 closure test 必须走真实 public path，不允许用测试文件里自己写一个模拟 helper 来证明 production 已实现。**

### 最高原则

```text
No evidence != safe
Unable to prove != allow
Contract compile failure != fallback
Catalog unavailable != unrestricted
Security config missing != full access
Snapshot unknown != pinned
Calendar missing != same-day
Helper test passed != execution path closed
```

---

# 1. 当前审计结论

## 1.1 当前 HEAD

```text
main = 58490a634eef36efc92ecaec945fc4a26016661c
commit message:
Sync FE R27 throughput / R28 audit work and DA R25 platform closure.
```

该 commit 相对上一轮 `37008c7...` 已提交 R25 的大批代码与测试。

## 1.2 当前 Freeze 判定

```text
R25 FULL PLATFORM FREEZE = REVOKED PENDING R26
```

原因不是 R25 方向错误，而是当前 GitHub committed tree 仍存在 confirmed blocker：

- `dataaccess/security/credentials.py` 在当前 HEAD 不存在；
- `.gitignore` 的 `credentials.*` 会直接匹配并忽略 `credentials.py`，很可能是根因；
- 但 `dataaccess/security/__init__.py`、`cos/remote.py`、`runtime/startup_gate.py`、R25 security tests 都在 import 它；
- `pyproject.toml` 显式 `packages=[...]`，却没有包含 R25 新增的 `security / contract / runtime / snapshot` 子包；
- 当前 HEAD 的 GitHub combined status 与 workflow runs 都为空；
- R25 报告声称 875 passed，但 committed tree 中 security test 自身依赖一个不存在的模块，因此该结果最多代表某个本地工作树，**不能代表当前 clean clone**；
- R25 新增的 Snapshot / Governor / Cache / GovernedFrame / StartupGate / PreparedRead 等部分未真正接入全部 production public path；
- 仍存在 PIT、权限、remote glob、filter scope 等实际执行漏洞。

---

# 2. 已经确认落地的 R25 内容 —— 不要重复返工

Coding agent 先读这一节。以下内容除非 R26 测试发现 regression，否则**不要重新发明第三套实现**：

1. `FilterRequirement` 已支持 `RuntimeDatasetContract.filters`。
2. Store 已新增 `_enforce_runtime_contract_filters()`，并把 `required_event_filters` 纳入 dataset gate。
3. US finance 已新增 `period_files / PERIOD_END_FILE` 基础布局。
4. StockCapital 已新增 `file_selector / filename_template` 基础能力。
5. `RunMode` 已区分：
   - `interactive_research`
   - `automated_research`
   - `production`
6. `automated_research` 已进入 strict semantic 判定。
7. `ResolvedSourceSnapshot / ResolvedObject / SnapshotVerifier` 类型已经创建。
8. `GlobalResourceGovernor / ResourceReservation` 已经创建。
9. `CacheManager` 已经创建。
10. `AvailabilityResult / CalendarUnavailableError` 已经创建。
11. mirror generation/pointer 的基础实现已经存在。
12. schema 首访检查、QueryBudget 基础校验、PIT date_label/instant 基础区分已经存在。
13. FactorEngine 已有 `factor_engine/security/access.py` 的 source tag 派生与 sensitivity 基础模型。

R26 要做的是：

> **把这些机制真正接成唯一执行链，并修正其自身仍存在的 bug。**

---

# 3. R26 P0 — 必须全部关闭后才能再次 Freeze

---

## R26-P0-001 — Clean checkout 不完整：`credentials.py` 被 Git 忽略 / committed tree 缺模块

### 当前事实

当前 HEAD：

```text
dataaccess/security/credentials.py
→ GitHub 404 Not Found
```

但以下模块均依赖它：

```text
dataaccess/security/__init__.py
dataaccess/cos/remote.py
dataaccess/runtime/startup_gate.py
dataaccess/tests/unit/test_r25_security_fe_2026_08.py
...
```

`.gitignore` 当前存在：

```gitignore
credentials.*
```

这会匹配：

```text
credentials.py
```

因此非常可能是：

> 本地有 `credentials.py`，测试从本地工作树通过，但 Git 永远没有提交它。

### 风险

这是 clean checkout 级 blocker：

```text
git clone
→ import data_access.security
→ ModuleNotFoundError
```

R25 “875 passed”不能作为 current GitHub tree 的证据。

### 必须整改

1. 调整 `.gitignore`：
   - 不要使用会吞掉源码文件的 `credentials.*`；
   - 至少增加显式 exception：

```gitignore
!dataaccess/security/credentials.py
```

更推荐改为只忽略真实凭证文件后缀/目录，而不是源码名。

2. 把**不含任何真实 secret** 的 `credentials.py` 正式纳入 Git。
3. Credential source code 与 credential material 必须彻底区分。
4. 建立 clean-tree source inventory 检查：
   - 所有 Python import target 必须在 Git index 或合法依赖中存在；
   - 禁止“本地 untracked module 才能跑”。

### 必测

```bash
git clean -xfd
git status --porcelain   # 必须为空
python -c "import data_access.security"
python -c "from data_access.security.credentials import CredentialProvider"
```

并新增 CI：

```text
T-R26-CLEAN-001
clean checkout import data_access/security/cos/runtime
```

---

## R26-P0-002 — Packaging 不完整：R25 新子包没有进入 wheel

### 当前事实

`dataaccess/pyproject.toml` 使用显式包列表：

```toml
[tool.setuptools]
packages = [
    "data_access",
    "data_access.core",
    "data_access.registry",
    "data_access.read",
    "data_access.write",
    "data_access.cos",
    "data_access.service",
    "data_access.clickhouse",
    "data_access.quality",
]
```

但 R25 新增：

```text
data_access.security
data_access.contract
data_access.runtime
data_access.snapshot
```

没有列入。

因为这里不是自动 discovery，而是 explicit packages，因此 source tree 测试通过并不能证明：

```text
pip install wheel
```

后还能 import。

### 必须整改

优先改为 setuptools package discovery：

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["data_access*"]
```

并正确处理 `package-dir`。

如果继续显式列表，则必须完整列出所有子包，但不推荐人工维护。

### 必测

必须新增**安装后测试，而非 repo source 测试**：

```bash
python -m build dataaccess
python -m venv /tmp/r26-wheel-test
/tmp/r26-wheel-test/bin/pip install dataaccess/dist/*.whl
cd /tmp
/tmp/r26-wheel-test/bin/python - <<'PY'
import data_access
import data_access.security
import data_access.contract
import data_access.runtime
import data_access.snapshot
from data_access.security.credentials import CredentialProvider
PY
```

CI 必须把 wheel smoke test 设为 required。

---

## R26-P0-003 — `PreparedRead` 被 R25 架构声明为核心，但当前 committed tree 没有实现文件

### 当前事实

R25 report/runtime doc 均宣称：

```text
Request
→ RuntimeDatasetContract
→ PreparedRead
→ SourceSnapshotResolver
→ ResourceGovernor
→ SnapshotVerifier
→ Backend
```

`dataaccess/runtime/__init__.py` doc 也写了：

```text
prepared_read PreparedRead
```

但当前 HEAD：

```text
dataaccess/runtime/prepared_read.py
→ 404
```

`runtime/__init__.py` 也没有导出 `PreparedRead`。

### 风险

当前真实系统仍然是多个路径分别解释：

```text
registry
COS contract
semantic catalog
mirror spec
params
predicate
```

R25 目标的：

```text
一个 immutable executable plan
```

没有成为事实。

### 必须整改

实现正式：

```python
@dataclass(frozen=True)
class PreparedRead:
    request_identity
    dataset
    runtime_contract
    security_context
    effective_filters
    temporal_plan
    physical_scope
    resolved_source_snapshot
    query_budget
    resource_reservation
    backend_plan
    lineage_seed
```

并要求：

```text
所有 public read path
先 prepare()
再 execute(prepared_read)
```

禁止 backend 自己重新解释 contract。

---

## R26-P0-004 — R25 新架构只“存在”，没有真正接管 Store execution path

### 当前事实

R25 新增：

```text
snapshot/
runtime/resource_governor.py
runtime/cache_manager.py
security/governed_frame.py
```

但当前主 Store/HTTP execution path 没有形成：

```text
resolve snapshot
→ governor admission
→ pin cache
→ verify
→ execute
→ verify
→ release
```

Store 最新 commit 的核心接线主要是 filter gate，而不是上述完整链路。

### 必须整改

建立唯一：

```python
store.prepare_read(...)
store.execute_prepared_read(...)
```

所有路径复用：

```text
read_result
read_arrow
read_frame
read
read_auto
read_arrow_stream
scan_polars
read_joined
sql
read_uri
read_factors
factor_matrix
HTTP
FactorEngine DataAccessSource
```

不得出现：

```text
某些 path 有 snapshot
某些 path 没有
某些 path 有 governor
某些 path 直接 engine.execute
```

### 必测

每个 public path 注入 fake instrumentation：

```text
contract_gate_called == 1
auth_called == 1
snapshot_resolved == 1
resource_admitted == 1
snapshot_verified_before == 1
backend_execute == 1
resource_released == 1
```

所有 path 必须通过同一 invariant test。

---

## R26-P0-005 — HTTP API principal 没有真正进入 Store 的嵌套执行上下文

### 当前事实

HTTP 层已经解析：

```text
X-API-Key
→ DataPrincipal
→ AccessPolicy
→ _ApiCallContext
```

但 backend 调用仍然是：

```python
get_store().read_result(...)
get_store().read_arrow_stream(...)
get_store().read_factors(...)
```

没有把 request principal / authorizer / credential scope 传入 Store。

Store 自己保存：

```text
self._principal
self._authorizer_sec
self._access_policy
```

它们是 process-level store creation state。

### 风险

顶层 HTTP 可能先授权 dataset A，但内部嵌套读仍可能使用 process principal：

```text
calendar
universe
factor catalog
factor tags
dependent datasets
remote credential
cache scope
audit
lineage
```

这不是完整 request-scoped authorization。

并发 API key A/B 时还可能产生身份串扰。

### 必须整改

不要每个 request 改全局 `set_authorizer()`。

建立 ContextVar：

```python
DataAccessExecutionContext(
    principal,
    authorizer,
    access_policy,
    credential_provider,
    request_id,
    run_mode,
    security_digest,
)
```

例如：

```python
with data_access.execution_scope(ctx):
    store.read(...)
```

所有 nested reads 必须自动继承。

### 必测

并发 destructive test：

```text
Thread A: basic key，只能 basic dataset
Thread B: premium key，可读 premium
1000 次交错
```

要求：

```text
A 永远不能读 premium
B 不被 A 的权限降级
calendar/universe/derived/factor nested read 都保持各自 principal
audit request_id/principal 一致
```

---

## R26-P0-006 — `AccessPolicy` 的空集合语义是 fail-open，与注释/部署直觉冲突

### 当前代码行为

```python
if not allowed_datasets:
    return True

if not allowed_factor_namespaces:
    return True

if not allowed_actions:
    return True
```

也就是说：

```json
"allowed_datasets": []
```

实际不是 deny all，而是 unrestricted。

### 风险

部署者最容易把：

```text
[]
```

理解成：

```text
暂时不给任何权限
```

当前却变成：

```text
全部权限
```

属于典型 fail-open config bug。

### 必须整改

三态必须显式：

```text
None / "*"  => unrestricted
[]          => deny all
["a","b"]   => allow exact set
```

不要用一个 empty set 同时代表“没配置”和“允许全部”。

同样适用于：

```text
allowed_actions
allowed_factor_namespaces
```

### 必测

```text
allowed_datasets=[]              -> deny all
allowed_datasets=None/"*"        -> unrestricted（只在显式允许的场景）
allowed_datasets=["a"]           -> only a
```

---

## R26-P0-007 — Production 默认授权仍然过宽；“没配策略”不能继续启动

### 当前问题

`_authorizer_from_env()` 未配置：

```text
DATA_ACCESS_ALLOWED_DATASETS
```

时返回：

```text
DEFAULT_ACCESS_POLICY
```

该 policy 基本是：

```text
all registered datasets + all actions
```

### 正确 production 语义

```text
production
AND
no explicit principal/policy
=> startup hard fail
```

不能：

```text
缺配置 => 本地默认超级用户
```

### 必须整改

Production Startup Gate 检查：

```text
explicit principal configured
explicit policy configured
policy source version/hash known
no wildcard privilege unless explicit
```

research 才允许 DEFAULT_LOCAL_PRINCIPAL。

---

## R26-P0-008 — Security config 使用 `bool("false")`，可把关闭配置解析成开启

### 当前问题 A：API principal

```python
allow_uri = bool(entry.get("allow_uri_read"))
allow_sensitive = bool(entry.get("allow_metadata_sensitive"))
```

所以：

```python
bool("false") is True
```

### 当前问题 B：Factor declassification

`factor_engine/security/access.py`：

```python
approved = bool(raw.get("approved"))
```

同样：

```json
"approved": "false"
```

可能被解析为 `True`。

### 必须整改

统一 strict config parser：

```text
bool 只接受真正 bool
enum 严格
unknown key forbid
missing sensitive field deny
```

推荐 Pydantic strict model：

```python
ConfigDict(extra="forbid", strict=True)
```

### 必测

```text
false      -> False
true       -> True
"false"    -> reject
"true"     -> reject
1/0        -> reject（安全配置）
unknown key -> reject
```

---

## R26-P0-009 — Factor derived access gate 仍然是 `any(tag)`，且 catalog 故障 fail-open

### 当前 Store 逻辑

```python
if not any(t in allowed for t in tags):
    denied
```

这意味着 factor：

```text
["market.basic", "alt.premium"]
```

用户只拥有：

```text
market.basic
```

仍可能通过。

此外：

```python
try:
    catalog = get_factor_catalog()
except Exception:
    return
```

即：

```text
catalog 不可用
→ 无法证明权限
→ 允许
```

### 必须整改

将权限分为：

1. classification/clearance：
   ```text
   public < basic < fundamental < premium < restricted
   ```
2. required entitlements/compartments：
   ```text
   premium.wind
   premium.massive
   alt.news
   ...
   ```

授权必须：

```text
principal.clearance >= factor.classification
AND
factor.required_entitlements ⊆ principal.entitlements
```

不是 `any()`。

Catalog unavailable：

```text
production => deny
```

Unknown factor metadata：

```text
production => deny
```

---

## R26-P0-010 — StockCapital split remote path 仍然会把 `shares_*.parquet` 混进来

### 当前事实

local registry 对 split 是正确的：

```yaml
us_stock_capital_split:
  glob: "[0-9]*.parquet"
```

用于排除：

```text
shares_*.parquet
```

但 remote `_remote_prefixed_date()`：

```python
selector = spec.file_selector or ""
return [f"{base}/{selector}*.parquet"]
```

split 的 `file_selector=None`，因此得到：

```text
*.parquet
```

会同时命中：

```text
2024-01-01.parquet
shares_2024-01-01.parquet
```

两类 schema。

R25 test 只测了：

```text
shares remote => shares_*.parquet
```

没有测：

```text
split remote
```

### 必须整改

不要用“空 prefix + *”表达排除逻辑。

推荐：

```text
remote LIST exact objects
→ basename regex ^\d{4}-\d{2}-\d{2}\.parquet$
→ exact split object list
```

shares：

```text
^shares_\d{4}-\d{2}-\d{2}\.parquet$
```

local/remote/mirror/snapshot resolver 共用同一 `FileSelector` IR。

### 破坏性测试

同一 fake bucket：

```text
2024-01-01.parquet               schema A
shares_2024-01-01.parquet        schema B
```

四模式：

```text
local
mirror
remote
auto
```

split 只能读 A；
shares 只能读 B。

---

## R26-P0-011 — RuntimeContract compile failure 被吞，最终等价于“关闭新 gate”

### 当前 Store

```python
try:
    rc = compile_runtime_contract(...)
except Exception:
    rc = None

if rc is None or not rc.filters:
    return
```

即：

```text
Runtime Contract 编译异常
→ 新 FilterRequirement gate 不执行
→ 继续走
```

`physical_partition_for()` 也有：

```python
try:
    compile_runtime_contract(...)
except Exception:
    pass
```

`runtime_contract.py` 对非法 contract layout：

```python
except ValidationError:
    contract_layout = None
```

然后 fallback mirror / default daily。

### 风险

authoritative contract typo/bug 可能变成：

```text
“contract 不可用”
→ “旧逻辑继续”
```

而不是 fail。

### 必须整改

Production / automated_research：

```text
RuntimeDatasetContract compilation failure
=> ContractCompilationError
=> hard fail
```

仅对**明确无 contract 的合法 dataset**允许走声明好的 legacy-compatible contract。

禁止：

```text
except Exception: pass
except Exception: return []
except ValidationError: fallback
```

出现在 security/PIT/contract/snapshot boundary。

---

## R26-P0-012 — FilterRequirement 仍存在跨 dataset 污染，且 `scope/read_mode` 没真正生效

### 当前 validator

会把：

```python
for ds_filters in filters_by_dataset.values():
    merged_filters.update(ds_filters)
```

合并所有 dataset filter。

例如：

```text
A.timeframe required
B.timeframe="quarterly"
```

B 的 filter 可以被 A 当成自己满足了 requirement。

此外：

```text
read_mode
```

参数当前没有真正用于：

```text
FilterRequirement.scope = panel/event/dimension/all
```

Store 甚至固定：

```text
read_mode="auto"
```

### 必须整改

Validator 必须是 per-dataset：

```python
validate_filter_requirements(
    dataset="A",
    contract=A_contract,
    params=A_params,
    global_predicate=...,
    dataset_predicate=filters_by_dataset["A"],
    read_mode=actual_mode,
)
```

不允许把 B 的过滤器 merge 进 A。

同时过滤条件统一先编译到：

```text
PredicateConstraint IR
```

不要只接受 raw Mapping。

### 必测

- A requirement 不能被 B filter 满足；
- panel requirement 只在 panel scope 生效；
- event requirement 在 event/pit scope 生效；
- `OR / NOT / != / IS NOT NULL` 不能伪装成“已限制到 allowed domain”。

---

## R26-P0-013 — SQL PIT 仍存在 same-day fail-open

### 当前问题

`_session_avail_sql()` 仍生成：

```sql
COALESCE(
    _cal._next_td,
    CAST(knowledge_time AS DATE)
) AS _avail_from
```

也就是说：

```text
calendar mapping 找不到
→ fallback knowledge date
→ same-day usable
```

这与 R25 的：

```text
calendar missing / right boundary unknown => fail closed
```

不一致。

### 必须整改

Production SQL 路径必须：

```text
mapping missing
=> CalendarUnavailableError / PITUnavailable
```

不能 `COALESCE` 到 knowledge。

Research 如显式 degradation：

```text
authoritative=False
degradation_reason=...
```

并禁止进入 automated mining / production publish。

---

## R26-P0-014 — Request 可以显式把 authoritative PIT contract 降级为更宽松语义

### 当前 `_effective_join_specs()`

优先级是：

```text
Field default
→ Contract default
→ Explicit request 最后覆盖
```

显式 request 被称为：

```text
“最高优先”
```

因此理论上系统要求：

```text
next_session_open
```

用户可能显式指定：

```text
same_day
```

而覆盖系统 floor。

另外 contract fallback 构造 `TemporalJoinSpec` 时没有显式带 `availability`，会吃默认：

```text
same_day
```

### 必须整改

建立：

```text
PITPolicyFloor
```

请求只能：

```text
same or stricter
```

不能 loosen：

```text
availability
future_cutoff
revision policy
period-selection safety
effective-time permission
```

Contract fallback 必须携带 authoritative availability。

### 必测

```text
contract next_session_open
request same_day
=> reject downgrade
```

---

## R26-P0-015 — SourceSnapshotResolver 的 snapshot 身份仍不够“可证明”，且没有完整接线

### 当前问题

### A. Manifest 可接受没有版本证据的 object

`ResolvedObject` 允许：

```text
etag=None
version_id=None
content_length=None
```

`content_digest_of_objects()` 仍会用：

```text
URI + "" + "" + 0
```

生成 digest。

这不是真正 content identity。

### B. `parse_source_manifest()` 可接受 raw string / 空 key

没有：

```text
URI normalization
registered prefix validation
duplicate object validation
complete marker validation
etag/version/size requirement
```

### C. test manifest 使用 `key="t/file.parquet"`

会被保存成：

```text
t/file.parquet
```

不是 `s3://...`。

SnapshotVerifier 会把它当本地 path。

### D. `_common_prefix()` 有 wildcard typo

当前：

```python
for marker in ("*", "? ", "{"):
```

`"? "` 是问号+空格，不是真正 `?`。

也没有完整支持：

```text
[
]
**
```

### E. policy 声称 latest/pin/fail_if_changed

实际只对：

```text
policy == "pin"
```

做特殊处理。

未知 policy 也不 reject。

### 必须整改

定义 strict：

```python
SnapshotPolicy = Literal["latest", "pin", "fail_if_changed"]
```

Production exact object 至少需要：

```text
canonical URI
AND (
    version_id
    OR etag + content_length
    OR cryptographic checksum
)
```

Publisher manifest 需要：

```text
manifest_version
dataset
generation
complete=true
objects[]
object_count
content_digest
prefix/root identity
published_at
```

所有 object 必须验证在 dataset registered boundary 下。

---

## R26-P0-016 — SnapshotVerifier before/after 逻辑不足，不能保证“执行期间没换文件”

### 当前问题

Remote before：

- 主要比 ETag；
- 没系统比 `version_id`；
- remote size 比较不完整。

Local before：

- 实际只比 size；
- 文档说 path+size+mtime/checksum，但代码没有把 mtime identity 真正校验起来。

After：

```text
只确认 HEAD 还存在
```

没有重新比较：

```text
etag
version_id
content_length
```

并且存在：

```python
if meta is None and self.strict:
```

这里用了原始 `self.strict`，而不是 `_effective_strict()`。

当：

```text
self.strict is None
环境实际上 production strict
```

after verification 可能不按 strict 失败。

### 必须整改

统一 immutable `ResolvedObjectIdentity`：

```text
remote:
uri + version_id
or uri + etag + content_length

local:
realpath + inode/device(optional) + size + mtime_ns + checksum/generation
```

Before/after 必须比较**同一 identity object**。

长 streaming read：

```text
开始 pin
每 batch/结束 verify strategy
close/release
```

---

## R26-P0-017 — ResourceGovernor 是 dead control：存在但未成为 production admission gate

### 当前事实

`GlobalResourceGovernor` 已新增，但 Store/HTTP 主链没有统一：

```text
get_global_governor().admit(...)
...
release(...)
```

### 额外自身问题

- `max_duckdb_concurrency` 只存了，没有执行；
- `remote_requests` 字段没参与 admission；
- duplicate `query_id` 可以覆盖 `_active[query_id]`；
- `released` 字段没参与状态；
- limit 参数缺完整 invariant validation；
- multi-worker contract 仅 warning/return False，不阻止 production。

### 必须整改

`PreparedRead` admission：

```text
snapshot exact objects
→ estimate bytes/memory/remote calls
→ QueryBudget
→ GlobalResourceGovernor.admit
→ backend slot
```

用 context manager：

```python
with governor.reserve(...):
    execute
```

任何：

```text
success
exception
timeout
client disconnect
generator close
```

都必须 release exactly once。

---

## R26-P0-018 — HTTP `_api_budget()` 丢掉 R25 QueryBudget v2 字段

### 当前问题

R25 新增：

```text
max_scan_objects
max_scan_bytes
max_remote_list_objects
max_remote_requests
max_estimated_memory
```

但 HTTP `_api_budget()` 构造新 QueryBudget 时只复制旧字段：

```text
max_rows
max_result_bytes
max_elapsed_ms
max_scan_files
require_columns
require_time_range
```

因此 web service path 可以把 v2 resource floor 丢掉。

Factor endpoint `read_factors()` 甚至没有传统一 `query_budget`。

### 必须整改

不要手写复制 budget。

实现：

```python
budget.tighten(...)
budget.with_overrides(...)
```

或者 dataclass replace，并保证所有字段保留。

HTTP、FE、CLI 全部使用同一 effective budget。

---

## R26-P0-019 — CacheManager 仍是模型，不是完整 cache governance

### 当前问题

虽然类存在，但：

```text
ttl
per_principal_quota
```

没有真正执行。

`CacheEntry` 没有 principal 字段，因此不可能落实 per-principal quota。

`_gc_locked()` 还存在 accounting 问题：

- entry 从 dict pop 后，`_total_bytes_locked()` 已经下降；
- 再减 `freed` 会 double-count；
- 可能提前认为到达 low watermark。

此外：

```python
Path(e.path).unlink()
```

对 cache directory 无效；
但 entry 已经从 manager 删除，可能出现：

```text
管理器认为释放
磁盘实际上没释放
```

### 必须整改

Cache entry identity 至少：

```text
cache_key
principal_scope
security_digest
source_snapshot_id
path
size
created_at
last_access
expires_at
refcount
state
```

目录使用安全递归删除，且只有在物理删除成功后更新 accounting。

TTL/quota/pin 必须有真实 end-to-end test。

---

## R26-P0-020 — Production Startup Gate 没接入服务启动，而且 gate 本身也少检查

### 当前事实

`runtime/startup_gate.py` 已存在，但：

```text
dataaccess/service/__main__.py
```

仍直接：

```python
uvicorn.run(...)
```

没有调用：

```text
run_startup_gate()
```

### Gate 声称检查

```text
security context
credential provider
critical calendar
cache
source snapshot provider
single-worker...
```

实际默认 runner 主要只有：

```text
ContractIR
CredentialProvider
legacy home
cache permissions
```

没有完整检查：

```text
security policy explicitness
critical calendar authoritative
source snapshot reachability
single-worker/shared governor
package/import completeness
```

### 必须整改

服务 startup lifecycle：

```text
load registry
compile all RuntimeContracts
security context validate
credentials validate
calendar validate
snapshot provider validate
cache permissions
worker model
engine
contract drift
=> only then listen socket
```

任何 critical fail：

```text
process exits non-zero
```

---

## R26-P0-021 — `/ready` 当前实现自身存在错误，而且失败状态仍可能返回 ready

### 当前问题 A

`/ready`：

```python
store._engine.execute("SELECT 1")
```

而当前 `DuckDBEngine` 对外是：

```text
execute_arrow
execute_reader
relation
...
```

没有看到 `.execute()` public method。

### 当前问题 B

Credential：

```text
provider.resolve() failed
→ cred_ok="failed"
→ 仍可返回 status="ready"
```

### 当前问题 C

strict 下发现：

```text
legacy /home/shw fallback
```

只设置：

```text
legacy_ok=False
```

仍可返回：

```json
{
  "status": "ready",
  "legacy_home_fallback": "in_use"
}
```

### 必须整改

`/ready` 直接复用 startup health state，不自己再写一套逻辑。

```text
startup gate failed => service 不应该启动
runtime dependency later degraded => /ready 503
```

---

## R26-P0-022 — Schema evolution closure 目前主要是测试文件里的模拟 helper，不是 production runtime gate

### 当前事实

R25 `test_r25_resource_schema_2026_08.py` 里：

```python
_schema_epoch_checker(...)
_check_dtype_compatibility(...)
```

是**测试文件自己定义的模拟函数**。

这只能证明：

```text
测试 helper 逻辑
```

不能证明：

```text
Store production read 跨 2024/2025 parquet 时真会 reject
```

现有 schema validation 的核心是：

```text
DESCRIBE read_parquet(... union_by_name=...)
→ 对合并后的 schema 与 declared schema 检查
```

它不等价于：

```text
逐 physical object / schema epoch 验证
```

`union_by_name=True` 可能把某一 epoch 缺字段变成 union 后有字段，旧 partition 产生 null，而不被识别为跨 epoch contract violation。

### 必须整改

真正实现：

```text
SchemaEpoch
SchemaEvolutionPolicy
SchemaMigration
```

Snapshot resolution 后，对 exact objects 的 parquet footer：

```text
object -> schema fingerprint
```

形成 epoch groups。

跨 epoch request 必须验证：

```text
requested field exists in every required epoch
dtype compatible
unit definition compatible
semantic definition version compatible
approved migration exists
```

### 必测

必须用真实 parquet：

```text
2024/a.parquet: close, roe(double)
2025/b.parquet: close              # roe missing
```

调用：

```python
store.read(...)
```

production 必须 reject。

另测：

```text
2024 roe double
2025 roe string
```

reject。

不能再用测试文件里的 simulator 作为 closure evidence。

---

## R26-P0-023 — `GovernedFrame` 可以被伪造，而且目前没有真正接入 FactorEngine 主读取链

### 当前问题 A

`require_governed_provenance()`：

```python
if isinstance(frame, GovernedFrame):
    return frame
```

没有检查：

```text
frame.has_provenance
execution_environment
security context binding
snapshot validity
```

所以：

```python
GovernedFrame(table_or_frame=df)
```

在 production 也能直接返回。

### 当前问题 B

`security_digest` 只是任意字符串：

```text
"d"
```

没有证明它来自真实 security context。

### 当前问题 C

`allow_production_publish` 参数目前没有发挥实际 gate 作用。

### 当前问题 D

当前 `factor_engine/storage/sources/data_access_source.py` 没有真正消费/返回 `GovernedFrame` 的主链证据；R25 FE test 只是直接测试 helper。

### 必须整改

`GovernedFrame` production validation：

```text
source_snapshot valid
lineage valid
execution_environment present
security_digest == current execution context digest
contract fingerprint matches
run_mode valid
```

并要求 FactorEngine DataAccessSource 的 production result：

```text
要么显式 GovernedFrame
要么内部不可剥离 provenance envelope
```

裸 pandas/Polars/Arrow 不能在中间层悄悄丢 provenance 后继续 publish。

---

## R26-P0-024 — `build_sql_snapshot()` 是 public metadata path，但没有逐 dataset authorization

### 当前代码

```python
for name in read_datasets:
    ds = registry.get(name)
    paths = _prepare_dataset_read(...)
    ...
```

没有：

```python
authorize_dataset(name, action="metadata:read")
```

### 风险

即使不返回真实数据，也可能暴露：

```text
dataset existence
physical layout
snapshot identity
paths/files
```

### 必须整改

所有 public metadata APIs：

```text
describe_dataset
build_sql_snapshot
manifest_version
factor metadata
catalog
registry inspection
```

明确分类：

```text
trusted internal
governed public
```

governed public 必须 auth。

---

## R26-P0-025 — Factor HTTP catalog/read 仍有 metadata 泄露与 per-factor authorization 闭环问题

### 当前 `/v1/factors`

返回：

```python
{
    "root": str(catalog.root),
    "count": len(catalog),
    "factors": all_records...
}
```

问题：

1. `catalog.root` 暴露服务器本地目录。
2. `count` 是全 catalog 数量，不是当前 principal 可见数量。
3. factor list 先授权 `factor_lake`，但没有先按每个 factor 的 derived classification/tag 过滤 catalog。
4. premium factor 的名字/存在性本身也是 metadata。
5. factor read 顶层 ctx 与 Store 内 factor gate 的 principal 仍不是同一个 request context（见 P0-005）。

### 必须整改

构造：

```text
VisibleFactorCatalog(principal)
```

再：

```text
count
summary
metadata
read
```

全部基于 visible set。

删除外部 API 的 local root。

---

# 4. R26 P1 — P0 关闭后继续处理；多数属于长期漂移/并发/可维护性风险

---

## R26-P1-001 — RuntimeDatasetContract fingerprint 漏掉 `temporal_axes` 与 `schema`

当前 contract `to_dict()` 包含：

```text
temporal_axes
schema
```

但 fingerprint payload 没包含。

因此：

```text
date_label -> instant
schema change
```

可能 contract fingerprint 不变。

### 修复

不要人工维护 fingerprint field list。

```python
payload = contract.to_dict()
payload.pop("fingerprint")
hash(canonical(payload))
```

---

## R26-P1-002 — Frozen dataclass 并不等于 deep immutable

以下仍是 mutable：

```text
temporal_axes Mapping
units Mapping
SemanticSchemaContract.schema dict
```

外部可以原地修改，fingerprint 不变。

### 修复

编译后深冻结：

```text
MappingProxyType
tuple
frozenset
immutable canonical tree
```

---

## R26-P1-003 — ContractCompiler 是 first-registry global singleton

当前 `_compiler` 是进程全局。

第一次：

```text
registry A
```

绑定后，后续：

```text
registry B / test registry / namespace registry
```

仍可能用 A。

### 修复

Compiler 应由 Store ownership：

```python
self.contract_compiler = ContractCompiler(self.registry)
```

或 cache key 至少绑定 registry fingerprint。

---

## R26-P1-004 — RuntimeContract.market 仍优先按 dataset name 猜

当前使用：

```text
_market_of_name(dataset)
```

而不是：

```text
contract.market
```

### 正确优先级

```text
explicit contract.market
> registry market
> name inference
```

冲突应 startup hard fail。

---

## R26-P1-005 — Storage layout 与 Physical layout 仍然保留两个 truth

`StorageContract.layout` 与：

```text
RuntimeDatasetContract.physical_partition.layout
```

可能不一致。

### 修复

StorageContract 只表达：

```text
backend
format
prefix/location
```

物理布局只由 `PhysicalPartitionSpec` 表达。

---

## R26-P1-006 — Authoritative layout parse error 仍有 fallback 到 daily 的路径

例如：

```python
except ValidationError:
    contract_layout = None
```

或：

```python
except Exception:
    layout = DAILY_TRADE_DATE
```

### 修复

Authoritative declaration typo：

```text
startup fail
```

无声明才允许明确：

```text
PLAIN_GLOB / UNSPECIFIED
```

不要默认：

```text
DAILY_TRADE_DATE
```

---

## R26-P1-007 — Runtime contract 的默认值过于乐观

当前类似：

```text
PIT fidelity = knowledge_date_pit
security classification = public
cross_market_comparable = True
```

### 建议

改成：

```text
UNKNOWN / UNCLASSIFIED / NOT_APPLICABLE
```

Production / automated mining 对 unknown：

```text
reject
```

---

## R26-P1-008 — TemporalAxis 仍可能把一套表级 representation 复制给 knowledge/effective/period

真实场景可同时存在：

```text
knowledge = UTC instant
effective = local date label
period = accounting date label
```

必须支持 per-axis：

```text
representation
precision
storage timezone
semantic timezone
```

不能一套字段复制三次。

---

## R26-P1-009 — RuntimeSecurityContext 允许 `production=True, strict_semantics=False`

当前：

```python
production = explicit or env
strict = is_strict_semantics()
```

二者没有强制 floor。

### 修复

```python
strict = bool(production) or is_strict_semantics()
```

Production 永远不能被显式参数降低。

---

## R26-P1-010 — API Principal Registry 权限撤销需要进程 reset/restart

`_api_registry` 为 process cache。

需要明确二选一：

### 方案 A

安全配置变更：

```text
必须滚动重启
```

写入部署 contract。

### 方案 B

加：

```text
config generation / file mtime / TTL reload
```

权限撤销不能无限期使用旧 cache。

---

## R26-P1-011 — HTTP request_id 有双重生成

Middleware：

```text
header or UUID A
```

Dependency：

```text
header or UUID B
```

客户端没传 header 时可能：

```text
response request id = A
security context id = B
```

### 修复

Middleware 唯一生成：

```python
request.state.request_id
ContextVar
```

全链复用。

---

## R26-P1-012 — Audit 对 `paths/params/error/extra` 没统一递归脱敏

当前 audit 直接记录：

```text
paths
params
error
extra
```

虽然有 `redaction.py`，但没有统一 sanitize。

### 修复

建立：

```python
sanitize_audit_payload()
```

递归处理：

```text
secret
token
authorization
password
signed URL
credential
session token
query signature
```

signed URI 至少去 query。

---

## R26-P1-013 — Audit 文件权限依赖 umask

`path.open("a")` 没显式建立：

```text
0600 / 0640
```

审计日志含内部路径、principal、dataset、params。

### 修复

创建时明确 mode，并启动时检查 owner/group/mode。

---

## R26-P1-014 — CacheManager GC accounting 与物理删除要修正

除 P0-019 外，还需要：

- directory deletion；
- symlink safety；
- disk stat 与 logical size reconciliation；
- quarantined file lifecycle；
- crash recovery；
- stale orphan inventory；
- manager restart 后重建 inventory。

---

## R26-P1-015 — ResourceGovernor duplicate query_id 可覆盖已有 reservation

当前：

```python
self._active[reservation.query_id] = reservation
```

没有 duplicate guard。

### 修复

同 query ID 再 admit：

```text
reject
```

并增加 reservation token/lease identity。

---

## R26-P1-016 — `max_duckdb_concurrency` 当前声明但没执行

要么删除假能力，要么真正：

```text
duckdb slot acquire/release
```

不要 report/配置声称有限制但实际没 gate。

---

## R26-P1-017 — `enforce_single_worker_contract()` 不能只 warning

Production：

```text
workers > 1
AND no shared limiter
=> startup fail
```

Research 可以 warn。

---

## R26-P1-018 — Shared DuckDB Engine 缺 fork/PID 防护

如果未来：

```text
gunicorn --preload
multiprocessing fork
factor mining process pool
```

fork 前创建 DuckDB connection，child 会继承 native connection / lock / pool。

### 修复

```python
self.owner_pid = os.getpid()
```

入口检查 pid。

并：

```python
os.register_at_fork(after_in_child=reset_dataaccess_process_state)
```

重建：

```text
engine
store
pool
S3 state
locks
cache in-memory registry
```

---

## R26-P1-019 — Multi-principal STS 与 shared DuckDB S3 credential 可能产生 credential race

如果以后按 HTTP principal 分 STS：

```text
Request A configure credential A
Request B configure credential B
Request A execute shared conn
```

可能串 credential。

### 必须明确部署模型

二选一：

### A. server-scoped cloud credential

HTTP logical principal 只做应用层 dataset gate。

### B. principal-scoped cloud credential

连接池必须按：

```text
credential_scope_id
```

隔离。

不能在 shared connection 上不断替换 secret。

---

## R26-P1-020 — Internal metadata probes 也必须受 QueryBudget / Governor

例如 factor version/snapshot/universe consistency probe 不能：

```text
deadline=None
全窗口 DISTINCT
```

系统内部 validation 也会扫数据。

所有 helper 都要进入：

```text
InternalReadBudget
GlobalGovernor
```

---

## R26-P1-021 — ReadURIRequest / FactorReadRequest 的资源验证不完整

`ReadRequest` 有列数、instrument cardinality 等 validator。

但 `ReadURIRequest` / `FactorReadRequest` 没完全复用同样的：

```text
columns count/length
instrument cardinality
limit upper bound
time window
format enum
```

### 修复

抽公共 constrained fields/model mixin。

所有 HTTP data-returning endpoints 用同一资源上限。

---

## R26-P1-022 — `DatasetInfo` response schema 对 nullable time/instrument column 要对齐

当前 response model：

```python
time_column: str
instrument_column: str
```

但真实 registry 里可能存在：

```text
None
```

静态维表/特殊 dataset 可能触发 FastAPI response validation error。

改成：

```python
str | None
```

并测试所有 registry dataset 的 metadata endpoint。

---

## R26-P1-023 — Source manifest 还需要 object duplicate / prefix escape / empty URI 防护

即使 P0-015 完成，也单独加入：

```text
duplicate URI reject
empty URI reject
../ escape reject
cross bucket reject
object count mismatch reject
manifest digest mismatch reject
complete=false reject
```

---

## R26-P1-024 — Remote glob language不要自己用字符串启发式判断

统一：

```text
GlobPattern / FileSelector IR
```

处理：

```text
*
?
[]
**
prefix exact
regex date filename
```

不要各处：

```python
if "*" in ...
```

---

## R26-P1-025 — PIT SQL / Python / Polars 应消费同一 Availability IR

当前 Python availability 已明显比 SQL date-level mapping 更完整。

长期目标：

```text
compile Availability IR once
→ Python evaluator
→ DuckDB SQL compiler
→ Polars compiler
```

必须有 golden parity test：

```text
A股午休
收盘
周末
国庆
美股 early close
DST
pre-market
after-hours
date_label
UTC instant
```

三 backend 同结果。

---

## R26-P1-026 — Availability instant 内部应统一 UTC aware，而不是继续扩散 naive datetime

内部约定：

```text
date_label => date
instant    => timezone-aware UTC datetime
```

只有 display/market-calendar boundary 做本地时区。

禁止一个 naive datetime 同时表示：

```text
local market time
UTC storage time
```

---

## R26-P1-027 — `mode="pit"` 命名可能误导

Generic：

```python
read(..., mode="pit")
```

目前更多表达：

```text
这个数据集允许作为 PIT event source
```

不等于：

```text
已经按 decision time 构造 point-in-time state
```

建议改成更明确的 API/文档：

```text
mode="pit_event"
read_as_of_state(...)
```

至少文档明确禁止开发者误以为 `mode=pit` 自动解决所有 look-ahead。

---

## R26-P1-028 — Cross-market currency 规则要以“canonical FX conversion”而不是 boolean 放行

Money：

```text
CNY != USD
```

不能只因为：

```text
cross_market_comparable=True
requires_fx=False
```

就直接比较 level。

允许跨市场 money level 的证据必须是：

```text
explicit FX dataset
FX PIT snapshot
target canonical currency
conversion timestamp/policy
```

否则只能：

```text
within-market rank/zscore
```

---

## R26-P1-029 — Factor catalog/security metadata 要把 classification 本身纳入 snapshot/version

Factor security tag/classification 变化：

```text
不能只改变 catalog 文本
```

必须改变：

```text
factor metadata version
security digest
cache scope
matrix manifest identity
```

旧 cache 不得继续被低权限读取。

---

## R26-P1-030 — `physical_partition_for()` 不应每次自行 `load_registry()` 并吞异常

Physical locator 应消费 Store 当前已经绑定的：

```text
RuntimeDatasetContract
```

而不是自己：

```text
load_registry()
compile again
fallback mirror
```

否则 custom registry / test registry / hot reload / namespace 容易漂移。

---

# 5. R26 P2 — Hardening / 可观测性 / 运维收口

以下不是当前首要 blocker，但建议 R26 一并清掉：

1. 对所有 `except Exception:` 做语义审计：
   - security / PIT / contract / snapshot：原则上不得 fallback；
   - telemetry / cleanup：可吞，但要有明确 comment。
2. 对所有 `bool(raw.get(...))` 做仓库级扫描。
3. 对所有 direct：
   - `pd.read_parquet`
   - `pl.read_parquet`
   - `pyarrow.parquet.read_table`
   - `duckdb read_parquet`
   做 production allowlist 审计。
4. 对所有 `os.environ` 动态安全配置建立 single resolver，防止模块各自解释。
5. 对所有 global singleton 建 reset / PID / config-generation 规则。
6. 对 error message 建统一 external redaction。
7. `EXPLAIN` / `profile_analyze` 当前异常仍可能包含 SQL 片段；外部 service 不应透传。
8. 所有 cache/temp/staging 文件建立 owner/mode policy。
9. API rate limit 与 global governor 统一，避免 HTTP semaphore 与 governor 两套独立计数。
10. stream client disconnect 必须测试：
    - reader close
    - remote slot release
    - governor release
    - cache unpin
    - connection return/discard
11. metrics 不应暴露敏感 dataset/factor label。
12. startup report 输出 config fingerprint，不输出 secret。
13. 对 ContractIR / RuntimeDatasetContract / SemanticCatalog 做 versioned schema。
14. 所有 generated closure report 必须记录：
    - exact commit SHA
    - `git status --porcelain`
    - test command
    - clean checkout / installed wheel test
    - Python/DuckDB/Polars/PyArrow versions
15. 不允许 closure report 写：
    ```text
    final HEAD: 工作树未提交
    ```
    然后据此宣布 Freeze。
16. CI 必须跑在 clean GitHub checkout，不允许工作区残留文件影响 import。
17. 对 `.gitignore` 新增 source-code guard：
    ```text
    如果 git check-ignore 命中 *.py 源文件 => CI fail
    ```
18. 建立 import graph audit：local import target 不存在 => CI fail。
19. 建立 `python -m compileall` / wheel import test。
20. 建立 repo-wide duplicate semantic implementation audit：
    - availability
    - filter rules
    - physical layout
    - security scope
    - snapshot identity
    同一语义只能有一个 compiler。

---

# 6. 必须新增的 R26 真实端到端测试

本节是本轮最重要的 DoD。

## 6.1 Clean checkout / packaging

### T-R26-CLEAN-001

```text
fresh git clone
git clean -xfd
pip install -e dataaccess[all]
import 全部 R25 子包
```

### T-R26-CLEAN-002

```text
python -m build
新 venv install wheel
从 /tmp（不在 repo cwd）import data_access*
```

### T-R26-CLEAN-003

```text
git ls-files dataaccess/security/credentials.py
必须存在
```

### T-R26-CLEAN-004

扫描：

```text
git check-ignore dataaccess/**/*.py
```

除明确 generated source 外，任何 production `.py` 被 ignore → fail。

---

## 6.2 Runtime single-path enforcement

### T-R26-PIPE-001

对：

```text
read_result
read_arrow
read_auto
read_arrow_stream
scan_polars
read_joined
sql
read_uri
read_factors
factor_matrix
```

instrument：

```text
prepare
auth
contract
snapshot
budget
governor
verify
execute
release
```

每个 exactly once。

---

## 6.3 Security

### T-R26-SEC-001

```text
AccessPolicy(allowed_datasets=[])
=> deny all
```

### T-R26-SEC-002

```text
production no explicit policy
=> startup fail
```

### T-R26-SEC-003

```json
{"allow_uri_read":"false"}
```

=> config reject。

### T-R26-SEC-004

```json
{"approved":"false"}
```

=> declassification config reject。

### T-R26-SEC-005

Factor tags：

```text
[basic, premium]
principal only basic
=> deny
```

### T-R26-SEC-006

```text
FactorCatalog unavailable
=> production deny
```

### T-R26-SEC-007

并发两个 API principal 1000 次 nested reads，不串身份。

---

## 6.4 Remote physical layout

### T-R26-PHY-001

StockCapital directory：

```text
2024-01-01.parquet
shares_2024-01-01.parquet
```

remote split 只能 exact list 第一类。

### T-R26-PHY-002

local/mirror/remote/auto 四模式输出 schema/rows 一致。

---

## 6.5 PIT

### T-R26-PIT-001

calendar mapping 不覆盖 knowledge：

```text
SQL path
=> reject
```

不能 same-day fallback。

### T-R26-PIT-002

```text
contract next_session_open
request same_day
=> downgrade rejected
```

### T-R26-PIT-003

Python/DuckDB/Polars golden parity：

```text
A股午休
收盘
节假日
US early close
DST
pre/after market
```

---

## 6.6 Snapshot

### T-R26-SNAP-001

Manifest object：

```text
URI only, no etag/version/checksum
=> production reject
```

### T-R26-SNAP-002

同 URI、同 size、内容替换：

```text
=> snapshot changed
```

### T-R26-SNAP-003

execute 前 ETag=v1，执行后 ETag=v2：

```text
=> SourceSnapshotChanged
```

### T-R26-SNAP-004

`?`, `[]`, `**` wildcard 都不能绕过 unresolved-glob gate。

### T-R26-SNAP-005

unknown policy：

```text
resolve(policy="whatever")
=> reject
```

### T-R26-SNAP-006

manifest object 越出 registered prefix：

```text
=> reject
```

---

## 6.7 Resource

### T-R26-RES-001

HTTP 真实 `/v1/read`：

```text
remote object count > max_scan_objects
=> execute 前 4xx/429/413（按规范）
backend execute counter == 0
```

### T-R26-RES-002

20 并发 public reads 真正进入 global governor，不是直接测试 governor helper。

### T-R26-RES-003

stream client 中途断开：

```text
active governor = 0
remote slot = 0
cache pin = 0
connection no leak
```

### T-R26-RES-004

multi-worker + no shared limiter：

```text
production startup fail
```

---

## 6.8 Cache

### T-R26-CACHE-001

TTL 到期后 stale/evict。

### T-R26-CACHE-002

principal A 的 cache 不被 B 复用。

### T-R26-CACHE-003

pinned generation 永远不删。

### T-R26-CACHE-004

实际 directory GC 后：

```text
disk bytes 真下降
manager accounting == disk inventory
```

---

## 6.9 Schema evolution

### T-R26-SCHEMA-001

真实 parquet 跨 epoch missing field，Store public read reject。

### T-R26-SCHEMA-002

真实 parquet double→string，reject。

### T-R26-SCHEMA-003

同 dtype 但 unit/definition version 改变：

```text
没有 migration => reject
```

不要测试自定义模拟 helper。

---

## 6.10 Governed provenance

### T-R26-PROV-001

```python
GovernedFrame(table_or_frame=df)
```

production reject。

### T-R26-PROV-002

伪造 security_digest reject。

### T-R26-PROV-003

FactorEngine 从 DataAccess 读 → 计算 → materialize → publish，全链 provenance 不丢。

### T-R26-PROV-004

research unsafe frame：

```text
可以研究
不能 production publish
```

---

## 6.11 HTTP readiness / startup

### T-R26-SVC-001

credential invalid => service startup fail。

### T-R26-SVC-002

legacy root strict => startup fail。

### T-R26-SVC-003

engine probe 使用真实 public engine API。

### T-R26-SVC-004

critical calendar missing => startup fail。

### T-R26-SVC-005

source snapshot provider unavailable => 对需要 remote authoritative source 的 production service startup fail。

---

# 7. Mandatory Repo-wide Audit Commands

Coding agent 在完成上述已知问题后，必须继续跑第二轮仓库级审计，不能做到已知列表就停。

## 7.1 Fail-open pattern

扫描：

```text
except Exception:
    pass

except Exception:
    return ...

except ValidationError:
    fallback...
```

重点目录：

```text
dataaccess/security
dataaccess/contract
dataaccess/snapshot
dataaccess/runtime
dataaccess/read
dataaccess/cos
dataaccess/store.py
factor_engine/storage/sources
factor_engine/security
```

分类：

```text
security/PIT/contract/snapshot -> 默认不得吞
cleanup/telemetry -> 可吞但必须有明确理由
```

---

## 7.2 Config coercion

搜索：

```python
bool(raw.get(
bool(entry.get(
str(...).lower() in ...
int(raw...)
```

所有 security/semantic config 进入 strict parser。

---

## 7.3 Direct raw IO bypass

仓库级搜索：

```text
pd.read_parquet
pandas.read_parquet
pl.read_parquet
pl.scan_parquet
pq.read_table
pyarrow.parquet
duckdb.read_parquet
read_csv
open(... parquet/csv)
```

生产 FactorEngine path 未在明确 allowlist 的必须整改。

Allowlist 只允许：

```text
offline evidence certification
migration
repair
test fixtures
one-off admin
```

不得用“这是脚本”自动豁免。

---

## 7.4 Global mutable security/runtime state

搜索：

```text
_global_
singleton
threading.local
os.environ
set_authorizer
set_credential_provider
```

核对：

```text
request scope
thread scope
process scope
fork scope
config generation
```

是否正确。

---

## 7.5 Duplicate truth audit

以下语义禁止存在两套 runtime compiler：

```text
physical partition
availability
required filters
allowed values
security scope
source snapshot
unit conversion
calendar
query budget
```

旧 helper 要么：

```text
delegate new compiler
```

要么删除。

---

# 8. 推荐实现后的最终架构

```text
                       ┌────────────────────┐
Request / FE request ─→│ ExecutionContext   │
                       │ principal          │
                       │ run_mode           │
                       │ request_id         │
                       │ credentials scope  │
                       └─────────┬──────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │ ContractCompiler   │
                       │ RuntimeDataset     │
                       │ Contract           │
                       └─────────┬──────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │ PreparedRead       │
                       │ predicate IR       │
                       │ PIT floor          │
                       │ physical selector  │
                       │ exact filters      │
                       └─────────┬──────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │ SourceSnapshot     │
                       │ exact objects      │
                       │ generation/digest  │
                       └─────────┬──────────┘
                                 │
                 ┌───────────────┴────────────────┐
                 ▼                                ▼
       QueryBudget v2                    Global Governor
       object/bytes/mem                  principal/global
                 └───────────────┬────────────────┘
                                 ▼
                         Cache pin / remote slot
                                 │
                                 ▼
                        SnapshotVerifier BEFORE
                                 │
                                 ▼
                  DuckDB / Polars / PyArrow / Stream
                                 │
                                 ▼
                         SnapshotVerifier AFTER
                                 │
                                 ▼
                    Governed Result + Full Lineage
                                 │
                                 ▼
                  FE compute / factor materialization
                                 │
                                 ▼
                    Provenance + security inheritance
                                 │
                                 ▼
                     Production publish transaction
```

关键：

```text
所有 public path 都必须穿过这条链。
```

---

# 9. R26 Closure / Freeze 条件

只有同时满足以下条件，才允许重新写：

```text
DATAACCESS FULL PLATFORM FREEZE = YES
```

## 9.1 Git / packaging

- [ ] `git status --porcelain` 空。
- [ ] `credentials.py` 等所有源码被 Git 跟踪。
- [ ] clean clone 可 import。
- [ ] source install 可用。
- [ ] wheel install 可用。
- [ ] R25 新包全部进入 distribution。
- [ ] GitHub Actions 在**同一个最终 commit SHA** 上 green。
- [ ] combined required checks 非空且通过。

## 9.2 P0

- [ ] R26-P0-001..025 全部关闭。
- [ ] 无“暂时跳过 P0”。
- [ ] 无“helper 已实现但主路径后续再接”。

## 9.3 Security

- [ ] production no-policy fail。
- [ ] empty scope deny-all 语义明确。
- [ ] API principal request-scoped。
- [ ] nested read 同 principal。
- [ ] factor derived permissions all-required。
- [ ] catalog unavailable fail-closed。
- [ ] strict bool config。
- [ ] no credential cross-request race。

## 9.4 PIT

- [ ] SQL/Python/Polars parity。
- [ ] calendar missing 不 same-day fallback。
- [ ] request 不能 loosen PIT floor。
- [ ] date_label / instant / early-close / DST golden tests 全过。

## 9.5 Snapshot

- [ ] public read 真正 resolve exact source snapshot。
- [ ] unknown identity production reject。
- [ ] before/after verifier 真正对比 identity。
- [ ] same-key overwrite 可检测。
- [ ] manifest prefix/complete/digest 验证。

## 9.6 Resource/cache

- [ ] HTTP/FE/Store 全部进 governor。
- [ ] QueryBudget v2 字段不丢。
- [ ] stream disconnect 无 reservation leak。
- [ ] multi-worker contract 真正 enforce。
- [ ] cache TTL/quota/pin/GC 真正执行。

## 9.7 Schema

- [ ] 跨 epoch production 实现存在于 runtime，不是 test helper。
- [ ] missing/dtype/unit definition drift 真实 public read destructive test 通过。

## 9.8 Provenance

- [ ] `GovernedFrame` 不可用随便构造的字符串伪造可信 provenance。
- [ ] FactorEngine production 主路径实际消费受管 provenance。
- [ ] unsafe research result 不能 publish。

---

# 10. Coding Agent 执行要求

把本文件直接作为整改提示词时，必须遵循：

1. **不要只修改测试让测试绿。**
2. **不要用 monkeypatch helper test 替代真实 public path。**
3. 每修一个 P0：
   - 先写 destructive regression；
   - 必须从 public API 触发；
   - 再改实现。
4. 不要重新建立第三套 contract/security/snapshot。
5. 删除/委托旧逻辑，确保 single source of truth。
6. 遇到：
   ```text
   无法证明安全
   无法证明 PIT
   无法证明 snapshot
   无法证明 schema
   ```
   production / automated_research 均 fail-closed。
7. 不要为了兼容旧行为保留：
   ```text
   except Exception -> fallback
   ```
   除非是 interactive research 且显式 degraded。
8. 修复后继续做 repo-wide audit，不要因为本文件列完就停。
9. 最终重新生成：
   ```text
   R26_FULL_PLATFORM_CLOSURE_REPORT.md
   ```
   报告必须以**已提交并推送的最终 Git SHA**为基线。
10. 最终报告至少记录：
    - final SHA
    - clean git status
    - clean clone import
    - wheel build/install
    - full pytest
    - destructive test matrix
    - GitHub CI status
    - contract drift audit
    - direct IO audit
    - remaining limitations
11. **只有 current GitHub clean checkout 可以复现实验结果时才能 Freeze。**
12. 不要写：
    ```text
    875 passed
    ```
    却依赖本地 untracked/ignored Python 源文件。
13. P1 中凡是安全、PIT、数据错误可能导致 silent wrong result 的，在实施中自动升级 P0。
14. 所有修复完成后，再整体读一遍：
    ```text
    dataaccess/
    factor_engine/storage/sources/
    factor_engine/security/
    ```
    继续寻找本文件未覆盖的问题；如发现 concrete blocker，必须一并修复后再收官。

---

# 11. 最终一句

R26 的目标不是继续把 DataAccess 变得更复杂，而是：

> **让 R25 已经设计出来的 Contract、Security、Snapshot、Resource、PIT、Provenance 真正成为一条不可绕过的执行链，并保证从 GitHub clean checkout 到 production service 的行为一致。**

在此之前，不建议继续新增 DataAccess 功能，也不建议再次声明 FULL PLATFORM FREEZE。
